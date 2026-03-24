"""
AnalysisAgent — Construit le contexte, appelle le LLM, valide les résultats.

Contient (migrés depuis incremental_analyzer.py) :
  - build_system_impact_section() : section [IMPACT SUR LE SYSTÈME] pour le prompt
  - build_context()               : construit le dict de contexte pour le LLM
  - analyze()                     : appelle llm_service + valide les blocs
"""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from validators.fix_validator import fix_validator, FixBlock



# Parser de réponse agentique

def parse_llm_response(raw: str) -> dict:
    """
    Lit la réponse agentique de Gemini et extrait :
      - strategy   : "full_class" | "targeted_methods" | "block_fix"
      - scope      : description du périmètre
      - reason     : pourquoi cette stratégie
      - payload    : contenu adapté à la stratégie
        full_class        → {"class_code": str, "changes": list}
        targeted_methods  → {"methods": [{"name": str, "code": str, "why": str}],
                              "remaining_blocks": list}
        block_fix         → {"blocks": list}  (format parse_fix_blocks existant)
    """
    import re

    result = {
        "strategy": "block_fix",
        "scope":    "",
        "reason":   "",
        "payload":  {},
        "raw":      raw,
    }

    #  Lire le bloc DECISION
    dec_m = re.search(
        r"---DECISION---\s*(.*?)\s*---DECISION END---",
        raw, re.DOTALL | re.IGNORECASE
    )
    if dec_m:
        dec = dec_m.group(1)
        strat_m = re.search(r"STRATEGY\s*:\s*(\w+)", dec, re.IGNORECASE)
        scope_m  = re.search(r"SCOPE\s*:\s*(.+)", dec, re.IGNORECASE)
        reason_m = re.search(r"REASON\s*:\s*(.+)", dec, re.IGNORECASE)
        if strat_m:
            s = strat_m.group(1).lower().strip()
            if s in ("full_class", "targeted_methods", "block_fix"):
                result["strategy"] = s
        if scope_m:
            result["scope"]  = scope_m.group(1).strip()
        if reason_m:
            result["reason"] = reason_m.group(1).strip()

    strategy = result["strategy"]

    # ── Extraire le payload selon la stratégie ────────────────────────────────

    if strategy == "full_class":
        sol_m = re.search(r"---SOLUTION START---[^`]*```\w*\n(.*?)```[^-]*---SOLUTION END---", raw, re.DOTALL | re.IGNORECASE)
        class_code = sol_m.group(1).rstrip() if sol_m else ""

        changes = []
        ch_section = raw.split("CHANGES MADE:")[-1] if "CHANGES MADE:" in raw else ""
        for line in ch_section.splitlines():
            line = line.strip()
            if line.startswith("- ") and ":" in line:
                changes.append(line[2:])

        result["payload"] = {"class_code": class_code, "changes": changes}

    elif strategy == "targeted_methods":
        methods = []
        for m in re.finditer(r"---METHOD START:\s*(\w+)---[^`]*```\w*\n(.*?)```[^-]*---METHOD END---(?:[^W]*WHY:\s*(.+?)(?=\n---|\Z))?", raw, re.DOTALL | re.IGNORECASE):
            methods.append({
                "name": m.group(1).strip(),
                "code": m.group(2).rstrip(),
                "why":  (m.group(3) or "").strip(),
            })

        from output.console_renderer import parse_fix_blocks
        remaining = parse_fix_blocks(raw)

        # Fallback : Gemini a choisi targeted_methods mais n'a produit aucun
        # ---METHOD START--- block (il a probablement produit ---FIX START--- à la place)
        # → traiter comme block_fix pour ne rien perdre
        if not methods and remaining:
            import logging as _log
            _log.getLogger(__name__).debug(
                "targeted_methods sans blocs METHOD → fallback block_fix (%d blocs FIX)",
                len(remaining)
            )
            result["strategy"] = "block_fix"
            result["payload"]  = {"blocks": remaining}
        else:
            result["payload"] = {"methods": methods, "remaining_blocks": remaining}

    else:  # block_fix
        from output.console_renderer import parse_fix_blocks
        result["payload"] = {"blocks": parse_fix_blocks(raw)}

    return result


# ── Section [IMPACT SUR LE SYSTÈME] ──────────────────────────────────────────

def build_system_impact_section(file_name: str, neighborhood: Dict[str, Any]) -> str:
    """
    Génère la section [IMPACT SUR LE SYSTÈME] injectée dans le prompt LLM.
    Le LLM sait exactement quels fichiers et méthodes sont à risque.

    Migré depuis incremental_analyzer._build_system_impact_section() (lignes 589-688)
    """
    preds    = neighborhood.get("predecessors", [])
    succs    = neighborhood.get("successors", [])
    indirect = neighborhood.get("indirect_impacted", [])
    pred_ent = neighborhood.get("predecessor_entities", {})
    succ_ent = neighborhood.get("successor_entities", {})

    if not preds and not succs:
        return ""   # fichier isolé → pas de section système

    lines = [
        "", "═" * 68, "  [IMPACT SUR LE SYSTÈME]", "═" * 68,
        f"  Tu analyses            : {file_name}",
    ]

    if preds:
        pred_names = [Path(p).name for p in preds]
        lines.append(f"  ⚠️  Appelé par (RISQUE)  : {', '.join(pred_names)}")
        for fp in preds:
            fname    = Path(fp).name
            entities = pred_ent.get(fname, [])
            if entities:
                sigs = [f"{e['name']}({e['params']})" if e.get("params") else e["name"]
                        for e in entities[:8]]
                crit     = entities[0].get("criticality", 0)
                crit_tag = f"  [criticité {crit}]" if crit > 0 else ""
                lines.append(f"      • {fname}{crit_tag} appelle : " + ", ".join(sigs))

    if succs:
        succ_names = [Path(p).name for p in succs]
        lines.append(f"  🔗 Utilise              : {', '.join(succ_names)}")
        for fp in succs:
            fname    = Path(fp).name
            entities = succ_ent.get(fname, [])
            if entities:
                sigs = [f"{e['name']}({e['params']})" if e.get("params") else e["name"]
                        for e in entities[:8]]
                lines.append(f"      • {fname} fournit : " + ", ".join(sigs))

    if indirect:
        indirect_names = [Path(p).name for p in indirect[:5]]
        extra = f" +{len(indirect)-5}" if len(indirect) > 5 else ""
        lines.append(f"  📡 Impact indirect      : {', '.join(indirect_names)}{extra}")

    lines += [
        "", "  RÈGLES ARCHITECTURALES :",
        "  • Ne JAMAIS renommer une méthode publique sans vérifier chaque appelant.",
        "  • Tout changement de signature doit rester rétro-compatible.",
        "  • Si tu proposes un fix, vérifie qu'il ne casse pas les appelants listés.",
        "  • Signale explicitement si un fix impacte un fichier dépendant.",
        "═" * 68, "",
    ]
    return "\n".join(lines)


def build_context(file_path: Path, neighborhood: Dict[str, Any],
                  project_indexer=None, change_info: Dict = None) -> Dict[str, Any]:
    """
    Construit le dictionnaire de contexte passé au LLM.
    """
    from agents.code_agent import code_agent

    ctx = {
        "file_path":         str(file_path),
        "language":          code_agent.detect_language(file_path),
        "dependencies":      neighborhood.get("successors", []),
        "dependents":        neighborhood.get("predecessors", []),
        "criticality_score": neighborhood.get("criticality", 0),
        "is_entry_point":    neighborhood.get("criticality", 0) == 0,
        "system_impact_section": build_system_impact_section(file_path.name, neighborhood),
        "neighborhood":      neighborhood,
    }

    if project_indexer:
        ctx["project_context"] = project_indexer.format_for_llm(file_path)

    if change_info:
        ctx["change_type"]   = change_info.get("change_type", "")
        ctx["lines_changed"] = change_info.get("lines_changed", 0)

    return ctx


#AnalysisAgent 

class AnalysisAgent:
    """
    Appelle le LLM avec le contexte enrichi et valide chaque correction proposée.

    Usage :
        result = analysis_agent.analyze(code, context, docs, scores)
    """

    def __init__(self, llm_service=None):
        self._svc = llm_service

    def set_llm_service(self, llm_service):
        self._svc = llm_service

    def analyze(self, code: str, context: Dict[str, Any],
                docs: List[Document] = None, scores: List[float] = None) -> Dict[str, Any]:
        if self._svc is None:
            return {"analysis": "Erreur : LLM service non initialisé.",
                    "relevant_knowledge": [], "validated_blocks": []}

        result = self._svc.analyze_code_with_rag(
            code=code, context=context,
            precomputed_docs=docs, precomputed_scores=scores)

        result["validated_blocks"] = self._validate_blocks(
            result["analysis"], code, context.get("language", "python"))
        return result

    def _validate_blocks(self, raw_analysis: str, source_code: str,
                         language: str) -> List[Dict]:
        """Parse et valide chaque bloc ---FIX START--- proposé par le LLM."""
        blocks = []
        parts  = re.split(r'-{3,}\s*FIX START\s*-{3,}', raw_analysis, flags=re.IGNORECASE)
        for raw in parts[1:]:
            end = re.search(r'-{3,}\s*FIX END\s*-{3,}', raw, re.IGNORECASE)
            if end: raw = raw[:end.start()]

            def _f(name):
                m = re.search(r'\*\*' + re.escape(name) + r'\*\*\s*:?\s*(.+?)(?=\n\s*\*\*|\Z)',
                              raw, re.DOTALL | re.IGNORECASE)
                return m.group(1).strip() if m else ""

            def _code(section):
                m = re.search(r'\*\*' + re.escape(section) + r'\*\*.*?```\w*\n(.*?)```',
                              raw, re.DOTALL | re.IGNORECASE)
                return m.group(1).rstrip() if m else ""

            sev_raw  = _f("SEVERITY").upper().split()[0] if _f("SEVERITY") else "MEDIUM"
            location = _f("LOCATION")
            line_m   = re.search(r'[:\s](\d{1,5})\b', location)
            problem  = _f("PROBLEM")
            if not problem: continue

            block = FixBlock(
                problem=problem,
                severity=sev_raw if sev_raw in {"CRITICAL","HIGH","MEDIUM","LOW"} else "MEDIUM",
                location=location, line_number=int(line_m.group(1)) if line_m else None,
                current_code=_code("CURRENT CODE"), fixed_code=_code("FIXED CODE"),
                why=_f("WHY"))
            is_valid, reason = fix_validator.validate(block, source_code, language)
            blocks.append({**block.__dict__, "is_valid": is_valid, "validation_reason": reason})
        return blocks



    def generate_solution(
        self,
        code:          str,
        context:       dict,
        analysis_text: str,
        docs:          list = None,
        scores:        list = None,
    ) -> dict:
        """
        Génère la classe entière réécrite en exploitant tout le contexte.
        Appelé après analyze() quand le nombre de problèmes critiques est élevé.

        Retourne :
            {
                "solution_text" : str  — classe complète réécrite
                "changes_made"  : list — résumé des corrections
                "language"      : str
            }
        """
        if self._svc is None:
            return {"solution_text": "", "changes_made": [], "language": "unknown"}

        from agents.code_agent import code_agent
        from pathlib import Path

        language = code_agent.detect_language(Path(context.get("file_path", "f.java")))

        # Construire le contexte KB depuis les docs RAG déjà calculés
        knowledge_context = ""
        if docs and scores:
            knowledge_context = self._svc._build_knowledge_context(docs, scores)

        solution_text = self._svc.generate_complete_solution(
            code              = code,
            context           = context,
            analysis_text     = analysis_text,
            knowledge_context = knowledge_context,
        )

        # Extraire le résumé des changements si présent
        changes = []
        if "CHANGES MADE:" in solution_text:
            changes_section = solution_text.split("CHANGES MADE:")[-1].strip()
            for line in changes_section.splitlines():
                line = line.strip()
                if line.startswith("- ") and ":" in line:
                    changes.append(line[2:])

        return {
            "solution_text": solution_text,
            "changes_made":  changes,
            "language":      language,
        }


analysis_agent = AnalysisAgent()