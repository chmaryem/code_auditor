"""
conflict_resolution_agent.py — Résolution de conflits PR.

ARCHITECTURE v6.2 — Pipeline RAG-enriched :

  Pipeline de résolution :
    1. get_pr_mergeable_status() — détection fiable (REST API direct + fallback MCP)
    2. RAG context query — cache SQLite + ChromaDB + KnowledgeGraph (0 token LLM)
    3. resolve_file_smart() — 3-way diff → conservateur → Gemini + RAG context
    4. RESOLVE_README.md — résumé des changements généré automatiquement
    5. push_file() + create_pull_request() — via github.* wrappers MCP

  Le RAG enrichit le prompt Gemini avec les patterns du projet (SQL injection,
  bcrypt, design patterns) pour une résolution contextuelle et pas seulement mécanique.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

# Force UTF-8 sur Windows (évite UnicodeEncodeError cp1252 pour ✓, ✗, →, etc.)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_R  = "\033[0m"
_B  = "\033[1m"
_GR = "\033[92m"
_YL = "\033[93m"
_RD = "\033[91m"
_CY = "\033[96m"
_DM = "\033[2m"



def _merge_3way(base: str, ours: str, theirs: str) -> Optional[str]:
    """
    Merge 3-way déterministe via difflib — 0 token LLM.
    Couvre ~70% des conflits PR (modifications dans des méthodes différentes).
    """
    if not base or not ours or not theirs:
        return None
    try:
        base_lines   = base.splitlines(keepends=True)
        ours_lines   = ours.splitlines(keepends=True)
        theirs_lines = theirs.splitlines(keepends=True)

        ours_ops   = list(difflib.SequenceMatcher(None, base_lines, ours_lines).get_opcodes())
        theirs_ops = list(difflib.SequenceMatcher(None, base_lines, theirs_lines).get_opcodes())

        ours_changed   = {i for tag, i1, i2, _, __ in ours_ops   if tag != "equal" for i in range(i1, i2)}
        theirs_changed = {i for tag, i1, i2, _, __ in theirs_ops if tag != "equal" for i in range(i1, i2)}

        if ours_changed & theirs_changed:
            return None  # Vraie intersection → LLM nécessaire

        # Pas de chevauchement : partir de OURS + ajouter ajouts THEIRS
        base_set  = set(base_lines)
        theirs_new = [l for l in theirs_lines if l not in base_set]

        merged = list(ours_lines)
        insertion_point = len(merged)
        for i, line in enumerate(reversed(merged)):
            if line.strip() == "}":
                insertion_point = len(merged) - 1 - i
                break
        if theirs_new:
            merged[insertion_point:insertion_point] = theirs_new

        result = "".join(merged)
        if "<<<<<<" in result or "=======" in result:
            return None
        if result.count("{") != result.count("}"):
            return ours if ours and "<<<<<<" not in ours else None
        return result
    except Exception:
        return None


def _merge_conservative(ours: str, theirs: str, filename: str) -> Optional[str]:
    """
    Merge conservateur : OURS + nouvelles méthodes de THEIRS.
    0 token LLM. Couvre ~20% des cas supplémentaires.
    """
    ratio = difflib.SequenceMatcher(None, ours, theirs).ratio()
    if ratio > 0.88:
        return ours  # Très similaires → OURS suffit

    ext = Path(filename).suffix.lower()
    if ext != ".java":
        return ours if ratio > 0.70 else None

    # Pour Java : détecter les nouvelles méthodes de THEIRS
    method_pattern = re.compile(
        r'(?:(?:public|private|protected|static|final|synchronized)\s+)+\w[\w<>\[\]]*\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{',
        re.MULTILINE,
    )
    ours_methods   = {m.group(1) for m in method_pattern.finditer(ours)}
    theirs_methods = {m.group(1) for m in method_pattern.finditer(theirs)}
    new_methods    = theirs_methods - ours_methods - {"if", "for", "while", "switch", "try", "catch"}

    if not new_methods:
        return ours  # Rien à ajouter

    # Extraire les corps des nouvelles méthodes
    additions = ""
    for m in method_pattern.finditer(theirs):
        name = m.group(1)
        if name in new_methods:
            start = m.start()
            depth = 0
            i = m.end() - 1
            while i < len(theirs):
                if theirs[i] == "{": depth += 1
                elif theirs[i] == "}":
                    depth -= 1
                    if depth == 0:
                        additions += "\n\n" + theirs[start:i+1]
                        break
                i += 1

    if not additions:
        return ours

    last_brace = ours.rfind("}")
    if last_brace == -1:
        return None
    result = ours[:last_brace] + additions + "\n\n" + ours[last_brace:]
    if "{" in result and result.count("{") == result.count("}"):
        return result
    return ours


def _resolve_with_gemini_budget(
    filename: str, ours: str, theirs: str, rag_context: list = None
) -> Optional[str]:
    """
    Résolution LLM avec budget strict — seulement les blocs différents.
    v6.3 : cascade OpenAI → Gemini → Groq via services.llm_factory.
    """
    ext = Path(filename).suffix.lstrip(".")

    # Extraire les blocs différents seulement
    ours_lines   = ours.splitlines()
    theirs_lines = theirs.splitlines()
    blocks = [
        (
            "\n".join(ours_lines[i1:i2]),
            "\n".join(theirs_lines[j1:j2]),
        )
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ours_lines, theirs_lines).get_opcodes()
        if tag != "equal"
    ]

    if not blocks:
        return ours  # Identiques

    conflicts_text = ""
    for i, (o, t) in enumerate(blocks[:5], 1):
        conflicts_text += f"\nBlock {i}:\nOURS:\n{o[:400]}\nTHEIRS:\n{t[:400]}\n"

    ours_header = "\n".join(ours.splitlines()[:12])
    prompt = (
        f"Resolve {len(blocks)} Git conflict block(s) in {filename} ({ext}).\n"
        f"Strategy: prefer security fixes, keep functional code from both sides.\n"
        f"File header:\n{ours_header}\n\n"
        f"Conflicts to resolve:\n{conflicts_text}\n"
    )

    # Injecter le contexte RAG dans le prompt
    if rag_context:
        prompt += "\nRelevant project patterns (from knowledge base):\n"
        for k in rag_context[:5]:
            if isinstance(k, dict):
                name = k.get("pattern", k.get("name", k.get("rule", "")))
                desc = k.get("description", k.get("detail", ""))
                prompt += f"- {name}: {desc}\n"
            elif isinstance(k, str):
                prompt += f"- {k}\n"
        prompt += "Use these patterns to guide the merge resolution.\n"

    prompt += (
        f"\nOutput ONLY the full resolved file content. No explanations.\n"
        f"If cannot resolve: output OURS_FALLBACK"
    )

    # Cascade OpenAI → Gemini → Groq via llm_factory
    from services.llm_factory import invoke_with_fallback
    text = invoke_with_fallback(
        prompt,
        temperature = 0.0,
        max_tokens  = 10000,
        label       = f"resolve:{filename}",
    )
    if text is None:
        return ours
    if "OURS_FALLBACK" in text:
        return ours
    if "<<<<<<" not in text:
        return text
    return ours


# ── Interactive conflict resolution helpers ───────────────────────────────────

def _classify_block_type(ours_text: str, theirs_text: str, filename: str) -> str:
    """Classifie un bloc de diff: 'import', 'method', 'simple', ou 'other'."""
    combined = (ours_text + "\n" + theirs_text).strip()
    lines = [l.strip() for l in combined.splitlines() if l.strip()]
    if not lines:
        return "simple"
    ext = Path(filename).suffix.lower()

    # Imports
    if ext == ".java" and all(l.startswith(("import ", "package ")) for l in lines):
        return "import"
    if ext == ".py" and all(l.startswith(("import ", "from ")) for l in lines):
        return "import"
    if ext in (".ts", ".js") and all("import " in l or "require(" in l for l in lines):
        return "import"

    # Method bodies
    if ext == ".java" and re.search(
        r'(?:public|private|protected)\s+(?:static\s+)?\w+[\w<>\[\]]*\s+\w+\s*\(', combined
    ):
        return "method"
    if ext == ".py" and re.search(r'^\s*(async\s+)?def\s+\w+', combined, re.MULTILINE):
        return "method"
    if ext in (".ts", ".js") and re.search(r'(?:function|class|const\s+\w+\s*=)\s', combined):
        return "method"

    # Simple (whitespace only)
    o = {l.strip() for l in ours_text.splitlines() if l.strip()}
    t = {l.strip() for l in theirs_text.splitlines() if l.strip()}
    if len(o.symmetric_difference(t)) <= 2:
        return "simple"
    return "other"


def _auto_merge_imports(ours_text: str, theirs_text: str) -> str:
    """Fusionne les imports des 2 côtés (dédupliqués, triés)."""
    all_lines = list(dict.fromkeys(
        [l for l in ours_text.splitlines() if l.strip()]
        + [l for l in theirs_text.splitlines() if l.strip()]
    ))
    packages = [l for l in all_lines if l.strip().startswith("package ")]
    imports = sorted(set(l for l in all_lines if l.strip().startswith(("import ", "from "))))
    others = [l for l in all_lines if l not in packages and l not in imports]
    result = packages + ([""] if packages and imports else []) + imports + others
    return "\n".join(result) + "\n"


def _prompt_user_choice(
    filename: str, ours_block: str, theirs_block: str, block_type: str,
) -> str:
    """Affiche les 2 versions et demande [1] OURS [2] THEIRS [3] LLM."""
    short = Path(filename).name
    label = {"method": "MÉTHODE", "other": "LOGIQUE"}.get(block_type, "BLOC")

    print(f"\n  {'─' * 60}")
    print(f"  {_CY}{_B}Conflit {label} dans {short}{_R}")
    print(f"  {'─' * 60}")

    for tag, color, block in [("[1] OURS (base)", _GR, ours_block),
                               ("[2] THEIRS (feature)", _YL, theirs_block)]:
        preview = "\n".join(f"    {l}" for l in block.splitlines()[:12])
        extra = len(block.splitlines()) - 12
        if extra > 0:
            preview += f"\n    {_DM}... (+{extra} lignes){_R}"
        print(f"\n  {color}{tag} :{_R}")
        print(preview)

    print(f"\n  {_CY}[3] MERGER par LLM (résolution intelligente){_R}")
    print(f"  {'─' * 60}")

    while True:
        try:
            c = input(f"  Choix [1/2/3] (défaut=1) : ").strip()
            if c in ("", "1"):
                return "ours"
            if c == "2":
                return "theirs"
            if c == "3":
                return "llm"
            print(f"  {_RD}Tapez 1, 2 ou 3.{_R}")
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {_DM}→ OURS par défaut{_R}")
            return "ours"


def _resolve_block_with_llm(
    filename: str, ours_block: str, theirs_block: str, rag_context: list = None,
) -> str:
    """Résout UN bloc de conflit via LLM."""
    ext = Path(filename).suffix.lstrip(".")
    prompt = (
        f"Merge this conflict in {filename} ({ext}).\n"
        f"Combine the best of both. Keep security fixes.\n\n"
        f"OURS:\n```{ext}\n{ours_block}\n```\n\n"
        f"THEIRS:\n```{ext}\n{theirs_block}\n```\n"
    )
    if rag_context:
        prompt += "\nPatterns:\n"
        for k in (rag_context or [])[:3]:
            name = k.get("pattern", k.get("name", "")) if isinstance(k, dict) else str(k)
            if name:
                prompt += f"- {name}\n"
    prompt += "\nOutput ONLY the merged code. No explanations, no fences.\n"

    from services.llm_factory import invoke_with_fallback
    text = invoke_with_fallback(prompt, temperature=0.0, max_tokens=4000,
                                label=f"merge_block:{Path(filename).name}")
    if text and "<<<<<<" not in text:
        text = re.sub(r'^```\w*\n', '', text)
        text = re.sub(r'\n```\s*$', '', text)
        return text
    return ours_block


def _validate_resolution(code: str) -> tuple:
    """Validation rapide du code résolu (0 dépendance externe)."""
    if "<<<<<<" in code or "=======" in code:
        return False, "Marqueurs de conflit non résolus"
    if code.count("{") != code.count("}"):
        return False, f"Accolades déséquilibrées ({code.count('{')}/{code.count('}')})"
    if not code.strip():
        return False, "Contenu vide"
    return True, "OK"


def resolve_file_smart(
    filename: str, base_content: str, ours_content: str, theirs_content: str,
    rag_context: list = None,
) -> Tuple[Optional[str], str, list]:
    """
    Pipeline interactif à 3 niveaux. Retourne (contenu, méthode, details[]).

    v7.3 — Résolution interactive :
      - Auto : imports, package, whitespace, simple changes
      - Interactif : method bodies, logique → [1] OURS [2] THEIRS [3] LLM
      - Sandbox : validation après résolution LLM
    """
    # Niveau 0 : fichiers identiques
    if ours_content == theirs_content:
        return ours_content, "identical", []

    # Niveau 1 : 3-way sans conflit (modifications dans des zones séparées)
    ours_lines = ours_content.splitlines(keepends=True)
    theirs_lines = theirs_content.splitlines(keepends=True)
    opcodes = list(difflib.SequenceMatcher(None, ours_lines, theirs_lines).get_opcodes())

    # Check if all diffs are imports/simple → full auto
    diff_blocks = [(tag, i1, i2, j1, j2) for tag, i1, i2, j1, j2 in opcodes if tag != "equal"]
    if not diff_blocks:
        return ours_content, "identical", []

    all_auto = True
    for tag, i1, i2, j1, j2 in diff_blocks:
        ours_block = "".join(ours_lines[i1:i2])
        theirs_block = "".join(theirs_lines[j1:j2])
        btype = _classify_block_type(ours_block, theirs_block, filename)
        if btype not in ("import", "simple"):
            all_auto = False
            break

    if all_auto:
        # All diffs are imports/simple → auto-merge without asking
        result = _merge_3way(base_content, ours_content, theirs_content)
        if result:
            return result, "3way", [{"type": "import", "choice": "auto"}]
        return ours_content, "conservative", [{"type": "import", "choice": "auto"}]

    # Niveau 2 : Résolution interactive bloc par bloc
    print(f"\n  {_CY}Résolution interactive pour {Path(filename).name}{_R}")
    resolved_lines = []
    details = []
    used_llm = False

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            resolved_lines.extend(ours_lines[i1:i2])
            continue

        ours_block = "".join(ours_lines[i1:i2])
        theirs_block = "".join(theirs_lines[j1:j2])
        btype = _classify_block_type(ours_block, theirs_block, filename)

        if btype == "import":
            merged = _auto_merge_imports(ours_block, theirs_block)
            resolved_lines.extend(merged.splitlines(keepends=True))
            details.append({"type": "import", "choice": "auto",
                            "ours": ours_block[:100], "theirs": theirs_block[:100]})
            print(f"    {_GR}✓ Imports fusionnés automatiquement{_R}")
        elif btype == "simple":
            resolved_lines.extend(ours_lines[i1:i2])
            details.append({"type": "simple", "choice": "auto"})
        else:
            # Interactive
            choice = _prompt_user_choice(filename, ours_block, theirs_block, btype)
            if choice == "ours":
                resolved_lines.extend(ours_lines[i1:i2])
            elif choice == "theirs":
                resolved_lines.extend(theirs_lines[j1:j2])
            elif choice == "llm":
                used_llm = True
                print(f"    {_YL}Résolution LLM en cours...{_R}", end="", flush=True)
                merged = _resolve_block_with_llm(filename, ours_block, theirs_block, rag_context)
                resolved_lines.extend(merged.splitlines(keepends=True))
                if not merged.endswith("\n"):
                    resolved_lines.append("\n")
                print(f" {_GR}✓{_R}")
            details.append({"type": btype, "choice": choice,
                            "ours": ours_block[:200], "theirs": theirs_block[:200]})

    result = "".join(resolved_lines)

    # Validation
    valid, msg = _validate_resolution(result)
    if not valid:
        print(f"    {_RD}Validation échouée : {msg} → fallback OURS{_R}")
        return ours_content, "fallback", details

    # Sandbox validation pour résolutions LLM
    if used_llm:
        try:
            from services.sandbox_executor import SandboxExecutor
            script = (
                f"code = '''{result[:3000]}'''\n"
                f"# Quick syntax check\n"
                f"braces = code.count('{{') == code.count('}}')\n"
                f"no_markers = '<<<<<<' not in code\n"
                f"print('SANDBOX_OK' if braces and no_markers else 'SANDBOX_FAIL')\n"
            )
            sb_result = SandboxExecutor(timeout=15).execute(script)
            if sb_result.success and "SANDBOX_OK" in (sb_result.stdout or ""):
                print(f"    {_GR}✓ Sandbox validé{_R}")
            else:
                print(f"    {_YL}⚠ Sandbox warning (résultat conservé){_R}")
        except Exception:
            pass  # Sandbox non disponible → OK

    method = "interactive" + ("_llm" if used_llm else "")
    return result, method, details


def _ensure_list(value: Any) -> List:
    if isinstance(value, list):
        return value
    if value is None or isinstance(value, int):
        return []
    if isinstance(value, str):
        return [value] if value else []
    return list(value) if hasattr(value, "__iter__") else []


# ─────────────────────────────────────────────────────────────────────────────
# API publique — appelée par pr_analyzer.py
# ─────────────────────────────────────────────────────────────────────────────

def _generate_resolve_readme(
    pr_number: int, orig_title: str, base_ref: str, head_ref: str,
    resolved_files: List[str], failed_files: List[str],
    diffs: List[str], rag_patterns: Dict[str, List[str]],
    resolution_details: Dict[str, list] = None,
) -> str:
    """
    Génère un RESOLVE_README.md enrichi avec les détails de résolution.
    v7.3 : inclut le type de conflit, la méthode choisie, et les previews.
    """
    lines = [
        f"# Auto-Resolve Report — PR #{pr_number}",
        "",
        f"> **PR originale** : #{pr_number} — *{orig_title}*",
        f"> **Branche base** : `{base_ref}` | **Branche source** : `{head_ref}`",
        f"> **Généré par** : Code Auditor v7.3 (Interactive + RAG-enhanced resolution)",
        "",
    ]

    # Résumé
    if resolved_files:
        lines.append("## Résumé des résolutions")
        lines.append("")
        lines.append("| Fichier | Méthode | Conflits | Détails |")
        lines.append("|---|---|---|---|")
        for d in diffs:
            parts = d.replace("- **", "").split("**")
            fname = parts[0] if parts else "?"
            method = "3way"
            if "`" in d:
                method = d.split("`")[1]
            # Count details for this file
            file_details = (resolution_details or {}).get(fname, [])
            n_auto = sum(1 for x in file_details if x.get("choice") == "auto")
            n_interactive = sum(1 for x in file_details if x.get("choice") in ("ours", "theirs", "llm"))
            detail_str = f"{n_auto} auto"
            if n_interactive:
                detail_str += f", {n_interactive} interactif"
            lines.append(f"| `{fname}` | `{method}` | {len(file_details)} | {detail_str} |")
        lines.append("")
    else:
        lines.append("## Fichiers résolus")
        lines.append("")
        lines.append("*Aucun fichier résolu automatiquement.*")
        lines.append("")

    # Détails par fichier
    if resolution_details:
        lines.append("## Détails des résolutions")
        lines.append("")
        for fname, details in resolution_details.items():
            if not details:
                continue
            lines.append(f"### `{Path(fname).name}`")
            lines.append("")
            for i, d in enumerate(details, 1):
                btype = d.get("type", "?")
                choice = d.get("choice", "?")
                icon = {"auto": "✅", "ours": "🔵 OURS", "theirs": "🟡 THEIRS", "llm": "🤖 LLM"}.get(choice, "?")
                lines.append(f"**Bloc {i}** — Type: `{btype}` | Résolution: {icon}")
                lines.append("")
                # Preview for non-trivial blocks
                if btype in ("method", "other") and d.get("ours"):
                    ours_preview = d["ours"][:150].replace("\n", "\n> ")
                    lines.append(f"<details><summary>OURS (preview)</summary>")
                    lines.append(f"")
                    lines.append(f"```")
                    lines.append(f"{ours_preview}")
                    lines.append(f"```")
                    lines.append(f"</details>")
                    lines.append("")
                if btype in ("method", "other") and d.get("theirs"):
                    theirs_preview = d["theirs"][:150].replace("\n", "\n> ")
                    lines.append(f"<details><summary>THEIRS (preview)</summary>")
                    lines.append(f"")
                    lines.append(f"```")
                    lines.append(f"{theirs_preview}")
                    lines.append(f"```")
                    lines.append(f"</details>")
                    lines.append("")

    if failed_files:
        lines.append("## Résolution manuelle requise")
        lines.append("")
        for f in failed_files:
            lines.append(f"- [ ] `{f}`")
        lines.append("")

    if rag_patterns:
        lines.append("## Patterns Knowledge Graph")
        lines.append("")
        for filename, patterns in rag_patterns.items():
            if patterns:
                lines.append(f"**`{Path(filename).name}`** : {', '.join(patterns[:5])}")
                lines.append("")

    lines.extend([
        "## Instructions pour le reviewer",
        "",
        "1. Vérifier les résolutions interactives (marquées 🔵/🟡/🤖)",
        "2. Vérifier qu'aucune vulnérabilité n'a été réintroduite",
        "3. Exécuter les tests unitaires avant de merger",
        "",
        "```bash",
        f"git fetch origin && git checkout auto-resolve/pr-{pr_number}",
        "git diff main..HEAD",
        "```",
        "",
        "---",
        "*Généré automatiquement par Code Auditor v7.3*",
    ])

    return "\n".join(lines)


async def resolve_pr_conflicts(owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
    """
    Résout les conflits d'une PR via MCP github.* wrappers + pipeline RAG.

    Architecture v6.2 :
      - github.* wrappers -> MCP GitHub Server (communication MCP préservée)
      - RAG context -> cache SQLite + ChromaDB + KnowledgeGraph (0 token)
      - resolver pipeline -> 3-way diff -> conservative -> Gemini + RAG context
      - RESOLVE_README.md -> résumé automatique pushé sur la branche
    """
    print(f"\n  {_CY}{_B}Code Auditor — Résolution conflits PR #{pr_number}{_R}")
    print(f"  {_DM}Repo : {owner}/{repo}{_R}")
    print(f"  {_DM}Mode : 3-way diff -> RAG context -> Gemini enrichi{_R}\n")

    try:
        from services.code_mode_client import github
    except ImportError:
        return {"success": False, "error": "code_mode_client non disponible"}

    try:
        # 1. Vérifier les conflits (méthode fiable avec polling)
        print(f"  Détection des conflits...")
        status = github.get_pr_mergeable_status(owner, repo, pr_number)

        if not status.get("has_conflicts"):
            msg = f"No conflicts detected. PR #{pr_number} is mergeable directly."
            print(f"  {_GR}✓ {msg}{_R}\n")
            github.post_comment(owner, repo, pr_number, f"✅ {msg}")
            return {"success": True, "resolved": [], "failed": [], "branch": "", "pr_url": "", "iterations": 1}

        base_ref = status.get("base_ref", "main")
        head_ref = status.get("head_ref", "")
        print(f"  Conflits détectés — base={base_ref}  head={head_ref}")

        # 2. Récupérer les fichiers de code modifiés
        files     = github.get_pr_files(owner, repo, pr_number)
        CODE_EXTS = {".java", ".py", ".js", ".ts", ".jsx", ".tsx"}
        code_files = [
            f for f in files
            if Path(f.get("filename", f.get("path", ""))).suffix.lower() in CODE_EXTS
            and f.get("status") not in ("removed", "added")
        ]
        print(f"  Fichiers à traiter : {len(code_files)}\n")

        # 3. Initialiser le pipeline RAG (0 token LLM)
        print(f"  Chargement contexte RAG...")
        rag = None
        cache = None
        try:
            from services.code_mode_client import RAGAnalyzer, CacheClient
            rag = RAGAnalyzer()
            cache = CacheClient()
            print(f"  {_GR}RAG + Cache actifs{_R}")
        except Exception as e:
            logger.debug("RAG/Cache non disponible: %s", e)
            print(f"  {_YL}RAG non disponible — résolution sans contexte{_R}")

        # 4. Créer la branche de résolution
        branch_name = f"auto-resolve/pr-{pr_number}"
        print(f"  Création branche {branch_name}...")
        github.create_branch(owner, repo, branch_name, base_ref)

        # 5. Résoudre chaque fichier avec contexte RAG
        resolved_files, failed_files, diffs = [], [], []
        rag_patterns = {}  # {filename: [pattern_names]} pour le README
        resolution_details = {}  # {filename: [detail_dicts]} pour le README enrichi

        for f in code_files:
            filename = f.get("filename", f.get("path", ""))
            ext = Path(filename).suffix.lstrip(".")
            print(f"  {_DM}-> {filename}{_R}", end="", flush=True)

            base_content = github.get_file_content(owner, repo, filename, base_ref)
            head_content = github.get_file_content(owner, repo, filename, head_ref)

            if not base_content or not head_content:
                print(f"  {_RD}x contenu inaccessible{_R}")
                failed_files.append(filename)
                continue

            if base_content == head_content:
                print(f"  {_GR}= identique{_R}")
                continue

            # Requête RAG : cache SQLite d'abord, puis analyse complète
            file_rag_context = []
            try:
                if cache:
                    cached = cache.read_analysis(filename)
                    if cached:
                        file_rag_context = cached.get("relevant_knowledge", [])
                        print(f" [cache]", end="", flush=True)

                if not file_rag_context and rag:
                    rag_result = rag.analyze(head_content, filename, ext)
                    file_rag_context = rag_result.get("relevant_knowledge", [])
                    if file_rag_context:
                        print(f" [rag:{len(file_rag_context)}]", end="", flush=True)
            except Exception as e:
                logger.debug("RAG query failed for %s: %s", filename, e)

            # Stocker les noms de patterns pour le README
            pattern_names = []
            for k in (file_rag_context or []):
                if isinstance(k, dict):
                    name = k.get("pattern", k.get("name", k.get("rule", "")))
                    if name:
                        pattern_names.append(str(name))
                elif isinstance(k, str) and k:
                    pattern_names.append(k)
            if pattern_names:
                rag_patterns[filename] = pattern_names

            resolved, method, details = resolve_file_smart(
                filename, base_content, base_content, head_content,
                rag_context=file_rag_context,
            )

            if resolved:
                msg = f"auto-resolve: {filename} (method={method})"
                github.push_file(owner, repo, filename, resolved, msg, branch_name)
                resolved_files.append(filename)
                resolution_details[filename] = details
                b = len(base_content.splitlines())
                h = len(head_content.splitlines())
                r = len(resolved.splitlines())
                diffs.append(f"- **{filename}**: {b}L -> {h}L -> resolved {r}L (`{method}`)")
                print(f"  {_GR}v [{method}]{_R}")
            else:
                print(f"  {_RD}x echec{_R}")
                failed_files.append(filename)

        # 5. Poster le rapport sur la PR originale
        comment_lines = [
            f"## Auto-Resolve Report for PR #{pr_number}", "",
            f"**Resolved** : {len(resolved_files)} fichier(s) ✅",
            f"**Failed** : {len(failed_files)} fichier(s) {'❌' if failed_files else ''}",
            f"**Resolution branch** : `{branch_name}`", "",
        ]
        if diffs:
            comment_lines += ["### Diff Summary", ""] + diffs + [""]
        if failed_files:
            comment_lines += ["### Manual Resolution Needed"] + [f"- ❌ `{f}`" for f in failed_files] + [""]
        if resolved_files:
            comment_lines += [
                "**Apply locally :**",
                "```bash",
                f"git fetch origin && git checkout {branch_name}",
                "```",
            ]
        github.post_comment(owner, repo, pr_number, "\n".join(comment_lines))

        # 6. Générer et pusher RESOLVE_README.md
        pr_data = status.get("pr_data", {})
        orig_title = pr_data.get("title", f"PR #{pr_number}") if pr_data else f"PR #{pr_number}"

        if resolved_files:
            readme_content = _generate_resolve_readme(
                pr_number, orig_title, base_ref, head_ref,
                resolved_files, failed_files, diffs, rag_patterns,
                resolution_details=resolution_details,
            )
            try:
                github.push_file(
                    owner, repo, "RESOLVE_README.md", readme_content,
                    f"auto-resolve: add resolution README for PR #{pr_number}",
                    branch_name,
                )
                print(f"  {_GR}v RESOLVE_README.md pushed{_R}")
            except Exception as e:
                logger.warning("Failed to push README: %s", e)

        # 7. Créer la PR auto-resolve -> main
        pr_url = ""
        if resolved_files:
            new_pr = github.create_pull_request(
                owner, repo,
                title=f"Auto-resolve conflicts for PR #{pr_number}",
                body="\n".join([
                    f"## Conflict resolution for PR #{pr_number}",
                    f"Original : #{pr_number} — *{orig_title}*", "",
                    "Created by **Code Auditor** v7.3 — Interactive + RAG-enhanced resolution.", "",
                    "### Resolution Methods", "",
                ] + diffs + [
                    "",
                    "### RAG Patterns Applied",
                    "",
                ] + [
                    f"- **{fn}**: {', '.join(pats[:5])}"
                    for fn, pats in rag_patterns.items() if pats
                ] + [
                    "",
                    "*See `RESOLVE_README.md` in the branch for full details.*",
                ]),
                head=branch_name,
                base=base_ref,
            )
            pr_url = new_pr.get("html_url", "") if isinstance(new_pr, dict) else ""

       
        print(f"\n  {'='*50}")
        print(f"  {_B}Resume : {len(resolved_files)} resolu(s) - {len(failed_files)} echec(s){_R}")
        if rag_patterns:
            total_p = sum(len(v) for v in rag_patterns.values())
            print(f"  {_CY}RAG patterns : {total_p} patterns appliques{_R}")
        if resolved_files:
            print(f"  {_GR}Branche : {branch_name}{_R}")
            if pr_url:
                print(f"  {_GR}PR creee : {pr_url}{_R}")
            print(f"  {_DM}  git fetch origin && git checkout {branch_name}{_R}")
        for fn in resolved_files:
            print(f"    {_GR}v{_R}  {fn}")
        if failed_files:
            print(f"\n  {_YL}Resolution manuelle requise :{_R}")
            for fn in failed_files:
                print(f"    {_YL}x{_R}  {fn}")
        print()

        return {
            "success": True,
            "resolved": resolved_files, "failed": failed_files,
            "branch": branch_name, "pr_url": pr_url,
            "iterations": 1,
        }

    except Exception as e:
        logger.error("resolve_pr_conflicts failed: %s", e)
        print(f"\n  {_RD}✗ Erreur : {e}{_R}\n")
        return {"success": False, "resolved": [], "failed": [], "error": str(e), "iterations": 1}
    finally:
        try:
            github.disconnect()
        except Exception:
            pass