"""
AnalysisAgent — Appelle le LLM et valide les résultats.

Rôle UNIQUE : construire le prompt, appeler Gemini, valider
chaque correction proposée avant de la retourner à l'utilisateur.

Différence avec l'ancien code :
  AVANT → les corrections inventées arrivaient directement à l'utilisateur.
  APRÈS → chaque bloc passe par FixValidator. Les corrections douteuses
           sont marquées [NON VÉRIFIÉ] au lieu d'être silencieusement affichées.
"""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from validators.fix_validator import fix_validator, FixBlock


class AnalysisAgent:
    """
    Prend le code + les docs RAG → appelle le LLM → valide → retourne.

    Usage :
        agent = AnalysisAgent(llm_service)
        result = agent.analyze(code, context, docs, scores)
    """

    def __init__(self, llm_service=None):
        self._svc = llm_service

    def set_llm_service(self, llm_service):
        self._svc = llm_service

    def analyze(
        self,
        code:    str,
        context: Dict[str, Any],
        docs:    List[Document] = None,
        scores:  List[float]    = None,
    ) -> Dict[str, Any]:
        if self._svc is None:
            return {"analysis": "Erreur : LLM service non initialisé.", "relevant_knowledge": []}

        result = self._svc.analyze_code_with_rag(
            code               = code,
            context            = context,
            precomputed_docs   = docs,
            precomputed_scores = scores,
        )

        result["validated_blocks"] = self._validate_blocks(
            raw_analysis = result["analysis"],
            source_code  = code,
            language     = context.get("language", "python"),
        )
        return result

    def _validate_blocks(self, raw_analysis: str, source_code: str, language: str) -> List[Dict]:
        blocks = []
        parts  = re.split(r'-{3,}\s*FIX START\s*-{3,}', raw_analysis, flags=re.IGNORECASE)

        for raw in parts[1:]:
            end = re.search(r'-{3,}\s*FIX END\s*-{3,}', raw, re.IGNORECASE)
            if end:
                raw = raw[:end.start()]

            def _f(name):
                m = re.search(
                    r'\*\*' + re.escape(name) + r'\*\*\s*:?\s*(.+?)(?=\n\s*\*\*|\Z)',
                    raw, re.DOTALL | re.IGNORECASE,
                )
                return m.group(1).strip() if m else ""

            def _code(section):
                m = re.search(
                    r'\*\*' + re.escape(section) + r'\*\*.*?```\w*\n(.*?)```',
                    raw, re.DOTALL | re.IGNORECASE,
                )
                return m.group(1).rstrip() if m else ""

            sev_raw  = _f("SEVERITY").upper().split()[0] if _f("SEVERITY") else "MEDIUM"
            location = _f("LOCATION")
            line_m   = re.search(r'[:\s](\d{1,5})\b', location)
            problem  = _f("PROBLEM")
            if not problem:
                continue

            block = FixBlock(
                problem      = problem,
                severity     = sev_raw if sev_raw in {"CRITICAL","HIGH","MEDIUM","LOW"} else "MEDIUM",
                location     = location,
                line_number  = int(line_m.group(1)) if line_m else None,
                current_code = _code("CURRENT CODE"),
                fixed_code   = _code("FIXED CODE"),
                why          = _f("WHY"),
            )
            is_valid, reason = fix_validator.validate(block, source_code, language)
            blocks.append({**block.__dict__, "is_valid": is_valid, "validation_reason": reason})

        return blocks


analysis_agent = AnalysisAgent()