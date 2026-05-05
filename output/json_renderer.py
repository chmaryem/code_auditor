"""
output/json_renderer.py — Rendu JSON pour API FastAPI + VS Code extension.

Convertit les réponses brutes du LLM en objets structurés AnalysisResultResponse.
Compatible avec le format LSP Diagnostic pour les squiggly lines dans VS Code.

Usage:
    renderer = JSONRenderer()
    result = renderer.render_analysis(raw_text, file_path, context, elapsed, score)
    # → AnalysisResultResponse prêt pour l'API
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from api.models import (
    AnalysisResultResponse,
    IssueDiagnostic,
    FixSuggestion,
)


# ── Mapping sévérité → LSP DiagnosticSeverity ────────────────────────────────
# VS Code utilise ces niveaux pour les couleurs des squiggly lines :
#   1 = Error (rouge)      → CRITICAL
#   2 = Warning (orange)   → HIGH
#   3 = Information (bleu) → MEDIUM
#   4 = Hint (gris)        → LOW

LSP_SEVERITY_MAP = {
    "CRITICAL": 1,
    "HIGH": 2,
    "MEDIUM": 3,
    "LOW": 4,
}


class JSONRenderer:
    """
    Transforme la sortie texte du LLM en objets structurés pour l'API.

    Gère les 3 stratégies de réponse du LLM :
      - block_fix        : blocs ---FIX START--- individuels
      - targeted_methods : blocs ---METHOD START--- avec méthodes réécrites
      - full_class       : classe entière réécrite dans ---SOLUTION START---

    Chaque stratégie produit le même format de sortie : AnalysisResultResponse.
    """

    def render_analysis(
        self,
        raw_text: str,
        file_path: Path,
        context: dict,
        elapsed: float,
        score: int = 0,
    ) -> AnalysisResultResponse:
        """
        Parse la réponse LLM complète et retourne un résultat structuré.

        Args:
            raw_text  : Réponse textuelle brute du LLM
            file_path : Chemin du fichier analysé
            context   : Dictionnaire de contexte (language, dependencies, etc.)
            elapsed   : Temps d'analyse en secondes
            score     : Score d'importance du changement (0-100)

        Returns:
            AnalysisResultResponse prêt pour sérialisation JSON
        """
        strategy = self._detect_strategy(raw_text)
        issues = self._parse_issues(raw_text)
        fixes = self._parse_fixes(raw_text)
        language = context.get("language", "unknown")

        # Compter les docs RAG utilisés depuis le contexte
        rag_docs = context.get("docs_used", 0)

        return AnalysisResultResponse(
            file_path=str(file_path),
            language=language,
            score=score,
            strategy=strategy,
            issues=issues,
            fixes=fixes,
            elapsed_seconds=round(elapsed, 2),
            rag_docs_used=rag_docs,
            raw_analysis=raw_text,
        )

    def render_clean(self, file_path: Path, reason: str) -> AnalysisResultResponse:
        """Résultat vide quand le fichier est propre ou le changement mineur."""
        return AnalysisResultResponse(
            file_path=str(file_path),
            score=0,
            strategy="none",
            raw_analysis=reason,
        )

    # ── Détection de la stratégie ─────────────────────────────────────────────

    @staticmethod
    def _detect_strategy(text: str) -> str:
        """Détecte la stratégie choisie par le LLM depuis le bloc ---DECISION---."""
        dec_match = re.search(
            r"---DECISION---\s*(.*?)\s*---DECISION END---",
            text, re.DOTALL | re.IGNORECASE,
        )
        if dec_match:
            strat_match = re.search(
                r"STRATEGY\s*:\s*(\w+)", dec_match.group(1), re.IGNORECASE
            )
            if strat_match:
                s = strat_match.group(1).lower().strip()
                if s in ("full_class", "targeted_methods", "block_fix"):
                    return s

        # Fallback : détecter par la présence de marqueurs
        if "--- SOLUTION START ---" in text or "---SOLUTION START---" in text:
            return "full_class"
        if "---METHOD START:" in text:
            return "targeted_methods"
        return "block_fix"

    # ── Parsing des issues (blocs ---FIX START---) ────────────────────────────

    @staticmethod
    def _parse_issues(text: str) -> List[IssueDiagnostic]:
        """
        Parse tous les blocs ---FIX START--- et retourne des IssueDiagnostic.

        Format LLM attendu :
            ---FIX START---
            **PROBLEM**: description
            **SEVERITY**: CRITICAL | HIGH | MEDIUM | LOW
            **LOCATION**: method_name, line N
            **CURRENT CODE**: ```lang\n...\n```
            **FIXED CODE**: ```lang\n...\n```
            **WHY**: explanation
            ---FIX END---
        """
        issues: List[IssueDiagnostic] = []
        parts = re.split(r'-{3,}\s*FIX START\s*-{3,}', text, flags=re.IGNORECASE)

        for raw in parts[1:]:
            end = re.search(r'-{3,}\s*FIX END\s*-{3,}', raw, re.IGNORECASE)
            if end:
                raw = raw[:end.start()]

            problem = _extract_field(raw, "PROBLEM")
            if not problem:
                continue

            sev_raw = _extract_field(raw, "SEVERITY").upper().split()[0] if _extract_field(raw, "SEVERITY") else "MEDIUM"
            severity = sev_raw if sev_raw in LSP_SEVERITY_MAP else "MEDIUM"

            location = _extract_field(raw, "LOCATION")
            line_match = re.search(r'[:\s](\d{1,5})\b', location)

            issues.append(IssueDiagnostic(
                severity=severity,
                message=problem,
                line=int(line_match.group(1)) if line_match else None,
                column=None,
                source="code-auditor",
                code_snippet=_extract_code_block(raw, "CURRENT CODE"),
                suggestion=_extract_field(raw, "WHY"),
            ))

        return issues

    # ── Parsing des fixes ─────────────────────────────────────────────────────

    @staticmethod
    def _parse_fixes(text: str) -> List[FixSuggestion]:
        """Parse les corrections proposées (current → fixed code)."""
        fixes: List[FixSuggestion] = []
        parts = re.split(r'-{3,}\s*FIX START\s*-{3,}', text, flags=re.IGNORECASE)

        for raw in parts[1:]:
            end = re.search(r'-{3,}\s*FIX END\s*-{3,}', raw, re.IGNORECASE)
            if end:
                raw = raw[:end.start()]

            fixed_code = _extract_code_block(raw, "FIXED CODE")
            if not fixed_code:
                continue

            fixes.append(FixSuggestion(
                location=_extract_field(raw, "LOCATION"),
                current_code=_extract_code_block(raw, "CURRENT CODE"),
                fixed_code=fixed_code,
                explanation=_extract_field(raw, "WHY"),
            ))

        # Aussi parser les blocs METHOD (targeted_methods strategy)
        for m in re.finditer(
            r"---METHOD START:\s*(\w+)---[^`]*```\w*\n(.*?)```[^-]*---METHOD END---",
            text, re.DOTALL | re.IGNORECASE,
        ):
            fixes.append(FixSuggestion(
                location=m.group(1),
                current_code="",
                fixed_code=m.group(2).rstrip(),
                explanation=f"Méthode {m.group(1)} réécrite",
            ))

        # Aussi parser SOLUTION (full_class strategy)
        sol_match = re.search(
            r"---SOLUTION START---[^`]*```\w*\n(.*?)```[^-]*---SOLUTION END---",
            text, re.DOTALL | re.IGNORECASE,
        )
        if sol_match:
            fixes.append(FixSuggestion(
                location="entire_file",
                current_code="",
                fixed_code=sol_match.group(1).rstrip(),
                explanation="Classe complète réécrite (full_class strategy)",
            ))

        return fixes


# ── Helpers internes ──────────────────────────────────────────────────────────

def _extract_field(raw: str, field_name: str) -> str:
    """Extrait la valeur d'un champ **FIELD**: value depuis le texte brut."""
    m = re.search(
        r'\*\*' + re.escape(field_name) + r'\*\*\s*:?\s*(.+?)(?=\n\s*\*\*|\Z)',
        raw, re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


def _extract_code_block(raw: str, section: str) -> str:
    """Extrait un bloc ```code``` après un champ **SECTION**."""
    m = re.search(
        r'\*\*' + re.escape(section) + r'\*\*.*?```\w*\n(.*?)```',
        raw, re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).rstrip() if m else ""
