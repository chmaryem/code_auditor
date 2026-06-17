"""
lc_git_commit_linter_agent.py — LangChain wrapper pour F3 : Commit Message Linter.

Intègre GitCommitLinter dans l'architecture multi-agent LangGraph.
Node dans le graph : node_commit_lint
Intent détecté par : LCGitDecisionAgent → "commit_lint"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from smart_git.git_commit_linter import lint_commit_message, lint_staged_commit, CommitLintReport

logger = logging.getLogger(__name__)


class LCGitCommitLinterAgent:
    """
    Agent LangChain pour la validation du message de commit.

    Utilisé par le LangGraph node `node_commit_lint`.
    Validation 100% locale (pas de LLM).
    """

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valide le message de commit et retourne le rapport.

        Input state fields:
          - commit_message (str)  : message à valider (optionnel)
          - project_path (str)    : chemin du projet (fallback COMMIT_EDITMSG)

        Output state fields:
          - commit_lint_report (dict) : rapport de validation
        """
        project_path   = Path(state.get("project_path", "."))
        commit_message = state.get("commit_message", "").strip()

        try:
            if commit_message:
                report = lint_commit_message(commit_message)
            else:
                report = lint_staged_commit(project_path)

            state["commit_lint_report"] = self._serialize(report)
            logger.info(
                "Commit lint: %d violations (score %d), valid=%s",
                len(report.violations),
                report.score,
                report.is_valid,
            )
        except Exception as e:
            logger.error("LCGitCommitLinterAgent error: %s", e)
            state["commit_lint_report"] = {
                "error":    str(e),
                "is_valid": False,
                "score":    0,
                "violations": [],
            }

        return state

    def _serialize(self, report: CommitLintReport) -> Dict[str, Any]:
        return {
            "original_message":  report.original_message,
            "is_valid":          report.is_valid,
            "has_warnings":      report.has_warnings,
            "score":             report.score,
            "suggested_message": report.suggested_message,
            "violations": [
                {
                    "rule":       v.rule,
                    "severity":   v.severity,
                    "message":    v.message,
                    "suggestion": v.suggestion,
                }
                for v in report.violations
            ],
            "error": report.error,
        }
