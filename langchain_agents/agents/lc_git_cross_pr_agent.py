"""
lc_git_cross_pr_agent.py — LangChain wrapper pour F6 : Cross-PR Conflict Detection.

Intègre GitCrossPRAnalyzer dans l'architecture multi-agent LangGraph.
Node dans le graph : node_cross_pr
Intent détecté par : LCGitDecisionAgent → "cross_pr_conflicts"
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from smart_git.git_cross_pr import detect_cross_pr_conflicts, CrossPRReport

logger = logging.getLogger(__name__)


class LCGitCrossPRAgent:
    """
    Agent LangChain pour la détection de conflits entre PRs ouvertes.

    Utilisé par le LangGraph node `node_cross_pr`.
    Requiert GitHub API (via MCPGitHubService).
    """

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Détecte les conflits cross-PR et retourne le rapport.

        Input state fields:
          - owner (str)       : organisation/utilisateur GitHub
          - repo (str)        : nom du dépôt
          - pr_number (int)   : PR courante à exclure (optionnel)
          - base (str)        : branche de base (défaut: main)

        Output state fields:
          - cross_pr_report (dict) : rapport de conflits
        """
        owner      = state.get("owner", "")
        repo       = state.get("repo", "")
        pr_number: Optional[int] = state.get("pr_number")
        base       = state.get("base", "main")

        if not owner or not repo:
            state["cross_pr_report"] = {
                "error": "owner et repo sont requis pour la détection cross-PR.",
                "conflicts": [],
                "total_open_prs": 0,
            }
            return state

        try:
            from services.code_mode_client import github

            report = detect_cross_pr_conflicts(owner, repo, github, base, pr_number)
            state["cross_pr_report"] = self._serialize(report)
            logger.info(
                "Cross-PR: %d conflicts in %d open PRs",
                len(report.conflicts),
                report.total_open_prs,
            )
        except Exception as e:
            logger.error("LCGitCrossPRAgent error: %s", e)
            state["cross_pr_report"] = {
                "error":         str(e),
                "conflicts":     [],
                "total_open_prs": 0,
            }

        return state

    def _serialize(self, report: CrossPRReport) -> Dict[str, Any]:
        return {
            "repo":            report.repo,
            "total_open_prs":  report.total_open_prs,
            "has_conflicts":   report.has_conflicts,
            "high_risk_count":   report.high_risk_count,
            "medium_risk_count": report.medium_risk_count,
            "conflicts": [
                {
                    "file_path":   c.file_path,
                    "pr_numbers":  c.pr_numbers,
                    "pr_titles":   c.pr_titles,
                    "risk_level":  c.risk_level,
                    "note":        c.note,
                }
                for c in report.conflicts
            ],
            "pr_summaries": report.pr_summaries,
            "error":        report.error,
        }
