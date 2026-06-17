"""
lc_git_pr_description_agent.py — LangChain wrapper pour F7 : PR Auto-Description.

Intègre GitPRDescriptionGenerator dans l'architecture multi-agent LangGraph.
Node dans le graph : node_pr_description
Intent détecté par : LCGitDecisionAgent → "pr_description"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from smart_git.git_pr_description import (
    generate_pr_description_local,
    generate_pr_description_github,
    PRDescriptionReport,
)

logger = logging.getLogger(__name__)


class LCGitPRDescriptionAgent:
    """
    Agent LangChain pour la génération automatique de description de PR.

    Utilisé par le LangGraph node `node_pr_description`.
    Mode local (git log) ou GitHub API selon les infos disponibles.
    """

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Génère la description de PR et la stocke dans l'état.

        Input state fields:
          - project_path (str)  : chemin du projet git
          - branch (str)        : branche feature
          - base (str)          : branche de base
          - owner (str)         : GitHub owner (optionnel — active mode GitHub)
          - repo (str)          : GitHub repo  (optionnel)
          - pr_number (int)     : numéro de PR (optionnel)

        Output state fields:
          - pr_description (dict) : rapport de description généré
        """
        project_path = Path(state.get("project_path", "."))
        branch       = state.get("branch", "HEAD")
        base         = state.get("base", "main")
        owner        = state.get("owner", "")
        repo_name    = state.get("repo", "")
        pr_number: Optional[int] = state.get("pr_number")

        try:
            # Mode GitHub si owner + repo + pr_number disponibles
            if owner and repo_name and pr_number:
                from services.code_mode_client import github
                pr_data = state.get("pr_report") or {}
                report = generate_pr_description_github(
                    owner, repo_name, pr_number, github, pr_data
                )
            else:
                # Mode local (git log)
                report = generate_pr_description_local(project_path, branch, base)

            state["pr_description"] = self._serialize(report)
            logger.info(
                "PR description generated: %d commits, %d files, from_cache=%s",
                report.commits_used,
                report.files_changed,
                report.from_cache,
            )
        except Exception as e:
            logger.error("LCGitPRDescriptionAgent error: %s", e)
            state["pr_description"] = {
                "error":       str(e),
                "title":       "",
                "description": "",
                "from_cache":  False,
            }

        return state

    def _serialize(self, report: PRDescriptionReport) -> Dict[str, Any]:
        return {
            "title":         report.title,
            "description":   report.description,
            "commits_used":  report.commits_used,
            "files_changed": report.files_changed,
            "additions":     report.additions,
            "deletions":     report.deletions,
            "from_cache":    report.from_cache,
            "error":         report.error,
        }
