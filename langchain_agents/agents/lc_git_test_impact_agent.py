"""
lc_git_test_impact_agent.py — LangChain wrapper pour F4 : Test Impact Analysis.

Intègre GitTestImpactAnalyzer dans l'architecture multi-agent LangGraph.
Node dans le graph : node_test_impact
Intent détecté par : LCGitDecisionAgent → "test_impact"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from smart_git.git_test_impact import (
    analyze_test_impact_staged,
    analyze_test_impact_files,
    TestImpactReport,
)

logger = logging.getLogger(__name__)


class LCGitTestImpactAgent:
    """
    Agent LangChain pour l'analyse d'impact sur les tests.

    Utilisé par le LangGraph node `node_test_impact`.
    Analyse 100% locale (naming conventions + grep).
    """

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyse l'impact sur les tests et retourne le rapport.

        Input state fields:
          - project_path (str)     : chemin du projet git
          - changed_files (list)   : fichiers modifiés (optionnel, sinon git diff --cached)
          - staged_files (list)    : alternative pour les fichiers stagés

        Output state fields:
          - test_impact_report (dict) : rapport d'impact
        """
        project_path  = Path(state.get("project_path", "."))
        changed_files: List[str] = (
            state.get("changed_files")
            or state.get("staged_files")
            or []
        )

        try:
            if changed_files:
                report = analyze_test_impact_files(changed_files, project_path)
            else:
                report = analyze_test_impact_staged(project_path)

            state["test_impact_report"] = self._serialize(report)
            logger.info(
                "Test impact: %d/%d files covered, %d test files",
                report.covered_files,
                report.total_files,
                len(report.all_test_files),
            )
        except Exception as e:
            logger.error("LCGitTestImpactAgent error: %s", e)
            state["test_impact_report"] = {
                "error": str(e),
                "impacts": [],
                "total_files": 0,
                "covered_files": 0,
                "uncovered_files": 0,
                "all_test_files": [],
            }

        return state

    def _serialize(self, report: TestImpactReport) -> Dict[str, Any]:
        return {
            "total_files":     report.total_files,
            "covered_files":   report.covered_files,
            "uncovered_files": report.uncovered_files,
            "coverage_ratio":  round(report.coverage_ratio, 2),
            "has_gaps":        report.has_gaps,
            "all_test_files":  report.all_test_files,
            "impacts": [
                {
                    "source_file":      i.source_file,
                    "test_files":       i.test_files,
                    "missing_tests":    i.missing_tests,
                    "discovery_method": i.discovery_method,
                }
                for i in report.impacts
            ],
            "error": report.error,
        }
