"""
lc_git_secret_agent.py — LangChain wrapper pour F1 : Secret Detector.

Intègre GitSecretScanner dans l'architecture multi-agent LangGraph.
Node dans le graph : node_secret_scan
Intent détecté par : LCGitDecisionAgent → "secret_scan"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from smart_git.git_secret_scanner import scan_staged_secrets, SecretScanReport

logger = logging.getLogger(__name__)


class LCGitSecretAgent:
    """
    Agent LangChain pour la détection de secrets dans les fichiers stagés.

    Utilisé par le LangGraph node `node_secret_scan`.
    Ne fait aucun appel LLM — 100% local (regex).
    """

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scanne les fichiers stagés et retourne le rapport de secrets.

        Input state fields:
          - project_path (str) : chemin du projet git

        Output state fields:
          - secret_scan_report (dict) : rapport sérialisé
        """
        project_path = Path(state.get("project_path", "."))

        try:
            report = scan_staged_secrets(project_path)
            state["secret_scan_report"] = self._serialize(report)
            logger.info(
                "Secret scan: %d findings in %d files",
                len(report.findings),
                report.scanned_files,
            )
        except Exception as e:
            logger.error("LCGitSecretAgent error: %s", e)
            state["secret_scan_report"] = {
                "error": str(e),
                "findings": [],
                "scanned_files": 0,
                "blocked": False,
            }

        return state

    def _serialize(self, report: SecretScanReport) -> Dict[str, Any]:
        return {
            "findings": [
                {
                    "file_path":    f.file_path,
                    "line_number":  f.line_number,
                    "secret_type":  f.secret_type,
                    "masked_text":  f.masked_text,
                    "severity":     f.severity,
                    "context":      f.context,
                }
                for f in report.findings
            ],
            "scanned_files": report.scanned_files,
            "blocked":       report.blocked,
            "has_secrets":   report.has_secrets,
            "critical_count": report.critical_count,
            "error":         report.error,
        }
