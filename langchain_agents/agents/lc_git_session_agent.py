"""
lc_git_session_agent.py — Smart Git Session Agent.

Role:
  Wrap GitSessionTracker and expose a clean dict output for LangGraph,
  ChatAgent, API, and plugin.

Uses existing:
  smart_git.git_session_tracker.GitSessionTracker
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


class LCGitSessionAgent:
    """
    SessionAgent answers:
      - current Git session risk
      - can I commit?
      - files at risk
      - unanalyzed modified files
    """

    def get_status(self, project_path: str) -> Dict[str, Any]:
        try:
            from smart_git.git_session_tracker import GitSessionTracker
        except Exception as e:
            return {
                "success": False,
                "error": f"Cannot import GitSessionTracker: {e}",
            }

        project = Path(project_path).resolve()

        cache_db = (
            project.parent
            / "code_auditor"
            / "data"
            / "cache"
            / "analysis_cache.db"
        )

        try:
            tracker = GitSessionTracker(
                project_path=project,
                cache_db=cache_db,
            )

            snapshot = tracker.force_check()

            if not snapshot:
                return {
                    "success": False,
                    "error": "Unable to calculate git session status",
                }

            return {
                "success": True,
                "score": snapshot.score,
                "level": snapshot.level,
                "minutes_since_commit": snapshot.minutes_since_commit,
                "time_multiplier": snapshot.time_multiplier,
                "total_critical": snapshot.total_critical,
                "total_high": snapshot.total_high,
                "total_bugs": snapshot.total_bugs,
                "files_at_risk": [
                    {
                        "path": file_risk.path,
                        "status": file_risk.status,
                        "staged": file_risk.staged,
                        "critical": file_risk.bugs_critical,
                        "high": file_risk.bugs_high,
                        "medium": file_risk.bugs_medium,
                        "low": file_risk.bugs_low,
                        "score": file_risk.score,
                        "has_analysis": file_risk.has_analysis,
                        "max_severity": file_risk.max_severity,
                    }
                    for file_risk in snapshot.files_at_risk
                ],
                "files_unanalyzed": snapshot.files_unanalyzed,
                "stats": snapshot.stats,
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Git session status failed: {e}",
            }


git_session_agent = LCGitSessionAgent()