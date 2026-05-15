"""
lc_git_session_agent.py — LangChain GitSessionAgent wrapper.

Wraps the GitSessionTracker snapshot for integration into the LangGraph WatchGraph.

Responsibilities:
  1. Read the session snapshot from Redis (written by GitSessionTracker)
  2. Format a context string for the LLM
  3. Return serializable dict for the WatchState
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LCGitSessionAgent:
    """
    LangChain-compatible GitSessionAgent.
    Reads from Redis cache populated by the background GitSessionTracker.
    """

    def __init__(self, project_path: Path):
        self.project_path = project_path

    def _gs_key(self) -> str:
        """Redis key for git session snapshot."""
        from services.mcp_redis_service import key_hash, KEY_PREFIX
        return f"{KEY_PREFIX}gs:{key_hash(str(self.project_path))}"

    def get_session_context(self) -> Optional[Dict[str, Any]]:
        """
        Read the session snapshot from Redis.
        Returns a dict with session info, or None if not available.
        """
        try:
            from services.mcp_redis_service import get_mcp_redis
            redis = get_mcp_redis()
            gs_key = self._gs_key()
            raw = redis.get(gs_key)
            if not raw:
                return None

            # Parse JSON with fallback for single-quotes (defensive)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                import ast
                data = ast.literal_eval(raw)

            # Extract key fields for LLM context
            return {
                "level": data.get("level", "CLEAN"),
                "score": data.get("score", 0),
                "minutes_since_commit": data.get("minutes_since_commit", 0),
                "files_at_risk_count": len(data.get("files_at_risk", [])),
                "files_unanalyzed_count": len(data.get("files_unanalyzed", [])),
                "time_multiplier": data.get("time_multiplier", 1.0),
                "has_data": True,
            }
        except Exception as e:
            logger.debug("LCGitSessionAgent.get_session_context erreur : %s", e)
            return None

    def format_alert(self, context: Dict[str, Any]) -> Optional[str]:
        """Format a session alert for the LLM if level is WARN or CRITICAL."""
        level = context.get("level", "CLEAN")
        if level not in ("WARN", "CRITICAL"):
            return None

        score = context.get("score", 0)
        minutes = context.get("minutes_since_commit", 0)
        files_at_risk = context.get("files_at_risk_count", 0)
        files_unanalyzed = context.get("files_unanalyzed_count", 0)

        alert = (
            f"⚠️ SESSION GIT {level} — "
            f"score={score}, {minutes}min depuis dernier commit, "
            f"{files_at_risk} fichiers à risque, {files_unanalyzed} fichiers non analysés"
        )
        return alert


# Singleton (re-instantiated in watch_graph with real project_path)
lc_git_session_agent = LCGitSessionAgent(Path("."))
