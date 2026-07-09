"""
git_session_tools.py — LangChain @tool wrappers for Git Session monitoring.

Tools:
  - tool_git_session_status : Retrieve git session snapshot from Redis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.tools import tool


@tool
def tool_git_session_status(project_path: str) -> Optional[Dict[str, Any]]:
    """Retrieve the current git session status from Redis cache.

    Reads the snapshot written by the background GitSessionTracker (180s interval).
    Returns session level, score, time since last commit, and risk metrics.

    Args:
        project_path: Absolute path to the project root.

    Returns:
        Dict with level, score, minutes_since_commit, files_at_risk_count,
        files_unanalyzed_count, time_multiplier, has_data; or None if unavailable.
    """
    from langchain_agents.tools.git_tools import tool_session_status
    return tool_session_status.invoke({"project_path": str(project_path)})
