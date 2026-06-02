"""
lc_git_diff_agent.py — Smart Git Diff Agent.

Role:
  Wrap existing Git diff helpers and commit message generator.

Uses existing:
  smart_git.git_diff_parser
  smart_git.git_commit_msg
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


class LCGitDiffAgent:
    """
    DiffAgent answers:
      - summarize current changes
      - list staged/uncommitted files
      - generate commit message
    """

    def get_changes(self, project_path: str) -> Dict[str, Any]:
        try:
            from smart_git.git_diff_parser import (
                get_uncommitted_files,
                get_staged_files,
                get_session_stats,
                is_git_repo,
            )
        except Exception as e:
            return {
                "success": False,
                "error": f"Cannot import git diff parser: {e}",
            }

        project = Path(project_path).resolve()

        try:
            if not is_git_repo(project):
                return {
                    "success": False,
                    "error": f"Not a git repository: {project}",
                }

            uncommitted = get_uncommitted_files(project)
            staged = get_staged_files(project)
            stats = get_session_stats(project)

            return {
                "success": True,
                "uncommitted_files": uncommitted,
                "staged_files": staged,
                "session_stats": stats,
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read git changes: {e}",
            }

    def generate_commit_message(self, project_path: str) -> Dict[str, Any]:
        try:
            from smart_git.git_commit_msg import generate_commit_message
        except Exception as e:
            return {
                "success": False,
                "commit_message": "",
                "error": f"Cannot import commit message generator: {e}",
            }

        project = Path(project_path).resolve()

        try:
            message = generate_commit_message(project)

            return {
                "success": bool(message),
                "commit_message": message or "",
                "error": "" if message else "No staged diff found",
            }

        except Exception as e:
            return {
                "success": False,
                "commit_message": "",
                "error": f"Commit message generation failed: {e}",
            }


git_diff_agent = LCGitDiffAgent()