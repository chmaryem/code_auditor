"""
lc_git_conflict_agent.py — Smart Git Conflict Agent.

Role:
  Safe conflict detection and dry-run resolution wrapper.

Uses existing:
  smart_git.git_conflict_resolver.detect_conflict_files

Important:
  This first LangGraph version is SAFE by default.
  It detects conflicts and returns a dry-run report.
  It does NOT write files and does NOT push branches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


class LCGitConflictAgent:
    """
    ConflictAgent answers:
      - are there conflicts?
      - which files are conflicted?
      - can we prepare a dry-run resolution?
    """

    def detect_conflicts(self, project_path: str) -> Dict[str, Any]:
        try:
            from smart_git.git_conflict_resolver import detect_conflict_files
        except Exception as e:
            return {
                "success": False,
                "error": f"Cannot import conflict resolver: {e}",
                "has_conflicts": False,
                "conflict_files": [],
            }

        project = Path(project_path).resolve()

        try:
            files = detect_conflict_files(project)

            return {
                "success": True,
                "has_conflicts": len(files) > 0,
                "conflict_files": files,
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Conflict detection failed: {e}",
                "has_conflicts": False,
                "conflict_files": [],
            }

    def dry_run_resolution(self, project_path: str) -> Dict[str, Any]:
        """
        Safe mode:
          - detect conflicts
          - return files
          - do not write
          - do not push

        Later:
          - generate preview content
          - create diff preview
          - require confirmation before apply
        """
        detected = self.detect_conflicts(project_path)

        if not detected.get("success"):
            return detected

        if not detected.get("has_conflicts"):
            return {
                "success": True,
                "has_conflicts": False,
                "conflict_files": [],
                "message": "No conflict files detected",
                "dry_run": True,
            }

        return {
            "success": True,
            "has_conflicts": True,
            "conflict_files": detected.get("conflict_files", []),
            "message": (
                "Conflicts detected. Safe dry-run mode is active. "
                "No file was modified."
            ),
            "dry_run": True,
            "next_step": (
                "Add preview generation using resolve_single_file() and show a diff "
                "before applying changes."
            ),
        }


git_conflict_agent = LCGitConflictAgent()