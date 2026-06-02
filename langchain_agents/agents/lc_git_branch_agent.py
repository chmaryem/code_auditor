"""
lc_git_branch_agent.py — Smart Git Branch Agent.

Role:
  Wrap GitBranchAnalyzer and expose a serializable dict report.

Uses existing:
  smart_git.git_branch_analyzer.GitBranchAnalyzer
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict


class LCGitBranchAgent:
    """
    BranchAgent answers:
      - is my branch ready to merge?
      - what are the risky files?
      - what is the merge verdict?
    """

    def analyze_branch(
        self,
        project_path: str,
        branch: str = "HEAD",
        base: str = "main",
        orchestrator=None,
    ) -> Dict[str, Any]:
        try:
            from smart_git.git_branch_analyzer import GitBranchAnalyzer
        except Exception as e:
            return {
                "success": False,
                "error": f"Cannot import GitBranchAnalyzer: {e}",
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
            analyzer = GitBranchAnalyzer(
                project_path=project,
                cache_db=cache_db,
                orchestrator=orchestrator,
            )

            report = analyzer.analyze(
                branch=branch or "HEAD",
                base=base or "main",
            )

            return {
                "success": True,
                "branch": report.branch,
                "base": report.base,
                "merge_base_hash": report.merge_base_hash,
                "commits": report.commits,
                "files": [asdict(file_report) for file_report in report.files],
                "conflict_risks": report.conflict_risks,
                "total_score": report.total_score,
                "verdict": report.verdict,
                "recommendation": report.recommendation,
                "total_critical": report.total_critical,
                "total_high": report.total_high,
                "files_clean_count": len(report.files_clean),
                "files_with_issues_count": len(report.files_with_issues),
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Branch analysis failed: {e}",
            }


git_branch_agent = LCGitBranchAgent()