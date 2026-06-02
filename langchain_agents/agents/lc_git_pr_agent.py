"""
lc_git_pr_agent.py — Smart Git Pull Request Agent.

Role:
  Wrap PR review and PR readiness systems.

Uses existing:
  smart_git.pr_review_agent.review_pr
  smart_git.merge_automation_agent.check_merge_readiness
"""

from __future__ import annotations

from typing import Any, Dict


class LCGitPRAgent:
    """
    PRAgent answers:
      - review PR
      - check PR merge readiness
    """

    async def review_pr(
        self,
        owner: str,
        repo: str,
        pr_number: int,
    ) -> Dict[str, Any]:
        if not owner or not repo or not pr_number:
            return {
                "success": False,
                "error": "Missing owner/repo/pr_number",
            }

        try:
            from smart_git.pr_review_agent import review_pr
        except Exception as e:
            return {
                "success": False,
                "error": f"Cannot import PR review agent: {e}",
            }

        try:
            result = await review_pr(owner, repo, pr_number)
            return result if isinstance(result, dict) else {
                "success": False,
                "error": "PR review returned non-dict result",
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"PR review failed: {e}",
            }

    async def readiness(
        self,
        owner: str,
        repo: str,
        pr_number: int,
    ) -> Dict[str, Any]:
        if not owner or not repo or not pr_number:
            return {
                "success": False,
                "error": "Missing owner/repo/pr_number",
            }

        try:
            from smart_git.merge_automation_agent import check_merge_readiness
        except Exception as e:
            return {
                "success": False,
                "error": f"Cannot import merge readiness agent: {e}",
            }

        try:
            result = await check_merge_readiness(owner, repo, pr_number)
            return result if isinstance(result, dict) else {
                "success": False,
                "error": "PR readiness returned non-dict result",
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"PR readiness failed: {e}",
            }


git_pr_agent = LCGitPRAgent()