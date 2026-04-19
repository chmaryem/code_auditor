"""
pr_analyzer.py — Analyse de Pull Requests via MCP Code Mode.

"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

async def analyze_pr(owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
    from smart_git.pr_review_agent import review_pr
    return await review_pr(owner, repo, pr_number)

async def resolve_pr_conflicts(owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
    """
    Résout les conflits d'une PR via MCP Code Mode.

    Délègue au ConflictResolutionAgent qui :
    1. Vérifie le statut mergeable
    2. Récupère les versions base/head
    3. Utilise resolve_single_file() (3-strategy retry)
    4. Pousse sur une branche de résolution

    Returns:
        dict: {success, resolved: [...], failed: [...], branch: "..."}
    """
    from smart_git.conflict_resolution_agent import resolve_pr_conflicts as _resolve
    return await _resolve(owner, repo, pr_number)

async def check_pr_merge_readiness(owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
    """
    Vérifie si une PR est prête à merger.

    Délègue au MergeAutomationAgent qui :
    1. Vérifie le statut mergeable
    2. Vérifie le CI/CD (check runs)
    3. Vérifie les reviews (approvals)
    4. Poste un rapport de readiness

    Returns:
        dict: {success, ready, mergeable, ci_pass, reviews_approved, ...}
    """
    from smart_git.merge_automation_agent import check_merge_readiness
    return await check_merge_readiness(owner, repo, pr_number)


def _parse_repo(repo_str: str):
    """Parse 'owner/repo' en (owner, repo)."""
    parts = repo_str.split("/")
    if len(parts) != 2:
        raise ValueError(f"Format attendu : owner/repo (reçu: {repo_str})")
    return parts[0], parts[1]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Code Auditor — PR Analyzer")
    parser.add_argument("action", choices=["check", "resolve"])
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--pr", type=int, required=True, help="PR number")
    args = parser.parse_args()

    owner, repo = _parse_repo(args.repo)

    if args.action == "check":
        asyncio.run(analyze_pr(owner, repo, args.pr))
    else:
        asyncio.run(resolve_pr_conflicts(owner, repo, args.pr))
