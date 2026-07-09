"""
lc_git_helpers.py — Shared leaf helpers for the Smart Git role-agents.

Kept dependency-free (stdlib + urllib only, no graph/agent imports) so both
WorkingCopyAgent and PRAgent can import it without circular imports. These
were previously inline in smart_git_dispatch.py; extracted verbatim so the
branch→PR resolution behavior validated today is preserved exactly.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.request
from typing import Any, Dict, List


def normalize_branch_name(name: str) -> str:
    return re.sub(r"[\s_]+", "-", (name or "").strip().strip("`'\"").lower())


def list_open_pr_heads(owner: str, repo: str) -> List[Dict[str, Any]]:
    """
    Direct REST call (same pattern as api/git_router.py's /prs — the
    github-mcp-server's list_open_prs reads are unreliable, see the
    GitHub REST-First migration). Only needs number + head ref.
    """
    token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    if not token or not owner or not repo:
        return []
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls?state=open&per_page=50"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "code-auditor/1.0",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            prs = json.loads(resp.read().decode())
        return [
            {"number": pr.get("number"), "head": pr.get("head", {}).get("ref", "")}
            for pr in prs if isinstance(pr, dict)
        ]
    except Exception:
        return []


async def resolve_branch_to_pr(owner: str, repo: str, branch_name: str) -> int:
    """
    If `branch_name` (named by the developer in free text) matches an open
    PR's head branch, return its number. Used so branch_readiness can
    delegate to the GitHub-backed PR readiness check instead of a
    local-only git analysis, which silently reports a false "MERGE_OK"
    when the named branch was never fetched into the local repo.
    """
    target = normalize_branch_name(branch_name)
    if not target:
        return 0
    prs = await asyncio.to_thread(list_open_pr_heads, owner, repo)
    for pr in prs:
        if normalize_branch_name(pr.get("head", "")) == target:
            return int(pr.get("number") or 0)
    return 0
