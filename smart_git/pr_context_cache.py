"""
pr_context_cache.py — Axe (b) of the multi-agent refactor: shared memory for
the PR Cockpit tabs (Readiness / Review / Description), which are independent,
on-demand REST calls (no combined endpoint — each dashboard tab click is its
own request).

Two things live here:

  1. Step 1 — shared PR metadata (short TTL, ~90s). Readiness, Review, and
     Description each call github.get_pr_info() independently — Review and
     Description also each call github.get_pr_files() independently — for the
     SAME PR whenever a developer clicks through tabs. Whichever endpoint runs
     first populates this cache; any other endpoint hit within the window
     reuses it instead of re-calling the GitHub API. Purely a performance
     optimization — returned data shape is identical to a fresh call, and
     cache misses (including "Redis unavailable") fall through to the exact
     same direct GitHub call as before this existed.

  2. Step 3 — opportunistic collaboration (longer TTL, 15 min). When Review
     runs, it caches its security findings (critical/high/medium counts +
     verdict). If Readiness runs afterwards for the SAME PR within the TTL
     window, it reads these findings and factors them into its verdict/score.
     If Review was never run (the common case — most developers check
     Readiness first), there is nothing to read and Readiness behaves EXACTLY
     as before this existed. This is real agent collaboration via shared
     memory, without forcing the two tabs into a single combined call.

Uses Redis HASH storage (hset_dict/hgetall), not plain set/get: the
redis-mcp-server backing get_mcp_redis() silently drops plain string values
starting with '{' or '[' (see services/mcp_redis_service.py _escape_hval
comment) — a JSON-serialized dict/list would be dropped by set()/get(). Hash
operations already escape this (same pattern as AnalysisCacheMemory in
langchain_agents/memory/redis_memory.py), so we reuse it here too.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

_TTL_SECONDS = 90
_REVIEW_FINDINGS_TTL_SECONDS = 900  # 15 min — long enough to switch tabs after reading Review
_KEY_PREFIX = "ca:mem:pr_context"


def _get_redis():
    from services.mcp_redis_service import get_mcp_redis
    return get_mcp_redis()


def _pr_key(owner: str, repo: str, pr_number: int, field: str) -> str:
    return f"{_KEY_PREFIX}:{owner}/{repo}#{pr_number}:{field}"


def _read_cached(key: str) -> Optional[Any]:
    try:
        data = _get_redis().hgetall(key)
        if not data or "data" not in data:
            return None
        return json.loads(data["data"])
    except Exception:
        return None


def _write_cached(key: str, value: Any, ttl_seconds: int) -> None:
    try:
        _get_redis().hset_dict(
            key, {"data": json.dumps(value, ensure_ascii=False, default=str)},
            expire_seconds=ttl_seconds,
        )
    except Exception:
        pass


def fetch_pr_info_cached(github_service, owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
    """Read-through cache wrapper for github_service.get_pr_info()."""
    key = _pr_key(owner, repo, pr_number, "pr_info")
    cached = _read_cached(key)
    if cached is not None:
        return cached
    pr_info = github_service.get_pr_info(owner, repo, pr_number) or {}
    if pr_info:
        _write_cached(key, pr_info, _TTL_SECONDS)
    return pr_info


def fetch_pr_files_cached(github_service, owner: str, repo: str, pr_number: int) -> List[Dict[str, Any]]:
    """Read-through cache wrapper for github_service.get_pr_files()."""
    key = _pr_key(owner, repo, pr_number, "pr_files")
    cached = _read_cached(key)
    if cached is not None:
        return cached
    files = github_service.get_pr_files(owner, repo, pr_number) or []
    if files:
        _write_cached(key, files, _TTL_SECONDS)
    return files


def cache_review_findings(
    owner: str, repo: str, pr_number: int,
    critical: int, high: int, medium: int, verdict: str,
) -> None:
    """Called by review_pr() after a completed review — makes the findings
    available to a subsequent Readiness check on the same PR."""
    key = _pr_key(owner, repo, pr_number, "review_findings")
    _write_cached(key, {
        "critical": critical, "high": high, "medium": medium, "verdict": verdict,
    }, _REVIEW_FINDINGS_TTL_SECONDS)


def get_cached_review_findings(owner: str, repo: str, pr_number: int) -> Optional[Dict[str, Any]]:
    """Read by check_merge_readiness() — None if Review was never run (or the
    TTL expired) for this PR, in which case Readiness behaves as before."""
    return _read_cached(_pr_key(owner, repo, pr_number, "review_findings"))


# ── Session git context (Chat conversational follow-ups) ─────────────────────
#
# The Chat classifies every message independently — a natural follow-up like
# "what's the fix?" or "how do I merge it?" doesn't repeat "PR #7", so it has
# no PR/branch to resolve and silently falls back to a local-only branch
# analysis that answers a different question. Remembering the last PR the
# Chat resolved for this session lets such follow-ups stay on-topic.

_SESSION_CONTEXT_TTL_SECONDS = 1800  # 30 min — long enough for a natural follow-up


def save_session_git_context(session_id: str, owner: str, repo: str, pr_number: int) -> None:
    """Called after a git_question resolves to a specific PR (explicitly or
    via branch-name matching), so later messages in the same session can
    fall back to it when they don't name a PR/branch themselves."""
    if not session_id or not owner or not repo or not pr_number:
        return
    _write_cached(
        f"{_KEY_PREFIX}:session:{session_id}",
        {"owner": owner, "repo": repo, "pr_number": pr_number},
        _SESSION_CONTEXT_TTL_SECONDS,
    )


def get_session_git_context(session_id: str) -> Optional[Dict[str, Any]]:
    """None if no PR was discussed yet in this session (or the TTL expired) —
    callers should fall back to their pre-existing behavior in that case."""
    if not session_id:
        return None
    return _read_cached(f"{_KEY_PREFIX}:session:{session_id}")
