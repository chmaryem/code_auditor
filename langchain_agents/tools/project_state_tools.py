"""
project_state_tools.py — ChatAgent tools that surface existing project intelligence.

These are thin, defensive wrappers over systems that already compute their own
state (Git risk, CI/CD readiness, test gaps, security/quality, secrets). Each
tool returns a small structured summary — never the raw payload — so the chat
LLM can reason over it cheaply. Every tool degrades gracefully to
{"available": False, "reason": ...} instead of raising, since most of these
sources depend on local repo state, cached analysis, or optional network/auth
that may not be present in every project.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _unavailable(reason: str) -> Dict[str, Any]:
    return {"available": False, "reason": reason}


def tool_chat_git_risk(project_path: str) -> Dict[str, Any]:
    """Uncommitted-change risk for the working tree (local, no network)."""
    try:
        from smart_git.git_session_tracker import GitSessionTracker
        cache_db = Path(project_path) / ".codeaudit" / "session_cache.db"
        tracker = GitSessionTracker(project_path=Path(project_path), cache_db=cache_db)
        snapshot = tracker.get_snapshot() or tracker.force_check()
        if snapshot is None:
            return _unavailable("no session snapshot (clean tree or tracker not initialized)")
        return {
            "available": True,
            "score": snapshot.score,
            "level": snapshot.level,
            "files_at_risk": [
                {"file": fr.file, "critical": getattr(fr, "critical", 0), "high": getattr(fr, "high", 0)}
                for fr in (snapshot.files_at_risk or [])[:10]
            ],
            "files_unanalyzed": (snapshot.files_unanalyzed or [])[:10],
            "total_critical": snapshot.total_critical,
            "total_high": snapshot.total_high,
            "total_bugs": snapshot.total_bugs,
            "minutes_since_commit": snapshot.minutes_since_commit,
        }
    except Exception as e:
        logger.debug("tool_chat_git_risk failed: %s", e)
        return _unavailable(f"git risk check failed: {e}")


def tool_chat_secrets(project_path: str) -> Dict[str, Any]:
    """Secret scan of staged changes (local git diff, no network)."""
    try:
        from smart_git.git_secret_scanner import scan_staged_secrets
        report = scan_staged_secrets(Path(project_path))
        if report.error:
            return _unavailable(report.error)
        return {
            "available": True,
            "has_secrets": report.has_secrets,
            "critical_count": report.critical_count,
            "scanned_files": report.scanned_files,
            "blocked": report.blocked,
            "findings": [
                {"file": f.file, "rule": getattr(f, "rule", ""), "line": getattr(f, "line", 0)}
                for f in (report.findings or [])[:10]
            ],
        }
    except Exception as e:
        logger.debug("tool_chat_secrets failed: %s", e)
        return _unavailable(f"secret scan failed: {e}")


def tool_chat_test_gaps(project_path: str, target_file: Optional[str] = None) -> Dict[str, Any]:
    """Test coverage gap for the active file, if one is in context."""
    if not target_file:
        return _unavailable("no active file in context — open a file to check its test coverage")
    try:
        from agents.test_gap_agent import get_test_gap_agent
        agent = get_test_gap_agent(Path(project_path))
        status = agent.check(source_file=Path(target_file), parsed_entities=[])
        return {
            "available": True,
            "file": str(target_file),
            "missing": status.missing,
            "coverage_ratio": status.coverage_ratio,
            "untested_entities": status.untested_entities[:10],
            "framework": status.framework,
            "impact_score": status.impact_score,
            "needs_attention": status.needs_attention,
            "reason": status.reason,
        }
    except Exception as e:
        logger.debug("tool_chat_test_gaps failed: %s", e)
        return _unavailable(f"test gap check failed: {e}")


def tool_chat_security_quality(project_path: str, target_file: Optional[str] = None) -> Dict[str, Any]:
    """Most recent known issues for the active file, read from the analysis cache
    (does not re-run Watch analysis)."""
    if not target_file:
        return _unavailable("no active file in context — open a file to see its known issues")
    try:
        from services.cache_service import cache_service
        cached = cache_service.get_cached_analysis(Path(target_file))
        if not cached:
            return _unavailable("file has not been analyzed yet (no cache entry)")
        return {
            "available": True,
            "file": str(target_file),
            "analysis_excerpt": (cached.get("analysis") or "")[:1500],
            "relevant_knowledge": (cached.get("relevant_knowledge") or [])[:5],
        }
    except Exception as e:
        logger.debug("tool_chat_security_quality failed: %s", e)
        return _unavailable(f"cache lookup failed: {e}")


def tool_chat_ci_readiness(
    project_path: str,
    repo: Optional[str] = None,
    owner: Optional[str] = None,
    pr_number: Optional[int] = None,
) -> Dict[str, Any]:
    """Deployment readiness score, reusing the same scorer as the CI/CD dashboard.

    Requires repo/owner (GitHub) — without them there is no commit/PR to score,
    so this degrades to unavailable rather than guessing.
    """
    if not repo or not owner:
        return _unavailable("no GitHub repo/owner resolved for this project — open the CI/CD page once to link it")
    try:
        import subprocess
        head_sha = subprocess.run(
            ["git", "-C", project_path, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if not head_sha:
            return _unavailable("could not resolve current commit sha")

        from ci_cd.cd_release_scorer import CDReleaseScorer
        scorer = CDReleaseScorer()
        report = scorer.score(repo=repo, owner=owner, commit_sha=head_sha, pr_number=pr_number)
        return {
            "available": True,
            "score": report.score,
            "verdict": report.verdict,
            "blocking_reasons": report.blocking_reasons[:5],
            "warnings": report.warnings[:5],
            "component_scores": report.component_scores,
        }
    except Exception as e:
        logger.debug("tool_chat_ci_readiness failed: %s", e)
        return _unavailable(f"CI readiness scoring failed: {e}")


def tool_chat_project_state_summary(
    project_path: str,
    target_file: Optional[str] = None,
    repo: Optional[str] = None,
    owner: Optional[str] = None,
) -> Dict[str, Any]:
    """Aggregate snapshot across all project-state tools — used for the
    'what's the state of my project / can I deploy / next actions' intents.
    Each section degrades independently so partial data still answers."""
    return {
        "git_risk": tool_chat_git_risk(project_path),
        "secrets": tool_chat_secrets(project_path),
        "test_gaps": tool_chat_test_gaps(project_path, target_file),
        "security_quality": tool_chat_security_quality(project_path, target_file),
        "ci_readiness": tool_chat_ci_readiness(project_path, repo=repo, owner=owner),
    }
