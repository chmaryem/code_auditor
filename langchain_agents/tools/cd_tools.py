"""
cd_tools.py — LangChain @tool wrappers for the CDGraph.

Groups:
  A. Deployment tracking tools  — record, update, query deploy state
  B. Release readiness tools    — score before deploy
  C. Health check tools         — check environment after deploy
  D. Rollback tools             — advise on rollback strategy
  E. Monitor tools              — store/retrieve monitor sessions
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ── A. Deployment Tracking Tools ─────────────────────────────────────────────

@tool
def tool_record_deploy(
    repo:        str,
    environment: str,
    commit_sha:  str,
    version:     str,
    branch:      str       = "main",
    deploy_url:  str       = "",
    run_id:      str       = "",
) -> Dict[str, Any]:
    """
    Records a new deployment as 'deploying' in Redis.

    Args:
        repo:        "{owner}/{repo}"
        environment: "production" | "staging" | "dev"
        commit_sha:  Full commit SHA being deployed
        version:     Docker image tag (ex: "sha-abc1234")
        branch:      Source branch
        deploy_url:  URL to check health after deploy
        run_id:      GitHub Actions run ID

    Returns:
        {"deploy_id": str, "status": "deploying"}
    """
    try:
        from ci_cd.cd_deploy_tracker import CDDeployTracker
        tracker   = CDDeployTracker()
        deploy_id = tracker.record_deploy(
            repo=repo, environment=environment, commit_sha=commit_sha,
            version=version, branch=branch, deploy_url=deploy_url, run_id=run_id,
        )
        return {"deploy_id": deploy_id, "status": "deploying"}
    except Exception as e:
        logger.error("tool_record_deploy failed: %s", e)
        return {"deploy_id": "", "status": "error", "error": str(e)}


@tool
def tool_mark_deploy_success(
    deploy_id: str,
    health_ok: bool = True,
) -> bool:
    """
    Marks a deployment as successful and updates the rollback anchor.

    Args:
        deploy_id: The deploy_id from tool_record_deploy
        health_ok: Whether the health check passed

    Returns:
        True if updated successfully
    """
    try:
        from ci_cd.cd_deploy_tracker import CDDeployTracker
        CDDeployTracker().mark_deploy_success(deploy_id, health_ok=health_ok)
        return True
    except Exception as e:
        logger.error("tool_mark_deploy_success failed: %s", e)
        return False


@tool
def tool_mark_deploy_failure(
    deploy_id:      str,
    failure_reason: str = "",
) -> bool:
    """
    Marks a deployment as failed (preserves last-success rollback anchor).

    Args:
        deploy_id:      The deploy_id from tool_record_deploy
        failure_reason: Human-readable description of failure

    Returns:
        True if updated successfully
    """
    try:
        from ci_cd.cd_deploy_tracker import CDDeployTracker
        CDDeployTracker().mark_deploy_failure(deploy_id, failure_reason=failure_reason)
        return True
    except Exception as e:
        logger.error("tool_mark_deploy_failure failed: %s", e)
        return False


@tool
def tool_get_env_state(
    repo:        str,
    environment: str,
) -> Dict[str, Any]:
    """
    Returns the current state of a deployment environment.

    Args:
        repo:        "{owner}/{repo}"
        environment: "production" | "staging" | "dev"

    Returns:
        {current_deploy_id, status, version, commit_sha, branch,
         deploy_url, last_ok_deploy_id, last_ok_version, last_updated}
    """
    try:
        from ci_cd.cd_deploy_tracker import CDDeployTracker
        return CDDeployTracker().get_env_state(repo, environment)
    except Exception as e:
        logger.error("tool_get_env_state failed: %s", e)
        return {}


@tool
def tool_get_last_successful_deploy(
    repo:        str,
    environment: str,
) -> Dict[str, Any]:
    """
    Returns the last successful deployment (rollback target).

    Args:
        repo:        "{owner}/{repo}"
        environment: Target environment

    Returns:
        {deploy_id, version, commit_sha, branch, deploy_url, success_at}
        or {} if no successful deploy found
    """
    try:
        from ci_cd.cd_deploy_tracker import CDDeployTracker
        result = CDDeployTracker().get_last_successful_deploy(repo, environment)
        return result or {}
    except Exception as e:
        logger.error("tool_get_last_successful_deploy failed: %s", e)
        return {}


# ── B. Release Readiness Tools ───────────────────────────────────────────────

@tool
def tool_score_release_readiness(
    repo:        str,
    owner:       str,
    commit_sha:  str,
    run_id:      str = "",
    project_key: str = "",
    pr_number:   int = 0,
    environment: str = "production",
) -> Dict[str, Any]:
    """
    Computes the release readiness score before a deployment.

    Args:
        repo:        "{owner}/{repo}"
        owner:       Repository owner
        commit_sha:  HEAD commit SHA
        run_id:      GitHub Actions run ID
        project_key: SonarCloud project key
        pr_number:   Associated PR number (0 = direct push)
        environment: Target environment

    Returns:
        {score: float, verdict: str, blocking_reasons: list,
         warnings: list, component_scores: dict}
    """
    try:
        from ci_cd.cd_release_scorer import CDReleaseScorer
        scorer = CDReleaseScorer()
        report = scorer.score(
            repo=repo, owner=owner, commit_sha=commit_sha,
            run_id=run_id, project_key=project_key,
            pr_number=pr_number or None,
            environment=environment,
        )
        return {
            "score":            report.score,
            "verdict":          report.verdict,
            "blocking_reasons": report.blocking_reasons,
            "warnings":         report.warnings,
            "component_scores": report.component_scores,
            "markdown":         report.to_markdown(),
        }
    except Exception as e:
        logger.error("tool_score_release_readiness failed: %s", e)
        return {
            "score": 0.0, "verdict": "DEPLOY_BLOCKED",
            "blocking_reasons": [str(e)], "warnings": [],
            "component_scores": {}, "markdown": "",
        }


# ── C. Health Check Tools ────────────────────────────────────────────────────

@tool
def tool_check_health(
    deploy_url:  str,
    use_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Performs a single health check snapshot on a deployed service.

    Args:
        deploy_url:   URL to check (e.g. "https://myapp.com")
        use_fallback: Try /health /api/health paths if base URL fails

    Returns:
        {healthy: bool, grade: "OK"|"DEGRADED"|"DOWN",
         http_status: int, latency_ms: float, issues: list}
    """
    try:
        from ci_cd.cd_health_checker import CDHealthChecker
        checker = CDHealthChecker(deploy_url)
        report  = checker.check_with_fallback() if use_fallback else checker.check()
        return {
            "healthy":     report.healthy,
            "grade":       report.grade,
            "http_status": report.http_status,
            "latency_ms":  report.latency_ms,
            "issues":      report.issues,
            "deploy_url":  report.deploy_url,
        }
    except Exception as e:
        logger.error("tool_check_health failed: %s", e)
        return {
            "healthy": False, "grade": "DOWN",
            "http_status": 0, "latency_ms": 0.0,
            "issues": [str(e)], "deploy_url": deploy_url,
        }


# ── D. Rollback Advisory Tools ───────────────────────────────────────────────

@tool
def tool_advise_rollback(
    repo:             str,
    owner:            str,
    environment:      str,
    failed_deploy_id: str = "",
    failed_version:   str = "",
    failed_sha:       str = "",
) -> Dict[str, Any]:
    """
    Generates a rollback recommendation for a failed deployment.
    NEVER auto-triggers rollback — only suggests.

    Args:
        repo:             "{owner}/{repo}"
        owner:            Repository owner
        environment:      Target environment
        failed_deploy_id: deploy_id of the failed deployment
        failed_version:   Docker tag that failed (alternative to deploy_id)
        failed_sha:       Commit SHA that failed

    Returns:
        {target_version, command, risk_level, risk_reasons,
         diff_files, commits_apart, comment_body}
        or {"available": False} if no previous successful deploy found
    """
    try:
        from ci_cd.cd_rollback_advisor import CDRollbackAdvisor
        advisor = CDRollbackAdvisor()
        advice  = advisor.advise(
            repo=repo, owner=owner, environment=environment,
            failed_deploy_id=failed_deploy_id,
            failed_version=failed_version,
            failed_sha=failed_sha,
        )
        if not advice:
            return {"available": False, "reason": "No previous successful deploy found"}
        return {
            "available":      True,
            "target_version": advice.target_version,
            "target_sha":     advice.target_sha,
            "command":        advice.command,
            "risk_level":     advice.risk_level,
            "risk_reasons":   advice.risk_reasons,
            "diff_files":     advice.diff_files,
            "commits_apart":  advice.commits_apart,
            "comment_body":   advice.comment_body,
        }
    except Exception as e:
        logger.error("tool_advise_rollback failed: %s", e)
        return {"available": False, "error": str(e)}


# ── E. Monitor Tools ─────────────────────────────────────────────────────────

@tool
def tool_get_monitor_session(deploy_id: str) -> Dict[str, Any]:
    """
    Retrieves the stored post-deploy monitor session for a deployment.

    Args:
        deploy_id: The deploy_id from tool_record_deploy

    Returns:
        {final_grade, availability, avg_latency_ms, regression, issues, ...}
        or {} if not found
    """
    try:
        from services.mcp_redis_service import get_mcp_redis
        redis = get_mcp_redis()
        return redis.hgetall(f"cd:monitor:{deploy_id}") or {}
    except Exception as e:
        logger.error("tool_get_monitor_session failed: %s", e)
        return {}


@tool
def tool_get_deploy_stats(
    repo:        str,
    environment: str = "",
) -> Dict[str, Any]:
    """
    Returns deployment statistics for a repository.

    Args:
        repo:        "{owner}/{repo}"
        environment: Filter by environment (optional, "" = all)

    Returns:
        {total, successes, failures, success_rate, avg_duration_s,
         last_deploy_at, last_deploy_status}
    """
    try:
        from ci_cd.cd_deploy_tracker import CDDeployTracker
        return CDDeployTracker().get_deploy_stats(repo, environment)
    except Exception as e:
        logger.error("tool_get_deploy_stats failed: %s", e)
        return {"repo": repo, "total": 0, "successes": 0, "failures": 0}
