
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict

from langgraph.graph import StateGraph, END 

from langchain_agents.graphs.state import CDState

logger = logging.getLogger(__name__)



def node_fetch_deployment(state: CDState) -> CDState:
   
    run_id = state.get("run_id", "")
    repo   = state.get("repo", "")
    owner  = state.get("owner", "")

    deploy_url = os.environ.get("DEPLOY_URL", "")
    environment = os.environ.get("DEPLOY_ENV", "production")
    version = ""

    # Try to extract version from run logs
    try:
        from langchain_agents.tools.ci_tools import tool_fetch_run_logs
        result = tool_fetch_run_logs.invoke({
            "owner": owner,
            "repo":  repo.split("/")[-1] if "/" in repo else repo,
            "run_id": run_id,
        })
        logs = result.get("logs", "")

        # Look for image tag patterns in logs
        import re
        for pattern in [
            r"sha-([a-f0-9]{7,8})",
            r"tag[s]?\s*[:=]\s*([^\s,\n]+)",
            r"image\s*[:=]\s*[^\s]+:([^\s\n]+)",
        ]:
            m = re.search(pattern, logs, re.IGNORECASE)
            if m:
                version = m.group(1)[:30]
                break

        if not version:
            sha = state.get("head_sha", "")
            version = f"sha-{sha[:7]}" if sha else "latest"

        # Extract DEPLOY_URL from logs if not in env
        if not deploy_url:
            m = re.search(r"https?://[^\s<>\"']+", logs)
            if m:
                deploy_url = m.group(0)[:200]

    except Exception as e:
        logger.debug("[CDGraph] fetch_deployment logs error: %s", e)
        sha = state.get("head_sha", "")
        version = f"sha-{sha[:7]}" if sha else "latest"

    # Record deployment in tracker — skipped in dry_run (preview) mode: no
    # real deploy happened, so nothing should land in the Redis deploy
    # history. Leaving deploy_id empty also naturally short-circuits the
    # mark_success/mark_failure calls in monitor_health/index_result below,
    # since those are already guarded by `if deploy_id:`.
    deploy_id = ""
    if not state.get("dry_run"):
        try:
            from langchain_agents.tools.cd_tools import tool_record_deploy
            result = tool_record_deploy.invoke({
                "repo":        repo,
                "environment": environment,
                "commit_sha":  state.get("head_sha", ""),
                "version":     version,
                "branch":      state.get("pr_branch", "main"),
                "deploy_url":  deploy_url,
                "run_id":      run_id,
            })
            deploy_id = result.get("deploy_id", "")
        except Exception as e:
            logger.error("[CDGraph] tool_record_deploy failed: %s", e)
    else:
        logger.info("[CDGraph] dry_run=True — skipping deploy record persistence")

    logger.info(
        "[CDGraph] fetch_deployment — env=%s ver=%s url=%s deploy_id=%s",
        environment, version, deploy_url, deploy_id
    )

    return {
        **state,
        "version":      version,
        "deploy_url":   deploy_url,
        "environment":  environment,
        "deploy_id":    deploy_id,
        "deploy_status": "deploying",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — pre_deploy_risk_score
# ─────────────────────────────────────────────────────────────────────────────

def node_pre_deploy_risk_score(state: CDState) -> CDState:
    """
    Computes the release readiness score BEFORE marking deploy success/fail.
    Result is used in the final PR comment and notification.
    """
    try:
        from langchain_agents.tools.cd_tools import tool_score_release_readiness
        result = tool_score_release_readiness.invoke({
            "repo":        state.get("repo", ""),
            "owner":       state.get("owner", ""),
            "commit_sha":  state.get("head_sha", ""),
            "run_id":      state.get("run_id", ""),
            "project_key": state.get("project_key", ""),
            "pr_number":   state.get("pr_number") or 0,
            "environment": state.get("environment", "production"),
        })
        logger.info(
            "[CDGraph] readiness score=%.0f verdict=%s",
            result.get("score", 0), result.get("verdict", "?")
        )
        return {
            **state,
            "readiness_score":            result.get("score", 0.0),
            "readiness_verdict":          result.get("verdict", "DEPLOY_WARN"),
            "readiness_blocking_reasons": result.get("blocking_reasons", []),
            "readiness_warnings":         result.get("warnings", []),
            "readiness_components":       result.get("component_scores", {}),
            "readiness_markdown":         result.get("markdown", ""),
        }
    except Exception as e:
        logger.error("[CDGraph] pre_deploy_risk_score failed: %s", e)
        return {**state, "readiness_score": 50.0, "readiness_verdict": "DEPLOY_WARN"}


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — check_environment
# ─────────────────────────────────────────────────────────────────────────────

def node_check_environment(state: CDState) -> CDState:
    """
    Fetches current env state and last successful deploy from Redis.
    Used as baseline for rollback and regression detection.
    """
    try:
        from langchain_agents.tools.cd_tools import (
            tool_get_env_state, tool_get_last_successful_deploy,
        )
        repo = state.get("repo", "")
        env  = state.get("environment", "production")

        env_state   = tool_get_env_state.invoke({"repo": repo, "environment": env})
        last_ok     = tool_get_last_successful_deploy.invoke({"repo": repo, "environment": env})

        logger.info(
            "[CDGraph] env_state status=%s last_ok_ver=%s",
            env_state.get("status", "?"), last_ok.get("version", "none")
        )
        return {**state, "env_state": env_state, "last_ok_deploy": last_ok}
    except Exception as e:
        logger.error("[CDGraph] check_environment failed: %s", e)
        return {**state, "env_state": {}, "last_ok_deploy": {}}


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — monitor_health
# ─────────────────────────────────────────────────────────────────────────────

def node_monitor_health(state: CDState) -> CDState:
    """
    Runs a short health check window (or single check if no URL).
    Uses a 5-minute window for production, single check for others.
    """
    deploy_url  = state.get("deploy_url", "")
    conclusion  = state.get("run_conclusion", "")
    environment = state.get("environment", "production")
    deploy_id   = state.get("deploy_id", "")

    if not deploy_url:
        logger.info("[CDGraph] monitor_health — no deploy_url, skipping")
        return {
            **state,
            "monitor_grade":          "HEALTHY" if conclusion == "success" else "DOWN",
            "monitor_availability":   100.0 if conclusion == "success" else 0.0,
            "monitor_avg_latency_ms": 0.0,
            "monitor_regression":     False,
            "monitor_issues":         ["No DEPLOY_URL configured"],
        }

    if conclusion != "success":
        # Deploy job failed — skip monitoring, go straight to failure analysis
        logger.info("[CDGraph] monitor_health — deploy failed, skip monitor")
        return {
            **state,
            "monitor_grade":      "DOWN",
            "monitor_regression": False,
            "monitor_issues":     ["Deploy job did not succeed"],
        }

    try:
        # Single health check snapshot (fast path) — one probe is enough for
        # the graph flow.
        from langchain_agents.tools.cd_tools import tool_check_health
        result = tool_check_health.invoke({
            "deploy_url":   deploy_url,
            "use_fallback": True,
        })

        health_ok = result.get("healthy", False)
        grade     = result.get("grade", "DOWN")

        logger.info(
            "[CDGraph] health check → %s latency=%.0fms status=%d",
            grade, result.get("latency_ms", 0), result.get("http_status", 0)
        )

        # Mark deploy success/failure in tracker
        if deploy_id:
            from langchain_agents.tools.cd_tools import (
                tool_mark_deploy_success, tool_mark_deploy_failure
            )
            if health_ok:
                tool_mark_deploy_success.invoke({
                    "deploy_id": deploy_id,
                    "health_ok": True,
                })
            else:
                tool_mark_deploy_failure.invoke({
                    "deploy_id":      deploy_id,
                    "failure_reason": f"Health check failed: {grade}",
                })

        return {
            **state,
            "deploy_status":          "success" if health_ok else "failed",
            "monitor_grade":          grade,
            "monitor_availability":   100.0 if health_ok else 0.0,
            "monitor_avg_latency_ms": result.get("latency_ms", 0.0),
            "monitor_regression":     not health_ok,
            "monitor_issues":         result.get("issues", []),
        }
    except Exception as e:
        logger.error("[CDGraph] monitor_health failed: %s", e)
        return {
            **state,
            "monitor_grade":   "DOWN",
            "monitor_issues":  [str(e)],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Router — after monitor_health
# ─────────────────────────────────────────────────────────────────────────────

def route_after_monitor(state: CDState) -> str:
    grade      = state.get("monitor_grade", "HEALTHY")
    conclusion = state.get("run_conclusion", "")
    status     = state.get("deploy_status", "")

    if grade == "DOWN" or status == "failed" or conclusion == "failure":
        return "analyze_failure"
    return "post_deploy_report"


# ─────────────────────────────────────────────────────────────────────────────
# Node 5 — analyze_failure
# ─────────────────────────────────────────────────────────────────────────────

def node_analyze_failure(state: CDState) -> CDState:
    """
    Uses LLM to analyze deploy failure root cause from logs.
    Same LLM factory pattern as CIGraph.
    """
    try:
        from ci_cd.pipeline_failure_analyzer import PipelineFailureAnalyzer
        analyzer = PipelineFailureAnalyzer()

        logs        = state.get("logs", "") or ""
        stage       = state.get("stage_failed", "deploy")
        monitor_iss = " | ".join(state.get("monitor_issues", []))
        combined    = f"Stage: {stage}\nHealth issues: {monitor_iss}\nLogs:\n{logs[-2000:]}"

        result = analyzer.analyze(
            logs=combined,
            stage_failed=stage or "deploy",
            repo=state.get("repo", ""),
            run_id=state.get("run_id", ""),
        )
        logger.info("[CDGraph] analyze_failure — root_cause extracted")
        return {
            **state,
            "deploy_failure_reason": result.get("root_cause", ""),
            "deploy_suggested_fix":  result.get("suggested_fix", ""),
        }
    except Exception as e:
        logger.error("[CDGraph] analyze_failure failed: %s", e)
        monitor_iss = state.get("monitor_issues", [])
        return {
            **state,
            "deploy_failure_reason": f"Deploy failure detected. Health check issues: {monitor_iss}",
            "deploy_suggested_fix":  "Check deploy logs and server connectivity.",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 6 — suggest_rollback
# ─────────────────────────────────────────────────────────────────────────────

def node_suggest_rollback(state: CDState) -> CDState:
    """Calls CDRollbackAdvisor and stores result in state."""
    try:
        from langchain_agents.tools.cd_tools import tool_advise_rollback
        result = tool_advise_rollback.invoke({
            "repo":             state.get("repo", ""),
            "owner":            state.get("owner", ""),
            "environment":      state.get("environment", "production"),
            "failed_deploy_id": state.get("deploy_id", ""),
            "failed_version":   state.get("version", ""),
            "failed_sha":       state.get("head_sha", ""),
        })
        available = result.get("available", False)
        logger.info(
            "[CDGraph] rollback advice — available=%s risk=%s",
            available, result.get("risk_level", "N/A")
        )
        return {
            **state,
            "rollback_available":   available,
            "rollback_command":     result.get("command", ""),
            "rollback_risk":        result.get("risk_level", "HIGH"),
            "rollback_risk_reasons": result.get("risk_reasons", []),
            "rollback_comment_body": result.get("comment_body", ""),
        }
    except Exception as e:
        logger.error("[CDGraph] suggest_rollback failed: %s", e)
        return {**state, "rollback_available": False}


# ─────────────────────────────────────────────────────────────────────────────
# Node 7 — post_deploy_report
# ─────────────────────────────────────────────────────────────────────────────

def node_post_deploy_report(state: CDState) -> CDState:
    """
    Builds and posts a structured PR comment with:
      - Release readiness score
      - Health check results
      - Rollback advice (if applicable)
    """
    pr_number = state.get("pr_number")
    owner     = state.get("owner", "")
    repo_name = state.get("repo", "").split("/")[-1]

    # Build comment body
    sections = []

    readiness_md = state.get("readiness_markdown", "")
    if readiness_md:
        sections.append(readiness_md)

    grade = state.get("monitor_grade", "")
    if grade:
        grade_icon = {"HEALTHY": "✅", "DEGRADED": "⚠️", "DOWN": "🔴"}.get(grade, "❓")
        avail = state.get("monitor_availability", 0.0)
        lat   = state.get("monitor_avg_latency_ms", 0.0)
        sections.append(
            f"\n## {grade_icon} Post-Deploy Health — `{grade}`\n\n"
            f"| Metric | Value |\n|---|---|\n"
            f"| Availability | {avail:.1f}% |\n"
            f"| Avg Latency | {lat:.0f}ms |\n"
        )

    reason = state.get("deploy_failure_reason", "")
    fix    = state.get("deploy_suggested_fix", "")
    if reason:
        sections.append(
            f"\n## 🔍 Failure Analysis\n\n**Root cause:**\n{reason}\n\n"
            + (f"**Suggested fix:**\n{fix}" if fix else "")
        )

    rollback_body = state.get("rollback_comment_body", "")
    if rollback_body:
        sections.append(f"\n{rollback_body}")

    comment_posted = False
    body = ""
    if sections:
        body = "\n\n---\n".join(sections)
        body = f"# 🚀 Code Auditor — CD Intelligence Report\n\n{body}"

        # dry_run (dashboard preview) → build the report for display but never
        # post it. Avoids spamming a PR with a comment for a "deployment" that
        # didn't actually happen.
        if not state.get("dry_run") and pr_number and owner and repo_name:
            try:
                from langchain_agents.tools.ci_tools import tool_post_pr_comment
                comment_posted = tool_post_pr_comment.invoke({
                    "owner":     owner,
                    "repo":      repo_name,
                    "pr_number": pr_number,
                    "body":      body[:65000],
                })
                logger.info("[CDGraph] PR comment posted: %s", comment_posted)
            except Exception as e:
                logger.error("[CDGraph] post_deploy_report PR comment failed: %s", e)

    return {**state, "comment_posted": comment_posted, "report_markdown": body}


# ─────────────────────────────────────────────────────────────────────────────
# Node 8 — index_result
# ─────────────────────────────────────────────────────────────────────────────

def node_index_result(state: CDState) -> CDState:
    """Finalizes the deploy record in Redis based on actual outcome."""
    deploy_id  = state.get("deploy_id", "")
    status     = state.get("deploy_status", "")
    conclusion = state.get("run_conclusion", "")

    if deploy_id:
        try:
            from ci_cd.cd_deploy_tracker import CDDeployTracker
            tracker = CDDeployTracker()
            if status == "success" or (conclusion == "success" and not status):
                tracker.mark_deploy_success(
                    deploy_id,
                    health_ok=state.get("monitor_grade") == "HEALTHY",
                )
            elif status == "failed" or conclusion == "failure":
                tracker.mark_deploy_failure(
                    deploy_id,
                    failure_reason=state.get("deploy_failure_reason", "")[:300],
                )
        except Exception as e:
            logger.error("[CDGraph] index_result tracker update failed: %s", e)

    return {**state, "indexed": True}


# ─────────────────────────────────────────────────────────────────────────────
# Node 9 — notify
# ─────────────────────────────────────────────────────────────────────────────

def node_notify(state: CDState) -> CDState:
    """Renders rich terminal notification via LCCDNotifier."""
    try:
        from langchain_agents.agents.lc_cd_notifier import LCCDNotifier
        notifier = LCCDNotifier()
        level    = notifier.notify(state)
        return {**state, "notification_level": level}
    except Exception as e:
        logger.error("[CDGraph] notify failed: %s", e)
        return {**state, "notification_level": "WARN"}


# ─────────────────────────────────────────────────────────────────────────────
# Graph assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_cd_graph():
    """Compiles and returns the CDGraph StateGraph."""
    g = StateGraph(CDState)

    g.add_node("fetch_deployment",       node_fetch_deployment)
    g.add_node("pre_deploy_risk_score",  node_pre_deploy_risk_score)
    g.add_node("check_environment",      node_check_environment)
    g.add_node("monitor_health",         node_monitor_health)
    g.add_node("analyze_failure",        node_analyze_failure)
    g.add_node("suggest_rollback",       node_suggest_rollback)
    g.add_node("post_deploy_report",     node_post_deploy_report)
    g.add_node("index_result",           node_index_result)
    g.add_node("notify",                 node_notify)

    g.set_entry_point("fetch_deployment")

    g.add_edge("fetch_deployment",       "pre_deploy_risk_score")
    g.add_edge("pre_deploy_risk_score",  "check_environment")
    g.add_edge("check_environment",      "monitor_health")

    g.add_conditional_edges(
        "monitor_health",
        route_after_monitor,
        {
            "analyze_failure":   "analyze_failure",
            "post_deploy_report": "post_deploy_report",
        },
    )

    g.add_edge("analyze_failure",   "suggest_rollback")
    g.add_edge("suggest_rollback",  "post_deploy_report")
    g.add_edge("post_deploy_report", "index_result")
    g.add_edge("index_result",       "notify")
    g.add_edge("notify",             END)

    return g.compile()


# ── Public invoke function ────────────────────────────────────────────────────

_cd_graph = None


def invoke_cd_run(
    run_id:      str,
    repo:        str,
    owner:       str,
    project_key: str = "",
    pr_number:   int = None,
    head_sha:    str = "",
    pr_branch:   str = "",
    stage_failed: str = "",
    run_conclusion: str = "",
    run_duration_seconds: int = 0,
    logs:        str = "",
    dry_run:     bool = False,
) -> Dict[str, Any]:
    """
    Invokes the CDGraph for a completed deploy workflow run.

    Args match invoke_ci_run() for drop-in compatibility with CIPoller.

    dry_run: when True, runs the full graph (readiness score, health check,
    failure analysis, rollback advice) without any side effect — no Redis
    deploy record, no PR comment. Used by the dashboard's on-demand preview,
    since the automatic trigger path (CIPoller reacting to a real
    publish/deploy job) is never reachable from workflow_dispatch-triggered
    runs (see node_fetch_deployment / node_post_deploy_report).
    """
    global _cd_graph
    if _cd_graph is None:
        _cd_graph = build_cd_graph()

    initial: CDState = {
        "run_id":              str(run_id),
        "repo":                repo,
        "owner":               owner,
        "project_key":         project_key or "",
        "pr_number":           pr_number,
        "head_sha":            head_sha or "",
        "pr_branch":           pr_branch or "",
        "environment":         os.environ.get("DEPLOY_ENV", "production"),
        "deploy_id":           "",
        "version":             "",
        "deploy_url":          os.environ.get("DEPLOY_URL", ""),
        "run_conclusion":      run_conclusion or "",
        "deploy_status":       "",
        "stage_failed":        stage_failed or "",
        "logs":                logs or "",
        # Defaults
        "readiness_score":             0.0,
        "readiness_verdict":           "DEPLOY_WARN",
        "readiness_blocking_reasons":  [],
        "readiness_warnings":          [],
        "readiness_components":        {},
        "readiness_markdown":          "",
        "env_state":                   {},
        "last_ok_deploy":              {},
        "monitor_grade":               "",
        "monitor_availability":        0.0,
        "monitor_avg_latency_ms":      0.0,
        "monitor_regression":          False,
        "monitor_issues":              [],
        "deploy_failure_reason":       "",
        "deploy_suggested_fix":        "",
        "rollback_available":          False,
        "rollback_command":            "",
        "rollback_risk":               "",
        "rollback_risk_reasons":       [],
        "rollback_comment_body":       "",
        "comment_posted":              False,
        "report_markdown":             "",
        "indexed":                     False,
        "notification_level":          "OK",
        "dry_run":                     dry_run,
    }

    try:
        start  = time.time()
        result = _cd_graph.invoke(initial)
        elapsed = round(time.time() - start, 2)
        logger.info(
            "[CDGraph] run=%s grade=%s verdict=%s (%.1fs)",
            run_id,
            result.get("monitor_grade", "?"),
            result.get("notification_level", "?"),
            elapsed,
        )
        return result
    except Exception as e:
        logger.error("[CDGraph] invoke failed for run=%s: %s", run_id, e)
        return {**initial, "deploy_failure_reason": f"CDGraph error: {e}"}
