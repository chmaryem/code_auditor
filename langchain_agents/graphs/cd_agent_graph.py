"""
cd_agent_graph.py — Agentic CDGraph (supervisor + specialist agents).

Parallel, feature-flagged alternative to cd_graph.py. invoke_cd_agent_run
mirrors invoke_cd_run (same signature incl. dry_run and same return shape), so
the dashboard preview keeps working unchanged.

Hybrid design:
  - Deterministic STRUCTURED nodes are REUSED from cd_graph.py
    (fetch_deployment, pre_deploy_risk_score, check_environment,
    monitor_health, post_deploy_report, index_result, notify). These populate
    the numeric fields the dashboard needs (readiness score/verdict, health
    grade/latency).
  - The REASONING is done by real tool-calling agents (cd_agents.py): a
    supervisor routes to a Deploy Analysis Agent and a Rollback Agent, which
    autonomously call cd_tools.

Flow:
  fetch_deployment → pre_deploy_risk_score → check_environment → monitor_health
    → supervise → [deploy_analysis] → [rollback] → consolidate
    → post_deploy_report → index_result → notify → END
  (conditional edges: only the specialists picked by the supervisor's route
  are actually visited — a healthy deploy skips both and goes straight to
  consolidate.)
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from langgraph.graph import StateGraph, END

from langchain_agents.graphs.state import CDState
from langchain_agents.graphs.cd_graph import (
    node_fetch_deployment,
    node_pre_deploy_risk_score,
    node_check_environment,
    node_monitor_health,
    node_post_deploy_report,
    node_index_result,
    node_notify,
)
from langchain_agents.graphs.ci_agent_graph import (
    _parse_root_cause_fix, _extract_section, _is_llm_error_text,
)

logger = logging.getLogger(__name__)


class CDAgentState(CDState, total=False):
    """CDState + agentic orchestration fields."""
    route: List[str]
    deploy_analysis_output: str
    rollback_output: str
    rollback_raw: Dict[str, Any]
    deploy_used_tools: bool        # did the deploy-analysis agent call any tool?
    quality_score: float           # deterministic quality guard (0-1)
    quality_flags: List[str]


# ── Agent nodes ────────────────────────────────────────────────────────────────

def node_supervise_cd(state: CDAgentState) -> CDAgentState:
    from langchain_agents.agents.cd_agents import get_cd_supervisor
    route = get_cd_supervisor().route(
        monitor_grade=state.get("monitor_grade", ""),
        run_conclusion=state.get("run_conclusion", ""),
        deploy_status=state.get("deploy_status", ""),
    )
    logger.info("[CDAgentGraph] supervisor route=%s", route or "none")
    return {**state, "route": route}


def _cd_context(state: CDAgentState) -> Dict[str, Any]:
    repo = state.get("repo", "")
    return {
        "repo": repo,
        "owner": state.get("owner", ""),
        "environment": state.get("environment", "production"),
        "deploy_url": state.get("deploy_url", ""),
        "deploy_id": state.get("deploy_id", ""),
        "version": state.get("version", ""),
        "head_sha": state.get("head_sha", ""),
        "monitor_grade": state.get("monitor_grade", ""),
        "monitor_issues": ", ".join(state.get("monitor_issues", []) or []),
        "run_conclusion": state.get("run_conclusion", ""),
    }


def node_deploy_analysis(state: CDAgentState) -> CDAgentState:
    """Only reached when the graph's conditional edge routed here."""
    from langchain_agents.agents.cd_agents import get_deploy_analysis_agent
    result = get_deploy_analysis_agent().run(
        task="Analyze this failed/unhealthy deployment. Give ROOT CAUSE + SUGGESTED FIX.",
        context=_cd_context(state),
    )
    response = result.get("response", "")
    used_tools = bool(result.get("tools_called"))
    # Clean fallback: on LLM error, produce a deterministic message instead of
    # leaking a 429/400 blob into the dashboard's Failure Analysis block.
    if result.get("llm_error") or _is_llm_error_text(response):
        grade = state.get("monitor_grade", "DOWN")
        issues = ", ".join(state.get("monitor_issues", []) or []) or "deploy did not succeed"
        response = (
            f"ROOT CAUSE:\nDeployment health is {grade} ({issues}). "
            "Automated deep analysis is temporarily unavailable (model rate limit).\n\n"
            "SUGGESTED FIX:\nReview the deploy job logs and server connectivity, then retry."
        )
        used_tools = True  # grounded in the real monitor grade / issues
    return {**state, "deploy_analysis_output": response, "deploy_used_tools": used_tools}


def node_rollback(state: CDAgentState) -> CDAgentState:
    """Only reached when the graph's conditional edge routed here."""
    from langchain_agents.agents.cd_agents import get_rollback_agent
    result = get_rollback_agent().run(
        task="Decide whether to roll back this deployment and assess the risk.",
        context=_cd_context(state),
    )
    return {
        **state,
        "rollback_output": result.get("response", ""),
        "rollback_raw": result.get("raw_tool_results", {}) or {},
    }


def node_consolidate_cd(state: CDAgentState) -> CDAgentState:
    """
    Fold agent outputs into the CDState fields that the reused
    node_post_deploy_report (and the /cd/analyze response mapping) expect.
    """
    updates: Dict[str, Any] = {}

    # Deploy failure narrative from the analysis agent.
    analysis = state.get("deploy_analysis_output", "") or ""
    if analysis:
        rc, fix = _parse_root_cause_fix(analysis)
        if rc:
            updates["deploy_failure_reason"] = rc
        if fix:
            updates["deploy_suggested_fix"] = fix
        # Deterministic quality guard (0 LLM call): score the answer's
        # structural reliability and surface it. Context = the real monitor
        # signals the analysis is supposed to be grounded in.
        if rc:
            from langchain_agents.graphs.response_quality import assess_quality
            ctx = " ".join([
                state.get("monitor_grade", "") or "",
                ", ".join(state.get("monitor_issues", []) or []),
                state.get("logs", "") or "",
            ])
            q = assess_quality(
                root_cause=rc, suggested_fix=fix,
                context_text=ctx, used_tools=bool(state.get("deploy_used_tools")),
            )
            updates["quality_score"] = q["score"]
            updates["quality_flags"] = q["flags"]

    # Structured rollback fields — read the REAL dict the rollback agent got
    # back from tool_advise_rollback (kept in raw_tool_results).
    raw = state.get("rollback_raw", {}) or {}
    advice = raw.get("tool_advise_rollback")
    if isinstance(advice, dict) and advice.get("available"):
        updates["rollback_available"] = True
        updates["rollback_command"] = advice.get("command", "")
        updates["rollback_risk"] = advice.get("risk_level", "")
        updates["rollback_risk_reasons"] = advice.get("risk_reasons", []) or []
        updates["rollback_comment_body"] = advice.get("comment_body", "")

    return {**state, **updates}


# ── Conditional routing ──────────────────────────────────────────────────────
# Fixed specialist order (deploy_analysis → rollback) — only presence in
# `route` decides whether a node is actually visited by the graph.

def _route_after_supervise_cd(state: CDAgentState) -> str:
    route = state.get("route", [])
    if "deploy_analysis" in route:
        return "deploy_analysis"
    if "rollback" in route:
        return "rollback"
    return "consolidate"


def _route_after_deploy_analysis(state: CDAgentState) -> str:
    return "rollback" if "rollback" in state.get("route", []) else "consolidate"


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_cd_agent_graph():
    g = StateGraph(CDAgentState)

    g.add_node("fetch_deployment",      node_fetch_deployment)
    g.add_node("pre_deploy_risk_score", node_pre_deploy_risk_score)
    g.add_node("check_environment",     node_check_environment)
    g.add_node("monitor_health",        node_monitor_health)
    g.add_node("supervise",             node_supervise_cd)
    g.add_node("deploy_analysis",       node_deploy_analysis)
    g.add_node("rollback",              node_rollback)
    g.add_node("consolidate",           node_consolidate_cd)
    g.add_node("post_deploy_report",    node_post_deploy_report)
    g.add_node("index_result",          node_index_result)
    g.add_node("notify",                node_notify)

    g.set_entry_point("fetch_deployment")
    g.add_edge("fetch_deployment",      "pre_deploy_risk_score")
    g.add_edge("pre_deploy_risk_score", "check_environment")
    g.add_edge("check_environment",     "monitor_health")
    g.add_edge("monitor_health",        "supervise")

    g.add_conditional_edges("supervise", _route_after_supervise_cd, {
        "deploy_analysis": "deploy_analysis",
        "rollback":        "rollback",
        "consolidate":     "consolidate",
    })
    g.add_conditional_edges("deploy_analysis", _route_after_deploy_analysis, {
        "rollback":    "rollback",
        "consolidate": "consolidate",
    })
    g.add_edge("rollback",              "consolidate")

    g.add_edge("consolidate",           "post_deploy_report")
    g.add_edge("post_deploy_report",    "index_result")
    g.add_edge("index_result",          "notify")
    g.add_edge("notify",                END)

    return g.compile()


# ── Public invoke (mirrors invoke_cd_run) ──────────────────────────────────────

_cd_agent_graph = None


def invoke_cd_agent_run(
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
    """Agentic counterpart of invoke_cd_run — same signature and return shape."""
    import os
    global _cd_agent_graph
    if _cd_agent_graph is None:
        _cd_agent_graph = build_cd_agent_graph()

    initial: CDAgentState = {
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
        # agentic fields
        "route":                   [],
        "deploy_analysis_output":  "",
        "rollback_output":         "",
        "rollback_raw":            {},
    }

    try:
        start = time.time()
        result = _cd_agent_graph.invoke(initial)
        logger.info(
            "[CDAgentGraph] run=%s grade=%s route=%s (%.1fs)",
            run_id, result.get("monitor_grade", "?"),
            result.get("route", []), round(time.time() - start, 2),
        )
        return result
    except Exception as e:
        logger.error("[CDAgentGraph] invoke failed for run=%s: %s", run_id, e)
        return {**initial, "deploy_failure_reason": f"CDAgentGraph error: {e}"}
