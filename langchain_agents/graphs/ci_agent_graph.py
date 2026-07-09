"""
ci_agent_graph.py — Agentic CIGraph (supervisor + specialist agents).

Parallel, feature-flagged alternative to ci_graph.py. Same inputs/outputs
(invoke_ci_agent_run mirrors invoke_ci_run), so it can be swapped in behind a
flag with the deterministic graph kept as a fallback.

Hybrid design:
  - Deterministic prefetch/post/notify nodes are REUSED from ci_graph.py
    (log download, classification, PR comment, indexing, notification).
  - The reasoning steps are REAL tool-calling agents (ci_agents.py): the LLM
    decides which tools to call. A supervisor agent decides which specialists
    run.

Flow:
  fetch_run → classify_failure → supervise
    → diagnostic → security_triage → remediation   (each no-ops if not routed)
    → consolidate → post_pr_comment → index_result → notify → END
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from langgraph.graph import StateGraph, END

from langchain_agents.graphs.state import CIState
# Reuse the deterministic nodes unchanged — prefetch + side-effects.
from langchain_agents.graphs.ci_graph import (
    node_fetch_run,
    node_classify_failure,
    node_post_pr_comment,
    node_index_result,
    node_notify,
)

logger = logging.getLogger(__name__)


class CIAgentState(CIState, total=False):
    """CIState + agentic orchestration fields."""
    route: List[str]              # specialists selected by the supervisor
    diagnostic_output: str
    security_output: str
    remediation_output: str


# ── Agent nodes ────────────────────────────────────────────────────────────────

def node_supervise(state: CIAgentState) -> CIAgentState:
    """CI Supervisor (LLM) decides which specialists to activate."""
    # Targeted dashboard analysis (a single issue/job → job_id is set) only
    # needs the Diagnostic agent. Skipping the supervisor LLM call and the
    # remediation agent here removes 2 LLM round-trips per "Analyze" click,
    # which is the main latency / Groq-rate-limit driver on that path.
    if state.get("job_id"):
        logger.info("[CIAgentGraph] targeted analysis (job_id) → diagnostic only")
        return {**state, "route": ["diagnostic"]}

    from langchain_agents.agents.ci_agents import get_ci_supervisor
    route = get_ci_supervisor().route(
        outcome=state.get("outcome", ""),
        failure_type=state.get("failure_type", "unknown"),
        logs_snippet=(state.get("logs", "") or "")[:600],
    )
    logger.info("[CIAgentGraph] supervisor route=%s", route or "none")
    return {**state, "route": route}


def _agent_context(state: CIAgentState) -> Dict[str, Any]:
    """Shared context injected into every specialist so the LLM can fill tool
    arguments (owner/repo/run_id/job_id/etc.)."""
    repo = state.get("repo", "")
    return {
        "owner": state.get("owner", ""),
        "repo": repo.split("/")[-1] if "/" in repo else repo,
        "full_repo": repo,
        "run_id": state.get("run_id", ""),
        "job_id": state.get("job_id", ""),
        "project_key": state.get("project_key", ""),
        "head_sha": state.get("head_sha", ""),
        "stage_failed": state.get("stage_failed", ""),
        "failure_type": state.get("failure_type", "unknown"),
        "logs_excerpt": (state.get("logs", "") or "")[:2000],
    }


def node_diagnostic(state: CIAgentState) -> CIAgentState:
    if "diagnostic" not in state.get("route", []):
        return state
    from langchain_agents.agents.ci_agents import get_diagnostic_agent
    result = get_diagnostic_agent().run(
        task="Diagnose this pipeline failure and give ROOT CAUSE + SUGGESTED FIX.",
        context=_agent_context(state),
    )
    response = result.get("response", "")
    # Clean fallback: if the agent's LLM failed (429/400/timeout), use the
    # deterministic analyzer instead of surfacing a raw error to the user.
    if result.get("llm_error") or _is_llm_error_text(response):
        response = _deterministic_ci_analysis(state)
    return {**state, "diagnostic_output": response}


def node_security_triage(state: CIAgentState) -> CIAgentState:
    if "security_triage" not in state.get("route", []):
        return state
    from langchain_agents.agents.ci_agents import get_security_triage_agent
    result = get_security_triage_agent().run(
        task="Triage the security findings for this run and give a SECURITY VERDICT.",
        context=_agent_context(state),
    )
    # On LLM error, drop the output (consolidate falls back to diagnostic) —
    # never store an error blob.
    if result.get("llm_error") or _is_llm_error_text(result.get("response", "")):
        return state
    return {**state, "security_output": result.get("response", "")}


def node_remediation(state: CIAgentState) -> CIAgentState:
    if "remediation" not in state.get("route", []):
        return state
    from langchain_agents.agents.ci_agents import get_remediation_agent
    ctx = _agent_context(state)
    ctx["diagnosis"] = state.get("diagnostic_output", "")[:1500]
    result = get_remediation_agent().run(
        task="Given the diagnosis, produce a concrete FIX. Memorize it if reliable.",
        context=ctx,
    )
    if result.get("llm_error") or _is_llm_error_text(result.get("response", "")):
        return state
    return {**state, "remediation_output": result.get("response", "")}


def node_consolidate(state: CIAgentState) -> CIAgentState:
    """
    Fold the agent outputs into root_cause/suggested_fix so the reused
    node_post_pr_comment / node_notify work unchanged.
    """
    diag = state.get("diagnostic_output", "") or ""
    root_cause, suggested_fix = _parse_root_cause_fix(diag)

    # Remediation fix (if any) overrides/augments the diagnostic's fix.
    rem = state.get("remediation_output", "") or ""
    if rem:
        rem_fix = _extract_section(rem, "FIX:")
        if rem_fix:
            suggested_fix = rem_fix

    sec = state.get("security_output", "") or ""
    if sec and not root_cause:
        root_cause = sec.strip()[:500]

    # Final safety net: never surface an error blob as the root cause.
    if root_cause and _is_llm_error_text(root_cause):
        root_cause = _deterministic_ci_analysis(state)
        root_cause, det_fix = _parse_root_cause_fix(root_cause)
        if det_fix and not suggested_fix:
            suggested_fix = det_fix

    # If no agent ran (e.g. success), leave fields as-is.
    updates: Dict[str, Any] = {}
    if root_cause:
        updates["root_cause"] = root_cause
    if suggested_fix:
        updates["suggested_fix"] = suggested_fix
    # Agentic path = LLM analysis; keep confidence below the Redis-cache
    # threshold so index_result stores the fix as a learned one.
    updates.setdefault("confidence", state.get("confidence", 0.0))
    return {**state, **updates}


# ── Fallback helpers ────────────────────────────────────────────────────────────

def _is_llm_error_text(text: str) -> bool:
    """Detect an agent output that is actually an LLM/tool error, so it is never
    shown to the user as a 'root cause'."""
    if not text:
        return True
    low = text.lower()
    return any(m in low for m in (
        "(llm error", "rate_limit", "rate limit", "error code: 429",
        "error code: 400", "tool call validation failed", "unavailable:",
        "(no response)",
    ))


def _deterministic_ci_analysis(state: CIAgentState) -> str:
    """Fallback to the deterministic PipelineFailureAnalyzer (text cascade,
    different rate-limit budget) and format as ROOT CAUSE / SUGGESTED FIX."""
    try:
        from ci_cd.pipeline_failure_analyzer import PipelineFailureAnalyzer
        res = PipelineFailureAnalyzer().analyze(
            run_id=state.get("run_id", ""),
            repo=state.get("repo", ""),
            owner=state.get("owner", ""),
            logs=state.get("logs", ""),
            stage_failed=state.get("stage_failed", ""),
            failure_type=state.get("failure_type", "unknown"),
            project_key=state.get("project_key", ""),
            pr_number=state.get("pr_number"),
            head_sha=state.get("head_sha", ""),
        )
        rc = (res.get("root_cause") or "").strip()
        fix = (res.get("suggested_fix") or "").strip()
        if rc or fix:
            return f"ROOT CAUSE:\n{rc}\n\nSUGGESTED FIX:\n{fix}"
    except Exception as e:
        logger.debug("[CIAgentGraph] deterministic fallback failed: %s", e)
    return (
        "ROOT CAUSE:\nAutomated analysis is temporarily unavailable "
        "(model rate limit). Please retry in a minute.\n\nSUGGESTED FIX:\n"
        "Check the GitHub Actions logs for the failed step."
    )


# ── Parsing helpers ─────────────────────────────────────────────────────────────

def _extract_section(text: str, header: str) -> str:
    """Return the block after `header` up to the next ALL-CAPS 'WORD:' header."""
    import re
    idx = text.upper().find(header.upper())
    if idx == -1:
        return ""
    rest = text[idx + len(header):]
    m = re.search(r"\n[A-Z][A-Z ]{2,}:", rest)
    return (rest[: m.start()] if m else rest).strip()


def _parse_root_cause_fix(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    rc = _extract_section(text, "ROOT CAUSE:")
    fix = _extract_section(text, "SUGGESTED FIX:")
    # Fallback: no headers → treat the whole thing as the root cause.
    if not rc and not fix:
        return text.strip()[:800], ""
    return rc, fix


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_ci_agent_graph():
    g = StateGraph(CIAgentState)

    g.add_node("fetch_run",        node_fetch_run)
    g.add_node("classify_failure", node_classify_failure)
    g.add_node("supervise",        node_supervise)
    g.add_node("diagnostic",       node_diagnostic)
    g.add_node("security_triage",  node_security_triage)
    g.add_node("remediation",      node_remediation)
    g.add_node("consolidate",      node_consolidate)
    g.add_node("post_pr_comment",  node_post_pr_comment)
    g.add_node("index_result",     node_index_result)
    g.add_node("notify",           node_notify)

    g.set_entry_point("fetch_run")
    g.add_edge("fetch_run",        "classify_failure")
    g.add_edge("classify_failure", "supervise")
    g.add_edge("supervise",        "diagnostic")
    g.add_edge("diagnostic",       "security_triage")
    g.add_edge("security_triage",  "remediation")
    g.add_edge("remediation",      "consolidate")
    g.add_edge("consolidate",      "post_pr_comment")
    g.add_edge("post_pr_comment",  "index_result")
    g.add_edge("index_result",     "notify")
    g.add_edge("notify",           END)

    return g.compile()


# ── Public invoke (mirrors invoke_ci_run) ──────────────────────────────────────

_ci_agent_graph = None


def invoke_ci_agent_run(
    run_id: str,
    repo: str,
    owner: str,
    project_key: str = "",
    pr_number: int = None,
    head_sha: str = "",
    pr_branch: str = "",
    stage_failed: str = "",
    job_id: str = "",
    run_conclusion: str = "",
    run_duration_seconds: int = 0,
) -> Dict[str, Any]:
    """Agentic counterpart of invoke_ci_run — same signature and return shape."""
    global _ci_agent_graph
    if _ci_agent_graph is None:
        _ci_agent_graph = build_ci_agent_graph()

    initial: CIAgentState = {
        "run_id": str(run_id),
        "repo": repo,
        "owner": owner,
        "project_key": project_key or "",
        "pr_number": pr_number,
        "head_sha": head_sha or "",
        "pr_branch": pr_branch or "",
        "stage_failed": stage_failed or "",
        "job_id": job_id or "",
        "run_conclusion": run_conclusion or "",
        "run_duration_seconds": run_duration_seconds,
        "logs": "",
        "outcome": "",
        "failure_type": "unknown",
        "severity": "INFO",
        "sonar_gate": {},
        "sonar_metrics": {},
        "sonar_issues": [],
        "similar_fixes": [],
        "confidence": 0.0,
        "root_cause": None,
        "suggested_fix": None,
        "comment_posted": False,
        "indexed": False,
        "notification_level": "INFO",
        "codeql_alerts": [],
        "trivy_report": {},
        "auto_retried": False,
        # agentic fields
        "route": [],
        "diagnostic_output": "",
        "security_output": "",
        "remediation_output": "",
    }

    try:
        start = time.time()
        result = _ci_agent_graph.invoke(initial)
        logger.info(
            "[CIAgentGraph] run=%s outcome=%s route=%s (%.1fs)",
            run_id, result.get("outcome", "?"),
            result.get("route", []), round(time.time() - start, 2),
        )
        return result
    except Exception as e:
        logger.error("[CIAgentGraph] invoke failed for run=%s: %s", run_id, e)
        return {**initial, "root_cause": f"CIAgentGraph error: {e}"}
