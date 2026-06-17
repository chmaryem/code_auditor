"""
smart_git_graph.py — LangGraph multi-agent Smart Git system.

Goal:
  Turn the existing Smart Git modules into a clean multi-agent graph.

Flow:
  decide
    → session
    → diff
    → branch
    → pr
    → conflict
    → synthesize

This graph is safe by default:
  - no automatic merge
  - no automatic push
  - no automatic conflict write
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from langchain_agents.graphs.smart_git_state import SmartGitState


# ══════════════════════════════════════════════════════════════════════════════
# Nodes
# ══════════════════════════════════════════════════════════════════════════════

def node_decide(state: SmartGitState) -> Dict[str, Any]:
    from langchain_agents.agents.lc_git_decision_agent import git_decision_agent

    plan = git_decision_agent.decide(
        state.get("user_message", ""),
        {
            "repo": state.get("repo", ""),
            "owner": state.get("owner", ""),
            "pr_number": state.get("pr_number", 0),
            "branch": state.get("branch", ""),
            "base": state.get("base", ""),
        },
    )

    return {
        "intent": plan.get("intent", "git_status"),
        "confidence": float(plan.get("confidence", 0.5)),
        "selected_agents": plan.get("selected_agents", []),
        "needs_confirmation": bool(plan.get("needs_confirmation", False)),
        "safe_mode": bool(plan.get("safe_mode", True)),
        "reason": plan.get("reason", ""),
        "pr_number": state.get("pr_number") or plan.get("pr_number", 0),
    }


def node_session(state: SmartGitState) -> Dict[str, Any]:
    from langchain_agents.agents.lc_git_session_agent import git_session_agent

    snapshot = git_session_agent.get_status(
        state.get("project_path", ".")
    )

    return {
        "session_snapshot": snapshot,
    }


def node_diff(state: SmartGitState) -> Dict[str, Any]:
    from langchain_agents.agents.lc_git_diff_agent import git_diff_agent

    intent = state.get("intent", "git_status")

    if intent == "commit_message":
        result = git_diff_agent.generate_commit_message(
            state.get("project_path", ".")
        )
        return {
            "commit_message": result.get("commit_message", ""),
            "changes": result,
        }

    changes = git_diff_agent.get_changes(
        state.get("project_path", ".")
    )

    return {
        "changes": changes,
    }


def node_branch(state: SmartGitState) -> Dict[str, Any]:
    from langchain_agents.agents.lc_git_branch_agent import git_branch_agent

    report = git_branch_agent.analyze_branch(
        project_path=state.get("project_path", "."),
        branch=state.get("branch") or "HEAD",
        base=state.get("base") or "main",
    )

    return {
        "branch_report": report,
    }


async def node_pr(state: SmartGitState) -> Dict[str, Any]:
    from langchain_agents.agents.lc_git_pr_agent import git_pr_agent

    owner = state.get("owner", "")
    repo = state.get("repo", "")
    pr_number = int(state.get("pr_number", 0) or 0)

    if not owner or not repo or not pr_number:
        error = {
            "success": False,
            "error": "Missing owner/repo/pr_number",
        }

        if state.get("intent") == "pr_readiness":
            return {"readiness_report": error}

        return {"pr_report": error}

    if state.get("intent") == "pr_readiness":
        readiness = await git_pr_agent.readiness(owner, repo, pr_number)
        return {
            "readiness_report": readiness,
        }

    review = await git_pr_agent.review_pr(owner, repo, pr_number)
    return {
        "pr_report": review,
    }


def node_conflict(state: SmartGitState) -> Dict[str, Any]:
    from langchain_agents.agents.lc_git_conflict_agent import git_conflict_agent

    report = git_conflict_agent.dry_run_resolution(
        state.get("project_path", ".")
    )

    return {
        "conflict_report": report,
    }


def node_secret_scan(state: SmartGitState) -> Dict[str, Any]:
    from langchain_agents.agents.lc_git_secret_agent import LCGitSecretAgent
    return LCGitSecretAgent().run(dict(state))


def node_commit_lint(state: SmartGitState) -> Dict[str, Any]:
    from langchain_agents.agents.lc_git_commit_linter_agent import LCGitCommitLinterAgent
    return LCGitCommitLinterAgent().run(dict(state))


def node_test_impact(state: SmartGitState) -> Dict[str, Any]:
    from langchain_agents.agents.lc_git_test_impact_agent import LCGitTestImpactAgent
    return LCGitTestImpactAgent().run(dict(state))


def node_cross_pr(state: SmartGitState) -> Dict[str, Any]:
    from langchain_agents.agents.lc_git_cross_pr_agent import LCGitCrossPRAgent
    return LCGitCrossPRAgent().run(dict(state))


def node_pr_description(state: SmartGitState) -> Dict[str, Any]:
    from langchain_agents.agents.lc_git_pr_description_agent import LCGitPRDescriptionAgent
    return LCGitPRDescriptionAgent().run(dict(state))


def node_synthesize(state: SmartGitState) -> Dict[str, Any]:
    from langchain_agents.agents.lc_git_synthesis_agent import git_synthesis_agent

    response = git_synthesis_agent.synthesize(dict(state))

    return {
        "response": response,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Routing
# ══════════════════════════════════════════════════════════════════════════════

def route_after_decision(state: SmartGitState) -> str:
    intent = state.get("intent", "git_status")

    if intent == "can_commit":
        return "session"

    if intent == "git_status":
        return "session"

    if intent in ("summarize_changes", "commit_message"):
        return "diff"

    if intent == "branch_readiness":
        return "branch"

    if intent in ("pr_review", "pr_readiness"):
        return "pr"

    if intent == "conflict_resolution_dry_run":
        return "conflict"

    if intent == "secret_scan":
        return "secret_scan"

    if intent == "commit_lint":
        return "commit_lint"

    if intent == "test_impact":
        return "test_impact"

    if intent == "cross_pr_conflicts":
        return "cross_pr"

    if intent == "pr_description":
        return "pr_description"

    return "session"


def route_after_session(state: SmartGitState) -> str:
    """
    can_commit needs:
      - session risk
      - current changes/staged files
    """
    if state.get("intent") == "can_commit":
        return "diff"

    return "synthesize"


# ══════════════════════════════════════════════════════════════════════════════
# Graph builder
# ══════════════════════════════════════════════════════════════════════════════

def build_smart_git_graph():
    graph = StateGraph(SmartGitState)

    graph.add_node("decide", node_decide)
    graph.add_node("session", node_session)
    graph.add_node("diff", node_diff)
    graph.add_node("branch", node_branch)
    graph.add_node("pr", node_pr)
    graph.add_node("conflict", node_conflict)
    graph.add_node("secret_scan", node_secret_scan)
    graph.add_node("commit_lint", node_commit_lint)
    graph.add_node("test_impact", node_test_impact)
    graph.add_node("cross_pr", node_cross_pr)
    graph.add_node("pr_description", node_pr_description)
    graph.add_node("synthesize", node_synthesize)

    graph.set_entry_point("decide")

    graph.add_conditional_edges(
        "decide",
        route_after_decision,
        {
            "session":       "session",
            "diff":          "diff",
            "branch":        "branch",
            "pr":            "pr",
            "conflict":      "conflict",
            "secret_scan":   "secret_scan",
            "commit_lint":   "commit_lint",
            "test_impact":   "test_impact",
            "cross_pr":      "cross_pr",
            "pr_description": "pr_description",
        },
    )

    graph.add_conditional_edges(
        "session",
        route_after_session,
        {
            "diff": "diff",
            "synthesize": "synthesize",
        },
    )

    graph.add_edge("diff", "synthesize")
    graph.add_edge("branch", "synthesize")
    graph.add_edge("pr", "synthesize")
    graph.add_edge("conflict", "synthesize")
    graph.add_edge("secret_scan", "synthesize")
    graph.add_edge("commit_lint", "synthesize")
    graph.add_edge("test_impact", "synthesize")
    graph.add_edge("cross_pr", "synthesize")
    graph.add_edge("pr_description", "synthesize")

    graph.add_edge("synthesize", END)

    return graph.compile()


_smart_git_graph = None


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

async def ainvoke_smart_git(
    message: str,
    project_path: str = ".",
    repo: str = "",
    owner: str = "",
    pr_number: int = 0,
    branch: str = "HEAD",
    base: str = "main",
    session_id: str = "",
) -> Dict[str, Any]:
    """
    Async entry point for:
      - ChatAgent
      - FastAPI
      - Plugin backend
    """
    global _smart_git_graph

    if _smart_git_graph is None:
        _smart_git_graph = build_smart_git_graph()

    initial: SmartGitState = {
        "user_message": message,
        "project_path": str(Path(project_path).resolve()),
        "repo": repo,
        "owner": owner,
        "pr_number": pr_number,
        "branch": branch,
        "base": base,
        "session_id": session_id,
        "safe_mode": True,
        "actions": [],
        "warnings": [],
        "errors": [],
        "stats": {},
    }

    start = time.time()
    result = await _smart_git_graph.ainvoke(initial)

    result.setdefault("stats", {})
    result["stats"]["elapsed"] = round(time.time() - start, 2)

    return result


def invoke_smart_git(
    message: str,
    project_path: str = ".",
    repo: str = "",
    owner: str = "",
    pr_number: int = 0,
    branch: str = "HEAD",
    base: str = "main",
    session_id: str = "",
) -> Dict[str, Any]:
    """
    Sync entry point for CLI/tests.
    """
    return asyncio.run(
        ainvoke_smart_git(
            message=message,
            project_path=project_path,
            repo=repo,
            owner=owner,
            pr_number=pr_number,
            branch=branch,
            base=base,
            session_id=session_id,
        )
    )