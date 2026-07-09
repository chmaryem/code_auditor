"""
smart_git_graph.py — Smart Git multi-agent orchestration (LangGraph).

Genuine multi-agent architecture for Smart Git, consistent with the platform's
other graphs (ci_graph.py, cd_graph.py): a deterministic-then-conditional
topology with a single convergence (join) node.

Topology (same shape as CIGraph/CDGraph):

    entry → node_router            (RouterAgent — LLM classify → intent)
    node_router → [route_by_intent]  (conditional edges)
        → node_working_copy   (WorkingCopyAgent — local/pre-commit + branch)
        → node_pr             (PRAgent — GitHub PR review/readiness/desc/cross)
        → node_conflict       (ConflictAgent — conflict dry-run + subgraph)
    {3 workers} → node_synthesize  (SynthesisAgent — JOIN → response) → END

Four pillars, each on existing infra:
  - Agents        : 5 LangChain agents (router, working_copy, pr, conflict, synthesis)
  - Tools         : langchain_agents/tools/git_tools.py (each agent owns a subset)
  - Memory        : AgentRedisMemory per agent + shared blackboard/session
                    memory (smart_git/pr_context_cache.py)
  - Orchestration : this StateGraph

The conflict path composes an existing subgraph
(smart_git/conflict_resolution_graph.py) for actual resolution — real graph
composition, already live in production.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import urllib.request
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from langchain_agents.graphs.state import SmartGitState

logger = logging.getLogger(__name__)



async def node_router(state: SmartGitState) -> Dict[str, Any]:
    """RouterAgent: classify the developer's message (LLM + regex fallback)."""
    from langchain_agents.agents.lc_git_decision_agent import git_decision_agent

    message = state.get("user_message", "") or ""
    plan = git_decision_agent.decide(message, {
        "repo": state.get("repo", ""), "owner": state.get("owner", ""),
        "pr_number": state.get("pr_number", 0), "branch": state.get("branch", "HEAD"),
        "base": state.get("base", "main"),
    })
    intent = plan.get("intent", "git_status")
    resolved_pr = int(state.get("pr_number", 0) or 0) or int(plan.get("pr_number", 0) or 0)

    return {
        "intent": intent,
        "pr_number": resolved_pr,
        "confidence": float(plan.get("confidence", 0.5)),
        "branch_name": plan.get("branch_name", ""),
        "selected_agents": plan.get("selected_agents", []),
        "needs_confirmation": bool(plan.get("needs_confirmation", False)),
        "safe_mode": bool(plan.get("safe_mode", True)),
        "reason": plan.get("reason", ""),
    }


async def node_working_copy(state: SmartGitState) -> Dict[str, Any]:
    """WorkingCopyAgent: local/pre-commit checks + local branch readiness."""
    from langchain_agents.agents.lc_git_working_copy_agent import working_copy_agent
    return await working_copy_agent.run(dict(state))


async def node_pr(state: SmartGitState) -> Dict[str, Any]:
    """PRAgent: GitHub PR review / readiness / fix-guidance / description / cross-PR."""
    from langchain_agents.agents.lc_git_pr_agent import git_pr_agent
    return await git_pr_agent.run(dict(state))


async def node_conflict(state: SmartGitState) -> Dict[str, Any]:
    """ConflictAgent: safe conflict dry-run scan (+ resolution subgraph capability)."""
    from langchain_agents.agents.lc_git_conflict_agent import git_conflict_agent
    return await git_conflict_agent.run(dict(state))


def _github_token() -> str:
    return os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN") or os.environ.get("GITHUB_TOKEN", "")


def _github_repo_overview(owner: str, repo: str, token: str) -> Dict[str, Any]:
    """Aperçu essentiel du dépôt via l'API REST GitHub (1 à 2 appels, best-effort)."""
    headers = {
        "Authorization": f"token {token}",
        "Accept":        "application/vnd.github.v3+json",
        "User-Agent":    "code-auditor/1.0",
    }
    url = f"https://api.github.com/repos/{owner}/{repo}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        d = json.loads(resp.read().decode())

    # Nombre de PR ouvertes — appel léger, dégradé silencieusement en cas d'échec.
    open_prs = None
    try:
        purl = f"https://api.github.com/repos/{owner}/{repo}/pulls?state=open&per_page=100"
        preq = urllib.request.Request(purl, headers=headers)
        with urllib.request.urlopen(preq, timeout=15) as presp:
            open_prs = len(json.loads(presp.read().decode()))
    except Exception as e:
        logger.debug("open PR count failed: %s", e)

    return {
        "success":        True,
        "source":         "github",
        "full_name":      d.get("full_name", f"{owner}/{repo}"),
        "description":    d.get("description") or "",
        "default_branch": d.get("default_branch", ""),
        "visibility":     "privé" if d.get("private") else "public",
        "language":       d.get("language") or "",
        "open_prs":       open_prs,
        "stars":          d.get("stargazers_count", 0),
        "pushed_at":      d.get("pushed_at", ""),
        "html_url":       d.get("html_url", ""),
    }


def _local_repo_overview(project_path: str) -> Dict[str, Any]:
    """Repli local (aucun réseau) : branche, remote, commits, fichiers suivis."""
    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", project_path, *args],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        except Exception:
            return ""

    return {
        "success":        True,
        "source":         "local",
        "full_name":      _git("rev-parse", "--show-toplevel").split("/")[-1] or project_path,
        "default_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "remote":         _git("remote", "get-url", "origin"),
        "commits":        _git("rev-list", "--count", "HEAD"),
        "last_commit":    _git("log", "-1", "--format=%s (%cr)"),
    }


def build_repo_overview(owner: str, repo: str, project_path: str) -> Dict[str, Any]:
    """Hybride : aperçu GitHub si dépôt connecté + token dispo, sinon repli git local."""
    token = _github_token()
    if owner and repo and token:
        try:
            return _github_repo_overview(owner, repo, token)
        except Exception as e:
            logger.info("repo_overview GitHub échoué (%s) → repli local", e)
    return _local_repo_overview(project_path)


async def node_repo_overview(state: SmartGitState) -> Dict[str, Any]:
    """RepoAgent: aperçu essentiel du dépôt (GitHub API si connecté, sinon git local)."""
    import asyncio
    overview = await asyncio.to_thread(
        build_repo_overview,
        state.get("owner", "") or "",
        state.get("repo", "") or "",
        state.get("project_path", ".") or ".",
    )
    return {"repo_overview_report": overview}


def node_synthesize(state: SmartGitState) -> Dict[str, Any]:
    """SynthesisAgent (JOIN): format whatever report is present into `response`."""
    from langchain_agents.agents.lc_git_synthesis_agent import git_synthesis_agent
    return {"response": git_synthesis_agent.synthesize(dict(state))}



_WORKING_COPY_INTENTS = {
    "git_status", "can_commit", "commit_message", "summarize_changes",
    "commit_lint", "secret_scan", "test_impact", "branch_readiness",
}
_PR_INTENTS = {
    "pr_review", "pr_readiness", "pr_fix_guidance", "pr_description",
    "cross_pr_conflicts",
}


def route_by_intent(state: SmartGitState) -> str:
    """Conditional router: intent → worker node (like CIGraph's route_after_classify)."""
    intent = state.get("intent", "git_status")
    if intent == "repo_overview":
        return "repo_overview"
    if intent in _PR_INTENTS:
        return "pr"
    if intent == "conflict_resolution_dry_run":
        return "conflict"
    return "working_copy"  # all working-copy intents + safe default




def build_smart_git_graph():
    graph = StateGraph(SmartGitState)

    graph.add_node("router", node_router)
    graph.add_node("working_copy", node_working_copy)
    graph.add_node("pr", node_pr)
    graph.add_node("conflict", node_conflict)
    graph.add_node("repo_overview", node_repo_overview)
    graph.add_node("synthesize", node_synthesize)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        route_by_intent,
        {"working_copy": "working_copy", "pr": "pr", "conflict": "conflict",
         "repo_overview": "repo_overview"},
    )

    # All workers converge on the single synthesis (join) node.
    for worker in ("working_copy", "pr", "conflict", "repo_overview"):
        graph.add_edge(worker, "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


_SMART_GIT_GRAPH = None


def get_smart_git_graph():
    """Compiled-graph singleton (same pattern as the other graphs)."""
    global _SMART_GIT_GRAPH
    if _SMART_GIT_GRAPH is None:
        _SMART_GIT_GRAPH = build_smart_git_graph()
    return _SMART_GIT_GRAPH
