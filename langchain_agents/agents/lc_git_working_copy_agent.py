"""
lc_git_working_copy_agent.py — WorkingCopyAgent (role-based Smart Git agent).

Role: everything about the developer's LOCAL working copy before/around a
commit, plus local branch readiness. Consolidates the former thin wrappers
(session, diff, secret, commit-linter, test-impact, branch) into one agent
that OWNS a toolbox and selects the right tool by intent.

Pillars:
  - Tools : self.tools (LangChain @tool objects from git_tools.py).
  - Memory: self.memory (AgentRedisMemory) for per-agent state; shared
            blackboard + session memory via smart_git.pr_context_cache.

Node in the graph: node_working_copy. Selected by RouterAgent for intents:
  git_status, can_commit, commit_message, summarize_changes,
  commit_lint, secret_scan, test_impact, branch_readiness.
"""
from __future__ import annotations

from typing import Any, Dict

from langchain_agents.tools.git_tools import (
    tool_session_status, tool_changes, tool_commit_message, tool_commit_lint,
    tool_secret_scan, tool_test_impact, tool_branch_readiness, tool_pr_readiness,
)


class LCGitWorkingCopyAgent:
    name = "working_copy_agent"

    def __init__(self) -> None:
        # Pilier TOOLS — boîte à outils inspectable (working_copy_agent.tools)
        self.tools = [
            tool_session_status, tool_changes, tool_commit_message,
            tool_commit_lint, tool_secret_scan, tool_test_impact,
            tool_branch_readiness,
        ]
        # Pilier MEMORY — mémoire propre à l'agent
        from langchain_agents.memory.redis_memory import AgentRedisMemory
        self.memory = AgentRedisMemory(self.name)

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch by intent to the matching tool, writing the report field."""
        import asyncio

        intent = state.get("intent", "git_status")
        project_path = state.get("project_path", ".") or "."

        if intent == "can_commit":
            state["session_snapshot"] = await asyncio.to_thread(
                tool_session_status.invoke, {"project_path": project_path})
            state["changes"] = await asyncio.to_thread(
                tool_changes.invoke, {"project_path": project_path})

        elif intent == "commit_message":
            result = await asyncio.to_thread(
                tool_commit_message.invoke, {"project_path": project_path})
            state["commit_message"] = result.get("commit_message", "")
            state["changes"] = result

        elif intent == "summarize_changes":
            state["changes"] = await asyncio.to_thread(
                tool_changes.invoke, {"project_path": project_path})

        elif intent == "commit_lint":
            state["commit_lint_report"] = await asyncio.to_thread(
                tool_commit_lint.invoke, {"project_path": project_path})

        elif intent == "secret_scan":
            state["secret_scan_report"] = await asyncio.to_thread(
                tool_secret_scan.invoke, {"project_path": project_path})

        elif intent == "test_impact":
            state["test_impact_report"] = await asyncio.to_thread(
                tool_test_impact.invoke, {"project_path": project_path})

        elif intent == "branch_readiness":
            await self._branch_readiness(state)

        else:  # git_status (default)
            state["session_snapshot"] = await asyncio.to_thread(
                tool_session_status.invoke, {"project_path": project_path})

        return state

    async def _branch_readiness(self, state: Dict[str, Any]) -> None:
        """
        Named branch → GitHub PR readiness if it matches an open PR (so Chat
        and dashboard never disagree); else local branch analysis. Falls back
        to the session's last-resolved PR for context-free follow-ups.
        Behavior preserved verbatim from smart_git_dispatch.py.
        """
        import asyncio
        from langchain_agents.agents.lc_git_helpers import resolve_branch_to_pr
        from smart_git.pr_context_cache import (
            get_session_git_context, save_session_git_context,
        )

        project_path = state.get("project_path", ".") or "."
        owner = state.get("owner", "")
        repo = state.get("repo", "")
        branch = state.get("branch", "HEAD")
        base = state.get("base", "main")
        session_id = state.get("session_id", "")

        resolved_number = 0
        branch_name = state.get("branch_name", "")
        if branch_name and owner and repo:
            resolved_number = await resolve_branch_to_pr(owner, repo, branch_name)

        if not resolved_number and owner and repo:
            ctx = get_session_git_context(session_id)
            if ctx and ctx.get("owner") == owner and ctx.get("repo") == repo:
                resolved_number = int(ctx.get("pr_number") or 0)

        if resolved_number:
            # Matches an open PR → delegate to GitHub-backed readiness (same
            # data the dashboard's Readiness tab shows). Reroute intent so the
            # synthesis agent renders a PR readiness report.
            state["intent"] = "pr_readiness"
            state["pr_number"] = resolved_number
            state["readiness_report"] = await tool_pr_readiness.ainvoke(
                {"owner": owner, "repo": repo, "pr_number": resolved_number})
            save_session_git_context(session_id, owner, repo, resolved_number)
        else:
            state["branch_report"] = await asyncio.to_thread(
                tool_branch_readiness.invoke,
                {"project_path": project_path, "branch": branch or "HEAD",
                 "base": base or "main"})


working_copy_agent = LCGitWorkingCopyAgent()
