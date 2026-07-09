from __future__ import annotations
from typing import Any, Dict

class LCGitPRAgent:

    def __init__(self) -> None:
        from langchain_agents.tools.git_tools import (
            tool_pr_review, tool_pr_readiness, tool_pr_description, tool_cross_pr,
        )
        # Pilier TOOLS — boîte à outils inspectable (pr_agent.tools)
        self.tools = [tool_pr_review, tool_pr_readiness, tool_pr_description, tool_cross_pr]
        # Pilier MEMORY
        from langchain_agents.memory.redis_memory import AgentRedisMemory
        self.memory = AgentRedisMemory("pr_agent")

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch by intent to the matching PR tool(s), writing the report field."""
        import asyncio
        from langchain_agents.tools.git_tools import (
            tool_pr_review, tool_pr_readiness, tool_pr_description, tool_cross_pr,
        )
        from smart_git.pr_context_cache import (
            get_session_git_context, save_session_git_context,
        )

        intent = state.get("intent", "pr_readiness")
        owner = state.get("owner", "")
        repo = state.get("repo", "")
        base = state.get("base", "main")
        branch = state.get("branch", "HEAD")
        project_path = state.get("project_path", ".") or "."
        session_id = state.get("session_id", "")
        pr_number = int(state.get("pr_number", 0) or 0)

        # Session-context fallback: a conversational follow-up may omit the PR.
        if not pr_number and owner and repo and intent in (
                "pr_review", "pr_readiness", "pr_fix_guidance", "pr_description"):
            ctx = get_session_git_context(session_id)
            if ctx and ctx.get("owner") == owner and ctx.get("repo") == repo:
                pr_number = int(ctx.get("pr_number") or 0)
                state["pr_number"] = pr_number

        if intent == "cross_pr_conflicts":
            state["cross_pr_report"] = await asyncio.to_thread(
                tool_cross_pr.invoke,
                {"owner": owner, "repo": repo, "pr_number": pr_number, "base": base})
            return state

        if intent == "pr_description":
            state["pr_description"] = await asyncio.to_thread(
                tool_pr_description.invoke,
                {"project_path": project_path, "owner": owner, "repo": repo,
                 "pr_number": pr_number, "branch": branch, "base": base,
                 "pr_report": state.get("pr_report")})
            if pr_number:
                save_session_git_context(session_id, owner, repo, pr_number)
            return state

        # pr_review / pr_readiness / pr_fix_guidance — all need owner/repo/pr
        if not owner or not repo or not pr_number:
            error = {"success": False, "error": "Missing owner/repo/pr_number"}
            state["pr_report" if intent == "pr_review" else "readiness_report"] = error
            return state

        if intent == "pr_review":
            state["pr_report"] = await tool_pr_review.ainvoke(
                {"owner": owner, "repo": repo, "pr_number": pr_number})
        else:
            # pr_readiness and pr_fix_guidance both use the readiness data;
            # the synthesis agent renders fix_guidance as an action plan.
            state["readiness_report"] = await tool_pr_readiness.ainvoke(
                {"owner": owner, "repo": repo, "pr_number": pr_number})
        save_session_git_context(session_id, owner, repo, pr_number)
        return state

    async def review_pr(
        self,
        owner: str,
        repo: str,
        pr_number: int,
    ) -> Dict[str, Any]:
        if not owner or not repo or not pr_number:
            return {
                "success": False,
                "error": "Missing owner/repo/pr_number",
            }

        try:
            from smart_git.pr_review_agent import review_pr
        except Exception as e:
            return {
                "success": False,
                "error": f"Cannot import PR review agent: {e}",
            }

        try:
            result = await review_pr(owner, repo, pr_number)
            return result if isinstance(result, dict) else {
                "success": False,
                "error": "PR review returned non-dict result",
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"PR review failed: {e}",
            }

    async def readiness(
        self,
        owner: str,
        repo: str,
        pr_number: int,
    ) -> Dict[str, Any]:
        if not owner or not repo or not pr_number:
            return {
                "success": False,
                "error": "Missing owner/repo/pr_number",
            }

        try:
            from smart_git.merge_automation_agent import check_merge_readiness
        except Exception as e:
            return {
                "success": False,
                "error": f"Cannot import merge readiness agent: {e}",
            }

        try:
            result = await check_merge_readiness(owner, repo, pr_number)
            return result if isinstance(result, dict) else {
                "success": False,
                "error": "PR readiness returned non-dict result",
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"PR readiness failed: {e}",
            }


git_pr_agent = LCGitPRAgent()