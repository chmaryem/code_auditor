
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from langchain_agents.chat_agents.base import BlackboardAgent

logger = logging.getLogger(__name__)


class FileAgent(BlackboardAgent):
 

    name = "file"
    wave = 0  # doit produire file_code AVANT RagAgent

    def should_activate(self, state: Dict[str, Any]) -> bool:
        if state.get("scope") == "dashboard":
            return bool(state.get("attached_files"))
        params = state.get("intent_params") or {}
        return bool(state.get("target_file") or params.get("file_hint"))

    async def contribute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        from langchain_agents.graphs.chat_graph import node_load_file_context
        return await asyncio.to_thread(node_load_file_context, dict(state))


class RagAgent(BlackboardAgent):
    """Récupération RAG (ChromaDB). Auto-désactivé en dashboard (pas de scan projet)."""

    name = "rag"

    def should_activate(self, state: Dict[str, Any]) -> bool:
        if state.get("scope") == "dashboard":
            return False
        return bool(state.get("needs_rag", True)) and bool(state.get("ai_use_rag", True))

    async def contribute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        from langchain_agents.agents.lc_chat_agent import lc_chat_agent
        try:
            res = await asyncio.to_thread(
                lc_chat_agent.retrieve,
                project_path=state.get("project_path", "."),
                query=state.get("user_message", ""),
                target_file=state.get("target_file", "") or "",
                file_code=state.get("file_code", "") or "",
                language=state.get("target_lang", "unknown"),
            )
            return {"rag_docs": res.get("rag_docs", []), "rag_scores": res.get("rag_scores", [])}
        except Exception as e:
            logger.debug("RagAgent failed: %s", e)
            return {"rag_docs": [], "rag_scores": []}


class GitAgent(BlackboardAgent):
    """Instantané de session Git (GitSessionTracker). Actif si needs_git."""

    name = "git"

    def should_activate(self, state: Dict[str, Any]) -> bool:
        return bool(state.get("needs_git"))

    async def contribute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from smart_git.git_session_tracker import GitSessionTracker
            tracker = GitSessionTracker(state.get("project_path", "."))
            snap = await asyncio.to_thread(tracker.get_session_status)
            return {"git_context": snap or {}}
        except Exception as e:
            logger.debug("GitAgent failed: %s", e)
            return {}


class CiAgent(BlackboardAgent):
    """Score de readiness CI/CD (même scorer que le dashboard CI/CD). Actif si ci."""

    name = "ci"

    def should_activate(self, state: Dict[str, Any]) -> bool:
        return state.get("intent") == "ci_question" or bool(state.get("needs_ci"))

    async def contribute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from langchain_agents.tools.project_state_tools import tool_chat_ci_readiness
            from langchain_agents.graphs.chat_graph import _resolve_owner_repo
            pp = state.get("project_path", ".") or "."
            owner, repo = _resolve_owner_repo(pp)
            ci = await asyncio.to_thread(tool_chat_ci_readiness, pp, repo, owner)
            return {"project_state_context": {"ci_readiness": ci}}
        except Exception as e:
            logger.debug("CiAgent failed: %s", e)
            return {}


class ProjectStateAgent(BlackboardAgent):
   

    name = "project_state"

    def should_activate(self, state: Dict[str, Any]) -> bool:
        return state.get("intent") == "project_state"

    async def contribute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from langchain_agents.tools.project_state_tools import tool_chat_project_state_summary
            from langchain_agents.graphs.chat_graph import _resolve_owner_repo
            pp = state.get("project_path", ".") or "."
            owner, repo = _resolve_owner_repo(pp)
            summary = await asyncio.to_thread(
                tool_chat_project_state_summary, pp, state.get("target_file"), repo, owner
            )
            return {"project_state_context": summary}
        except Exception as e:
            logger.debug("ProjectStateAgent failed: %s", e)
            return {}


class SummaryAgent(BlackboardAgent):
    """Résumé léger du projet (langages, comptage). Auto-désactivé en dashboard."""

    name = "summary"

    def should_activate(self, state: Dict[str, Any]) -> bool:
        return state.get("scope") != "dashboard"

    async def contribute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        from langchain_agents.agents.lc_chat_agent import lc_chat_agent
        try:
            return {"project_summary": await asyncio.to_thread(
                lc_chat_agent.project_summary, state.get("project_path", ".")
            )}
        except Exception as e:
            logger.debug("SummaryAgent failed: %s", e)
            return {}
