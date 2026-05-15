"""
chat_graph.py — LangGraph ChatGraph Phase 1.

Goal:
  Provide a conversational entry point over the existing Code Auditor systems.

Flow:
  intent_router → load_memory → load_file_context → project_summary
  → rag_retrieve → answer_question → memory_save → format_response → END
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from langchain_agents.graphs.state import ChatState

logger = logging.getLogger(__name__)


def node_intent_router(state: ChatState) -> Dict[str, Any]:
    """Detect chat intent using LCChatAgent deterministic router."""
    from langchain_agents.agents.lc_chat_agent import lc_chat_agent

    result = lc_chat_agent.detect_intent(
        state.get("user_message", ""),
        state.get("target_file", "") or "",
    )
    return {
        "intent": result.get("intent", "question"),
        "intent_params": result.get("intent_params", {}),
    }


def node_load_memory(state: ChatState) -> Dict[str, Any]:
    """Load conversation history from Redis."""
    from langchain_agents.agents.lc_chat_agent import lc_chat_agent
    from services.chat_memory_service import chat_memory_service

    session_id = state.get("session_id") or chat_memory_service.new_session_id()
    history = lc_chat_agent.load_history(session_id)
    return {
        "session_id": session_id,
        "history": history,
        "memory_key": chat_memory_service.history_key(session_id),
    }


def node_load_file_context(state: ChatState) -> Dict[str, Any]:
    """Load target file, cached analysis and dependency context."""
    from langchain_agents.agents.lc_chat_agent import lc_chat_agent

    params = state.get("intent_params", {}) or {}
    target_file = state.get("target_file") or params.get("file_hint", "")

    ctx = lc_chat_agent.load_file_context(
        project_path=state.get("project_path", "."),
        target_file=target_file,
        user_message=state.get("user_message", ""),
    )

    return {
        "target_file": ctx.get("target_file", target_file or ""),
        "target_lang": ctx.get("target_lang", "unknown"),
        "file_code": ctx.get("file_code", ""),
        "file_analysis": ctx.get("file_analysis", {}),
        "dependencies": ctx.get("dependencies", []),
        "dependents": ctx.get("dependents", []),
        # method_hint: specific method name the dev asked about (e.g. "get_cached_analysis")
        "intent_params": {
            **(state.get("intent_params") or {}),
            "method_hint": ctx.get("method_hint", ""),
        },
    }


def node_project_summary(state: ChatState) -> Dict[str, Any]:
    """Load lightweight project summary for prompt grounding."""
    from langchain_agents.agents.lc_chat_agent import lc_chat_agent
    summary = lc_chat_agent.project_summary(state.get("project_path", "."))
    return {"project_summary": summary}


def node_rag_retrieve(state: ChatState) -> Dict[str, Any]:
    """Retrieve RAG docs for the question and target file."""
    from langchain_agents.agents.lc_chat_agent import lc_chat_agent

    result = lc_chat_agent.retrieve(
        project_path=state.get("project_path", "."),
        query=state.get("user_message", ""),
        target_file=state.get("target_file", ""),
        file_code=state.get("file_code", ""),
        language=state.get("target_lang", "unknown"),
    )
    return {
        "rag_docs": result.get("rag_docs", []),
        "rag_scores": result.get("rag_scores", []),
    }


def node_answer_question(state: ChatState) -> Dict[str, Any]:
    """Generate final project-aware answer."""
    from langchain_agents.agents.lc_chat_agent import lc_chat_agent
    response = lc_chat_agent.answer(dict(state))
    return {"response": response}


def node_memory_save(state: ChatState) -> Dict[str, Any]:
    """Persist user/assistant exchange in Redis."""
    from langchain_agents.agents.lc_chat_agent import lc_chat_agent

    metadata = {
        "intent": state.get("intent"),
        "target_file": state.get("target_file"),
        "language": state.get("target_lang"),
    }
    lc_chat_agent.save_exchange(
        session_id=state.get("session_id", ""),
        user_message=state.get("user_message", ""),
        response=state.get("response", ""),
        metadata=metadata,
    )
    return {}


def node_format_response(state: ChatState) -> Dict[str, Any]:
    """Final formatting hook."""
    response = state.get("response", "")
    session_id = state.get("session_id", "")
    formatted = response.strip()
    if session_id:
        formatted += f"\n\n---\n_session: `{session_id}`_"
    return {"formatted_response": formatted}


def node_load_project_patterns(state: ChatState) -> Dict[str, Any]:
    """Detect project conventions for code generation (Phase 2 only)."""
    from services.code_generator_service import detect_conventions
    lang = state.get("generation_language") or state.get("target_lang") or "python"
    try:
        patterns = detect_conventions(state.get("project_path", "."), lang)
    except Exception as e:
        logger.warning("detect_conventions failed: %s", e)
        patterns = {}
    return {"project_patterns": patterns}


def node_generate_completion(state: ChatState) -> Dict[str, Any]:
    """Generate function body completion (Phase 2)."""
    from langchain_agents.agents.lc_chat_agent import lc_chat_agent
    result = lc_chat_agent.complete_function(dict(state))
    return result


def node_generate_class(state: ChatState) -> Dict[str, Any]:
    """Generate a new class from scratch (Phase 2)."""
    from langchain_agents.agents.lc_chat_agent import lc_chat_agent
    result = lc_chat_agent.generate_class(dict(state))
    return result


def node_validate_generated(state: ChatState) -> Dict[str, Any]:
    """Validate syntax of generated code (Phase 2)."""
    from services.code_generator_service import validate_syntax
    code = state.get("generated_code", "")
    lang = state.get("generation_language") or state.get("target_lang") or "python"
    if not code:
        return {"generation_valid": False, "generation_errors": ["No code was generated."]}
    valid, errors = validate_syntax(code, lang)
    return {"generation_valid": valid, "generation_errors": errors}


def _route_by_intent(state: ChatState) -> str:
    """Conditional edge: route after project_summary based on intent."""
    intent = state.get("intent", "question")
    if intent == "complete_fn":
        return "load_project_patterns"
    if intent == "new_class":
        return "load_project_patterns"
    return "rag_retrieve"


def build_chat_graph():
    """Build and compile ChatGraph (Phase 1 + Phase 2).

    Phase 1 flow (question / explain):
      intent_router → load_memory → load_file_context → project_summary
      → rag_retrieve → answer_question → memory_save → format_response → END

    Phase 2 flow (complete_fn / new_class):
      intent_router → load_memory → load_file_context → project_summary
      → load_project_patterns → rag_retrieve
      → [generate_completion | generate_class]
      → validate_generated → memory_save → format_response → END
    """
    graph = StateGraph(ChatState)

    # ── Phase 1 nodes ──────────────────────────────────────────────────────
    graph.add_node("intent_router",     node_intent_router)
    graph.add_node("load_memory",       node_load_memory)
    graph.add_node("load_file_context", node_load_file_context)
    graph.add_node("project_summary",   node_project_summary)
    graph.add_node("rag_retrieve",      node_rag_retrieve)
    graph.add_node("answer_question",   node_answer_question)
    graph.add_node("memory_save",       node_memory_save)
    graph.add_node("format_response",   node_format_response)

    # ── Phase 2 nodes ──────────────────────────────────────────────────────
    graph.add_node("load_project_patterns", node_load_project_patterns)
    graph.add_node("generate_completion",   node_generate_completion)
    graph.add_node("generate_class",        node_generate_class)
    graph.add_node("validate_generated",    node_validate_generated)

    # ── Fixed edges ────────────────────────────────────────────────────────
    graph.set_entry_point("intent_router")
    graph.add_edge("intent_router",         "load_memory")
    graph.add_edge("load_memory",           "load_file_context")
    graph.add_edge("load_file_context",     "project_summary")

    # ── Conditional routing after project_summary ─────────────────────────
    graph.add_conditional_edges(
        "project_summary",
        _route_by_intent,
        {
            "rag_retrieve":          "rag_retrieve",
            "load_project_patterns": "load_project_patterns",
        },
    )

    # ── Phase 2 path ───────────────────────────────────────────────────────
    graph.add_edge("load_project_patterns", "rag_retrieve")

    # After rag_retrieve: route to answer_question OR generate_* based on intent
    # NOTE: do NOT add a direct edge from rag_retrieve — only one edge type per source
    def _route_after_rag(state: ChatState) -> str:
        intent = state.get("intent", "question")
        if intent == "complete_fn":
            return "generate_completion"
        if intent == "new_class":
            return "generate_class"
        return "answer_question"

    graph.add_conditional_edges(
        "rag_retrieve",
        _route_after_rag,
        {
            "answer_question":     "answer_question",
            "generate_completion": "generate_completion",
            "generate_class":      "generate_class",
        },
    )

    # Phase 1 tail
    graph.add_edge("answer_question",   "memory_save")

    # Phase 2 tail
    graph.add_edge("generate_completion", "validate_generated")
    graph.add_edge("generate_class",      "validate_generated")
    graph.add_edge("validate_generated",  "memory_save")

    # ── Common tail ────────────────────────────────────────────────────────
    graph.add_edge("memory_save",       "format_response")
    graph.add_edge("format_response",   END)

    return graph.compile()


_chat_graph = None


def invoke_chat(
    message: str,
    project_path: str = ".",
    session_id: str = "",
    target_file: str = "",
    **services: Any,
) -> Dict[str, Any]:
    """Convenience wrapper for CLI/API usage."""
    global _chat_graph
    if _chat_graph is None:
        _chat_graph = build_chat_graph()

    initial: ChatState = {
        "session_id": session_id,
        "user_message": message,
        "project_path": str(Path(project_path).resolve()),
        "target_file": target_file or "",
        "target_lang": "unknown",
        "intent": "question",
        "intent_params": {},
        "history": [],
        "rag_docs": [],
        "rag_scores": [],
        "response": "",
        "formatted_response": "",
        "code_blocks": [],
        "suggested_files": [],
        **services,
    }

    start = time.time()
    result = _chat_graph.invoke(initial)
    result.setdefault("stats", {})
    result["stats"]["elapsed"] = round(time.time() - start, 2)
    return result
