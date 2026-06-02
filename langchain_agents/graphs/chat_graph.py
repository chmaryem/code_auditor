"""
chat_graph.py — LangGraph ChatGraph.

Goal:
  Provide a conversational entry point over the existing Code Auditor systems.

Architecture:
  load_memory → intent_router → decision_agent
    → route decision:
        - CI/Git questions can skip file context
        - code questions load file context
    → route file context:
        - fast explain
        - contextual Q&A
        - code generation
        - test generation placeholder
    → memory_save → format_response → END

Streaming:
  stream_chat() uses LangGraph astream_events() and yields SSE events:
    status | token | code | done | error
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from langchain_agents.graphs.state import ChatState

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Initial state helper
# ══════════════════════════════════════════════════════════════════════════════

def _initial_state(
    message: str,
    project_path: str = ".",
    session_id: str = "",
    target_file: str = "",
    cursor_line: int = 0,
    active_function: str = "",
    selected_text: str = "",
    visible_range: list | None = None,
    **services: Any,
) -> ChatState:
    return {
        "session_id": session_id,
        "user_message": message,
        "project_path": str(Path(project_path).resolve()),
        "target_file": target_file or "",
        "target_lang": "unknown",

        # IDE cursor context
        "cursor_line":       cursor_line,
        "active_function":   active_function,
        "selected_text":     selected_text,
        "visible_range":     visible_range or [0, 0],

        # Git / CI context (populated lazily)
        "git_context":  {},
        "ci_context":   {},

        # Intent / decision
        "intent": "question",
        "intent_params": {},
        "decision_plan": {},
        "context_level": "context",
        "selected_agents": [],
        "needs_rag": True,
        "needs_git": False,
        "needs_ci": False,
        "needs_generation": False,
        "needs_tests": False,

        # Context
        "history": [],
        "memory_key": "",
        "file_code": "",
        "file_analysis": {},
        "dependencies": [],
        "dependents": [],
        "project_summary": {},
        "rag_docs": [],
        "rag_scores": [],
        "file_cache": {},

        # Generation
        "generation_target": "",
        "generation_language": "",
        "generated_code": "",
        "generation_valid": True,
        "generation_errors": [],
        "project_patterns": {},
        "apply_to_disk": False,

        # Output
        "response": "",
        "formatted_response": "",
        "code_blocks": [],
        "suggested_files": [],

        **services,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Nodes — Memory + Intent + Decision
# ══════════════════════════════════════════════════════════════════════════════

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


def node_intent_router(state: ChatState) -> Dict[str, Any]:
    """Detect base chat intent using LCChatAgent deterministic router."""
    from langchain_agents.agents.lc_chat_agent import lc_chat_agent

    result = lc_chat_agent.detect_intent(
        state.get("user_message", ""),
        state.get("target_file", "") or "",
    )

    return {
        "intent": result.get("intent", "question"),
        "intent_params": result.get("intent_params", {}),
    }


def node_decision_agent(state: ChatState) -> Dict[str, Any]:
    """
    LLM-powered decision agent (v2).
    Passes cursor context for IDE-aware routing.
    Falls back to regex if LLM unavailable or low confidence.
    """
    from langchain_agents.agents.lc_chat_decision_agent import chat_decision_agent

    plan = chat_decision_agent.decide(
        user_message         = state.get("user_message", ""),
        target_file          = state.get("target_file", "") or "",
        base_intent          = state.get("intent", "question"),
        intent_params        = state.get("intent_params", {}) or {},
        conversation_history = state.get("history", []),
        # cursor context
        cursor_line          = state.get("cursor_line", 0),
        active_function      = state.get("active_function", ""),
        selected_text        = state.get("selected_text", ""),
    )

    old_params = state.get("intent_params", {}) or {}
    merged_params = {
        **old_params,
        "file_hint":        old_params.get("file_hint")        or plan.get("target_file", ""),
        "method_hint":      old_params.get("method_hint")      or plan.get("target_symbol", ""),
        "generation_target":old_params.get("generation_target") or plan.get("generation_target", ""),
    }

    return {
        "decision_plan":    plan,
        "intent":           plan.get("intent", state.get("intent", "question")),
        "intent_params":    merged_params,
        "context_level":    plan.get("context_level", "context"),
        "selected_agents":  plan.get("agents", []),
        "needs_rag":        bool(plan.get("needs_rag", True)),
        "needs_git":        bool(plan.get("needs_git", False)),
        "needs_ci":         bool(plan.get("needs_ci", False)),
        "needs_generation": bool(plan.get("needs_generation", False)),
        "needs_tests":      bool(plan.get("needs_tests", False)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Nodes — Context
# ══════════════════════════════════════════════════════════════════════════════

def node_load_file_context(state: ChatState) -> Dict[str, Any]:
    """Load target file, cached analysis and dependency context."""
    from langchain_agents.agents.lc_chat_agent import lc_chat_agent

    params = state.get("intent_params", {}) or {}
    target_file = state.get("target_file") or params.get("file_hint", "")

    # Per-invocation cache. A Redis-backed session cache can be added later.
    file_cache = state.get("file_cache", {}) or {}
    if target_file and target_file in file_cache:
        cached = file_cache[target_file]
        return {
            "target_file": target_file,
            "target_lang": cached["target_lang"],
            "file_code": cached["file_code"],
            "file_analysis": cached["file_analysis"],
            "dependencies": cached["dependencies"],
            "dependents": cached["dependents"],
            "intent_params": {
                **(state.get("intent_params") or {}),
                "method_hint": cached.get("method_hint", ""),
            },
            "file_cache": file_cache,
        }

    ctx = lc_chat_agent.load_file_context(
        project_path=state.get("project_path", "."),
        target_file=target_file,
        user_message=state.get("user_message", ""),
    )

    found_target_file = ctx.get("target_file", target_file or "")
    if found_target_file and ctx.get("file_code"):
        file_cache[found_target_file] = {
            "target_lang": ctx.get("target_lang", "unknown"),
            "file_code": ctx.get("file_code", ""),
            "file_analysis": ctx.get("file_analysis", {}),
            "dependencies": ctx.get("dependencies", []),
            "dependents": ctx.get("dependents", []),
            "method_hint": ctx.get("method_hint", ""),
        }

    return {
        "target_file": found_target_file,
        "target_lang": ctx.get("target_lang", "unknown"),
        "file_code": ctx.get("file_code", ""),
        "file_analysis": ctx.get("file_analysis", {}),
        "dependencies": ctx.get("dependencies", []),
        "dependents": ctx.get("dependents", []),
        "intent_params": {
            **(state.get("intent_params") or {}),
            "method_hint": ctx.get("method_hint", ""),
        },
        "file_cache": file_cache,
    }


async def node_parallel_context(state: ChatState) -> Dict[str, Any]:
    """
    Parallel context loader — replaces sequential project_summary → rag_retrieve.

    Runs concurrently:
      - project_summary  (lightweight metadata)
      - rag_retrieve     (ChromaDB vector search)
      - neighborhood     (Knowledge Graph dep lookup)
      - git_context      (GitSessionTracker snapshot — only if needs_git)

    ~60% latency reduction vs sequential execution.
    """
    from langchain_agents.agents.lc_chat_agent import lc_chat_agent

    project_path = state.get("project_path", ".")
    user_message = state.get("user_message", "")
    target_file  = state.get("target_file", "") or ""
    file_code    = state.get("file_code", "") or ""
    language     = state.get("target_lang", "unknown")
    needs_git    = state.get("needs_git", False)
    needs_rag    = state.get("needs_rag", True)

    async def _summary():
        try:
            return lc_chat_agent.project_summary(project_path)
        except Exception as e:
            logger.warning("project_summary failed: %s", e)
            return {}

    async def _rag():
        if not needs_rag:
            return {"rag_docs": [], "rag_scores": []}
        try:
            result = await asyncio.to_thread(
                lc_chat_agent.retrieve,
                project_path=project_path,
                query=user_message,
                target_file=target_file,
                file_code=file_code,
                language=language,
            )
            return {"rag_docs": result.get("rag_docs", []), "rag_scores": result.get("rag_scores", [])}
        except Exception as e:
            logger.warning("rag_retrieve failed: %s", e)
            return {"rag_docs": [], "rag_scores": []}

    async def _git():
        if not needs_git:
            return {}
        try:
            from smart_git.git_session_tracker import GitSessionTracker
            tracker = GitSessionTracker(project_path)
            snapshot = await asyncio.to_thread(tracker.get_session_status)
            return snapshot or {}
        except Exception as e:
            logger.debug("git_context failed: %s", e)
            return {}

    summary, rag_result, git_snap = await asyncio.gather(
        _summary(), _rag(), _git()
    )

    return {
        "project_summary": summary,
        "rag_docs":        rag_result["rag_docs"],
        "rag_scores":      rag_result["rag_scores"],
        "git_context":     git_snap,
    }


def node_rag_retrieve(state: ChatState) -> Dict[str, Any]:
    """Standalone RAG retrieve — used only by generation sub-path."""
    from langchain_agents.agents.lc_chat_agent import lc_chat_agent

    result = lc_chat_agent.retrieve(
        project_path=state.get("project_path", "."),
        query=state.get("user_message", ""),
        target_file=state.get("target_file", ""),
        file_code=state.get("file_code", ""),
        language=state.get("target_lang", "unknown"),
    )

    return {
        "rag_docs":   result.get("rag_docs", []),
        "rag_scores": result.get("rag_scores", []),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Nodes — Answer / Specialized paths
# ══════════════════════════════════════════════════════════════════════════════

async def node_fast_answer(state: ChatState, config: Any = None) -> Dict[str, Any]:
    """Fast streamed response for simple explain/summarize questions."""
    from langchain_agents.agents.lc_chat_agent import lc_chat_agent

    response = await lc_chat_agent.afast_answer(dict(state), config=config)
    return {"response": response}


async def node_answer_question(state: ChatState, config: Any = None) -> Dict[str, Any]:
    """Generate final project-aware answer."""
    from langchain_agents.agents.lc_chat_agent import lc_chat_agent

    response = await lc_chat_agent.aanswer(dict(state), config=config)
    return {"response": response}


async def node_git_question(state: ChatState) -> Dict[str, Any]:
    """
    Route Git-related questions to SmartGitGraph.

    Phase B enhancement — "Explain what changed":
      If the message is about understanding recent changes (diff, what did I modify,
      résume mes changements…), we inject the actual git diff into the prompt so the
      ChatAgent can answer conversationally without routing to SmartGitGraph at all.

    For heavier operations (PR review, conflict resolution…), SmartGitGraph is used.
    """
    import re

    message = state.get("user_message", "") or ""
    project_path = state.get("project_path", ".") or "."

    # ── "Explain what changed" shortcut ──────────────────────────────────────
    EXPLAIN_DIFF_PATTERNS = [
        r"what (did|have) i (changed|modified|done)",
        r"(explain|describe|résume|résumé|summarize|summary)\s+(what|my|les|mes)?\s*(changed|changes|modifications|diff|recent)",
        r"(qu'est-ce que|qu'ai-je)\s+(j'ai\s+)?(changé|modifié|fait)",
        r"résume\s+mes\s+changements",
        r"what.*(different|changed).*(since|from)",
        r"(show|explain).*(diff|patch)",
    ]
    is_explain_diff = any(
        re.search(p, message, re.IGNORECASE) for p in EXPLAIN_DIFF_PATTERNS
    )

    if is_explain_diff:
        diff_text = ""
        try:
            import subprocess
            result_proc = await asyncio.to_thread(
                subprocess.run,
                ["git", "-C", project_path, "diff", "--stat", "--unified=3"],
                capture_output=True, text=True, timeout=10,
            )
            diff_text = result_proc.stdout or ""
            if not diff_text.strip():
                # Try staged
                result_proc = await asyncio.to_thread(
                    subprocess.run,
                    ["git", "-C", project_path, "diff", "--cached", "--stat", "--unified=3"],
                    capture_output=True, text=True, timeout=10,
                )
                diff_text = result_proc.stdout or ""
        except Exception as e:
            logger.debug("git diff failed: %s", e)

        if diff_text.strip():
            from langchain_agents.agents.lc_chat_agent import lc_chat_agent
            enriched_state = dict(state)
            enriched_state["user_message"] = (
                f"{message}\n\n"
                f"[Git diff context — current working tree]\n"
                f"```diff\n{diff_text[:3000]}\n```"
            )
            response = await lc_chat_agent.aanswer(enriched_state)
            return {"response": response, "git_diff_used": True}
        else:
            return {
                "response": (
                    "## Git Changes\n\n"
                    "✅ Aucune modification non commitée détectée dans le working tree.\n\n"
                    "Si tu viens de committer, utilise :\n"
                    "```bash\ngit show --stat\n```"
                ),
                "git_diff_used": False,
            }

    # ── Full SmartGitGraph for heavier operations ─────────────────────────────
    from langchain_agents.graphs.smart_git_graph import ainvoke_smart_git

    result = await ainvoke_smart_git(
        message=message,
        project_path=project_path,
        session_id=state.get("session_id", ""),
    )

    return {
        "response":   result.get("response", ""),
        "git_result": result,
    }


def node_ci_question(state: ChatState) -> Dict[str, Any]:
    """Placeholder for CIGraph integration."""
    return {
        "response": (
            "## CI/CD Intelligence\n\n"
            "Cette demande doit être traitée par le **CI/CD System**.\n\n"
            "L’intégration directe `ChatAgent → CIGraph` est prévue dans la prochaine étape.\n\n"
            "Pour l’instant, utilise le module CI existant :\n\n"
            "```bash\n"
            "python main.py ci-analyze --repo OWNER/REPO --pr PR_NUMBER --project-key SONAR_PROJECT_KEY\n"
            "```"
        )
    }


def node_test_generation(state: ChatState) -> Dict[str, Any]:
    """Placeholder for TestAgent integration."""
    return {
        "response": (
            "## Test Generation\n\n"
            "Cette demande doit être traitée par le **Test System**.\n\n"
            "L’intégration directe `ChatAgent → TestAgent` est prévue dans la prochaine étape.\n\n"
            "Pour l’instant, utilise la commande existante du plugin :\n\n"
            "`Code Auditor: Generate Tests for Current File`"
        )
    }


# ══════════════════════════════════════════════════════════════════════════════
# Nodes — Phase 2 generation
# ══════════════════════════════════════════════════════════════════════════════

def node_load_project_patterns(state: ChatState) -> Dict[str, Any]:
    """Detect project conventions for code generation."""
    from services.code_generator_service import detect_conventions

    lang = state.get("generation_language") or state.get("target_lang") or "python"
    if lang == "unknown":
        lang = "python"

    try:
        patterns = detect_conventions(state.get("project_path", "."), lang)
    except Exception as e:
        logger.warning("detect_conventions failed: %s", e)
        patterns = {}

    return {"project_patterns": patterns}


def node_generate_completion(state: ChatState) -> Dict[str, Any]:
    """Generate function body completion."""
    from langchain_agents.agents.lc_chat_agent import lc_chat_agent

    result = lc_chat_agent.complete_function(dict(state))
    return result


def node_generate_class(state: ChatState) -> Dict[str, Any]:
    """Generate a new class from scratch."""
    from langchain_agents.agents.lc_chat_agent import lc_chat_agent

    result = lc_chat_agent.generate_class(dict(state))
    return result


def node_validate_generated(state: ChatState) -> Dict[str, Any]:
    """Validate syntax of generated code."""
    from services.code_generator_service import validate_syntax

    code = state.get("generated_code", "")
    lang = state.get("generation_language") or state.get("target_lang") or "python"

    if lang == "unknown":
        lang = "python"

    if not code:
        return {
            "generation_valid": False,
            "generation_errors": ["No code was generated."],
        }

    valid, errors = validate_syntax(code, lang)
    response = state.get("response", "")

    if valid:
        response += "\n\n---\n**Validation:** OK"
    else:
        response += (
            "\n\n---\n"
            "**Validation:** FAILED\n\n"
            + "\n".join(f"- {e}" for e in errors)
        )

    return {
        "generation_valid": valid,
        "generation_errors": errors,
        "response": response,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Nodes — Memory + Formatting
# ══════════════════════════════════════════════════════════════════════════════

def node_memory_save(state: ChatState) -> Dict[str, Any]:
    """Persist user/assistant exchange in Redis + semantic memory."""
    from langchain_agents.agents.lc_chat_agent import lc_chat_agent

    plan = state.get("decision_plan") or {}

    metadata = {
        "intent":           state.get("intent"),
        "target_file":      state.get("target_file"),
        "language":         state.get("target_lang"),
        "context_level":    state.get("context_level"),
        "selected_agents":  state.get("selected_agents", []),
        "decision_reason":  plan.get("reason", ""),
        "generated_code":   state.get("generated_code", ""),
        "cursor_line":      state.get("cursor_line", 0),
        "active_function":  state.get("active_function", ""),
    }

    session_id   = state.get("session_id", "")
    user_message = state.get("user_message", "")
    response     = state.get("response", "")

    # Redis history (existing)
    lc_chat_agent.save_exchange(
        session_id=session_id,
        user_message=user_message,
        response=response,
        metadata=metadata,
    )

    # ── Phase C1: Semantic memory write (background, non-blocking) ────────────
    def _write_semantic():
        try:
            from langchain_agents.memory.lc_semantic_memory import semantic_memory
            semantic_memory.write_memory(
                session_id=session_id,
                user_message=user_message,
                assistant_message=response,
                metadata={"intent": state.get("intent", ""), "file": state.get("target_file", "")},
            )
        except Exception as e:
            logger.debug("semantic memory write failed: %s", e)

    import threading
    threading.Thread(target=_write_semantic, daemon=True).start()

    # Notify LearningAgent only as "suggested", not "accepted".
    intent = state.get("intent", "")
    if intent in ("complete_fn", "new_class", "code_generation"):
        generated = state.get("generated_code", "")
        lang = state.get("generation_language") or state.get("target_lang") or "unknown"
        if generated:
            try:
                from langchain_agents.agents.lc_learning_agent import learning_agent
                learning_agent.submit_feedback(
                    block={
                        "problem": f"code_generation:{intent}",
                        "fixed_code": generated,
                        "severity": "LOW",
                    },
                    action="suggested",
                    language=lang,
                )
            except Exception as e:
                logger.warning("Could not notify LearningAgent: %s", e)

    return {}


def node_format_response(state: ChatState) -> Dict[str, Any]:
    """Final formatting — adds session ID and proactive suggestion count."""
    response   = state.get("response", "")
    session_id = state.get("session_id", "")
    proactive  = state.get("proactive_suggestions", {}) or {}

    formatted = response.strip()

    # Append proactive summary if any suggestions
    if proactive.get("has_critical"):
        n = proactive.get("total", 0)
        formatted += f"\n\n---\n> ⚠️ **{n} suggestion(s) proactive(s)** — consulte `/api/chat/proactive`"

    if session_id:
        routing = (state.get("decision_plan") or {}).get("_routing", "")
        formatted += f"\n\n---\n_session: `{session_id}`_"
        if routing:
            formatted += f" · _routing: {routing}_"

    return {"formatted_response": formatted}


# ══════════════════════════════════════════════════════════════════════════════
# Routing
# ══════════════════════════════════════════════════════════════════════════════

def _route_after_decision(state: ChatState) -> str:
    intent        = state.get("intent", "question")
    context_level = state.get("context_level", "context")

    if intent == "ci_question":
        return "ci_question"
    if intent == "git_question":
        return "git_question"
    # Phase C3: deep context_level → tool-calling pair programmer
    if context_level == "deep" and intent == "question":
        return "semantic_recall"
    return "load_file_context"


def _route_after_file_context(state: ChatState) -> str:
    intent        = state.get("intent", "question")
    context_level = state.get("context_level", "context")

    if intent == "test_generation":
        return "test_generation"

    if context_level == "fast" and state.get("file_code"):
        return "fast_answer"

    # All other paths go through parallel context (RAG + summary + git)
    return "parallel_context"


def _route_after_parallel_context(state: ChatState) -> str:
    intent        = state.get("intent", "question")
    context_level = state.get("context_level", "context")

    if intent in ("complete_fn", "new_class", "code_generation"):
        return "load_project_patterns"

    # Phase C3: very complex questions → tool-calling
    if context_level == "deep" and intent in ("question", "explain"):
        return "tool_calling_answer"

    return "answer_question"


# ══════════════════════════════════════════════════════════════════════════════
# Graph builder
# ══════════════════════════════════════════════════════════════════════════════

def build_chat_graph():
    graph = StateGraph(ChatState)

    # Core orchestration nodes
    graph.add_node("load_memory",        node_load_memory)
    graph.add_node("intent_router",      node_intent_router)
    graph.add_node("decision_agent",     node_decision_agent)
    graph.add_node("load_file_context",  node_load_file_context)
    graph.add_node("parallel_context",   node_parallel_context)
    graph.add_node("rag_retrieve",       node_rag_retrieve)

    # Phase C: semantic recall + tool-calling
    graph.add_node("semantic_recall",    node_semantic_recall)
    graph.add_node("tool_calling_answer",node_tool_calling_answer)

    # Answer nodes
    graph.add_node("fast_answer",        node_fast_answer)
    graph.add_node("answer_question",    node_answer_question)
    graph.add_node("git_question",       node_git_question)
    graph.add_node("ci_question",        node_ci_question)
    graph.add_node("test_generation",    node_test_generation)

    # Generation nodes
    graph.add_node("load_project_patterns", node_load_project_patterns)
    graph.add_node("generate_completion",   node_generate_completion)
    graph.add_node("generate_class",        node_generate_class)
    graph.add_node("validate_generated",    node_validate_generated)

    # Tail nodes
    graph.add_node("memory_save",        node_memory_save)
    graph.add_node("format_response",    node_format_response)

    # ── Entry ────────────────────────────────────────────────────────────────
    graph.set_entry_point("load_memory")
    graph.add_edge("load_memory",   "intent_router")
    graph.add_edge("intent_router", "decision_agent")

    graph.add_conditional_edges(
        "decision_agent",
        _route_after_decision,
        {
            "load_file_context": "load_file_context",
            "git_question":      "git_question",
            "ci_question":       "ci_question",
            "semantic_recall":   "semantic_recall",   # Phase C3 fast track
        },
    )

    # semantic_recall → tool_calling_answer (always)
    graph.add_edge("semantic_recall", "tool_calling_answer")

    graph.add_conditional_edges(
        "load_file_context",
        _route_after_file_context,
        {
            "fast_answer":      "fast_answer",
            "parallel_context": "parallel_context",
            "test_generation":  "test_generation",
        },
    )

    graph.add_conditional_edges(
        "parallel_context",
        _route_after_parallel_context,
        {
            "answer_question":      "answer_question",
            "load_project_patterns":"load_project_patterns",
            "tool_calling_answer":  "tool_calling_answer",  # Phase C3 deep path
        },
    )

    # Generation sub-path
    graph.add_edge("load_project_patterns", "rag_retrieve")
    graph.add_conditional_edges(
        "rag_retrieve",
        lambda s: (
            "generate_completion" if s.get("intent") in ("complete_fn", "code_generation")
            else "generate_class" if s.get("intent") == "new_class"
            else "answer_question"
        ),
        {
            "answer_question":    "answer_question",
            "generate_completion":"generate_completion",
            "generate_class":     "generate_class",
        },
    )

    graph.add_edge("generate_completion", "validate_generated")
    graph.add_edge("generate_class",      "validate_generated")

    # All answer paths → memory_save → format → END
    for node in ("fast_answer", "answer_question", "git_question",
                 "ci_question", "test_generation", "validate_generated",
                 "tool_calling_answer"):
        graph.add_edge(node, "memory_save")

    graph.add_edge("memory_save",    "format_response")
    graph.add_edge("format_response", END)

    return graph.compile()

# ══════════════════════════════════════════════════════════════════════════════
# Nodes — Phase C3 : Semantic recall + Tool-calling pair programmer
# ══════════════════════════════════════════════════════════════════════════════

def node_semantic_recall(state: ChatState) -> Dict[str, Any]:
    """
    Phase C1+C3: Recall relevant facts from the semantic memory store.
    Injects past context into the state for use by node_tool_calling_answer.
    """
    session_id = state.get("session_id", "")
    query      = state.get("user_message", "")
    if not session_id or not query:
        return {"semantic_context": []}
    try:
        from langchain_agents.memory.lc_semantic_memory import semantic_memory
        facts = semantic_memory.recall_memory(session_id=session_id, query=query)
        return {"semantic_context": facts}
    except Exception as e:
        logger.debug("node_semantic_recall: %s", e)
        return {"semantic_context": []}


async def node_tool_calling_answer(state: ChatState) -> Dict[str, Any]:
    """
    Phase C3: LLM-driven pair programmer with real tool calls.

    The LLM autonomously decides which tools to call
    (search_codebase, get_file, get_git_diff, analyze_file, get_dependencies, get_ci_status)
    and synthesizes a final answer from the tool results.
    """
    from langchain_agents.agents.lc_tool_calling_agent import lc_tool_calling_agent

    result = await lc_tool_calling_agent.arun(
        message      = state.get("user_message", ""),
        project_path = state.get("project_path", "."),
        session_id   = state.get("session_id", ""),
        history      = state.get("history", []),
        semantic_context = state.get("semantic_context", []),
        target_file  = state.get("target_file", "") or "",
    )

    return {
        "response":      result.get("response", ""),
        "tools_called":  result.get("tools_called", []),
    }



_chat_graph = None


def _get_chat_graph():
    global _chat_graph
    if _chat_graph is None:
        _chat_graph = build_chat_graph()
    return _chat_graph


async def ainvoke_chat(
    message: str,
    project_path: str = ".",
    session_id: str = "",
    target_file: str = "",
    cursor_line: int = 0,
    active_function: str = "",
    selected_text: str = "",
    visible_range: list | None = None,
    **services: Any,
) -> Dict[str, Any]:
    """Async chat invocation for FastAPI and async LangGraph nodes."""
    graph = _get_chat_graph()
    initial = _initial_state(
        message=message,
        project_path=project_path,
        session_id=session_id,
        target_file=target_file,
        cursor_line=cursor_line,
        active_function=active_function,
        selected_text=selected_text,
        visible_range=visible_range,
        **services,
    )

    start = time.time()
    result = await graph.ainvoke(initial)
    result.setdefault("stats", {})
    result["stats"]["elapsed"] = round(time.time() - start, 2)
    return result


def invoke_chat(
    message: str,
    project_path: str = ".",
    session_id: str = "",
    target_file: str = "",
    **services: Any,
) -> Dict[str, Any]:
    """Synchronous wrapper for CLI usage."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            ainvoke_chat(
                message=message,
                project_path=project_path,
                session_id=session_id,
                target_file=target_file,
                **services,
            )
        )

    raise RuntimeError(
        "invoke_chat() cannot be called from an active event loop. Use await ainvoke_chat()."
    )


def _sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


async def stream_chat(
    message: str,
    project_path: str = ".",
    session_id: str = "",
    target_file: str = "",
    **services: Any,
):
    """
    Streaming equivalent of ainvoke_chat().

    Yields SSE data events:
      - {type: status, content: str}
      - {type: token, content: str}
      - {type: code, content: str}
      - {type: done, ...metadata}
      - {type: error, content: str}
    """
    graph = _get_chat_graph()
    initial = _initial_state(
        message=message,
        project_path=project_path,
        session_id=session_id,
        target_file=target_file,
        **services,
    )

    started_at = time.time()
    final_session_id = session_id or ""
    final_intent = "question"
    final_target_file = target_file or ""
    final_context_level = "context"
    final_response = ""

    yield _sse({"type": "status", "content": "Démarrage..."})

    node_status = {
        "load_memory": "Chargement de la mémoire...",
        "intent_router": "Détection de l’intention...",
        "decision_agent": "Planification multi-agent...",
        "load_file_context": "Lecture du fichier...",
        "project_summary": "Résumé du projet...",
        "rag_retrieve": "Recherche dans le projet...",
        "fast_answer": "Génération rapide...",
        "answer_question": "Génération de la réponse...",
        "load_project_patterns": "Détection des conventions projet...",
        "generate_completion": "Génération du code...",
        "generate_class": "Génération de la classe...",
        "validate_generated": "Validation du code généré...",
        "memory_save": "Sauvegarde de la conversation...",
    }

    try:
        async for event in graph.astream_events(initial, version="v1"):
            kind = event.get("event", "")
            node_name = event.get("name", "")
            data = event.get("data", {}) or {}

            if kind == "on_chain_start" and node_name in node_status:
                yield _sse({"type": "status", "node": node_name, "content": node_status[node_name]})

            elif kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                content = getattr(chunk, "content", "")
                if isinstance(content, list):
                    content = "".join(str(x) for x in content)
                if content:
                    yield _sse({"type": "token", "content": content})

            elif kind == "on_chain_end":
                output = data.get("output", {})

                if node_name == "load_memory" and isinstance(output, dict):
                    final_session_id = output.get("session_id", final_session_id)

                elif node_name == "decision_agent" and isinstance(output, dict):
                    final_intent = output.get("intent", final_intent)
                    final_context_level = output.get("context_level", final_context_level)
                    yield _sse({
                        "type": "plan",
                        "intent": final_intent,
                        "context_level": final_context_level,
                        "selected_agents": output.get("selected_agents", []),
                    })

                elif node_name == "load_file_context" and isinstance(output, dict):
                    final_target_file = output.get("target_file", final_target_file)

                elif node_name in ("generate_completion", "generate_class") and isinstance(output, dict):
                    if output.get("generated_code"):
                        yield _sse({"type": "code", "content": output["generated_code"]})

                elif node_name == "format_response" and isinstance(output, dict):
                    final_response = output.get("formatted_response", final_response)

        yield _sse({
            "type": "done",
            "session_id": final_session_id,
            "intent": final_intent,
            "target_file": final_target_file,
            "context_level": final_context_level,
            "elapsed_seconds": round(time.time() - started_at, 2),
            "response": final_response,
        })

    except Exception as e:
        logger.exception("stream_chat failed: %s", e)
        yield _sse({"type": "error", "content": str(e)})
