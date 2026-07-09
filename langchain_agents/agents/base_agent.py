"""
base_agent.py — Reusable tool-calling LangChain agent (real multi-agent core).

Unlike the deterministic CIGraph/CDGraph nodes (which call @tool functions
directly in Python), an LCBaseAgent binds a set of LangChain tools to an LLM
and lets the LLM DECIDE which tools to call, in a loop, until it produces a
final answer. This is the building block for the agentic CI/CD graphs
(ci_agent_graph.py / cd_agent_graph.py).

Design notes:
  - Cascade Groq → OpenRouter → Gemini (same providers as
    lc_analysis_agent._build_llm_with_fallback), BUT tools are bound to each
    provider individually before combining with .with_fallbacks(), so the
    tool-calling capability survives a provider failover (binding tools on the
    RunnableWithFallbacks wrapper does NOT propagate — that is a real gotcha).
  - Graceful degradation: if no provider/tool-binding is available, the agent
    falls back to a single-shot text answer via invoke_with_fallback, so an
    agent node never hard-crashes the graph.
  - Tools are the existing LangChain @tool objects from ci_tools.py /
    cd_tools.py — reused as-is, now actually selected by the LLM.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,       # noqa: F401  (kept for type clarity / future use)
    ToolMessage,
)

logger = logging.getLogger(__name__)


def _build_tool_calling_llm(tools: List[BaseTool]):
    """
    Build a cascade LLM (Groq → OpenRouter → Gemini) with `tools` bound to
    EACH provider, so tool-calling survives a provider failover.

    Returns a Runnable exposing .invoke(messages) that returns an AIMessage
    (possibly carrying .tool_calls), or None if no provider is configured.
    """
    from config import config
    import os

    bound: list = []

    # Primary: Groq (fast + reliable)
    groq_key = config.api.groq_api_key or os.getenv("GROQ_API_KEY", "")
    if groq_key:
        try:
            from langchain_openai import ChatOpenAI
            m = ChatOpenAI(
                model=config.api.groq_model,
                api_key=groq_key,
                base_url="https://api.groq.com/openai/v1",
                temperature=config.api.temperature,
                max_tokens=min(config.api.max_tokens, 8192),
                max_retries=1,
            )
            bound.append(m.bind_tools(tools))
        except Exception as e:
            logger.debug("[Agent] Groq tool-LLM build failed: %s", e)

    # Fallback 1: OpenRouter
    or_key = config.api.openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")
    if or_key:
        try:
            from langchain_openai import ChatOpenAI
            m = ChatOpenAI(
                model=config.api.openrouter_model,
                api_key=or_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=config.api.temperature,
                max_tokens=config.api.max_tokens,
                max_retries=0,
                default_headers={
                    "HTTP-Referer": "https://github.com/code-auditor",
                    "X-Title": "Code Auditor",
                },
            )
            bound.append(m.bind_tools(tools))
        except Exception as e:
            logger.debug("[Agent] OpenRouter tool-LLM build failed: %s", e)

    # Fallback 2: Gemini
    gemini_key = config.api.gemini_api_key or os.getenv("GOOGLE_API_KEY", "")
    if gemini_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            m = ChatGoogleGenerativeAI(
                model=config.api.gemini_model,
                google_api_key=gemini_key,
                temperature=config.api.temperature,
                max_output_tokens=config.api.max_tokens,
                convert_system_message_to_human=True,
                max_retries=1,
            )
            bound.append(m.bind_tools(tools))
        except Exception as e:
            logger.debug("[Agent] Gemini tool-LLM build failed: %s", e)

    if not bound:
        return None
    primary = bound[0]
    return primary.with_fallbacks(bound[1:]) if len(bound) > 1 else primary


class LCBaseAgent:
    """
    A real tool-calling LangChain agent.

    An instance is defined by:
      - name:          short identifier (used in logs)
      - system_prompt: the agent's role / instructions
      - tools:         LangChain @tool objects the LLM may call autonomously

    run(task, context) executes the agentic loop:
      1. LLM sees system prompt + task + context + bound tools
      2. LLM emits tool_calls → we execute them → feed results back
      3. Repeat until the LLM returns a final text answer (or max_iterations)
    """

    MAX_ITERATIONS = 3

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: List[BaseTool],
        max_iterations: int = 0,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools
        self._tool_map = {t.name: t for t in tools}
        self.max_iterations = max_iterations or self.MAX_ITERATIONS
        self._llm = None
        self._llm_built = False

    # ── LLM (lazy) ──────────────────────────────────────────────────────────

    def _get_llm(self):
        if not self._llm_built:
            self._llm = _build_tool_calling_llm(self.tools)
            self._llm_built = True
        return self._llm

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the agent on a task.

        Args:
            task:    instruction / question for the agent
            context: optional dict injected into the first human message

        Returns:
            {response: str, tools_called: list, tool_results: dict, iterations: int}
        """
        llm = self._get_llm()
        tools_called: List[str] = []
        tool_results: Dict[str, str] = {}
        raw_tool_results: Dict[str, Any] = {}

        ctx_block = self._format_context(context)

        # No LLM available → degrade to a single-shot text answer.
        if llm is None:
            return self._text_fallback(task, ctx_block, tools_called, tool_results)

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"{task}{ctx_block}"),
        ]

        iterations = 0
        final = ""
        llm_error = False
        while iterations < self.max_iterations:
            iterations += 1
            try:
                ai = llm.invoke(messages)
            except Exception as e:
                logger.warning("[Agent:%s] LLM invoke failed: %s", self.name, e)
                final = f"(LLM error: {e})"
                llm_error = True
                break

            tool_calls = getattr(ai, "tool_calls", None) or []
            if not tool_calls:
                final = getattr(ai, "content", "") or ""
                break

            messages.append(ai)
            for tc in tool_calls:
                tname = tc.get("name", "")
                targs = tc.get("args", {}) or {}
                tid = tc.get("id", f"call_{iterations}_{tname}")
                result = self._execute_tool(tname, targs)
                tools_called.append(tname)
                tool_results[tname] = str(result)[:1500]
                # Keep the last structured result per tool — CD agents need the
                # real dict (score, grade, rollback fields) for the dashboard.
                raw_tool_results[tname] = result
                # Trimmed payload fed back to the LLM keeps token usage (and the
                # Groq TPM rate-limit) under control.
                messages.append(ToolMessage(
                    content=str(result)[:1500],
                    tool_call_id=tid,
                ))

        # A truncated agentic loop that never produced a final answer is also
        # treated as a soft failure so callers can fall back cleanly.
        if not final:
            final = "(no response)"

        logger.info(
            "[Agent:%s] done — iterations=%d tools=%s error=%s",
            self.name, iterations, tools_called or "none", llm_error,
        )
        return {
            "response": final,
            "tools_called": tools_called,
            "tool_results": tool_results,
            "raw_tool_results": raw_tool_results,
            "iterations": iterations,
            "llm_error": llm_error,
        }

    # ── Internals ───────────────────────────────────────────────────────────

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
        tool = self._tool_map.get(name)
        if tool is None:
            return f"[unknown tool: {name}]"
        try:
            return tool.invoke(args)
        except Exception as e:
            logger.warning("[Agent:%s] tool %s failed: %s", self.name, name, e)
            return f"[tool {name} error: {e}]"

    @staticmethod
    def _format_context(context: Optional[Dict[str, Any]]) -> str:
        if not context:
            return ""
        lines = []
        for k, v in context.items():
            sval = str(v)
            if len(sval) > 1500:
                sval = sval[:1500] + "…"
            lines.append(f"- {k}: {sval}")
        return "\n\n## Context\n" + "\n".join(lines)

    def _text_fallback(
        self,
        task: str,
        ctx_block: str,
        tools_called: List[str],
        tool_results: Dict[str, str],
    ) -> Dict[str, Any]:
        err = False
        try:
            from services.llm_factory import invoke_with_fallback
            text = invoke_with_fallback(
                f"{self.system_prompt}\n\n{task}{ctx_block}",
                label=f"agent_{self.name}_fallback",
                max_tokens=1024,
            )
        except Exception as e:
            text = f"(agent {self.name} unavailable: {e})"
            err = True
        return {
            "response": text,
            "tools_called": tools_called,
            "tool_results": tool_results,
            "raw_tool_results": {},
            "iterations": 0,
            "llm_error": err,
        }
