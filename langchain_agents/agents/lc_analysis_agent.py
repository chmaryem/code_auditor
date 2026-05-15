"""
lc_analysis_agent.py — LangChain AnalysisAgent.

4 Pillars:
  LLM     : OpenRouter/Gemini cascade via RunnableWithFallbacks
  Tools   : tool_llm_analyze, tool_build_context, tool_validate_fix_blocks
  Memory  : AnalysisCacheMemory (Redis) — cached results, post-solution flags
  Planning: Decision tree in prompt (full_class / targeted_methods / block_fix)

This agent is the core LLM reasoning engine of the system.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_agents.tools.analysis_tools import (
    tool_build_context,
    tool_llm_analyze,
    tool_validate_fix_blocks,
)
from langchain_agents.memory.redis_memory import AgentRedisMemory, AnalysisCacheMemory

logger = logging.getLogger(__name__)


def _build_llm_with_fallback():
    """
    Build LLM with cascade fallback: OpenRouter → Gemini.

    Uses RunnableWithFallbacks from LangChain for automatic failover.
    """
    from config import config
    import os

    llms = []

    # Primary: OpenRouter
    api_key = config.api.openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")
    if api_key:
        try:
            from langchain_openai import ChatOpenAI
            llms.append(ChatOpenAI(
                model=config.api.openrouter_model,
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=config.api.temperature,
                max_tokens=config.api.max_tokens,
                default_headers={
                    "HTTP-Referer": "https://github.com/code-auditor",
                    "X-Title": "Code Auditor",
                },
            ))
        except Exception as e:
            logger.debug("OpenRouter LLM build failed: %s", e)

    # Fallback: Gemini
    gemini_key = config.api.gemini_api_key or os.getenv("GOOGLE_API_KEY", "")
    if gemini_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llms.append(ChatGoogleGenerativeAI(
                model=config.api.gemini_model,
                google_api_key=gemini_key,
                temperature=config.api.temperature,
                max_output_tokens=config.api.max_tokens,
                convert_system_message_to_human=True,
            ))
        except Exception as e:
            logger.debug("Gemini LLM build failed: %s", e)

    if not llms:
        raise ValueError("No LLM provider available (check API keys)")

    # Build with fallbacks
    primary = llms[0]
    if len(llms) > 1:
        return primary.with_fallbacks(llms[1:])
    return primary


class LCAnalysisAgent:
    """
    LangChain AnalysisAgent — core LLM reasoning engine.

    Anatomy:
      - LLM:      RunnableWithFallbacks (OpenRouter→Gemini)
      - Tools:    llm_analyze, build_context, validate_fix_blocks
      - Memory:   AnalysisCacheMemory (Redis), post-solution flags
      - Planning: Strategy decision tree (full_class/targeted_methods/block_fix)
    """

    def __init__(self):
        self.memory = AgentRedisMemory("analysis_agent")
        self.cache = AnalysisCacheMemory()
        self.tools = [tool_llm_analyze, tool_build_context, tool_validate_fix_blocks]
        self._llm = None

    @property
    def llm(self):
        """Lazy-init LLM with fallback chain (LLM pillar)."""
        if self._llm is None:
            self._llm = _build_llm_with_fallback()
        return self._llm

    # ── Memory: cache checks ─────────────────────────────────────────────────

    def get_cached_analysis(self, file_path: str, content_hash: str) -> Optional[Dict]:
        """Check Redis cache for existing analysis (Memory pillar)."""
        if not self.cache.is_stale(file_path, content_hash):
            return self.cache.get_cached(file_path)
        return None

    def cache_analysis(self, file_path: str, result: Dict, content_hash: str) -> None:
        """Store analysis in Redis cache (Memory pillar)."""
        self.cache.set_cached(file_path, result, content_hash)

    def is_post_solution(self, file_path: str) -> bool:
        """Check if this file was already fixed in current session (Memory pillar)."""
        return self.memory.get(f"post_solution:{file_path}") == "true"

    def mark_post_solution(self, file_path: str) -> None:
        """Mark file as already fixed (Memory pillar)."""
        self.memory.set(f"post_solution:{file_path}", "true", expire_seconds=3600)

    # ── Tool: full analysis ──────────────────────────────────────────────────

    def analyze(
        self,
        code: str,
        context: Dict[str, Any],
        docs: List[Any],
        scores: List[float],
    ) -> Dict[str, Any]:
        """
        Run full LLM analysis (Tool pillar).

        Pipeline:
          1. Check cache → return if fresh
          2. Build prompt with context + RAG docs
          3. Invoke LLM (OpenRouter→Gemini cascade)
          4. Validate fix blocks
          5. Cache result
          6. Return structured result

        Planning pillar is embedded in the prompt:
          → STEP 1: DECIDE REPAIR STRATEGY (full_class / targeted_methods / block_fix)
          → STEP 2: GENERATE THE FIX MATCHING YOUR DECISION
        """
        return tool_llm_analyze.invoke({
            "code": code,
            "context": context,
            "docs": docs,
            "scores": scores,
        })

    # ── Planning: post-solution mode ─────────────────────────────────────────

    def enrich_context_post_solution(
        self,
        context: Dict[str, Any],
        file_path: str,
    ) -> Dict[str, Any]:
        """
        Planning pillar — if file was already fixed, inject post-solution mode.

        This forces the LLM to choose block_fix/targeted_methods instead of
        rewriting the entire file again.
        """
        if self.is_post_solution(file_path):
            context["post_solution_mode"] = True
            context["post_solution_hint"] = (
                "This file was ALREADY rewritten by you in a previous analysis. "
                "DO NOT rewrite the entire class again. Use block_fix only."
            )
        return context


# Singleton
lc_analysis_agent = LCAnalysisAgent()


from agents.analysis_agent import (
    parse_llm_response,
    build_context,
    build_system_impact_section,
)

__all__ = [
    "LCAnalysisAgent",
    "lc_analysis_agent",
    "parse_llm_response",
    "build_context",
    "build_system_impact_section",
]
