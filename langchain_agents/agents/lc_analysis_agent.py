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

import json
import re as _re

logger = logging.getLogger(__name__)


def parse_structured_output(text: str) -> dict:
    """
    Extrait et valide le bloc JSON <STRUCTURED_OUTPUT> de la réponse LLM.

    Le LLM est invité (via le prompt) à terminer sa réponse par un bloc :
        <STRUCTURED_OUTPUT>
        { "strategy": "...", "strategy_reason": "...", "issues": [...], "fixes": [...] }
        </STRUCTURED_OUTPUT>

    Si ce bloc est absent ou malformé, retourne un dict vide — les appelants
    (node_emit_ws_events) reviennent alors aux parseurs regex de fallback.

    Args:
        text: Texte brut de la réponse LLM (analysis["analysis"]).

    Returns:
        Dict avec les clés :
            strategy        (str)  : "full_class" | "targeted_methods" | "block_fix"
            strategy_reason (str)  : explication courte, max 100 chars
            issues          (list) : liste de dicts conformes à WSIssueV2
            fixes           (list) : liste de dicts conformes à WSFixV2
        Ou {} si le bloc n'est pas trouvé / JSON invalide.
    """
    if not text:
        return {}

    # Chercher le bloc délimité
    match = _re.search(
        r"<STRUCTURED_OUTPUT>\s*(.*?)\s*</STRUCTURED_OUTPUT>",
        text,
        _re.DOTALL,
    )
    if not match:
        return {}

    raw_json = match.group(1).strip()

    # Retirer les balises de code markdown si présentes : ```json ... ```
    raw_json = _re.sub(r"^```[a-z]*\s*", "", raw_json)
    raw_json = _re.sub(r"\s*```$", "", raw_json)

    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.debug("parse_structured_output: JSON parse failed: %s", exc)
        return {}

    if not isinstance(data, dict):
        return {}

    # Normalisation légère — garantit les clés attendues
    return {
        "strategy":        str(data.get("strategy", "block_fix")),
        "strategy_reason": str(data.get("strategy_reason", ""))[:120],
        "issues":          _normalize_issues(data.get("issues", [])),
        "fixes":           _normalize_fixes(data.get("fixes", [])),
    }


def _normalize_issues(raw: list) -> list:
    """Sanitise la liste d'issues pour garantir les types attendus."""
    if not isinstance(raw, list):
        return []
    normalized = []
    _sev_valid = {"critical", "error", "warning", "info"}
    for item in raw:
        if not isinstance(item, dict):
            continue
        sev = str(item.get("severity", "warning")).lower()
        if sev not in _sev_valid:
            sev = "warning"
        normalized.append({
            "title":      str(item.get("title", ""))[:120],
            "message":    str(item.get("message", item.get("title", "")))[:500],
            "line":       _safe_int(item.get("line")),
            "column":     _safe_int(item.get("column")),
            "end_line":   _safe_int(item.get("end_line")),
            "end_column": _safe_int(item.get("end_column")),
            "severity":   sev,
            "rule":       str(item.get("rule", "code-auditor"))[:80],
            "suggestion": str(item.get("suggestion", ""))[:300],
        })
    return normalized[:20]   # cap at 20 issues


def _normalize_fixes(raw: list) -> list:
    """Sanitise la liste de fixes pour garantir les types attendus."""
    if not isinstance(raw, list):
        return []
    normalized = []
    _mode_valid = {"replace_snippet", "replace_method", "full_file"}
    for item in raw:
        if not isinstance(item, dict):
            continue
        mode = str(item.get("apply_mode", "replace_snippet"))
        if mode not in _mode_valid:
            mode = "replace_snippet"
        current = str(item.get("current_code", "")).strip()
        fixed   = str(item.get("fixed_code",   "")).strip()
        if not current or not fixed or current == fixed:
            continue
        normalized.append({
            "title":        str(item.get("title", "Suggested fix"))[:160],
            "explanation":  str(item.get("explanation", ""))[:500],
            "line":         _safe_int(item.get("line")),
            "apply_mode":   mode,
            "current_code": current,
            "fixed_code":   fixed,
        })
    return normalized[:10]   # cap at 10 fixes


def _safe_int(value) -> "int | None":
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _build_llm_with_fallback():
    """
    Build LLM with cascade fallback: OpenRouter → Groq → Gemini.

    Uses RunnableWithFallbacks from LangChain for automatic failover.
    """
    from config import config
    import os

    llms = []

    # Primary: Groq (rapide + fiable — OpenRouter free est très souvent saturé en 429,
    # ce qui faisait perdre un aller-retour à CHAQUE analyse avant ce changement)
    groq_key = config.api.groq_api_key or os.getenv("GROQ_API_KEY", "")
    if groq_key:
        try:
            from langchain_openai import ChatOpenAI
            llms.append(ChatOpenAI(
                model=config.api.groq_model,
                api_key=groq_key,
                base_url="https://api.groq.com/openai/v1",
                temperature=config.api.temperature,
                max_tokens=min(config.api.max_tokens, 8192),  # marge sous la TPM Groq (30k)
                max_retries=1,
            ))
        except Exception as e:
            logger.debug("Groq LLM build failed: %s", e)

    # Fallback 1: OpenRouter (utilisé seulement si Groq est indisponible)
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
                max_retries=0,
                default_headers={
                    "HTTP-Referer": "https://github.com/code-auditor",
                    "X-Title": "Code Auditor",
                },
            ))
        except Exception as e:
            logger.debug("OpenRouter LLM build failed: %s", e)

    # Fallback 2: Gemini
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
                max_retries=1,  # évite le spam de 8 retries quand le free tier Gemini est à 0
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
    "parse_structured_output",
    "parse_llm_response",
    "build_context",
    "build_system_impact_section",
]
