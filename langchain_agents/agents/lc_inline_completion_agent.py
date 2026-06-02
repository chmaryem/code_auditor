"""
lc_inline_completion_agent.py — Cursor-aware inline code completion.

Phase B — Feature B2.

Unlike the function-level complete_fn (which needs a function name),
this agent works like Copilot: it receives the code BEFORE and AFTER the
cursor position and generates the missing code fragment.

Usage:
    POST /api/chat/complete/inline
    {
      "file_path":   "src/service.py",
      "cursor_line": 42,
      "prefix_code": "def calculate_score(self, user_id:",
      "suffix_code": "\\n    return result",
      "language":    "python",
      "project_path": "."
    }

    Response:
    {
      "completion":   "user: User, weight: float = 1.0) -> float:",
      "confidence":   0.92,
      "tokens_used":  64,
      "source":       "llm" | "cache"
    }
"""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

_INLINE_PROMPT = """\
You are an expert {language} code completion engine embedded in an IDE.

Complete the code at the CURSOR position. Output ONLY the completion text \
(no explanation, no markdown, no code fences). The completion must:
- Fit naturally between prefix and suffix
- Match the project's coding style
- Be concise (typically 1-5 lines)
- Be syntactically valid {language}

Project context:
  File: {file_path}
  Language: {language}

Relevant code patterns from project (RAG):
{rag_snippet}

--- PREFIX (code before cursor) ---
{prefix}
--- CURSOR HERE ---
--- SUFFIX (code after cursor) ---
{suffix}
---

Return ONLY the completion text to insert at the cursor:"""


class LCInlineCompletionAgent:
    """
    Inline code completion agent.

    Pipeline:
      1. Hash (prefix + suffix) → check Redis cache (TTL 5min)
      2. Optional RAG retrieval for project patterns
      3. LLM call (fast model, max 128 tokens)
      4. Cache result
    """

    CACHE_TTL = 300   # 5 minutes
    MAX_TOKENS = 128

    def __init__(self):
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            try:
                from langchain_agents.agents.lc_analysis_agent import _build_llm_with_fallback
                self._llm = _build_llm_with_fallback()
            except Exception:
                pass
        return self._llm

    # ── Cache ─────────────────────────────────────────────────────────────────

    def _cache_key(self, prefix: str, suffix: str, language: str) -> str:
        raw = f"{language}:{prefix[-200:]}:{suffix[:100]}"
        return "inline:completion:" + hashlib.md5(raw.encode()).hexdigest()

    def _cache_get(self, key: str) -> str | None:
        try:
            from services.mcp_redis_service import get_mcp_redis
            val = get_mcp_redis().get(key)
            return val.decode() if isinstance(val, bytes) else val
        except Exception:
            return None

    def _cache_set(self, key: str, value: str) -> None:
        try:
            from services.mcp_redis_service import get_mcp_redis
            get_mcp_redis().setex(key, self.CACHE_TTL, value)
        except Exception:
            pass

    # ── RAG snippet ───────────────────────────────────────────────────────────

    def _get_rag_snippet(
        self,
        project_path: str,
        query: str,
        language: str,
        max_chars: int = 800,
    ) -> str:
        """Get a short RAG snippet from the project for style context."""
        try:
            from langchain_agents.tools.chat_tools import tool_chat_rag_retrieve
            result = tool_chat_rag_retrieve.invoke({
                "project_path": project_path,
                "query":        query[:300],
                "target_file":  "",
                "file_code":    "",
                "language":     language,
            })
            docs = result.get("rag_docs", [])[:2]
            snippet = "\n\n".join(d.get("content", "")[:400] for d in docs)
            return snippet[:max_chars] if snippet else "(no project patterns found)"
        except Exception as e:
            logger.debug("RAG snippet failed: %s", e)
            return "(RAG unavailable)"

    # ── Main entry ────────────────────────────────────────────────────────────

    def complete(
        self,
        prefix_code: str,
        suffix_code: str,
        language: str = "python",
        file_path: str = "",
        project_path: str = ".",
        use_rag: bool = True,
    ) -> Dict[str, Any]:
        """
        Synchronous inline completion.

        Returns:
            {completion, confidence, source, elapsed_ms}
        """
        t0 = time.time()
        language = language or "python"

        # 1. Cache check
        cache_key = self._cache_key(prefix_code, suffix_code, language)
        cached = self._cache_get(cache_key)
        if cached:
            return {
                "completion":  cached,
                "confidence":  0.85,
                "source":      "cache",
                "elapsed_ms":  round((time.time() - t0) * 1000),
            }

        # 2. RAG snippet (optional)
        rag_snippet = ""
        if use_rag and project_path:
            # Use the last meaningful token from prefix as query
            query = prefix_code.strip().split("\n")[-1][:100] or file_path
            rag_snippet = self._get_rag_snippet(project_path, query, language)

        # 3. Build prompt
        prompt = _INLINE_PROMPT.format(
            language    = language,
            file_path   = file_path or "(unknown)",
            rag_snippet = rag_snippet or "(none)",
            prefix      = prefix_code[-1500:],   # last 1500 chars
            suffix      = suffix_code[:300],      # next 300 chars
        )

        # 4. LLM call
        completion = ""
        source     = "llm"
        try:
            if self.llm is not None:
                from langchain_core.messages import HumanMessage
                result = self.llm.invoke(
                    [HumanMessage(content=prompt)],
                    config={"max_tokens": self.MAX_TOKENS},
                )
                completion = getattr(result, "content", str(result)).strip()
            else:
                from services.llm_factory import invoke_with_fallback
                completion = invoke_with_fallback(
                    prompt, label="inline_completion", max_tokens=self.MAX_TOKENS
                ).strip()
        except Exception as e:
            logger.warning("inline completion LLM failed: %s", e)
            return {
                "completion":  "",
                "confidence":  0.0,
                "source":      "error",
                "elapsed_ms":  round((time.time() - t0) * 1000),
                "error":       str(e),
            }

        # Strip any accidental fences
        import re
        completion = re.sub(r"^```[a-z]*\n?", "", completion)
        completion = re.sub(r"\n?```$",        "", completion.strip())

        # 5. Cache
        if completion:
            self._cache_set(cache_key, completion)

        elapsed = round((time.time() - t0) * 1000)
        confidence = 0.9 if len(completion) > 5 else 0.5

        logger.debug(
            "inline_completion: lang=%s len=%d source=%s elapsed=%dms",
            language, len(completion), source, elapsed,
        )

        return {
            "completion":  completion,
            "confidence":  confidence,
            "source":      source,
            "elapsed_ms":  elapsed,
        }


lc_inline_completion_agent = LCInlineCompletionAgent()
