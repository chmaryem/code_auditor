"""
lc_inline_completion_agent.py — Smart cursor-aware inline code completion.

3-tier pipeline (per keystroke):
  Tier 1 — In-memory cache          < 1 ms   repeated patterns
  Tier 2 — Graph symbol fast path   < 5 ms   unambiguous name completion (no LLM)
  Tier 3 — LLM with project context ~500 ms  everything else (Groq-first cascade)

Project context for the LLM is built from the DependencyGraph already loaded at
server startup — zero additional I/O per keystroke.
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

_INLINE_PROMPT = """\
Complete the {language} code at <CURSOR>. Output ONLY the inserted text — \
no explanation, no markdown fences.
Keep it concise (1–5 lines). Must be syntactically valid {language}.
{context_section}
<prefix>{prefix}</prefix><CURSOR><suffix>{suffix}</suffix>

Completion:"""


# ── Fast LLM builder ──────────────────────────────────────────────────────────

def _build_fast_llm():
    """
    Low-latency LLM cascade: Groq → Gemini-flash → OpenRouter.
    Every client is bound to MAX_TOKENS so completions stay short and fast.
    """
    from config import config
    import os

    llms = []

    groq_key = config.api.groq_api_key or os.getenv("GROQ_API_KEY", "")
    if groq_key:
        try:
            from langchain_openai import ChatOpenAI
            llms.append(ChatOpenAI(
                model=config.api.groq_model,
                api_key=groq_key,
                base_url="https://api.groq.com/openai/v1",
                temperature=0.0,
                max_tokens=LCInlineCompletionAgent.MAX_TOKENS,
                max_retries=0,
            ))
        except Exception as e:
            logger.debug("Groq fast LLM build failed: %s", e)

    gemini_key = config.api.gemini_api_key or os.getenv("GOOGLE_API_KEY", "")
    if gemini_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llms.append(ChatGoogleGenerativeAI(
                model=config.api.gemini_model,
                google_api_key=gemini_key,
                temperature=0.0,
                max_output_tokens=LCInlineCompletionAgent.MAX_TOKENS,
                max_retries=0,
            ))
        except Exception as e:
            logger.debug("Gemini fast LLM build failed: %s", e)

    or_key = config.api.openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")
    if or_key:
        try:
            from langchain_openai import ChatOpenAI
            llms.append(ChatOpenAI(
                model=config.api.openrouter_model,
                api_key=or_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=0.0,
                max_tokens=LCInlineCompletionAgent.MAX_TOKENS,
                max_retries=0,
            ))
        except Exception as e:
            logger.debug("OpenRouter fast LLM build failed: %s", e)

    if not llms:
        raise ValueError("No fast LLM available (check GROQ/GOOGLE/OPENROUTER keys)")

    primary = llms[0]
    return primary.with_fallbacks(llms[1:]) if len(llms) > 1 else primary


# ── Agent ──────────────────────────────────────────────────────────────────────

class LCInlineCompletionAgent:
    """
    Smart inline completion agent.

    Pipeline:
      1. In-memory cache                      — sub-ms, repeated patterns
      2. Graph fast path (Phase 1)            — unambiguous symbol, 0 LLM
         • local file symbols visible before cursor
         • imported symbols (by name, typed into the import list)
      3. LLM call with enriched project context:
         Phase A — local symbols visible before cursor
         Phase B — imported symbols' signatures
         Phase C — project-wide prefix-token match
    """

    CACHE_TTL     = 300   # 5 min  — completion in-memory TTL
    CTX_CACHE_TTL = 120   # 2 min  — project context block TTL per file
    MAX_TOKENS    = 128   # ~4 lines; keeps LLM latency low

    def __init__(self):
        self._fast_llm = None
        self._completion_cache: Dict[str, tuple] = {}  # hash → (ts, text)
        self._ctx_cache:        Dict[str, tuple] = {}  # file → (ts, ctx_str)

    @property
    def fast_llm(self):
        if self._fast_llm is None:
            try:
                self._fast_llm = _build_fast_llm()
            except Exception as e:
                logger.warning("fast LLM unavailable: %s", e)
        return self._fast_llm

    # ── Tier 1 — In-memory cache ──────────────────────────────────────────────

    def _cache_key(self, prefix: str, suffix: str, language: str) -> str:
        raw = f"{language}:{prefix[-200:]}:{suffix[:100]}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _cache_get(self, key: str) -> Optional[str]:
        # Never read MCP Redis: it returns error strings for missing keys,
        # which would render as verbatim ghost text.
        hit = self._completion_cache.get(key)
        if hit and (time.time() - hit[0]) < self.CACHE_TTL:
            return hit[1]
        return None

    def _cache_set(self, key: str, value: str) -> None:
        self._completion_cache[key] = (time.time(), value)
        try:
            from services.mcp_redis_service import get_mcp_redis
            get_mcp_redis().set(f"inline:{key}", value, expire_seconds=self.CACHE_TTL)
        except Exception:
            pass

    # ── Tier 2 — Graph fast path (Phase 1) ───────────────────────────────────

    def _last_partial_token(self, prefix_code: str) -> str:
        """
        Extract the partial identifier the developer is currently typing.
        Returns '' if the context is ambiguous (inside string, comment, after '.').
        """
        if not prefix_code:
            return ""

        last_line = prefix_code.rstrip("\n").split("\n")[-1]

        # Skip comment lines
        stripped = last_line.lstrip()
        if stripped.startswith("#") or stripped.startswith("//"):
            return ""

        # Skip if inside an unclosed string (odd number of non-escaped quotes)
        sq = last_line.count("'") - last_line.count("\\'")
        dq = last_line.count('"') - last_line.count('\\"')
        if (sq % 2) != 0 or (dq % 2) != 0:
            return ""

        # Must end on an identifier character
        if not last_line or not (last_line[-1].isalnum() or last_line[-1] == "_"):
            return ""

        tokens = re.findall(r"[a-zA-Z_]\w*", last_line)
        if not tokens:
            return ""

        token = tokens[-1]
        if len(token) < 3:
            return ""

        # Skip attribute access (after '.')
        idx = last_line.rfind(token)
        if idx > 0 and last_line[idx - 1] == ".":
            return ""

        return token

    def _entity_sig(self, e: Any) -> str:
        params    = ", ".join(e.parameters) if getattr(e, "parameters", None) else ""
        ret       = f" -> {e.return_type}" if getattr(e, "return_type", None) else ""
        async_pfx = "async " if getattr(e, "is_async", False) else ""
        if e.type == "class":
            return f"class {e.name}"
        return f"{async_pfx}def {e.name}({params}){ret}"

    def _try_graph_completion(
        self,
        file_path: str,
        prefix_code: str,
        cursor_line: int,
    ) -> Optional[str]:
        """
        Tier 2: if a partial token matches EXACTLY ONE symbol in scope, return the
        completion suffix without calling the LLM.
        Only triggers for unambiguous, single-candidate matches.
        """
        token = self._last_partial_token(prefix_code)
        if not token or not file_path:
            return None

        candidates: List[Any] = []

        try:
            from services.graph_service import dependency_builder

            norm      = str(Path(file_path).resolve())
            threshold = cursor_line if cursor_line > 0 else 10_000

            # Local symbols visible before the cursor
            local = (
                dependency_builder.file_entities.get(file_path)
                or dependency_builder.file_entities.get(norm)
                or []
            )
            for e in local:
                if (
                    e.name != token
                    and e.name.startswith(token)
                    and getattr(e, "start_line", 0) < threshold
                    and e.type in ("function", "method", "class")
                ):
                    candidates.append(e)

            # Imported symbols
            file_imports = (
                dependency_builder.file_imports.get(file_path)
                or dependency_builder.file_imports.get(norm)
                or []
            )
            for imp in file_imports:
                module_stem = imp.module.lstrip(".").split(".")[-1]
                if not module_stem:
                    continue
                for fpath, entities in dependency_builder.file_entities.items():
                    fp = Path(fpath)
                    if fp.stem == module_stem or fp.name == f"{module_stem}.py":
                        for e in entities:
                            if (
                                e.name != token
                                and e.name.startswith(token)
                                and e.type in ("function", "class", "method")
                                and (not imp.items or e.name in imp.items)
                            ):
                                candidates.append(e)
                        break

        except Exception as exc:
            logger.debug("_try_graph_completion error: %s", exc)
            return None

        if len(candidates) != 1:
            return None  # ambiguous or no match → fall through to LLM

        e           = candidates[0]
        name_suffix = e.name[len(token):]  # portion left to type after the prefix

        if e.type == "class":
            return name_suffix

        # Function/method: append name tail + call signature
        params = ", ".join(e.parameters) if getattr(e, "parameters", None) else ""
        return f"{name_suffix}({params})"

    # ── Tier 3 helper — Project context block (Phases A + B + C) ─────────────

    def _build_project_context(
        self,
        file_path: str,
        prefix_code: str,
        cursor_line: int,
    ) -> str:
        """
        Build project-aware context injected into the LLM prompt.
        Uses the DependencyGraph already built at server startup — no I/O.
        Result cached per file (invalidated by file mtime).
        """
        if not file_path:
            return ""

        try:
            file_mtime = Path(file_path).stat().st_mtime
        except OSError:
            file_mtime = 0.0

        ctx_key = f"{file_path}:{int(file_mtime)}"
        hit = self._ctx_cache.get(ctx_key)
        if hit and (time.time() - hit[0]) < self.CTX_CACHE_TTL:
            return hit[1]

        parts: List[str] = []

        try:
            from services.graph_service import dependency_builder

            norm      = str(Path(file_path).resolve())
            threshold = cursor_line if cursor_line > 0 else 10_000

            # ── Phase A: local symbols before cursor ─────────────────────────
            local = (
                dependency_builder.file_entities.get(file_path)
                or dependency_builder.file_entities.get(norm)
                or []
            )
            visible = [
                e for e in local
                if getattr(e, "start_line", 0) < threshold
                and e.type in ("function", "method", "class")
            ]
            if visible:
                parts.append(
                    "Local symbols:\n"
                    + "\n".join(f"  {self._entity_sig(e)}" for e in visible[:12])
                )

            # ── Phase B: imported symbols' signatures ────────────────────────
            file_imports = (
                dependency_builder.file_imports.get(file_path)
                or dependency_builder.file_imports.get(norm)
                or []
            )
            imp_sigs: List[str] = []
            for imp in file_imports[:10]:
                module_stem = imp.module.lstrip(".").split(".")[-1]
                if not module_stem:
                    continue
                for fpath, entities in dependency_builder.file_entities.items():
                    fp = Path(fpath)
                    if fp.stem != module_stem and fp.name != f"{module_stem}.py":
                        continue
                    if imp.items:
                        relevant = [e for e in entities if e.name in imp.items]
                    else:
                        relevant = [
                            e for e in entities
                            if not e.name.startswith("_")
                            and e.type in ("function", "class", "method")
                        ][:5]
                    for e in relevant[:4]:
                        imp_sigs.append(f"{self._entity_sig(e)}  # {fp.name}")
                    break

            if imp_sigs:
                parts.append(
                    "Imported symbols:\n"
                    + "\n".join(f"  {s}" for s in imp_sigs[:12])
                )

            # ── Phase C: project-wide prefix-token match ─────────────────────
            token = self._last_partial_token(prefix_code)
            if token:
                matches: List[str] = []
                low = token.lower()
                for fpath, entities in dependency_builder.file_entities.items():
                    if fpath == file_path or fpath == norm:
                        continue  # already in Phase A
                    fp_name = Path(fpath).name
                    for e in entities:
                        if (
                            e.name.lower().startswith(low)
                            and e.type in ("function", "class", "method")
                        ):
                            matches.append(
                                f"{self._entity_sig(e)}  # {fp_name}"
                            )
                    if len(matches) >= 6:
                        break
                if matches:
                    parts.append(
                        f"Symbols matching '{token}...':\n"
                        + "\n".join(f"  {m}" for m in matches[:6])
                    )

        except Exception as exc:
            logger.debug("_build_project_context failed: %s", exc)
            return ""

        ctx = "\n\n".join(parts)
        self._ctx_cache[ctx_key] = (time.time(), ctx)
        return ctx

    # ── Public entry ──────────────────────────────────────────────────────────

    def complete(
        self,
        prefix_code:  str,
        suffix_code:  str,
        language:     str = "python",
        file_path:    str = "",
        project_path: str = ".",
        cursor_line:  int = 0,
        use_rag:      bool = False,
    ) -> Dict[str, Any]:
        """
        Synchronous inline completion — 3-tier pipeline.

        Returns:
            {completion, confidence, source, elapsed_ms}
        """
        t0       = time.time()
        language = language or "python"

        # ── Tier 1: in-memory cache (< 1ms) ──────────────────────────────────
        cache_key = self._cache_key(prefix_code, suffix_code, language)
        cached    = self._cache_get(cache_key)
        if cached:
            return {
                "completion": cached,
                "confidence": 0.85,
                "source":     "cache",
                "elapsed_ms": round((time.time() - t0) * 1000),
            }

        # ── Tier 2: graph fast path (< 5ms, no LLM) ──────────────────────────
        fast = self._try_graph_completion(file_path, prefix_code, cursor_line)
        if fast is not None:
            self._cache_set(cache_key, fast)
            logger.debug(
                "inline_completion [graph]: %r  elapsed=%dms",
                fast, round((time.time() - t0) * 1000),
            )
            return {
                "completion": fast,
                "confidence": 0.95,
                "source":     "graph",
                "elapsed_ms": round((time.time() - t0) * 1000),
            }

        # ── Tier 3: LLM with project context ─────────────────────────────────
        ctx = self._build_project_context(file_path, prefix_code, cursor_line)
        context_section = (
            f"\nProject context (match style and types):\n{ctx}\n" if ctx else ""
        )

        prompt = _INLINE_PROMPT.format(
            language        = language,
            context_section = context_section,
            prefix          = prefix_code[-1500:],
            suffix          = suffix_code[:300],
        )

        completion = ""
        source     = "llm"
        try:
            if self.fast_llm is not None:
                from langchain_core.messages import HumanMessage
                result     = self.fast_llm.invoke([HumanMessage(content=prompt)])
                completion = getattr(result, "content", str(result)).strip()
            else:
                from services.llm_factory import invoke_with_fallback
                completion = invoke_with_fallback(
                    prompt, label="inline_completion", max_tokens=self.MAX_TOKENS
                ).strip()
        except Exception as e:
            logger.warning("inline completion LLM failed: %s", e)
            return {
                "completion": "",
                "confidence": 0.0,
                "source":     "error",
                "elapsed_ms": round((time.time() - t0) * 1000),
                "error":      str(e),
            }

        # Strip accidental fences
        completion = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", completion)
        completion = re.sub(r"\n?```$", "", completion.strip())

        if completion:
            self._cache_set(cache_key, completion)

        elapsed    = round((time.time() - t0) * 1000)
        confidence = 0.9 if len(completion) > 5 else 0.5

        logger.debug(
            "inline_completion [llm]: lang=%s len=%d ctx=%s elapsed=%dms",
            language, len(completion), "yes" if ctx else "no", elapsed,
        )

        return {
            "completion": completion,
            "confidence": confidence,
            "source":     source,
            "elapsed_ms": elapsed,
        }


lc_inline_completion_agent = LCInlineCompletionAgent()
