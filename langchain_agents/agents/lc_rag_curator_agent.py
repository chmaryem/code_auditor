"""
lc_rag_curator_agent.py — LangChain RAGCuratorAgent.

4 Pillars:
  LLM     : OpenRouter/Groq/Gemini cascade via RunnableWithFallbacks (reuses
            _build_llm_with_fallback() from lc_analysis_agent.py — no raw HTTP)
  Tools   : N/A — single primitive (curate()), exposed as a bare public method,
            no @tool wrapper (Pattern B already established for this module —
            no bind_tools()/agentic loop, so a @tool would add ceremony without
            value; see plan decision 1.2)
  Memory  : N/A — stateless per-request filtering, no notion of "retry" or
            "remembering a file" is relevant here
  Planning: fixed role (Pattern B) — one LLM call per curate() invocation,
            never an autonomous tool-calling loop

Role: filters obvious false-positives out of the RAG context retrieved by
retrieve_rag_context() (test_patterns_kb + ProjectCodeIndexer), BEFORE it
reaches build_prompt(). Keeps the existing similarity-based ranking as-is —
this is additive filtering only, never a full rerank.

Zero-regression contract: curate() returns a dict of the EXACT SAME SHAPE as
its input ({"test_patterns": [...], "project_examples": [...], "total_docs": int}).
Any failure (LLM error, malformed response) returns the ORIGINAL rag_context
unchanged — fail-silent, never raises.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_REMOVE_INDICES_RE = re.compile(r"<REMOVE_INDICES>\s*([\d,\s]*)\s*</REMOVE_INDICES>", re.IGNORECASE)


class RAGCuratorAgent:
    """
    LangChain RAGCuratorAgent — additive false-positive filtering for RAG context.
    """

    def __init__(self):
        self._llm = None

    @property
    def llm(self):
        """Lazy-init LLM with fallback cascade (LLM pillar)."""
        if self._llm is None:
            from langchain_agents.agents.lc_analysis_agent import _build_llm_with_fallback
            self._llm = _build_llm_with_fallback()
        return self._llm

    def curate(
        self,
        rag_context: Dict[str, Any],
        untested_signatures: List[Dict[str, Any]],
        source_path: Path,
    ) -> Dict[str, Any]:
        """
        Filters obvious false-positives out of rag_context. Returns a dict of the
        SAME SHAPE ({"test_patterns", "project_examples", "total_docs"}).

        Fail-silent: any exception, LLM error, or malformed response returns
        rag_context UNCHANGED. Never raises.
        """
        try:
            total_docs = rag_context.get("total_docs", 0)
            if not total_docs:
                # Nothing to filter — skip the LLM call entirely.
                return rag_context

            test_patterns = list(rag_context.get("test_patterns", []))
            project_examples = list(rag_context.get("project_examples", []))
            entries = test_patterns + project_examples
            if not entries:
                return rag_context

            prompt = self._build_curation_prompt(entries, untested_signatures, source_path)
            raw = self.llm.invoke(prompt)
            text = raw.content if hasattr(raw, "content") else str(raw)

            return self._parse_curation_response(text, test_patterns, project_examples)
        except Exception as e:
            logger.debug("RAGCuratorAgent.curate erreur (fail-silent): %s", e)
            return rag_context

    def _build_curation_prompt(
        self,
        entries: List[Dict[str, Any]],
        untested_signatures: List[Dict[str, Any]],
        source_path: Path,
    ) -> str:
        sig_names = ", ".join(s.get("name", "") for s in untested_signatures[:10]) or "(none)"

        entries_text = ""
        for i, entry in enumerate(entries, 1):
            content = str(entry.get("content", ""))[:300]
            entries_text += f"\n[{i}] {content}\n"

        return f"""You are filtering retrieved documents for relevance before they are
used as context to generate unit tests for "{source_path.name}".

Methods to test: {sig_names}

Retrieved documents:
{entries_text}

Identify ONLY documents that are clearly irrelevant (unrelated topic, wrong
language, no useful pattern for testing the methods above). Keep documents
that are even loosely useful — when in doubt, KEEP the document.

Respond with EXACTLY one line, nothing else:
<REMOVE_INDICES>comma,separated,indices</REMOVE_INDICES>
Use an empty tag <REMOVE_INDICES></REMOVE_INDICES> if nothing should be removed."""

    def _parse_curation_response(
        self,
        text: str,
        test_patterns: List[Dict[str, Any]],
        project_examples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        match = _REMOVE_INDICES_RE.search(text or "")
        if not match:
            # Malformed response — keep everything, do not guess.
            return {
                "test_patterns": test_patterns,
                "project_examples": project_examples,
                "total_docs": len(test_patterns) + len(project_examples),
            }

        raw_indices = match.group(1).strip()
        if not raw_indices:
            return {
                "test_patterns": test_patterns,
                "project_examples": project_examples,
                "total_docs": len(test_patterns) + len(project_examples),
            }

        try:
            remove = {int(x.strip()) for x in raw_indices.split(",") if x.strip()}
        except ValueError:
            return {
                "test_patterns": test_patterns,
                "project_examples": project_examples,
                "total_docs": len(test_patterns) + len(project_examples),
            }

        entries = test_patterns + project_examples
        kept = [e for i, e in enumerate(entries, 1) if i not in remove]
        n_patterns = len(test_patterns)
        # Preserve the original test_patterns/project_examples split among kept entries.
        kept_patterns = [e for i, e in enumerate(entries[:n_patterns], 1) if i not in remove]
        kept_examples = [e for i, e in enumerate(entries[n_patterns:], n_patterns + 1) if i not in remove]

        return {
            "test_patterns": kept_patterns,
            "project_examples": kept_examples,
            "total_docs": len(kept_patterns) + len(kept_examples),
        }


rag_curator_agent = RAGCuratorAgent()
