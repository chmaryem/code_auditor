"""
lc_code_agent.py — LangChain CodeAgent.

4 Pillars:
  LLM     : None (deterministic — no LLM calls)
  Tools   : tool_parse_file, tool_analyze_change, tool_detect_language
  Memory  : AgentRedisMemory → stores old_content for diff comparison
  Planning: Score-based gating → score < 20 = skip, score ≥ 20 = continue

This agent handles file parsing, change detection, and significance scoring.
It is the FIRST agent in the WatchGraph pipeline.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_agents.tools.code_tools import (
    tool_analyze_change,
    tool_detect_language,
    tool_parse_file,
)
from langchain_agents.memory.redis_memory import AgentRedisMemory

logger = logging.getLogger(__name__)


class LCCodeAgent:
    """
    LangChain-style CodeAgent — deterministic file analysis.

    Anatomy:
      - LLM:      None
      - Tools:    parse_file, analyze_change, detect_language
      - Memory:   Redis (old file content for diff)
      - Planning: Score threshold (skip if < 20)
    """

    SIGNIFICANCE_THRESHOLD = 20

    def __init__(self):
        self.memory = AgentRedisMemory("code_agent")
        self.tools = [tool_parse_file, tool_analyze_change, tool_detect_language]

    # ── Memory: old content ──────────────────────────────────────────────────

    def _get_old_content(self, file_path: str) -> str:
        """Retrieve previously stored content for this file (Memory pillar)."""
        return self.memory.get(f"content:{file_path}") or ""

    def _save_content(self, file_path: str, content: str) -> None:
        """Store current content for future diff comparison (Memory pillar)."""
        self.memory.set(
            f"content:{file_path}",
            content,
            expire_seconds=86400 * 7,  # 7 days TTL
        )

    # ── Tools: wrappers ──────────────────────────────────────────────────────

    def parse(self, file_path: Path) -> Dict[str, Any]:
        """Parse file via tool_parse_file (Tool pillar)."""
        return tool_parse_file.invoke({"file_path": str(file_path)})

    def detect_language(self, file_path: Path) -> str:
        """Detect language via tool (Tool pillar)."""
        return tool_detect_language.invoke({"file_path": str(file_path)})

    def analyze_change(self, file_path: str, new_content: str) -> Dict[str, Any]:
        """
        Compare new content against stored old content (Memory + Tool).

        Returns change_info dict with: significant, score, lines_changed,
        change_type, reason.
        """
        old_content = self._get_old_content(file_path)
        result = tool_analyze_change.invoke({
            "old_content": old_content,
            "new_content": new_content,
        })
        # Update memory with new content
        self._save_content(file_path, new_content)
        return result

    # ── Planning: significance gating ────────────────────────────────────────

    def should_analyze(self, change_info: Dict[str, Any]) -> bool:
        """
        Planning pillar — decide if LLM analysis is warranted.

        Returns True if the change is significant enough.
        """
        if not change_info.get("significant", True):
            return False
        score = change_info.get("score", 100)
        return score >= self.SIGNIFICANCE_THRESHOLD

    # ── Hash check ───────────────────────────────────────────────────────────

    @staticmethod
    def compute_hash(content: str) -> str:
        """Compute SHA-256 hash for content change detection."""
        return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]

    def has_content_changed(self, file_path: str, current_content: str) -> bool:
        """Check if file content hash has changed since last analysis."""
        current_hash = self.compute_hash(current_content)
        stored_hash = self.memory.get(f"hash:{file_path}")
        if stored_hash == current_hash:
            return False
        # Update stored hash
        self.memory.set(f"hash:{file_path}", current_hash)
        return True


# Singleton
lc_code_agent = LCCodeAgent()
