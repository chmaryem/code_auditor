"""
lc_test_gap_agent.py — LangChain TestGapAgent wrapper.

Wraps the legacy TestGapAgent for integration into the LangGraph WatchGraph.

Responsibilities:
  1. Detect if the file IS already a test file (skip)
  2. Check coverage via TestDiscoveryService
  3. Compute impact score (0 token LLM)
  4. Return a serializable dict for the WatchState

Patterns by language:
  Java     : *Test.java, Test*.java
  Python   : test_*.py, *_test.py
  JS/TS    : *.test.js, *.spec.js, *.test.ts, *.spec.ts
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Test file detection patterns ────────────────────────────────────────────
_TEST_PATTERNS: Dict[str, List[str]] = {
    "java":     ["Test.java", "Tests.java"],
    "python":   ["test_", "_test.py"],
    "javascript": [".test.js", ".spec.js"],
    "typescript": [".test.ts", ".spec.ts", ".test.tsx", ".spec.tsx"],
}


def _is_test_file(file_path: Path, language: str = "") -> bool:
    """Return True if the file itself is a test file (not a source file)."""
    name = file_path.name
    lower_name = name.lower()
    ext = file_path.suffix.lower()

    # Language-agnostic heuristic first
    if "test" in lower_name or "spec" in lower_name:
        # But exclude things like "testing_guide.md" — only code files
        if ext in (".java", ".py", ".js", ".ts", ".jsx", ".tsx"):
            return True

    # Language-specific suffix checks
    lang = language.lower() if language else ""
    for pat in _TEST_PATTERNS.get(lang, []):
        if name.endswith(pat) or name.startswith(pat):
            return True

    return False


class LCTestGapAgent:
    """
    LangChain-compatible TestGapAgent.
    Reuses TestDiscoveryService + TestGapStatus from the legacy module.
    """

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self._discovery = None
        self._agent = None

    def _init_services(self):
        """Lazy-init discovery + gap agent (avoid heavy imports at module level)."""
        if self._discovery is None:
            from services.test_discovery import TestDiscoveryService
            from agents.test_gap_agent import TestGapAgent
            self._discovery = TestDiscoveryService(self.project_path)
            self._agent = TestGapAgent(self.project_path)

    def check(
        self,
        source_file: str,
        parsed_entities: List[Dict[str, Any]],
        change_info: Optional[Dict[str, Any]] = None,
        language: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Check if a source file has missing tests.

        Returns:
            Serializable dict with test-gap info, or None if the file IS a test
            file or no gap was detected.
        """
        fp = Path(source_file)

        # ── 1. Skip test files ───────────────────────────────────────────────
        if _is_test_file(fp, language):
            logger.debug("Skipping test file: %s", fp.name)
            return None

        # ── 2. Lazy-init services ────────────────────────────────────────────
        self._init_services()

        try:
            # Use legacy TestGapAgent for full logic
            status = self._agent.check(
                source_file=fp,
                parsed_entities=parsed_entities,
                change_info=change_info,
            )
        except Exception as e:
            logger.debug("TestGapAgent check failed for %s: %s", fp.name, e)
            return None

        # ── 3. Serialize for WatchState ──────────────────────────────────────
        # Only return if attention is needed (avoid polluting state)
        if not status.needs_attention:
            return None

        return {
            "source_file": str(status.source_file),
            "source_name": status.source_file.name,
            "test_file": str(status.test_file) if status.test_file else None,
            "missing": status.missing,
            "coverage_ratio": status.coverage_ratio,
            "untested_entities": status.untested_entities,
            "tested_entities": status.tested_entities,
            "framework": status.framework,
            "impact_score": status.impact_score,
            "reason": status.reason,
            "needs_attention": status.needs_attention,
        }

    def is_test_file(self, file_path: str, language: str = "") -> bool:
        """Public API — check if a file is a test file."""
        return _is_test_file(Path(file_path), language)


# Singleton (re-instantiated in watch_graph with real project_path)
lc_test_gap_agent = LCTestGapAgent(Path("."))
