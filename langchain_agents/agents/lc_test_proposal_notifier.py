"""
lc_test_proposal_notifier.py — LangChain wrapper for TestProposalNotifier.

4 Pillars:
  LLM     : None (0 token)
  Tools   : None (console output only)
  Memory  : None
  Planning: Delegates to the legacy TestProposalNotifier (preserves all display logic)

This wrapper allows watch_graph.py to stay within the langchain_agents/ namespace
while reusing the rich notification display of the legacy TestProposalNotifier.

It accepts the serializable dict produced by LCTestGapAgent.check() and
reconstructs a native TestGapStatus for the legacy notifier — so all display
levels (INFO / WARN / URGENT) are preserved exactly.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LCTestProposalNotifier:
    """
    LangChain-compatible wrapper around the legacy TestProposalNotifier.

    Accepts the serializable gap dict produced by LCTestGapAgent.check()
    and delegates display to the legacy notifier — preserving all formatting.

    Usage inside watch_graph nodes:
        notifier = LCTestProposalNotifier(print_lock=state.get("_print_lock"))
        notifier.notify(gap_dict)
    """

    def __init__(self, print_lock: Optional[threading.Lock] = None):
        self._lock = print_lock or threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def notify(self, gap: Dict[str, Any]) -> None:
        """
        Display test-gap notification for a single gap dict.

        Args:
            gap: The dict returned by LCTestGapAgent.check() —
                 must contain source_file, missing, coverage_ratio,
                 untested_entities, tested_entities, framework,
                 impact_score, reason.
        """
        if not gap:
            return

        status = self._dict_to_status(gap)
        if status is None:
            # Fallback: plain print if reconstruction fails
            self._fallback_print(gap)
            return

        try:
            from agents.test_proposal_notifier import TestProposalNotifier
            notifier = TestProposalNotifier(print_lock=self._lock)
            notifier.notify(status)
        except Exception as e:
            logger.warning("LCTestProposalNotifier delegation failed: %s", e)
            self._fallback_print(gap)

    def notify_batch(self, gaps: list) -> None:
        """
        Display notification for multiple gap dicts (batch mode).

        Args:
            gaps: List of gap dicts from LCTestGapAgent.check().
        """
        if not gaps:
            return

        statuses = []
        for g in gaps:
            s = self._dict_to_status(g)
            if s:
                statuses.append(s)

        if not statuses:
            return

        try:
            from agents.test_proposal_notifier import TestProposalNotifier
            notifier = TestProposalNotifier(print_lock=self._lock)
            notifier.notify_batch(statuses)
        except Exception as e:
            logger.warning("LCTestProposalNotifier batch failed: %s", e)
            for g in gaps:
                self._fallback_print(g)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _dict_to_status(gap: Dict[str, Any]):
        """
        Reconstruct a native TestGapStatus from an LCTestGapAgent gap dict.

        Note: needs_attention is a @property on TestGapStatus — it must NOT
        be passed as a constructor argument.
        """
        try:
            from agents.test_gap_agent import TestGapStatus

            return TestGapStatus(
                source_file=Path(gap["source_file"]),
                test_file=Path(gap["test_file"]) if gap.get("test_file") else None,
                missing=gap.get("missing", True),
                coverage_ratio=gap.get("coverage_ratio", 0.0),
                untested_entities=gap.get("untested_entities", []),
                tested_entities=gap.get("tested_entities", []),
                framework=gap.get("framework"),
                impact_score=gap.get("impact_score", 0),
                reason=gap.get("reason", ""),
            )
        except Exception as e:
            logger.debug("_dict_to_status failed: %s", e)
            return None

    @staticmethod
    def _fallback_print(gap: Dict[str, Any]) -> None:
        """Plain-text fallback when the rich notifier is unavailable."""
        name = gap.get("source_name", Path(gap.get("source_file", "?")).name)
        reason = gap.get("reason", "missing tests")
        score = gap.get("impact_score", 0)
        print(f"    🧪 Test gap: {name} — {reason} (impact={score})")


# Singleton (re-instantiated in watch_graph with real print_lock)
lc_test_proposal_notifier = LCTestProposalNotifier()
