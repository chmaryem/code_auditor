"""
compat.py — Legacy compatibility layer for langchain_agents/.

PURPOSE
-------
Centralises ALL imports from the legacy `agents/` package into one place.
This solves two problems:

  1. sys.path safety: ensures the project root is in sys.path before any
     legacy import, regardless of how/where the LC module is imported from
     (main thread, worker thread, unit test, etc.).

  2. Single source of truth: every LC module imports legacy items from here,
     never directly from `agents.*`. Changing a legacy class name? Fix it here.

USAGE (in any langchain_agents/ module)
----------------------------------------
    # Instead of:  from agents.test_gap_agent import TestGapAgent
    # Do:
    from langchain_agents.compat import TestGapAgent, TestGapStatus

    # Instead of:  from agents.analysis_agent import build_context
    # Do:
    from langchain_agents.compat import build_context
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── Ensure project root is in sys.path ───────────────────────────────────────
# langchain_agents/compat.py → langchain_agents/ → project_root
_ROOT = Path(__file__).parent.parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Legacy: code_agent ────────────────────────────────────────────────────────
from agents.code_agent import code_agent as legacy_code_agent          # noqa: E402

# ── Legacy: analysis_agent ────────────────────────────────────────────────────
from agents.analysis_agent import (                                     # noqa: E402
    AnalysisAgent,
    parse_llm_response,
    build_context,
    build_system_impact_section,
)

# ── Legacy: retriever_agent ───────────────────────────────────────────────────
from agents.retriever_agent import retriever_agent as legacy_retriever_agent  # noqa: E402

# ── Legacy: test_gap_agent ────────────────────────────────────────────────────
from agents.test_gap_agent import (                                     # noqa: E402
    TestGapAgent,
    TestGapStatus,
    test_gap_agent as legacy_test_gap_agent,
)

# ── Legacy: test_proposal_notifier ────────────────────────────────────────────
from agents.test_proposal_notifier import TestProposalNotifier          # noqa: E402

# ── Legacy: learning_agent ────────────────────────────────────────────────────
from agents.learning_agent import learning_agent as legacy_learning_agent  # noqa: E402

__all__ = [
    # code
    "legacy_code_agent",
    # analysis
    "AnalysisAgent",
    "parse_llm_response",
    "build_context",
    "build_system_impact_section",
    # retriever
    "legacy_retriever_agent",
    # test gap
    "TestGapAgent",
    "TestGapStatus",
    "legacy_test_gap_agent",
    # test proposal
    "TestProposalNotifier",
    # learning
    "legacy_learning_agent",
]
