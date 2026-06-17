"""
hardening_state.py — State for the HardeningGraph (agentic file-hardening loop).

The HardeningGraph is the first CYCLIC graph in the platform.
Unlike WatchGraph (acyclic fan-out), this graph loops:

  analyze → plan → fix → verify → decide ──► loop back to plan
                                          └──► done / escalate

The State accumulates across iterations so the agent REMEMBERS what it tried
and never repeats a failed approach on the same issue.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


# ─── Sub-types ────────────────────────────────────────────────────────────────

class IssueAttempt(TypedDict, total=False):
    """One attempt to fix one issue — stored in attempts[] to avoid retrying."""
    issue_id:    str      # unique key: f"{severity}:{line}:{rule}"
    approach:    str      # description of what was tried (for LLM context)
    patch:       str      # the actual patched code tried
    verdict:     str      # "static_fail" | "tests_fail" | "accepted" | "skipped"
    fail_reason: str      # human reason (from _fix_breaks_code or test stderr)


class StagedPatch(TypedDict, total=False):
    """A verified patch waiting for developer approval (never written to disk)."""
    issue_id:      str
    issue_summary: str   # "ERROR: SQL injection · line 12"
    approach:      str
    original_code: str
    patched_code:  str
    static_ok:     bool
    tests_ok:      Optional[bool]   # None = tests still running / skipped
    test_summary:  str


# ─── Main State ───────────────────────────────────────────────────────────────

class HardeningState(TypedDict, total=False):
    """
    State for the HardeningGraph agentic loop.

    Flows through every node and persists across iterations.
    The graph mutates this dict in-place (LangGraph merge semantics).
    """

    # ── Input (set once at launch, never mutated) ──────────────────────────────
    file_path:      str
    project_path:   str
    language:       str
    goal_score:     int            # stop when score >= this (default 90)
    max_iterations: int            # hard brake (default 8)
    mode:           str            # "step_by_step" | "autonomous" | "supervised"

    # ── Loop counters ─────────────────────────────────────────────────────────
    iteration:      int            # current loop index (starts at 0)

    # ── Live state of the file (refreshed each analyze pass) ──────────────────
    source_code:    str            # current in-memory content (NOT on disk yet)
    issues:         List[Dict[str, Any]]   # parsed issues from last analyze
    score:          int            # quality score from last analyze (0-100)
    score_initial:  int            # score at launch (for final delta report)

    # ── Architecture pipeline (same fields as WatchState) ─────────────────────
    # Computed once in node_analyze, reused across fix/verify/decide iterations
    # so we don't hit ChromaDB + NetworkX on every single fix attempt.
    neighborhood:   Dict[str, Any]         # NetworkX dependency graph neighborhood
    rag_docs:       List[Dict[str, Any]]   # serialized ChromaDB 2-pass RAG docs
    rag_scores:     List[float]            # reranker scores for rag_docs
    patterns:       List[str]             # KG vulnerability pattern names
    context:        Dict[str, Any]         # enriched LLM context (same as WatchState.context)

    # ── Per-iteration working vars (overwritten each loop) ─────────────────────
    current_issue:  Optional[Dict[str, Any]]   # the issue being worked on
    current_fix:    Optional[Dict[str, Any]]   # raw fix from LLM ({current_code, fixed_code, …})
    verify_static:  Optional[str]              # None = pass, else = break reason
    verify_tests:   Optional[bool]             # None = not run yet

    # ── Memory — accumulated across ALL iterations ─────────────────────────────
    attempts:       List[IssueAttempt]         # every try, win or fail
    staged_patches: List[StagedPatch]          # accepted patches awaiting dev approval

    # ── Routing ───────────────────────────────────────────────────────────────
    status: Literal[
        "planning",    # choosing next issue
        "fixing",      # LLM generating a fix
        "verifying",   # static + test gates
        "awaiting",    # step-by-step: waiting for dev Keep/Reject/Skip
        "done",        # goal reached or no more fixable issues
        "stuck",       # max_iter reached or no progress N turns
        "escalate",    # issue needs human — agent cannot fix it
    ]
    skip_reason:    Optional[str]     # why we skipped an issue
    done_reason:    str               # final message for the report

    # ── Speculative pipeline ───────────────────────────────────────────────────
    # Pre-computed next fix so the UI shows it instantly when dev clicks Keep.
    next_issue:     Optional[Dict[str, Any]]
    next_fix:       Optional[Dict[str, Any]]

    # ── Streaming callback ────────────────────────────────────────────────────
    # callback(event: dict) → None  pushed to SSE / WS for live plugin updates.
    # Same pattern as WatchState._ws_broadcast.
    _broadcast:     Any

    # ── Injected services (same lazy-import pattern as watch_graph nodes) ──────
    _runner:        Any   # TestRunner instance (injected at launch)
