"""
hardening_graph.py — Agentic file-hardening loop (HardeningGraph).

THE FIRST CYCLIC GRAPH in the platform. Unlike WatchGraph (acyclic fan-out),
this graph loops until the file is clean, the goal score is reached,
or the agent is stuck.

Graph topology:
  START → analyze → plan ──► fix → verify → decide ──┐
                     │                                  │
                     │  (loop back)  ◄──────────────────┘
                     └──► done / stuck / escalate → END

Conditional edges:
  - plan:   no_fixable_issue → done
  - decide: keep      → plan   (next issue)
            retry     → fix    (same issue, different approach)
            stuck     → done   (max retries on this issue)
            max_iter  → done   (hard brake)

Design principles:
  - All nodes use lazy imports (same as watch_graph.py)
  - State accumulates attempts[] so the agent NEVER retries a failed approach
  - _fix_breaks_code is the FIRST verify gate (< 50ms, no LLM)
  - TestRunner runs AFTER static gate — only on impacted tests
  - _broadcast callback pushes live events to SSE/WS (same as _ws_broadcast)
  - Nothing is written to disk — staged_patches holds in-memory patches only
"""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Dict, Literal

from langgraph.graph import END, StateGraph

from langchain_agents.graphs.hardening_state import HardeningState, IssueAttempt, StagedPatch

logger = logging.getLogger(__name__)

_R  = "\033[0m"
_GR = "\033[92m"
_YL = "\033[93m"
_CY = "\033[96m"
_RD = "\033[91m"


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _issue_id(issue: Dict[str, Any]) -> str:
    """Stable unique key for an issue — used to track attempts."""
    sev  = str(issue.get("severity", ""))
    line = str(issue.get("line", ""))
    rule = str(issue.get("rule", issue.get("message", ""))[:40])
    return f"{sev}:{line}:{rule}"


def _broadcast(state: HardeningState, event: Dict[str, Any]) -> None:
    """Push a live event to the SSE/WS callback if wired."""
    cb = state.get("_broadcast")
    if callable(cb):
        try:
            cb(event)
        except Exception:
            pass


def _already_tried(state: HardeningState, issue_id: str) -> int:
    """Return number of times we already attempted this issue."""
    return sum(1 for a in state.get("attempts", []) if a.get("issue_id") == issue_id)


def _already_accepted(state: HardeningState, issue_id: str) -> bool:
    """Return True if this issue was already accepted into staged_patches."""
    return any(
        p.get("issue_id") == issue_id
        for p in state.get("staged_patches", [])
    )


MAX_RETRIES_PER_ISSUE = 2   # after 2 failed approaches, skip the issue
MAX_ITERATIONS        = 8   # absolute hard brake
SCORE_FLOOR_PROGRESS  = 3   # if score doesn't move after N iters, stop


# ═══════════════════════════════════════════════════════════════════════════════
# Node: analyze
# ═══════════════════════════════════════════════════════════════════════════════

def _score_from_issues(issues: list) -> int:
    """
    Derive a quality score (0-100) from the issue list.
    Mirrors the penalty weights used in the analysis prompt so the
    hardening loop converges toward the same target as the dashboard.
    """
    penalty = 0
    weights = {"critical": 25, "error": 25, "high": 20, "warning": 10, "medium": 10, "low": 5, "info": 2}
    for issue in issues:
        sev = str(issue.get("severity", "info")).lower()
        penalty += weights.get(sev, 5)
    return max(0, 100 - penalty)


def node_analyze(state: HardeningState) -> Dict[str, Any]:
    """
    Full-intelligence analysis on current in-memory source_code.

    Respects the SAME pipeline as watch_graph:
      1. get_neighborhood    → NetworkX dependency graph
      2. retrieve_with_context → ChromaDB 2-pass RAG + reranker
      3. detect_patterns      → KnowledgeGraph vulnerability patterns
      4. build_system_impact  → enriched LLM context
      5. lc_analysis_agent.analyze(docs=rag_docs) → LLM with full KB context
    """
    from langchain_agents.agents.lc_analysis_agent import (
        lc_analysis_agent, parse_structured_output, build_system_impact_section,
    )
    from langchain_agents.agents.lc_retriever_agent import lc_retriever_agent
    from langchain_agents.graphs.watch_graph import _parse_issues_from_llm

    file_path   = state.get("file_path", "")
    source_code = state.get("source_code", "")
    language    = state.get("language", "python")
    iteration   = state.get("iteration", 0)

    print(f"\n  {_CY}[iter {iteration}] analyze{_R} — {Path(file_path).name}")
    t0 = time.time()

    # ── Step 1: NetworkX neighborhood (same as node_get_neighborhood) ──────────
    try:
        neighborhood = lc_retriever_agent.get_neighborhood(file_path)
        criticality  = neighborhood.get("criticality", 0)
        if criticality:
            print(f"    Criticality: {criticality} dependents")
    except Exception as e:
        logger.warning("get_neighborhood failed: %s", e)
        neighborhood = {}

    # ── Step 2: RAG retrieval (same as node_rag_retrieve) ─────────────────────
    try:
        rag_result = lc_retriever_agent.retrieve_with_context(
            code=source_code,
            file_path=file_path,
            language=language,
            neighborhood=neighborhood,
        )
        rag_docs_raw = rag_result.get("docs", [])
        rag_scores   = rag_result.get("scores", [])
        # Serialize Document objects → dicts (same as node_rag_retrieve)
        rag_docs = [
            {"content": d.page_content, "metadata": d.metadata}
            for d in rag_docs_raw
        ]
        print(f"    RAG: {len(rag_docs)} documents retrieved")
    except Exception as e:
        logger.warning("RAG retrieval failed: %s", e)
        rag_docs, rag_scores = [], []

    # ── Step 3: KG patterns (same as node_rag_retrieve) ───────────────────────
    try:
        patterns = lc_retriever_agent.detect_patterns(source_code, language)
        if patterns:
            print(f"    KG patterns: {', '.join(patterns[:3])}")
    except Exception as e:
        logger.warning("KG pattern detection failed: %s", e)
        patterns = []

    # ── Step 4: Build enriched context (same as node_build_context) ───────────
    try:
        system_impact = build_system_impact_section(Path(file_path).name, neighborhood)
    except Exception:
        system_impact = ""

    context = {
        "file_path":            file_path,
        "language":             language,
        "criticality_score":    neighborhood.get("criticality", 0),
        "dependencies":         neighborhood.get("successors", []),
        "dependents":           neighborhood.get("predecessors", []),
        "is_entry_point":       neighborhood.get("is_entry_point", False),
        "system_impact_section": system_impact,
        "kg_patterns":          patterns,
        # Hardening-specific additions
        "hardening":            True,
        "iteration":            iteration,
        "goal_score":           state.get("goal_score", 90),
    }

    # ── Step 5: LLM analysis with FULL intelligence ────────────────────────────
    try:
        result   = lc_analysis_agent.analyze(
            code=source_code,
            context=context,
            docs=rag_docs,
            scores=rag_scores,
        )
        raw_text = result.get("analysis", "") if isinstance(result, dict) else str(result)

        structured = parse_structured_output(raw_text)
        if structured.get("issues") is not None:
            issues = structured["issues"]
        else:
            issues = _parse_issues_from_llm(raw_text, file_path)

        score = _score_from_issues(issues)

    except Exception as e:
        logger.exception("node_analyze LLM failed: %s", e)
        issues = state.get("issues", [])
        score  = state.get("score", 0)

    elapsed = time.time() - t0
    print(f"    {_GR}done{_R} ({elapsed:.1f}s) — {len(issues)} issues · score {score}")

    _broadcast(state, {
        "type":      "hardening_analyze",
        "iteration": iteration,
        "issues":    issues,
        "score":     score,
        "elapsed":   round(elapsed, 2),
    })

    updates: Dict[str, Any] = {
        "issues":       issues,
        "score":        score,
        "neighborhood": neighborhood,
        "rag_docs":     rag_docs,
        "rag_scores":   rag_scores,
        "patterns":     patterns,
        "context":      context,
    }
    if iteration == 0:
        updates["score_initial"] = score

    return updates


# ═══════════════════════════════════════════════════════════════════════════════
# Node: plan
# ═══════════════════════════════════════════════════════════════════════════════

def node_plan(state: HardeningState) -> Dict[str, Any]:
    """
    Choose the next issue to attack.

    Strategy:
      - Skip issues already accepted into staged_patches
      - Skip issues that exceeded MAX_RETRIES_PER_ISSUE
      - Pick the highest-severity remaining issue
      - If none left → status = done
    """
    issues     = state.get("issues", [])
    iteration  = state.get("iteration", 0)
    max_iter   = state.get("max_iterations", MAX_ITERATIONS)
    goal_score = state.get("goal_score", 90)
    score      = state.get("score", 0)

    print(f"  {_CY}[iter {iteration}] plan{_R}")

    # Hard brakes — check BEFORE incrementing so the counter never exceeds max
    if iteration >= max_iter:
        print(f"    {_YL}max_iterations reached ({max_iter}){_R}")
        return {"status": "done", "iteration": iteration,
                "done_reason": f"Max iterations ({max_iter}) reached"}

    if score >= goal_score:
        print(f"    {_GR}goal score reached ({score}/{goal_score}){_R}")
        return {"status": "done", "iteration": iteration,
                "done_reason": f"Goal score reached ({score}/{goal_score})"}

    # Increment AFTER brakes, so iteration = #times we actually run fix→decide
    iteration = iteration + 1

    # Severity order for picking next issue
    severity_rank = {"CRITICAL": 0, "ERROR": 0, "HIGH": 1, "WARNING": 2, "MEDIUM": 2, "LOW": 3, "INFO": 4}

    # Single-word meta-labels from the parser (e.g. "DECISION", "ANALYSIS") are
    # not real code bugs — the LLM emits them as structural markers when it lacks
    # a real finding. Trying to fix them wastes iterations and produces garbage.
    _META_LABELS = frozenset({"decision", "analysis", "meta", "advisory", "note", "summary"})

    candidates = []
    for issue in issues:
        iid = _issue_id(issue)
        if _already_accepted(state, iid):
            continue
        tries = _already_tried(state, iid)
        if tries >= MAX_RETRIES_PER_ISSUE:
            continue
        msg_lower = str(issue.get("message", "")).strip().lower()
        if msg_lower in _META_LABELS:
            print(f"    skip meta-label: [{iid}]")
            continue
        rank = severity_rank.get(str(issue.get("severity", "INFO")).upper(), 5)
        candidates.append((rank, iid, issue))

    if not candidates:
        staged = len(state.get("staged_patches", []))
        print(f"    {_GR}no more fixable issues — {staged} patches staged{_R}")
        return {
            "status":        "done",
            "iteration":     iteration,
            "done_reason":   f"All fixable issues resolved · {staged} patch(es) staged",
            "current_issue": None,
        }

    candidates.sort(key=lambda x: x[0])
    _, chosen_id, chosen = candidates[0]
    tries = _already_tried(state, chosen_id)

    print(f"    picked: [{chosen.get('severity','?')}] {str(chosen.get('message',''))[:60]} (try {tries+1})")

    _broadcast(state, {
        "type":      "hardening_plan",
        "iteration": iteration,
        "issue":     chosen,
        "try_num":   tries + 1,
    })

    return {
        "status":        "fixing",
        "iteration":     iteration,
        "current_issue": chosen,
        "current_fix":   None,
        "verify_static": None,
        "verify_tests":  None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node: fix
# ═══════════════════════════════════════════════════════════════════════════════

def node_fix(state: HardeningState) -> Dict[str, Any]:
    """
    Generate a targeted fix for current_issue.

    Reuses the architecture pipeline already built in node_analyze:
      - neighborhood from state  → already computed by NetworkX, no re-fetch
      - rag_docs/scores from state → KB rules already retrieved by ChromaDB
      - build enriched context   → same structure as node_build_context
      - lc_analysis_agent.analyze(docs=rag_docs) → LLM with FULL KB intelligence

    Adds hardening-specific context on top:
      - target_issue / target_severity / target_line / target_message
      - prior_failures injected as negative examples (never repeat same approach)
      - instruction pin: fix ONLY this specific issue
    """
    from langchain_agents.agents.lc_analysis_agent import (
        lc_analysis_agent, parse_structured_output, build_system_impact_section,
    )
    from langchain_agents.graphs.watch_graph import _extract_fixes_from_llm

    issue        = state.get("current_issue", {})
    source_code  = state.get("source_code", "")
    language     = state.get("language", "python")
    file_path    = state.get("file_path", "")
    iteration    = state.get("iteration", 0)
    issue_id     = _issue_id(issue)

    # ── Reuse pipeline artifacts already in state (no redundant I/O) ──────────
    neighborhood = state.get("neighborhood", {})
    rag_docs     = state.get("rag_docs", [])
    rag_scores   = state.get("rag_scores", [])
    patterns     = state.get("patterns", [])

    prior_failures = [
        a for a in state.get("attempts", [])
        if a.get("issue_id") == issue_id and a.get("verdict") != "accepted"
    ]

    print(f"  {_CY}[iter {iteration}] fix{_R} — {str(issue.get('message',''))[:50]}")
    if prior_failures:
        print(f"    {_YL}{len(prior_failures)} prior failure(s) injected as negative context{_R}")
    print(f"    RAG: {len(rag_docs)} docs · KG patterns: {len(patterns)}")

    prior_str = ""
    if prior_failures:
        prior_str = "\n".join(
            f"- Tried: {f.get('approach','?')[:80]} → FAILED: {f.get('fail_reason','?')[:80]}"
            for f in prior_failures
        )

    # ── Build enriched context (same structure as node_build_context) ──────────
    try:
        system_impact = build_system_impact_section(Path(file_path).name, neighborhood)
    except Exception:
        system_impact = ""

    context = {
        # Pipeline fields (same as WatchState.context)
        "file_path":             file_path,
        "language":              language,
        "criticality_score":     neighborhood.get("criticality", 0),
        "dependencies":          neighborhood.get("successors", []),
        "dependents":            neighborhood.get("predecessors", []),
        "is_entry_point":        neighborhood.get("is_entry_point", False),
        "system_impact_section": system_impact,
        "kg_patterns":           patterns,
        # Hardening-specific targeting
        "hardening_mode":        True,
        "target_issue":          issue,
        "target_severity":       issue.get("severity", ""),
        "target_line":           issue.get("line", ""),
        "target_message":        issue.get("message", ""),
        "prior_failures":        prior_str,
        "instruction": (
            f"Fix ONLY this specific issue: [{issue.get('severity','?')}] "
            f"{issue.get('message','')} at line {issue.get('line','?')}.\n"
            f"CRITICAL: You are hardening {Path(file_path).name} ONLY. "
            f"In every fix block, 'current_code' MUST be a snippet that exists "
            f"verbatim in {Path(file_path).name}. "
            f"NEVER use code from other files as 'current_code'.\n"
            + (f"Do NOT repeat these failed approaches:\n{prior_str}" if prior_str else "")
        ),
    }

    # ── LLM fix with FULL KB intelligence ─────────────────────────────────────
    t0 = time.time()
    current_fix = None
    try:
        result   = lc_analysis_agent.analyze(
            code=source_code,
            context=context,
            docs=rag_docs,       # KB rules from ChromaDB — same as watch_graph
            scores=rag_scores,
        )
        raw_text = result.get("analysis", "") if isinstance(result, dict) else str(result)

        structured = parse_structured_output(raw_text)
        if structured.get("fixes"):
            fixes_raw = structured["fixes"]
        else:
            fixes_raw = _extract_fixes_from_llm(
                raw_text, file_path, language,
                source_code=source_code, strategy="block_fix",
            )

        current_fix = fixes_raw[0] if fixes_raw else None

    except Exception as e:
        logger.exception("node_fix LLM failed: %s", e)

    elapsed = time.time() - t0
    has_fix = bool(current_fix and current_fix.get("fixed_code", "").strip())
    print(f"    {_GR if has_fix else _YL}{'fix generated' if has_fix else 'no fix produced'}{_R} ({elapsed:.1f}s)")

    _broadcast(state, {
        "type":      "hardening_fix",
        "iteration": iteration,
        "issue":     issue,
        "has_fix":   has_fix,
    })

    return {
        "status":      "verifying",
        "current_fix": current_fix,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node: verify
# ═══════════════════════════════════════════════════════════════════════════════

def node_verify(state: HardeningState) -> Dict[str, Any]:
    """
    Two-tier verification gate — reuses existing infrastructure:

    Tier 1 (sync, < 50ms): _fix_breaks_code from watch_graph
      → rejects fixes that introduce syntax errors or undefined names
    Tier 2 (async-ish): TestRunner on impacted test files
      → optional, only when a test file exists for this source file
    """
    from langchain_agents.graphs.watch_graph import _fix_breaks_code

    issue       = state.get("current_issue", {})
    current_fix = state.get("current_fix")
    source_code = state.get("source_code", "")
    language    = state.get("language", "python")
    file_path   = state.get("file_path", "")
    iteration   = state.get("iteration", 0)

    print(f"  {_CY}[iter {iteration}] verify{_R}")

    # ── No fix was produced ────────────────────────────────────────────────────
    if not current_fix or not current_fix.get("fixed_code", "").strip():
        print(f"    {_YL}no fix to verify → skip{_R}")
        _broadcast(state, {"type": "hardening_verify", "iteration": iteration,
                           "tier1": "no_fix", "tier2": None})
        return {"verify_static": "no fix produced", "verify_tests": None}

    patched = current_fix.get("fixed_code", "").strip()

    # ── Tier 1: static gate ────────────────────────────────────────────────────
    # Apply the patch in-memory to get the full file, then compare full↔full.
    # Passing the raw snippet to _fix_breaks_code would make function params
    # look "undefined" (they're not in scope of the snippet alone).
    full_patched = _apply_patch(source_code, current_fix, language)
    t0 = time.time()
    static_reason = _fix_breaks_code(source_code, full_patched, language)
    static_ms = round((time.time() - t0) * 1000)

    if static_reason:
        print(f"    {_RD}tier1 FAIL{_R} ({static_ms}ms) — {static_reason}")
        _broadcast(state, {"type": "hardening_verify", "iteration": iteration,
                           "tier1": "fail", "reason": static_reason, "tier2": None})
        return {"verify_static": static_reason, "verify_tests": None}

    print(f"    {_GR}tier1 pass{_R} ({static_ms}ms)")

    # ── Tier 2: run impacted tests ─────────────────────────────────────────────
    runner   = state.get("_runner")
    tests_ok: "bool | None" = None
    test_summary = ""

    if runner is not None:
        # Find the test file for this source (convention: test_<name>.py)
        src_stem   = Path(file_path).stem
        src_dir    = Path(file_path).parent
        candidates = [
            src_dir / f"test_{src_stem}.py",
            src_dir / f"{src_stem}_test.py",
            Path(state.get("project_path", ".")) / "tests" / f"test_{src_stem}.py",
        ]
        test_file = next((p for p in candidates if p.exists()), None)

        if test_file:
            print(f"    running {test_file.name}…")
            t1 = time.time()
            run_result = runner.run(test_file, language)
            elapsed_t = round(time.time() - t1, 2)
            tests_ok     = run_result.success
            test_summary = run_result.error_summary if not run_result.success else "all pass"
            status_str   = f"{_GR}tests pass{_R}" if tests_ok else f"{_RD}tests FAIL{_R}"
            print(f"    {status_str} ({elapsed_t}s) — {test_summary[:80]}")
        else:
            print(f"    {_YL}no test file found — skipping tier2{_R}")
    else:
        print(f"    {_YL}no runner injected — skipping tier2{_R}")

    _broadcast(state, {
        "type":         "hardening_verify",
        "iteration":    iteration,
        "tier1":        "pass",
        "tier2":        tests_ok,
        "test_summary": test_summary,
    })

    return {"verify_static": None, "verify_tests": tests_ok}


# ═══════════════════════════════════════════════════════════════════════════════
# Node: decide
# ═══════════════════════════════════════════════════════════════════════════════

def node_decide(state: HardeningState) -> Dict[str, Any]:
    """
    After verify, decide what to do:
      - verify failed → record attempt, retry or skip
      - verify passed, mode == step_by_step → stage patch, status = awaiting
      - verify passed, mode == autonomous   → stage patch, continue to plan
    """
    issue        = state.get("current_issue", {})
    current_fix  = state.get("current_fix") or {}
    static_fail  = state.get("verify_static")
    tests_fail   = state.get("verify_tests") is False
    language     = state.get("language", "python")
    mode         = state.get("mode", "step_by_step")
    iteration    = state.get("iteration", 0)
    issue_id     = _issue_id(issue)
    source_code  = state.get("source_code", "")

    patched      = current_fix.get("fixed_code", "").strip()
    approach     = current_fix.get("explanation", current_fix.get("approach", "LLM fix"))

    # ── FAIL: static or tests ──────────────────────────────────────────────────
    if static_fail or tests_fail:
        fail_reason = static_fail or "tests failed"
        attempt: IssueAttempt = {
            "issue_id":    issue_id,
            "approach":    approach,
            "patch":       patched,
            "verdict":     "static_fail" if static_fail else "tests_fail",
            "fail_reason": fail_reason,
        }
        attempts = list(state.get("attempts", [])) + [attempt]
        retries  = sum(1 for a in attempts if a.get("issue_id") == issue_id)

        print(f"  {_RD}decide{_R}: fail ({fail_reason[:60]}) — try {retries}/{MAX_RETRIES_PER_ISSUE}")

        _broadcast(state, {
            "type":       "hardening_decide",
            "iteration":  iteration,
            "verdict":    "fail",
            "reason":     fail_reason,
            "retries":    retries,
        })

        next_status = "fixing" if retries < MAX_RETRIES_PER_ISSUE else "planning"
        return {"attempts": attempts, "status": next_status}

    # ── PASS ──────────────────────────────────────────────────────────────────
    # Guard: don't double-stage or double-record if this issue was already accepted
    if _already_accepted(state, issue_id):
        return {"status": "planning"}

    patch: StagedPatch = {
        "issue_id":      issue_id,
        "issue_summary": f"{issue.get('severity','?')}: {str(issue.get('message',''))[:60]}",
        "approach":      approach,
        "original_code": current_fix.get("current_code", ""),
        "patched_code":  patched,
        "static_ok":     True,
        "tests_ok":      state.get("verify_tests"),
        "test_summary":  "",
    }

    attempt_accepted: IssueAttempt = {
        "issue_id":    issue_id,
        "approach":    approach,
        "patch":       patched,
        "verdict":     "accepted",
        "fail_reason": "",
    }

    staged   = list(state.get("staged_patches", [])) + [patch]
    attempts = list(state.get("attempts", []))        + [attempt_accepted]

    # Apply patch to in-memory source (so next analyze sees updated code)
    new_source = _apply_patch(source_code, current_fix, language)

    print(f"  {_GR}decide{_R}: accepted — {len(staged)} patch(es) staged")

    _broadcast(state, {
        "type":      "hardening_decide",
        "iteration": iteration,
        "verdict":   "accepted",
        "patch":     patch,
        "mode":      mode,
    })

    next_status = "awaiting" if mode == "step_by_step" else "planning"

    return {
        "staged_patches": staged,
        "attempts":       attempts,
        "source_code":    new_source,
        "status":         next_status,
    }


def _apply_patch(source: str, fix: Dict[str, Any], language: str) -> str:
    """
    Apply the accepted fix to the in-memory source.
    Tries current_code → fixed_code replacement first (anchored),
    falls back to full-file replace when current_code not found.
    Never writes to disk.
    """
    patched    = fix.get("fixed_code", "").strip()
    current    = (fix.get("current_code") or "").strip()

    if not patched:
        return source

    if current and current in source:
        return source.replace(current, patched, 1)

    # Full-file replace (apply_mode = full_file)
    if len(patched) > 50:
        return patched

    return source


# ═══════════════════════════════════════════════════════════════════════════════
# Conditional edge routers
# ═══════════════════════════════════════════════════════════════════════════════

def route_after_plan(state: HardeningState) -> Literal["fix", "end"]:
    status = state.get("status", "planning")
    if status == "done":
        return "end"
    return "fix"


def route_after_decide(state: HardeningState) -> Literal["plan", "fix", "end"]:
    """
    - awaiting   → end (step-by-step: plugin waits for dev Keep/Reject/Skip)
    - planning   → plan (next issue)
    - fixing     → fix (retry same issue, different approach)
    - done/stuck → end
    """
    status = state.get("status", "planning")
    if status in ("awaiting", "done", "stuck", "escalate"):
        return "end"
    if status == "fixing":
        return "fix"
    return "plan"


# ═══════════════════════════════════════════════════════════════════════════════
# Graph builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_hardening_graph() -> Any:
    """
    Build and compile the cyclic HardeningGraph.

    Topology:
      analyze → plan ──[done]──► END
                 │
                 ▼
                fix → verify → decide ──[awaiting/done]──► END
                                  │
                                  ├──[planning]──► plan   (loop ✓)
                                  └──[fixing]───► fix    (retry ✓)
    """
    graph = StateGraph(HardeningState)

    graph.add_node("analyze", node_analyze)
    graph.add_node("plan",    node_plan)
    graph.add_node("fix",     node_fix)
    graph.add_node("verify",  node_verify)
    graph.add_node("decide",  node_decide)

    # Fixed edges
    graph.add_edge("analyze", "plan")
    graph.add_edge("fix",     "verify")
    graph.add_edge("verify",  "decide")

    # Conditional edges (the loops live here)
    graph.add_conditional_edges("plan",   route_after_plan,   {"fix": "fix", "end": END})
    graph.add_conditional_edges("decide", route_after_decide, {
        "plan": "plan",
        "fix":  "fix",
        "end":  END,
    })

    graph.set_entry_point("analyze")

    return graph.compile()


# Singleton — compiled once, reused across requests (same as watch_graph pattern)
hardening_graph = build_hardening_graph()


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

async def ainvoke_hardening(
    file_path:      str,
    project_path:   str,
    language:       str       = "python",
    goal_score:     int       = 90,
    max_iterations: int       = MAX_ITERATIONS,
    mode:           str       = "step_by_step",
    broadcast_cb:   Any       = None,
) -> HardeningState:
    """
    Launch one hardening pass on a file.

    In step_by_step mode the graph stops at status='awaiting' after each
    accepted fix (waiting for dev Keep/Reject/Skip from the plugin).
    The caller resumes by calling ainvoke_continue().

    Returns the final HardeningState.
    """
    from services.test_runner import TestRunner

    fp = Path(file_path)
    if not fp.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    source_code = fp.read_text(encoding="utf-8", errors="replace")

    initial_state: HardeningState = {
        "file_path":      file_path,
        "project_path":   project_path,
        "language":       language,
        "goal_score":     goal_score,
        "max_iterations": max_iterations,
        "mode":           mode,
        "iteration":      0,
        "source_code":    source_code,
        "issues":         [],
        "score":          0,
        "score_initial":  0,
        # Architecture pipeline fields (populated by node_analyze)
        "neighborhood":   {},
        "rag_docs":       [],
        "rag_scores":     [],
        "patterns":       [],
        "context":        {},
        # Per-iteration working vars
        "current_issue":  None,
        "current_fix":    None,
        "verify_static":  None,
        "verify_tests":   None,
        "attempts":       [],
        "staged_patches": [],
        "status":         "planning",
        "skip_reason":    None,
        "done_reason":    "",
        "next_issue":     None,
        "next_fix":       None,
        "_broadcast":     broadcast_cb,
        "_runner":        TestRunner(Path(project_path)) if project_path else None,
    }

    result = await hardening_graph.ainvoke(initial_state)
    return result


async def ainvoke_continue(
    state:    HardeningState,
    decision: Literal["keep", "reject", "skip"],
) -> HardeningState:
    """
    Resume a paused step-by-step session after dev makes a decision.

    keep   → current patch stays, move to next issue
    reject → discard patch, record attempt, retry with different approach
    skip   → skip this issue entirely (mark as skipped in attempts)
    """
    staged   = list(state.get("staged_patches", []))
    attempts = list(state.get("attempts", []))
    issue    = state.get("current_issue", {})
    issue_id = _issue_id(issue) if issue else ""
    fix      = state.get("current_fix") or {}

    if decision == "keep":
        # Already staged in node_decide — just continue to plan
        next_status = "planning"

    elif decision == "reject":
        # Un-stage the last patch, record failed attempt
        if staged and staged[-1].get("issue_id") == issue_id:
            staged = staged[:-1]
        # Revert in-memory source to before the patch
        original = fix.get("current_code", "")
        if original and original in state.get("source_code", ""):
            state["source_code"] = state["source_code"].replace(
                fix.get("fixed_code", ""), original, 1
            )
        attempts.append({
            "issue_id":    issue_id,
            "approach":    fix.get("explanation", ""),
            "patch":       fix.get("fixed_code", ""),
            "verdict":     "rejected_by_dev",
            "fail_reason": "developer rejected",
        })
        retries = sum(1 for a in attempts if a.get("issue_id") == issue_id)
        next_status = "fixing" if retries < MAX_RETRIES_PER_ISSUE else "planning"

    else:  # skip
        if staged and staged[-1].get("issue_id") == issue_id:
            staged = staged[:-1]
        attempts.append({
            "issue_id":    issue_id,
            "approach":    "skipped by developer",
            "patch":       "",
            "verdict":     "skipped",
            "fail_reason": "developer skipped",
        })
        next_status = "planning"

    updated = {**state, "staged_patches": staged, "attempts": attempts, "status": next_status}
    result  = await hardening_graph.ainvoke(updated)
    return result
