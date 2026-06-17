"""
watch_graph.py — LangGraph WatchGraph (Orchestrator).

This is the ONLY LangGraph component. It connects the 4 LangChain agents
(CodeAgent, RetrieverAgent, AnalysisAgent, LearningAgent) in a StateGraph
with conditional edges.

Replaces: core/orchestrator.py Orchestrator._analyze_file() (12 steps)

Graph topology:
  hash_check → read_file → change_filter → parse_ast → index_chromadb
  → update_kg → update_dep_graph → get_neighborhood → rag_retrieve
  → build_context → llm_analyze → cache_results → learn_feedback
  → [conditional] analyze_dependents → END

Conditional edges:
  - hash_check:     unchanged → END (skip)
  - change_filter:  minor     → END (skip)
  - learn_feedback: has_deps  → analyze_dependents
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Literal

from langgraph.graph import END, StateGraph

from langchain_agents.graphs.state import WatchState

logger = logging.getLogger(__name__)

# ANSI colors for console output
_R  = "\033[0m"
_B  = "\033[1m"
_GR = "\033[92m"
_YL = "\033[93m"
_CY = "\033[96m"
_DM = "\033[2m"


# ═══════════════════════════════════════════════════════════════════════════════
# Node Functions — each takes WatchState, returns partial WatchState updates
# ═══════════════════════════════════════════════════════════════════════════════


def node_hash_check(state: WatchState) -> Dict[str, Any]:
    """Node 1: Check if file content has changed (hash comparison)."""
    from langchain_agents.agents.lc_code_agent import lc_code_agent

    file_path = state["file_path"]
    fp = Path(file_path)

    if not fp.exists():
        return {"skip_reason": f"File not found: {fp.name}"}

    try:
        content = fp.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"skip_reason": f"Read error: {e}"}

    if not lc_code_agent.has_content_changed(file_path, content):
        return {"skip_reason": "unchanged (same hash)"}

    return {
        "code": content,
        "content_hash": lc_code_agent.compute_hash(content),
        "skip_reason": None,
    }


def node_read_file(state: WatchState) -> Dict[str, Any]:
    """Node 2: Read file and detect language."""
    from langchain_agents.agents.lc_code_agent import lc_code_agent

    file_path = state["file_path"]
    language = lc_code_agent.detect_language(Path(file_path))

    from config import config
    if language not in config.analysis.supported_languages:
        return {"skip_reason": f"Unsupported language: {language}"}

    print(f"  {_CY}→ {Path(file_path).name}{_R} ({language})")
    return {"language": language, "skip_reason": None}


def node_change_filter(state: WatchState) -> Dict[str, Any]:
    """Node 3: Analyze change significance (CodeAgent planning)."""
    from langchain_agents.agents.lc_code_agent import lc_code_agent

    file_path = state["file_path"]
    code = state["code"]

    change_info = lc_code_agent.analyze_change(file_path, code)

    if not lc_code_agent.should_analyze(change_info):
        reason = change_info.get("reason", "minor change")
        print(f"    {_DM}Skip: {reason}{_R}")
        return {"change_info": change_info, "skip_reason": f"minor: {reason}"}

    score = change_info.get("score", 0)
    change_type = change_info.get("change_type", "unknown")
    print(f"    {_GR}Significant{_R} (score={score}, type={change_type})")
    return {"change_info": change_info, "skip_reason": None}


def node_parse_ast(state: WatchState) -> Dict[str, Any]:
    """Node 4: AST parsing (CodeAgent tool)."""
    from langchain_agents.agents.lc_code_agent import lc_code_agent

    parsed = lc_code_agent.parse(Path(state["file_path"]))
    entities = parsed.get("entities", [])
    print(f"    Parsed: {len(entities)} entities")
    return {"parsed": parsed}


def node_index_chromadb(state: WatchState) -> Dict[str, Any]:
    """Node 5: Index file into ChromaDB for project code retrieval."""
    indexer = state.get("_project_indexer")
    if not indexer:
        return {}

    try:
        indexer.index_file(
            file_path=Path(state["file_path"]),
            content=state["code"],
            entities=state.get("parsed", {}).get("entities", []),
        )
    except Exception as e:
        logger.debug("ChromaDB indexing failed: %s", e)
    return {}


def node_update_kg(state: WatchState) -> Dict[str, Any]:
    """Node 6: Incremental Knowledge Graph update."""
    try:
        from services.knowledge_graph import knowledge_graph
        knowledge_graph.update_file(
            file_path=Path(state["file_path"]),
            project_indexer=state.get("_project_indexer"),
            llm=None,
        )
    except Exception as e:
        logger.debug("KG update failed: %s", e)
    return {}


def node_update_dep_graph(state: WatchState) -> Dict[str, Any]:
    """Node 7: Update dependency graph with parsed entities."""
    extractor = state.get("_extractor")
    if not extractor:
        return {"dependents_to_analyze": []}

    try:
        parsed = state.get("parsed", {})
        extractor.update_file(
            file_path=Path(state["file_path"]),
            entities=parsed.get("entities", []),
            imports=parsed.get("imports", []),
        )
        # Find dependents (files that import this one)
        dependents = extractor.get_dependents(Path(state["file_path"]))
        return {"dependents_to_analyze": [str(d) for d in dependents]}
    except Exception as e:
        logger.debug("Dep graph update failed: %s", e)
        return {"dependents_to_analyze": []}


def node_test_gap_detect(state: WatchState) -> Dict[str, Any]:
    """Node 8: Detect missing tests for the current source file (0 token).

    Uses only langchain_agents/ agents:
      - LCTestGapAgent          : detects the gap, returns serializable dict
      - LCTestProposalNotifier  : rich display (wraps legacy notifier internally)
    """
    from langchain_agents.agents.lc_test_gap_agent import LCTestGapAgent
    from langchain_agents.agents.lc_test_proposal_notifier import LCTestProposalNotifier

    file_path = state["file_path"]
    project_path = state.get("project_path", ".")
    language = state.get("language", "")
    parsed = state.get("parsed", {})
    change_info = state.get("change_info", {})
    # ── 1. Skip test files ────────────────────────────────────────────────────
    agent = LCTestGapAgent(Path(project_path))
    if agent.is_test_file(file_path, language):
        return {"test_gap": None}

    # ── 2. Detect gap → serializable dict (LCTestGapAgent) ───────────────────
    try:
        gap = agent.check(
            source_file=file_path,
            parsed_entities=parsed.get("entities", []),
            change_info=change_info,
            language=language,
        )
    except Exception as e:
        logger.warning("LCTestGapAgent.check failed for %s: %s", Path(file_path).name, e)
        return {"test_gap": None}

    # ── 3. Notify via LCTestProposalNotifier (rich display, all levels) ───────
    if gap:
        import threading
        print_lock = state.get("_print_lock")
        notifier = LCTestProposalNotifier(print_lock=print_lock or threading.Lock())
        notifier.notify(gap)

    return {"test_gap": gap}


def node_get_neighborhood(state: WatchState) -> Dict[str, Any]:
    """Node 9: Extract graph neighborhood (RetrieverAgent tool)."""
    from langchain_agents.agents.lc_retriever_agent import lc_retriever_agent

    neighborhood = lc_retriever_agent.get_neighborhood(state["file_path"])
    criticality = neighborhood.get("criticality", 0)
    if criticality > 0:
        print(f"    Criticality: {criticality} dependents")
    return {"neighborhood": neighborhood}


def node_rag_retrieve(state: WatchState) -> Dict[str, Any]:
    """Node 9: System-aware RAG retrieval (RetrieverAgent 2-pass pipeline)."""
    from langchain_agents.agents.lc_retriever_agent import lc_retriever_agent

    result = lc_retriever_agent.retrieve_with_context(
        code=state["code"],
        file_path=state["file_path"],
        language=state["language"],
        neighborhood=state.get("neighborhood"),
    )

    docs = result.get("docs", [])
    scores = result.get("scores", [])
    print(f"    RAG: {len(docs)} documents retrieved")

    # Serialize docs for state (Documents → dicts)
    serialized = [
        {"content": d.page_content, "metadata": d.metadata}
        for d in docs
    ]

    # Detect KG patterns
    patterns = lc_retriever_agent.detect_patterns(state["code"], state["language"])
    if patterns:
        print(f"    KG patterns: {', '.join(patterns[:3])}")

    return {"rag_docs": serialized, "rag_scores": scores, "patterns": patterns}


def node_git_session(state: WatchState) -> Dict[str, Any]:
    """Node 9b: Git session monitoring (reads from GitSessionTracker, 0 LLM token)."""
    from langchain_agents.agents.lc_git_session_agent import LCGitSessionAgent

    project_path = state.get("project_path", ".")
    agent = LCGitSessionAgent()

    git_session = agent.get_status(project_path)

    if not git_session.get("success"):
        logger.debug("git_session unavailable: %s", git_session.get("error"))
        return {"git_session": None}

    # Normalise: add has_data flag used by node_emit_ws_events
    git_session["has_data"] = True
    git_session["files_at_risk_count"] = len(git_session.get("files_at_risk", []))

    level = git_session.get("level", "CLEAN")
    score = git_session.get("score", 0)
    if level in ("WARN", "CRITICAL"):
        print(f"    🔥 Git session: {level} (score={score}, "
              f"critical={git_session.get('total_critical', 0)})")
    else:
        print(f"    📊 Git session: {level} (score={score})")

    return {"git_session": git_session}


def node_build_context(state: WatchState) -> Dict[str, Any]:
    """Node 10: Build enriched context for LLM prompt."""
    from langchain_agents.agents.lc_analysis_agent import lc_analysis_agent

    file_path = state["file_path"]
    neighborhood = state.get("neighborhood", {})
    change_info = state.get("change_info", {})

    # Build base context
    context = {
        "file_path": file_path,
        "language": state.get("language", "unknown"),
        "criticality_score": neighborhood.get("criticality", 0),
        "dependencies": neighborhood.get("successors", []),
        "dependents": neighborhood.get("predecessors", []),
        "is_entry_point": neighborhood.get("is_entry_point", False),
        "change_type": change_info.get("change_type", "unknown"),
        "lines_changed": change_info.get("lines_changed", 0),
    }

    # Add system impact section if available
    from langchain_agents.agents.lc_analysis_agent import build_system_impact_section
    try:
        context["system_impact_section"] = build_system_impact_section(
            Path(file_path).name, neighborhood
        )
    except Exception:
        context["system_impact_section"] = ""

    # Inject test gap info into LLM context
    test_gap = state.get("test_gap")
    if test_gap:
        context["test_gap_warning"] = (
            f"⚠️ TESTS MANQUANTS : {test_gap['reason']} "
            f"(impact={test_gap['impact_score']}, "
            f"couverture={test_gap.get('coverage_ratio', 0):.0%})"
        )
        context["test_gap_untested"] = test_gap.get("untested_entities", [])
        context["test_gap_framework"] = test_gap.get("framework", "inconnu")

    # Inject git session info into LLM context
    git_session = state.get("git_session")
    if git_session and git_session.get("has_data"):
        level = git_session.get("level", "CLEAN")
        if level in ("WARN", "CRITICAL"):
            context["git_session_alert"] = (
                f"⚠️ SESSION GIT {level} — "
                f"score={git_session.get('score', 0)}, "
                f"{git_session.get('minutes_since_commit', 0)}min depuis dernier commit, "
                f"{git_session.get('files_at_risk_count', 0)} fichiers à risque"
            )

    # Planning: post-solution mode check
    context = lc_analysis_agent.enrich_context_post_solution(context, file_path)

    return {"context": context}


def node_llm_analyze(state: WatchState) -> Dict[str, Any]:
    """Node 11: LLM analysis (AnalysisAgent — core reasoning)."""
    from langchain_agents.agents.lc_analysis_agent import lc_analysis_agent

    file_path = state["file_path"]
    content_hash = state.get("content_hash", "")

    # Memory: check cache first
    cached = lc_analysis_agent.get_cached_analysis(file_path, content_hash)
    if cached:
        print(f"    {_GR}Cache hit{_R} — skipping LLM")
        return {"analysis": cached}

    print(f"    {_YL}LLM analyzing...{_R}")
    t0 = time.time()

    result = lc_analysis_agent.analyze(
        code=state["code"],
        context=state["context"],
        docs=state.get("rag_docs", []),
        scores=state.get("rag_scores", []),
    )

    elapsed = time.time() - t0
    print(f"    {_GR}Analysis complete{_R} ({elapsed:.1f}s)")

    # Extract strategy from analysis text
    strategy = "block_fix"
    analysis_text = result.get("analysis", "")
    if "STRATEGY: full_class" in analysis_text:
        strategy = "full_class"
    elif "STRATEGY: targeted_methods" in analysis_text:
        strategy = "targeted_methods"

    return {"analysis": result, "strategy": strategy}


def node_cache_results(state: WatchState) -> Dict[str, Any]:
    """Node 12: Cache analysis results + display output with rich console rendering."""
    from langchain_agents.agents.lc_analysis_agent import lc_analysis_agent
    from langchain_agents.agents.lc_analysis_agent import parse_llm_response
    from output.console_renderer import (
        print_results, print_solution, print_targeted_methods, parse_fix_blocks,
    )

    analysis = state.get("analysis", {})
    if analysis and state.get("content_hash"):
        lc_analysis_agent.cache_analysis(
            state["file_path"], analysis, state["content_hash"]
        )

    # Update legacy cache too
    cache = state.get("_cache")
    if cache and analysis:
        cache.update_file_cache(
            Path(state["file_path"]), analysis,
            state.get("neighborhood", {}).get("successors", []),
            state.get("neighborhood", {}).get("predecessors", []),
        )

    # Mark post-solution if full_class was used
    strategy = state.get("strategy", "block_fix")
    if strategy == "full_class":
        lc_analysis_agent.mark_post_solution(state["file_path"])

    # Update file counter stats
    counter = state.get("_file_counter")
    if counter:
        counter["analyzed"] = counter.get("analyzed", 0) + 1
        ct = state.get("change_info", {}).get("change_type", "unknown")
        counter.setdefault("by_type", {})[ct] = counter.get("by_type", {}).get(ct, 0) + 1

    # ── Rich console display ─────────────────────────────────────────────────
    if not analysis:
        return {}

    result_text = analysis.get("analysis", "")
    if not result_text and isinstance(analysis, str):
        result_text = analysis

    file_name = Path(state["file_path"]).name
    language = state.get("language", "?")
    elapsed = time.time() - state.get("stats", {}).get("start_time", time.time())
    preds = state.get("neighborhood", {}).get("predecessors", [])
    change_score = state.get("change_info", {}).get("score", 0)
    analyzed_count = counter.get("analyzed", 0) if counter else 0

    print_lock = state.get("_print_lock")

    try:
        if print_lock:
            print_lock.acquire()

        parsed_resp = parse_llm_response(result_text)

        # Rich console output only in interactive terminal (CLI/dev mode).
        # In API server mode (non-tty stdout) use structured logger to avoid
        # flooding the server console with ANSI code dumps.
        import sys as _sys
        if _sys.stdout.isatty():
            if parsed_resp["strategy"] == "full_class":
                print(f"  Strategy : {_CY}full_class{_R} — {parsed_resp['reason'][:80]}")
                print_solution(
                    solution_text=result_text, file_name=file_name,
                    changes=parsed_resp["payload"].get("changes", []),
                    language=language, elapsed=elapsed,
                    analyzed_count=analyzed_count, score=change_score,
                    impacted=preds,
                )
            elif parsed_resp["strategy"] == "targeted_methods":
                methods = parsed_resp["payload"].get("methods", [])
                print(f"  Strategy : {_CY}targeted_methods{_R} ({len(methods)} méthode(s))")
                print_targeted_methods(
                    methods=methods, file_name=file_name,
                    remaining=parsed_resp["payload"].get("remaining_blocks", []),
                    elapsed=elapsed, analyzed_count=analyzed_count,
                    score=change_score, impacted=preds,
                )
            else:
                if parsed_resp["reason"]:
                    print(f"  Strategy : {_CY}block_fix{_R} — {parsed_resp['reason'][:80]}")
                print_results(
                    text=result_text, file_name=file_name,
                    context=state.get("context", {}), elapsed=elapsed,
                    analyzed_count=analyzed_count, score=change_score,
                    impacted=preds,
                )
        else:
            logger.info(
                "[CACHE] file=%s strategy=%s elapsed=%.1fs score=%d",
                file_name,
                parsed_resp.get("strategy", strategy),
                elapsed,
                change_score,
            )
    finally:
        if print_lock:
            try:
                print_lock.release()
            except RuntimeError:
                pass

    return {}


def node_learn_feedback(state: WatchState) -> Dict[str, Any]:
    """Node 13: Learning feedback (LearningAgent — self-improvement + recurring patterns)."""
    from langchain_agents.agents.lc_learning_agent import lc_learning_agent
    from output.console_renderer import parse_fix_blocks

    analysis = state.get("analysis", {})
    language = state.get("language", "unknown")

    # ── 1. LangChain learning agent (simple path) ────────────────────────────
    result = lc_learning_agent.process_feedback(analysis, language)
    pending = result.get("pending_rules", [])
    if pending:
        names = ", ".join(r["pattern"] for r in pending)
        print(f"    {_GR}KB rules staged for approval: {names}{_R}")

        # Notify the developer so they can Accept/Reject. Rules no longer enter
        # the KB silently — the plugin shows an approval prompt per suggestion.
        ws_broadcast = state.get("_ws_broadcast")
        if callable(ws_broadcast):
            for r in pending:
                try:
                    ws_broadcast({
                        "type":      "kb_suggestion",
                        "pattern":   r["pattern"],
                        "language":  r["language"],
                        "severity":  r.get("severity", "MEDIUM"),
                        "count":     r.get("count", 0),
                        "rule_md":   r.get("rule_md", ""),
                        "file_path": state.get("file_path", ""),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    })
                except Exception as _e:
                    logger.debug("kb_suggestion broadcast failed: %s", _e)

    # ── 2. Legacy LearningAgent (enriched with project context) ──────────────
    learning_agent = state.get("_learning_agent")
    if learning_agent:
        analysis_text = analysis.get("analysis", "") if isinstance(analysis, dict) else str(analysis)
        fix_blocks = parse_fix_blocks(analysis_text)
        if fix_blocks:
            try:
                learning_agent.collect_feedback(
                    blocks=fix_blocks,
                    code_before=state.get("code", ""),
                    language=language,
                    file_name=Path(state["file_path"]).name,
                    project_indexer=state.get("_project_indexer"),
                    dependency_graph=state.get("_dep_graph"),
                )
            except Exception as e:
                logger.debug("Legacy learning feedback: %s", e)

    # ── 3. Display recurring patterns + broadcast to plugin ──────────────────
    try:
        from langchain_agents.memory.redis_memory import PatternMemory
        pm = PatternMemory()
        top = pm.get_top_patterns(language, n=3)
        recurring = [p for p in top if p["count"] >= 3]
        if recurring:
            print(f"\n    {_YL}⚠  Patterns récurrents ({language}) :{_R}")
            for p in recurring:
                print(f"      {_YL}• {p['pattern']} — vu {p['count']}× → vérifiez la KB{_R}")
            # Broadcast kb_suggestion for patterns not already staged in step 1
            already_staged = {r["pattern"] for r in pending}
            ws_broadcast_r = state.get("_ws_broadcast")
            if callable(ws_broadcast_r):
                for p in recurring:
                    if p["pattern"] in already_staged:
                        continue  # already broadcast above
                    try:
                        ws_broadcast_r({
                            "type":      "kb_suggestion",
                            "pattern":   p["pattern"],
                            "language":  language,
                            "severity":  p.get("severity", "MEDIUM"),
                            "count":     p.get("count", 0),
                            "rule_md":   p.get("fix", ""),
                            "file_path": state.get("file_path", ""),
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        })
                    except Exception as _e:
                        logger.debug("kb_suggestion (recurring) broadcast failed: %s", _e)
    except Exception:
        pass

    return {"learning_result": result}


def _is_test_path(path: str) -> bool:
    """True if `path` is a test file (Python / JS-TS / Java conventions).

    Test files are written to verify code, not application code to audit; they must
    not enter the watch pipeline as primary files or as dependents.
    """
    from pathlib import Path as _P
    p = _P(str(path))
    name = p.name.lower()
    parts = {x.lower() for x in p.parts}
    if parts & {"tests", "test", "__tests__", "__mocks__"}:
        return True
    if "test" in parts and "src" in parts:  # Java src/test/java
        return True
    if name.endswith(".py"):
        return name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"
    if name.endswith((".js", ".jsx", ".ts", ".tsx")):
        stem = name.rsplit(".", 1)[0]
        return stem.endswith(".test") or stem.endswith(".spec")
    if name.endswith(".java"):
        base = name[:-5]
        return base.endswith("test") or base.endswith("tests") or base.endswith("it")
    return False


def node_analyze_dependents(state: WatchState) -> Dict[str, Any]:
    """Node 14: Analyze impacted dependent files with LLM (upstream compatibility check)."""
    dependents = state.get("dependents_to_analyze", [])

    # Fallback: DependencyExtractor only re-adds Python relative imports during incremental
    # updates (resolve_import). For Java/TS, update_graph adds no edges, so
    # dependents_to_analyze is always empty. Use neighborhood.predecessors from the
    # pre-built initial project graph instead — it correctly resolved all languages.
    if not dependents:
        dependents = state.get("neighborhood", {}).get("predecessors", [])
        if dependents:
            logger.debug(
                "analyze_dependents: dependents_to_analyze empty, "
                "falling back to %d neighborhood predecessors",
                len(dependents),
            )

    # Ground dependents in the ACTUAL watched project. The dependency graph can
    # still contain entries from the backend repo or other indexed projects, which
    # previously caused bogus dependents (config.py/main.py) to be analyzed and
    # their fixes mis-applied. Keep only files that exist on disk AND live under
    # project_path, excluding the changed file itself. This mirrors the filter in
    # node_emit_ws_events so the analysis path and the WS event agree.
    import os as _os
    _proj = state.get("project_path", "") or ""
    if dependents and _proj:
        _proj_abs = _os.path.normcase(_os.path.abspath(_proj))
        _self_abs = _os.path.normcase(_os.path.abspath(state.get("file_path", "")))
        _grounded: list = []
        _seen: set = set()
        for _d in dependents:
            try:
                _da = _os.path.normcase(_os.path.abspath(str(_d)))
            except Exception:
                continue
            if _da in _seen or _da == _self_abs:
                continue
            if not _os.path.exists(_d) or not _da.startswith(_proj_abs):
                continue
            if _is_test_path(_d):  # never audit generated test files as dependents
                continue
            _seen.add(_da)
            _grounded.append(str(_d))
        if len(_grounded) != len(dependents):
            logger.info(
                "analyze_dependents: grounded %d → %d dependents inside project",
                len(dependents), len(_grounded),
            )
        dependents = _grounded

    if not dependents:
        return {}

    # Throttle: skip LLM analysis of dependents if the main analysis already
    # took too long. This prevents cascading LLM calls from stalling the watch
    # loop (root cause of the 481s latency). The dependency_impact WS event is
    # still emitted by node_emit_ws_events regardless.
    elapsed_so_far = time.time() - state.get("stats", {}).get("start_time", time.time())
    if elapsed_so_far > 45.0:
        logger.info(
            "analyze_dependents SKIPPED (elapsed=%.0fs > 45s threshold). "
            "File: %s. Would have analyzed: %s",
            elapsed_so_far,
            Path(state.get("file_path", "?")).name,
            [Path(d).name for d in dependents[:3]],
        )
        return {}

    # Skip if post-solution mode (avoid cascading rewrites)
    if state.get("post_solution_mode"):
        return {}

    from config import config
    from langchain_agents.agents.lc_analysis_agent import build_context
    from langchain_agents.agents.lc_code_agent import lc_code_agent
    from output.console_renderer import print_results, parse_fix_blocks

    max_deps = config.watcher.max_impacted_files
    to_analyze = dependents[:max_deps]
    changed_file = Path(state["file_path"]).name
    _raw_analysis = state.get("analysis", {}).get("analysis", "")
    # Strip code blocks from the analysis summary injected into dependent context.
    # The raw LLM text contains fix blocks with before/after code from the CHANGED
    # file. Without stripping, the dependent LLM sees that code in its context and
    # copies it into its own current_code anchor — causing "original code not found"
    # because that code belongs to a different file.
    import re as _re_dep
    _analysis_no_code = _re_dep.sub(r'(?s)```[^\n]*\n.*?```', '[code omitted]', _raw_analysis)
    analysis_text = _analysis_no_code[:1200]
    print_lock = state.get("_print_lock")
    ws_broadcast = state.get("_ws_broadcast")
    loop_start = state.get("stats", {}).get("start_time", time.time())
    # Hard per-cascade budget. The primary result has already been broadcast, so
    # dependents are best-effort: if the cascade runs long (rate-limit backoff,
    # cold reranker), stop rather than stall the watch loop for minutes.
    DEPENDENT_BUDGET_S = 60.0
    analyzed = 0

    for dep_str in to_analyze:
        # Re-check budget EVERY iteration (the old single pre-loop check let a
        # fast primary unlock an unbounded dependent cascade → 297s latency).
        if time.time() - loop_start > DEPENDENT_BUDGET_S:
            logger.info(
                "analyze_dependents: budget %.0fs exhausted, stopping after %d dep(s)",
                DEPENDENT_BUDGET_S, analyzed,
            )
            break

        dep_path = Path(dep_str)
        if not dep_path.exists():
            logger.warning("Dependent not found on disk, skipping: %s", dep_str)
            continue

        try:
            dep_content = dep_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning("Could not read dependent %s: %s", dep_path.name, e)
            continue

        print(f"\n{'─'*70}")
        print(f" 🔗 Dépendant : {dep_path.name}  ← {changed_file} vient de changer")

        # Parse dependent
        parsed_dep = lc_code_agent.parse(dep_path)
        if isinstance(parsed_dep, str):
            parsed_dep = {"entities": [], "imports": []}
        if parsed_dep.get("error"):
            continue

        # Neighborhood + RAG for dependent
        from langchain_agents.agents.lc_retriever_agent import lc_retriever_agent
        neighborhood_dep = lc_retriever_agent.get_neighborhood(str(dep_path))
        if isinstance(neighborhood_dep, str):
            neighborhood_dep = {}
        dep_lang = lc_code_agent.detect_language(dep_path)
        neighborhood_dep["language"] = dep_lang
        neighborhood_dep["_parsed_entities"] = parsed_dep.get("entities", [])

        result_dep = lc_retriever_agent.retrieve_with_context(
            code=dep_content, file_path=str(dep_path),
            language=dep_lang, neighborhood=neighborhood_dep,
        )
        docs_dep = result_dep.get("docs", [])
        scores_dep = result_dep.get("scores", [])

        # Build context with upstream change info
        context_dep = build_context(
            file_path=dep_path, neighborhood=neighborhood_dep,
            project_indexer=state.get("_project_indexer"),
        )
        if isinstance(context_dep, str):
            context_dep = {}

        context_dep["upstream_change"] = (
            f"IMPORTANT: {changed_file} was just refactored. "
            f"Verify that {dep_path.name} still compiles and works correctly with it.\n"
            f"=== Summary of changes in {changed_file} "
            f"(read-only context — DO NOT copy this code into your fix blocks) ===\n"
            f"{analysis_text[:800]}\n"
            f"=== End of {changed_file} context ===\n\n"
            f"CRITICAL: You are analyzing {dep_path.name} ONLY. "
            f"In every fix block, 'current_code' MUST be a snippet that exists "
            f"verbatim in {dep_path.name}. "
            f"NEVER use code from {changed_file} as current_code."
        )
        context_dep["post_solution_mode"] = True
        context_dep["post_solution_hint"] = (
            f"{changed_file} was refactored. Check for: broken imports, "
            f"wrong method calls, signature mismatches, compilation errors. "
            f"Use block_fix only. If compatible, say so with 0 fix blocks."
        )
        print(f"  {_DM}↩  Mode compatibilité — vérifie l'impact de {changed_file}{_R}")

        # LLM analysis
        try:
            from langchain_agents.tools.analysis_tools import tool_llm_analyze
            serialized_docs = [
                {"content": d.page_content, "metadata": d.metadata} for d in docs_dep
            ]
            analysis_dep = tool_llm_analyze.invoke({
                "code": dep_content, "context": context_dep,
                "docs": serialized_docs, "scores": scores_dep,
            })
        except Exception as e:
            logger.error("Analyse dépendant %s : %s", dep_path.name, e)
            continue

        if isinstance(analysis_dep, str):
            analysis_dep = {"analysis": analysis_dep, "relevant_knowledge": []}

        # Cache
        cache = state.get("_cache")
        if cache:
            cache.update_file_cache(dep_path, analysis_dep, [], [])

        # Display with rich renderer
        result_text_dep = analysis_dep.get("analysis", "")
        _do_print = print_lock.acquire if print_lock else lambda: True
        try:
            if print_lock:
                print_lock.acquire()
            print_results(
                text=result_text_dep, file_name=dep_path.name,
                context=context_dep, elapsed=0.0,
                analyzed_count=0, score=0, impacted=[],
            )
        finally:
            if print_lock:
                try:
                    print_lock.release()
                except RuntimeError:
                    pass

        # ── Stream this dependent's result to the plugin immediately ─────────
        # Same builder as the primary → dependents get identical fix-anchoring /
        # apply safety. Broadcast now so each dependent appears as it finishes
        # instead of waiting for the whole cascade.
        if callable(ws_broadcast):
            try:
                dep_built = build_analysis_result_event(
                    file_path=str(dep_path),
                    language=dep_lang,
                    raw_text=result_text_dep,
                    source_code=dep_content,
                    strategy_default="block_fix",
                    change_score=0,
                    rag_docs_used=len(docs_dep),
                    elapsed_ms=0,
                )
                dep_event = dep_built["event"]
                # Tag as an upstream-impact result so the plugin can group it
                # under the file the developer actually edited.
                dep_event["triggered_by"] = state["file_path"]
                dep_event["is_dependent"] = True
                ws_broadcast(dep_event)
            except Exception as exc:
                logger.warning("[WS] dependent broadcast failed for %s: %s",
                               dep_path.name, exc)

        # Self-Improving RAG for dependent
        fix_blocks_dep = parse_fix_blocks(result_text_dep)
        learning_agent = state.get("_learning_agent")
        if fix_blocks_dep and learning_agent:
            try:
                learning_agent.collect_feedback(
                    blocks=fix_blocks_dep, code_before=dep_content,
                    language=dep_lang, file_name=dep_path.name,
                )
            except Exception as e:
                logger.debug("Feedback dépendant %s : %s", dep_path.name, e)

        analyzed += 1

    if analyzed > 0:
        print(f"\n  ✓ {analyzed} dépendant(s) analysé(s) suite au changement de {changed_file}\n")

    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# Node 15 — WebSocket Event Builder (VS Code plugin output)
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_issues_from_llm(text: str, file_path: str) -> list:
    """
    Extract structured issues from raw LLM analysis text.

    Looks for patterns like:
      - ISSUE: <title> | line <N> | severity <S> | rule <R>
      - ## <title> (line N)
      - ⚠ / ❌ / 🔴 prefixed lines with line numbers
      - FIX BLOCK headers

    Returns list of dicts compatible with vscode.Diagnostic.
    """
    import re

    issues = []
    seen: set = set()

    # ── Severity keyword helpers ───────────────────────────────────────────────
    _SEV_MAP = {
        "critical": "critical", "error": "error",
        "warning": "warning",   "warn": "warning",
        "medium": "warning",    "info": "info",
        "low": "info",          "high": "error",
    }

    def _norm_sev(raw: str) -> str:
        return _SEV_MAP.get(raw.strip().lower(), "warning")

    # ── Pattern 1: explicit ISSUE: lines ──────────────────────────────────────
    # ISSUE: SQL Injection | line 42 | severity critical | rule security.sql_injection
    p1 = re.compile(
        r"ISSUE:\s*(?P<title>[^|\n]+)"
        r"(?:\s*\|\s*line\s*(?P<line>\d+))?"
        r"(?:\s*\|\s*severity\s*(?P<sev>\w+))?"
        r"(?:\s*\|\s*rule\s*(?P<rule>[^\s|]+))?",
        re.IGNORECASE,
    )
    for m in p1.finditer(text):
        title = m.group("title").strip()
        key = (title, m.group("line"))
        if key in seen:
            continue
        seen.add(key)
        issues.append({
            "title":    title,
            "message":  title,
            "line":     int(m.group("line")) if m.group("line") else None,
            "severity": _norm_sev(m.group("sev") or "warning"),
            "rule":     m.group("rule") or "code-auditor",
            "suggestion": "",
        })

    # ── Pattern 2: FIX BLOCK headers — «### Fix N: <title> (line N, severity)» ─
    p2 = re.compile(
        r"(?:#{1,3}|FIX\s*\d+[:.]?)\s*(?P<title>[^\n(]+)"
        r"(?:\((?:[Ll]ine\s*)?(?P<line>\d+)(?:,\s*(?P<sev>\w+))?\))?",
        re.IGNORECASE,
    )
    for m in p2.finditer(text):
        title = m.group("title").strip(" :-")
        if not title or len(title) < 5:
            continue
        key = (title, m.group("line"))
        if key in seen:
            continue
        seen.add(key)
        issues.append({
            "title":    title,
            "message":  title,
            "line":     int(m.group("line")) if m.group("line") else None,
            "severity": _norm_sev(m.group("sev") or "warning"),
            "rule":     "code-auditor",
            "suggestion": "",
        })

    # ── Pattern 3: emoji-prefixed lines with line numbers ─────────────────────
    # ❌ Line 12: Missing null check
    p3 = re.compile(
        r"(?:❌|⚠️?|🔴|🟡|CRITICAL|ERROR|WARNING|BUG)\s*"
        r"(?:[Ll]ine\s*(?P<line>\d+)\s*[:\-–]?\s*)?"
        r"(?P<title>[^\n]{5,120})",
    )
    for m in p3.finditer(text):
        title = m.group("title").strip()
        key = (title[:40], m.group("line"))
        if key in seen:
            continue
        seen.add(key)
        # Derive severity from prefix
        prefix = text[m.start(): m.start() + 10]
        sev = "error" if any(x in prefix for x in ("❌", "🔴", "CRITICAL", "ERROR")) else "warning"
        issues.append({
            "title":    title[:120],
            "message":  title[:120],
            "line":     int(m.group("line")) if m.group("line") else None,
            "severity": sev,
            "rule":     "code-auditor",
            "suggestion": "",
        })

    # ── Enrich suggestions from nearby «Suggestion:» lines ────────────────────
    sug_pattern = re.compile(r"[Ss]uggestion[:\-–]\s*(?P<sug>[^\n]{5,200})")
    suggestions = [m.group("sug").strip() for m in sug_pattern.finditer(text)]
    for i, issue in enumerate(issues):
        if i < len(suggestions):
            issue["suggestion"] = suggestions[i]

    return issues[:20]   # cap at 20 issues per file

def _extract_fixes_from_llm(
    text: str,
    file_path: str,
    language: str,
    source_code: str = "",
    strategy: str = "",
) -> list:
    """
    Extract suggested fixes from the raw LLM output for VS Code.

    Produces fixes that the plugin can apply only when both fields exist:
      - current_code
      - fixed_code

    Supports:
      1. parse_fix_blocks()
      2. diff snippets: - old / + new
      3. full_class code blocks: replace whole file
      4. targeted Java method blocks: replace one method when possible
    """
    import re
    from pathlib import Path

    fixes = []
    file_name = Path(file_path).name

    def _safe_line(value):
        try:
            return int(value) if value is not None else None
        except Exception:
            return None

    def _add_fix(
        title: str,
        current_code: str,
        fixed_code: str,
        explanation: str = "",
        line=None,
        apply_mode: str = "replace_snippet",
    ):
        current_code = str(current_code or "").strip()
        fixed_code = str(fixed_code or "").strip()

        if not current_code or not fixed_code:
            return

        if current_code == fixed_code:
            return

        # Avoid extremely tiny unsafe replacements
        if len(current_code) < 3 or len(fixed_code) < 3:
            return

        fixes.append({
            "title": str(title or "Suggested fix")[:160],
            "file_path": file_path,
            "file": file_name,
            "line": _safe_line(line),
            "current_code": current_code,
            "fixed_code": fixed_code,
            "explanation": str(explanation or "Suggested Code Auditor fix."),
            "fix_preview": _build_fix_preview(current_code, fixed_code),
            "language": language,
            "apply_mode": apply_mode,
        })

    def _build_fix_preview(current_code: str, fixed_code: str, max_chars: int = 1200) -> str:
        current_code = str(current_code or "").strip()
        fixed_code = str(fixed_code or "").strip()

        preview = f"- {current_code}\n+ {fixed_code}"

        if len(preview) > max_chars:
            preview = preview[:max_chars] + "\n... preview truncated ..."

        return preview

    def _extract_code_blocks(raw: str) -> list:
        blocks = []

        code_block_pattern = re.compile(
            r"```(?P<lang>[a-zA-Z0-9_+-]*)\s*(?P<code>.*?)```",
            re.DOTALL,
        )

        for match in code_block_pattern.finditer(raw or ""):
            block_lang = (match.group("lang") or "").strip().lower()
            code = (match.group("code") or "").strip()

            if not code:
                continue

            blocks.append({
                "language": block_lang,
                "code": code,
            })

        return blocks

    def _looks_like_full_java_class(code: str) -> bool:
        return (
            "class " in code
            and ("public class " in code or "class " in code)
            and ("{" in code and "}" in code)
        )

    def _extract_java_method_name(code: str) -> str:
        m = re.search(
            r"(?:public|private|protected)?\s*(?:static\s+)?[\w<>\[\], ?]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*\{",
            code,
        )
        return m.group(1) if m else ""

    def _find_java_method_in_source(source: str, method_name: str) -> str:
        if not source or not method_name:
            return ""

        signature = re.search(
            rf"(?:public|private|protected)?\s*(?:static\s+)?[\w<>\[\], ?]+\s+{re.escape(method_name)}\s*\([^)]*\)\s*\{{",
            source,
        )

        if not signature:
            return ""

        start = signature.start()
        brace_start = source.find("{", signature.start())

        if brace_start < 0:
            return ""

        depth = 0
        for i in range(brace_start, len(source)):
            ch = source[i]

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1

                if depth == 0:
                    return source[start:i + 1].strip()

        return ""

    # ── 1. Existing parser from console_renderer ─────────────────────────────
    try:
        from output.console_renderer import parse_fix_blocks

        blocks = parse_fix_blocks(text) or []

        for idx, block in enumerate(blocks):
            current_code = (
                block.get("current_code")
                or block.get("before")
                or block.get("old_code")
                or block.get("original")
                or ""
            )

            fixed_code = (
                block.get("fixed_code")
                or block.get("after")
                or block.get("new_code")
                or block.get("replacement")
                or ""
            )

            title = (
                block.get("title")
                or block.get("problem")
                or block.get("location")
                or f"Suggested fix {idx + 1}"
            )

            explanation = (
                block.get("why")
                or block.get("explanation")
                or block.get("problem")
                or "Suggested fix"
            )

            _add_fix(
                title=title,
                current_code=current_code,
                fixed_code=fixed_code,
                explanation=explanation,
                line=block.get("line"),
                apply_mode="replace_snippet",
            )

    except Exception as e:
        logger.debug("parse_fix_blocks failed while extracting WS fixes: %s", e)

    # ── 2. Diff-style snippets ───────────────────────────────────────────────
    diff_pattern = re.compile(
        r"(?m)^\s*-\s*(?P<old>[^\n]+)\n\s*\+\s*(?P<new>[^\n]+)"
    )

    for match in diff_pattern.finditer(text or ""):
        old = match.group("old").strip()
        new = match.group("new").strip()

        before = text[max(0, match.start() - 400):match.start()]
        line_match = re.search(r"(?:line|Line|ligne|Ligne)\s*[:#]?\s*(\d+)", before)
        line = int(line_match.group(1)) if line_match else None

        _add_fix(
            title=f"Replace code at line {line or '?'}",
            current_code=old,
            fixed_code=new,
            explanation="Suggested replacement extracted from analysis diff.",
            line=line,
            apply_mode="replace_snippet",
        )

    # ── 3. Code block fallback ───────────────────────────────────────────────
    code_blocks = _extract_code_blocks(text)

    for block in code_blocks:
        code = block["code"]

        # Full Java class: replace whole file.
        if language == "java" and _looks_like_full_java_class(code):
            if source_code.strip():
                _add_fix(
                    title=f"Replace full file {file_name}",
                    current_code=source_code,
                    fixed_code=code,
                    explanation="Full-class fix generated by Code Auditor. Review before applying.",
                    line=1,
                    apply_mode="full_file",
                )
                continue

        # Targeted Java method: replace method if we can find current method.
        if language == "java":
            method_name = _extract_java_method_name(code)

            if method_name:
                current_method = _find_java_method_in_source(source_code, method_name)

                if current_method:
                    _add_fix(
                        title=f"Replace method {method_name}",
                        current_code=current_method,
                        fixed_code=code,
                        explanation=f"Suggested replacement for method {method_name}.",
                        line=None,
                        apply_mode="replace_method",
                    )

    # ── 4. Deduplicate ───────────────────────────────────────────────────────
    deduped = []
    seen = set()

    for fix in fixes:
        key = (
            fix.get("apply_mode", ""),
            fix.get("current_code", "")[:120],
            fix.get("fixed_code", "")[:120],
        )

        if key in seen:
            continue

        seen.add(key)
        deduped.append(fix)

    logger.info(
        "[WS] fixes extracted for %s: %d",
        file_name,
        len(deduped),
    )

    return deduped[:10]


def _anchor_current_code(current: str, source: str) -> "str | None":
    """
    Anchor a fix's `current_code` to the EXACT text present in `source`.

    The LLM frequently returns `current_code` that is paraphrased, re-indented,
    or (with weak models) partially hallucinated — so an exact `current in source`
    test fails and the plugin reports "original code not found", or worse, applies
    the patch at the wrong place and corrupts the file.

    Resolution order:
      1. Exact substring        → return `current` unchanged.
      2. Whitespace-normalized  → find the contiguous block of source lines whose
         per-line stripped form equals `current`'s, and return that EXACT source
         slice (so downstream exact-match apply succeeds).
      3. Un-anchorable          → return None (caller marks the fix non-applicable).
    """
    current = (current or "").strip("\n")
    if not current or not source:
        return None

    # 1. Exact
    if current in source:
        return current

    def _norm(line: str) -> str:
        return " ".join(line.split())

    cur_lines = [l for l in current.splitlines() if l.strip()]
    if not cur_lines:
        return None
    cur_norm = [_norm(l) for l in cur_lines]

    src_lines = source.splitlines()
    src_norm = [_norm(l) for l in src_lines]
    window = len(cur_norm)

    # 2. Sliding-window match on normalized lines, ignoring blank source lines
    #    inside the window is intentionally NOT done — we need a contiguous slice
    #    so the returned text is a real substring of `source`.
    for i in range(0, len(src_lines) - window + 1):
        if src_norm[i:i + window] == cur_norm:
            return "\n".join(src_lines[i:i + window])

    # 3. Single-line fixes: tolerate surrounding context by matching the one line
    if window == 1:
        target = cur_norm[0]
        for i, sn in enumerate(src_norm):
            if sn == target:
                return src_lines[i]

    return None


def _align_fix_to_source(anchored: str, fixed: str, source: str):
    """Re-align an anchored fix to the source's indentation.

    The LLM returns ``current_code``/``fixed_code`` de-indented. When ``current``
    then matches as a mid-line / un-indented substring, substituting a multi-line
    ``fixed`` drops the indentation of every *added* line (it lands at column 0),
    producing IndentationErrors and statements merged onto one line — the
    file-corruption bug.

    This expands ``anchored`` to span whole source lines (capturing the leading
    indentation) and re-indents ``fixed`` to that same base indentation while
    preserving its own relative nesting. Returns ``(current_full, fixed_aligned)``.
    If the match is not cleanly line-aligned (real code before/after it on the
    line), the inputs are returned unchanged — a safe no-op.
    """
    import textwrap

    idx = source.find(anchored)
    if idx < 0:
        return anchored, fixed

    line_start = source.rfind("\n", 0, idx) + 1
    pre = source[line_start:idx]
    if pre.strip():
        return anchored, fixed  # real code precedes the match on its line

    end = idx + len(anchored)
    nl = source.find("\n", end)
    line_end = nl if nl != -1 else len(source)
    if source[end:line_end].strip():
        return anchored, fixed  # real code follows the match on its line

    current_full = source[line_start:line_end]
    first_line = current_full.splitlines()[0] if current_full.splitlines() else current_full
    base_indent = first_line[: len(first_line) - len(first_line.lstrip())]

    fixed_norm = textwrap.dedent(fixed).strip("\n")
    aligned = "\n".join(
        (base_indent + line) if line.strip() else ""
        for line in fixed_norm.splitlines()
    )
    return current_full, aligned


def _py_compiles(code: str) -> bool:
    """True unless `code` has a Python *syntax* error (runtime errors don't count)."""
    try:
        compile(code, "<chk>", "exec")
        return True
    except SyntaxError:
        return False
    except Exception:
        # Non-syntax issues (e.g. null bytes) — don't claim the code is broken.
        return True


def _py_undefined_names(code: str) -> set:
    """Names that are READ but never BOUND anywhere in `code` (flat, non-scoped).

    Deliberately conservative: a name bound *anywhere* in the module counts as
    defined, so we only flag names that are used yet defined nowhere — which keeps
    false positives near zero while still catching the motivating case (a fix that
    drops ``rows = cursor.fetchall()`` but keeps ``return ... rows``).
    """
    import ast
    import builtins

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()  # syntax is handled by the compile gate

    bound: set = set()
    loaded: set = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                bound.add(node.id)        # covers assignments, for/with/comprehension targets, unpacking
            elif isinstance(node.ctx, ast.Load):
                loaded.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    return set()  # star import → can't reason; never reject
                bound.add(alias.asname or alias.name)

    safe = set(dir(builtins)) | {
        "__name__", "__file__", "__doc__", "__builtins__",
        "__spec__", "__loader__", "__package__",
    }
    return {n for n in loaded if n not in bound and n not in safe}


def _fix_breaks_code(original: str, patched: str, language: str) -> "str | None":
    """Return a human reason if `patched` is provably WORSE than `original`
    (introduces a syntax error or a newly-undefined name), else None.

    Only languages we can statically validate are gated (Python today). For
    everything else we return None — we never reject a fix we cannot prove broken,
    so good fixes are never lost. This is the Phase-A trust gate: the plugin must
    never offer a one-click fix that breaks the file.
    """
    if (language or "").lower() != "python":
        return None
    if original == patched:
        return None

    orig_ok = _py_compiles(original)
    patched_ok = _py_compiles(patched)
    if orig_ok and not patched_ok:
        return "introduce a syntax error"
    if orig_ok and patched_ok:
        new_undef = _py_undefined_names(patched) - _py_undefined_names(original)
        if new_undef:
            return "leave undefined name(s): " + ", ".join(sorted(new_undef))
    return None


def _attach_fixes_to_issues(issues: list, fixes: list) -> list:
    """
    Attach fix fields to issues so the VS Code plugin can show
    Apply / Explain / Ignore without guessing.
    """
    if not issues or not fixes:
        return issues

    enriched = []

    for idx, issue in enumerate(issues):
        item = dict(issue)
        issue_line = item.get("line")
        matching_fix = None

        if issue_line is not None:
            for fix in fixes:
                if fix.get("line") == issue_line:
                    matching_fix = fix
                    break

        if matching_fix is None and idx < len(fixes):
            matching_fix = fixes[idx]

        if matching_fix:
            # Only attach replacement code when the fix is anchorable/applicable.
            # Otherwise the plugin would think a one-click fix exists and apply it
            # at the wrong place (the file-corruption bug). Advisory fixes still
            # contribute their explanation so the user gets guidance.
            if matching_fix.get("applicable"):
                item["current_code"] = matching_fix.get("current_code", "")
                item["fixed_code"] = matching_fix.get("fixed_code", "")
                item["fix_preview"] = matching_fix.get("fix_preview", "")
            item["fix_explanation"] = matching_fix.get("explanation", "")

            if not item.get("suggestion"):
                item["suggestion"] = matching_fix.get("explanation", "")

        enriched.append(item)

    return enriched


# Tokens that prove a "issue"/"fix" is actually leaked prompt scaffolding or the
# weak model echoing the output template — never a real finding. Matched
# case-insensitively against an issue title/message or a fix's code.
_PARSER_ARTIFACTS = (
    "structured_output", "structured output", "---decision",
    "**location**", "**problem**", "**severity**", "**current code**",
    "**fixed code**", "fix start", "```", "<structured", "</structured",
)

# Phrases the model uses when it found NOTHING — these must not surface as issues.
_NO_ISSUE_PHRASES = (
    "no specific issue", "no issues found", "no issue found",
    "does not contain", "no obvious issue", "appears to be clean",
    "no problems found", "n/a",
)


def _is_artifact(*texts: str) -> bool:
    """True if any text is empty/garbage or contains parser scaffolding."""
    blob = " ".join(t for t in texts if t).lower()
    if not blob.strip():
        return True
    return any(tok in blob for tok in _PARSER_ARTIFACTS)


def _sanitize_issues(issues_raw: list) -> list:
    """Drop parser artifacts and 'nothing found' non-issues.

    The weak LLM frequently echoes the output template (titles like
    ``</STRUCTURED_OUTPUT>`` or ``**LOCATION**: execute_query, line 1``) or
    reports 'No specific issues found' as if it were a finding. Surfacing these
    as actionable issues is what made the plugin's issue list look broken.
    """
    clean = []
    for issue in issues_raw or []:
        title = str(issue.get("title", "")).strip()
        message = str(issue.get("message", "")).strip()
        if _is_artifact(title, message):
            continue
        low = f"{title} {message}".lower()
        if any(p in low for p in _NO_ISSUE_PHRASES):
            continue
        clean.append(issue)
    return clean


def _sanitize_fixes(fixes_raw: list) -> list:
    """Drop fixes whose code is parser scaffolding or has no real content."""
    clean = []
    for fix in fixes_raw or []:
        current = str(fix.get("current_code", ""))
        fixed = str(fix.get("fixed_code", ""))
        if _is_artifact(current) or _is_artifact(fixed):
            continue
        clean.append(fix)
    return clean


def build_analysis_result_event(
    file_path: str,
    language: str,
    raw_text: str,
    source_code: str,
    strategy_default: str = "block_fix",
    change_score: int = 0,
    rag_docs_used: int = 0,
    elapsed_ms: int = 0,
) -> Dict[str, Any]:
    """
    Build a single schema-v2.0 `analysis_result` event from one file's LLM output.

    Shared by node_emit_ws_events (primary file) and node_analyze_dependents
    (each dependent), so both go through the SAME structured-parse + fix-anchoring
    path — guaranteeing dependents get the same apply-fix safety as the primary.
    """
    import uuid

    analyzed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    request_id  = str(uuid.uuid4())

    # ── Structured output first, regex fallback ──────────────────────────────
    try:
        from langchain_agents.agents.lc_analysis_agent import parse_structured_output
        structured = parse_structured_output(raw_text)
    except Exception:
        structured = {}

    if structured.get("issues") is not None:
        issues_raw      = structured.get("issues", [])
        fixes_raw       = structured.get("fixes", [])
        strategy        = structured.get("strategy") or strategy_default
        strategy_reason = structured.get("strategy_reason", "")
        logger.info(
            "[WS] structured output used for %s: strategy=%s issues=%d fixes=%d",
            Path(file_path).name, strategy, len(issues_raw), len(fixes_raw),
        )
    else:
        issues_raw = _parse_issues_from_llm(raw_text, file_path)
        fixes_raw  = _extract_fixes_from_llm(
            raw_text, file_path, language,
            source_code=source_code, strategy=strategy_default,
        )
        strategy        = strategy_default
        strategy_reason = ""
        logger.info(
            "[WS] regex fallback used for %s: issues=%d fixes=%d raw_len=%d",
            Path(file_path).name, len(issues_raw), len(fixes_raw), len(raw_text or ""),
        )

    # Strip leaked prompt scaffolding / "nothing found" non-issues before anything
    # downstream treats them as real, actionable findings.
    _n_issues, _n_fixes = len(issues_raw), len(fixes_raw)
    issues_raw = _sanitize_issues(issues_raw)
    fixes_raw  = _sanitize_fixes(fixes_raw)
    if len(issues_raw) != _n_issues or len(fixes_raw) != _n_fixes:
        logger.info(
            "[WS] sanitized %s: issues %d→%d, fixes %d→%d (dropped parser artifacts)",
            Path(file_path).name, _n_issues, len(issues_raw), _n_fixes, len(fixes_raw),
        )

    try:
        from api.diff_utils import compute_diff_hunks, truncate_code_for_ws
        _diff_available = True
    except ImportError:
        _diff_available = False

    enriched_fixes = []
    for fix in fixes_raw:
        current    = str(fix.get("current_code", "")).strip()
        fixed      = str(fix.get("fixed_code",   "")).strip("\n")  # keep indentation for re-alignment
        apply_mode = str(fix.get("apply_mode", "replace_snippet"))
        fix_id     = str(uuid.uuid4())

        # Anchor current_code to the EXACT file text (see _anchor_current_code).
        applicable = False
        if apply_mode != "full_file" and source_code and current and fixed:
            anchored = _anchor_current_code(current, source_code)
            if anchored is not None:
                # Re-align current/fixed to the source indentation so multi-line
                # fixes don't land at column 0 (the file-corruption bug). See
                # _align_fix_to_source.
                current, fixed = _align_fix_to_source(anchored, fixed, source_code)
                applicable = True
            else:
                logger.info(
                    "[WS] fix not anchorable in %s (current_code not found) — "
                    "marked advisory: %.60s",
                    Path(file_path).name, current.replace("\n", "\\n"),
                )

        diff_hunks = []
        patched = None
        if applicable and current != fixed:
            try:
                patched = source_code.replace(current, fixed, 1)
                if _diff_available:
                    diff_hunks = compute_diff_hunks(source_code, patched)
            except Exception as exc:
                logger.debug("diff_hunks failed: %s", exc)

        if apply_mode == "full_file" and current and fixed:
            applicable = True
            patched = fixed

        # ── Validation gate (Phase A — trust) ───────────────────────────────────
        # Never offer a one-click fix that makes the file WORSE: a new syntax error
        # or a newly-undefined name (e.g. a fix that drops `rows = cursor.fetchall()`
        # but keeps `return pd.DataFrame(rows)`). Such fixes are downgraded to
        # advisory — the dev still gets the guidance, but the plugin won't auto-apply
        # broken code. Only statically-checkable languages are gated; others pass.
        if applicable and patched is not None and patched != source_code:
            break_reason = _fix_breaks_code(source_code, patched, language)
            if break_reason:
                applicable = False
                diff_hunks = []
                logger.info(
                    "[WS] fix rejected in %s (would %s) — marked advisory",
                    Path(file_path).name, break_reason,
                )

        if _diff_available and apply_mode == "full_file" and (
            len(current) > 300 or len(fixed) > 300
        ):
            current_ws = truncate_code_for_ws(current)
            fixed_ws   = truncate_code_for_ws(fixed)
        else:
            current_ws = current
            fixed_ws   = fixed

        # Derive the TRUE 1-based line from the anchored position in source. The
        # LLM's reported line is unreliable (weak models miscount, reporting e.g.
        # line 5 for code that lives at line 16) — which threw the gutter dots off
        # and could send a line-based apply to the wrong line. Fall back to the
        # LLM line only when the fix isn't anchored (advisory / full_file).
        fix_line = fix.get("line")
        if applicable and apply_mode != "full_file" and current and source_code:
            _pos = source_code.find(current)
            if _pos >= 0:
                fix_line = source_code.count("\n", 0, _pos) + 1

        enriched_fixes.append({
            "id":           fix_id,
            "issue_id":     fix.get("issue_id"),
            "title":        str(fix.get("title", "Suggested fix"))[:160],
            "apply_mode":   apply_mode if applicable else "manual",
            "applicable":   applicable,
            "file_path":    file_path,
            "line":         fix_line,
            "diff_hunks":   diff_hunks,
            "current_code": current_ws,
            "fixed_code":   fixed_ws,
            "explanation":  str(fix.get("explanation", fix.get("why", "")))[:500],
            "language":     language,
        })

    enriched_issues = []
    for idx, issue in enumerate(issues_raw):
        matched_fix = enriched_fixes[idx] if idx < len(enriched_fixes) else None
        fix_id_ref  = matched_fix["id"] if matched_fix else None
        fix_ok      = bool(matched_fix and matched_fix.get("applicable"))
        # Prefer the fix's anchored line for the gutter / "Go to line" (the real
        # position) over the LLM's guess. Keep the LLM line only for advisory
        # issues that have no anchored fix to derive from.
        issue_line = issue.get("line")
        if matched_fix and matched_fix.get("applicable") and matched_fix.get("line"):
            issue_line = matched_fix["line"]
        enriched_issues.append({
            **issue,
            "line":          issue_line,
            "id":            str(uuid.uuid4()),
            "fix_available": fix_ok,
            "fix_id":        fix_id_ref,
        })

    enriched_issues = _attach_fixes_to_issues(enriched_issues, enriched_fixes)

    _sev_order = {"critical": 4, "error": 3, "warning": 2, "info": 1}
    top_sev    = max(
        (_sev_order.get(str(i.get("severity", "info")).lower(), 1) for i in enriched_issues),
        default=1,
    )
    sev_label  = {4: "critical", 3: "error", 2: "warning", 1: "info"}[top_sev]

    event = {
        "type":            "analysis_result",
        "schema_version":  "2.0",
        "request_id":      request_id,
        "file_path":       file_path,
        "file_name":       Path(file_path).name,
        "language":        language,
        "severity":        sev_label,
        "strategy":        strategy,
        "strategy_reason": strategy_reason,
        "change_score":    int(change_score),
        "issues":          enriched_issues,
        "fixes":           enriched_fixes,
        "elapsed_ms":      elapsed_ms,
        "analyzed_at":     analyzed_at,
        "rag_docs_used":   rag_docs_used,
    }
    return {
        "event":           event,
        "enriched_issues": enriched_issues,
        "enriched_fixes":  enriched_fixes,
        "strategy":        strategy,
    }


def node_emit_ws_events(state: WatchState) -> Dict[str, Any]:
    """
    Node 15: Build all WebSocket events for the VS Code plugin (schema v2.0).

    Improvements over v1:
      - parse_structured_output() replaces fragile regex parsers
        (falls back to regex if LLM did not produce <STRUCTURED_OUTPUT>)
      - diff_hunks computed via difflib for targeted code changes
      - request_id (UUID) for event correlation plugin ↔ server
      - elapsed_ms, analyzed_at, schema_version, file_name, change_score
      - fix_available / fix_id cross-references between issues and fixes
      - full_class fixes: current_code/fixed_code truncated; diff_hunks used

    Events emitted:
      • analysis_result    — structured issues[] + fixes[] (schema_version=2.0)
      • dependency_impact  — impacted files list
      • test_gap           — missing test warning
      • git_recommendation — commit recommended / blocked
      • known_issue        — recurring Redis pattern if any
    """
    import uuid

    file_path   = state.get("file_path", "")
    language    = state.get("language", "unknown")
    analysis    = state.get("analysis", {})
    raw_text    = analysis.get("analysis", "") if isinstance(analysis, dict) else str(analysis)
    source_code = state.get("code", "")

    # ── Metadata ──────────────────────────────────────────────────────────────
    request_id  = str(uuid.uuid4())
    analyzed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start_time  = state.get("stats", {}).get("start_time", time.time())
    elapsed_ms  = int((time.time() - start_time) * 1000)

    events: list = []

    # ── 1. analysis_result (shared builder) ───────────────────────────────────
    built = build_analysis_result_event(
        file_path=file_path,
        language=language,
        raw_text=raw_text,
        source_code=source_code,
        strategy_default=state.get("strategy", "block_fix"),
        change_score=int(state.get("change_info", {}).get("score", 0)),
        rag_docs_used=len(state.get("rag_docs", [])),
        elapsed_ms=elapsed_ms,
    )
    ar_event        = built["event"]
    enriched_issues = built["enriched_issues"]
    enriched_fixes  = built["enriched_fixes"]
    strategy        = built["strategy"]
    # Re-stamp request_id so downstream events correlate with this emission.
    ar_event["request_id"] = request_id
    ar_event["analyzed_at"] = analyzed_at
    events.append(ar_event)

    # ── 2. dependency_impact ──────────────────────────────────────────────────
    impacted = (
        state.get("dependents_to_analyze")
        or state.get("neighborhood", {}).get("predecessors", [])
    )

    # Keep only files that REALLY exist inside the analyzed project. The shared
    # dependency graph can contain entries from other indexed projects (or stale
    # paths); without this filter a 4-file project wrongly reported "5 dependent
    # files". This grounds the impact in the actual project on disk.
    import os as _os
    _proj = state.get("project_path", "") or ""
    if impacted:
        _proj_abs = _os.path.normcase(_os.path.abspath(_proj)) if _proj else ""

        def _in_project(f: str) -> bool:
            try:
                if not f or not _os.path.exists(f):
                    return False
                if _proj_abs:
                    return _os.path.normcase(_os.path.abspath(f)).startswith(_proj_abs)
                return True
            except Exception:
                return False

        # de-dup + filter to real project files, excluding the analyzed file itself
        _self_abs = _os.path.normcase(_os.path.abspath(file_path)) if file_path else ""
        seen = set()
        filtered = []
        for f in impacted:
            fa = _os.path.normcase(_os.path.abspath(f)) if f else ""
            if fa and fa != _self_abs and fa not in seen and _in_project(f):
                seen.add(fa)
                filtered.append(f)
        impacted = filtered

    if impacted:
        n    = len(impacted)
        risk = "high" if n > 5 else ("medium" if n >= 2 else "low")

        events.append({
            "type":           "dependency_impact",
            "schema_version": "2.0",
            "request_id":     request_id,
            "file_path":      file_path,
            "impacted_files": impacted,
            "risk":           risk,
            "reason": (
                f"{Path(file_path).name} exports symbols used by "
                f"{n} dependent file(s)"
            ),
            "analyzed_at":    analyzed_at,
        })

    # ── 3. test_gap ───────────────────────────────────────────────────────────
    test_gap = state.get("test_gap")

    if test_gap:
        related = [test_gap.get("test_file")] if test_gap.get("test_file") else []

        events.append({
            "type":               "test_gap",
            "schema_version":     "2.0",
            "request_id":         request_id,
            "file_path":          file_path,
            "severity":           "warning",
            "missing_tests":      test_gap.get("missing", True),
            "related_test_files": related,
            "recommendation":     test_gap.get("reason", "Generate or update tests"),
            "analyzed_at":        analyzed_at,
        })

    # ── 4. git_recommendation ─────────────────────────────────────────────────
    git_session = state.get("git_session")

    if git_session and git_session.get("has_data"):
        level      = git_session.get("level", "CLEAN")
        n_critical = git_session.get("total_critical", 0)
        files_risk = git_session.get("files_at_risk_count", 0)

        should_commit = level == "CLEAN" and n_critical == 0
        status_str    = "recommended" if should_commit else "not_recommended"
        risk_str      = "low" if should_commit else ("high" if level == "CRITICAL" else "medium")

        blocking: list = []
        if n_critical > 0:
            blocking.append(f"{n_critical} critical issue(s) not yet committed")
        if files_risk > 0:
            blocking.append(f"{files_risk} file(s) at risk")
        if test_gap:
            blocking.append(f"Missing tests for {Path(file_path).name}")

        msg = (
            "Your changes look stable. Commit recommended."
            if should_commit
            else f"Commit not recommended: {'; '.join(blocking) or 'issues detected'}."
        )

        events.append({
            "type":                     "git_recommendation",
            "schema_version":           "2.0",
            "request_id":               request_id,
            "status":                   status_str,
            "should_commit":            should_commit,
            "message":                  msg,
            "modified_files":           len(git_session.get("files_at_risk", [])),
            "staged_files":             0,
            "risk":                     risk_str,
            "blocking_reasons":         blocking,
            "suggested_commit_message": "",
            "analyzed_at":              analyzed_at,
        })

    # ── 5. known_issue ────────────────────────────────────────────────────────
    try:
        from langchain_agents.memory.redis_memory import PatternMemory

        pm  = PatternMemory()
        top = pm.get_top_patterns(language, n=5)

        for p in top:
            if p.get("count", 0) >= 3 and p.get("pattern", "") in raw_text:
                events.append({
                    "type":           "known_issue",
                    "schema_version": "2.0",
                    "request_id":     request_id,
                    "file_path":      file_path,
                    "issue_title":    p["pattern"],
                    "similarity":     round(min(p["count"] / 10.0, 1.0), 2),
                    "previous_fix":   p.get("fix", ""),
                    "seen_count":     p["count"],
                    "analyzed_at":    analyzed_at,
                })
                break
    except Exception:
        pass

    logger.info(
        "[WS] events built for %s: types=%s strategy=%s issues=%d fixes=%d elapsed=%dms",
        Path(file_path).name,
        [e["type"] for e in events],
        strategy,
        len(enriched_issues),
        len(enriched_fixes),
        elapsed_ms,
    )

    # ── Push the PRIMARY result to the plugin immediately ────────────────────
    # This runs BEFORE node_analyze_dependents, so the developer sees the result
    # of the file they just edited in ~real time instead of waiting for the whole
    # dependent cascade (root cause of the 297s lag). Dependents stream in after.
    broadcasted = False
    ws_broadcast = state.get("_ws_broadcast")
    if callable(ws_broadcast):
        for event in events:
            try:
                ws_broadcast(event)
            except Exception as exc:
                logger.warning("[WS] primary broadcast failed: %s", exc)
        broadcasted = True

    return {
        "issues":         enriched_issues,
        "ws_events":      events,
        "ws_broadcasted": broadcasted,
    }



# ═══════════════════════════════════════════════════════════════════════════════
# Conditional Edge Functions
# ═══════════════════════════════════════════════════════════════════════════════


def should_continue_or_skip(state: WatchState) -> Literal["continue", "skip"]:
    """Route: continue processing or skip to END."""
    if state.get("skip_reason"):
        return "skip"
    return "continue"


def has_dependents(state: WatchState) -> Literal["yes", "no"]:
    """Route: check if there are dependent files to re-analyze.

    Primary source  : dependents_to_analyze (from update_dep_graph via DependencyExtractor).
    Fallback source : neighborhood.predecessors (pre-built full project graph — covers Java,
                      TypeScript, and absolute Python imports that update_graph cannot resolve
                      incrementally).
    """
    if state.get("dependents_to_analyze"):
        return "yes"
    # Fallback: neighbourhood predecessors from the initial full-project graph build
    if state.get("neighborhood", {}).get("predecessors"):
        return "yes"
    return "no"


# ═══════════════════════════════════════════════════════════════════════════════
# Graph Builder
# ═══════════════════════════════════════════════════════════════════════════════


def build_watch_graph():
    """
    Build and compile the WatchGraph — the LangGraph orchestrator.

    Graph topology:
        hash_check ──→ read_file ──→ change_filter ──→ parse_ast
        ──→ index_chromadb ──→ update_kg ──→ update_dep_graph
        ──→ test_gap_detect ──→ get_neighborhood ──→ rag_retrieve ──→ git_session
        ──→ build_context ──→ llm_analyze ──→ cache_results
        ──→ emit_ws_events (broadcast PRIMARY result in real time)
        ──→ learn_feedback (KB promotion — after the developer already has results)
        ──→ [if deps] analyze_dependents (stream each dependent) ──→ END
        ──→ [no deps]                                            ──→ END
    """
    graph = StateGraph(WatchState)

    # ── Add nodes ────────────────────────────────────────────────────────────
    graph.add_node("hash_check",         node_hash_check)
    graph.add_node("read_file",          node_read_file)
    graph.add_node("change_filter",      node_change_filter)
    graph.add_node("parse_ast",          node_parse_ast)
    graph.add_node("index_chromadb",     node_index_chromadb)
    graph.add_node("update_kg",          node_update_kg)
    graph.add_node("update_dep_graph",   node_update_dep_graph)
    graph.add_node("test_gap_detect",    node_test_gap_detect)
    graph.add_node("get_neighborhood",   node_get_neighborhood)
    graph.add_node("rag_retrieve",       node_rag_retrieve)
    graph.add_node("git_session",        node_git_session)
    graph.add_node("build_context",      node_build_context)
    graph.add_node("llm_analyze",        node_llm_analyze)
    graph.add_node("cache_results",      node_cache_results)
    graph.add_node("learn_feedback",     node_learn_feedback)
    graph.add_node("analyze_dependents", node_analyze_dependents)
    graph.add_node("emit_ws_events",     node_emit_ws_events)   # NEW ← Node 15

    # ── Entry point ──────────────────────────────────────────────────────────
    graph.set_entry_point("hash_check")

    # ── Edges ────────────────────────────────────────────────────────────────
    graph.add_conditional_edges("hash_check", should_continue_or_skip, {
        "continue": "read_file",
        "skip":     END,
    })
    graph.add_conditional_edges("read_file", should_continue_or_skip, {
        "continue": "change_filter",
        "skip":     END,
    })
    graph.add_conditional_edges("change_filter", should_continue_or_skip, {
        "continue": "parse_ast",
        "skip":     END,
    })

    # Sequential pipeline
    graph.add_edge("parse_ast",         "index_chromadb")
    graph.add_edge("index_chromadb",    "update_kg")
    graph.add_edge("update_kg",         "update_dep_graph")
    graph.add_edge("update_dep_graph",  "test_gap_detect")
    graph.add_edge("test_gap_detect",   "get_neighborhood")
    graph.add_edge("get_neighborhood",  "rag_retrieve")
    graph.add_edge("rag_retrieve",      "git_session")
    graph.add_edge("git_session",       "build_context")
    graph.add_edge("build_context",     "llm_analyze")
    graph.add_edge("llm_analyze",       "cache_results")

    # Emit the PRIMARY result FIRST (real-time), THEN do KB learning + dependents.
    # Previously emit ran after analyze_dependents AND after the per-issue learning
    # LLM calls, so the primary result was held hostage for the whole cascade
    # (up to ~300s). Now emit broadcasts immediately after caching; learning and
    # the dependent stream happen afterwards without blocking the developer.
    graph.add_edge("cache_results",  "emit_ws_events")
    graph.add_edge("emit_ws_events", "learn_feedback")
    graph.add_conditional_edges("learn_feedback", has_dependents, {
        "yes": "analyze_dependents",
        "no":  END,
    })
    graph.add_edge("analyze_dependents", END)

    return graph.compile()


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience function for CLI integration
# ═══════════════════════════════════════════════════════════════════════════════


def invoke_watch(
    file_path: str,
    project_path: str = ".",
    project_indexer: Any = None,
    extractor: Any = None,
    rag_system: Any = None,
    dep_graph: Any = None,
    cache: Any = None,
    print_lock: Any = None,
    learning_agent: Any = None,
    file_counter: Any = None,
    ws_broadcast: Any = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper to invoke the WatchGraph for a single file.

    Args:
        file_path: Absolute path to the file to analyze.
        project_path: Project root directory.
        project_indexer: ProjectCodeIndexer instance (optional).
        extractor: DependencyExtractor instance (optional).
        rag_system: CodeRAGSystemAPI instance (optional).
        dep_graph: nx.DiGraph — dependency graph (optional).
        cache: CacheService instance (optional).
        print_lock: threading.Lock for console output (optional).
        learning_agent: Legacy LearningAgent instance (optional).
        file_counter: Shared dict for session stats (optional).

    Returns:
        Final WatchState dict with all analysis results.
    """
    graph = build_watch_graph()

    initial_state: WatchState = {
        "file_path": str(file_path),
        "project_path": str(project_path),
        "_project_indexer": project_indexer,
        "_extractor": extractor,
        "_rag_system": rag_system,
        "_dep_graph": dep_graph,
        "_cache": cache,
        "_print_lock": print_lock,
        "_learning_agent": learning_agent,
        "_file_counter": file_counter,
        "_ws_broadcast": ws_broadcast,
        "skip_reason": None,
        "post_solution_mode": False,
        "dependents_to_analyze": [],
        "stats": {"start_time": time.time()},
    }

    print(f"\n  {'═' * 60}")
    print(f"  {_B}WatchGraph{_R} — {Path(file_path).name}")
    print(f"  {'═' * 60}")

    result = graph.invoke(initial_state)

    elapsed = time.time() - initial_state["stats"]["start_time"]
    skip = result.get("skip_reason")
    if skip:
        print(f"  {_DM}Skipped: {skip}{_R} ({elapsed:.1f}s)")
    else:
        strategy = result.get("strategy", "?")
        print(f"  {_GR}Done{_R} (strategy={strategy}, {elapsed:.1f}s)")
    print(f"  {'═' * 60}\n")

    return result

