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
    """Node 9b: Git session monitoring (reads from Redis, 0 LLM token)."""
    from langchain_agents.agents.lc_git_session_agent import LCGitSessionAgent

    project_path = state.get("project_path", ".")
    agent = LCGitSessionAgent(Path(project_path))

    git_session = agent.get_session_context()
    if git_session and git_session.get("has_data"):
        level = git_session.get("level", "CLEAN")
        if level in ("WARN", "CRITICAL"):
            alert = agent.format_alert(git_session)
            if alert:
                print(f"    🔥 {alert}")
        else:
            print(f"    📊 Git session: {level} (score={git_session.get('score', 0)})")
        return {"git_session": git_session}
    print("    📊 Git session: pas de données (vérifiez si des fichiers sont non-commités)")
    return {"git_session": None}


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
    promoted = result.get("rules_promoted", [])
    if promoted:
        print(f"    {_GR}KB rules promoted: {', '.join(promoted)}{_R}")

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

    # ── 3. Display recurring patterns ────────────────────────────────────────
    try:
        from langchain_agents.memory.redis_memory import PatternMemory
        pm = PatternMemory()
        top = pm.get_top_patterns(language, n=3)
        recurring = [p for p in top if p["count"] >= 3]
        if recurring:
            print(f"\n    {_YL}⚠  Patterns récurrents ({language}) :{_R}")
            for p in recurring:
                print(f"      {_YL}• {p['pattern']} — vu {p['count']}× → vérifiez la KB{_R}")
    except Exception:
        pass

    return {"learning_result": result}


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

    if not dependents:
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
    analysis_text = state.get("analysis", {}).get("analysis", "")[:1500]
    print_lock = state.get("_print_lock")
    analyzed = 0

    for dep_str in to_analyze:
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
            f"Verify that THIS file still compiles and works correctly with it.\n"
            f"Summary of changes in {changed_file}:\n{analysis_text[:800]}"
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

    Returns a CompiledGraph that can be invoked with:
        graph.invoke({"file_path": "/path/to/file.py", ...})

    Graph topology:
        hash_check ──→ read_file ──→ change_filter ──→ parse_ast
        ──→ index_chromadb ──→ update_kg ──→ update_dep_graph
        ──→ test_gap_detect ──→ get_neighborhood ──→ rag_retrieve ──→ git_session
        ──→ build_context ──→ llm_analyze ──→ cache_results ──→ learn_feedback
        ──→ [if deps] analyze_dependents ──→ END
    """
    graph = StateGraph(WatchState)

    # ── Add nodes ────────────────────────────────────────────────────────────
    graph.add_node("hash_check",        node_hash_check)
    graph.add_node("read_file",         node_read_file)
    graph.add_node("change_filter",     node_change_filter)
    graph.add_node("parse_ast",         node_parse_ast)
    graph.add_node("index_chromadb",    node_index_chromadb)
    graph.add_node("update_kg",         node_update_kg)
    graph.add_node("update_dep_graph",  node_update_dep_graph)
    graph.add_node("test_gap_detect",   node_test_gap_detect)
    graph.add_node("get_neighborhood",  node_get_neighborhood)
    graph.add_node("rag_retrieve",      node_rag_retrieve)
    graph.add_node("git_session",       node_git_session)
    graph.add_node("build_context",     node_build_context)
    graph.add_node("llm_analyze",       node_llm_analyze)
    graph.add_node("cache_results",     node_cache_results)
    graph.add_node("learn_feedback",    node_learn_feedback)
    graph.add_node("analyze_dependents", node_analyze_dependents)

    # ── Entry point ──────────────────────────────────────────────────────────
    graph.set_entry_point("hash_check")

    # ── Edges ────────────────────────────────────────────────────────────────
    # Node 1 → conditional: skip or continue
    graph.add_conditional_edges("hash_check", should_continue_or_skip, {
        "continue": "read_file",
        "skip":     END,
    })

    # Node 2 → conditional: unsupported language → skip
    graph.add_conditional_edges("read_file", should_continue_or_skip, {
        "continue": "change_filter",
        "skip":     END,
    })

    # Node 3 → conditional: minor change → skip
    graph.add_conditional_edges("change_filter", should_continue_or_skip, {
        "continue": "parse_ast",
        "skip":     END,
    })

    # Nodes 4-13: sequential pipeline
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
    graph.add_edge("cache_results",     "learn_feedback")

    # Node 13 → conditional: dependents or END
    graph.add_conditional_edges("learn_feedback", has_dependents, {
        "yes": "analyze_dependents",
        "no":  END,
    })

    # Node 14 → END
    graph.add_edge("analyze_dependents", END)

    # ── Compile ──────────────────────────────────────────────────────────────
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

