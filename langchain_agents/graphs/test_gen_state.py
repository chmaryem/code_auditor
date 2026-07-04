"""
test_gen_state.py — State for the TestGenGraph (multi-agent test generation).

Refactors the monolithic TestGeneratorAgent.generate_for_file() (15 sequential
steps) into a real LangGraph pipeline of 6 agent nodes:

  Discovery → Analysis → RAG → Generation → Validation → Execution

The State carries every value that used to be a local variable inside
generate_for_file(), plus the injected RAG/KG services (same lazy-injection
pattern as HardeningState._runner / WatchState._ws_broadcast).

ZERO REGRESSION is the contract: this State is only a transport — all business
logic lives verbatim in test_gen_nodes/_helpers.py, moved (not rewritten) from
agents/test_generator_agent.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict


class TestGenState(TypedDict, total=False):
    """
    State threaded through every TestGenGraph node.

    Mirrors the locals of the original generate_for_file(); field groups are
    tagged with the node that first populates them.
    """

    # ── Input (set once at launch, never mutated) ──────────────────────────────
    source_path:   Path
    project_path:  Path
    write:         bool
    incremental:   bool

    # ── Injected services (same lazy pattern as HardeningState._runner) ────────
    _discovery:            Any   # TestDiscoveryService
    _test_kb:              Any   # TestKnowledgeLoader (ChromaDB test_patterns_kb)
    _project_code_indexer: Any   # ProjectCodeIndexer
    _knowledge_graph:      Any   # KnowledgeGraph (NetworkX)
    _runner:               Any   # TestRunner

    # ── DiscoveryAgent ─────────────────────────────────────────────────────────
    framework:     str
    convention:    str
    language:      str
    source_code:   str
    target_path:   Path
    cache_hit:     bool

    # ── AnalysisAgent ──────────────────────────────────────────────────────────
    signatures:          List[Dict[str, Any]]   # ALL signatures (public + private)
    imports:             List[str]
    dependency_classes:  List[Any]              # [(class_name, usage), ...]
    untested_signatures: List[Dict[str, Any]]   # incremental filter result
    is_incremental:      bool
    existing_test_code:  str

    # ── RAGAgent ───────────────────────────────────────────────────────────────
    rag_context:   Dict[str, Any]
    kg_context:    str

    # ── GenerationAgent ────────────────────────────────────────────────────────
    prompt:            str   # base prompt, memorised for retries
    safe_source_code:  str   # secret-redacted source
    generated_code:    str
    _pre_retry_code:   str   # original code kept if a structural retry fails
    _runtime_fix:      bool  # True if the runtime retry produced a fix

    # ── ValidationAgent ────────────────────────────────────────────────────────
    validated:         bool
    validation_error:  Optional[str]

    # ── ExecutionAgent ─────────────────────────────────────────────────────────
    run_result:        Any               # TestRunResult or None
    run_success:       Optional[bool]
    run_error_summary: Optional[str]
    run_error:         Any               # TestRunResult carried into runtime retry

    # ── Retry counters (fidelity: structural max 1, runtime max 1) ─────────────
    retry_structural:  int
    retry_runtime:     int

    # ── Transient routing markers (set by a node, read by its conditional edge) ─
    _gen_route:   str   # "validate" | "execute" | "finalize" | "end"
    _val_route:   str   # "generate" | "proceed"
    _exec_route:  str   # "generate" | "finalize" | "end"

    # ── Output ─────────────────────────────────────────────────────────────────
    final_output:  Dict[str, Any]        # the exact dict returned by generate_for_file()
    error:         Optional[str]


# Fidelity caps — the original code did exactly ONE structural retry and ONE
# runtime retry. Keep these at 1 to guarantee zero regression.
MAX_RETRY_STRUCTURAL = 1
MAX_RETRY_RUNTIME = 1
