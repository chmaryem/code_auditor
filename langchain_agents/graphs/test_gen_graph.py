"""
test_gen_graph.py — TestGenGraph: multi-agent unit-test generation.

Refactors TestGeneratorAgent.generate_for_file() (a 1176-line monolith of 15
sequential steps) into a real LangGraph StateGraph of 6 agent nodes, WITHOUT any
behavioural change. All business logic lives verbatim in test_gen_nodes/_helpers.py.

Topology
--------
  START → discovery ──[cache_hit]──────────────────────────────► END
              │
          analysis ──[incremental & fully tested]──────────────► END
              │
            rag → generation
                       ▲   │  ├─[no LLM code]────────────────────► END
   [structural retry]  │   │  ├─[validate]──► validation
                       │   │  ├─[execute]───► execution   (runtime retry fix)
                       │   │  └─[finalize]──► finalize     (runtime retry, no fix)
                       │   │
   validation ─[proceed & write]──► execution              [proceed & !write] ─► finalize
              └─[retry]──► generation
                       │
   execution ─[fail & retries left]──► generation (runtime)
             ├─[success | exhausted]──► finalize
             └─[disk write failed]────► END

              finalize → END

Fidelity notes (see decisions baked in):
  - Structural retry capped at 1; the ORIGINAL first-pass code is restored if the
    retry still fails. Validation NEVER stops the pipeline (it only gated caching).
  - Runtime retry capped at 1; the retried code is NOT re-validated (goes straight
    back to execution), exactly like the original.
  - ExecutionAgent runs only when write=True (write=False skips straight to finalize).

Singleton pattern: the graph is compiled once and reused (same as watch_graph /
hardening_graph).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Literal

from langgraph.graph import END, StateGraph

from langchain_agents.graphs.test_gen_state import TestGenState
from langchain_agents.graphs.test_gen_nodes.discovery_node import node_discovery
from langchain_agents.graphs.test_gen_nodes.analysis_node import node_analysis
from langchain_agents.graphs.test_gen_nodes.rag_node import node_rag
from langchain_agents.graphs.test_gen_nodes.generation_node import node_generation
from langchain_agents.graphs.test_gen_nodes.validation_node import node_validation
from langchain_agents.graphs.test_gen_nodes.execution_node import node_execution, node_finalize

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Conditional edge routers
# ═══════════════════════════════════════════════════════════════════════════════

def route_after_discovery(state: TestGenState) -> Literal["analysis", "end"]:
    if state.get("cache_hit"):
        return "end"
    if state.get("final_output") is not None:   # read-error early exit
        return "end"
    return "analysis"


def route_after_analysis(state: TestGenState) -> Literal["rag", "end"]:
    # AnalysisAgent sets final_output only on the incremental "fully tested" exit.
    if state.get("final_output") is not None:
        return "end"
    return "rag"


def route_after_generation(state: TestGenState) -> Literal["validation", "execution", "finalize", "end"]:
    route = state.get("_gen_route")
    if route == "end":
        return "end"
    if route == "execute":
        return "execution"
    if route == "finalize":
        return "finalize"
    return "validation"


def route_after_validation(state: TestGenState) -> Literal["generation", "execution", "finalize"]:
    if state.get("_val_route") == "generate":
        return "generation"
    # proceed: original wrote + executed only when write=True
    if state.get("write"):
        return "execution"
    return "finalize"


def route_after_execution(state: TestGenState) -> Literal["generation", "finalize", "end"]:
    route = state.get("_exec_route")
    if route == "generate":
        return "generation"
    if route == "end":
        return "end"
    return "finalize"


# ═══════════════════════════════════════════════════════════════════════════════
# Graph builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_test_gen_graph() -> Any:
    """Build and compile the TestGenGraph (compiled once, reused per request)."""
    graph = StateGraph(TestGenState)

    graph.add_node("discovery",  node_discovery)
    graph.add_node("analysis",   node_analysis)
    graph.add_node("rag",        node_rag)
    graph.add_node("generation", node_generation)
    graph.add_node("validation", node_validation)
    graph.add_node("execution",  node_execution)
    graph.add_node("finalize",   node_finalize)

    graph.set_entry_point("discovery")

    # Fixed edges
    graph.add_edge("rag", "generation")
    graph.add_edge("finalize", END)

    # Conditional edges
    graph.add_conditional_edges("discovery", route_after_discovery, {
        "analysis": "analysis", "end": END,
    })
    graph.add_conditional_edges("analysis", route_after_analysis, {
        "rag": "rag", "end": END,
    })
    graph.add_conditional_edges("generation", route_after_generation, {
        "validation": "validation", "execution": "execution",
        "finalize": "finalize", "end": END,
    })
    graph.add_conditional_edges("validation", route_after_validation, {
        "generation": "generation", "execution": "execution", "finalize": "finalize",
    })
    graph.add_conditional_edges("execution", route_after_execution, {
        "generation": "generation", "finalize": "finalize", "end": END,
    })

    return graph.compile()


# Singleton — compiled once, reused across requests (same as watch/hardening graphs)
test_gen_graph = build_test_gen_graph()


# ═══════════════════════════════════════════════════════════════════════════════
# Public API — used by the thin TestGeneratorAgent wrapper
# ═══════════════════════════════════════════════════════════════════════════════

def generate_tests_via_graph(
    source_path:          Path,
    project_path:         Path,
    write:                bool,
    incremental:          bool,
    discovery:            Any,
    runner:               Any,
    test_kb:              Any = None,
    project_code_indexer: Any = None,
    knowledge_graph:      Any = None,
) -> Dict[str, Any]:
    """
    Run one test-generation pass through the graph and return the exact dict that
    the original generate_for_file() produced.

    The caller (TestGeneratorAgent) still owns the source_path.exists() guard, so
    by the time we get here the file is known to exist.
    """
    initial_state: TestGenState = {
        # Input
        "source_path": source_path,
        "project_path": project_path,
        "write": write,
        "incremental": incremental,
        # Injected services
        "_discovery": discovery,
        "_runner": runner,
        "_test_kb": test_kb,
        "_project_code_indexer": project_code_indexer,
        "_knowledge_graph": knowledge_graph,
        # Retry counters
        "retry_structural": 0,
        "retry_runtime": 0,
        # Working defaults
        "validation_error": None,
        "run_error": None,
        "final_output": None,
    }

    result = test_gen_graph.invoke(initial_state)
    out = result.get("final_output")
    if not isinstance(out, dict):
        # Defensive: should never happen — every terminal path sets final_output.
        logger.warning("TestGenGraph produced no final_output; returning error dict")
        return {"error": "Génération de tests : aucun résultat produit.", "test_file": None}
    return out
