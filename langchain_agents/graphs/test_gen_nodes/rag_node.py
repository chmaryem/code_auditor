"""
rag_node.py — RAGAgent.

Wraps steps 6–7 of generate_for_file():
  - retrieve_rag_context: test_patterns_kb (threshold 0.75, k=5)
                          + ProjectCodeIndexer (threshold 0.80, k=5, test files only)
                          + filesystem fallback when no indexer is available
  - get_kg_context: NetworkX Knowledge Graph relations for the source file

The RAG query uses untested_signatures so incremental runs retrieve context
scoped to the methods that still need tests.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_agents.graphs.test_gen_nodes._helpers import (
    get_kg_context,
    retrieve_rag_context,
)

logger = logging.getLogger(__name__)


def node_rag(state: Dict[str, Any]) -> Dict[str, Any]:
    source_path = state["source_path"]
    source_code = state["source_code"]
    project_path = state["project_path"]
    untested_signatures = state["untested_signatures"]

    # 6. RAG — patterns de test + exemples du projet
    rag_context = retrieve_rag_context(
        source_path,
        source_code,
        untested_signatures,
        state.get("_test_kb"),
        state.get("_project_code_indexer"),
        project_path,
    )

    # 7. Contexte Knowledge Graph (dépendances, classes liées)
    kg_context = get_kg_context(source_path, state.get("_knowledge_graph"))

    return {
        "rag_context": rag_context,
        "kg_context":  kg_context,
    }
