"""
rag_tools.py — LangChain @tool wrappers for RAG operations.

Tools:
  - tool_rag_retrieve         : System-aware RAG (2-pass: ChromaDB → reranker)
  - tool_kg_detect_patterns   : Knowledge Graph pattern detection
  - tool_index_file_chromadb  : Index a file into ChromaDB
  - tool_update_knowledge_graph : Incremental KG update for a file
  - tool_get_neighborhood     : Graph neighborhood extraction
"""
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.tools import tool

_fallback_rag_instance = None


@tool
def tool_rag_retrieve(
    code: str,
    neighborhood: dict,
    file_name: str,
    language: str,
) -> Dict[str, Any]:
    """Retrieve relevant RAG documents using System-Aware RAG pipeline.

    Priority: retriever_agent (full pipeline + cross-encoder) → CodeRAGSystemAPI (basic).

    Args:
        code: Source code content.
        neighborhood: Graph neighborhood dict (predecessors, successors, etc).
        file_name: Name of the current file.
        language: Programming language.

    Returns:
        Dict with 'docs' (list of doc contents) and 'scores' (list of floats).
    """
    # Try full System-Aware RAG (cross-encoder + KG + project code)
    try:
        from agents.retriever_agent import retriever_agent
        if retriever_agent._vector_store:
            neighborhood["language"] = language
            docs, scores = retriever_agent.retrieve_system_aware(
                current_code=code,
                neighborhood=neighborhood,
                current_file_name=file_name,
                networkx_graph=retriever_agent._extractor.graph if retriever_agent._extractor else None,
            )
            return {
                "docs": [{"content": d.page_content, "metadata": d.metadata} for d in docs],
                "scores": scores,
            }
    except Exception:
        pass

    # Fallback: basic CodeRAGSystemAPI (ChromaDB only, no cross-encoder)
    global _fallback_rag_instance
    if _fallback_rag_instance is None:
        from services.llm_service import CodeRAGSystemAPI
        _fallback_rag_instance = CodeRAGSystemAPI()

    docs, scores = _fallback_rag_instance._retrieve_relevant_knowledge(
        query=code[:2000],
        language=language,
    )

    return {
        "docs": [{"content": d.page_content, "metadata": d.metadata} for d in docs],
        "scores": scores,
    }


@tool
def tool_kg_detect_patterns(code: str, language: str) -> List[str]:
    """Detect vulnerability patterns in code using the Knowledge Graph.

    Args:
        code: Source code to analyze.
        language: Programming language.

    Returns:
        List of detected pattern names (e.g. SqlInjectionVulnerability).
    """
    from services.knowledge_graph import knowledge_graph

    if not knowledge_graph._built:
        return []

    detected = knowledge_graph.detect_patterns(code, language)
    return [name for name, _score in detected]


@tool
def tool_index_file_chromadb(file_path: str, content: str, entities: list) -> int:
    """Index a single file into ChromaDB via ProjectCodeIndexer.

    Args:
        file_path: Absolute file path.
        content: File content.
        entities: Parsed AST entities list.

    Returns:
        Number of chunks indexed.
    """
    # This is called by the graph node; the indexer must be passed via state
    # We return 0 here as a fallback — the real indexing happens in the graph node
    return 0


@tool
def tool_update_knowledge_graph(file_path: str) -> bool:
    """Incrementally update the Knowledge Graph for a single file.

    Args:
        file_path: Absolute file path.

    Returns:
        True if update succeeded.
    """
    from services.knowledge_graph import knowledge_graph

    try:
        knowledge_graph.update_file(file_path=Path(file_path), project_indexer=None, llm=None)
        return True
    except Exception:
        return False


@tool
def tool_get_neighborhood(file_path: str) -> Dict[str, Any]:
    """Extract the dependency graph neighborhood for a file.

    Returns predecessors (callers), successors (dependencies), indirect impact,
    entities, and criticality score.

    Args:
        file_path: Absolute file path.

    Returns:
        Neighborhood dict with predecessors, successors, indirect_impacted,
        predecessor_entities, successor_entities, criticality.
    """
    try:
        from agents.retriever_agent import retriever_agent
        if retriever_agent._extractor:
            return retriever_agent.get_neighborhood(Path(file_path))
    except Exception:
        pass

    # Fallback: empty neighborhood (no dependency graph available)
    return {
        "predecessors": [],
        "successors": [],
        "indirect_impacted": [],
        "predecessor_entities": [],
        "successor_entities": [],
        "criticality": 0,
        "is_entry_point": False,
    }
