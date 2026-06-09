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
_fallback_indexer_instance = None


def _get_project_code_indexer():
    """Return a live ProjectCodeIndexer instance for tool_index_file_chromadb.

    Priority:
      1. Orchestrator's initialized instance (retriever_agent._project_code_indexer)
         — preferred: shares the already-loaded Jina embeddings and the same
         `project_code_index` collection used by the rest of the pipeline.
      2. Lazy fallback: build one reusing assistant_agent's embeddings.
         Writes to the same store/collection (COLLECTION_NAME is a class constant),
         so retrieval stays consistent regardless of which path created it.

    Returns None only if no embeddings/indexer can be obtained at all.
    """
    # 1. Live instance wired by the orchestrator
    try:
        from agents.retriever_agent import retriever_agent
        if retriever_agent._project_code_indexer is not None:
            return retriever_agent._project_code_indexer
    except Exception:
        pass

    # 2. Lazy fallback — reuse the Jina embeddings already loaded by assistant_agent
    global _fallback_indexer_instance
    if _fallback_indexer_instance is None:
        try:
            from services.knowledge_loader import ProjectCodeIndexer
            from services.llm_service import assistant_agent
            _fallback_indexer_instance = ProjectCodeIndexer(
                embeddings=assistant_agent.embeddings
            )
        except Exception:
            return None
    return _fallback_indexer_instance


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

    Reindexes the file's code (method-by-method when entities are provided,
    otherwise generic chunks) into the `project_code_index` collection, so later
    System-Aware RAG can surface similar real project code. The old version of
    the file is replaced first, keeping the collection in sync.

    Args:
        file_path: Absolute file path.
        content: File content.
        entities: Parsed AST entities list (dicts or CodeEntity objects).

    Returns:
        Number of chunks indexed (0 if no indexer is available or ChromaDB is
        momentarily locked — the call is always non-blocking).
    """
    indexer = _get_project_code_indexer()
    if indexer is None:
        return 0
    try:
        return indexer.index_file(
            file_path=Path(file_path),
            content=content,
            entities=entities or [],
        )
    except Exception:
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
