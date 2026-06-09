"""
lc_retriever_agent.py — LangChain RetrieverAgent.

4 Pillars:
  LLM     : Cross-encoder reranker (local model for pass 2)
  Tools   : tool_rag_retrieve, tool_kg_detect_patterns, tool_get_neighborhood
  Memory  : ChromaDB (vector store), Knowledge Graph (NetworkX), ProjectCodeIndexer
  Planning: 2-pass strategy → pass 1 broad (20 candidates) → pass 2 rerank (8 final)

This agent is a custom LangChain BaseRetriever that implements the
System-Aware RAG pipeline.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_agents.tools.rag_tools import (
    tool_get_neighborhood,
    tool_kg_detect_patterns,
    tool_rag_retrieve,
)
from langchain_agents.tools.code_tools import tool_detect_language

logger = logging.getLogger(__name__)


class LCRetrieverAgent(BaseRetriever):
    """
    System-Aware RAG Retriever — LangChain BaseRetriever implementation.

    Anatomy:
      - LLM:      Cross-encoder reranker (pass 2)
      - Tools:    rag_retrieve, kg_detect_patterns, get_neighborhood
      - Memory:   ChromaDB vector store + Knowledge Graph (NetworkX)
      - Planning: 2-pass retrieval strategy

    This retriever is composable with any LangChain chain via the | operator.
    """

    # Optional context. When set, _get_relevant_documents (the interface used by
    # the `|` operator) becomes system-aware instead of forcing the basic path.
    # Both default to the previous behavior → fully backward compatible.
    language: str = "unknown"
    file_path: str = ""

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """
        Retrieve documents using the 2-pass system-aware pipeline.

        LangChain BaseRetriever interface (used by the `|` operator). When a
        `file_path`/`language` context is configured (e.g. via `for_file()`),
        it resolves the language from the extension and pulls the dependency-
        graph neighborhood, so composing the retriever in a chain no longer
        forces the basic fallback. With no context set, it degrades gracefully
        to basic retrieval (previous behavior).

        Args:
            query: The code content to find relevant knowledge for.

        Returns:
            List of relevant Document objects.
        """
        language = self.language or "unknown"
        file_name = "unknown"
        neighborhood: Dict[str, Any] = {}

        if self.file_path:
            file_name = Path(self.file_path).name
            if language == "unknown":
                try:
                    language = tool_detect_language.invoke(
                        {"file_path": self.file_path}
                    ) or "unknown"
                except Exception:
                    language = "unknown"
            # System-aware: dependency-graph neighborhood for this file
            try:
                neighborhood = tool_get_neighborhood.invoke(
                    {"file_path": self.file_path}
                )
            except Exception:
                neighborhood = {}

        result = tool_rag_retrieve.invoke({
            "code": query,
            "neighborhood": neighborhood,
            "file_name": file_name,
            "language": language,
        })

        docs = []
        for d in result.get("docs", []):
            if isinstance(d, dict):
                docs.append(Document(
                    page_content=d.get("content", ""),
                    metadata=d.get("metadata", {}),
                ))
        return docs

    def for_file(
        self,
        file_path: str,
        language: str = "unknown",
    ) -> "LCRetrieverAgent":
        """
        Return a NEW retriever bound to a file's context.

        Keeps the retriever system-aware when composed via `|` without mutating
        the shared singleton (thread-safe for concurrent multi-agent graphs):

            chain = lc_retriever_agent.for_file(path) | prompt | llm
        """
        return self.__class__(file_path=str(file_path), language=language)

    def retrieve_with_context(
        self,
        code: str,
        file_path: str,
        language: str,
        neighborhood: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Full system-aware retrieval with all context.

        Planning pillar:
          Pass 1 — Broad search with multi-query (ChromaDB × 3 sources)
          Pass 2 — Cross-encoder reranking (top-8)

        Args:
            code: Source code content.
            file_path: File path for context.
            language: Programming language.
            neighborhood: Graph neighborhood dict (optional).

        Returns:
            Dict with 'docs' and 'scores'.
        """
        if neighborhood is None:
            neighborhood = tool_get_neighborhood.invoke({"file_path": file_path})

        result = tool_rag_retrieve.invoke({
            "code": code,
            "neighborhood": neighborhood,
            "file_name": Path(file_path).name,
            "language": language,
        })

        # Convert to Document objects
        docs = []
        for d in result.get("docs", []):
            if isinstance(d, dict):
                docs.append(Document(
                    page_content=d.get("content", ""),
                    metadata=d.get("metadata", {}),
                ))
        return {
            "docs": docs,
            "scores": result.get("scores", []),
        }

    def detect_patterns(self, code: str, language: str) -> List[str]:
        """
        Detect vulnerability patterns via Knowledge Graph (Tool pillar).

        Args:
            code: Source code.
            language: Programming language.

        Returns:
            List of detected pattern names.
        """
        return tool_kg_detect_patterns.invoke({
            "code": code,
            "language": language,
        })

    def get_neighborhood(self, file_path: str) -> Dict[str, Any]:
        """
        Extract dependency graph neighborhood (Tool pillar).

        Args:
            file_path: Absolute file path.

        Returns:
            Neighborhood dict.
        """
        return tool_get_neighborhood.invoke({"file_path": file_path})


# Singleton
lc_retriever_agent = LCRetrieverAgent()
