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

        This is the LangChain BaseRetriever interface method.

        Args:
            query: The code content to find relevant knowledge for.

        Returns:
            List of relevant Document objects.
        """
        # For basic retrieval, use the tool directly
        result = tool_rag_retrieve.invoke({
            "code": query,
            "neighborhood": {},
            "file_name": "unknown",
            "language": "unknown",
        })

        docs = []
        for d in result.get("docs", []):
            if isinstance(d, dict):
                docs.append(Document(
                    page_content=d.get("content", ""),
                    metadata=d.get("metadata", {}),
                ))
        return docs

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
