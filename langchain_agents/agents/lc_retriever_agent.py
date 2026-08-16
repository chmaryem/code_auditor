
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
      
        return self.__class__(file_path=str(file_path), language=language)

    def retrieve_with_context(
        self,
        code: str,
        file_path: str,
        language: str,
        neighborhood: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
       
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
        
        return tool_kg_detect_patterns.invoke({
            "code": code,
            "language": language,
        })

    def get_neighborhood(self, file_path: str) -> Dict[str, Any]:
      
        return tool_get_neighborhood.invoke({"file_path": file_path})


# Singleton
lc_retriever_agent = LCRetrieverAgent()
