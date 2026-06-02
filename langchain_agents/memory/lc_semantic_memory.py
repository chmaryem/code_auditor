"""
lc_semantic_memory.py — Semantic session memory for ChatAgent (Phase C1).

Unlike Redis history (raw turn storage), this agent:
  - Maintains a per-session VECTOR STORE of key facts extracted from conversation
  - On each new message, retrieves the most relevant past context
  - Continuously learns user preferences, recurring patterns, and project-specific knowledge

Architecture:
  write_memory(session_id, turn) → extract facts → embed → store in ChromaDB
  recall_memory(session_id, query) → vector search → top-k facts
  get_profile(session_id) → structured profile {expertise, preferences, patterns}

Storage: ChromaDB collection "chat_memory_{session_id}" (persistent per session)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_FACT_EXTRACT_PROMPT = """\
Extract key facts from this developer conversation turn that are worth remembering.
Focus on: recurring problems, user expertise level, preferred patterns, project-specific knowledge.
Return JSON list of short fact strings (max 5, each ≤ 80 chars). Empty list if nothing notable.

User: {user_message}
Assistant: {assistant_message}

JSON:"""

_PROFILE_PROMPT = """\
Based on these conversation facts, build a developer profile in JSON:
{{
  "expertise_level": "junior|mid|senior",
  "preferred_language": "python|java|...",
  "recurring_issues": ["list"],
  "preferred_patterns": ["list"],
  "project_context": "one line summary"
}}

Facts:
{facts}

JSON profile:"""


class SemanticMemoryAgent:
    """
    Per-session semantic memory using ChromaDB.

    Each session gets its own collection: chat_memory_<session_id>
    Facts are extracted by LLM and stored as embeddings for semantic recall.
    """

    COLLECTION_PREFIX = "chat_mem_"
    MAX_FACTS_PER_SESSION = 200
    RECALL_TOP_K = 5

    def __init__(self):
        self._collections: Dict[str, Any] = {}  # session_id → chroma collection
        self._chroma_client = None
        self._embed_fn = None

    def _get_client(self):
        if self._chroma_client is None:
            try:
                import chromadb
                from config import config
                persist_dir = str(Path(config.CHROMA_PERSIST_DIR) / "semantic_memory")
                self._chroma_client = chromadb.PersistentClient(path=persist_dir)
            except Exception as e:
                logger.warning("ChromaDB init failed for semantic memory: %s", e)
        return self._chroma_client

    def _get_collection(self, session_id: str):
        if session_id in self._collections:
            return self._collections[session_id]
        client = self._get_client()
        if not client:
            return None
        try:
            col = client.get_or_create_collection(
                name=f"{self.COLLECTION_PREFIX}{session_id[:32]}",
                metadata={"hnsw:space": "cosine"},
            )
            self._collections[session_id] = col
            return col
        except Exception as e:
            logger.debug("get_collection failed: %s", e)
            return None

    def _extract_facts(self, user_message: str, assistant_message: str) -> List[str]:
        """LLM-based fact extraction from a conversation turn."""
        if not user_message or not assistant_message:
            return []
        try:
            from services.llm_factory import invoke_with_fallback
            import re
            prompt = _FACT_EXTRACT_PROMPT.format(
                user_message=user_message[:400],
                assistant_message=assistant_message[:400],
            )
            raw = invoke_with_fallback(prompt, label="fact_extract", max_tokens=128)
            raw = raw.strip()
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw.strip())
            facts = json.loads(raw)
            if isinstance(facts, list):
                return [str(f)[:80] for f in facts[:5]]
        except Exception as e:
            logger.debug("fact extraction failed: %s", e)
        return []

    def write_memory(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Extract facts from a turn and store in session vector store.
        Returns list of extracted facts (for logging).
        """
        facts = self._extract_facts(user_message, assistant_message)
        if not facts:
            return []

        col = self._get_collection(session_id)
        if col is None:
            return facts

        try:
            ts = int(time.time())
            col.add(
                documents=facts,
                ids=[f"{session_id}:{ts}:{i}" for i in range(len(facts))],
                metadatas=[{
                    "session_id":   session_id,
                    "timestamp":    ts,
                    "source":       "conversation",
                    **(metadata or {}),
                }] * len(facts),
            )
            logger.debug("Stored %d facts for session %s", len(facts), session_id)
        except Exception as e:
            logger.debug("write_memory failed: %s", e)

        return facts

    def recall_memory(self, session_id: str, query: str) -> List[str]:
        """
        Semantic recall: retrieve most relevant past facts for a query.
        Returns list of fact strings (empty if no memory or ChromaDB unavailable).
        """
        col = self._get_collection(session_id)
        if col is None:
            return []

        try:
            count = col.count()
            if count == 0:
                return []
            k = min(self.RECALL_TOP_K, count)
            results = col.query(query_texts=[query[:300]], n_results=k)
            docs = results.get("documents", [[]])[0]
            return [d for d in docs if d]
        except Exception as e:
            logger.debug("recall_memory failed: %s", e)
            return []

    def get_profile(self, session_id: str) -> Dict[str, Any]:
        """
        Build a developer profile from all stored facts.
        Returns empty dict if insufficient data.
        """
        col = self._get_collection(session_id)
        if col is None:
            return {}

        try:
            count = col.count()
            if count < 5:
                return {}

            # Get all facts (up to 50)
            results = col.get(limit=50)
            docs = results.get("documents", [])
            if not docs:
                return {}

            facts_text = "\n".join(f"- {d}" for d in docs[:30])
            from services.llm_factory import invoke_with_fallback
            import re
            raw = invoke_with_fallback(
                _PROFILE_PROMPT.format(facts=facts_text),
                label="profile_build",
                max_tokens=200,
            )
            raw = re.sub(r"^```[a-z]*\n?", "", raw.strip())
            raw = re.sub(r"\n?```$", "", raw.strip())
            return json.loads(raw)
        except Exception as e:
            logger.debug("get_profile failed: %s", e)
            return {}

    def clear_session(self, session_id: str) -> bool:
        """Delete all semantic memory for a session."""
        client = self._get_client()
        if not client:
            return False
        try:
            client.delete_collection(f"{self.COLLECTION_PREFIX}{session_id[:32]}")
            self._collections.pop(session_id, None)
            return True
        except Exception as e:
            logger.debug("clear_session failed: %s", e)
            return False


semantic_memory = SemanticMemoryAgent()
