"""
RetrieverAgent — Cherche les règles pertinentes dans la base de connaissances.

Rôle UNIQUE : étant donné du code parsé, trouver les règles KB pertinentes.

Amélioration clé vs ancien système :
  AVANT → un seul fichier entier envoyé comme requête (signal brouillé)
  APRÈS → une requête PAR entité (fonction/classe) → signal précis et ciblé
"""
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document


class RetrieverAgent:
    """
    Recherche dans ChromaDB les règles pertinentes pour un fichier de code.

    Usage :
        agent = RetrieverAgent(vector_store)
        docs, scores = agent.retrieve(parsed_code)
    """

    def __init__(self, vector_store=None):
        self._store = vector_store

    def set_vector_store(self, vector_store):
        self._store = vector_store

    def retrieve(
        self,
        parsed_code: Dict[str, Any],
        top_k:       int   = 8,
        threshold:   float = 0.45,
    ) -> Tuple[List[Document], List[float]]:
        """
        Cherche les règles KB pertinentes pour le code parsé.
        Stratégie : une requête par entité au lieu d'une seule pour tout le fichier.
        """
        if self._store is None:
            return [], []

        language = parsed_code.get("language", "unknown")
        entities = parsed_code.get("entities", [])
        code     = parsed_code.get("code", "")

        queries = self._build_queries(entities, language, code)

        # Recherche pour chaque requête + déduplication par contenu
        all_docs: Dict[str, Tuple[Document, float]] = {}
        for query in queries:
            for doc, score in self._search_one(query, language, k=3, threshold=threshold):
                key = doc.page_content[:80]
                if key not in all_docs or score < all_docs[key][1]:
                    all_docs[key] = (doc, score)

        ranked = sorted(all_docs.values(), key=lambda x: x[1])[:top_k]
        return [d for d, _ in ranked], [s for _, s in ranked]

    def _build_queries(self, entities: list, language: str, code: str) -> List[str]:
        """Une requête par fonction/classe + une requête de sécurité globale."""
        queries = []
        for entity in entities[:10]:
            name = entity.get("name", "")
            sig  = entity.get("signature", "")
            if name:
                queries.append(f"{name} {sig} {language}".strip())
        if code:
            queries.append(f"security vulnerability {language} {code[:500]}")
        if not entities:
            queries.append(code[:1000])
        return queries

    def _search_one(
        self,
        query:     str,
        language:  str,
        k:         int   = 3,
        threshold: float = 0.45,
    ) -> List[Tuple[Document, float]]:
        try:
            results = self._store.similarity_search_with_score(query, k=k * 2)
        except Exception:
            return []
        filtered = [(doc, score) for doc, score in results if score <= threshold]
        filtered.sort(key=lambda p: (
            0 if p[0].metadata.get("language", "") == language else 1, p[1]
        ))
        return filtered[:k]


retriever_agent = RetrieverAgent()