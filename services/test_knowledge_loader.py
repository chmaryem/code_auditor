"""
test_knowledge_loader.py — Gestion de la collection ChromaDB séparée pour les patterns de test.

Architecture :
  3 collections ChromaDB dans le système :
    code_kb_jina_v2    ← KnowledgeBaseLoader  (règles génériques dans les .md)
    project_code_index ← ProjectCodeIndexer   (code RÉEL du projet)
    test_patterns_kb   ← TestKnowledgeLoader  (patterns de test par langage)  ← CE FICHIER

  Pourquoi une collection séparée pour les tests ?
    - Les patterns de test (JUnit, pytest, Jest) sont spécialisés et structurés
    - La recherche RAG doit pouvoir cibler uniquement les patterns de test
    - Les styles de test varient fortement entre langages → filtrage par metadata.language
    - Évite de polluer la collection d'audit avec du contenu test

  Utilisation :
    loader = TestKnowledgeLoader(embeddings=existing_embeddings)
    loader.load()  # Charge les patterns depuis knowledge_base/*/testing/
    docs, scores = loader.search("JDBC connection test", language="java")
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import config

logger = logging.getLogger(__name__)

# Nom de la collection ChromaDB dédiée aux patterns de test
TEST_KB_COLLECTION = "test_patterns_kb"

# Répertoire de stockage séparé (évite le verrou SQLite partagé sur Windows)
TEST_KB_STORE_DIR = config.DATA_DIR / "test_patterns_store"


class TestKnowledgeLoader:
    """
    Charge et recherche les patterns de test unitaire dans une collection ChromaDB dédiée.

    Cycle de vie :
      1. __init__(embeddings) — réutilise les embeddings Jina déjà chargés
      2. load()              — scanne knowledge_base/*/testing/*.md et ingère
      3. search(query, lang) — recherche sémantique avec filtrage par langage
    """

    def __init__(self, embeddings: HuggingFaceEmbeddings | None = None):
        """
        Args:
            embeddings: Embeddings Jina déjà chargés (réutilisation depuis assistant_agent).
                       Si None, charge un modèle frais (fallback).
        """
        self._embeddings = embeddings
        self._store: Optional[Chroma] = None
        self._lock = threading.Lock()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n## ", "\n### ", "\n```\n", "\n\n", "\n", " ", ""],
        )

    # ── Store management ──────────────────────────────────────────────────────

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            logger.info("TestKnowledgeLoader: chargement embeddings frais...")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=config.rag.embedding_model,
                model_kwargs={
                    "device": config.rag.embedding_device,
                    "trust_remote_code": True,
                },
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": 32,
                },
            )
        return self._embeddings

    def _get_store(self) -> Chroma:
        if self._store is None:
            with self._lock:
                if self._store is None:
                    TEST_KB_STORE_DIR.mkdir(parents=True, exist_ok=True)
                    self._store = Chroma(
                        persist_directory=str(TEST_KB_STORE_DIR),
                        embedding_function=self._get_embeddings(),
                        collection_name=TEST_KB_COLLECTION,
                    )
        return self._store

 
    def load(self, force: bool = False) -> int:
        """
        Charge les patterns de test depuis knowledge_base/*/testing/*.md.

        Args:
            force: Si True, vide la collection et réingère tout.

        Returns:
            Nombre de chunks ingérés.
        """
        store = self._get_store()
        existing = store._collection.count()

        if existing > 0 and not force:
            logger.info(
                "test_patterns_kb déjà peuplé (%d chunks) — skip.",
                existing,
            )
            return existing

        if force and existing > 0:
            logger.info("Force réingestion test_patterns_kb...")
            try:
                store._collection.delete(
                    where={"category": {"$eq": "testing"}}
                )
            except Exception:
                # Fallback: vider tout
                try:
                    ids = store._collection.get()["ids"]
                    if ids:
                        store._collection.delete(ids=ids)
                except Exception:
                    pass

        # Scanner les fichiers de test dans la KB
        kb_root = config.KNOWLEDGE_BASE_DIR
        test_files = self._scan_test_files(kb_root)
        logger.info("TestKnowledgeLoader: %d fichiers de patterns de test trouvés", len(test_files))

        all_chunks: List[Document] = []
        for file_path in test_files:
            chunks = self._process_file(file_path, kb_root)
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("Aucun pattern de test à ingérer.")
            return 0

        # Ingestion par lots
        BATCH_SIZE = 30
        total = 0
        for i in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[i: i + BATCH_SIZE]
            try:
                store.add_documents(batch)
                total += len(batch)
            except Exception as e:
                logger.error("Erreur ingestion test patterns lot %d: %s", i // BATCH_SIZE, e)

        logger.info("test_patterns_kb: %d chunks ingérés", total)
        return total

    def _scan_test_files(self, kb_root: Path) -> List[Path]:
        """Scanne les fichiers de patterns de test dans knowledge_base/*/testing/."""
        files = []
        if not kb_root.exists():
            return files

        for test_dir in kb_root.rglob("testing"):
            if test_dir.is_dir():
                for f in test_dir.rglob("*"):
                    if f.is_file() and f.suffix.lower() in (".md", ".txt"):
                        files.append(f)

        return sorted(files)

    def _process_file(self, file_path: Path, kb_root: Path) -> List[Document]:
        """Parse un fichier de patterns de test et le découpe en chunks."""
        try:
            raw = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            logger.error("Lecture impossible %s: %s", file_path, e)
            return []

        if not raw.strip():
            return []

        # Parser le front-matter
        from services.knowledge_loader import parse_front_matter, metadata_from_path
        front_meta, body = parse_front_matter(raw)
        path_meta = metadata_from_path(file_path, kb_root)

        # Fusion (front-matter prioritaire)
        merged = {**path_meta, **front_meta}
        merged["category"] = "testing"  # Force la catégorie

        # Normaliser les tags
        if isinstance(merged.get("tags"), list):
            merged["tags"] = ",".join(merged["tags"])

        str_meta = {k: str(v) for k, v in merged.items()}

        # Découper en chunks
        base_doc = Document(page_content=body, metadata=str_meta)
        chunks = self._splitter.split_documents([base_doc])

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = str(i)
            chunk.metadata["total_chunks"] = str(len(chunks))
            chunk.metadata["kb_type"] = "test_pattern"

        return chunks

    # ── Recherche sémantique ──────────────────────────────────────────────────

    def search(
        self,
        query: str,
        language: str = "",
        k: int = 5,
        threshold: float = 0.55,
    ) -> Tuple[List[Document], List[float]]:
        """
        Recherche des patterns de test pertinents.

        Args:
            query: Code source ou description du pattern recherché.
            language: Langage à privilégier (boost dans le tri).
            k: Nombre max de résultats.
            threshold: Score cosinus max (plus bas = plus pertinent).

        Returns:
            (documents, scores) — triés par pertinence.
        """
        store = self._get_store()

        # Vérifier que la collection n'est pas vide
        if store._collection.count() == 0:
            logger.debug("test_patterns_kb vide — aucun résultat.")
            return [], []

        results = store.similarity_search_with_score(query, k=k * 2)

        if not results:
            return [], []

        # Filtre par seuil
        filtered = [
            (doc, score) for doc, score in results
            if score <= threshold
        ]

        # Boost par langage
        lang_lower = language.lower() if language else ""
        if lang_lower:
            filtered.sort(
                key=lambda pair: (
                    0 if pair[0].metadata.get("language", "") == lang_lower else 1,
                    pair[1],
                )
            )

        filtered = filtered[:k]

        docs = [doc for doc, _ in filtered]
        scores = [score for _, score in filtered]

        logger.debug(
            "test_patterns_kb: %d/%d résultats pour lang=%s",
            len(docs), len(results), language,
        )
        return docs, scores

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Retourne les statistiques de la collection."""
        store = self._get_store()
        total = store._collection.count()
        stats = {"total_chunks": total, "by_language": {}}

        if total > 0:
            try:
                results = store._collection.get(include=["metadatas"])
                metas = results.get("metadatas", [])
                from collections import Counter
                by_lang = Counter(m.get("language", "unknown") for m in metas)
                stats["by_language"] = dict(by_lang)
            except Exception:
                pass

        return stats
