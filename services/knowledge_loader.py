"""
knowledge_loader.py — Script d'ingestion de la Knowledge Base vers ChromaDB
============================================================================

Rôle dans l'architecture RAG :
┌─────────────────────────────────────────────────────────────────────────┐
│  knowledge_base/                                                         │
│  ├── java/security/sql_injection.md        ← Fichiers source (.md/.txt) │
│  ├── java/patterns/solid_encapsulation.md                                │
│  └── ...                                                                 │
│          │                                                               │
│          ▼  knowledge_loader.py (CE FICHIER)                             │
│  1. Lit chaque .md et extrait le texte + front-matter YAML               │
│  2. Déduit les métadonnées depuis le chemin : java/security → lang+type  │
│  3. Découpe en chunks (RecursiveCharacterTextSplitter)                   │
│  4. Génère les embeddings (jinaai/jina-embeddings-v2-base-code)          │
│  5. Stocke dans ChromaDB avec métadonnées indexées                       │
│          │                                                               │
│          ▼                                                               │
│  data/vector_store/  ← Collection ChromaDB persistée                    │
└─────────────────────────────────────────────────────────────────────────┘

Utilisation :
    # Première fois (ou après modification des .md)
    python knowledge_loader.py

    # Forcer la réingestion complète (ex: changement de modèle d'embedding)
    python knowledge_loader.py --force

    # Afficher les statistiques de la collection sans réingérer
    python knowledge_loader.py --stats

    # Tester la recherche
    python knowledge_loader.py --test "SQL injection java PreparedStatement"

Comment les métadonnées influencent le LLM :
    Chaque chunk stocké dans ChromaDB porte des métadonnées :
    {
        "language":      "java",          ← filtrage par langage du fichier analysé
        "category":      "security",      ← type de règle
        "source_file":   "sql_injection.md",
        "severity":      "CRITICAL",      ← extrait du front-matter YAML
        "tags":          "sql-injection,jdbc,prepared-statement",
        "chunk_index":   2,               ← position dans le fichier source
        "total_chunks":  5,
    }

    Dans analyze_code_with_rag(), on effectue une recherche filtrée :
    Si language="java" → on pondère les résultats java avant les généraux.
    Si le score cosinus > 0.75 → le chunk est ignoré (non pertinent).
    Les chunks restants forment le BEST PRACTICES block du prompt LLM.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("knowledge_loader")




# Mapping chemin → métadonnées déduits automatiquement
# Format : "segment_du_chemin" → valeur de métadonnée
LANGUAGE_FROM_PATH: dict[str, str] = {
    "java":       "java",
    "python":     "python",
    "typescript": "typescript",
    "javascript": "javascript",
    "general":    "general",
}

CATEGORY_FROM_PATH: dict[str, str] = {
    "security":     "security",
    "patterns":     "patterns",
    "performance":  "performance",
    "quality":      "quality",
    "architecture": "architecture",
    "testing":      "testing",
}

# Séparateurs optimisés pour le Markdown technique avec blocs de code
MARKDOWN_SEPARATORS = [
    "\n## ",      # Section H2 — séparation principale
    "\n### ",     # Sous-section H3
    "\n```\n",    # Fin de bloc de code
    "\n\n",       # Paragraphe
    "\n",
    " ",
    "",
]


# ─────────────────────────────────────────────────────────────────────────────
# Parsing du Front-Matter YAML
# ─────────────────────────────────────────────────────────────────────────────

def parse_front_matter(content: str) -> tuple[dict[str, Any], str]:
    """
    Extrait le front-matter YAML des fichiers .md (entre --- et ---).

    Retourne (metadata_dict, content_sans_front_matter).

    Exemple de front-matter attendu :
        ---
        language: java
        category: security
        rule_type: vulnerability
        severity: CRITICAL
        tags: [sql-injection, jdbc]
        ---
    """
    front_matter: dict[str, Any] = {}

    # Vérifier si le fichier commence par ---
    if not content.startswith("---"):
        return front_matter, content

    # Chercher la fermeture ---
    end_match = re.search(r"\n---\n", content[3:])
    if not end_match:
        return front_matter, content

    yaml_block = content[3: end_match.start() + 3]
    remaining  = content[end_match.end() + 3:]

    # Parser manuellement (évite la dépendance pyyaml)
    for line in yaml_block.strip().splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, _, raw_value = line.partition(":")
        key       = key.strip()
        raw_value = raw_value.strip()

        # Gérer les listes YAML : [a, b, c] ou - item
        if raw_value.startswith("[") and raw_value.endswith("]"):
            items = [i.strip().strip("'\"") for i in raw_value[1:-1].split(",")]
            front_matter[key] = items
        elif raw_value:
            front_matter[key] = raw_value.strip("'\"")

    return front_matter, remaining.lstrip("\n")


# ─────────────────────────────────────────────────────────────────────────────
# Extraction des métadonnées depuis le chemin
# ─────────────────────────────────────────────────────────────────────────────

def metadata_from_path(file_path: Path, kb_root: Path) -> dict[str, str]:
    """
    Déduit les métadonnées depuis la position du fichier dans l'arborescence.

    Exemple :
      knowledge_base/java/security/sql_injection.md
     {"language": "java", "category": "security", "source_file": "sql_injection.md"}
    """
    try:
        rel = file_path.relative_to(kb_root)
    except ValueError:
        rel = file_path

    parts = rel.parts  # ex: ('java', 'security', 'sql_injection.md')

    language = "general"
    category = "general"

    for part in parts:
        part_lower = part.lower()
        if part_lower in LANGUAGE_FROM_PATH:
            language = LANGUAGE_FROM_PATH[part_lower]
        if part_lower in CATEGORY_FROM_PATH:
            category = CATEGORY_FROM_PATH[part_lower]

    return {
        "language":    language,
        "category":    category,
        "source_file": file_path.name,
        "source_path": str(rel),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Loader principal
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeBaseLoader:
    """
    Charge les fichiers de knowledge base dans ChromaDB.

    Flux complet :
      scan() → parse_file() → split_into_chunks() → build_metadata() → upsert()
    """

    def __init__(self, kb_dir: Path | None = None):
        self.kb_dir      = config.KNOWLEDGE_BASE_DIR
        self.splitter    = RecursiveCharacterTextSplitter(
            chunk_size    = config.rag.chunk_size,
            chunk_overlap = config.rag.chunk_overlap,
            separators    = MARKDOWN_SEPARATORS,
            length_function = len,
        )
        self._embeddings: HuggingFaceEmbeddings | None = None
        self._store:       Chroma | None               = None

  

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            logger.info("Chargement du modèle d'embeddings : %s (device=%s)",
                        config.rag.embedding_model, config.rag.embedding_device)
            self._embeddings = HuggingFaceEmbeddings(
                model_name   = config.rag.embedding_model,
                model_kwargs = {
                    "device":         config.rag.embedding_device,
                    "trust_remote_code": True,  # Requis pour Jina v2
                },
                encode_kwargs = {
                    "normalize_embeddings": True,   # Cosine similarity fiable
                    "batch_size":           32,     # Optimise le throughput
                },
            )
        return self._embeddings

    def _get_store(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                persist_directory  = str(config.VECTOR_STORE_DIR),
                embedding_function = self._get_embeddings(),
                collection_name    = config.CHROMA_COLLECTION,
            )
        return self._store

   
    def scan_files(self) -> list[Path]:
        """Retourne tous les .md et .txt dans la knowledge base."""
        if not self.kb_dir.exists():
            logger.warning("Knowledge base directory not found: %s", self.kb_dir)
            return []

        files = sorted(
            f for f in self.kb_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in (".md", ".txt")
        )
        logger.info("Fichiers trouvés dans knowledge_base: %d", len(files))
        return files

    # ── Traitement d'un fichier ───────────────────────────────────────────────

    def process_file(self, file_path: Path) -> list[Document]:
        """
        Lit un fichier, extrait front-matter + texte, découpe en chunks.
        Retourne une liste de Documents LangChain avec métadonnées complètes.
        """
        try:
            raw_content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            logger.error("Impossible de lire %s : %s", file_path, e)
            return []

        if not raw_content.strip():
            logger.debug("Fichier vide ignoré : %s", file_path.name)
            return []

        # ── 1. Parser le front-matter YAML ────────────────────────────────────
        front_meta, body = parse_front_matter(raw_content)

        # ── 2. Métadonnées depuis le chemin (source de vérité structurelle) ───
        path_meta = metadata_from_path(file_path, self.kb_dir)

        # ── 3. Fusion : front-matter > path metadata (le YAML a priorité) ────
        merged_meta: dict[str, Any] = {**path_meta, **front_meta}

        # Normaliser tags (liste → string pour ChromaDB qui ne supporte pas les listes)
        if isinstance(merged_meta.get("tags"), list):
            merged_meta["tags"] = ",".join(merged_meta["tags"])

        # S'assurer que toutes les valeurs sont des strings (ChromaDB l'exige)
        str_meta = {k: str(v) for k, v in merged_meta.items()}

        # ── 4. Découper le corps en chunks ────────────────────────────────────
        base_doc = Document(page_content=body, metadata=str_meta)
        chunks   = self.splitter.split_documents([base_doc])

        # ── 5. Enrichir chaque chunk avec son index ───────────────────────────
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"]  = str(i)
            chunk.metadata["total_chunks"] = str(len(chunks))

        logger.debug(
            "  %s → %d chunks (lang=%s, cat=%s)",
            file_path.name,
            len(chunks),
            str_meta.get("language", "?"),
            str_meta.get("category", "?"),
        )
        return chunks

    # ── Ingestion principale ──────────────────────────────────────────────────

    def load(self, force: bool = False) -> int:
        """
        Charge tous les fichiers dans ChromaDB.

        Args:
            force: Si True, vide la collection avant de réingérer.

        Returns:
            Nombre de chunks ingérés.
        """
        store = self._get_store()

        # Vérifier si déjà peuplé
        existing_count = store._collection.count()
        if existing_count > 0 and not force:
            logger.info(
                "Collection '%s' déjà peuplée (%d chunks). "
                "Utilisez --force pour réingérer.",
                config.CHROMA_COLLECTION, existing_count
            )
            return existing_count

        if force and existing_count > 0:
            logger.info(
                "Force réingestion : suppression des %d chunks existants...",
                existing_count
            )
            store._collection.delete(
                where={"source_file": {"$ne": "__sentinel__"}}  # Supprime tout
            )
            logger.info("Collection vidée.")

        # Traiter tous les fichiers
        files       = self.scan_files()
        all_chunks: list[Document] = []

        for file_path in files:
            chunks = self.process_file(file_path)
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("Aucun chunk à ingérer. Vérifiez le contenu de %s", self.kb_dir)
            return 0

        # Ingestion par lots (évite les timeouts sur grandes KB)
        BATCH_SIZE = 50
        total_ingested = 0

        logger.info("Ingestion de %d chunks en lots de %d...", len(all_chunks), BATCH_SIZE)

        for i in range(0, len(all_chunks), BATCH_SIZE):
            batch = all_chunks[i: i + BATCH_SIZE]
            try:
                store.add_documents(batch)
                total_ingested += len(batch)
                logger.info(
                    "  Lot %d/%d ingéré (%d chunks)",
                    i // BATCH_SIZE + 1,
                    (len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE,
                    len(batch),
                )
            except Exception as e:
                logger.error("Erreur lors de l'ingestion du lot %d : %s", i // BATCH_SIZE + 1, e)
                raise

        logger.info(" Ingestion terminée : %d chunks dans '%s'",
                    total_ingested, config.CHROMA_COLLECTION)
        return total_ingested

    # ── Statistiques ─────────────────────────────────────────────────────────

    def print_stats(self) -> None:
        """Affiche les statistiques détaillées de la collection."""
        store = self._get_store()
        total = store._collection.count()

        print(f"\n{'═' * 60}")
        print(f"  Collection ChromaDB : {config.CHROMA_COLLECTION}")
        print(f"  Modèle d'embeddings : {config.rag.embedding_model}")
        print(f"  Device              : {config.rag.embedding_device}")
        print(f"  Dimension           : {config.rag.embedding_dimension}")
        print(f"  Total chunks        : {total}")
        print(f"{'═' * 60}")

        if total == 0:
            print("Collection vide — lancez : python knowledge_loader.py")
            return

        # Récupérer tous les metadata pour les stats
        try:
            results = store._collection.get(include=["metadatas"])
            metadatas = results.get("metadatas", [])

            # Comptages par langage
            from collections import Counter
            lang_counts  = Counter(m.get("language",  "unknown") for m in metadatas)
            cat_counts   = Counter(m.get("category",  "unknown") for m in metadatas)
            sev_counts   = Counter(m.get("severity",  "")        for m in metadatas if m.get("severity"))
            file_counts  = Counter(m.get("source_file","unknown") for m in metadatas)

            print("\n  Par langage :")
            for lang, count in sorted(lang_counts.items()):
                bar = "█" * (count // 2)
                print(f"    {lang:<15} {count:>4} chunks  {bar}")

            print("\n  Par catégorie :")
            for cat, count in sorted(cat_counts.items()):
                print(f"    {cat:<15} {count:>4} chunks")

            if sev_counts:
                print("\n  Par sévérité (front-matter) :")
                for sev, count in sev_counts.most_common():
                    print(f"    {sev:<12} {count:>4} chunks")

            print(f"\n  Fichiers sources : {len(file_counts)}")
            for fname, count in sorted(file_counts.items()):
                print(f"    {fname:<40} {count:>3} chunks")

        except Exception as e:
            logger.debug("Stats détaillées non disponibles : %s", e)

        print(f"{'═' * 60}\n")

    # ── Test de recherche ─────────────────────────────────────────────────────

    def test_search(self, query: str, language: str | None = None, k: int = 5) -> None:
        """
        Teste la recherche dans la collection avec affichage des scores.
        Permet de vérifier que le seuil de pertinence fonctionne bien.
        """
        store = self._get_store()
        threshold = config.rag.relevance_threshold

        print(f"\n{'─' * 60}")
        print(f"  Requête : {query!r}")
        print(f"  Seuil   : {threshold} (score > {threshold} = ignoré)")
        print(f"{'─' * 60}")

        # Recherche avec scores
        results = store.similarity_search_with_score(query, k=k)

        kept   = [(doc, score) for doc, score in results if score <= threshold]
        filtered = [(doc, score) for doc, score in results if score > threshold]

        print(f"\n   Conservés ({len(kept)}/{k}) :")
        for doc, score in kept:
            meta = doc.metadata
            print(f"    [{score:.3f}] {meta.get('source_file','?')} "
                  f"({meta.get('language','?')}/{meta.get('category','?')})")
            print(f"           {doc.page_content[:120].strip()!r}...")

        if filtered:
            print(f"\n   Filtrés (score > {threshold}) ({len(filtered)}) :")
            for doc, score in filtered:
                meta = doc.metadata
                print(f"    [{score:.3f}] {meta.get('source_file','?')} — NON PERTINENT")

        print(f"{'─' * 60}\n")




# ─────────────────────────────────────────────────────────────────────────────
# ProjectCodeIndexer — Changement 1
# ─────────────────────────────────────────────────────────────────────────────

class ProjectCodeIndexer:
    """
    Indexe le code source du projet dans une collection ChromaDB séparée.

    Deux collections ChromaDB dans le système :
      code_kb_jina_v2    ← KnowledgeBaseLoader  (règles génériques dans les .md)
      project_code_index ← ProjectCodeIndexer   (code RÉEL du projet, méthode par méthode)

    Pourquoi une collection séparée ?
      La KB de règles explique COMMENT corriger un problème (générique).
      project_code_index montre comment TON projet gère les mêmes patterns.
      Exemple : UserService.java a un resource leak sur Statement.
        → KB retourne : "utilise try-with-resources" (règle générique)
        → project_code_index retourne : findByUsername() dans UserRepository.java
          qui a exactement le même problème → le LLM voit que c'est systémique.

    Cycle de vie :
      1. initialize() dans IncrementalAnalyzer → index_project() une seule fois
      2. À chaque Ctrl+S → index_file() remplace les chunks du fichier sauvegardé
         → la collection reste toujours synchronisée avec le projet

    Réutilise les embeddings de assistant_agent (déjà chargés en mémoire).
    Pas de double chargement du modèle Jina.
    """

    COLLECTION_NAME = "project_code_index"

    # Extensions de fichiers à indexer
    CODE_EXTENSIONS = {".py", ".java", ".js", ".ts", ".jsx", ".tsx"}

    # Dossiers à ignorer
    EXCLUDED_DIRS = {
        "__pycache__", ".git", "node_modules", ".venv", "venv",
        "dist", "build", "target", ".idea", ".vscode",
    }

    def __init__(self, embeddings):
        """
        Args:
            embeddings : HuggingFaceEmbeddings déjà initialisé dans assistant_agent.
                         Réutilisation directe — évite de recharger Jina (2-3 Go RAM).
        """
        import threading
        self._embeddings = embeddings
        self._store: Chroma | None = None
        self._lock = threading.Lock()   # protège _store contre les accès concurrents
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size    = 500,
            chunk_overlap = 60,
            separators    = ["\n\n", "\n", " ", ""],
        )

    def _get_store(self) -> Chroma:
        # Double-checked locking — thread-safe sans verrouiller à chaque appel
        if self._store is None:
            with self._lock:
                if self._store is None:
                    # Répertoire SÉPARÉ de la KB principale (code_kb_jina_v2).
                    # Sur Windows, ChromaDB utilise SQLite — deux collections dans
                    # le même répertoire partagent le même fichier .db → verrou partagé
                    # → blocage garanti quand les deux écrivent simultanément.
                    project_store_dir = config.VECTOR_STORE_DIR.parent / "project_code_store"
                    project_store_dir.mkdir(parents=True, exist_ok=True)
                    self._store = Chroma(
                        persist_directory  = str(project_store_dir),
                        embedding_function = self._embeddings,
                        collection_name    = self.COLLECTION_NAME,
                    )
        return self._store

    # ── Indexation initiale du projet ─────────────────────────────────────────

    def index_project(self, project_path: Path, force: bool = False) -> int:
        """
        Indexe tous les fichiers du projet au démarrage.
        Skip si la collection est déjà peuplée (sauf si force=True).

        Appelé une seule fois dans IncrementalAnalyzer.initialize().

        Returns:
            Nombre de chunks indexés (0 si déjà peuplé et force=False).
        """
        store = self._get_store()
        existing = store._collection.count()

        if existing > 0 and not force:
            logger.info(
                "project_code_index déjà peuplé (%d chunks) — skip indexation initiale.",
                existing,
            )
            return existing

        if force and existing > 0:
            store._collection.delete(where={"collection": {"$eq": "project_code"}})
            logger.info("project_code_index vidé pour réindexation.")

        # Scanner et indexer tous les fichiers
        files = self._scan_project(project_path)
        logger.info("ProjectCodeIndexer : %d fichiers à indexer...", len(files))

        total = 0
        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                if content.strip():
                    n = self._do_index_file(file_path, content, entities=[])
                    total += n
            except Exception as e:
                logger.debug("Erreur indexation %s : %s", file_path.name, e)

        logger.info("ProjectCodeIndexer : %d chunks indexés.", total)
        return total

    # ── Mise à jour incrémentale (appelée à chaque Ctrl+S) ───────────────────

    def index_file(
        self,
        file_path: Path,
        content:   str,
        entities:  list,
    ) -> int:
        """
        Réindexe un fichier après modification.
        Appelé automatiquement dans IncrementalAnalyzer._analyze_file() à l'ÉTAPE 4.5.

        Fix Windows ChromaDB : toute l'opération est wrappée dans un try/except global.
        Sur Windows, SQLite (backend ChromaDB) peut verrouiller la DB juste après
        index_project() → la suppression bloquerait tout le pipeline.
        Si ChromaDB est occupé on retourne 0 silencieusement — le pipeline continue.
        """
        try:
            store = self._get_store()

            # Supprimer l'ancienne version — non bloquant grâce au try/except global
            try:
                store._collection.delete(
                    where={"source_file": {"$eq": file_path.name}}
                )
            except Exception:
                # ChromaDB verrouillé ou collection vide → on continue sans supprimer
                pass

            return self._do_index_file(file_path, content, entities)

        except Exception as e:
            logger.debug("ProjectCodeIndexer.index_file() ignoré pour %s : %s", file_path.name, e)
            return 0

    def _do_index_file(
        self,
        file_path: Path,
        content:   str,
        entities:  list,
    ) -> int:
        """
        Logique d'indexation commune à index_project() et index_file().
        """
        store    = self._get_store()
        language = file_path.suffix.lstrip(".").lower()
        docs: list[Document] = []
        lines = content.splitlines()

        # ── Stratégie 1 : découpe par méthode (si entités disponibles) ────────
        method_entities = [
            e for e in entities
            if (e.get("type") if isinstance(e, dict) else getattr(e, "type", ""))
            in ("method", "function", "class")
        ]

        if method_entities:
            for entity in method_entities:
                # Compatibilité dict (project_indexer) et objet (code_parser)
                if isinstance(entity, dict):
                    name       = entity.get("name", "")
                    etype      = entity.get("type", "method")
                    start_line = entity.get("start_line", 1)
                    end_line   = entity.get("end_line", start_line + 10)
                    params     = entity.get("parameters", [])
                else:
                    name       = getattr(entity, "name", "")
                    etype      = getattr(entity, "type", "method")
                    start_line = getattr(entity, "start_line", 1)
                    end_line   = getattr(entity, "end_line", start_line + 10)
                    params     = getattr(entity, "parameters", [])

                method_code = "\n".join(lines[start_line - 1 : end_line])
                if len(method_code.strip()) < 15:
                    continue

                docs.append(Document(
                    page_content = method_code,
                    metadata     = {
                        "source_file":  file_path.name,
                        "source_path":  str(file_path),
                        "entity_name":  name,
                        "entity_type":  etype,
                        "parameters":   ", ".join(params[:6]) if params else "",
                        "language":     language,
                        "start_line":   str(start_line),
                        "collection":   "project_code",
                    },
                ))

        # ── Stratégie 2 : découpe générique (fallback) ────────────────────────
        else:
            chunks = self._splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 15:
                    continue
                docs.append(Document(
                    page_content = chunk,
                    metadata     = {
                        "source_file": file_path.name,
                        "source_path": str(file_path),
                        "entity_name": "",
                        "entity_type": "chunk",
                        "language":    language,
                        "chunk_index": str(i),
                        "collection":  "project_code",
                    },
                ))

        if not docs:
            return 0

        # Ingestion par lots de 20
        for i in range(0, len(docs), 20):
            store.add_documents(docs[i : i + 20])

        return len(docs)

    # ── Recherche ─────────────────────────────────────────────────────────────

    def search(
        self,
        query:        str,
        k:            int   = 4,
        exclude_file: str   = "",
        threshold:    float = 0.75,
    ) -> list:
        """
        Cherche du code similaire dans le projet.

        Args:
            query        : code ou description du pattern recherché
            k            : nombre de résultats max
            exclude_file : nom de fichier à exclure (évite l'autoréférence)
            threshold    : score cosinus max (plus bas = plus similaire)

        Returns:
            Liste de (Document, score) triée par score croissant.
        """
        store   = self._get_store()
        results = store.similarity_search_with_score(query, k=k * 2)

        filtered = [
            (doc, score) for doc, score in results
            if score <= threshold
            and doc.metadata.get("source_file", "") != exclude_file
        ]
        filtered.sort(key=lambda x: x[1])
        return filtered[:k]

    # ── Stats ─────────────────────────────────────────────────────────────────

    def print_stats(self) -> None:
        """Affiche les statistiques de la collection project_code_index."""
        store = self._get_store()
        total = store._collection.count()
        print(f"\n  project_code_index : {total} chunks de code projet")
        if total > 0:
            results  = store._collection.get(include=["metadatas"])
            metas    = results.get("metadatas", [])
            from collections import Counter
            by_file  = Counter(m.get("source_file", "?") for m in metas)
            by_lang  = Counter(m.get("language",    "?") for m in metas)
            print(f"  Langages : {dict(by_lang)}")
            print(f"  Fichiers ({len(by_file)}) :")
            for fname, count in sorted(by_file.items()):
                print(f"    {fname:<40} {count:>3} chunks")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _scan_project(self, project_path: Path) -> list[Path]:
        """Scanne récursivement le projet en ignorant les dossiers exclus."""
        files = []
        for f in project_path.rglob("*"):
            if not f.is_file():
                continue
            if f.suffix.lower() not in self.CODE_EXTENSIONS:
                continue
            if any(part in self.EXCLUDED_DIRS for part in f.parts):
                continue
            files.append(f)
        return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Charge la knowledge base dans ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python knowledge_loader.py                                      # Ingestion si vide
  python knowledge_loader.py --force                             # Réingestion complète
  python knowledge_loader.py --stats                             # Statistiques seules
  python knowledge_loader.py --test "SQL injection java"         # Test de recherche
  python knowledge_loader.py --test "async await typescript" --lang typescript
        """,
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Vide la collection ChromaDB et réingère tout depuis les fichiers .md"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Affiche les statistiques de la collection sans ingérer"
    )
    parser.add_argument(
        "--test", metavar="QUERY",
        help="Teste une recherche dans la collection avec affichage des scores"
    )
    parser.add_argument(
        "--lang", metavar="LANGUAGE",
        help="Filtre par langage dans le test de recherche (java|python|typescript)"
    )
    parser.add_argument(
        "--kb-dir", metavar="PATH",
        help="Chemin alternatif vers le dossier knowledge_base"
    )
    args = parser.parse_args()

    kb_dir = Path(args.kb_dir) if args.kb_dir else None
    loader = KnowledgeBaseLoader(kb_dir=kb_dir)

    if args.stats:
        loader.print_stats()
        return

    if args.test:
        # Si stats non affichées mais test demandé, vérifier que la collection est peuplée
        store = loader._get_store()
        if store._collection.count() == 0:
            logger.warning("Collection vide — ingestion automatique...")
            loader.load()
        loader.test_search(args.test, language=args.lang)
        return

    # Ingestion
    print("\n" + "═" * 60)
    print(f"  Knowledge Base Loader")
    print(f"  Source : {loader.kb_dir}")
    print(f"  Modèle : {config.rag.embedding_model}")
    print(f"  Device : {config.rag.embedding_device}")
    print("═" * 60 + "\n")

    count = loader.load(force=args.force)
    loader.print_stats()

    if count > 0:
        print(f" {count} chunks prêts dans ChromaDB")
        print(f"   Lancez maintenant : python main.py watch <projet>\n")
    else:
        print("  Aucun chunk ingéré — vérifiez knowledge_base/")


if __name__ == "__main__":
    main()