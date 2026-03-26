"""
RetrieverAgent — Recherche RAG consciente du système.

Contient :
  - GraphNeighborhoodExtractor : extrait le voisinage NetworkX d'un fichier
    (predecessors = appelants, successors = dépendances, impact indirect)
  - SystemAwareRAG             : pipeline RAG en 2 passes
      Passe 1 : ChromaDB multi-query → top-20 candidats
        • Queries structurelles (code brut + signatures appelants + dépendances)
        • KG expand_queries (depth=2) — nœuds détectés → règles connexes
        • KG n_hop_retrieval (depth=3) — NetworkX + KG → règles N-hop
        • Project code index — code projet similaire
      Passe 2 : Cross-encoder reranking → top-8 vraiment pertinents
      Fusion + déduplication + tri final.
  - RetrieverAgent             : façade publique qui orchestre les deux

Nouveautés :
  - Cross-encoder reranking (sentence-transformers, modèle local)
  - _build_rerank_query() : query enrichie avec patterns KG pour le reranker
  - TOP_K_CANDIDATES = 20 (passe 1 large) → TOP_K = 8 (passe 2 précise)
  - Fallback automatique si sentence-transformers absent
  - _detected_patterns injecté dans neighborhood pour le reranking
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from config import config as _cfg

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# GraphNeighborhoodExtractor
# ─────────────────────────────────────────────────────────────────────────────

class GraphNeighborhoodExtractor:
    """
    Interroge le graphe NetworkX pour extraire le voisinage complet d'un fichier.

    Sources combinées :
      1. NetworkX predecessors/successors (imports résolus)
      2. ProjectIndexer.get_related_files() (convention de nommage — filet de sécurité)

    Retourne :
      predecessors          : fichiers qui utilisent ce fichier (risque de casse)
      successors            : fichiers que ce fichier utilise
      indirect_impacted     : predecessors des predecessors (profondeur 2)
      predecessor_entities  : {fichier: [{name, params, criticality}]}
      successor_entities    : {fichier: [{name, params, criticality}]}
      criticality           : nb de predecessors directs
    """

    def __init__(self, graph, project_indexer):
        self.graph           = graph
        self.project_indexer = project_indexer

    def get_neighborhood(self, file_path: Path) -> Dict[str, Any]:
        node_id = f"file:{file_path}"

        # Source 1 : NetworkX
        if self.graph.has_node(node_id):
            pred_paths_nx = [
                n.replace("file:", "")
                for n in self.graph.predecessors(node_id)
                if n.startswith("file:")
            ]
            succ_paths_nx = [
                n.replace("file:", "")
                for n in self.graph.successors(node_id)
                if n.startswith("file:")
            ]
        else:
            pred_paths_nx = []
            succ_paths_nx = []

        # Source 2 : conventions de nommage (filet de sécurité)
        related_by_name = []
        if self.project_indexer and self.project_indexer.context:
            try:
                related_by_name = self.project_indexer.get_related_files(file_path)
            except Exception:
                pass

        # Union sans doublons (dict.fromkeys préserve l'ordre)
        all_preds = list(dict.fromkeys(pred_paths_nx + related_by_name))
        all_succs = list(dict.fromkeys(succ_paths_nx))

        # Entités enrichies (name + params + criticality)
        pred_entities = self._collect_entities_rich(all_preds)
        succ_entities = self._collect_entities_rich(all_succs)

        # Impact indirect (profondeur 2)
        indirect_impacted = set()
        for fp in all_preds:
            pred_node = f"file:{fp}"
            if self.graph.has_node(pred_node):
                for grand_pred in self.graph.predecessors(pred_node):
                    if grand_pred.startswith("file:") and grand_pred != node_id:
                        indirect_impacted.add(grand_pred.replace("file:", ""))

        return {
            "predecessors":         all_preds,
            "successors":           all_succs,
            "indirect_impacted":    list(indirect_impacted),
            "predecessor_entities": pred_entities,
            "successor_entities":   succ_entities,
            "criticality":          len(all_preds),
        }

    def _collect_entities_rich(
        self, file_paths: List[str]
    ) -> Dict[str, List[Dict]]:
        """
        Collecte les entités enrichies (name + params + criticality)
        depuis project_indexer.
        """
        result = {}
        if not self.project_indexer or not self.project_indexer.context:
            return result

        context_files = self.project_indexer.context.files
        for fp in file_paths:
            file_name = Path(fp).name
            file_info = context_files.get(fp, {})
            entities  = file_info.get("entities", [])
            file_crit = file_info.get("criticality", 0)
            rich = []
            for e in entities:
                if e.get("type") not in ("method", "function", "class"):
                    continue
                raw_params = e.get("parameters", [])
                params_str = ", ".join(raw_params[:6])
                if len(raw_params) > 6:
                    params_str += ", ..."
                rich.append({
                    "name":        e.get("name", ""),
                    "params":      params_str,
                    "criticality": file_crit,
                })
            if rich:
                result[file_name] = rich
        return result

    @staticmethod
    def empty() -> Dict[str, Any]:
        return {
            "predecessors":         [],
            "successors":           [],
            "indirect_impacted":    [],
            "predecessor_entities": {},
            "successor_entities":   {},
            "criticality":          0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SystemAwareRAG
# ─────────────────────────────────────────────────────────────────────────────

class SystemAwareRAG:
    """
    RAG conscient du système — pipeline en 2 passes.

    Passe 1 (large) : ChromaDB multi-query → jusqu'à TOP_K_CANDIDATES=20 candidats
      Sources A : queries structurelles (code + signatures appelants + dépendances)
      Sources B : KG expand_queries depth=2
      Sources C : KG n_hop_retrieval depth=3
      Sources D : project_code_index (code réel du projet)

    Passe 2 (précise) : Cross-encoder reranking → TOP_K=8 docs finals
      Modèle : cross-encoder/ms-marco-MiniLM-L-6-v2 (local, ~80MB)
      Score final = 0.7 * ce_score_normalized + 0.3 * l2_score_inverted
      Fallback : tri L2 simple si sentence-transformers absent

    IMPORTANT : distance L2 Jina (THRESHOLD=1.2) ≠ cosinus (relevance_threshold=0.45)
    """

    # ── Attributs de classe ───────────────────────────────────────────────────

    # Singleton cross-encoder — chargé une seule fois pour tout le processus
    _reranker        = None
    _reranker_loaded = False

    # Seuil L2 par défaut — écrasé par config.rag.rag_l2_threshold si présent
    THRESHOLD = 1.2

    # Passe 1 : nb de candidats avant reranking
    TOP_K_CANDIDATES = 20

    # Passe 2 : nb de docs retenus après reranking
    TOP_K = 8

    # ── Initialisation ────────────────────────────────────────────────────────

    def __init__(
        self,
        vector_store,
        language:             str,
        project_code_indexer  = None,
        knowledge_graph       = None,
    ):
        self.vs                   = vector_store
        self.language             = language.lower()
        self.project_code_indexer = project_code_indexer
        self._kg                  = knowledge_graph

        # Seuil L2 depuis config (modifiable sans toucher au code)
        try:
            self.THRESHOLD = _cfg.rag.rag_l2_threshold
        except Exception:
            pass  # garder la valeur par défaut (1.2)

        # Construire le KG si nécessaire
        if self._kg and not self._kg._built:
            self._kg.build()

    # ── Cross-encoder singleton ───────────────────────────────────────────────

    @classmethod
    def _get_reranker(cls):
        """
        Charge le cross-encoder une seule fois (lazy singleton).
        Retourne None si sentence-transformers n'est pas installé.
        → Le pipeline continue sans reranking (fallback L2).
        """
        if not cls._reranker_loaded:
            try:
                from sentence_transformers import CrossEncoder
                cls._reranker = CrossEncoder(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    max_length=512,
                )
                logger.info("Cross-encoder reranker chargé ✓")
            except ImportError:
                logger.debug(
                    "sentence-transformers absent — reranking désactivé. "
                    "Installez avec : pip install sentence-transformers"
                )
                cls._reranker = None
            except Exception as e:
                logger.warning("Cross-encoder chargement échoué : %s", e)
                cls._reranker = None
            cls._reranker_loaded = True
        return cls._reranker

    # ── Pipeline principal ────────────────────────────────────────────────────

    def retrieve(
        self,
        current_code:      str,
        neighborhood:      Dict[str, Any],
        current_file_name: str = "",
        networkx_graph     = None,
    ) -> Tuple[List[Document], List[float]]:
        """
        Retourne (docs, scores) fusionnés depuis toutes les sources.

        Étapes :
          1. Construire toutes les queries (A + B + C)
          2. Rechercher dans ChromaDB + project_code_index (passe 1)
          3. Fusionner et dédupliquer → top-20 candidats
          4. Cross-encoder reranking → top-8 finals (passe 2)
          5. Fallback L2 si reranker absent ou erreur
        """

        # ── Source A : queries structurelles ──────────────────────────────────
        queries = self._build_queries(current_code, neighborhood)

        # ── Sources B + C : KG queries ────────────────────────────────────────
        kg_queries   = []
        nhop_queries = []

        if self._kg:
            parsed_entities = neighborhood.get("_parsed_entities", [])

            # Détecter les patterns présents dans le code
            detected_with_scores = self._kg.detect_patterns(
                current_code, self.language,
                parsed_entities=parsed_entities,
            )
            detected_patterns = [n for n, _ in detected_with_scores]

            # Injecter dans neighborhood pour _build_rerank_query()
            neighborhood["_detected_patterns"] = detected_patterns

            # Source B : expand_queries depth=2
            kg_queries = self._kg.expand_queries(
                detected_nodes=detected_with_scores,
                language=self.language,
                depth=2,
            )

            # Source C : n_hop_retrieval depth=3
            if current_file_name:
                nhop_queries = self._kg.n_hop_retrieval(
                    modified_file=current_file_name,
                    networkx_graph=networkx_graph,
                    language=self.language,
                    depth=3,
                )

            # Affichage console
            if detected_patterns:
                print(f"   • KG patterns : {detected_patterns}")
            total_kg = len(kg_queries) + len(nhop_queries)
            if total_kg:
                print(
                    f"   • KG queries  : {len(kg_queries)} expand"
                    f" + {len(nhop_queries)} n-hop"
                )

        # ── Fusion des queries (sans doublons) ────────────────────────────────
        all_queries = queries.copy()
        for q in kg_queries + nhop_queries:
            if q not in all_queries:
                all_queries.append(q)

        # ── Passe 1 : recherche ChromaDB (large) ─────────────────────────────
        # k=12 par query (au lieu de 8) pour alimenter le reranker avec plus de candidats
        seen: Dict[str, Tuple[Document, float]] = {}

        for query in all_queries:
            if not query.strip():
                continue
            try:
                results = self.vs.similarity_search_with_score(query, k=12)
            except Exception as e:
                logger.debug("ChromaDB search erreur pour query '%s': %s", query[:50], e)
                continue

            # Afficher les scores de la première query (diagnostic)
            if results and query == all_queries[0]:
                scores_str = ", ".join(f"{s:.3f}" for _, s in results[:4])
                print(f"   • Scores RAG : [{scores_str}] (seuil={self.THRESHOLD})")

            for doc, score in results:
                if score > self.THRESHOLD:
                    continue
                meta = doc.metadata
                key  = (
                    "kb:"
                    + meta.get("source_file", "")
                    + str(meta.get("chunk_index", hash(doc.page_content)))
                )
                # Garder le meilleur score pour chaque chunk unique
                if key not in seen or score < seen[key][1]:
                    seen[key] = (doc, score)

        # ── Source D : project_code_index ────────────────────────────────────
        if self.project_code_indexer:
            try:
                project_results = self.project_code_indexer.search(
                    query        = current_code[:600],
                    k            = 4,
                    exclude_file = current_file_name,
                    threshold    = self.THRESHOLD,
                )
                for doc, score in project_results:
                    meta = doc.metadata
                    key  = (
                        "proj:"
                        + meta.get("source_file", "")
                        + meta.get("entity_name", "")
                        + str(meta.get("start_line", ""))
                    )
                    if key not in seen or score < seen[key][1]:
                        seen[key] = (doc, score)
            except Exception as e:
                logger.debug("ProjectCodeIndexer search erreur : %s", e)

        if not seen:
            return [], []

        # ── Tri intermédiaire → top-20 candidats ─────────────────────────────
        # Critères : boost langage > boost KB > score L2
        candidates = sorted(
            seen.values(),
            key=lambda pair: (
                0 if pair[0].metadata.get("language", "") == self.language else 1,
                0 if pair[0].metadata.get("collection", "") != "project_code" else 1,
                pair[1],
            ),
        )[: self.TOP_K_CANDIDATES]

        # ── Passe 2 : Cross-encoder reranking ────────────────────────────────
        reranker = self._get_reranker()

        if reranker is not None and len(candidates) > self.TOP_K:
            top_docs, top_scores = self._rerank(
                reranker, candidates, current_code, neighborhood
            )
            if top_docs:
                print(
                    f"   • Reranking   : {len(candidates)} → {len(top_docs)} docs"
                    f" (cross-encoder actif)"
                )
                return top_docs, top_scores
            # Si _rerank retourne vide (erreur interne) → fallback

        # ── Fallback : tri L2 simple ──────────────────────────────────────────
        top = candidates[: self.TOP_K]
        return [d for d, _ in top], [s for _, s in top]

    # ── Reranking ─────────────────────────────────────────────────────────────

    def _rerank(
        self,
        reranker,
        candidates: List[Tuple[Document, float]],
        current_code: str,
        neighborhood: Dict[str, Any],
    ) -> Tuple[List[Document], List[float]]:
        """
        Applique le cross-encoder sur les candidats de la passe 1.

        Score final = 0.7 × ce_score_normalisé + 0.3 × l2_score_inversé

        Retourne ([], []) si le reranker échoue → le caller bascule en fallback L2.
        """
        try:
            rerank_query = self._build_rerank_query(current_code, neighborhood)

            # Paires (query, contenu_chunk) — limité à 400 chars par chunk
            pairs = [
                (rerank_query, doc.page_content[:400])
                for doc, _ in candidates
            ]

            # Prédiction cross-encoder (scores bruts, non bornés)
            ce_scores = reranker.predict(pairs)

            combined = []
            for i, (doc, l2_score) in enumerate(candidates):
                ce_score = float(ce_scores[i])

                # Normaliser le score cross-encoder avec sigmoid → [0, 1]
                ce_norm = 1.0 / (1.0 + math.exp(-ce_score))

                # Inverser le score L2 → [0, 1] (0 = loin, 1 = proche)
                l2_norm = 1.0 - min(l2_score / self.THRESHOLD, 1.0)

                # Score final pondéré
                final_score = 0.7 * ce_norm + 0.3 * l2_norm
                combined.append((doc, l2_score, final_score))

            # Trier par score final décroissant (plus pertinent en premier)
            combined.sort(key=lambda x: -x[2])

            top_docs   = [d for d, _, _ in combined[: self.TOP_K]]
            top_scores = [s for _, s, _ in combined[: self.TOP_K]]

            logger.debug(
                "Cross-encoder reranking : %d candidats → %d docs retenus",
                len(candidates), len(top_docs),
            )
            return top_docs, top_scores

        except Exception as e:
            logger.debug(
                "Cross-encoder reranking erreur : %s — fallback L2", e
            )
            return [], []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_rerank_query(
        self,
        current_code: str,
        neighborhood: Dict[str, Any],
    ) -> str:
        """
        Construit une query enrichie pour le cross-encoder.
        Plus précise qu'une simple requête de code brut :
          - patterns KG détectés (SqlInjection, ResourceLeak...)
          - langage
          - début du code (300 chars)
        """
        parts = []

        # Patterns KG détectés → signal très fort pour le reranker
        patterns = neighborhood.get("_detected_patterns", [])
        if patterns:
            parts.append("vulnerability patterns: " + ", ".join(patterns[:6]))

        # Langage — aide le reranker à filtrer les règles du bon langage
        parts.append(f"language: {self.language}")

        # Début du code — contexte global
        code_snippet = current_code[:300].strip()
        if code_snippet:
            parts.append(code_snippet)

        return " | ".join(parts) if parts else current_code[:400]

    def _build_queries(
        self,
        current_code: str,
        neighborhood: Dict[str, Any],
    ) -> List[str]:
        """
        Construit les queries structurelles (Source A) :
          Query 1 : code brut (800 premiers chars)
          Query 2 : signatures des méthodes des appelants (predecessors)
          Query 3 : signatures des méthodes des dépendances (successors)
        """
        queries = [current_code[:800]]

        # Query 2 : signatures des appelants
        pred_entities = neighborhood.get("predecessor_entities", {})
        if pred_entities:
            sigs = [
                f"{e['name']}({e['params']})" if e.get("params") else e["name"]
                for ents in pred_entities.values()
                for e in ents[:5]
            ]
            if sigs:
                queries.append(
                    f"caller methods {self.language}: "
                    + ", ".join(sigs[:15])
                )

        # Query 3 : signatures des dépendances
        succ_entities = neighborhood.get("successor_entities", {})
        if succ_entities:
            sigs = [
                f"{e['name']}({e['params']})" if e.get("params") else e["name"]
                for ents in succ_entities.values()
                for e in ents[:5]
            ]
            if sigs:
                queries.append(
                    f"dependency interface {self.language}: "
                    + ", ".join(sigs[:15])
                )

        return queries


# ─────────────────────────────────────────────────────────────────────────────
# RetrieverAgent — façade publique
# ─────────────────────────────────────────────────────────────────────────────

class RetrieverAgent:
    """
    Façade publique pour tout ce qui concerne la récupération de contexte.

    Usage depuis l'Orchestrateur :
        neighborhood = retriever_agent.get_neighborhood(file_path)
        docs, scores = retriever_agent.retrieve_system_aware(
            code, neighborhood, file_name, graph
        )
    """

    def __init__(self):
        self._extractor:            Optional[GraphNeighborhoodExtractor] = None
        self._vector_store                                               = None
        self._project_code_indexer                                       = None
        self._knowledge_graph                                            = None

    def initialize(
        self,
        graph,
        project_indexer,
        vector_store,
        project_code_indexer = None,
        knowledge_graph      = None,
    ):
        """Appelé par l'Orchestrateur après la construction du graphe."""
        self._extractor            = GraphNeighborhoodExtractor(graph, project_indexer)
        self._vector_store         = vector_store
        self._project_code_indexer = project_code_indexer
        self._knowledge_graph      = knowledge_graph

    def get_neighborhood(self, file_path: Path) -> Dict[str, Any]:
        """Retourne le voisinage complet (predecessors, successors, indirect, entities)."""
        if self._extractor is None:
            return GraphNeighborhoodExtractor.empty()
        return self._extractor.get_neighborhood(file_path)

    def retrieve_system_aware(
        self,
        current_code:      str,
        neighborhood:      Dict[str, Any],
        current_file_name: str,
        networkx_graph,
    ) -> Tuple[List[Document], List[float]]:
        """
        Retourne (docs, scores) RAG enrichis par le graphe de dépendances et le KG.
        Pipeline 2 passes : ChromaDB multi-query → cross-encoder reranking.
        """
        if self._vector_store is None:
            return [], []

        rag = SystemAwareRAG(
            vector_store         = self._vector_store,
            language             = neighborhood.get("language", "unknown"),
            project_code_indexer = self._project_code_indexer,
            knowledge_graph      = self._knowledge_graph,
        )
        return rag.retrieve(
            current_code      = current_code,
            neighborhood      = neighborhood,
            current_file_name = current_file_name,
            networkx_graph    = networkx_graph,
        )


# Instance globale
retriever_agent = RetrieverAgent()