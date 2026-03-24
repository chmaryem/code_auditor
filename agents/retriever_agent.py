"""
RetrieverAgent — Recherche RAG consciente du système.


  - GraphNeighborhoodExtractor : extrait le voisinage NetworkX d'un fichier
    (predecessors = appelants, successors = dépendances, impact indirect)
  - SystemAwareRAG             : 3 + N recherches ChromaDB fusionnées
    (code brut + signatures appelants + signatures dépendances + KG queries)
  - RetrieverAgent             : façade publique qui orchestre les deux

"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from config import config as _cfg

logger = logging.getLogger(__name__)


# GraphNeighborhoodExtractor 

class GraphNeighborhoodExtractor:
    """
    Interroge le graphe NetworkX pour extraire le voisinage complet d'un fichier.

    Sources combinées :
      1. NetworkX predecessors/successors (imports résolus — certains)
      2. ProjectIndexer.get_related_files() (convention de nommage — filet de sécurité)

    Retourne :
      predecessors       : fichiers qui utilisent ce fichier (risque de casse)
      successors         : fichiers que ce fichier utilise
      indirect_impacted  : predecessors des predecessors (profondeur 2)
      predecessor_entities / successor_entities : {fichier: [{name, params, criticality}]}
      criticality        : nb de predecessors directs

  
    """

    def __init__(self, graph, project_indexer):
        self.graph           = graph
        self.project_indexer = project_indexer

    def get_neighborhood(self, file_path: Path) -> Dict[str, Any]:
        node_id = f"file:{file_path}"

        # Source 1 : NetworkX
        if self.graph.has_node(node_id):
            pred_paths_nx = [n.replace("file:", "") for n in self.graph.predecessors(node_id)
                             if n.startswith("file:")]
            succ_paths_nx = [n.replace("file:", "") for n in self.graph.successors(node_id)
                             if n.startswith("file:")]
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

    def _collect_entities_rich(self, file_paths: List[str]) -> Dict[str, List[Dict]]:
        """Collecte les entités enrichies (name + params + criticality) depuis project_indexer."""
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
                if len(raw_params) > 6: params_str += ", ..."
                rich.append({"name": e.get("name", ""), "params": params_str,
                             "criticality": file_crit})
            if rich:
                result[file_name] = rich
        return result

    @staticmethod
    def empty() -> Dict[str, Any]:
        return {"predecessors": [], "successors": [], "indirect_impacted": [],
                "predecessor_entities": {}, "successor_entities": {}, "criticality": 0}


# SystemAwareRAG 

class SystemAwareRAG:
    """
    RAG conscient du système.

    Pipeline :
      1. Queries structurelles (code brut + signatures appelants + signatures dépendances)
      2. KG expand_queries (depth=2) — nœuds détectés → règles connexes
      3. KG n_hop_retrieval (depth=3) — NetworkX + KG → règles N-hop
      4. Project code index — code projet similaire
      Fusion + déduplication + tri final.

    """

    # Distance L2 max retenue (Jina embeddings).
    # Chargé depuis config.rag.rag_l2_threshold au __init__.
    # NE PAS confondre avec config.rag.relevance_threshold (cosinus) — métriques différentes.
    THRESHOLD = 1.2
    TOP_K     = 8

    def __init__(self, vector_store, language: str, project_code_indexer=None,
                 knowledge_graph=None):
        self.vs                   = vector_store
        self.language             = language.lower()
        self.project_code_indexer = project_code_indexer
        self._kg                  = knowledge_graph
        # Charger le seuil L2 depuis config (modifiable sans toucher au code)
        try:
            self.THRESHOLD = _cfg.rag.rag_l2_threshold
        except Exception:
            pass  # garder la valeur de classe par défaut (1.2)
        if self._kg and not self._kg._built:
            self._kg.build()

    def retrieve(self, current_code: str, neighborhood: Dict[str, Any],
                 current_file_name: str = "", networkx_graph=None) -> Tuple[list, list]:
        """Retourne (docs, scores) fusionnés depuis toutes les sources."""
        # A : Queries structurelles
        queries = self._build_queries(current_code, neighborhood)

        # B : KG expand_queries
        kg_queries, nhop_queries = [], []
        if self._kg:
            parsed_entities = neighborhood.get("_parsed_entities", [])
            detected_with_scores = self._kg.detect_patterns(
                current_code, self.language, parsed_entities=parsed_entities)
            detected_patterns = [n for n, _ in detected_with_scores]
            kg_queries = self._kg.expand_queries(
                detected_nodes=detected_with_scores, language=self.language, depth=2)

            # C : N-Hop Retrieval
            if current_file_name:
                nhop_queries = self._kg.n_hop_retrieval(
                    modified_file=current_file_name, networkx_graph=networkx_graph,
                    language=self.language, depth=3)

            if detected_patterns:
                print(f"   • KG patterns : {detected_patterns}")
            total_kg = len(kg_queries) + len(nhop_queries)
            if total_kg:
                print(f"   • KG queries  : {len(kg_queries)} expand + {len(nhop_queries)} n-hop")

        # Fusionner sans doublons
        all_queries = queries.copy()
        for q in kg_queries + nhop_queries:
            if q not in all_queries:
                all_queries.append(q)

        seen: Dict[str, Tuple[Any, float]] = {}

        # Recherche KB règles
        for query in all_queries:
            if not query.strip(): continue
            results = self.vs.similarity_search_with_score(query, k=self.TOP_K)
            if results and query == all_queries[0]:
                scores_str = ", ".join(f"{s:.3f}" for _, s in results[:4])
                print(f"   • Scores RAG : [{scores_str}] (seuil={self.THRESHOLD})")
            for doc, score in results:
                if score > self.THRESHOLD: continue
                meta = doc.metadata
                key  = "kb:" + meta.get("source_file","") + str(meta.get("chunk_index", hash(doc.page_content)))
                if key not in seen or score < seen[key][1]:
                    seen[key] = (doc, score)

        # Recherche code projet
        if self.project_code_indexer:
            project_results = self.project_code_indexer.search(
                query=current_code[:600], k=4,
                exclude_file=current_file_name, threshold=self.THRESHOLD)
            for doc, score in project_results:
                meta = doc.metadata
                key  = "proj:" + meta.get("source_file","") + meta.get("entity_name","") + str(meta.get("start_line",""))
                if key not in seen or score < seen[key][1]:
                    seen[key] = (doc, score)

        if not seen:
            return [], []

        ranked = sorted(seen.values(), key=lambda pair: (
            0 if pair[0].metadata.get("language","") == self.language else 1,
            0 if pair[0].metadata.get("collection","") != "project_code" else 1,
            pair[1],
        ))[:self.TOP_K]

        return [d for d, _ in ranked], [s for _, s in ranked]

    def _build_queries(self, current_code: str, neighborhood: Dict[str, Any]) -> List[str]:
        queries = [current_code[:800]]
        pred_entities = neighborhood.get("predecessor_entities", {})
        if pred_entities:
            sigs = [f"{e['name']}({e['params']})" if e.get("params") else e["name"]
                    for ents in pred_entities.values() for e in ents[:5]]
            if sigs:
                queries.append(f"caller methods {self.language}: " + ", ".join(sigs[:15]))
        succ_entities = neighborhood.get("successor_entities", {})
        if succ_entities:
            sigs = [f"{e['name']}({e['params']})" if e.get("params") else e["name"]
                    for ents in succ_entities.values() for e in ents[:5]]
            if sigs:
                queries.append(f"dependency interface {self.language}: " + ", ".join(sigs[:15]))
        return queries


#RetrieverAgent — façade publique 

class RetrieverAgent:
    """
    Façade publique pour tout ce qui concerne la récupération de contexte.

    Usage depuis l'Orchestrateur :
        neighborhood = retriever_agent.get_neighborhood(file_path)
        docs, scores = retriever_agent.retrieve_system_aware(
            code, neighborhood, file_name, graph)
    """

    def __init__(self):
        self._extractor: Optional[GraphNeighborhoodExtractor] = None
        self._vector_store  = None
        self._project_code_indexer = None
        self._knowledge_graph      = None

    def initialize(self, graph, project_indexer, vector_store,
                   project_code_indexer=None, knowledge_graph=None):
        """Appelé par l'Orchestrateur après la construction du graphe."""
        self._extractor = GraphNeighborhoodExtractor(graph, project_indexer)
        self._vector_store         = vector_store
        self._project_code_indexer = project_code_indexer
        self._knowledge_graph      = knowledge_graph

    def get_neighborhood(self, file_path: Path) -> Dict[str, Any]:
        """Retourne le voisinage complet (predecessors, successors, indirect, entities)."""
        if self._extractor is None:
            return GraphNeighborhoodExtractor.empty()
        return self._extractor.get_neighborhood(file_path)

    def retrieve_system_aware(self, current_code: str, neighborhood: Dict[str, Any],
                              current_file_name: str, networkx_graph) -> Tuple[list, list]:
        """Retourne (docs, scores) RAG enrichis par le graphe de dépendances et le KG."""
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


retriever_agent = RetrieverAgent()