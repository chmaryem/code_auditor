"""
test_generator_agent.py — Génération on-demand de tests unitaires (thin wrapper).

Architecture v3 (LangGraph multi-agents) :
  Ce module n'est plus qu'un point d'entrée. Le pipeline en 15 étapes a été
  refactoré en un vrai StateGraph LangGraph de 6 nœuds agents :

    Discovery → Analysis → RAG → Generation → Validation → Execution

  Voir langchain_agents/graphs/test_gen_graph.py (topologie + edges conditionnels)
  et langchain_agents/graphs/test_gen_nodes/ (implémentation des nœuds).
  Toute la logique métier vit VERBATIM dans test_gen_nodes/_helpers.py — extraite
  telle quelle depuis la v2 de ce fichier, sans modification de comportement.

  La signature publique generate_for_file(source_path, write, incremental) et le
  dict retourné sont STRICTEMENT identiques à la v2 : aucune régression observable
  depuis l'API /generate-tests ni depuis l'extension VS Code.

Historique (v2, désormais réparti entre nœuds) :
  1. Discovery   → convention de test + framework détecté (+ cache Redis in)
  2. RAG         → ChromaDB test_patterns_kb + ProjectCodeIndexer + Knowledge Graph
  3. Generation  → LLM direct via invoke_with_fallback() (redaction secrets)
  4. Validation  → compile() / validation structurelle par langage
  5. Execution   → run tests (pytest/mvn/jest) + retry runtime + cache Redis out
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from services.test_discovery import TestDiscoveryService
from services.test_runner import TestRunner

logger = logging.getLogger(__name__)


class TestGeneratorAgent:
    """
    Agent de génération de tests unitaires avec RAG sémantique.

    Thin wrapper : instancie les services partagés et délègue à la TestGenGraph.
    """

    def __init__(
        self,
        project_path: Path,
        test_kb=None,
        project_code_indexer=None,
        knowledge_graph=None,
    ):
        """
        Args:
            project_path: Racine du projet à tester.
            test_kb: TestKnowledgeLoader — collection ChromaDB des patterns de test.
            project_code_indexer: ProjectCodeIndexer — code réel du projet (tests existants).
            knowledge_graph: KnowledgeGraph — relations entre entités pour context.
        """
        self.project_path = project_path
        self._discovery = TestDiscoveryService(project_path)
        self._test_kb = test_kb
        self._project_code_indexer = project_code_indexer
        self._knowledge_graph = knowledge_graph
        self._runner = TestRunner(project_path)

    # ── API publique ─────────────────────────────────────────────────────────

    def generate_for_file(
        self,
        source_path: Path,
        write: bool = False,
        incremental: bool = False,
    ) -> Dict[str, Any]:
        """
        Génère un fichier de test pour `source_path` via la TestGenGraph.

        Args:
            source_path: Fichier source à tester.
            write: Écrire le résultat sur disque.
            incremental: Si True et qu'un fichier de test existe déjà,
                génère uniquement les méthodes manquantes et les insère
                à la fin du fichier existant plutôt que de l'écraser.

        Returns:
            {
                "test_file":   Path or None,
                "test_code":   str,
                "framework":   str,
                "error":       str or None,
                "rag_docs_used": int,
                "validated":   bool,
                "incremental": bool,
                "run_success": bool or None,
                "run_error_summary": str or None,
            }
        """
        if not source_path.exists():
            return {"error": f"Fichier introuvable : {source_path}", "test_file": None}

        from langchain_agents.graphs.test_gen_graph import generate_tests_via_graph

        return generate_tests_via_graph(
            source_path=source_path,
            project_path=self.project_path,
            write=write,
            incremental=incremental,
            discovery=self._discovery,
            runner=self._runner,
            test_kb=self._test_kb,
            project_code_indexer=self._project_code_indexer,
            knowledge_graph=self._knowledge_graph,
        )
