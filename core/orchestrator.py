"""
Orchestrator — Chef d'orchestre du projet. Remplace IncrementalAnalyzer.

Contient le pipeline complet migré depuis incremental_analyzer.py :
  - initialize()      : construit graphe, KG, indexeur, branche les agents
  - queue_analysis()  : enfile une analyse (thread-safe)
  - _worker_loop()    : thread de fond qui consomme la queue
  - _analyze_file()   : pipeline complet en 11 étapes (System-Aware)
  - _handle_deletion(): nettoyage quand un fichier est supprimé
  - stop()            : arrêt propre + statistiques

Architecture :
  Orchestrator appelle les agents dans l'ordre :
    code_agent       → parse + analyze_change
    graph_service    → update_graph
    retriever_agent  → get_neighborhood + retrieve_system_aware
    analysis_agent   → build_context + analyze (LLM)
    cache_service    → update
    console_renderer → print_results
    learning_agent   → collect_feedback (async)
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Point d'entrée unique pour toute demande d'analyse.

    Usage :
        orch = Orchestrator(project_path)
        orch.initialize()
        orch.handle(file_changed_event(Path("fichier.py")))
        orch.stop()
    """

    def __init__(self, project_path: Path):
        self.project_path    = project_path

        # Services / agents (initialisés dans initialize())
        self._cache          = None
        self._dependency_graph = None
        self._project_indexer  = None

        # Worker
        self._queue          = Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._is_running     = False
        self._print_lock     = threading.Lock()
        self._last_hash: Dict[str, str] = {}
        self._file_contents: Dict[str, str] = {}
        # Fichiers en cours d'analyse — évite les doublons dans la queue
        self._in_progress: set = set()
        self._in_progress_lock = threading.Lock()
        # Fichiers qui ont reçu une solution full_class cette session
        # Format : {str(file_path): content_hash_de_la_solution}
        # Quand ce fichier est sauvegardé, on injecte un flag "solution déjà générée"
        # pour que Gemini ne produise pas une seconde réécriture complète.
        self._solution_applied: Dict[str, str] = {}

        self._stats = {
            "analyzed":      0,
            "skipped_hash":  0,
            "skipped_minor": 0,
            "time_total":    0.0,
            "by_type":       {},
        }

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialize(self):
        """
        Initialise tous les composants et branche les agents.
        Migré depuis IncrementalAnalyzer.initialize()
        """
        from services.cache_service    import CacheService
        from services.graph_service    import dependency_builder
        from services.project_indexer  import get_project_index
        from services.knowledge_graph  import knowledge_graph
        from services.knowledge_loader import ProjectCodeIndexer
        from services.llm_service      import assistant_agent
        from agents.retriever_agent    import retriever_agent
        from agents.analysis_agent     import analysis_agent
        from agents.learning_agent     import learning_agent

        print(" Initialisation System-Aware RAG...")

        # Cache SQLite
        self._cache = CacheService()

        # Index projet + graphe de dépendances
        print(" Indexation du projet...")
        self._project_indexer  = get_project_index(self.project_path)
        self._dependency_graph = dependency_builder.build_from_project(self.project_path)

        # ProjectCodeIndexer (réutilise les embeddings déjà chargés)
        self._project_code_indexer = ProjectCodeIndexer(embeddings=assistant_agent.embeddings)
        n_chunks = self._project_code_indexer.index_project(self.project_path)
        print(f" ProjectCodeIndexer : {n_chunks} chunks dans ChromaDB")

        # Knowledge Graph
        print(" Knowledge Graph RAG...")
        knowledge_graph.build(
            project_indexer  = self._project_indexer,
            dependency_graph = self._dependency_graph,
            llm              = assistant_agent.llm,
        )
        kg_n = knowledge_graph._graph.number_of_nodes()
        kg_e = knowledge_graph._graph.number_of_edges()
        print(f" KG : {kg_n} concepts, {kg_e} relations")
        print(f" Graphe : {self._dependency_graph.number_of_nodes()} nœuds, "
              f"{self._dependency_graph.number_of_edges()} arêtes")

        # Brancher RetrieverAgent
        retriever_agent.initialize(
            graph                = self._dependency_graph,
            project_indexer      = self._project_indexer,
            vector_store         = assistant_agent.vector_store,
            project_code_indexer = self._project_code_indexer,
            knowledge_graph      = knowledge_graph,
        )
        self._retriever_agent = retriever_agent

        # Brancher AnalysisAgent
        analysis_agent.set_llm_service(assistant_agent)
        self._analysis_agent = analysis_agent

        # Brancher LearningAgent
        try:
            from services.knowledge_loader import KnowledgeBaseLoader
            kb_loader = KnowledgeBaseLoader()
            learning_agent.initialize(
                llm             = assistant_agent.llm,
                vector_store    = assistant_agent.vector_store,
                kb_dir          = kb_loader.kb_dir,
                kb_loader       = kb_loader,
                knowledge_graph = knowledge_graph,
            )
            learning_agent.start()
            print(" Self-Improving RAG activé — fixes validés → KB enrichie")
        except Exception as e:
            logger.debug("LearningAgent non initialisé : %s", e)
        self._learning_agent = learning_agent

        print(" System-Aware RAG + Knowledge Graph + Self-Improving activés\n")

        # Démarrer le worker
        self._is_running    = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    # ── Interface publique ────────────────────────────────────────────────────

    def handle(self, event):
        """Reçoit un Event et l'enfile."""
        from core.events import EventType
        if event.type in (EventType.FILE_CHANGED, EventType.MANUAL_ANALYZE):
            self.queue_analysis(event.file_path, deleted=False)
        elif event.type == EventType.FILE_DELETED:
            self.queue_analysis(event.file_path, deleted=True)
        elif event.type == EventType.GIT_COMMIT:
            for fi in event.payload.get("changed_files", []):
                p = Path(fi.get("path", ""))
                if p.exists():
                    self.queue_analysis(p, deleted=False)

    def queue_analysis(self, file_path: Path, deleted: bool = False):
        if deleted:
            self._queue.put({"file_path": file_path, "deleted": True})
            return
        # Bloquer si ce fichier est déjà dans la queue ou en cours d'analyse
        file_key = str(file_path)
        with self._in_progress_lock:
            if file_key in self._in_progress:
                logger.debug("Doublon ignoré (déjà en cours) : %s", file_path.name)
                return
            if self._cache and not self._cache.has_file_changed(file_path):
                return
            self._in_progress.add(file_key)
        self._queue.put({"file_path": file_path, "deleted": False})

    def stop(self):
        print("\n Arrêt...")
        self._is_running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        if self._cache:
            self._cache.save()

        # LearningAgent.stop() déclenche flush_session() → bilan KB affiché ici
        if hasattr(self, "_learning_agent"):
            self._learning_agent.stop()

        print(f"\n Statistiques d'analyse :")
        print(f"   Analysés      : {self._stats['analyzed']}")
        print(f"   Ignorés hash  : {self._stats['skipped_hash']}")
        print(f"   Ignorés mineur: {self._stats['skipped_minor']}")
        if self._stats["analyzed"] > 0:
            avg = self._stats["time_total"] / self._stats["analyzed"]
            print(f"   Temps moyen   : {avg:.1f}s")
        if self._stats["by_type"]:
            print(f"   Par type      : {self._stats['by_type']}")
        if hasattr(self, "_learning_agent"):
            kb = self._learning_agent.get_stats()
            total_kb = kb.get("auto_promoted", 0) + kb.get("batch_promoted", 0)
            if total_kb:
                print(f"   KB enrichie   : {total_kb} règle(s) cette session")

    # ── Worker ────────────────────────────────────────────────────────────────

    def _worker_loop(self):
        while self._is_running:
            try:
                task = self._queue.get(timeout=1)
                if task.get("deleted"):
                    self._handle_deletion(task["file_path"])
                else:
                    self._analyze_file(task["file_path"])
                self._queue.task_done()
            except Empty:
                pass
            except Exception as e:
                logger.error("Worker erreur : %s", e)

    # ── Suppression ───────────────────────────────────────────────────────────

    def _handle_deletion(self, file_path: Path):
        print(f"\n {file_path.name} supprimé")
        if self._cache:
            self._cache.remove_file_from_cache(file_path)
        node_id = f"file:{file_path}"
        if self._dependency_graph and self._dependency_graph.has_node(node_id):
            self._dependency_graph.remove_node(node_id)
        if self._cache:
            self._cache.save()
        print()

    # ── Pipeline principal (11 étapes) ────────────────────────────────────────

    def _analyze_file(self, file_path: Path):
        """
        Pipeline System-Aware complet — migré depuis IncrementalAnalyzer._analyze_file()

        Étape 1  : Hash check         → ignore si fichier inchangé
        Étape 2  : Lecture            → contenu brut
        Étape 3  : ChangeAnalyzer     → filtre changements mineurs (RAS, 0 token)
        Étape 4  : Parsing AST        → code_agent.parse()
        Étape 4.5: ProjectCodeIndexer → mise à jour non-bloquante (timeout 4s)
        Étape 4.6: KG update          → mise à jour incrémentale
        Étape 5  : update_graph       → graph_service.update_graph()
        Étape 6  : Voisinage          → retriever_agent.get_neighborhood()
        Étape 7  : SystemAwareRAG     → retriever_agent.retrieve_system_aware()
        Étape 8  : Contexte           → analysis_agent.build_context()
        Étape 9  : LLM                → analysis_agent.analyze()
        Étape 10 : Cache + affichage
        Étape 11 : Feedback           → learning_agent.collect_feedback()
        """
        from agents.code_agent     import code_agent
        from agents.analysis_agent import build_context
        from services.graph_service import update_graph
        from services.knowledge_graph import knowledge_graph
        from output.console_renderer import print_results, print_minor_change, parse_fix_blocks, print_targeted_methods, print_solution, _GR, _R, _CY, _DM

        start = time.time()
        print(f"\n{'─'*70}")
        print(f" {file_path.name}")

        # ── ÉTAPE 1 : Hash check ──────────────────────────────────────────────
        if self._cache and not self._cache.has_file_changed(file_path):
            print("  Ignoré : Hash identique\n")
            self._stats["skipped_hash"] += 1
            return

        # ── ÉTAPE 2 : Lecture ─────────────────────────────────────────────────
        try:
            new_content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f" Erreur lecture : {e}\n")
            with self._in_progress_lock:
                self._in_progress.discard(str(file_path))
            return

        # file_key défini ici — utilisé dans étape 8.6, affichage et cache
        file_key = str(file_path)

        # ── ÉTAPE 3 : Filtre intelligent — changements mineurs ────────────────
        old_content = self._file_contents.get(str(file_path), "")
        change_info = code_agent.analyze_change(old_content, new_content)
        print(f" {change_info['reason']} (score: {change_info['score']}/100)")

        if not change_info["significant"]:
            print_minor_change(change_info["reason"])
            self._stats["skipped_minor"] += 1
            self._stats["by_type"][change_info["change_type"]] = (
                self._stats["by_type"].get(change_info["change_type"], 0) + 1)
            if self._cache:
                self._cache.update_file_cache(
                    file_path, {"analysis": "RAS — changement mineur", "relevant_knowledge": []}, [], [])
            self._file_contents[str(file_path)] = new_content
            with self._in_progress_lock:
                self._in_progress.discard(str(file_path))
            return

        print(" Analyse System-Aware lancée...")

        # ── ÉTAPE 4 : Parsing AST ─────────────────────────────────────────────
        parsed = code_agent.parse(file_path)
        if parsed.get("error"):
            print(f" Erreur parsing : {parsed['error']}\n"); return
        entities = len(parsed.get("entities", []))
        imports  = len(parsed.get("imports",  []))
        print(f"   • {entities} entité(s), {imports} import(s)")

        # ── ÉTAPE 4.5 : ProjectCodeIndexer (non-bloquant, timeout 4s) ────────
        if self._project_code_indexer:
            ex = ThreadPoolExecutor(max_workers=1)
            try:
                future = ex.submit(self._project_code_indexer.index_file,
                                   file_path, new_content, parsed.get("entities", []))
                future.result(timeout=4)
            except FuturesTimeout:
                logger.debug("ProjectCodeIndexer timeout — pipeline non bloqué")
            except Exception as e:
                logger.debug("ProjectCodeIndexer ignoré pour %s : %s", file_path.name, e)
            finally:
                ex.shutdown(wait=False)

        # ── ÉTAPE 4.6 : KG update incrémental ────────────────────────────────
        try:
            knowledge_graph.update_file(
                file_path=file_path, project_indexer=self._project_indexer, llm=None)
        except Exception as e:
            logger.debug("KG update_file ignoré pour %s : %s", file_path.name, e)

        # ── ÉTAPE 5 : Mise à jour graphe NetworkX ─────────────────────────────
        update_graph(self._dependency_graph, file_path, parsed)

        # ── ÉTAPE 6 : Voisinage ───────────────────────────────────────────────
        neighborhood = self._retriever_agent.get_neighborhood(file_path)
        preds    = neighborhood["predecessors"]
        succs    = neighborhood["successors"]
        indirect = neighborhood["indirect_impacted"]
        crit     = neighborhood["criticality"]

        if preds:
            pred_names = [Path(p).name for p in preds[:3]]
            extra = f" +{len(preds)-3}" if len(preds) > 3 else ""
            print(f" ⚠️  Appelé par : {', '.join(pred_names)}{extra}  ({crit} dépendant(s))")
        if succs:
            print(f" 🔗 Utilise     : {', '.join(Path(p).name for p in succs[:3])}")
        if indirect:
            print(f" 📡 Impact indirect : {len(indirect)} fichier(s)")
        print(f"{'🔴' if crit > 5 else '🟡' if crit > 0 else '🟢'} Criticité : {crit}")

        # ── ÉTAPE 7 : SystemAwareRAG ──────────────────────────────────────────
        language = code_agent.detect_language(file_path)
        print(f" RAG System-Aware ({language})...", flush=True)

        # Injecter les entités pour detect_patterns (project_indexer prioritaire)
        file_info = (self._project_indexer.context.files.get(str(file_path), {})
                     if self._project_indexer and self._project_indexer.context else {})
        neighborhood["_parsed_entities"] = (
            file_info.get("entities", []) or parsed.get("entities", []))
        neighborhood["language"] = language

        relevant_docs, rag_scores = self._retriever_agent.retrieve_system_aware(
            current_code      = new_content,
            neighborhood      = neighborhood,
            current_file_name = file_path.name,
            networkx_graph    = self._dependency_graph,
        )
        print(f"   • {len(relevant_docs)} chunk(s) RAG (KB rules + project code)")

        # ── ÉTAPE 8 : Contexte enrichi ────────────────────────────────────────
        context = build_context(
            file_path       = file_path,
            neighborhood    = neighborhood,
            project_indexer = self._project_indexer,
            change_info     = change_info,
        )

        # ── ÉTAPE 8.5 : Lire l'ancien résultat (pour le delta) ─────────────
        previous_analysis = ""
        if self._cache:
            cached = self._cache.get_cached_analysis(file_path)
            if cached and cached.get("analysis"):
                previous_analysis = cached["analysis"]

        # ── ÉTAPE 8.6 : Injecter flag post-solution si ce fichier en a eu une ──
        # Si une solution full_class a été générée pour ce fichier dans cette session,
        # on force Gemini à ne faire que des block_fix résiduels — pas de nouvelle
        # réécriture complète qui créerait une boucle infinie.
        if file_key in self._solution_applied:
            context["post_solution_mode"] = True
            context["post_solution_hint"] = (
                "A complete full_class solution was ALREADY generated for this file "
                "in this session. The developer just saved that solution. "
                "DO NOT generate another full_class rewrite. "
                "Use ONLY block_fix for any small residual issues you find. "
                "If the code is substantially correct, choose block_fix with 0-2 issues max."
            )
            print(f"  {_DM}↩  Mode post-solution — block_fix uniquement{_R}", flush=True)

        # ── ÉTAPE 9 : LLM ────────────────────────────────────────────────────
        print(" Analyse LLM (System-Aware)...", flush=True)
        analysis = self._analysis_agent.analyze(
            code=new_content, context=context,
            docs=relevant_docs, scores=rag_scores)

        # ── ÉTAPE 10 : Cache ──────────────────────────────────────────────────
        if self._cache:
            self._cache.update_file_cache(
                file_path, analysis,
                context.get("dependencies", []),
                context.get("dependents",   []))
        self._file_contents[str(file_path)] = new_content

        # ── Affichage ─────────────────────────────────────────────────────────
        elapsed     = time.time() - start
        result_text = analysis["analysis"]
        result_hash = hashlib.md5(result_text.encode("utf-8", errors="replace")).hexdigest()

        self._stats["analyzed"]   += 1
        self._stats["time_total"] += elapsed
        self._stats["by_type"][change_info["change_type"]] = (
            self._stats["by_type"].get(change_info["change_type"], 0) + 1)

        if self._last_hash.get(file_key) == result_hash:
            return  # même résultat déjà affiché (watchdog double-fire)
        self._last_hash[file_key] = result_hash

        # Parser la réponse agentique : Gemini a décidé la stratégie
        from agents.analysis_agent import parse_llm_response
        parsed = parse_llm_response(result_text)
        strategy = parsed["strategy"]

        with self._print_lock:
            if strategy == "full_class":
                print(f"  Strategy : {_CY}full_class{_R} — {parsed['reason'][:80]}")
                print_solution(
                    solution_text = result_text,
                    file_name     = file_path.name,
                    changes       = parsed["payload"].get("changes", []),
                    language      = language,
                    elapsed       = elapsed,
                    analyzed_count= self._stats["analyzed"],
                    score         = change_info["score"],
                    impacted      = preds,
                )
                # Mémoriser que ce fichier vient de recevoir une solution complète.
                # Le prochain save déclenchera une analyse en mode "post-solution"
                # (block_fix seulement, pas de nouvelle réécriture complète).
                code_block = parsed["payload"].get("class_code", "")
                if code_block:
                    self._solution_applied[file_key] = hashlib.md5(
                        code_block.encode("utf-8", errors="replace")
                    ).hexdigest()
                # Afficher aussi les blocs fix s'il y en a dans la réponse
                remaining = parsed["payload"].get("remaining_blocks", [])
                if remaining:
                    print_results(
                        text=result_text, file_name=file_path.name,
                        context=context, elapsed=elapsed,
                        analyzed_count=self._stats["analyzed"],
                        score=change_info["score"], impacted=preds,
                        previous_analysis=previous_analysis,
                    )

            elif strategy == "targeted_methods":
                methods = parsed["payload"].get("methods", [])
                print(f"  Strategy : {_CY}targeted_methods{_R} ({len(methods)} méthode(s)) — {parsed['reason'][:80]}")
                print_targeted_methods(
                    methods    = methods,
                    file_name  = file_path.name,
                    remaining  = parsed["payload"].get("remaining_blocks", []),
                    elapsed    = elapsed,
                    analyzed_count = self._stats["analyzed"],
                    score      = change_info["score"],
                    impacted   = preds,
                    previous_analysis = previous_analysis,
                )

            else:  # block_fix
                print(f"  Strategy : {_CY}block_fix{_R} — {parsed['reason'][:80]}" if parsed['reason'] else "")
                print_results(
                    text=result_text, file_name=file_path.name,
                    context=context, elapsed=elapsed,
                    analyzed_count=self._stats["analyzed"],
                    score=change_info["score"], impacted=preds,
                    previous_analysis=previous_analysis,
                )

        # ── ÉTAPE 11 : Self-Improving RAG ─────────────────────────────────────
        fix_blocks = parse_fix_blocks(result_text)
        if hasattr(self, "_learning_agent") and self._learning_agent:
            if fix_blocks:
                try:
                    self._learning_agent.collect_feedback(
                        blocks           = fix_blocks,
                        code_before      = new_content,
                        language         = language,
                        file_name        = file_path.name,
                        project_indexer  = self._project_indexer,
                        dependency_graph = self._dependency_graph,
                    )
                except Exception as e:
                    logger.debug("Feedback Loop erreur : %s", e)

        # Libérer le verrou — ce fichier peut maintenant être re-analysé
        with self._in_progress_lock:
            self._in_progress.discard(str(file_path))