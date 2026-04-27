"""
Orchestrator — Chef d'orchestre du projet (Architecture Async v2).

Architecture v2 (asyncio) :
  - L'event loop asyncio tourne dans un thread daemon unique
  - asyncio.PriorityQueue remplace threading.Queue (FIFO → priorité)
  - _analyze_file() est une coroutine async (LLM via asyncio.to_thread)
  - _analyze_dependents() utilise asyncio.gather() pour paralléliser
  - Debounce coalesce : N événements en <1s → 1 seul batch
  - Cancellation : si un fichier est re-modifié, l'analyse en cours est annulée

L'API publique reste identique :
  orch = Orchestrator(project_path)
  orch.initialize()
  orch.handle(file_changed_event(Path("fichier.py")))
  orch.stop()
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Délai de coalesce : on attend ce temps après le dernier événement reçu
# avant de lancer le batch d'analyses.
_COALESCE_DELAY = 0.8  # secondes


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

        # ── Async infrastructure ──────────────────────────────────────────
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._queue: Optional[asyncio.PriorityQueue] = None
        self._is_running     = False
        self._print_lock     = threading.Lock()

        # Cancellation : {file_key: asyncio.Task}
        self._pending_tasks: Dict[str, asyncio.Task] = {}

        # Debounce : contenu précédent et hashes
        self._last_hash: Dict[str, str] = {}
        self._file_contents: Dict[str, str] = {}

        # Fichiers qui ont reçu une solution full_class cette session
        self._solution_applied: Dict[str, str] = {}

        # Nombre max de dépendants analysés après chaque save
        self._max_deps_per_analysis: int = 2

        # Compteur monotone pour le tri FIFO à priorité égale
        self._seq = 0
        self._seq_lock = threading.Lock()

        self._stats = {
            "analyzed":      0,
            "skipped_hash":  0,
            "skipped_minor": 0,
            "cancelled":     0,
            "time_total":    0.0,
            "by_type":       {},
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Initialisation
    # ══════════════════════════════════════════════════════════════════════════

    def initialize(self):
        """
        Initialise tous les composants et branche les agents.
        Démarre l'event loop asyncio dans un thread daemon.
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
        # Construire un LLM compatible (SemanticLinker utilise .invoke() optionnellement)
        from services.llm_factory import build_llm_cascade_for_agent
        _cascade = build_llm_cascade_for_agent(temperature=0.0, max_tokens=256)
        _kg_llm  = _cascade[0][1] if _cascade else None  # premier provider disponible
        knowledge_graph.build(
            project_indexer  = self._project_indexer,
            dependency_graph = self._dependency_graph,
            llm              = _kg_llm,
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
            from services.llm_factory import build_llm_cascade_for_agent
            kb_loader = KnowledgeBaseLoader()
            _la_cascade = build_llm_cascade_for_agent(temperature=0.0, max_tokens=2048)
            _la_llm = _la_cascade[0][1] if _la_cascade else None
            learning_agent.initialize(
                llm             = _la_llm,
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

        print(" System-Aware RAG + Knowledge Graph + Self-Improving activés")
        print(" Architecture Async activée (asyncio.PriorityQueue)\n")

        # ── Démarrer l'event loop async dans un thread daemon ─────────────
        self._is_running = True
        self._loop_thread = threading.Thread(
            target=self._run_event_loop, daemon=True, name="orch-async-loop"
        )
        self._loop_thread.start()

        # Attendre que l'event loop soit prête
        for _ in range(50):  # max 5s
            if self._loop is not None and self._loop.is_running():
                break
            time.sleep(0.1)

    # ══════════════════════════════════════════════════════════════════════════
    # Interface publique (thread-safe, synchrone)
    # ══════════════════════════════════════════════════════════════════════════

    def handle(self, event):
        """Reçoit un Event et l'enfile dans la queue async (thread-safe)."""
        from core.events import EventType
        if event.type in (EventType.FILE_CHANGED, EventType.MANUAL_ANALYZE):
            self.queue_analysis(event.file_path, deleted=False, priority=event.priority)
        elif event.type == EventType.FILE_DELETED:
            self.queue_analysis(event.file_path, deleted=True, priority=event.priority)
        elif event.type == EventType.GIT_COMMIT:
            for fi in event.payload.get("changed_files", []):
                if isinstance(fi, dict):
                    raw_path = fi.get("path", "")
                    status   = fi.get("status", "M")
                elif isinstance(fi, str):
                    raw_path = fi
                    status   = "M"
                else:
                    continue

                if not raw_path or status == "D":
                    continue

                p = Path(raw_path)
                if not p.is_absolute():
                    p = event.payload.get("repo_path", Path(".")) / p
                if p.exists():
                    self.queue_analysis(p, deleted=False, priority=event.priority)

    def queue_analysis(self, file_path: Path, deleted: bool = False,
                       priority: int = 50):
        """
        Poste un événement dans la queue async (thread-safe).
        Annule l'analyse en cours pour ce fichier si elle existe.
        """
        if not self._loop or not self._loop.is_running():
            logger.warning("Event loop non démarrée — événement ignoré : %s", file_path)
            return

        file_key = str(file_path)

        # ── Cancellation : annuler l'analyse en cours pour ce fichier ─────
        if file_key in self._pending_tasks:
            old_task = self._pending_tasks[file_key]
            if not old_task.done():
                self._loop.call_soon_threadsafe(old_task.cancel)
                self._stats["cancelled"] += 1
                logger.info("Analyse annulée (nouveau changement) : %s", file_path.name)

        # ── Compteur FIFO pour l'ordre à priorité égale ───────────────────
        with self._seq_lock:
            seq = self._seq
            self._seq += 1

        # ── Enqueue thread-safe ───────────────────────────────────────────
        task_item = {
            "file_path": file_path,
            "deleted": deleted,
        }

        def _put():
            if self._queue is not None:
                self._queue.put_nowait((priority, seq, task_item))

        self._loop.call_soon_threadsafe(_put)

    def stop(self):
        """Arrêt propre : annule les tâches, arrête l'event loop, affiche les stats."""
        print("\n Arrêt...")
        self._is_running = False

        # Annuler toutes les tâches async en cours
        if self._loop and self._loop.is_running():
            for file_key, task in list(self._pending_tasks.items()):
                if not task.done():
                    self._loop.call_soon_threadsafe(task.cancel)

            # Arrêter l'event loop
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._loop_thread:
            self._loop_thread.join(timeout=5)

        if self._cache:
            self._cache.save()

        # LearningAgent.stop() déclenche flush_session()
        if hasattr(self, "_learning_agent"):
            self._learning_agent.stop()

        print(f"\n Statistiques d'analyse :")
        print(f"   Analysés      : {self._stats['analyzed']}")
        print(f"   Ignorés hash  : {self._stats['skipped_hash']}")
        print(f"   Ignorés mineur: {self._stats['skipped_minor']}")
        print(f"   Annulés       : {self._stats['cancelled']}")
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

    # ══════════════════════════════════════════════════════════════════════════
    # Event Loop (thread daemon)
    # ══════════════════════════════════════════════════════════════════════════

    def _run_event_loop(self):
        """Crée et démarre l'event loop asyncio dans ce thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._queue = asyncio.PriorityQueue()

        try:
            self._loop.run_until_complete(self._async_worker())
        except Exception as e:
            if self._is_running:
                logger.error("Event loop crash : %s", e)
        finally:
            # Nettoyage
            try:
                pending = asyncio.all_tasks(self._loop)
                for t in pending:
                    t.cancel()
                if pending:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:
                pass
            self._loop.close()

    async def _async_worker(self):
        """
        Worker async principal avec debounce coalesce.

        Stratégie :
          1. Attendre le premier événement de la queue
          2. Après réception, attendre _COALESCE_DELAY pour collecter les suivants
          3. Dédupliquer par fichier (le dernier événement gagne)
          4. Lancer les analyses en parallèle via asyncio.create_task()
        """
        while self._is_running:
            # ── Étape 1 : Attendre le premier événement ───────────────────
            batch: Dict[str, dict] = {}
            try:
                priority, seq, task = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
                file_key = str(task["file_path"])
                batch[file_key] = task
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # ── Étape 2 : Coalesce — collecter les événements suivants ────
            await asyncio.sleep(_COALESCE_DELAY)

            while not self._queue.empty():
                try:
                    p, s, extra_task = self._queue.get_nowait()
                    extra_key = str(extra_task["file_path"])
                    batch[extra_key] = extra_task  # Le dernier gagne
                except asyncio.QueueEmpty:
                    break

            # ── Étape 3 : Traiter le batch ────────────────────────────────
            if len(batch) > 1:
                logger.info("Coalesce : %d fichier(s) dans ce batch", len(batch))

            for file_key, task in batch.items():
                if not self._is_running:
                    break

                file_path = task["file_path"]

                if task.get("deleted"):
                    self._handle_deletion(file_path)
                else:
                    # Vérifier le cache avant de lancer l'analyse
                    if self._cache and not self._cache.has_file_changed(file_path):
                        self._stats["skipped_hash"] += 1
                        continue

                    # Annuler l'ancienne tâche si elle existe
                    if file_key in self._pending_tasks:
                        old = self._pending_tasks[file_key]
                        if not old.done():
                            old.cancel()
                            self._stats["cancelled"] += 1

                    # Lancer la nouvelle analyse comme tâche async
                    coro = self._analyze_file(file_path)
                    t = asyncio.create_task(coro, name=f"analyze:{file_path.name}")
                    self._pending_tasks[file_key] = t

    # ══════════════════════════════════════════════════════════════════════════
    # Suppression
    # ══════════════════════════════════════════════════════════════════════════

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

    # ══════════════════════════════════════════════════════════════════════════
    # Pipeline principal (async) — 11 étapes
    # ══════════════════════════════════════════════════════════════════════════

    async def _analyze_file(self, file_path: Path):
        """
        Pipeline System-Aware complet — version async.

        Les étapes CPU-bound (hash, lecture, parsing) restent synchrones (rapides).
        Les étapes I/O-bound (LLM, RAG reranker) sont wrappées dans asyncio.to_thread().
        """
        from agents.code_agent     import code_agent
        from agents.analysis_agent import build_context
        from services.graph_service import update_graph
        from services.knowledge_graph import knowledge_graph
        from output.console_renderer import (
            print_results, print_minor_change, parse_fix_blocks,
            print_targeted_methods, print_solution, _GR, _R, _CY, _DM
        )

        start = time.time()
        file_key = str(file_path)

        print(f"\n{'─'*70}")
        print(f" {file_path.name}")

        # ── ÉTAPE 1 : Hash check ──────────────────────────────────────────
        if self._cache and not self._cache.has_file_changed(file_path):
            print("  Ignoré : Hash identique\n")
            self._stats["skipped_hash"] += 1
            return

        # ── ÉTAPE 2 : Lecture ─────────────────────────────────────────────
        try:
            new_content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f" Erreur lecture : {e}\n")
            return

        # ── ÉTAPE 3 : Filtre intelligent — changements mineurs ────────────
        old_content = self._file_contents.get(file_key, "")
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
            self._file_contents[file_key] = new_content
            return

        print(" Analyse System-Aware lancée...")

        # ── Checkpoint cancellation ───────────────────────────────────────
        await asyncio.sleep(0)  # Yield pour vérifier si la tâche est annulée

        # ── ÉTAPE 4 : Parsing AST ─────────────────────────────────────────
        parsed = await asyncio.to_thread(code_agent.parse, file_path)
        if isinstance(parsed, str):
            parsed = {"error": parsed, "entities": [], "imports": []}
        if parsed.get("error"):
            print(f" Erreur parsing : {parsed['error']}\n"); return
        entities = len(parsed.get("entities", []))
        imports  = len(parsed.get("imports",  []))
        print(f"   • {entities} entité(s), {imports} import(s)")

        # ── ÉTAPE 4.5 : ProjectCodeIndexer (non-bloquant, timeout 4s) ────
        if self._project_code_indexer:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        self._project_code_indexer.index_file,
                        file_path, new_content, parsed.get("entities", [])
                    ),
                    timeout=4.0,
                )
            except asyncio.TimeoutError:
                logger.debug("ProjectCodeIndexer timeout — pipeline non bloqué")
            except Exception as e:
                logger.debug("ProjectCodeIndexer ignoré pour %s : %s", file_path.name, e)

        # ── ÉTAPE 4.6 : KG update incrémental ────────────────────────────
        try:
            await asyncio.to_thread(
                knowledge_graph.update_file,
                file_path=file_path, project_indexer=self._project_indexer, llm=None
            )
        except Exception as e:
            logger.debug("KG update_file ignoré pour %s : %s", file_path.name, e)

        # ── ÉTAPE 5 : Mise à jour graphe NetworkX ─────────────────────────
        update_graph(self._dependency_graph, file_path, parsed)

        # ── Checkpoint cancellation ───────────────────────────────────────
        await asyncio.sleep(0)

        # ── ÉTAPE 6 : Voisinage ───────────────────────────────────────────
        neighborhood = await asyncio.to_thread(
            self._retriever_agent.get_neighborhood, file_path
        )
        if isinstance(neighborhood, str):
            neighborhood = {}
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

        # ── ÉTAPE 7 : SystemAwareRAG ──────────────────────────────────────
        language = code_agent.detect_language(file_path)

        file_info = {}
        if (self._project_indexer and self._project_indexer.context and
            hasattr(self._project_indexer.context, 'files') and
            isinstance(self._project_indexer.context.files, dict)):
            cached_value = self._project_indexer.context.files.get(str(file_path), {})
            file_info = cached_value if isinstance(cached_value, dict) else {}

        neighborhood["_parsed_entities"] = (
            file_info.get("entities", []) or parsed.get("entities", []))
        neighborhood["language"] = language

        # ── Checkpoint cancellation ───────────────────────────────────────
        await asyncio.sleep(0)

        # RAG retrieval (I/O-bound : reranker CPU + ChromaDB)
        relevant_docs, rag_scores = await asyncio.to_thread(
            self._retriever_agent.retrieve_system_aware,
            current_code      = new_content,
            neighborhood      = neighborhood,
            current_file_name = file_path.name,
            networkx_graph    = self._dependency_graph,
        )
        print(f"   • {len(relevant_docs)} chunk(s) RAG (KB rules + project code)")

        # ── ÉTAPE 8 : Contexte enrichi ────────────────────────────────────
        context = build_context(
            file_path       = file_path,
            neighborhood    = neighborhood,
            project_indexer = self._project_indexer,
            change_info     = change_info,
        )
        if isinstance(context, str):
            context = {}

        # ── ÉTAPE 8.5 : Lire l'ancien résultat (pour le delta) ───────────
        previous_analysis = ""
        if self._cache:
            cached = self._cache.get_cached_analysis(file_path)
            if cached and cached.get("analysis"):
                previous_analysis = cached["analysis"]

        # ── ÉTAPE 8.6 : Flag post-solution ────────────────────────────────
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

        # ── Checkpoint cancellation ───────────────────────────────────────
        await asyncio.sleep(0)

        # ── ÉTAPE 9 : LLM (le plus long — wrappé dans to_thread) ─────────
        analysis = await asyncio.to_thread(
            self._analysis_agent.analyze,
            code=new_content, context=context,
            docs=relevant_docs, scores=rag_scores
        )
        if isinstance(analysis, str):
            analysis = {"analysis": analysis, "relevant_knowledge": [], "validated_blocks": []}

        # ── ÉTAPE 10 : Cache ──────────────────────────────────────────────
        if self._cache:
            self._cache.update_file_cache(
                file_path, analysis,
                context.get("dependencies", []),
                context.get("dependents",   []))
        self._file_contents[file_key] = new_content

        # ── Affichage ─────────────────────────────────────────────────────
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

        # Parser la réponse agentique : stratégie décidée par Gemini
        from agents.analysis_agent import parse_llm_response
        parsed_resp = parse_llm_response(result_text)
        strategy = parsed_resp["strategy"]

        with self._print_lock:
            if strategy == "full_class":
                print(f"  Strategy : {_CY}full_class{_R} — {parsed_resp['reason'][:80]}")
                print_solution(
                    solution_text = result_text,
                    file_name     = file_path.name,
                    changes       = parsed_resp["payload"].get("changes", []),
                    language      = language,
                    elapsed       = elapsed,
                    analyzed_count= self._stats["analyzed"],
                    score         = change_info["score"],
                    impacted      = preds,
                )
                code_block = parsed_resp["payload"].get("class_code", "")
                if code_block:
                    self._solution_applied[file_key] = hashlib.md5(
                        code_block.encode("utf-8", errors="replace")
                    ).hexdigest()
                remaining = parsed_resp["payload"].get("remaining_blocks", [])
                if remaining:
                    print_results(
                        text=result_text, file_name=file_path.name,
                        context=context, elapsed=elapsed,
                        analyzed_count=self._stats["analyzed"],
                        score=change_info["score"], impacted=preds,
                        previous_analysis=previous_analysis,
                    )

            elif strategy == "targeted_methods":
                methods = parsed_resp["payload"].get("methods", [])
                print(f"  Strategy : {_CY}targeted_methods{_R} ({len(methods)} méthode(s)) — {parsed_resp['reason'][:80]}")
                print_targeted_methods(
                    methods    = methods,
                    file_name  = file_path.name,
                    remaining  = parsed_resp["payload"].get("remaining_blocks", []),
                    elapsed    = elapsed,
                    analyzed_count = self._stats["analyzed"],
                    score      = change_info["score"],
                    impacted   = preds,
                    previous_analysis = previous_analysis,
                )

            else:  # block_fix
                print(f"  Strategy : {_CY}block_fix{_R} — {parsed_resp['reason'][:80]}" if parsed_resp['reason'] else "")
                print_results(
                    text=result_text, file_name=file_path.name,
                    context=context, elapsed=elapsed,
                    analyzed_count=self._stats["analyzed"],
                    score=change_info["score"], impacted=preds,
                    previous_analysis=previous_analysis,
                )

        # ── ÉTAPE 11 : Self-Improving RAG ────────────────────────────────
        fix_blocks = parse_fix_blocks(result_text)

        # Mémoire épisodique
        if fix_blocks and self._cache:
            session_id = getattr(self, "_session_id", None)
            for block in fix_blocks:
                if not isinstance(block, dict):
                    logger.debug("Bloc ignoré : type inattendu %s", type(block).__name__)
                    continue
                pattern_type = self._extract_pattern_type(block)
                try:
                    self._cache.record_pattern(
                        file_path    = str(file_path),
                        pattern_type = pattern_type,
                        severity     = block.get("severity", "MEDIUM"),
                        session_id   = session_id,
                    )
                except Exception as e:
                    logger.debug("record_pattern erreur : %s", e)

        # Patterns récurrents
        if self._cache:
            try:
                recurring = self._cache.get_recurring_patterns(
                    str(file_path), min_count=2
                )
                if recurring:
                    self._print_recurring_warning(file_path.name, recurring)
            except Exception as e:
                logger.debug("get_recurring_patterns erreur : %s", e)

        # LearningAgent (non-bloquant)
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

        # ── ÉTAPE 12 : Analyse proactive des dépendants (async) ───────────
        if preds and not context.get("post_solution_mode"):
            await self._analyze_dependents(
                dependents        = preds,
                changed_file_name = file_path.name,
                change_summary    = result_text[:1500],
                language          = language,
            )

        # Nettoyage de la tâche
        self._pending_tasks.pop(file_key, None)

    # ══════════════════════════════════════════════════════════════════════════
    # Analyse proactive des dépendants (async + gather)
    # ══════════════════════════════════════════════════════════════════════════

    async def _analyze_dependents(
        self,
        dependents:        List[str],
        changed_file_name: str,
        change_summary:    str,
        language:          str,
    ) -> None:
        """
        Analyse les dépendants en parallèle via asyncio.gather().
        Chaque dépendant est traité comme une coroutine indépendante.
        """
        tasks = []
        for dep_path_str in dependents[:self._max_deps_per_analysis]:
            dep_path = Path(dep_path_str)
            if not dep_path.exists():
                continue
            tasks.append(
                self._analyze_single_dependent(
                    dep_path, changed_file_name, change_summary, language
                )
            )

        if not tasks:
            return

        results = await asyncio.gather(*tasks, return_exceptions=True)

        analyzed_count = sum(1 for r in results if r is True)
        for r in results:
            if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError):
                logger.error("Erreur analyse dépendant : %s", r)

        if analyzed_count > 0:
            print(f"\n  ✓ {analyzed_count} dépendant(s) analysé(s) suite au changement de {changed_file_name}\n")

    async def _analyze_single_dependent(
        self,
        dep_path:          Path,
        changed_file_name: str,
        change_summary:    str,
        language:          str,
    ) -> bool:
        """Analyse un seul dépendant. Retourne True si l'analyse a réussi."""
        from agents.code_agent     import code_agent
        from agents.analysis_agent import build_context
        from output.console_renderer import print_results, parse_fix_blocks, _CY, _R, _DM

        try:
            dep_content = dep_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.debug("Lecture dépendant %s : %s", dep_path.name, e)
            return False

        print(f"\n{'─'*70}")
        print(f" 🔗 Dépendant : {dep_path.name}  ← {changed_file_name} vient de changer")

        # Parsing
        parsed_dep = await asyncio.to_thread(code_agent.parse, dep_path)
        if isinstance(parsed_dep, str):
            parsed_dep = {"entities": [], "imports": []}
        if parsed_dep.get("error"):
            logger.debug("Parsing dépendant %s : %s", dep_path.name, parsed_dep["error"])
            return False

        # Voisinage + RAG
        neighborhood_dep = await asyncio.to_thread(
            self._retriever_agent.get_neighborhood, dep_path
        )
        if isinstance(neighborhood_dep, str):
            neighborhood_dep = {}
        dep_lang = code_agent.detect_language(dep_path)
        neighborhood_dep["language"] = dep_lang
        neighborhood_dep["_parsed_entities"] = parsed_dep.get("entities", [])

        # Checkpoint cancellation
        await asyncio.sleep(0)

        relevant_docs, rag_scores = await asyncio.to_thread(
            self._retriever_agent.retrieve_system_aware,
            current_code      = dep_content,
            neighborhood      = neighborhood_dep,
            current_file_name = dep_path.name,
            networkx_graph    = self._dependency_graph,
        )

        # Contexte enrichi avec info upstream
        context_dep = build_context(
            file_path       = dep_path,
            neighborhood    = neighborhood_dep,
            project_indexer = self._project_indexer,
        )
        if isinstance(context_dep, str):
            context_dep = {}

        context_dep["upstream_change"] = (
            f"IMPORTANT: {changed_file_name} was just refactored. "
            f"Verify that THIS file still compiles and works correctly with it.\n"
            f"Summary of changes in {changed_file_name}:\n{change_summary[:800]}"
        )
        context_dep["post_solution_mode"] = True
        context_dep["post_solution_hint"] = (
            f"{changed_file_name} was refactored. Check for: broken imports, "
            f"wrong method calls, signature mismatches, compilation errors caused "
            f"by the upstream change. Use block_fix only. "
            f"If everything is compatible, say so clearly with 0 fix blocks."
        )

        print(f"  {_DM}↩  Mode compatibilité — vérifie l'impact de {changed_file_name}{_R}")

        # Checkpoint cancellation
        await asyncio.sleep(0)

        # Analyse LLM (le gros morceau — via to_thread)
        try:
            analysis_dep = await asyncio.to_thread(
                self._analysis_agent.analyze,
                code    = dep_content,
                context = context_dep,
                docs    = relevant_docs,
                scores  = rag_scores,
            )
        except Exception as e:
            logger.error("Analyse dépendant %s : %s", dep_path.name, e)
            return False

        if isinstance(analysis_dep, str):
            analysis_dep = {"analysis": analysis_dep, "relevant_knowledge": [], "validated_blocks": []}

        result_text_dep = analysis_dep.get("analysis", "")

        # Cache
        if self._cache:
            self._cache.update_file_cache(dep_path, analysis_dep, [], [])

        # Affichage
        with self._print_lock:
            print_results(
                text           = result_text_dep,
                file_name      = dep_path.name,
                context        = context_dep,
                elapsed        = 0.0,
                analyzed_count = self._stats["analyzed"],
                score          = 0,
                impacted       = [],
            )

        # Self-Improving RAG pour le dépendant
        fix_blocks_dep = parse_fix_blocks(result_text_dep)
        if fix_blocks_dep and hasattr(self, "_learning_agent") and self._learning_agent:
            try:
                self._learning_agent.collect_feedback(
                    blocks      = fix_blocks_dep,
                    code_before = dep_content,
                    language    = dep_lang,
                    file_name   = dep_path.name,
                )
            except Exception as e:
                logger.debug("Feedback dépendant %s : %s", dep_path.name, e)

        return True

    # ══════════════════════════════════════════════════════════════════════════
    # Utilitaires
    # ══════════════════════════════════════════════════════════════════════════

    def _extract_pattern_type(self, block: dict) -> str:
        """
        Extrait un type de pattern normalisé depuis le problem text.
        Ex: "SQL injection in findByUsername" → "SqlInjection"
        """
        problem = block.get("problem", "").lower()

        patterns = {
            "sql injection":         "SqlInjection",
            "prepared statement":    "SqlInjection",
            "resource leak":         "ResourceLeak",
            "try-with-resources":    "ResourceLeak",
            "resultset":             "ResourceLeak",
            "plain text password":   "PlainTextPassword",
            "bcrypt":                "PlainTextPassword",
            "undeclared":            "UndeclaredVariable",
            "null pointer":          "NullPointer",
            "single responsibility": "SRPViolation",
            "pagination":            "MissingPagination",
            "n+1":                   "N1Query",
        }

        for keyword, pattern_name in patterns.items():
            if keyword in problem:
                return pattern_name

        import re
        return re.sub(r'[^a-zA-Z0-9]', '', problem[:40].title())

    def _print_recurring_warning(
        self,
        file_name: str,
        recurring: list,
    ):
        """Affiche un avertissement pour les patterns récurrents."""
        from output.console_renderer import _YL, _RD, _R, _BD, _DM

        print(f"\n  {_YL}⟳ Patterns récurrents dans {file_name} :{_R}")
        for p in recurring[:3]:
            color = _RD if p["severity"] == "CRITICAL" else _YL
            days_ago = ""
            try:
                from datetime import datetime
                first = datetime.fromisoformat(p["first_seen"])
                delta = (datetime.now() - first).days
                if delta > 0:
                    days_ago = f" depuis {delta} jour(s)"
            except Exception:
                pass

            kb_tag = f" {_DM}[dans KB]{_R}" if p["in_kb"] else ""
            print(
                f"    {color}• {p['pattern']}{_R}"
                f" — {_BD}{p['count']}x{_R}{days_ago}{kb_tag}"
            )