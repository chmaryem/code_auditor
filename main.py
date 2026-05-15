"""
main.py — Point d'entrée unique du projet Code Auditor AI.

Commandes disponibles :
  python main.py file    <fichier>        # analyser un seul fichier
  python main.py project <dossier>        # analyser un projet complet
  python main.py watch   <dossier>        # surveillance temps réel
  python main.py git     <dossier>        # analyser le dernier commit Git
  python main.py hook    <dossier>        # installer le pre-commit hook Git

  # Smart Git Merge
  python main.py resolve-conflicts <dossier>       # résoudre les conflits locaux via LLM
  python main.py merge-hook <dossier>               # installer le pre-merge hook

  # MCP Code Mode (agents autonomes)
  python main.py pr-check       --repo owner/repo --pr N  # revue PR via agent
  python main.py pr-resolve     --repo owner/repo --pr N  # résoudre conflits PR via agent
  python main.py pr-merge-check --repo owner/repo --pr N  # vérifier readiness merge
"""
import sys
import io
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))



# ── Couleurs console ──────────────────────────────────────────────────────────
G  = "\033[92m"
R  = "\033[91m"
Y  = "\033[93m"
C  = "\033[96m"
B  = "\033[1m"
E  = "\033[0m"

def ok(msg):   print(f"{G}✓ {msg}{E}")
def err(msg):  print(f"{R}✗ {msg}{E}")
def info(msg): print(f"{C}  {msg}{E}")
def hdr(msg):
    print(f"\n{B}{'='*70}{E}")
    print(f"{B}{msg.center(70)}{E}")
    print(f"{B}{'='*70}{E}\n")


# ── Commande : file ───────────────────────────────────────────────────────────

def cmd_file(args):
    """Analyse un seul fichier."""
    from services.llm_service import assistant_agent

    file_path = Path(args.path)
    if not file_path.is_file():
        err(f"Fichier introuvable : {file_path}")
        return

    hdr(f"ANALYSE FICHIER : {file_path.name}")
    info(f"Fichier : {file_path}")

    code = file_path.read_text(encoding="utf-8", errors="replace")
    info(f"Taille  : {len(code)} caractères\n")
    info("Analyse avec RAG en cours...")

    result = assistant_agent.analyze_code_with_rag(
        code    = code,
        context = {
            "file_path": str(file_path),
            "language":  file_path.suffix.replace(".", ""),
        },
    )

    hdr("RÉSULTATS DE L'ANALYSE")
    print(result["analysis"])

    if result.get("relevant_knowledge"):
        hdr("BEST PRACTICES CONSULTÉES")
        for kb in result["relevant_knowledge"]:
            print(f"  • {kb.get('source_file', 'unknown')} ({kb.get('category', '')})")

    ok("Analyse terminée")


# ── Commande : project ────────────────────────────────────────────────────────

def cmd_project(args):
    """Analyse un projet complet."""
    from core.project_analyzer import project_analyzer

    project_path = Path(args.path)
    if not project_path.is_dir():
        err(f"Dossier introuvable : {project_path}")
        return

    hdr(f"ANALYSE PROJET : {project_path.name}")
    results = project_analyzer.analyze_full_project(project_path, args.max_files)

    structure   = results["structure_analysis"]
    entry_pts   = structure.get("entry_points", [])
    cycles      = structure.get("circular_dependencies", [])
    orphans     = structure.get("orphaned_modules", [])
    conflicts   = results.get("conflicts", [])
    file_analyses = results.get("file_analyses", {})

    # Architecture
    hdr("ARCHITECTURE DU PROJET")
    print(f"{B}Points d'entrée :{E}")
    for e in entry_pts[:5]:
        print(f"  • {e.split(':')[-1] if ':' in e else e}")
    if len(entry_pts) > 5:
        print(f"  ... et {len(entry_pts) - 5} autres")

    print(f"\n{B}Dépendances circulaires :{E}")
    if cycles:
        for c in cycles[:3]:
            print(f"  {Y}{'→'.join(x.split(':')[-1] for x in c)}{E}")
    else:
        ok("  Aucune dépendance circulaire")

    print(f"\n{B}Modules orphelins :{E}")
    if orphans:
        for o in orphans[:5]:
            print(f"  {Y}{o.split(':')[-1] if ':' in o else o}{E}")
    else:
        ok("  Aucun module orphelin")

    # Conflits
    if conflicts:
        hdr("CONFLITS DE REFACTORING")
        for i, c in enumerate(conflicts, 1):
            print(f"{B}Conflit #{i}:{E} {c['type']} — {c['severity']}")
            print(f"  {c['message']}\n")
    else:
        ok("\nAucun conflit détecté")

    # Plan
    hdr("PLAN DE REFACTORING")
    print(results.get("refactoring_plan", ""))

    # Analyses par fichier
    hdr("ANALYSES PAR FICHIER")
    for i, (fp, analysis) in enumerate(file_analyses.items(), 1):
        ctx = analysis.get("context", {})
        print(f"\n{B}{'─'*70}{E}")
        print(f"{B}Fichier {i}/{len(file_analyses)} : {Path(fp).name}{E}")
        print(f"{B}{'─'*70}{E}")
        print(f"  Criticité   : {ctx.get('criticality_score', 0)}")
        print(f"  Dépendances : {len(ctx.get('dependencies', []))}")
        print(f"\n{analysis.get('analysis', 'Aucune analyse')}\n")

    # Résumé
    hdr("RÉSUMÉ")
    print(f"Fichiers analysés      : {B}{len(file_analyses)}{E}")
    print(f"Conflits détectés      : {B}{len(conflicts)}{E}")
    print(f"Points d'entrée        : {B}{len(entry_pts)}{E}")
    print(f"Dépendances circulaires: {B}{len(cycles)}{E}")
    if conflicts:
        print(f"\n{Y}  Résolvez les conflits avant d'appliquer les corrections.{E}")
    else:
        ok("\nVous pouvez appliquer les corrections en toute sécurité.")


# ── Commande : watch ──────────────────────────────────────────────────────────

def _setup_langsmith_tracing():
    """Configure LangSmith tracing if enabled in config."""
    from config import config
    if config.langgraph.langsmith_tracing and config.langgraph.langsmith_api_key:
        import os
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = config.langgraph.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = config.langgraph.langsmith_project
        os.environ["LANGSMITH_ENDPOINT"] = config.langgraph.langsmith_endpoint
        info("LangSmith tracing enabled — project: " + config.langgraph.langsmith_project)


def cmd_watch(args):
    """Surveille un projet en temps réel + Smart Git Session Tracker."""
    try:
        import watchdog  # noqa
    except ImportError:
        err("Module 'watchdog' manquant — installe-le : pip install watchdog")
        return

    from watchers.file_watcher import FileWatcher
    from config import config

    project_path = Path(args.path)
    if not project_path.is_dir():
        err(f"Dossier introuvable : {project_path}")
        return

    use_langgraph = getattr(args, 'langgraph', False) or config.langgraph.enabled

    # ── LangGraph mode ────────────────────────────────────────────────────────
    if use_langgraph:
        hdr("MODE WATCH — LangGraph + LangChain Agents")
        info(f"Projet : {project_path}")
        info("Orchestration : LangGraph StateGraph (WatchGraph)")
        info("Agents : LangChain (CodeAgent, RetrieverAgent, AnalysisAgent, LearningAgent)\n")

        _setup_langsmith_tracing()

        # ── Initialize services (same as legacy Orchestrator) ────────────────
        from services.llm_service import CodeRAGSystemAPI
        info("Initialisation System-Aware RAG...")
        rag_system = CodeRAGSystemAPI()

        # Project Indexer
        project_indexer = None
        try:
            from services.project_indexer import get_project_index
            project_indexer = get_project_index(project_path)
            info(f" ProjectIndexer : {project_indexer.context.total_files} fichiers, "
                 f"{project_indexer.context.total_entities} entités")
        except Exception as e:
            import logging, traceback
            logging.getLogger(__name__).warning("ProjectIndexer init failed: %s", e)
            traceback.print_exc()

        # Dependency Graph
        dep_graph = None
        extractor = None
        try:
            from services.graph_service import dependency_builder, DependencyExtractor
            dep_graph = dependency_builder.build_from_project(project_path)
            extractor = DependencyExtractor(dep_graph)
            info(f" Graphe : {dep_graph.number_of_nodes()} nœuds, "
                 f"{dep_graph.number_of_edges()} arêtes")
        except Exception as e:
            import logging, traceback
            logging.getLogger(__name__).warning("DependencyGraph init failed: %s", e)
            traceback.print_exc()

        # Knowledge Graph
        try:
            from services.knowledge_graph import knowledge_graph
            if not knowledge_graph._built:
                knowledge_graph.build(project_indexer=project_indexer)
            info(f" KG : {knowledge_graph._graph.number_of_nodes()} concepts, "
                 f"{knowledge_graph._graph.number_of_edges()} relations")
        except Exception as e:
            import logging, traceback
            logging.getLogger(__name__).warning("KnowledgeGraph init failed: %s", e)
            traceback.print_exc()

        # Self-Improving RAG
        try:
            from agents.learning_agent import feedback_processor
            if feedback_processor:
                info(" Self-Improving RAG activé — fixes validés → KB enrichie")
        except Exception:
            pass

        # Test Gap Detection
        try:
            from services.test_knowledge_loader import test_knowledge_loader
            if test_knowledge_loader:
                info(f" test_patterns_kb : {test_knowledge_loader.chunk_count()} chunks de patterns de test")
                info(" Test Gap Detection activé — surveillance des tests manquants")
        except Exception:
            pass

        # Retriever Agent initialization (for cross-encoder + full RAG)
        try:
            from agents.retriever_agent import retriever_agent
            from services.knowledge_graph import knowledge_graph as kg_instance

            retriever_agent.initialize(
                graph=dep_graph,
                project_indexer=project_indexer,
                vector_store=rag_system.vector_store if rag_system else None,
                project_code_indexer=project_indexer,
                knowledge_graph=kg_instance if kg_instance._built else None,
            )
            info(" System-Aware RAG + Knowledge Graph + Cross-encoder activés")
        except Exception as e:
            import logging, traceback
            logging.getLogger(__name__).warning("RetrieverAgent init failed: %s", e)
            traceback.print_exc()

        info(" Architecture LangGraph activée (StateGraph)\n")

        # ── Smart Git Session Tracker ────────────────────────────────────────
        git_tracker = None
        try:
            from smart_git.git_diff_parser import is_git_repo
            if is_git_repo(project_path):
                from smart_git.git_session_tracker import GitSessionTracker
                from smart_git.git_notifier import GitNotifier
                import threading

                cache_db = config.CACHE_DIR / "analysis_cache.db"
                notifier = GitNotifier(print_lock=threading.Lock())
                git_tracker = GitSessionTracker(
                    project_path=project_path,
                    cache_db=cache_db,
                    notifier=notifier,
                    check_interval=180,
                )
                git_tracker.start()
                info("Smart Git Tracker actif — surveillance accumulation de bugs\n")
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("GitSessionTracker non démarré : %s", e)

        # ── LearningAgent lifecycle ──────────────────────────────────────────
        from agents.learning_agent import learning_agent as _learning_agent
        try:
            _learning_agent.initialize(
                llm=rag_system._llm if rag_system else None,
                vector_store=rag_system.vector_store if rag_system else None,
                kb_dir=config.KNOWLEDGE_BASE_DIR,
                kb_loader=getattr(rag_system, '_kb_loader', None),
                knowledge_graph=None,
            )
            _learning_agent.start()
            info("LearningAgent démarré (Self-Improving RAG)\n")
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("LearningAgent init failed: %s", e)
            _learning_agent = None

        # ── File counter + shared resources ──────────────────────────────────
        import threading
        print_lock = threading.Lock()
        file_counter = {
            "analyzed": 0, "skipped": 0,
            "skipped_hash": 0, "skipped_minor": 0,
            "by_type": {},
        }

        from langchain_agents.graphs.watch_graph import invoke_watch

        # ── Debounce / coalesce wrapper (0.8s) ───────────────────────────────
        _COALESCE_DELAY = 0.8
        _pending_timer = {}   # file_path → Timer
        _pending_lock = threading.Lock()

        def _debounced_invoke(file_path: Path):
            """Execute invoke_watch after coalesce delay expires."""
            with _pending_lock:
                _pending_timer.pop(str(file_path), None)

            result = invoke_watch(
                file_path=str(file_path),
                project_path=str(project_path),
                project_indexer=project_indexer,
                extractor=extractor,
                rag_system=rag_system,
                dep_graph=dep_graph,
                cache=None,
                print_lock=print_lock,
                learning_agent=_learning_agent,
                file_counter=file_counter,
            )
            if result.get("skip_reason"):
                skip = result["skip_reason"]
                file_counter["skipped"] += 1
                if "hash" in str(skip).lower():
                    file_counter["skipped_hash"] += 1
                elif "minor" in str(skip).lower() or "score" in str(skip).lower():
                    file_counter["skipped_minor"] += 1

        def on_change_lg(file_path: Path, deleted: bool = False):
            # ── File deletion handling ───────────────────────────────────
            if deleted:
                info(f"Fichier supprimé : {file_path.name}")
                # Clean dependency graph
                if dep_graph is not None:
                    node_id = f"file:{file_path}"
                    if dep_graph.has_node(node_id):
                        dep_graph.remove_node(node_id)
                        info(f"  Nœud supprimé du graphe de dépendances")
                # Clean Redis cache
                try:
                    from langchain_agents.memory.redis_memory import AgentRedisMemory
                    mem = AgentRedisMemory("code_agent")
                    mem.delete(f"hash:{file_path}")
                    mem.delete(f"content:{file_path}")
                except Exception:
                    pass
                return

            # ── Debounce: cancel previous timer, start new one ───────────
            fp_key = str(file_path)
            with _pending_lock:
                old = _pending_timer.get(fp_key)
                if old:
                    old.cancel()
                timer = threading.Timer(_COALESCE_DELAY, _debounced_invoke, args=[file_path])
                timer.daemon = True
                _pending_timer[fp_key] = timer
                timer.start()

        watcher = FileWatcher(project_path=project_path, callback=on_change_lg)
        try:
            watcher.watch()
        except KeyboardInterrupt:
            pass
        finally:
            watcher.stop()
            if git_tracker:
                git_tracker.stop()
            # Stop LearningAgent (flush pending KB rules)
            if _learning_agent:
                try:
                    _learning_agent.stop()
                except Exception:
                    pass
            # Cancel pending timers
            with _pending_lock:
                for t in _pending_timer.values():
                    t.cancel()
            # ── Detailed stats ───────────────────────────────────────────
            total = file_counter['analyzed'] + file_counter['skipped']
            print(f"\n {'═' * 60}")
            print(f"\n Statistiques d'analyse :")
            print(f"   Analysés           : {file_counter['analyzed']}")
            print(f"   Ignorés (total)    : {file_counter['skipped']}")
            if file_counter['skipped_hash']:
                print(f"     ↳ Même contenu   : {file_counter['skipped_hash']}")
            if file_counter['skipped_minor']:
                print(f"     ↳ Mineur (<20)   : {file_counter['skipped_minor']}")
            if file_counter.get('by_type'):
                print(f"   Par type :")
                for ct, n in file_counter['by_type'].items():
                    print(f"     • {ct:20s} : {n}")
            if _learning_agent:
                la_stats = _learning_agent.get_stats()
                if la_stats.get("received"):
                    print(f"   Self-Improving RAG :")
                    print(f"     Feedbacks reçus  : {la_stats['received']}")
                    print(f"     Règles promues   : {la_stats.get('auto_promoted', 0) + la_stats.get('batch_promoted', 0)}")
            print(f"\n {'═' * 60}\n")
        return

    # ── Legacy mode (Orchestrator) ────────────────────────────────────────────
    from core.orchestrator import Orchestrator
    from core.events import file_changed_event

    hdr("MODE WATCH — SURVEILLANCE TEMPS RÉEL")
    info(f"Projet : {project_path}\n")

    orchestrator = Orchestrator(project_path)
    orchestrator.initialize()

    def on_change(file_path: Path, deleted: bool = False):
        orchestrator.handle(file_changed_event(file_path, deleted=deleted))

    # ── Smart Git Session Tracker ────────────────────────────────────────────
    git_tracker = None
    try:
        from smart_git.git_diff_parser import is_git_repo
        if is_git_repo(project_path):
            from smart_git.git_session_tracker import GitSessionTracker
            from smart_git.git_notifier import GitNotifier

            cache_db    = config.CACHE_DIR / "analysis_cache.db"
            notifier    = GitNotifier(print_lock=orchestrator._print_lock)
            git_tracker = GitSessionTracker(
                project_path   = project_path,
                cache_db       = cache_db,
                notifier       = notifier,
                check_interval = 180,
            )
            git_tracker.start()
            info("Smart Git Tracker actif — surveillance accumulation de bugs\n")
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug("GitSessionTracker non démarré : %s", e)

    watcher = FileWatcher(project_path=project_path, callback=on_change)
    try:
        watcher.watch()
    except KeyboardInterrupt:
        pass
    finally:
        watcher.stop()
        if git_tracker:
            git_tracker.stop()
        orchestrator.stop()


# ── Commande : git status ─────────────────────────────────────────────────────

def cmd_git_status(args):
    """Affiche l'état de la session courante (accumulation de bugs non commités)."""
    from smart_git.git_session_tracker import GitSessionTracker
    from smart_git.git_report import session_report
    from smart_git.git_diff_parser import is_git_repo
    from config import config

    project_path = Path(args.path)
    if not is_git_repo(project_path):
        err(f"Pas un dépôt Git : {project_path}")
        return

    cache_db = config.CACHE_DIR / "analysis_cache.db"
    tracker  = GitSessionTracker(project_path=project_path, cache_db=cache_db)
    snapshot = tracker.force_check()

    if snapshot:
        print(session_report(snapshot))
        if snapshot.files_unanalyzed:
            info(f"{len(snapshot.files_unanalyzed)} fichier(s) sans analyse Watch.")
            info("Conseil : python main.py watch <projet> pour analyser en temps réel.")
    else:
        err("Impossible de calculer le score de session.")


# ── Commande : git branch ──────────────────────────────────────────────────────

def cmd_git_branch(args):
    """Analyse une branche feature vs sa base et donne un verdict de merge."""
    from smart_git.git_branch_analyzer import GitBranchAnalyzer
    from smart_git.git_report import branch_report, save_branch_report_json
    from smart_git.git_diff_parser import is_git_repo
    from config import config

    project_path = Path(args.path)
    if not is_git_repo(project_path):
        err(f"Pas un dépôt Git : {project_path}")
        return

    branch   = getattr(args, "branch", "HEAD")
    base     = getattr(args, "base",   "main")
    save_json = getattr(args, "report", False)

    hdr(f"ANALYSE BRANCHE — {branch} vs {base}")

    cache_db = config.CACHE_DIR / "analysis_cache.db"
    analyzer = GitBranchAnalyzer(project_path=project_path, cache_db=cache_db)

    try:
        report = analyzer.analyze(branch=branch, base=base)
    except RuntimeError as e:
        err(str(e))
        return

    print(branch_report(report))

    if save_json:
        out_dir  = config.CACHE_DIR.parent / "git_reports"
        out_path = save_branch_report_json(report, out_dir)
        ok(f"Rapport JSON sauvegardé : {out_path}")


# ── Commande : git commit ──────────────────────────────────────────────────────

def cmd_git(args):
    """Analyse les fichiers modifiés dans un commit Git donné."""
    from core.orchestrator import Orchestrator
    from core.events import git_commit_event
    from smart_git.git_diff_parser import get_changed_files, get_current_commit_hash, is_git_repo

    project_path = Path(args.path)
    if not is_git_repo(project_path):
        err(f"Pas un dépôt Git : {project_path}")
        return

    commit      = getattr(args, "commit", "HEAD")
    changed     = get_changed_files(commit=commit, project_path=project_path)
    commit_hash = get_current_commit_hash(project_path)

    if not changed:
        ok("Aucun fichier modifié dans ce commit.")
        return

    hdr(f"ANALYSE GIT — commit {commit_hash}")
    info(f"{len(changed)} fichier(s) modifié(s)\n")

    orchestrator = Orchestrator(project_path)
    orchestrator.initialize()
    orchestrator.handle(git_commit_event(commit_hash, changed, repo_path=project_path))
    orchestrator.stop()


# ── Commande : hook ───────────────────────────────────────────────────────────

def cmd_hook(args):
    """Installe ou désinstalle le pre-commit Git hook."""
    from smart_git.git_hook import install_hook, uninstall_hook
    project_path = Path(args.path)
    if getattr(args, "uninstall", False):
        uninstall_hook(project_path)
    else:
        strict = not getattr(args, "no_strict", False)
        install_hook(project_path, strict=strict)


# ── Commande : resolve-conflicts ───────────────────────────────────────────────

def cmd_resolve_conflicts(args):
    """
    Résout les conflits de merge locaux via le LLM.
    Après un 'git merge' qui produit des conflits, cette commande
    envoie chaque fichier en conflit au LLM pour résolution automatique.
    """
    from smart_git.git_conflict_resolver import resolve_all_conflicts
    project_path = Path(args.path).resolve()
    resolve_all_conflicts(project_path)


# ── Commande : merge-hook ──────────────────────────────────────────────────

def cmd_merge_hook(args):
    """
    Installe ou désinstalle le pre-merge-commit hook.
    Ce hook bloque 'git merge' si le code de la branche source
    contient des bugs CRITICAL (score ≥ 35).
    """
    from smart_git.git_merge_hook import install_merge_hook, uninstall_merge_hook
    project_path = Path(args.path).resolve()
    if getattr(args, "uninstall", False):
        uninstall_merge_hook(project_path)
    else:
        install_merge_hook(project_path)


# ── Commande : ci-analyze (analyse manuelle d'un run/PR) ───────────────────────

def cmd_ci_analyze(args):
    """
    Analyse manuelle d'un run GitHub Actions via le CIGraph IA.
    Déclenche immédiatement une analyse complète et poste le commentaire PR.

    Usage :
      python main.py ci-analyze --repo owner/repo --pr 17
      python main.py ci-analyze --repo owner/repo --branch feature/test-hook
      python main.py ci-analyze --repo owner/repo --run-id 25855849812 --pr 17 --project-key sonar_key
    """
    import os, urllib.request, urllib.parse, json as _json
    from pathlib import Path
    from dotenv import load_dotenv
    # Charger .env avec chemin absolu et override=True (force)
    _env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=_env_path, override=True)
    from langchain_agents.graphs.ci_graph import invoke_ci_run

    repo        = args.repo
    owner       = repo.split("/")[0]
    repo_name   = repo.split("/")[-1]
    pr_number   = getattr(args, "pr",          None)
    run_id      = getattr(args, "run_id",      "") or ""
    project_key = getattr(args, "project_key", "") or ""
    branch      = getattr(args, "branch",      "") or ""

    token = (
        os.environ.get("GITHUB_TOKEN")
        or os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    )

    # Debug : afficher les tokens disponibles
    if not token:
        print(f"{Y}[WARN] GITHUB_TOKEN non trouve dans l'environnement.{E}")
        print(f"  Verifiez que votre .env contient : GITHUB_TOKEN=ghp_xxx")
        print(f"  GITHUB_TOKEN={repr(os.environ.get('GITHUB_TOKEN'))} / GITHUB_PERSONAL_ACCESS_TOKEN={repr(os.environ.get('GITHUB_PERSONAL_ACCESS_TOKEN'))}")
        return

    # Auto-detect : si run_id non fourni -> dernier run du repo (ou de la branche)
    if not run_id:
        def _fetch_runs(branch_filter=""):
            params = {"status": "completed", "per_page": "20"}
            if branch_filter:
                params["branch"] = branch_filter
            url = (
                f"https://api.github.com/repos/{owner}/{repo_name}/actions/runs"
                f"?{urllib.parse.urlencode(params)}"
            )
            req = urllib.request.Request(url)
            req.add_header("Authorization", f"token {token}")
            req.add_header("Accept", "application/vnd.github.v3+json")
            with urllib.request.urlopen(req, timeout=15) as r:
                return _json.loads(r.read().decode()).get("workflow_runs", [])

        try:
            runs = []
            # Strategie 1 : chercher par branche base (main) si PR fournie
            if pr_number and branch:
                runs = _fetch_runs("main") or _fetch_runs("")
            elif branch:
                # Les runs CI se font souvent sur main, pas la branche feature
                runs = _fetch_runs(branch)
                if not runs:
                    info(f"Aucun run sur branch={branch}, recherche sans filtre...")
                    runs = _fetch_runs("")

            if not runs:
                runs = _fetch_runs("")

            if runs:
                # Si pr_number fourni, privilegier les runs de la PR
                target_run = runs[0]
                if pr_number:
                    for r in runs:
                        prs = r.get("pull_requests", [])
                        if any(p.get("number") == pr_number for p in prs):
                            target_run = r
                            info(f"Run associe a PR #{pr_number} trouve : {r['id']}")
                            break

                run_id = str(target_run["id"])
                det_branch = target_run.get("head_branch", "")
                info(f"Run ID auto-detecte : {run_id} (branch: {det_branch}, conclusion: {target_run.get('conclusion')})")
                if not branch:
                    branch = det_branch
            else:
                print(f"{R}[ERROR] Aucun run trouve pour {repo}{E}")
                return
        except Exception as e:
            print(f"{R}[ERROR] Impossible de recuperer le dernier run : {e}{E}")
            return

    if not run_id:
        print(f"{R}[ERROR] Fournissez --run-id ou verifiez votre GITHUB_TOKEN.{E}")
        return

    hdr(f"CI ANALYZE — Run #{run_id} | PR #{pr_number or '?'} | {repo}")
    info(f"Branch      : {branch or '(auto)'}")
    info(f"Project key : {project_key or '(non configuré)'}")
    print()

    result = invoke_ci_run(
        run_id=run_id,
        repo=repo,
        owner=owner,
        project_key=project_key,
        pr_number=pr_number,
        pr_branch=branch,
        run_conclusion="",
    )

    print()
    print(f"{'='*60}")
    print(f"  Outcome       : {result.get('outcome', '?')}")
    print(f"  Failure type  : {result.get('failure_type', '?')}")
    print(f"  Stage echoue  : {result.get('stage_failed') or 'N/A'}")
    print(f"  PR number     : {result.get('pr_number') or 'aucune'}")
    print(f"  Comment poste : {'[OK]' if result.get('comment_posted') else '[NO]'}")
    print(f"  Notification  : {result.get('notification_level', '?')}")
    if result.get("root_cause"):
        print(f"  Cause racine  : {result['root_cause'][:120]}...")
    print(f"{'='*60}")


# ── Commande : ci-poll (CI/CD Intelligence) ────────────────────────────────────

def cmd_ci_poll(args):
    """
    Mode polling CI Intelligence (T7+T8) — surveille les GitHub Actions runs
    et déclenche le CIGraph IA pour analyser chaque run terminé.

    T7 : Les échecs publish/deploy sont aussi connectés au CIGraph.
    T8 : Boucle polling configurable avec persistance Redis des runs vus.

    Usage :
      python main.py ci-poll --repo owner/repo
      python main.py ci-poll --repo owner/repo --project-key sonar_key --interval 60
      python main.py ci-poll --repo owner/repo --branch main --interval 30
    """
    from ci_cd.ci_poller import CIPoller

    repo        = args.repo
    project_key = getattr(args, "project_key", "") or ""
    interval    = getattr(args, "interval", 120)
    branch      = getattr(args, "branch", "") or ""
    watch_jobs  = getattr(args, "watch_jobs", None)

    hdr(f"MODE CI INTELLIGENCE — Polling ({repo})")
    info(f"SonarCloud project : {project_key or '(non configuré)'}")
    info(f"Intervalle polling : {interval}s")
    info(f"Branche            : {branch or 'toutes'}")
    info("T7 : publish/deploy connectés au CIGraph")
    info("T8 : boucle polling avec mémoire Redis")
    info("Ctrl+C pour arrêter\n")

    try:
        poller = CIPoller(
            repo=repo,
            interval=interval,
            branch=branch,
            project_key=project_key,
            watch_jobs=watch_jobs,
        )
        poller.run()
    except KeyboardInterrupt:
        print(f"\n{Y}  Polling CI arrêté.{E}")
    except Exception as e:
        print(f"{R}✗ Erreur poller : {e}{E}")


# ── Commande : ci-deploy (CI/CD) ──────────────────────────────────────────

def cmd_ci_deploy(args):
    """
    Déploie le workflow GitHub Actions CI/CD sur un repo distant via MCP.
    Détecte le langage (Java/Python/JS) et génère le YAML adapté.
    """
    import asyncio
    from smart_git.pr_analyzer import _parse_repo
    from ci_cd.ci_deploy_agent import deploy_ci_workflow

    owner, repo = _parse_repo(args.repo)
    auditor_repo   = getattr(args, "auditor_repo", "chmaryem/code_auditor")
    force          = getattr(args, "force", False)
    enable_deploy  = getattr(args, "enable_deploy", False)
    asyncio.run(deploy_ci_workflow(
        owner, repo,
        auditor_repo=auditor_repo,
        force=force,
        enable_deploy=enable_deploy,
    ))


# ── Commande : cd-score ──────────────────────────────────────────────

def cmd_cd_score(args):
    """
    Calcule le Release Readiness Score avant un déploiement.
    Aggrège CI, SonarCloud, sécurité, approvals PR, risk fichiers.
    Verdict: DEPLOY_OK / DEPLOY_WARN / DEPLOY_BLOCKED
    """
    from smart_git.pr_analyzer import _parse_repo
    from ci_cd.cd_release_scorer import CDReleaseScorer

    owner, repo_name = _parse_repo(args.repo)
    repo = f"{owner}/{repo_name}"

    hdr("RELEASE READINESS SCORE")
    info(f"Repo        : {repo}")
    info(f"Commit SHA  : {args.sha[:8] if args.sha else 'HEAD'}")
    info(f"Environment : {args.environment}")
    if args.project_key:
        info(f"SonarCloud  : {args.project_key}")

    scorer = CDReleaseScorer()
    report = scorer.score(
        repo=repo,
        owner=owner,
        commit_sha=args.sha or "",
        run_id=getattr(args, "run_id", "") or "",
        project_key=args.project_key or "",
        pr_number=args.pr or None,
        environment=args.environment,
    )

    verdict_colors = {
        "DEPLOY_OK":      G,
        "DEPLOY_WARN":    Y,
        "DEPLOY_BLOCKED": R,
    }
    vc = verdict_colors.get(report.verdict, Y)
    print(f"\n{vc}{B}  Verdict : {report.verdict}{E}")
    print(f"  Score   : {report.score:.0f}/100\n")

    print(f"{B}  Component scores:{E}")
    for name, s in report.component_scores.items():
        bar = f"{G}█{E}" if s >= 15 else (f"{Y}█{E}" if s >= 7 else f"{R}█{E}")
        print(f"    {bar} {name.replace('_', ' ').title():<22} {s:.1f}")

    if report.blocking_reasons:
        print(f"\n{R}{B}  Blocking:{E}")
        for r in report.blocking_reasons:
            print(f"{R}    ✗ {r}{E}")

    if report.warnings:
        print(f"\n{Y}  Warnings:{E}")
        for w in report.warnings:
            print(f"{Y}    ⚠ {w}{E}")


# ── Commande : cd-status ─────────────────────────────────────────────

def cmd_cd_status(args):
    """
    Affiche l'état courant d'un environnement de déploiement.
    Montre le dernier déploiement réussi et les stats (success rate, durée).
    """
    from smart_git.pr_analyzer import _parse_repo
    from ci_cd.cd_deploy_tracker import CDDeployTracker
    import datetime

    owner, repo_name = _parse_repo(args.repo)
    repo = f"{owner}/{repo_name}"
    env  = args.environment

    hdr(f"CD STATUS — {repo} / {env}")

    tracker = CDDeployTracker()
    state   = tracker.get_env_state(repo, env)
    last_ok = tracker.get_last_successful_deploy(repo, env)
    stats   = tracker.get_deploy_stats(repo, env)
    recent  = tracker.get_recent_deploys(repo, limit=5, env=env)

    if not state:
        print(f"{Y}  No deployment data found for {repo}/{env}{E}")
        print(f"  Run `python main.py ci-poll` to start tracking deployments.")
        return

    status = state.get("status", "unknown")
    sc = G if status == "success" else (R if status == "failed" else Y)
    print(f"\n{B}  Current Environment:{E}")
    print(f"    Status  : {sc}{status}{E}")
    print(f"    Version : {state.get('version', 'N/A')}")
    print(f"    SHA     : {state.get('commit_sha', 'N/A')[:8]}")
    print(f"    Branch  : {state.get('branch', 'N/A')}")
    print(f"    URL     : {state.get('deploy_url', 'N/A')}")

    if last_ok:
        ts = last_ok.get("success_at", 0)
        dt = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "N/A"
        print(f"\n{B}  Last Successful Deploy:{E}")
        print(f"    Version  : {last_ok.get('version', 'N/A')}")
        print(f"    SHA      : {last_ok.get('commit_sha', 'N/A')[:8]}")
        print(f"    At       : {dt}")

    print(f"\n{B}  Statistics (last 50 deploys):{E}")
    sr = stats.get('success_rate', 0)
    sc2 = G if sr >= 90 else (Y if sr >= 70 else R)
    print(f"    Total    : {stats.get('total', 0)}")
    print(f"    Success  : {stats.get('successes', 0)}")
    print(f"    Failed   : {stats.get('failures', 0)}")
    print(f"    Rate     : {sc2}{sr}%{E}")
    avg = stats.get('avg_duration_s', 0)
    print(f"    Avg time : {avg}s ({avg // 60}m{avg % 60:02d}s)")

    if recent:
        print(f"\n{B}  Recent deployments:{E}")
        for d in recent:
            ts  = d.get('started_at', 0)
            dt2 = datetime.datetime.fromtimestamp(ts).strftime("%m-%d %H:%M") if ts else "?"
            s   = d.get('status', '?')
            sc3 = G if s == 'success' else (R if s == 'failed' else Y)
            print(
                f"    {sc3}{s:<8}{E}  "
                f"ver={d.get('version', '?')[:20]:<20}  "
                f"{dt2}"
            )


# ── Commande : pr-check (MCP Code Mode) ──────────────────────────────────

def cmd_pr_check(args):
    """
    Analyse une Pull Request via MCP Code Mode.
    L'agent Gemini génère un script Python qui analyse la PR
    avec le pipeline RAG complet et poste un review structuré.
    """
    import asyncio
    from smart_git.pr_analyzer import analyze_pr, _parse_repo

    owner, repo = _parse_repo(args.repo)
    asyncio.run(analyze_pr(owner, repo, args.pr))


# ── Commande : pr-resolve (MCP Code Mode) ────────────────────────────────

def cmd_pr_resolve(args):
    """
    Résout les conflits d'une PR via MCP Code Mode.
    L'agent Gemini génère un script qui résout les conflits
    avec le resolver 3-strategy et pousse une branche de résolution.
    """
    import asyncio
    from smart_git.pr_analyzer import resolve_pr_conflicts, _parse_repo

    owner, repo = _parse_repo(args.repo)
    asyncio.run(resolve_pr_conflicts(owner, repo, args.pr))


# ── Commande : pr-merge-check (MCP Code Mode) ────────────────────────────

def cmd_pr_merge_check(args):
    """
    Vérifie si une PR est prête à merger via MCP Code Mode.
    L'agent vérifie le statut mergeable, les checks CI/CD,
    et les reviews, puis poste un rapport de readiness.
    NE MERGE JAMAIS automatiquement.
    """
    import asyncio
    from smart_git.pr_analyzer import check_pr_merge_readiness, _parse_repo

    owner, repo = _parse_repo(args.repo)
    asyncio.run(check_pr_merge_readiness(owner, repo, args.pr))


# ── Commande : generate-tests ──────────────────────────────────────────────

def cmd_generate_tests(args):
    """
    Génère des tests unitaires pour un fichier source donné.
    Utilise le TestGeneratorAgent avec RAG (test_patterns_kb + ProjectCodeIndexer)
    pour produire des tests cohérents avec les conventions du projet.
    """
    from pathlib import Path
    from agents.test_generator_agent import TestGeneratorAgent

    source_path = Path(args.path).resolve()
    if not source_path.exists():
        err(f"Fichier introuvable : {source_path}")
        return

    project_path = Path(args.project).resolve() if args.project else source_path.parent

    hdr(f"GÉNÉRATION DE TESTS : {source_path.name}")
    info(f"Projet : {project_path}")

    # ── Initialiser le RAG pour les tests ──────────────────────────────────
    test_kb = None
    project_code_indexer = None
    knowledge_graph_inst = None

    try:
        from services.llm_service import assistant_agent
        from services.test_knowledge_loader import TestKnowledgeLoader
        info("Chargement test_patterns_kb (RAG test patterns)...")
        test_kb = TestKnowledgeLoader(embeddings=assistant_agent.embeddings)
        n = test_kb.load()
        stats = test_kb.get_stats()
        info(f"test_patterns_kb : {n} chunks ({stats.get('by_language', {})})")
    except Exception as e:
        info(f"test_patterns_kb non disponible : {e}")

    try:
        from services.llm_service import assistant_agent
        from services.knowledge_loader import ProjectCodeIndexer
        info("Chargement ProjectCodeIndexer...")
        project_code_indexer = ProjectCodeIndexer(embeddings=assistant_agent.embeddings)
        project_code_indexer.index_project(project_path)
    except Exception as e:
        info(f"ProjectCodeIndexer non disponible : {e}")

    try:
        from services.knowledge_graph import knowledge_graph
        knowledge_graph_inst = knowledge_graph
    except Exception:
        pass

    # ── Générer les tests ──────────────────────────────────────────────────
    agent = TestGeneratorAgent(
        project_path=project_path,
        test_kb=test_kb,
        project_code_indexer=project_code_indexer,
        knowledge_graph=knowledge_graph_inst,
    )

    info("Génération en cours (LLM + RAG)...\n")
    result = agent.generate_for_file(source_path, write=args.write)

    if result.get("error"):
        err(result["error"])
        return

    # ── Affichage résultat ─────────────────────────────────────────────────
    if result.get("test_file"):
        ok(f"Tests générés : {result['test_file']}")
        info(f"Framework : {result.get('framework', '?')}")
        info(f"Docs RAG utilisés : {result.get('rag_docs_used', 0)}")
        if result.get("validated"):
            ok("Validation structurelle : OK")
        else:
            info("Validation structurelle : echec (retry effectue)")

        if not args.write:
            print(f"\n{Y}  Mode preview — le code ci-dessous n'a PAS été écrit sur disque.{E}")
            print(f"{Y}  Relancez avec --write pour écrire le fichier.{E}\n")
            print(f"{B}{'─' * 70}{E}")
            print(result.get("test_code", ""))
            print(f"{B}{'─' * 70}{E}")
        else:
            ok(f"Fichier écrit : {result['test_file']}")
    else:
        err(result.get("error", "Échec de la génération"))


#  CLI ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Code Auditor AI — analyse intelligente de code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=
        """
Exemples :
  python main.py file    mon_script.py     # un seul fichier
  python main.py project ./mon_projet      # projet complet
  python main.py watch   ./mon_projet      # surveillance temps réel
  python main.py git     ./mon_projet      # dernier commit Git
  python main.py hook    ./mon_projet      # installer le pre-commit hook

  # Smart Git Merge
  python main.py resolve-conflicts ./mon_projet          # résoudre conflits locaux
  python main.py merge-hook ./mon_projet                  # installer merge hook

  # CI/CD Pipeline
  python main.py ci-deploy      --repo owner/repo           # déployer workflow CI/CD

  # MCP Code Mode (agents autonomes)
  python main.py pr-check       --repo owner/repo --pr 42  # revue PR via agent
  python main.py pr-resolve     --repo owner/repo --pr 42  # résoudre conflits PR
  python main.py pr-merge-check --repo owner/repo --pr 42  # vérifier readiness merge
  python main.py generate-tests ./src/service.py --project ./  # générer tests unitaires
        """
    )
    sub = p.add_subparsers(dest="command")

    sp = sub.add_parser("file",    help="Analyser un seul fichier")
    sp.add_argument("path")

    sp = sub.add_parser("project", help="Analyser un projet complet")
    sp.add_argument("path")
    sp.add_argument("--max-files", type=int, default=10)

    sp = sub.add_parser("watch",   help="Surveiller en temps réel")
    sp.add_argument("path")
    sp.add_argument("--langgraph", action="store_true",
                    help="Use LangGraph orchestration (LangChain agents + StateGraph)")

    sp = sub.add_parser("git",        help="Analyser un commit Git donné")
    sp.add_argument("path")
    sp.add_argument("--commit", default="HEAD", help="Hash du commit (défaut: HEAD)")

    sp = sub.add_parser("git-status", help="Afficher l'état de la session (bugs accumulés non commités)")
    sp.add_argument("path")

    sp = sub.add_parser("git-branch", help="Analyser une branche avant merge")
    sp.add_argument("path")
    sp.add_argument("--branch", default="HEAD", help="Branche à analyser (défaut: HEAD)")
    sp.add_argument("--base",   default="main",  help="Branche de base (défaut: main)")
    sp.add_argument("--report", action="store_true", help="Sauvegarder le rapport en JSON")

    sp = sub.add_parser("hook",       help="Installer/désinstaller le pre-commit hook Git")
    sp.add_argument("path")
    sp.add_argument("--uninstall",  action="store_true", help="Désinstaller le hook")
    sp.add_argument("--no-strict",  action="store_true", help="Mode non-strict")

    # ── Smart Git Merge commands ────────────────────────────────────────────
    sp = sub.add_parser("resolve-conflicts", help="Résoudre les conflits de merge locaux via LLM")
    sp.add_argument("path", help="Chemin du projet Git")

    sp = sub.add_parser("merge-hook", help="Installer/désinstaller le pre-merge hook")
    sp.add_argument("path", help="Chemin du projet Git")
    sp.add_argument("--uninstall", action="store_true", help="Désinstaller le merge hook")

    # ── CI/CD Pipeline ──────────────────────────────────────────────────────
    sp = sub.add_parser("ci-deploy", help="Déployer le workflow CI/CD sur un repo GitHub")
    sp.add_argument("--repo",          required=True, help="owner/repo cible")
    sp.add_argument("--auditor-repo",  default="chmaryem/code_auditor", help="Repo de Code Auditor")
    sp.add_argument("--branch",        default="main", help="Branche cible (defaut: main)")
    sp.add_argument("--force",         action="store_true", help="Ecraser le workflow existant")
    sp.add_argument("--enable-deploy", action="store_true", dest="enable_deploy",
                    help="Activer Stage 6 : deploy SSH + docker compose (necessite DEPLOY_HOST, DEPLOY_USER, DEPLOY_SSH_KEY)")

    sp = sub.add_parser("ci-poll", help="Surveiller les GitHub Actions runs et analyser les failures (CI Intelligence IA)")
    sp.add_argument("--repo",        required=True,           help="owner/repo à surveiller")
    sp.add_argument("--project-key", default="",              dest="project_key", help="Clé projet SonarCloud")
    sp.add_argument("--interval",    type=int, default=120,   help="Intervalle de polling en secondes (défaut: 120)")
    sp.add_argument("--branch",      default="",              help="Surveiller uniquement cette branche (défaut: toutes)")
    sp.add_argument("--watch-jobs",  nargs="*",               dest="watch_jobs", help="Jobs spécifiques à surveiller")
    sp.add_argument("--max-runs",    type=int, default=5,     dest="max_runs",   help="Runs à récupérer par poll (défaut: 5)")

    sp = sub.add_parser("ci-analyze", help="Analyser manuellement un run GitHub Actions via le CIGraph IA")
    sp.add_argument("--repo",        required=True,   help="owner/repo")
    sp.add_argument("--run-id",      default="",      dest="run_id",      help="Run ID GitHub Actions")
    sp.add_argument("--pr",          type=int,        default=None,        help="Numéro de la PR")
    sp.add_argument("--branch",      default="",                           help="Branche (ex: feature/test-hook)")
    sp.add_argument("--project-key", default="",      dest="project_key", help="Clé SonarCloud")

    sp = sub.add_parser("cd-score", help="Release Readiness Score avant déploiement (CD Intelligence)")
    sp.add_argument("--repo",        required=True,   help="owner/repo")
    sp.add_argument("--sha",         default="",      help="Commit SHA à évaluer")
    sp.add_argument("--pr",          type=int,        default=None,  help="Numéro PR associée")
    sp.add_argument("--project-key", default="",      dest="project_key", help="Clé SonarCloud")
    sp.add_argument("--environment", default="production", help="Environnement cible (défaut: production)")

    sp = sub.add_parser("cd-status", help="État courant d'un environnement de déploiement")
    sp.add_argument("--repo",        required=True,   help="owner/repo")
    sp.add_argument("--environment", default="production", help="Environnement (défaut: production)")

    sp = sub.add_parser("pr-check", help="Analyser une PR GitHub via MCP")
    sp.add_argument("--repo", required=True, help="owner/repo")
    sp.add_argument("--pr", type=int, required=True, help="Numéro de la PR")

    sp = sub.add_parser("pr-resolve", help="Résoudre les conflits d'une PR via MCP Code Mode")
    sp.add_argument("--repo", required=True, help="owner/repo")
    sp.add_argument("--pr", type=int, required=True, help="Numéro de la PR")

    sp = sub.add_parser("pr-merge-check", help="Vérifier si une PR est prête à merger (MCP Code Mode)")
    sp.add_argument("--repo", required=True, help="owner/repo")
    sp.add_argument("--pr", type=int, required=True, help="Numéro de la PR")

    sp = sub.add_parser("generate-tests", help="Générer des tests unitaires pour un fichier source")
    sp.add_argument("path", help="Chemin du fichier source à tester")
    sp.add_argument("--project", help="Chemin racine du projet (défaut: dossier du fichier)")
    sp.add_argument("--write", action="store_true", help="Écrire le fichier de test sur disque")

    # ── Chat Agent ────────────────────────────────────────────────────────────
    sp = sub.add_parser("chat", help="Project-aware ChatAgent (Phase 1+2) — Q&A / explain / generate")
    sp.add_argument("--project",   default=".",  help="Project root path (default: current dir)")
    sp.add_argument("--ask",       default="",   help="One-shot question (non-interactive mode)")
    sp.add_argument("--session",   default="",   help="Existing session ID (continue a chat session)")
    sp.add_argument("--file",      default="",   help="Optional target file path or filename")
    # Phase 2 — code generation
    sp.add_argument("--complete",  default="",   help="[Phase 2] Complete a function: 'findByEmail in UserService.java'")
    sp.add_argument("--new-class", default="",   dest="new_class",
                    help="[Phase 2] Generate a new class: 'ProductService'")
    sp.add_argument("--lang",      default="",   help="[Phase 2] Language: java|python|javascript|typescript")
    sp.add_argument("--write",     action="store_true",
                    help="[Phase 2] Write generated file to disk (suggested path)")

    # ── API Server ─────────────────────────────────────────────────────────────
    sp = sub.add_parser("serve", help="Start FastAPI server (REST + WebSocket + ChatAgent API)")
    sp.add_argument("--host",    default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    sp.add_argument("--port",    type=int, default=8765, help="Port (default: 8765)")
    sp.add_argument("--project", default=".", help="Default project path for analysis endpoints")
    sp.add_argument("--reload",  action="store_true", help="Hot-reload (dev mode)")

    return p

# ── Commande : chat ───────────────────────────────────────────────────────────

def cmd_chat(args):
    """ChatAgent Phase 1+2 — Q&A / explain / complete function / generate class.

    Modes:
      --ask "..."       : one-shot question (Phase 1)
      --complete "..."  : complete a function body (Phase 2)
      --new-class NAME  : generate a new class (Phase 2)
      (no flags)        : interactive REPL
    """
    from langchain_agents.graphs.chat_graph import invoke_chat

    project = str(Path(args.project).resolve())

    # ── Phase 2 : function completion ───────────────────────────────────────
    if getattr(args, "complete", ""):
        message = f"complete {args.complete}"
        result  = invoke_chat(
            message=message,
            project_path=project,
            session_id=args.session,
            target_file=args.file,
        )
        _print_generation_result(result, args)
        return

    # ── Phase 2 : new class generation ──────────────────────────────────────
    if getattr(args, "new_class", ""):
        lang_part = f" in {args.lang}" if getattr(args, "lang", "") else ""
        message   = f"create a {args.new_class} class{lang_part}"
        result    = invoke_chat(
            message=message,
            project_path=project,
            session_id=args.session,
            target_file=args.file,
        )
        _print_generation_result(result, args)
        return

    # ── Phase 1 : one-shot Q&A ───────────────────────────────────────────────
    if args.ask:
        result = invoke_chat(
            message=args.ask,
            project_path=project,
            session_id=args.session,
            target_file=args.file,
        )
        output = result.get("formatted_response") or result.get("response", "")
        print(output)
        return

    # ── Interactive REPL ─────────────────────────────────────────────────────
    from services.chat_memory_service import chat_memory_service
    session = args.session or chat_memory_service.new_session_id()

    hdr("Code Auditor — ChatAgent (Phase 1+2)")
    info(f"Projet  : {project}")
    if args.file:
        info(f"Fichier : {args.file}")
    info(f"Session : {session}  (--session {session} pour reprendre)")
    print(f"{Y}  Tape 'exit' ou Ctrl-C pour quitter.{E}")
    print(f"{C}  Phase 2 : 'complete <fn> in <file>' | 'create a <ClassName> class'{E}\n")

    while True:
        try:
            msg = input(f"{B}you>{E} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        if msg.lower() in ("exit", "quit", "q"):
            break

        result  = invoke_chat(
            message=msg,
            project_path=project,
            session_id=session,
            target_file=args.file,
        )
        session = result.get("session_id", session)
        output  = result.get("formatted_response") or result.get("response", "")
        elapsed = result.get("stats", {}).get("elapsed", "?")
        print(f"\n{G}assistant>{E} {output}\n")
        print(f"{C}  [{elapsed}s]{E}\n")


def _print_generation_result(result: dict, args) -> None:
    """Print generated code result and optionally write to disk."""
    output  = result.get("formatted_response") or result.get("response", "")
    elapsed = result.get("stats", {}).get("elapsed", "?")

    # Windows cp1252 terminals can't print emojis — encode safely
    def _safe_print(text: str) -> None:
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
                sys.stdout.encoding or "utf-8", errors="replace"
            ))

    _safe_print(f"\n{G}assistant>{E}")
    _safe_print(output)
    _safe_print(f"\n{C}  [{elapsed}s]{E}")

    # Write to disk if --write and suggested file path available
    if getattr(args, "write", False):
        suggested = (result.get("suggested_files") or [""])[0]
        code      = result.get("generated_code", "")
        if suggested and code:
            out_path = Path(suggested)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(code, encoding="utf-8")
            ok(f"Written to {out_path}")
        else:
            err("No suggested file path — cannot write to disk automatically.")



def cmd_serve(args):
    """Start the FastAPI server exposing REST + WebSocket + ChatAgent API.

    Endpoints available after start:
      GET  /health                    — server readiness check
      POST /api/chat                  — Q&A / explain (Phase 1)
      POST /api/chat/complete         — function completion (Phase 2)
      POST /api/chat/generate         — new class generation (Phase 2)
      GET  /api/chat/history/{id}     — conversation history
      POST /analyze/file              — single file analysis
      POST /analyze/project           — full project analysis
      GET  /watch/status              — watch mode status
      WS   /ws                        — real-time events
      GET  /docs                      — Swagger UI

    The VS Code plugin connects to this server.
    """
    import uvicorn
    import logging as _logging
    from pathlib import Path as _Path
    import api.server as _srv

    project = str(_Path(args.project).resolve())
    _srv._default_project = _Path(project)

    hdr(f"Code Auditor — API Server")
    info(f"Projet  : {project}")
    info(f"Adresse : http://{args.host}:{args.port}")
    info(f"Docs    : http://{args.host}:{args.port}/docs")
    info(f"Chat    : http://{args.host}:{args.port}/api/chat")
    print(f"{C}  Ctrl-C pour arrêter.{E}\n")

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )



def main():
    parser = build_parser()
    args   = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    commands = {
        "file": cmd_file, "project": cmd_project, "watch": cmd_watch,
        "git": cmd_git, "git-status": cmd_git_status,
        "git-branch": cmd_git_branch, "hook": cmd_hook,
        # Smart Git Merge
        "resolve-conflicts": cmd_resolve_conflicts,
        "merge-hook": cmd_merge_hook,
        # CI/CD Pipeline
        "ci-deploy":  cmd_ci_deploy,
        "ci-poll":    cmd_ci_poll,
        "ci-analyze": cmd_ci_analyze,    # analyse manuelle
        # CD Intelligence
        "cd-score":   cmd_cd_score,
        "cd-status":  cmd_cd_status,
        # MCP Code Mode
        "pr-check":       cmd_pr_check,
        "pr-resolve":     cmd_pr_resolve,
        "pr-merge-check": cmd_pr_merge_check,
        "generate-tests": cmd_generate_tests,
        # Chat Agent
        "chat":           cmd_chat,
        # API Server
        "serve":          cmd_serve,
    }
    commands[args.command](args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Y}  Interrompu par l'utilisateur.{E}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{R}Erreur fatale : {e}{E}")
        import traceback; traceback.print_exc()
        sys.exit(1)
