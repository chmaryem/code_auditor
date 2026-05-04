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

def cmd_watch(args):
    """Surveille un projet en temps réel + Smart Git Session Tracker."""
    try:
        import watchdog  # noqa
    except ImportError:
        err("Module 'watchdog' manquant — installe-le : pip install watchdog")
        return

    from core.orchestrator import Orchestrator
    from core.events import file_changed_event
    from watchers.file_watcher import FileWatcher

    project_path = Path(args.path)
    if not project_path.is_dir():
        err(f"Dossier introuvable : {project_path}")
        return

    hdr("MODE WATCH — SURVEILLANCE TEMPS RÉEL")
    info(f"Projet : {project_path}\n")

    orchestrator = Orchestrator(project_path)
    orchestrator.initialize()

    def on_change(file_path: Path, deleted: bool = False):
        orchestrator.handle(file_changed_event(file_path, deleted=deleted))

    # ── Smart Git Session Tracker ────────────────────────────────────────────
    # Démarre en arrière-plan — lit le cache Watch et surveille l'accumulation
    # de bugs non commités. Notifie le développeur si le score monte.
    git_tracker = None
    try:
        from smart_git.git_diff_parser import is_git_repo
        if is_git_repo(project_path):
            from smart_git.git_session_tracker import GitSessionTracker
            from smart_git.git_notifier import GitNotifier
            from config import config

            cache_db    = config.CACHE_DIR / "analysis_cache.db"
            notifier    = GitNotifier(print_lock=orchestrator._print_lock)
            git_tracker = GitSessionTracker(
                project_path   = project_path,
                cache_db       = cache_db,
                notifier       = notifier,
                check_interval = 180,   # vérification toutes les 3 min
            )
            git_tracker.start()
            info("Smart Git Tracker actif — surveillance accumulation de bugs\n")
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug("GitSessionTracker non démarré : %s", e)
    # ────────────────────────────────────────────────────────────────────────

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
    auditor_repo = getattr(args, "auditor_repo", "chmaryem/code_auditor")
    force = getattr(args, "force", False)
    asyncio.run(deploy_ci_workflow(owner, repo, auditor_repo=auditor_repo, force=force))


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
    sp.add_argument("--repo", required=True, help="owner/repo cible")
    sp.add_argument("--auditor-repo", default="chmaryem/code_auditor", help="Repo de Code Auditor")
    sp.add_argument("--force", action="store_true", help="Écraser le workflow existant")

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

    return p


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
        "ci-deploy": cmd_ci_deploy,
        # MCP Code Mode
        "pr-check": cmd_pr_check,
        "pr-resolve": cmd_pr_resolve,
        "pr-merge-check": cmd_pr_merge_check,
        "generate-tests": cmd_generate_tests,
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
