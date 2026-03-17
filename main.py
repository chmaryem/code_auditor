"""
main.py — Point d'entrée unique du projet Code Auditor AI.

Commandes disponibles :
  python main.py file    <fichier>        # analyser un seul fichier
  python main.py project <dossier>        # analyser un projet complet
  python main.py watch   <dossier>        # surveillance temps réel
  python main.py git     <dossier>        # analyser le dernier commit Git
  python main.py hook    <dossier>        # installer le pre-commit hook Git
"""
import sys
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
    """Surveille un projet en temps réel."""
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

    watcher = FileWatcher(project_path=project_path, callback=on_change)
    try:
        watcher.watch()
    except KeyboardInterrupt:
        pass
    finally:
        watcher.stop()
        orchestrator.stop()



# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Code Auditor AI — analyse intelligente de code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python main.py file    mon_script.py     # un seul fichier
  python main.py project ./mon_projet      # projet complet
  python main.py watch   ./mon_projet      # surveillance temps réel
  python main.py git     ./mon_projet      # dernier commit Git
  python main.py hook    ./mon_projet      # installer le pre-commit hook
        """,
    )
    sub = p.add_subparsers(dest="command")

    sp = sub.add_parser("file",    help="Analyser un seul fichier")
    sp.add_argument("path")

    sp = sub.add_parser("project", help="Analyser un projet complet")
    sp.add_argument("path")
    sp.add_argument("--max-files", type=int, default=10)

    sp = sub.add_parser("watch",   help="Surveiller en temps réel")
    sp.add_argument("path")

    sp = sub.add_parser("git",     help="Analyser le dernier commit Git")
    sp.add_argument("path")
    sp.add_argument("--commit", default="HEAD")

    sp = sub.add_parser("hook",    help="Installer le pre-commit hook Git")
    sp.add_argument("path")

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    {"file": cmd_file, "project": cmd_project, "watch": cmd_watch,}[args.command](args)


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