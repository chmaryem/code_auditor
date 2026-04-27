"""
ci_runner.py — Point d'entrée pour GitHub Actions.

Ce script est exécuté par le workflow pr_review.yml.
Il fait le pont entre GitHub Actions et le pipeline d'analyse existant :

  1. Parse les arguments (--repo, --pr) passés par le workflow
  2. Poste un status "pending" sur le commit (le dev voit "Analyse en cours...")
  3. Appelle review_pr() — la MÊME fonction que `python main.py pr-check`
  4. Poste le status final (success/failure) basé sur le verdict
  5. Exit code 1 si merge bloqué (pour que le job GitHub Actions soit ❌)

Différences avec main.py pr-check :
  - Pas de terminal interactif
  - Gère le status check automatiquement
  - Exit code approprié pour CI/CD
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ajouter le dossier racine au path (comme main.py)
sys.path.insert(0, str(Path(__file__).parent))

# Force UTF-8 (GitHub Actions runners = Linux, mais au cas où)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("ci_runner")


def parse_args() -> argparse.Namespace:
    """Parse les arguments passés par le workflow GitHub Actions."""
    parser = argparse.ArgumentParser(
        description="Code Auditor CI — Analyse automatique de PR"
    )
    parser.add_argument(
        "--repo", required=True,
        help="Repo au format owner/repo (ex: chmaryem/test-project-)"
    )
    parser.add_argument(
        "--pr", type=int, required=True,
        help="Numéro de la PR à analyser"
    )
    return parser.parse_args()


def parse_repo(repo_str: str) -> tuple:
    """Parse 'owner/repo' en (owner, repo)."""
    parts = repo_str.split("/")
    if len(parts) != 2:
        print(f"❌ Format invalide : {repo_str} (attendu: owner/repo)")
        sys.exit(1)
    return parts[0], parts[1]


def main():
    args = parse_args()
    owner, repo = parse_repo(args.repo)
    pr_number = args.pr

    print(f"\n{'='*60}")
    print(f"  Code Auditor CI — PR #{pr_number}")
    print(f"  Repo : {owner}/{repo}")
    print(f"  Mode : GitHub Actions (automatique)")
    print(f"{'='*60}\n")

    # ── Step 1 : Récupérer le SHA du commit HEAD de la PR ──────────────
    # On en a besoin pour poster le status check AVANT l'analyse
    head_sha = ""
    try:
        from services.code_mode_client import github
        pr_info = github.get_pr_info(owner, repo, pr_number)
        if pr_info:
            head_sha = pr_info.get("head", {}).get("sha", "")
        # Déconnecter le MCP pour le reconnecter proprement dans review_pr()
        try:
            github.disconnect()
        except Exception:
            pass
    except Exception as e:
        logger.warning("Impossible de récupérer le SHA : %s", e)

    # ── Step 2 : Poster le status "pending" ────────────────────────────
    # Le développeur voit "Analyse en cours..." dans sa PR
    if head_sha:
        try:
            from ci_status_reporter import post_status
            post_status(
                owner, repo, head_sha,
                state="pending",
                description="🔄 Analyse en cours...",
            )
            print(f"  📡 Status 'pending' posté sur {head_sha[:8]}\n")
        except Exception as e:
            logger.warning("Status pending non posté : %s", e)

    # ── Step 3 : Lancer l'analyse (même code que pr-check) ─────────────
    try:
        from smart_git.pr_review_agent import review_pr
        result = asyncio.run(review_pr(owner, repo, pr_number))
    except Exception as e:
        logger.error("Analyse échouée : %s", e)

        # Poster un status "error" si l'analyse plante
        if head_sha:
            try:
                from ci_status_reporter import post_status
                post_status(
                    owner, repo, head_sha,
                    state="error",
                    description=f"💥 Erreur d'analyse : {str(e)[:100]}",
                )
            except Exception:
                pass

        sys.exit(1)

    # ── Step 4 : Poster le status final ────────────────────────────────
    if not result.get("success"):
        error_msg = result.get("error", "Analyse échouée")
        print(f"\n  ❌ Analyse échouée : {error_msg}")

        if head_sha:
            try:
                from ci_status_reporter import post_status
                post_status(
                    owner, repo, head_sha,
                    state="error",
                    description=f"Erreur : {error_msg[:120]}",
                )
            except Exception:
                pass

        sys.exit(1)

    # Utiliser le SHA retourné par review_pr() si disponible
    sha = result.get("head_sha", head_sha)

    if sha:
        try:
            from ci_status_reporter import post_review_status
            post_review_status(
                owner=owner,
                repo=repo,
                sha=sha,
                verdict=result.get("verdict", "APPROVE"),
                score=result.get("score", 0),
                critical=result.get("critical", 0),
                high=result.get("high", 0),
                medium=result.get("medium", 0),
            )
        except Exception as e:
            logger.error("Status final non posté : %s", e)
    else:
        print("  ⚠️ Pas de SHA disponible — status check non posté")

    # ── Step 5 : Exit code ─────────────────────────────────────────────
    # Exit 1 = le job GitHub Actions sera ❌ (rouge)
    # Exit 0 = le job sera ✅ (vert)
    verdict = result.get("verdict", "APPROVE")
    if verdict == "REQUEST_CHANGES":
        print(f"\n  🚫 CI EXIT 1 — Merge bloqué (score {result.get('score', 0):.0f})")
        sys.exit(1)
    else:
        print(f"\n  ✅ CI EXIT 0 — Merge autorisé (score {result.get('score', 0):.0f})")
        sys.exit(0)


if __name__ == "__main__":
    main()
