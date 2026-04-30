"""
ci_cd/ci_runner.py — Point d'entrée exécuté par le runner GitHub Actions.

Ce script est le pont entre GitHub Actions et le pipeline d'analyse :

  Flow complet :
    1. Parse --repo owner/repo et --pr N passés par le workflow YAML
    2. Récupère le SHA du commit HEAD de la PR (via GitHubClient)
    3. Poste le status "pending" (le dev voit le check en cours dans GitHub UI)
    4. Appelle review_pr() — la MÊME fonction que `python main.py pr-check`
       → RAG + KG + LLM → verdict APPROVE / COMMENT / REQUEST_CHANGES
    5. Poste le status final (success/failure) sur le commit
    6. Exit 1 si REQUEST_CHANGES (le merge est bloqué dans GitHub UI)
       Exit 0 sinon (merge autorisé)

CORRECTIONS v2 :
  - Imports depuis le package ci_cd (relatif) — pas d'import absolu fragile
  - sys.path.insert() pointe sur le bon dossier parent selon l'endroit d'exécution
  - Gestion du cas où head_sha est vide (PR fraîchement créée, SHA pas encore dispo)
  - Le status "error" est posté en cas d'exception dans review_pr()
  - Message de sortie terminal clair avec les métriques

Exécution :
  python -m ci_cd.ci_runner --repo owner/repo --pr 42
  (depuis le dossier racine du projet code_auditor_tool)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# ── Chemin racine ─────────────────────────────────────────────────────────────
# Le script peut être lancé depuis n'importe quel répertoire.
# On s'assure que le dossier parent de ci_cd/ (= racine du projet) est dans le path.
_THIS_DIR    = Path(__file__).resolve().parent   # .../code_auditor_tool/ci_cd/
_PROJECT_ROOT = _THIS_DIR.parent                  # .../code_auditor_tool/
sys.path.insert(0, str(_PROJECT_ROOT))

# Force UTF-8 (Windows cp1252 ne supporte pas les symboles Unicode)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("ci_runner")


# ─────────────────────────────────────────────────────────────────────────────
# Validation pré-exécution
# ─────────────────────────────────────────────────────────────────────────────

def _test_mcp_connection(timeout: int = 30) -> tuple[bool, str]:
    """
    Teste que le serveur MCP GitHub est accessible.
    Retourne (is_available, message)
    """
    import subprocess
    try:
        # Test simple: vérifier que npx peut exécuter le serveur MCP
        result = subprocess.run(
            ["npx", "--yes", "@modelcontextprotocol/server-github", "--version"],
            capture_output=True,
            timeout=timeout
        )
        if result.returncode == 0:
            version = result.stdout.decode().strip() if result.stdout else "unknown"
            return True, f"MCP server disponible (version: {version})"
        stderr = result.stderr.decode()[:200] if result.stderr else ""
        return False, f"MCP server erreur: {stderr}"
    except subprocess.TimeoutExpired:
        return False, f"MCP server timeout ({timeout}s) - problème réseau ou npm"
    except FileNotFoundError:
        return False, "npx non trouvé - Node.js/npm doit être installé"
    except Exception as e:
        return False, f"MCP server inaccessible: {str(e)}"


# ─────────────────────────────────────────────────────────────────────────────
# Parsing des arguments
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Code Auditor CI — Analyse automatique de PR via GitHub Actions"
    )
    parser.add_argument("--repo", required=True,  help="owner/repo (ex: chmaryem/my-project)")
    parser.add_argument("--pr",   required=True,  help="Numéro de la PR (int ou str)")
    return parser.parse_args()


def _parse_repo(repo_str: str) -> tuple[str, str]:
    """Parse 'owner/repo' → (owner, repo). Exit 1 si format invalide."""
    parts = repo_str.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        print(f"[ERROR] Format repo invalide: '{repo_str}' (attendu: owner/repo)")
        sys.exit(1)
    return parts[0].strip(), parts[1].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internes
# ─────────────────────────────────────────────────────────────────────────────

def _get_head_sha(owner: str, repo: str, pr_number: int) -> str:
    """
    Récupère le SHA du commit HEAD de la PR via GitHubClient.

    Retourne "" si la récupération échoue (la suite du pipeline peut continuer
    sans SHA — les statuses ne seront juste pas postés).
    """
    try:
        from services.code_mode_client import github
        pr_info = github.get_pr_info(owner, repo, pr_number)
        sha = pr_info.get("head", {}).get("sha", "") if pr_info else ""
        try:
            github.disconnect()
        except Exception:
            pass
        return sha
    except Exception as e:
        logger.warning("Impossible de récupérer le SHA de la PR : %s", e)
        return ""


def _post_pending(owner: str, repo: str, sha: str) -> None:
    """Poste le status 'pending' si on a un SHA valide."""
    if not sha:
        return
    try:
        from ci_cd.ci_status_reporter import post_status
        ok = post_status(
            owner, repo, sha,
            state       = "pending",
            description = "Analyse Code Auditor en cours...",
        )
        if ok:
            print(f"  Status 'pending' poste sur {sha[:8]}")
    except Exception as e:
        logger.warning("Status pending non poste : %s", e)


def _post_error(owner: str, repo: str, sha: str, message: str) -> None:
    """Poste le status 'error' en cas d'exception dans le pipeline."""
    if not sha:
        return
    try:
        from ci_cd.ci_status_reporter import post_status
        post_status(
            owner, repo, sha,
            state       = "error",
            description = f"Erreur analyse : {message[:100]}",
        )
    except Exception:
        pass  # On ne bloque pas sur une erreur de status


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée principal
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args      = _parse_args()
    owner, repo = _parse_repo(args.repo)

    # GitHub Actions injecte parfois le pr_number entre guillemets
    try:
        pr_number = int(str(args.pr).strip('"').strip("'"))
    except ValueError:
        print(f"[ERROR] --pr doit être un entier, recu: '{args.pr}'")
        sys.exit(1)

    # ── Bannière ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Code Auditor CI — PR #{pr_number}")
    print(f"  Repo    : {owner}/{repo}")
    print(f"  Mode    : GitHub Actions (runner automatique)")
    print(f"{'='*60}\n")

    # ── Step 0 : Test connexion MCP ───────────────────────────────────────────
    print("  [0/5] Test connexion MCP...")
    mcp_ok, mcp_msg = _test_mcp_connection()
    if not mcp_ok:
        print(f"  [WARN] {mcp_msg}")
        print("  Le pipeline va tenter de continuer, mais l'analyse PR risque d'échouer.")
        # On continue mais on log le warning
    else:
        print(f"  ✓ {mcp_msg}")
    print()

    # ── Step 1 : Récupérer le SHA du commit HEAD ──────────────────────────────
    print("  [1/5] Recuperation du SHA HEAD...")
    head_sha = _get_head_sha(owner, repo, pr_number)
    if head_sha:
        print(f"  SHA : {head_sha[:8]}...{head_sha[-4:]}")
    else:
        print("  SHA indisponible — les statuses GitHub ne seront pas postes")

    # ── Step 2 : Poster le status "pending" ───────────────────────────────────
    print("  [2/5] Status 'pending'...")
    _post_pending(owner, repo, head_sha)

    # ── Step 3 : Lancer l'analyse (même logique que `python main.py pr-check`) ─
    print(f"  [3/5] Lancement review_pr ({owner}/{repo} PR#{pr_number})...")
    try:
        from smart_git.pr_review_agent import review_pr
        result = asyncio.run(review_pr(owner, repo, pr_number))
    except ImportError as e:
        msg = f"Import error : {e} — verifiez que requirements.txt est installe"
        print(f"\n  [ERROR] {msg}")
        _post_error(owner, repo, head_sha, msg)
        sys.exit(1)
    except Exception as e:
        msg = str(e)
        print(f"\n  [ERROR] Analyse echouee : {msg}")
        logger.exception("review_pr exception")
        _post_error(owner, repo, head_sha, msg)
        sys.exit(1)

    # ── Step 4 : Poster le status final ──────────────────────────────────────
    print("  [4/5] Status final...")

    if not result.get("success"):
        error_msg = result.get("error", "Resultat analyse invalide")
        print(f"\n  [ERROR] Analyse sans succes : {error_msg}")
        _post_error(owner, repo, head_sha, error_msg)
        sys.exit(1)

    # Utiliser le SHA retourné par review_pr si disponible (plus fiable)
    final_sha = result.get("head_sha") or head_sha

    if final_sha:
        try:
            from ci_cd.ci_status_reporter import post_review_status
            post_review_status(
                owner    = owner,
                repo     = repo,
                sha      = final_sha,
                verdict  = result.get("verdict",  "APPROVE"),
                score    = result.get("score",     0.0),
                critical = result.get("critical",  0),
                high     = result.get("high",      0),
                medium   = result.get("medium",    0),
            )
        except Exception as e:
            # Ne pas bloquer le pipeline si le status échoue
            logger.error("Status final non poste : %s", e)
    else:
        print("  Pas de SHA disponible — status non poste")

    # ── Step 5 : Exit code ────────────────────────────────────────────────────
    verdict = result.get("verdict", "APPROVE")
    score   = result.get("score",   0.0)

    print(f"\n{'='*60}")
    if verdict == "REQUEST_CHANGES":
        print(f"  CI EXIT 1 — MERGE BLOQUE")
        print(f"  Score    : {score:.0f}")
        print(f"  Critical : {result.get('critical', 0)}")
        print(f"  High     : {result.get('high', 0)}")
        print(f"  Ouvrez le review GitHub pour voir les details.")
        print(f"{'='*60}\n")
        sys.exit(1)
    else:
        print(f"  CI EXIT 0 — MERGE AUTORISE ({verdict})")
        print(f"  Score : {score:.0f}")
        print(f"{'='*60}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()