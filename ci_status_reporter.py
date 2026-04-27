"""
ci_status_reporter.py — Poste un status check sur un commit GitHub.

Utilise l'API GitHub Statuses :
  POST /repos/{owner}/{repo}/statuses/{sha}

Ceci crée un "commit status" visible dans l'onglet PR de GitHub.
Quand la branch protection rule exige ce status, le bouton Merge
est bloqué tant que le status n'est pas "success".

Utilise urllib.request (stdlib) — même pattern que mcp_github_service.py.
Pas besoin d'installer requests ou httpx.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

# Nom du status check tel qu'il apparaîtra dans GitHub UI
# C'est ce nom qu'il faut cocher dans Settings → Branch Protection
STATUS_CONTEXT = "Code Auditor / PR Review"


def _get_token() -> str:
    """Récupère le token GitHub (Actions ou local)."""
    # Dans GitHub Actions, GITHUB_TOKEN est injecté automatiquement
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    if not token:
        # Fallback : lire le .env local
        try:
            from dotenv import load_dotenv
            load_dotenv()
            token = os.environ.get("GITHUB_TOKEN", "")
            if not token:
                token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
        except ImportError:
            pass
    return token


def post_status(
    owner: str,
    repo: str,
    sha: str,
    state: str,
    description: str,
    target_url: Optional[str] = None,
) -> bool:
    """
    Poste un commit status sur GitHub.

    Args:
        owner: Propriétaire du repo (ex: "chmaryem")
        repo: Nom du repo (ex: "test-project-")
        sha: SHA complet du commit HEAD de la PR
        state: "success" | "failure" | "pending" | "error"
        description: Message affiché dans GitHub UI (max 140 chars)
        target_url: URL optionnelle vers les détails (ex: lien vers le run)

    Returns:
        True si le status a été posté avec succès
    """
    token = _get_token()
    if not token:
        logger.error("Aucun token GitHub disponible pour poster le status")
        return False

    url = f"https://api.github.com/repos/{owner}/{repo}/statuses/{sha}"

    payload = {
        "state": state,
        "description": description[:140],  # GitHub limite à 140 chars
        "context": STATUS_CONTEXT,
    }
    if target_url:
        payload["target_url"] = target_url

    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }

    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status in (200, 201):
                logger.info(
                    "Status '%s' posté sur %s/%s@%s : %s",
                    state, owner, repo, sha[:8], description,
                )
                return True
            else:
                logger.error("Status API réponse inattendue: %d", resp.status)
                return False
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:200]
        logger.error("Status API erreur %d : %s", e.code, body)
        return False
    except Exception as e:
        logger.error("Status API erreur : %s", e)
        return False


def post_review_status(
    owner: str,
    repo: str,
    sha: str,
    verdict: str,
    score: float,
    critical: int,
    high: int,
    medium: int,
) -> bool:
    """
    Poste le status check basé sur le résultat de review_pr().

    Logique de décision (identique à pr_review_agent.py) :
      - CRITICAL > 0 ou score >= 35  →  failure (merge bloqué)
      - score >= 15                   →  success (avec warning)
      - sinon                         →  success (merge autorisé)
    """
    if verdict == "REQUEST_CHANGES":
        state = "failure"
        desc = f"❌ Score {score:.0f} | C:{critical} H:{high} M:{medium} — MERGE BLOQUÉ"
    elif verdict == "COMMENT":
        state = "success"
        desc = f"⚠️ Score {score:.0f} | C:{critical} H:{high} M:{medium} — Attention"
    else:  # APPROVE
        state = "success"
        desc = f"✅ Score {score:.0f} | C:{critical} H:{high} M:{medium} — OK"

    print(f"  📡 Status check → {state} : {desc}")
    return post_status(owner, repo, sha, state, desc)
