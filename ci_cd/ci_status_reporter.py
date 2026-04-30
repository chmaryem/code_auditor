"""
ci_cd/ci_status_reporter.py — Poste un status check sur un commit GitHub.

Utilise l'API GitHub Statuses :
  POST /repos/{owner}/{repo}/statuses/{sha}

Ce status check est visible dans l'onglet PR de GitHub sous le bouton Merge.
Quand la branch protection rule exige ce status (Settings → Branches →
Require status checks → "Code Auditor / PR Review"), le bouton Merge est
bloqué tant que le status n'est pas "success".

CORRECTIONS v2 :
  - Retry automatique (max 2 tentatives) si l'API répond avec 5xx
  - Log explicite quand GITHUB_TOKEN est manquant (erreur fréquente en local)
  - Constante STATUS_CONTEXT exportée pour que ci_runner.py puisse la référencer
  - Séparation claire entre post_status() (briques) et post_review_status() (logique métier)

Pattern : urllib.request stdlib — cohérent avec mcp_github_service.py,
          pas de dépendance externe (requests, httpx).
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

# Nom affiché dans GitHub UI sous l'onglet PR → Checks
# C'est CE nom qu'il faut cocher dans Settings → Branch protection rules
STATUS_CONTEXT = "Code Auditor / PR Review"

# Nombre de tentatives en cas d'erreur transitoire (429, 5xx)
_MAX_RETRIES = 2
_RETRY_DELAY = 3  # secondes


# ─────────────────────────────────────────────────────────────────────────────
# Récupération du token
# ─────────────────────────────────────────────────────────────────────────────

def _get_token() -> str:
    """
    Récupère le token GitHub.

    Priorité :
      1. GITHUB_TOKEN     → injecté automatiquement par GitHub Actions
      2. GITHUB_PERSONAL_ACCESS_TOKEN → token local depuis .env
      3. Fallback dotenv  → lecture du fichier .env

    Scopes requis pour post_status() : repo:status
    Dans GitHub Actions, le GITHUB_TOKEN injecté a ce scope par défaut.
    """
    # 1. Variable déjà présente dans l'environnement
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        return token

    token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    if token:
        return token

    # 2. Charger le .env (utile en développement local)
    try:
        from dotenv import load_dotenv
        load_dotenv()
        token = os.environ.get("GITHUB_TOKEN", "")
        if not token:
            token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    except ImportError:
        pass  # python-dotenv non installé — pas bloquant en Actions

    return token


# ─────────────────────────────────────────────────────────────────────────────
# Brique de base : POST /statuses/{sha}
# ─────────────────────────────────────────────────────────────────────────────

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
        owner:       Propriétaire du repo (ex: "chmaryem")
        repo:        Nom du repo (ex: "test-project")
        sha:         SHA complet du commit HEAD de la PR
        state:       "success" | "failure" | "pending" | "error"
        description: Message affiché dans GitHub UI (max 140 chars)
        target_url:  URL optionnelle vers les détails du run

    Returns:
        True si le status a été posté avec succès, False sinon.

    Note:
        En cas de sha tronqué (8 chars), GitHub accepte quand même.
        Mais un sha vide ("") retournera une 422 → on vérifie en amont.
    """
    if not sha:
        logger.warning("post_status: sha vide — status non posté")
        return False

    token = _get_token()
    if not token:
        logger.error(
            "post_status: aucun token GitHub disponible.\n"
            "  En local  : ajoutez GITHUB_PERSONAL_ACCESS_TOKEN dans votre .env\n"
            "  En Actions: vérifiez permissions: statuses: write dans le workflow"
        )
        return False

    url     = f"https://api.github.com/repos/{owner}/{repo}/statuses/{sha}"
    payload = {
        "state":       state,
        "description": description[:140],  # GitHub limite à 140 chars
        "context":     STATUS_CONTEXT,
    }
    if target_url:
        payload["target_url"] = target_url

    data    = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"token {token}",
        "Accept":        "application/vnd.github.v3+json",
        "Content-Type":  "application/json",
        "User-Agent":    "CodeAuditor-CI/2.0",
    }

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status in (200, 201):
                    logger.info(
                        "Status '%s' posté → %s/%s@%s : %s",
                        state, owner, repo, sha[:8], description,
                    )
                    return True
                logger.error("Status API: réponse inattendue %d", resp.status)
                return False

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:200]
            if e.code in (429, 500, 502, 503) and attempt < _MAX_RETRIES:
                logger.warning(
                    "Status API erreur %d (tentative %d/%d) — retry dans %ds",
                    e.code, attempt, _MAX_RETRIES, _RETRY_DELAY,
                )
                time.sleep(_RETRY_DELAY)
                continue
            logger.error("Status API erreur %d : %s", e.code, body)
            return False

        except Exception as e:
            logger.error("Status API erreur inattendue : %s", e)
            return False

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Logique métier : décision success/failure depuis le verdict du review
# ─────────────────────────────────────────────────────────────────────────────

def post_review_status(
    owner:    str,
    repo:     str,
    sha:      str,
    verdict:  str,
    score:    float,
    critical: int,
    high:     int,
    medium:   int,
) -> bool:
    """
    Poste le status check basé sur le résultat de review_pr().

    Logique identique à pr_review_agent.py (cohérence garantie) :
      verdict == "REQUEST_CHANGES" → failure  (bouton Merge bloqué)
      verdict == "COMMENT"         → success  (merge autorisé avec warning)
      verdict == "APPROVE"         → success  (merge autorisé)

    Le score et les counts sont affichés dans la description pour le dev.
    """
    if verdict == "REQUEST_CHANGES":
        state = "failure"
        icon  = "BLOQUE"
        desc  = f"[{icon}] Score {score:.0f} | C:{critical} H:{high} M:{medium} — MERGE BLOQUE"
    elif verdict == "COMMENT":
        state = "success"
        icon  = "WARN"
        desc  = f"[{icon}] Score {score:.0f} | C:{critical} H:{high} M:{medium} — Attention"
    else:  # APPROVE
        state = "success"
        icon  = "OK"
        desc  = f"[{icon}] Score {score:.0f} | C:{critical} H:{high} M:{medium} — Merge autorise"

    print(f"  Status check → {state} : {desc}")
    return post_status(owner, repo, sha, state, desc)