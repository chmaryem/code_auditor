"""
merge_automation_agent.py — Vérification de readiness de merge.

ARCHITECTURE v5 — Direct Python, 0 token LLM :

  Avant : CodeModeAgent générait un script (~2000 tokens) → sandbox l'exécutait.
  Après : Python direct → 0 token LLM consommé.

  Justification : la vérification de readiness est 100% factuelle.
    - mergeable : get_pr_mergeable_status()
    - CI/CD     : get_check_runs()
    - Reviews   : get_pr_reviews()
  Aucune intelligence artificielle nécessaire pour lire des booléens.
  Le MCP Code Mode est gardé pour pr-check (analyse de code — là l'IA apporte de la valeur).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

# Force UTF-8 sur Windows (évite UnicodeEncodeError cp1252)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_R  = "\033[0m"
_B  = "\033[1m"
_GR = "\033[92m"
_YL = "\033[93m"
_RD = "\033[91m"
_CY = "\033[96m"
_DM = "\033[2m"


async def check_merge_readiness(owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
    """
    Vérifie si une PR est prête à merger — aucun appel LLM.

    Utilise les MCP github.* wrappers (connexion MCP GitHub maintenue).
    Économie : ~2000 tokens (génération script) → 0 token.
    """
    print(f"\n  {_CY}{_B}Code Auditor — Vérification merge PR #{pr_number}{_R}")
    print(f"  {_DM}Repo : {owner}/{repo}{_R}")
    print(f"  {_DM}Mode : API GitHub directe (0 token LLM){_R}\n")

    try:
        from services.code_mode_client import github
    except ImportError:
        return {"success": False, "ready": False, "error": "code_mode_client non disponible"}

    try:
        # 1. Infos PR
        print(f"  Récupération PR #{pr_number}...")
        pr_info = github.get_pr_info(owner, repo, pr_number)
        if not pr_info:
            return {"success": False, "ready": False, "error": "PR introuvable"}

        pr_title = pr_info.get("title", f"PR #{pr_number}")
        head_sha  = pr_info.get("head", {}).get("sha", "")

        # 2. Statut mergeable (polling fiable)
        print(f"  Vérification conflits...")
        status = github.get_pr_mergeable_status(owner, repo, pr_number)
        has_conflicts = status.get("has_conflicts", False)
        mergeable     = not has_conflicts

        # 3. CI/CD checks
        sha_short = head_sha[:8] if head_sha else "N/A"
        print(f"  Vérification CI/CD (SHA: {sha_short})...")
        checks = github.get_check_runs(owner, repo, head_sha) if head_sha else []
        failing_checks = [
            c.get("name", "?") for c in checks
            if c.get("conclusion", "") not in ("success", "neutral", "skipped", "")
        ]
        ci_pass = len(failing_checks) == 0

        # 4. Reviews
        print(f"  Vérification reviews...")
        reviews          = github.get_pr_reviews(owner, repo, pr_number)
        reviews_approved = any(r.get("state") == "APPROVED"         for r in reviews)
        changes_req      = any(r.get("state") == "REQUEST_CHANGES"  for r in reviews)

        # 5. Verdict
        ready   = mergeable and ci_pass and not changes_req
        reasons = []
        if not mergeable:
            reasons.append("conflits à résoudre")
        if not ci_pass:
            reasons.append(f"CI/CD en échec : {', '.join(failing_checks[:3])}")
        if changes_req:
            reasons.append("review demande des modifications")
        if not reviews_approved and not changes_req:
            reasons.append("pas encore d'approbation")

        details = " · ".join(reasons) if reasons else "Tous les critères sont satisfaits"

        # 6. Construire le rapport Markdown
        lines = [
            f"## Merge Readiness Report — PR #{pr_number}: {pr_title}", "",
            "### 1. Statut Mergeable",
            f"- {'✅ Aucun conflit détecté.' if mergeable else '❌ Conflits présents — résoudre avant de merger.'}",
            "",
            f"### 2. CI/CD (SHA: {sha_short})",
        ]
        if not checks:
            lines.append("- ℹ️ Aucun check CI/CD configuré.")
        elif ci_pass:
            lines.append(f"- ✅ {len(checks)} check(s) passé(s) avec succès.")
        else:
            lines.append(f"- ❌ {len(failing_checks)} check(s) en échec : {', '.join(failing_checks)}")
        lines += [
            "",
            "### 3. Reviews",
        ]
        if not reviews:
            lines.append("- ℹ️ Aucune review soumise.")
        elif reviews_approved:
            n = sum(1 for r in reviews if r.get("state") == "APPROVED")
            lines.append(f"- ✅ Approuvée par {n} reviewer(s).")
        elif changes_req:
            lines.append("- ❌ Modifications demandées — traiter les commentaires de review.")
        else:
            lines.append("- ℹ️ Reviews existantes mais pas d'approbation.")
        lines += [
            "",
            "### Verdict global",
            f"- {'✅ **PRÊTE** : cette PR peut être mergée.' if ready else f'⚠️ **PAS PRÊTE** : {details}.'}",
        ]

        report = "\n".join(lines)
        github.post_comment(owner, repo, pr_number, report)

        # Affichage terminal
        if ready:
            print(f"\n  {_GR}{_B}✓  PR PRÊTE À MERGER{_R}")
        else:
            print(f"\n  {_RD}{_B}✗  PR PAS PRÊTE{_R}")
            print(f"  {_DM}{details}{_R}")

        print(f"  {_DM}Mergeable: {'Oui' if mergeable else 'Non'} | "
              f"CI: {'OK' if ci_pass else 'KO'} | "
              f"Approuvée: {'Oui' if reviews_approved else 'Non'}{_R}")
        print()

        return {
            "success": True,
            "ready": ready,
            "mergeable": mergeable,
            "ci_pass": ci_pass,
            "reviews_approved": reviews_approved,
            "details": details,
            "iterations": 0,   # 0 = pas d'appel LLM
        }

    except Exception as e:
        logger.error("check_merge_readiness failed: %s", e)
        print(f"\n  {_RD}✗ Erreur : {e}{_R}\n")
        return {"success": False, "ready": False, "error": str(e), "iterations": 0}
    finally:
        try:
            github.disconnect()
        except Exception:
            pass
