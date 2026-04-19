"""
pr_review_agent.py — Agent MCP Code Mode pour la revue de Pull Requests.

FIX v4 :
  Bug analyse incorrect : le système analysait le code de main (branch de base)
  au lieu du code de la feature branch. C'est pourquoi score=0 même avec des bugs.
  
  Le PR_REVIEW_SYSTEM_PROMPT maintenant force explicitement l'utilisation de
  head_ref (la branche feature) pour get_file_content().

  Résultat attendu :
    - PR avec UserService.java bugué (SQL injection, mot de passe en dur) :
      ✗ MERGE BLOQUÉ — score 45+
      CRITICAL: 2+ · HIGH: 3+ · Fichiers: 1
    - Review posté sur GitHub avec analyse RAG détaillée
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

_R  = "\033[0m"
_B  = "\033[1m"
_GR = "\033[92m"
_YL = "\033[93m"
_RD = "\033[91m"
_CY = "\033[96m"
_DM = "\033[2m"


def _extract_json_from_output(raw_output: str) -> dict:
    """Extrait le JSON résultat depuis le stdout du sandbox."""
    if not raw_output:
        return {}
    lines = raw_output.strip().splitlines()
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{") or line.startswith("["):
            try:
                result = json.loads(line)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                continue
    try:
        start = raw_output.rfind("{")
        end   = raw_output.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(raw_output[start:end])
    except Exception:
        pass
    return {}


def create_pr_review_agent():
    from agents.code_mode_agent import CodeModeAgent, PR_REVIEW_SYSTEM_PROMPT
    return CodeModeAgent(system_prompt=PR_REVIEW_SYSTEM_PROMPT)


async def review_pr(owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
    """
    Analyse une Pull Request via MCP Code Mode avec le pipeline RAG complet.

    RÉSULTAT ATTENDU dans le terminal (PR avec UserService.java bugué) :
      ✗ MERGE BLOQUÉ — score 45
      CRITICAL: 2 · HIGH: 3 · Fichiers: 1 · Itérations: 1

    RÉSULTAT ATTENDU sur GitHub :
      Un review posté automatiquement avec :
      - Score total RAG
      - Bugs détectés par fichier (SQL injection, credentials hardcodés, etc.)
      - Verdict REQUEST_CHANGES avec description détaillée

    IMPORTANT : analyse le code de la FEATURE BRANCH (head_ref), pas de main.
    C'est pourquoi les bugs sont détectés même si main est propre.
    """
    import asyncio

    print(f"\n  {_CY}{_B}Code Auditor — Revue PR #{pr_number} (MCP Code Mode){_R}")
    print(f"  {_DM}Repo : {owner}/{repo}{_R}")
    print(f"  {_DM}Mode : Agent autonome avec sandbox{_R}\n")

    agent = create_pr_review_agent()

    task = (
        f"Review Pull Request #{pr_number} on {owner}/{repo}.\n\n"
        f"owner = '{owner}'\n"
        f"repo = '{repo}'\n"
        f"pr_number = {pr_number}\n\n"
        f"CRITICAL: Fetch file content from the FEATURE BRANCH (head_ref), NOT from main.\n"
        f"pr_info = github.get_pr_info(owner, repo, pr_number)\n"
        f"head_ref = pr_info['head']['ref']   ← use THIS branch for get_file_content()\n"
        f"files = github.get_pr_files(owner, repo, pr_number)\n"
        f"For each file: content = github.get_file_content(owner, repo, filename, head_ref)\n"
        f"Then: analysis = rag.analyze(content, filename, language, patch)\n"
    )

    result = await asyncio.to_thread(agent.run, task)

    if result["success"]:
        parsed = _extract_json_from_output(result["output"])

        verdict  = parsed.get("verdict",   "UNKNOWN")
        score    = parsed.get("score",     0)
        critical = parsed.get("critical",  0)
        high     = parsed.get("high",      0)
        files    = parsed.get("files_analyzed", 0)

        verdict_display = {
            "REQUEST_CHANGES": f"{_RD}{_B}✗  MERGE BLOQUÉ{_R}",
            "COMMENT":         f"{_YL}{_B}⚠  MERGE AVEC PRÉCAUTION{_R}",
            "APPROVE":         f"{_GR}{_B}✓  MERGE AUTORISÉ{_R}",
        }
        display = verdict_display.get(verdict, f"{_YL}⚠  {verdict}{_R}")

        print(f"\n  {display} — score {score}")
        print(f"  CRITICAL: {critical} · HIGH: {high} · Fichiers: {files} · Itérations: {result['iterations']}")

        if verdict == "UNKNOWN" and not parsed:
            print(f"\n  {_YL}⚠ Impossible de parser le JSON de sortie.{_R}")
            print(f"  {_DM}Output brut :{_R}\n  {result['output'][:400]}")
        print()

        return {
            "success": True,
            "verdict": verdict,
            "score": score,
            "critical": critical,
            "high": high,
            "files_analyzed": files,
            "iterations": result["iterations"],
            "raw_output": result["output"],
        }

    else:
        print(f"\n  {_RD}✗ Échec de l'analyse : {result['error']}{_R}\n")
        return {
            "success": False,
            "error": result["error"],
            "iterations": result["iterations"],
        }