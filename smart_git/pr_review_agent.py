"""
pr_review_agent.py — Agent de revue de Pull Requests.

REFACTOR v7 — Direct Python (0 token CodeModeAgent) :

  AVANT (v4-v6) :
    CodeModeAgent génère un script Python (LLM #1 ~500-800 tokens)
    → Sandbox l'exécute
    → rag.analyze() appelle Gemini (LLM #2)
    TOTAL : 2 appels LLM par exécution + overhead de génération de script

  APRÈS (v7) :
    Appels directs github.* + rag.analyze() (LLM unique, 1 par fichier)
    → Même architecture que conflict_resolution_agent.py
    → Cache-first : 0 token LLM si fichier déjà en cache Watch
    TOTAL : 0-1 appel LLM par fichier (selon cache)

  Économie : -50% à -80% de consommation quota selon le cache.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

# Force UTF-8 sur Windows
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

CODE_EXTENSIONS = {".java", ".py", ".js", ".ts", ".jsx", ".tsx"}


def _detect_language(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return {
        ".py": "python", ".java": "java",
        ".js": "javascript", ".jsx": "javascript",
        ".ts": "typescript", ".tsx": "typescript",
    }.get(ext, "unknown")


async def review_pr(owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
    """
    Analyse une Pull Request via appels directs rag.analyze() (sans CodeModeAgent).

    Architecture v7 :
      1. github.get_pr_info()      → head_ref / base_ref
      2. github.get_pr_files()     → liste des fichiers modifiés
      3. Pour chaque fichier code :
           a. cache.read_analysis(filename)  [0 token si hit]
           b. Si miss : rag.analyze(content, filename, language, patch)  [1 token LLM]
           c. kg.detect_patterns(content, language)
      4. Calcul score total → verdict
      5. github.post_review() → review posté sur GitHub

    IMPORTANT : analyse le code de la FEATURE BRANCH (head_ref), pas de main.
    """
    import asyncio

    print(f"\n  {_CY}{_B}Code Auditor — Revue PR #{pr_number} (Direct Mode v7){_R}")
    print(f"  {_DM}Repo : {owner}/{repo}{_R}")
    print(f"  {_DM}Mode : Direct Python (0 overhead CodeModeAgent){_R}\n")

    try:
        from services.code_mode_client import github, rag, kg, cache
    except ImportError as e:
        print(f"  {_RD}✗ Erreur import code_mode_client : {e}{_R}\n")
        return {"success": False, "error": str(e)}

    try:
        # ── 1. Infos PR ──────────────────────────────────────────────────────
        print(f"  Récupération PR #{pr_number}...")
        pr_info = github.get_pr_info(owner, repo, pr_number)
        if not pr_info:
            return {"success": False, "error": "PR introuvable"}

        head_ref = pr_info.get("head", {}).get("ref", "")
        base_ref = pr_info.get("base", {}).get("ref", "main")
        pr_title = pr_info.get("title", f"PR #{pr_number}")
        print(f"  {_DM}head={head_ref}  base={base_ref}{_R}")

        # ── 2. Fichiers modifiés ──────────────────────────────────────────────
        print(f"  Récupération des fichiers...")
        files = github.get_pr_files(owner, repo, pr_number)
        code_files = [
            f for f in files
            if Path(f.get("filename", f.get("path", ""))).suffix.lower() in CODE_EXTENSIONS
        ]
        print(f"  {len(code_files)} fichier(s) à analyser sur {len(files)} modifiés\n")

        # ── 3. Analyse de chaque fichier ──────────────────────────────────────
        total_score    = 0.0
        total_critical = 0
        total_high     = 0
        total_medium   = 0
        files_analyzed = 0
        per_file_results: List[str] = []

        for f in code_files:
            filename = f.get("filename", f.get("path", ""))
            patch    = f.get("patch", "")
            language = _detect_language(filename)
            ext      = Path(filename).suffix.lstrip(".")

            print(f"  {_DM}→ {filename}{_R}", end="", flush=True)

            # a. Cache-first (0 token LLM si hit)
            cached_text = cache.read_analysis(filename)
            if cached_text:
                result = rag.count_severity(cached_text)
                result["source"] = "cache"
                print(f" {_GR}[cache]{_R}", end="", flush=True)
            else:
                # b. Télécharger le contenu de la feature branch
                content = github.get_file_content(owner, repo, filename, head_ref)
                if not content:
                    print(f" {_YL}[vide]{_R}")
                    continue

                # Validation : détecter contenu binaire/corrompu
                _printable = sum(1 for c in content[:1000] if c.isprintable() or c in '\n\r\t')
                if len(content) > 20 and _printable / min(len(content), 1000) < 0.80:
                    print(f" {_YL}[skip: contenu binaire/corrompu]{_R}")
                    continue

                # c. Analyse RAG complète (1 appel LLM)
                result = rag.analyze(content, filename, language, patch)
                print(f" {_DM}[rag score={result.get('score', 0):.0f}]{_R}", end="", flush=True)

            # d. Knowledge Graph patterns
            try:
                content_for_kg = (
                    github.get_file_content(owner, repo, filename, head_ref)
                    if not cached_text else cached_text
                )
                patterns = kg.detect_patterns(content_for_kg, language)
                if patterns:
                    print(f" [kg:{len(patterns)}]", end="", flush=True)
            except Exception:
                patterns = []

            score    = result.get("score",    0)
            critical = result.get("critical", 0)
            high     = result.get("high",     0)
            medium   = result.get("medium",   0)

            total_score    += score
            total_critical += critical
            total_high     += high
            total_medium   += medium
            files_analyzed += 1

            sev_icon = (
                f"{_RD}✗" if critical > 0 else
                f"{_YL}⚠" if high > 0     else
                f"{_GR}✓"
            ) + _R
            print(f" {sev_icon} C:{critical} H:{high} M:{medium}")

            # Résumé par fichier pour le review GitHub
            analysis_snippet = result.get("analysis", "")[:600].replace("\n", "  \n")
            per_file_results.append(
                f"### `{filename}` — score {score:.0f}\n"
                f"**CRITICAL**: {critical} | **HIGH**: {high} | **MEDIUM**: {medium}\n\n"
                + (f"**KG Patterns**: {', '.join(patterns[:5])}\n\n" if patterns else "")
                + f"<details><summary>Détails RAG</summary>\n\n{analysis_snippet}\n</details>\n"
            )

        # ── 4. Verdict ────────────────────────────────────────────────────────
        if total_critical > 0 or total_score >= 35:
            event = "REQUEST_CHANGES"
        elif total_score >= 15:
            event = "COMMENT"
        else:
            event = "APPROVE"

        # ── 5. Review body ────────────────────────────────────────────────────
        verdict_icon = {
            "REQUEST_CHANGES": "❌ MERGE BLOQUÉ",
            "COMMENT":         "⚠️ MERGE AVEC PRÉCAUTION",
            "APPROVE":         "✅ MERGE AUTORISÉ",
        }[event]

        body_lines = [
            f"## Code Auditor — PR Review #{pr_number}",
            f"> **PR** : {pr_title}",
            f"> **Verdict** : **{verdict_icon}**",
            f"> **Score global** : {total_score:.0f} "
            f"| CRITICAL: {total_critical} | HIGH: {total_high} | MEDIUM: {total_medium}",
            f"> **Fichiers analysés** : {files_analyzed} / {len(code_files)}",
            f"> **Pipeline** : RAG (ChromaDB + KnowledgeGraph + Multi-Model) — cache-first",
            "",
            "---",
            "",
        ] + per_file_results + [
            "---",
            "*Généré par Code Auditor v7 — [Direct Mode, 0 overhead CodeModeAgent]*",
        ]
        body = "\n".join(body_lines)

        # ── 6. Post Review ────────────────────────────────────────────────────
        print(f"\n  Posting review sur GitHub...")
        github.post_review(owner, repo, pr_number, body, event)

        # ── 7. Affichage terminal ─────────────────────────────────────────────
        verdict_display = {
            "REQUEST_CHANGES": f"{_RD}{_B}✗  MERGE BLOQUÉ{_R}",
            "COMMENT":         f"{_YL}{_B}⚠  MERGE AVEC PRÉCAUTION{_R}",
            "APPROVE":         f"{_GR}{_B}✓  MERGE AUTORISÉ{_R}",
        }[event]

        print(f"\n  {verdict_display} — score {total_score:.0f}")
        print(
            f"  CRITICAL: {total_critical} · HIGH: {total_high} · "
            f"MEDIUM: {total_medium} · Fichiers: {files_analyzed}"
        )
        print()

        return {
            "success":        True,
            "verdict":        event,
            "score":          total_score,
            "critical":       total_critical,
            "high":           total_high,
            "medium":         total_medium,
            "files_analyzed": files_analyzed,
            "iterations":     0,  # 0 = pas d'itération CodeModeAgent
            "head_sha":       pr_info.get("head", {}).get("sha", ""),
        }

    except Exception as e:
        logger.error("review_pr failed: %s", e)
        print(f"\n  {_RD}✗ Erreur : {e}{_R}\n")
        return {"success": False, "error": str(e), "iterations": 0}
    finally:
        try:
            github.disconnect()
        except Exception:
            pass