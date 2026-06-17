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
import re
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


# ── Inline review helpers (commentaires ancrés à la ligne + blocs suggestion) ──

_SEV_ICON  = {"critical": "🔴", "error": "🟠", "warning": "🟡", "info": "🔵"}
_SEV_LABEL = {"critical": "CRITICAL", "error": "HIGH", "warning": "MEDIUM", "info": "INFO"}
_SEV_RANK  = {"critical": 0, "error": 1, "warning": 2, "info": 3}

# Cap pour éviter un review GitHub géant (l'API rejette au-delà de ~50 commentaires)
_MAX_INLINE_COMMENTS = 40


def _diff_added_lines(patch: str) -> set:
    """
    Retourne l'ensemble des numéros de ligne (côté NOUVEAU fichier) présents
    dans le diff de la PR.

    GitHub n'accepte un commentaire inline QUE sur une ligne faisant partie du
    diff (ajout '+' ou contexte ' '). Une ancre hors-diff fait rejeter TOUT le
    review par l'API — d'où ce filtre de sécurité.
    """
    lines: set = set()
    if not patch:
        return lines
    new_ln = 0
    for raw in patch.splitlines():
        if raw.startswith("@@"):
            m = re.search(r"\+(\d+)", raw)
            new_ln = int(m.group(1)) if m else 0
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            lines.add(new_ln)
            new_ln += 1
        elif raw.startswith("-") and not raw.startswith("---"):
            continue  # suppression : n'avance pas le compteur du nouveau fichier
        else:
            new_ln += 1  # ligne de contexte
    return lines


def _parse_findings(analysis_text: str):
    """
    Extrait et valide (issues, fixes) du bloc <STRUCTURED_OUTPUT>.

    Pipeline :
      1. parse_structured_output() → dict brut
      2. Validation Pydantic → coerce les types + rejette les champs manquants
      3. Si parsing échoue → log + retourne listes vides (jamais de crash silencieux)
    """
    from typing import List, Optional
    try:
        from pydantic import BaseModel, Field as PField, validator
    except ImportError:
        from pydantic.v1 import BaseModel, Field as PField, validator  # type: ignore

    # Pydantic schemas — tolérant sur les champs optionnels
    class _Fix(BaseModel):
        title:        str            = ""
        explanation:  str            = ""
        line:         Optional[int]  = None
        apply_mode:   str            = "replace_snippet"
        current_code: str            = ""
        fixed_code:   str            = ""

    class _Issue(BaseModel):
        title:      str            = ""
        message:    str            = ""
        line:       Optional[int]  = None
        severity:   str            = "warning"
        rule:       str            = ""
        suggestion: str            = ""

        @validator("severity", pre=True, always=True)
        def _normalise_sev(cls, v):
            return (v or "warning").lower().strip()

    class _Output(BaseModel):
        issues: List[_Issue] = PField(default_factory=list)
        fixes:  List[_Fix]   = PField(default_factory=list)

    try:
        from langchain_agents.agents.lc_analysis_agent import parse_structured_output
    except Exception:
        return [], []

    raw = parse_structured_output(analysis_text or "") or {}
    if not raw:
        logger.warning("parse_findings: <STRUCTURED_OUTPUT> absent ou vide")
        return [], []

    try:
        validated = _Output.model_validate(raw)
    except Exception:
        try:
            validated = _Output.parse_obj(raw)          # pydantic v1 fallback
        except Exception as exc:
            logger.warning("parse_findings: validation Pydantic échouée: %s", exc)
            return [], []

    return (
        [i.dict() for i in validated.issues],
        [f.dict() for f in validated.fixes],
    )


def _compute_score_breakdown(findings: list) -> dict:
    """
    Calcule un breakdown multi-axe du score de qualité à partir des findings.

    Axes :
      security       — findings liés aux vulnérabilités (injection, auth, secrets…)
      quality        — problèmes de logique, nullability, types
      performance    — N+1, boucles, cache, mémoire
      maintainability — complexité, duplication, conventions

    Score 0–10 par axe (10 = parfait, 0 = critique).
    """
    _SEC  = {"injection", "xss", "csrf", "auth", "credential", "secret", "token",
              "password", "sanitiz", "escape", "insecur", "vuln", "exploit"}
    _PERF = {"performance", "n+1", "loop", "complex", "memory", "cache",
              "timeout", "latency", "query", "index"}
    _MAINT = {"duplication", "naming", "convention", "readability", "dead_code",
               "magic_number", "coupling", "cohesion"}

    sev_w = {"critical": 4.0, "error": 2.0, "warning": 0.8, "info": 0.2}

    sec_pen = qual_pen = perf_pen = maint_pen = 0.0

    for f in findings:
        w    = sev_w.get(f.get("severity", "info"), 0.0)
        rule = (f.get("rule", "") + " " + f.get("title", "")).lower()
        if any(kw in rule for kw in _SEC):
            sec_pen += w * 1.5
        elif any(kw in rule for kw in _PERF):
            perf_pen += w
        elif any(kw in rule for kw in _MAINT):
            maint_pen += w
        else:
            qual_pen += w

    def _score(penalty: float) -> int:
        return max(0, min(10, round(10 - penalty)))

    return {
        "security":        _score(sec_pen),
        "quality":         _score(qual_pen),
        "performance":     _score(perf_pen),
        "maintainability": _score(maint_pen),
    }


def _build_findings_and_comments(filename: str, issues: list, fixes: list, diff_lines: set):
    """
    Construit deux sorties à partir des issues/fixes structurés :
      - findings[] : pour le dashboard (TOUTES les issues, in_diff ou non)
      - comments[] : pour GitHub (seulement les issues ancrables dans le diff,
                     avec un bloc ```suggestion``` committable si un fix existe)

    Apparie chaque issue à son fix d'abord par ligne, sinon positionnellement
    (le prompt garantit issues[i] ↔ fixes[i]).
    """
    fixes_by_line: Dict[int, dict] = {}
    for fx in fixes:
        ln = fx.get("line")
        if ln is not None and ln not in fixes_by_line:
            fixes_by_line[ln] = fx

    findings, comments = [], []
    for idx, iss in enumerate(issues):
        line = iss.get("line")
        sev  = (iss.get("severity") or "warning").lower()
        fix  = fixes_by_line.get(line) or (fixes[idx] if idx < len(fixes) else None)
        fix  = fix or {}
        fixed_code   = fix.get("fixed_code", "")
        current_code = fix.get("current_code", "")
        apply_mode   = fix.get("apply_mode", "replace_snippet")
        in_diff = bool(line) and line in diff_lines

        findings.append({
            "file":         filename,
            "line":         line,
            "severity":     sev,
            "title":        iss.get("title", ""),
            "message":      iss.get("message", ""),
            "rule":         iss.get("rule", ""),
            "hint":         iss.get("suggestion", ""),
            "current_code": current_code,
            "fixed_code":   fixed_code,
            "apply_mode":   apply_mode,
            "in_diff":      in_diff,
        })

        if not in_diff:
            continue

        icon    = _SEV_ICON.get(sev, "🔵")
        label   = _SEV_LABEL.get(sev, sev.upper())
        body    = f"{icon} **{label}** · {iss.get('title', '')}\n\n{iss.get('message', '')}"
        comment = {"path": filename, "line": line, "side": "RIGHT"}

        # Étendue multi-ligne : si current_code couvre N lignes, ancrer la plage
        # entière (start_line..line) pour que la suggestion remplace tout le bloc.
        n_lines  = len(current_code.splitlines()) if current_code else 1
        span_ok  = all((line + k) in diff_lines for k in range(n_lines))

        if fixed_code and apply_mode in ("replace_snippet", "replace_method") and span_ok:
            if n_lines > 1:
                comment["start_line"] = line
                comment["start_side"] = "RIGHT"
                comment["line"]       = line + n_lines - 1
            body += f"\n\n```suggestion\n{fixed_code}\n```"
        elif iss.get("suggestion"):
            body += f"\n\n> 💡 {iss['suggestion']}"

        comment["body"] = body
        comments.append(comment)

    return findings, comments


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
        all_findings:  List[Dict[str, Any]] = []   # pour le dashboard
        all_comments:  List[Dict[str, Any]] = []   # pour GitHub (inline)

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
                # b. Télécharger le contenu de la feature branch (fichier entier)
                content = github.get_file_content(owner, repo, filename, head_ref, max_chars=0)
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
                    github.get_file_content(owner, repo, filename, head_ref, max_chars=0)
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

            # Findings structurés (ancrés à la ligne) → review inline + dashboard
            issues, fixes = _parse_findings(result.get("analysis", ""))
            f_findings, f_comments = _build_findings_and_comments(
                filename, issues, fixes, _diff_added_lines(patch)
            )
            all_findings.extend(f_findings)
            all_comments.extend(f_comments)

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

        # ── 6. Post Review (avec commentaires inline ancrés à la ligne) ───────
        inline_comments = all_comments[:_MAX_INLINE_COMMENTS]
        print(f"\n  Posting review sur GitHub ({len(inline_comments)} commentaire(s) inline)...")
        github.post_review(owner, repo, pr_number, body, event, comments=inline_comments)

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

        all_findings.sort(key=lambda f: _SEV_RANK.get(f.get("severity", "info"), 9))

        return {
            "success":         True,
            "verdict":         event,
            "score":           total_score,
            "score_breakdown": _compute_score_breakdown(all_findings),
            "critical":        total_critical,
            "high":            total_high,
            "medium":          total_medium,
            "files_analyzed":  files_analyzed,
            "iterations":      0,
            "head_sha":        pr_info.get("head", {}).get("sha", ""),
            "head_ref":        head_ref,
            "body":            body,
            "findings":        all_findings,
            "inline_count":    len(all_comments[:_MAX_INLINE_COMMENTS]),
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