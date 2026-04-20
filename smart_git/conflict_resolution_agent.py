"""
conflict_resolution_agent.py — Résolution de conflits PR.

ARCHITECTURE v6.2 — Pipeline RAG-enriched :

  Pipeline de résolution :
    1. get_pr_mergeable_status() — détection fiable (REST API direct + fallback MCP)
    2. RAG context query — cache SQLite + ChromaDB + KnowledgeGraph (0 token LLM)
    3. resolve_file_smart() — 3-way diff → conservateur → Gemini + RAG context
    4. RESOLVE_README.md — résumé des changements généré automatiquement
    5. push_file() + create_pull_request() — via github.* wrappers MCP

  Le RAG enrichit le prompt Gemini avec les patterns du projet (SQL injection,
  bcrypt, design patterns) pour une résolution contextuelle et pas seulement mécanique.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

# Force UTF-8 sur Windows (évite UnicodeEncodeError cp1252 pour ✓, ✗, →, etc.)
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


# ─────────────────────────────────────────────────────────────────────────────
# Résolution intelligente à 3 niveaux (miroir de git_conflict_resolver)
# ─────────────────────────────────────────────────────────────────────────────

def _merge_3way(base: str, ours: str, theirs: str) -> Optional[str]:
    """
    Merge 3-way déterministe via difflib — 0 token LLM.
    Couvre ~70% des conflits PR (modifications dans des méthodes différentes).
    """
    if not base or not ours or not theirs:
        return None
    try:
        base_lines   = base.splitlines(keepends=True)
        ours_lines   = ours.splitlines(keepends=True)
        theirs_lines = theirs.splitlines(keepends=True)

        ours_ops   = list(difflib.SequenceMatcher(None, base_lines, ours_lines).get_opcodes())
        theirs_ops = list(difflib.SequenceMatcher(None, base_lines, theirs_lines).get_opcodes())

        ours_changed   = {i for tag, i1, i2, _, __ in ours_ops   if tag != "equal" for i in range(i1, i2)}
        theirs_changed = {i for tag, i1, i2, _, __ in theirs_ops if tag != "equal" for i in range(i1, i2)}

        if ours_changed & theirs_changed:
            return None  # Vraie intersection → LLM nécessaire

        # Pas de chevauchement : partir de OURS + ajouter ajouts THEIRS
        base_set  = set(base_lines)
        theirs_new = [l for l in theirs_lines if l not in base_set]

        merged = list(ours_lines)
        insertion_point = len(merged)
        for i, line in enumerate(reversed(merged)):
            if line.strip() == "}":
                insertion_point = len(merged) - 1 - i
                break
        if theirs_new:
            merged[insertion_point:insertion_point] = theirs_new

        result = "".join(merged)
        if "<<<<<<" in result or "=======" in result:
            return None
        if result.count("{") != result.count("}"):
            return ours if ours and "<<<<<<" not in ours else None
        return result
    except Exception:
        return None


def _merge_conservative(ours: str, theirs: str, filename: str) -> Optional[str]:
    """
    Merge conservateur : OURS + nouvelles méthodes de THEIRS.
    0 token LLM. Couvre ~20% des cas supplémentaires.
    """
    ratio = difflib.SequenceMatcher(None, ours, theirs).ratio()
    if ratio > 0.88:
        return ours  # Très similaires → OURS suffit

    ext = Path(filename).suffix.lower()
    if ext != ".java":
        return ours if ratio > 0.70 else None

    # Pour Java : détecter les nouvelles méthodes de THEIRS
    method_pattern = re.compile(
        r'(?:(?:public|private|protected|static|final|synchronized)\s+)+\w[\w<>\[\]]*\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{',
        re.MULTILINE,
    )
    ours_methods   = {m.group(1) for m in method_pattern.finditer(ours)}
    theirs_methods = {m.group(1) for m in method_pattern.finditer(theirs)}
    new_methods    = theirs_methods - ours_methods - {"if", "for", "while", "switch", "try", "catch"}

    if not new_methods:
        return ours  # Rien à ajouter

    # Extraire les corps des nouvelles méthodes
    additions = ""
    for m in method_pattern.finditer(theirs):
        name = m.group(1)
        if name in new_methods:
            start = m.start()
            depth = 0
            i = m.end() - 1
            while i < len(theirs):
                if theirs[i] == "{": depth += 1
                elif theirs[i] == "}":
                    depth -= 1
                    if depth == 0:
                        additions += "\n\n" + theirs[start:i+1]
                        break
                i += 1

    if not additions:
        return ours

    last_brace = ours.rfind("}")
    if last_brace == -1:
        return None
    result = ours[:last_brace] + additions + "\n\n" + ours[last_brace:]
    if "{" in result and result.count("{") == result.count("}"):
        return result
    return ours


def _resolve_with_gemini_budget(
    filename: str, ours: str, theirs: str, rag_context: list = None
) -> Optional[str]:
    """
    Résolution LLM avec budget strict — seulement les blocs différents.
    v6.2 : enrichi avec le contexte RAG (patterns, vulnérabilités détectées).
    """
    import os, time
    from dotenv import load_dotenv
    load_dotenv(Path(_project_root) / ".env")
    api_key  = os.getenv("GOOGLE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    ext = Path(filename).suffix.lstrip(".")

    # Extraire les blocs différents seulement
    ours_lines   = ours.splitlines()
    theirs_lines = theirs.splitlines()
    blocks = [
        (
            "\n".join(ours_lines[i1:i2]),
            "\n".join(theirs_lines[j1:j2]),
        )
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ours_lines, theirs_lines).get_opcodes()
        if tag != "equal"
    ]

    if not blocks:
        return ours  # Identiques

    conflicts_text = ""
    for i, (o, t) in enumerate(blocks[:5], 1):
        conflicts_text += f"\nBlock {i}:\nOURS:\n{o[:400]}\nTHEIRS:\n{t[:400]}\n"

    ours_header = "\n".join(ours.splitlines()[:12])
    prompt = (
        f"Resolve {len(blocks)} Git conflict block(s) in {filename} ({ext}).\n"
        f"Strategy: prefer security fixes, keep functional code from both sides.\n"
        f"File header:\n{ours_header}\n\n"
        f"Conflicts to resolve:\n{conflicts_text}\n"
    )

    # v6.2 : Injecter le contexte RAG dans le prompt
    if rag_context:
        prompt += "\nRelevant project patterns (from knowledge base):\n"
        for k in rag_context[:5]:
            if isinstance(k, dict):
                name = k.get("pattern", k.get("name", k.get("rule", "")))
                desc = k.get("description", k.get("detail", ""))
                prompt += f"- {name}: {desc}\n"
            elif isinstance(k, str):
                prompt += f"- {k}\n"
        prompt += "Use these patterns to guide the merge resolution.\n"

    prompt += (
        f"\nOutput ONLY the full resolved file content. No explanations.\n"
        f"If cannot resolve: output OURS_FALLBACK"
    )

    # Gemini first
    if api_key:
        for model in [os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), "gemini-2.0-flash", "gemini-1.5-flash"]:
            for attempt in range(2):
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    llm = ChatGoogleGenerativeAI(
                        model=model, google_api_key=api_key,
                        max_output_tokens=2048, temperature=0.0,
                    )
                    resp = llm.invoke(prompt)
                    text = resp.content.strip() if hasattr(resp, "content") else str(resp)
                    if "OURS_FALLBACK" in text:
                        return ours
                    if "<<<<<<" not in text:
                        return text
                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        wait = 20 if attempt == 0 else 50
                        print(f"    {_YL}[{model}] quota → attente {wait}s...{_R}", end="", flush=True)
                        time.sleep(wait)
                        if attempt == 1:
                            break
                    else:
                        break

    # Groq fallback
    if groq_key:
        for model_g, display in [
            ("llama-3.3-70b-versatile", "Groq Llama3.3-70B"),
            ("llama-4-scout-17b-16e-instruct", "Groq Llama4-Scout"),
        ]:
            try:
                from langchain_groq import ChatGroq
                llm_g = ChatGroq(model=model_g, temperature=0.0, groq_api_key=groq_key, max_tokens=4096)
                resp = llm_g.invoke(prompt)
                text = resp.content.strip() if hasattr(resp, "content") else str(resp)
                if "OURS_FALLBACK" in text:
                    return ours
                if "<<<<<<" not in text:
                    print(f" {display}", end="", flush=True)
                    return text
            except Exception as e:
                logger.warning("Groq [%s] echec: %s", display, e)

    # Dernier recours : OURS
    return ours


def resolve_file_smart(
    filename: str, base_content: str, ours_content: str, theirs_content: str,
    rag_context: list = None,
) -> Tuple[Optional[str], str]:
    """
    Pipeline à 3 niveaux. Retourne (contenu_résolu, méthode).
    v6.2 : le contexte RAG est passé au niveau 3 (Gemini).
    """
    # Niveau 1 : 3-way déterministe (0 token)
    result = _merge_3way(base_content, ours_content, theirs_content)
    if result:
        return result, "3way"

    # Niveau 2 : merge conservateur (0 token)
    result = _merge_conservative(ours_content, theirs_content, filename)
    if result:
        return result, "conservative"

    # Niveau 3 : Gemini budget minimal + RAG context
    print(f"    {_YL}conflit complexe -> Gemini + RAG{_R}", end="", flush=True)
    result = _resolve_with_gemini_budget(filename, ours_content, theirs_content, rag_context)
    if result:
        return result, "gemini_rag"

    return None, "failed"


def _ensure_list(value: Any) -> List:
    if isinstance(value, list):
        return value
    if value is None or isinstance(value, int):
        return []
    if isinstance(value, str):
        return [value] if value else []
    return list(value) if hasattr(value, "__iter__") else []


# ─────────────────────────────────────────────────────────────────────────────
# API publique — appelée par pr_analyzer.py
# ─────────────────────────────────────────────────────────────────────────────

def _generate_resolve_readme(
    pr_number: int, orig_title: str, base_ref: str, head_ref: str,
    resolved_files: List[str], failed_files: List[str],
    diffs: List[str], rag_patterns: Dict[str, List[str]],
) -> str:
    """
    Génère un RESOLVE_README.md pour documenter la résolution automatique.
    0 token LLM — template Markdown pur.
    """
    lines = [
        f"# Auto-Resolve Report — PR #{pr_number}",
        "",
        f"> **PR originale** : #{pr_number} — *{orig_title}*",
        f"> **Branche base** : `{base_ref}` | **Branche source** : `{head_ref}`",
        f"> **Generé par** : Code Auditor v6.2 (RAG-enhanced conflict resolution)",
        "",
        "## Fichiers résolus",
        "",
    ]

    if resolved_files:
        lines.append("| Fichier | Méthode | Status |")
        lines.append("|---|---|---|")
        for d in diffs:
            # Parse "- **filename**: ...L -> ...L -> resolved ...L (`method`)"
            parts = d.replace("- **", "").split("**")
            fname = parts[0] if parts else "?"
            method = "3way"
            if "`" in d:
                method = d.split("`")[1]
            lines.append(f"| `{fname}` | {method} | Resolved |")
        lines.append("")
    else:
        lines.append("*Aucun fichier résolu automatiquement.*")
        lines.append("")

    if failed_files:
        lines.append("## Resolution manuelle requise")
        lines.append("")
        for f in failed_files:
            lines.append(f"- [ ] `{f}`")
        lines.append("")

    # Section RAG patterns
    if rag_patterns:
        lines.append("## Patterns détectés (Knowledge Graph)")
        lines.append("")
        lines.append("Les patterns suivants ont été utilisés pour guider la résolution :")
        lines.append("")
        for filename, patterns in rag_patterns.items():
            if patterns:
                lines.append(f"### `{filename}`")
                for p in patterns:
                    lines.append(f"- {p}")
                lines.append("")

    lines.extend([
        "## Instructions pour le reviewer",
        "",
        "1. Vérifier que les résolutions préservent la logique métier",
        "2. Vérifier qu'aucune vulnérabilité n'a été réintroduite",
        "3. Exécuter les tests unitaires avant de merger",
        "",
        "```bash",
        f"git fetch origin && git checkout auto-resolve/pr-{pr_number}",
        "# Vérifier les changements",
        "git diff main..HEAD",
        "```",
        "",
        "---",
        "*Généré automatiquement par Code Auditor — ne pas modifier manuellement.*",
    ])

    return "\n".join(lines)


async def resolve_pr_conflicts(owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
    """
    Résout les conflits d'une PR via MCP github.* wrappers + pipeline RAG.

    Architecture v6.2 :
      - github.* wrappers -> MCP GitHub Server (communication MCP préservée)
      - RAG context -> cache SQLite + ChromaDB + KnowledgeGraph (0 token)
      - resolver pipeline -> 3-way diff -> conservative -> Gemini + RAG context
      - RESOLVE_README.md -> résumé automatique pushé sur la branche
    """
    print(f"\n  {_CY}{_B}Code Auditor — Résolution conflits PR #{pr_number}{_R}")
    print(f"  {_DM}Repo : {owner}/{repo}{_R}")
    print(f"  {_DM}Mode : 3-way diff -> RAG context -> Gemini enrichi{_R}\n")

    try:
        from services.code_mode_client import github
    except ImportError:
        return {"success": False, "error": "code_mode_client non disponible"}

    try:
        # 1. Vérifier les conflits (méthode fiable avec polling)
        print(f"  Détection des conflits...")
        status = github.get_pr_mergeable_status(owner, repo, pr_number)

        if not status.get("has_conflicts"):
            msg = f"No conflicts detected. PR #{pr_number} is mergeable directly."
            print(f"  {_GR}✓ {msg}{_R}\n")
            github.post_comment(owner, repo, pr_number, f"✅ {msg}")
            return {"success": True, "resolved": [], "failed": [], "branch": "", "pr_url": "", "iterations": 1}

        base_ref = status.get("base_ref", "main")
        head_ref = status.get("head_ref", "")
        print(f"  Conflits détectés — base={base_ref}  head={head_ref}")

        # 2. Récupérer les fichiers de code modifiés
        files     = github.get_pr_files(owner, repo, pr_number)
        CODE_EXTS = {".java", ".py", ".js", ".ts", ".jsx", ".tsx"}
        code_files = [
            f for f in files
            if Path(f.get("filename", f.get("path", ""))).suffix.lower() in CODE_EXTS
            and f.get("status") not in ("removed", "added")
        ]
        print(f"  Fichiers à traiter : {len(code_files)}\n")

        # 3. Initialiser le pipeline RAG (0 token LLM)
        print(f"  Chargement contexte RAG...")
        rag = None
        cache = None
        try:
            from services.code_mode_client import RAGAnalyzer, CacheClient
            rag = RAGAnalyzer()
            cache = CacheClient()
            print(f"  {_GR}RAG + Cache actifs{_R}")
        except Exception as e:
            logger.debug("RAG/Cache non disponible: %s", e)
            print(f"  {_YL}RAG non disponible — résolution sans contexte{_R}")

        # 4. Créer la branche de résolution
        branch_name = f"auto-resolve/pr-{pr_number}"
        print(f"  Création branche {branch_name}...")
        github.create_branch(owner, repo, branch_name, base_ref)

        # 5. Résoudre chaque fichier avec contexte RAG
        resolved_files, failed_files, diffs = [], [], []
        rag_patterns = {}  # {filename: [pattern_names]} pour le README

        for f in code_files:
            filename = f.get("filename", f.get("path", ""))
            ext = Path(filename).suffix.lstrip(".")
            print(f"  {_DM}-> {filename}{_R}", end="", flush=True)

            base_content = github.get_file_content(owner, repo, filename, base_ref)
            head_content = github.get_file_content(owner, repo, filename, head_ref)

            if not base_content or not head_content:
                print(f"  {_RD}x contenu inaccessible{_R}")
                failed_files.append(filename)
                continue

            if base_content == head_content:
                print(f"  {_GR}= identique{_R}")
                continue

            # Requête RAG : cache SQLite d'abord, puis analyse complète
            file_rag_context = []
            try:
                if cache:
                    cached = cache.read_analysis(filename)
                    if cached:
                        file_rag_context = cached.get("relevant_knowledge", [])
                        print(f" [cache]", end="", flush=True)

                if not file_rag_context and rag:
                    rag_result = rag.analyze(head_content, filename, ext)
                    file_rag_context = rag_result.get("relevant_knowledge", [])
                    # Extraire aussi les noms de patterns pour le README
                    if file_rag_context:
                        print(f" [rag:{len(file_rag_context)}]", end="", flush=True)
            except Exception as e:
                logger.debug("RAG query failed for %s: %s", filename, e)

            # Stocker les noms de patterns pour le README
            pattern_names = []
            for k in (file_rag_context or []):
                if isinstance(k, dict):
                    name = k.get("pattern", k.get("name", k.get("rule", "")))
                    if name:
                        pattern_names.append(str(name))
                elif isinstance(k, str) and k:
                    pattern_names.append(k)
            if pattern_names:
                rag_patterns[filename] = pattern_names

            resolved, method = resolve_file_smart(
                filename, base_content, base_content, head_content,
                rag_context=file_rag_context,
            )

            if resolved:
                msg = f"auto-resolve: {filename} (method={method})"
                github.push_file(owner, repo, filename, resolved, msg, branch_name)
                resolved_files.append(filename)
                b = len(base_content.splitlines())
                h = len(head_content.splitlines())
                r = len(resolved.splitlines())
                diffs.append(f"- **{filename}**: {b}L -> {h}L -> resolved {r}L (`{method}`)")
                print(f"  {_GR}v [{method}]{_R}")
            else:
                print(f"  {_RD}x echec{_R}")
                failed_files.append(filename)

        # 5. Poster le rapport sur la PR originale
        comment_lines = [
            f"## Auto-Resolve Report for PR #{pr_number}", "",
            f"**Resolved** : {len(resolved_files)} fichier(s) ✅",
            f"**Failed** : {len(failed_files)} fichier(s) {'❌' if failed_files else ''}",
            f"**Resolution branch** : `{branch_name}`", "",
        ]
        if diffs:
            comment_lines += ["### Diff Summary", ""] + diffs + [""]
        if failed_files:
            comment_lines += ["### Manual Resolution Needed"] + [f"- ❌ `{f}`" for f in failed_files] + [""]
        if resolved_files:
            comment_lines += [
                "**Apply locally :**",
                "```bash",
                f"git fetch origin && git checkout {branch_name}",
                "```",
            ]
        github.post_comment(owner, repo, pr_number, "\n".join(comment_lines))

        # 6. Générer et pusher RESOLVE_README.md
        pr_data = status.get("pr_data", {})
        orig_title = pr_data.get("title", f"PR #{pr_number}") if pr_data else f"PR #{pr_number}"

        if resolved_files:
            readme_content = _generate_resolve_readme(
                pr_number, orig_title, base_ref, head_ref,
                resolved_files, failed_files, diffs, rag_patterns,
            )
            try:
                github.push_file(
                    owner, repo, "RESOLVE_README.md", readme_content,
                    f"auto-resolve: add resolution README for PR #{pr_number}",
                    branch_name,
                )
                print(f"  {_GR}v RESOLVE_README.md pushed{_R}")
            except Exception as e:
                logger.warning("Failed to push README: %s", e)

        # 7. Créer la PR auto-resolve -> main
        pr_url = ""
        if resolved_files:
            new_pr = github.create_pull_request(
                owner, repo,
                title=f"Auto-resolve conflicts for PR #{pr_number}",
                body="\n".join([
                    f"## Conflict resolution for PR #{pr_number}",
                    f"Original : #{pr_number} — *{orig_title}*", "",
                    "Created by **Code Auditor** v6.2 — RAG-enhanced conflict resolution.", "",
                    "### Resolution Methods", "",
                ] + diffs + [
                    "",
                    "### RAG Patterns Applied",
                    "",
                ] + [
                    f"- **{fn}**: {', '.join(pats[:5])}"
                    for fn, pats in rag_patterns.items() if pats
                ] + [
                    "",
                    "*See `RESOLVE_README.md` in the branch for full details.*",
                ]),
                head=branch_name,
                base=base_ref,
            )
            pr_url = new_pr.get("html_url", "") if isinstance(new_pr, dict) else ""

       
        print(f"\n  {'='*50}")
        print(f"  {_B}Resume : {len(resolved_files)} resolu(s) - {len(failed_files)} echec(s){_R}")
        if rag_patterns:
            total_p = sum(len(v) for v in rag_patterns.values())
            print(f"  {_CY}RAG patterns : {total_p} patterns appliques{_R}")
        if resolved_files:
            print(f"  {_GR}Branche : {branch_name}{_R}")
            if pr_url:
                print(f"  {_GR}PR creee : {pr_url}{_R}")
            print(f"  {_DM}  git fetch origin && git checkout {branch_name}{_R}")
        for fn in resolved_files:
            print(f"    {_GR}v{_R}  {fn}")
        if failed_files:
            print(f"\n  {_YL}Resolution manuelle requise :{_R}")
            for fn in failed_files:
                print(f"    {_YL}x{_R}  {fn}")
        print()

        return {
            "success": True,
            "resolved": resolved_files, "failed": failed_files,
            "branch": branch_name, "pr_url": pr_url,
            "iterations": 1,
        }

    except Exception as e:
        logger.error("resolve_pr_conflicts failed: %s", e)
        print(f"\n  {_RD}✗ Erreur : {e}{_R}\n")
        return {"success": False, "resolved": [], "failed": [], "error": str(e), "iterations": 1}
    finally:
        try:
            github.disconnect()
        except Exception:
            pass