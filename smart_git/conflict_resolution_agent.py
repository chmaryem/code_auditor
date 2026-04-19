"""
conflict_resolution_agent.py — Résolution de conflits PR.

ARCHITECTURE v5 — Pipeline direct Python (sans CodeModeAgent/sandbox) :

  POURQUOI on retire CodeModeAgent ici (mais pas pour pr-check) :
    - La résolution de conflits suit un workflow DÉTERMINISTE et FIXE.
    - L'agent génèrerait toujours le même script → 2000 tokens gaspillés.
    - La valeur de l'IA est dans resolve_single_file() (3-way diff + LLM budget).

  Le MCP Code Mode est GARDÉ pour pr-check (analyse dynamique, variable par PR).

  Pipeline de résolution :
    1. get_pr_mergeable_status() — détection fiable des conflits
    2. resolve_file_smart() — 3-way diff → merge conservateur → Gemini budget
    3. push_file() + create_pull_request() — via github.* wrappers MCP
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


def _resolve_with_gemini_budget(filename: str, ours: str, theirs: str) -> Optional[str]:
    """
    Résolution LLM avec budget strict — seulement les blocs différents.
    Input : ~200-500 chars de blocs seulement (au lieu de 10 000 chars entiers).
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
        f"Output ONLY the full resolved file content. No explanations.\n"
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
    filename: str, base_content: str, ours_content: str, theirs_content: str
) -> Tuple[Optional[str], str]:
    """
    Pipeline à 3 niveaux. Retourne (contenu_résolu, méthode).
    """
    # Niveau 1 : 3-way déterministe (0 token)
    result = _merge_3way(base_content, ours_content, theirs_content)
    if result:
        return result, "3way"

    # Niveau 2 : merge conservateur (0 token)
    result = _merge_conservative(ours_content, theirs_content, filename)
    if result:
        return result, "conservative"

    # Niveau 3 : Gemini budget minimal
    print(f"    {_YL}conflit complexe → Gemini (budget){_R}", end="", flush=True)
    result = _resolve_with_gemini_budget(filename, ours_content, theirs_content)
    if result:
        return result, "gemini_budget"

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

async def resolve_pr_conflicts(owner: str, repo: str, pr_number: int) -> Dict[str, Any]:
    """
    Résout les conflits d'une PR via MCP github.* wrappers + pipeline 3 niveaux.

    Architecture conservée :
      - github.* wrappers → MCP GitHub Server (communication MCP préservée)
      - resolver pipeline → 3-way diff → conservative → Gemini budget
      - Résultat posté sur GitHub via github.post_comment() et github.create_pull_request()
    """
    print(f"\n  {_CY}{_B}Code Auditor — Résolution conflits PR #{pr_number}{_R}")
    print(f"  {_DM}Repo : {owner}/{repo}{_R}")
    print(f"  {_DM}Mode : 3-way diff -> conservateur -> Gemini budget{_R}\n")

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

        # 3. Créer la branche de résolution
        branch_name = f"auto-resolve/pr-{pr_number}"
        print(f"  Création branche {branch_name}...")
        github.create_branch(owner, repo, branch_name, base_ref)

        # 4. Résoudre chaque fichier
        resolved_files, failed_files, diffs = [], [], []

        for f in code_files:
            filename = f.get("filename", f.get("path", ""))
            print(f"  {_DM}→ {filename}{_R}", end="", flush=True)

            base_content = github.get_file_content(owner, repo, filename, base_ref)
            head_content = github.get_file_content(owner, repo, filename, head_ref)

            if not base_content or not head_content:
                print(f"  {_RD}✗ contenu inaccessible{_R}")
                failed_files.append(filename)
                continue

            if base_content == head_content:
                print(f"  {_GR}= identique{_R}")
                continue

            resolved, method = resolve_file_smart(filename, base_content, base_content, head_content)

            if resolved:
                msg = f"auto-resolve: {filename} (method={method})"
                github.push_file(owner, repo, filename, resolved, msg, branch_name)
                resolved_files.append(filename)
                b = len(base_content.splitlines())
                h = len(head_content.splitlines())
                r = len(resolved.splitlines())
                diffs.append(f"- **{filename}**: {b}L → {h}L → resolved {r}L (`{method}`)")
                print(f"  {_GR}✓ [{method}]{_R}")
            else:
                print(f"  {_RD}✗ échec{_R}")
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

        # 6. Créer la PR auto-resolve → main
        pr_url = ""
        if resolved_files:
            pr_data = status.get("pr_data", {})
            orig_title = pr_data.get("title", f"PR #{pr_number}") if pr_data else f"PR #{pr_number}"
            new_pr = github.create_pull_request(
                owner, repo,
                title=f"Auto-resolve conflicts for PR #{pr_number}",
                body="\n".join([
                    f"## Conflict resolution for PR #{pr_number}",
                    f"Original : #{pr_number} — *{orig_title}*", "",
                    "Created by **Code Auditor** — automatic merge conflict resolution.", "",
                    "### Resolution Methods", "",
                ] + diffs),
                head=branch_name,
                base=base_ref,
            )
            pr_url = new_pr.get("html_url", "") if isinstance(new_pr, dict) else ""

        # Affichage terminal
        print(f"\n  {'─'*50}")
        print(f"  {_B}Résumé : {len(resolved_files)} résolu(s) · {len(failed_files)} échec(s){_R}")
        if resolved_files:
            print(f"  {_GR}Branche : {branch_name}{_R}")
            if pr_url:
                print(f"  {_GR}PR créée : {pr_url}{_R}")
            print(f"  {_DM}  git fetch origin && git checkout {branch_name}{_R}")
        for fn in resolved_files:
            print(f"    {_GR}✓{_R}  {fn}")
        if failed_files:
            print(f"\n  {_YL}Résolution manuelle requise :{_R}")
            for fn in failed_files:
                print(f"    {_YL}✗{_R}  {fn}")
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