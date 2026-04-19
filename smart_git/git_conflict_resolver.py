"""
git_conflict_resolver.py — Résolution automatique de conflits Git via LLM.

FIX v5 — Budget token contrôlé + cascade de modèles :

  PROBLÈME : les prompts envoyaient les 2 fichiers entiers (5000 chars chacun)
  + fichier conflicté (7000 chars) = ~17 000 chars (~4250 tokens) PAR fichier.
  Pour 2 fichiers en conflit = 8500 tokens → quota dépassé.

  SOLUTION :
  1. Tronquer les contenus :
     - OURS    : 2000 chars max (était 5000)
     - THEIRS  : 2000 chars max (était 5000)  
     - Conflicté : 3000 chars max (était 7000)
     → Réduction : 17 000 → 7 000 chars (~58% d'économie)

  2. max_output_tokens = 1024 (était 8192)
     → Gemini donne une réponse concise, pas 8000 tokens de code commenté

  3. Cascade de modèles : gemini-2.5-flash → gemini-2.0-flash → gemini-1.5-flash
     Chaque modèle a son propre quota indépendant.
     Backoff : 20s → 45s entre retries par modèle.

  4. Prompt condensé : stratégie en 1 ligne au lieu de 8 lignes.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

_R  = "\033[0m"
_B  = "\033[1m"
_GR = "\033[92m"
_YL = "\033[93m"
_RD = "\033[91m"
_CY = "\033[96m"
_DM = "\033[2m"

MAX_ATTEMPTS = 3

# Cascade de modèles (quota indépendant par modèle)
_LLM_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

# Budget tokens en chars
_BUDGET_OURS     = 2000
_BUDGET_THEIRS   = 2000
_BUDGET_CONFLICT = 3000
_MAX_OUTPUT      = 1024


def detect_conflict_files(project_path: Path) -> List[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=U"],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=str(project_path)
    )
    if result.returncode != 0:
        return []
    return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]


def has_conflict_markers(content: str) -> bool:
    return "<<<<<<" in content and "=======" in content and ">>>>>>>" in content


def get_ours_theirs(file_path: str, project_path: Path) -> Tuple[str, str]:
    ours = subprocess.run(
        ["git", "show", f":2:{file_path}"],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=str(project_path)
    )
    theirs = subprocess.run(
        ["git", "show", f":3:{file_path}"],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=str(project_path)
    )
    return (
        ours.stdout if ours.returncode == 0 else "",
        theirs.stdout if theirs.returncode == 0 else "",
    )


def _collect_imports_from_content(content: str) -> Set[str]:
    imports = set()
    for m in re.finditer(r"^import\s+(.+?)[\s;]*$", content, re.MULTILINE):
        imports.add(m.group(1).strip())
    for m in re.finditer(r"^from\s+(\S+)\s+import", content, re.MULTILINE):
        imports.add(m.group(1).strip())
    return imports


def _build_resolution_prompt(
    file_path: str,
    conflicted_content: str,
    ours_content: str,
    theirs_content: str,
    project_context: str = "",
    attempt: int = 1,
) -> str:
    """
    Prompt réduit de ~70% vs v4.
    Contenus tronqués, stratégie condensée en 1 ligne.
    """
    language = Path(file_path).suffix.lstrip(".")

    strategies = {
        1: "Combine OURS + THEIRS. For same method modified differently → keep more secure/complete version.",
        2: "Use OURS as base. Only add NEW methods from THEIRS not already in OURS. Do NOT modify OURS code.",
        3: "Return OURS exactly as-is. ONLY append brand-new methods from THEIRS at the end of the class.",
    }
    strategy = strategies.get(attempt, strategies[1])

    ours_trunc    = (ours_content or "")[:_BUDGET_OURS]
    theirs_trunc  = (theirs_content or "")[:_BUDGET_THEIRS]
    conflict_trunc = conflicted_content[:_BUDGET_CONFLICT]

    ctx_section = f"\nPROJECT CONTEXT:\n{project_context[:500]}\n" if project_context else ""

    return (
        f"Resolve this Git merge conflict in {file_path} ({language}).\n\n"
        f"STRATEGY (attempt {attempt}/3): {strategy}\n\n"
        f"PRIORITY: 1.Compilation 2.No-invented-imports 3.Security 4.Functionality\n"
        f"FORBIDDEN: add new imports, rename methods, change signatures.\n"
        f"{ctx_section}\n"
        f"OURS:\n```{language}\n{ours_trunc}\n```\n\n"
        f"THEIRS:\n```{language}\n{theirs_trunc}\n```\n\n"
        f"CONFLICTED:\n```\n{conflict_trunc}\n```\n\n"
        f"OUTPUT: resolved file content ONLY. No explanation. No markdown fences.\n"
        f"If cannot resolve safely: output exactly CANNOT_AUTO_RESOLVE"
    )


def _validate_resolved(
    resolved: str,
    file_path: str,
    ours_content: str,
    theirs_content: str,
    project_deps: Set[str] = None,
) -> Tuple[bool, str]:
    if has_conflict_markers(resolved):
        return False, "Conflict markers still present"
    if not resolved.strip():
        return False, "Empty file"

    ext = Path(file_path).suffix.lower()

    # Vérification imports
    resolved_imports = _collect_imports_from_content(resolved)
    allowed = _collect_imports_from_content(ours_content or "") | _collect_imports_from_content(theirs_content or "")
    std_java = {"java.util", "java.io", "java.lang", "java.sql", "java.net", "java.time", "javax.sql"}
    std_py   = {"os", "sys", "re", "json", "pathlib", "typing", "datetime", "logging", "collections"}
    invented = []
    for imp in resolved_imports:
        if imp in allowed:
            continue
        root = imp.split(".")[0]
        if ext == ".java" and any(imp.startswith(s) for s in std_java):
            continue
        if ext == ".py" and root in std_py:
            continue
        if project_deps and any(d.lower() in imp.lower() for d in project_deps):
            continue
        invented.append(imp)
    if invented:
        return False, f"Invented imports: {', '.join(invented[:3])}"

    # Syntaxe Python
    if ext == ".py":
        import ast
        try:
            ast.parse(resolved)
        except SyntaxError as e:
            return False, f"Python syntax: {e}"

    # Accolades Java/JS/TS
    if ext in (".java", ".js", ".ts", ".jsx", ".tsx"):
        if abs(resolved.count("{") - resolved.count("}")) > 1:
            return False, f"Unbalanced braces ({resolved.count('{')} open, {resolved.count('}')} close)"

    return True, "OK"


def resolve_single_file(
    file_path: str,
    conflicted_content: str,
    ours_content: str,
    theirs_content: str,
    project_context: str = "",
    project_deps: Set[str] = None,
) -> Optional[str]:
    """
    Résolution avec budget contrôlé et cascade de modèles.

    3 stratégies × cascade gemini-2.5-flash → 2.0-flash → 1.5-flash.
    max_output_tokens=1024 (réponse concise forcée).
    """
    from dotenv import load_dotenv
    load_dotenv(Path(_project_root) / ".env")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    strategies_name = ["intelligent", "conservative", "safe"]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt > 1:
            print(f"\n    {_YL}↻ Tentative {attempt}/{MAX_ATTEMPTS} ({strategies_name[attempt-1]})...{_R}",
                  end="", flush=True)

        prompt = _build_resolution_prompt(
            file_path, conflicted_content,
            ours_content, theirs_content,
            project_context, attempt,
        )

        # Cascade de modèles
        resolved = None
        for model in _LLM_MODELS:
            for retry in range(2):
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    llm = ChatGoogleGenerativeAI(
                        model=model,
                        google_api_key=api_key,
                        max_output_tokens=_MAX_OUTPUT,
                        temperature=max(0.0, 0.1 - (attempt * 0.03)),
                    )
                    response = llm.invoke(prompt)
                    resolved = response.content.strip()
                    break
                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        wait = [20, 45][min(retry, 1)]
                        print(f"\n    ⏳ [{model}] Quota 429 — {wait}s...", end="", flush=True)
                        time.sleep(wait)
                        if retry == 1:
                            resolved = None
                            break
                    else:
                        resolved = None
                        break
            if resolved is not None:
                break

        if resolved is None:
            print(f"\n    {_RD}Tous les modèles LLM indisponibles{_R}", end="")
            return None

        if resolved == "CANNOT_AUTO_RESOLVE":
            if attempt < MAX_ATTEMPTS:
                continue
            return None

        # Nettoyer markdown
        if resolved.startswith("```"):
            lines = resolved.splitlines()
            end = -1 if (lines and lines[-1].strip() == "```") else len(lines)
            resolved = "\n".join(lines[1:end])

        is_valid, reason = _validate_resolved(
            resolved, file_path, ours_content, theirs_content, project_deps
        )
        if is_valid:
            if attempt > 1:
                print(f"  {_GR}✓{_R}", end="")
            return resolved
        else:
            print(f"\n    {_YL}Validation échouée ({reason}){_R}", end="")
            if attempt >= MAX_ATTEMPTS:
                return None

    return None


def resolve_all_conflicts(project_path: Path) -> Dict[str, any]:
    """Pipeline complet de résolution locale."""
    print(f"\n  {_CY}{_B}Code Auditor — Résolution automatique de conflits (v5){_R}\n")

    conflict_files = detect_conflict_files(project_path)
    if not conflict_files:
        print(f"  {_GR}✓  Aucun conflit détecté.{_R}\n")
        return {"resolved": [], "failed": [], "total": 0}

    print(f"  Fichiers en conflit : {_B}{len(conflict_files)}{_R}\n")

    try:
        from smart_git.conflict_context_builder import ConflictContextBuilder
        ctx_builder = ConflictContextBuilder(project_path)
        project_deps = ctx_builder._get_project_dependencies()
    except Exception:
        project_deps = set()

    resolved_files = []
    failed_files   = []

    for filepath in conflict_files:
        abs_path = project_path / filepath
        print(f"\n  {_DM}→ {filepath}{_R}", end="", flush=True)

        try:
            conflicted = abs_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            failed_files.append(filepath)
            continue

        if not has_conflict_markers(conflicted):
            resolved_files.append(filepath)
            print(f"  {_GR}✓ Pas de marqueurs{_R}")
            continue

        ours, theirs = get_ours_theirs(filepath, project_path)

        try:
            from smart_git.conflict_context_builder import ConflictContextBuilder
            ctx = ConflictContextBuilder(project_path).build_context(filepath)
        except Exception:
            ctx = ""

        resolved = resolve_single_file(
            filepath, conflicted, ours, theirs, ctx, project_deps,
        )

        if resolved is None:
            print(f"  {_RD}✗ Échec{_R}")
            failed_files.append(filepath)
            continue

        try:
            abs_path.write_text(resolved, encoding="utf-8")
            subprocess.run(["git", "add", filepath], cwd=str(project_path), capture_output=True)
            print(f"  {_GR}✓ Résolu{_R}")
            resolved_files.append(filepath)
        except Exception as e:
            print(f"  {_RD}✗ Erreur écriture : {e}{_R}")
            failed_files.append(filepath)

    print(f"\n  {'─' * 50}")
    print(f"  {_B}Résumé : {len(resolved_files)}/{len(conflict_files)} résolu(s){_R}")
    if failed_files:
        print(f"  {_RD}{len(failed_files)} échec(s){_R}")
        for f in failed_files:
            print(f"    {_RD}✗{_R}  {f}")
    if resolved_files and not failed_files:
        print(f"\n  {_GR}✓ Tous les conflits résolus !{_R}")
        print(f"  {_DM}Finalisez : git merge --continue{_R}")
    print()

    return {"resolved": resolved_files, "failed": failed_files, "total": len(conflict_files)}