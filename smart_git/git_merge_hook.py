"""
git_merge_hook.py — Pre-merge hook intelligent.

╔══════════════════════════════════════════════════════════════════════╗
║  ÉTAPE 4 — Pre-merge Hook                                           ║
║                                                                      ║
║  Ce hook se déclenche AUTOMATIQUEMENT lors d'un `git merge`.         ║
║  Il bloque le merge si :                                             ║
║    - Le score de qualité ≥ 35 (bugs CRITICAL dans la branche)        ║
║    - Des conflits sont détectés (suggère resolve-conflicts)          ║
║                                                                      ║
║  Installation :                                                      ║
║    python main.py hook <projet> --merge                              ║
║    → installe .git/hooks/pre-merge-commit                            ║
║                                                                      ║
║  Contournement :                                                     ║
║    git merge --no-verify                                             ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from smart_git.git_hook import (
    _count_severity_from_blocks,
    _read_analysis_fresh,
    _compute_file_hash,
    SEVERITY_WEIGHTS,
    WATCHED_EXTENSIONS,
    BLOCK_THRESHOLD,
    WARN_THRESHOLD,
)

# ── ANSI ─────────────────────────────────────────────────────────────────────
_R  = "\033[0m"
_B  = "\033[1m"
_GR = "\033[92m"
_YL = "\033[93m"
_RD = "\033[91m"
_CY = "\033[96m"
_DM = "\033[2m"


# ─────────────────────────────────────────────────────────────────────────────
# Détection des infos de merge
# ─────────────────────────────────────────────────────────────────────────────

def _get_merge_info(project_path: Path) -> Dict[str, str]:
    """
    Récupère les informations sur le merge en cours.

    MERGE_HEAD contient le SHA du commit que l'on tente de merger.
    On en déduit la branche source.
    """
    merge_head_file = project_path / ".git" / "MERGE_HEAD"
    if not merge_head_file.exists():
        return {"merging": False}

    merge_sha = merge_head_file.read_text().strip()

    # Trouver le nom de la branche depuis le SHA
    result = subprocess.run(
        ["git", "name-rev", "--name-only", merge_sha],
        capture_output=True, text=True, cwd=str(project_path)
    )
    source_branch = result.stdout.strip() if result.returncode == 0 else merge_sha[:8]

    # Branche courante
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        capture_output=True, text=True, cwd=str(project_path)
    )
    target_branch = result.stdout.strip() if result.returncode == 0 else "HEAD"

    return {
        "merging": True,
        "source": source_branch,
        "target": target_branch,
        "merge_sha": merge_sha,
    }


def _get_merge_changed_files(project_path: Path) -> List[str]:
    """
    Liste les fichiers modifiés entre HEAD et MERGE_HEAD.

    Ces fichiers sont ceux que le merge va introduire.
    """
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD...MERGE_HEAD"],
        capture_output=True, text=True, cwd=str(project_path)
    )
    if result.returncode != 0:
        return []
    return [
        f.strip() for f in result.stdout.strip().split("\n")
        if f.strip() and Path(f.strip()).suffix.lower() in WATCHED_EXTENSIONS
    ]


def _detect_conflicts(project_path: Path) -> List[str]:
    """Détecte les fichiers en conflit (si le merge a déjà produit des conflits)."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=U"],
        capture_output=True, text=True, cwd=str(project_path)
    )
    if result.returncode != 0:
        return []
    return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Hook principal
# ─────────────────────────────────────────────────────────────────────────────

def run_pre_merge_hook(project_path: Path) -> int:
    """
    Pipeline du pre-merge hook :

      1. Récupérer les infos du merge (source → target)
      2. Lister les fichiers modifiés dans la branche source
      3. Pour chaque fichier, lire le cache SQLite
      4. Calculer le score de qualité
      5. Détecter les conflits
      6. Décision : autoriser ou bloquer le merge
    """
    print(f"\n  {_CY}{_B}Code Auditor — Pre-merge check{_R}\n")

    merge_info = _get_merge_info(project_path)
    if not merge_info.get("merging"):
        print(f"  {_DM}Pas de merge en cours.{_R}\n")
        return 0

    source = merge_info["source"]
    target = merge_info["target"]
    print(f"  Merge : {_B}{source}{_R} → {_B}{target}{_R}\n")

    # Localiser le cache
    cache_db = project_path.parent / "code_auditor" / "data" / "cache" / "analysis_cache.db"
    if not cache_db.exists():
        cache_db = Path(_project_root) / "data" / "cache" / "analysis_cache.db"

    # Fichiers modifiés par le merge
    changed_files = _get_merge_changed_files(project_path)
    if not changed_files:
        print(f"  {_GR}✓  Aucun fichier de code dans ce merge.{_R}\n")
        return 0

    print(f"  Fichiers modifiés : {_B}{len(changed_files)}{_R}")

    # Conflits
    conflicts = _detect_conflicts(project_path)
    if conflicts:
        print(f"  Conflits détectés : {_RD}{_B}{len(conflicts)} fichier(s){_R}")

    print()

    # Score de chaque fichier
    total_score = 0.0
    total_critical = 0
    total_high = 0
    file_reports = []
    W = 50

    print(f"  {_B}Fichiers de la branche source :{_R}")
    print(f"  {'─' * W}")
    print(f"  {'Fichier':<30}  {'C':>3}  {'H':>3}  {'Score':>6}")
    print(f"  {'─' * W}")

    for filepath in changed_files:
        abs_path = str((project_path / filepath).resolve())
        analysis = _read_analysis_fresh(abs_path, cache_db)

        if not analysis:
            print(f"  {filepath.split('/')[-1]:<30}  {'—':>3}  {'—':>3}  {'?':>6}  "
                  f"{_DM}[pas d'analyse]{_R}")
            file_reports.append({"path": filepath, "score": 0, "analyzed": False})
            continue

        c, h, m, score = _count_severity_from_blocks(analysis)
        total_score += score
        total_critical += c
        total_high += h

        c_col = f"{_RD}{c}{_R}" if c else "0"
        h_col = f"{_YL}{h}{_R}" if h else "0"
        print(f"  {filepath.split('/')[-1]:<30}  {c_col:>3}  {h_col:>3}  {score:>6.0f}")
        file_reports.append({"path": filepath, "score": score, "analyzed": True,
                             "critical": c, "high": h})

    print(f"  {'─' * W}")
    print(f"\n  Score total : {_B}{total_score:.0f}{_R}")
    print()

    # ── Décision ──────────────────────────────────────────────────────────────
    block_reasons = []

    if conflicts:
        block_reasons.append(
            f"{len(conflicts)} fichier(s) en conflit (résolution requise)"
        )

    if total_score >= BLOCK_THRESHOLD or total_critical > 0:
        block_reasons.append(
            f"Score {total_score:.0f} ≥ {BLOCK_THRESHOLD} "
            f"ou {total_critical} bug(s) CRITICAL"
        )

    if block_reasons:
        print(f"  {_RD}{_B}✗  MERGE BLOQUÉ{_R}")
        for reason in block_reasons:
            print(f"  {_RD}  • {reason}{_R}")
        if conflicts:
            print(f"\n  {_CY}💡 Résoudre les conflits :{_R}")
            print(f"  {_DM}   python main.py resolve-conflicts{_R}")
        print(f"\n  {_DM}  Pour forcer : git merge --no-verify{_R}\n")
        return 1

    if total_score >= WARN_THRESHOLD:
        print(f"  {_YL}{_B}⚠  MERGE AVEC AVERTISSEMENT{_R} — "
              f"score {total_score:.0f}")
        print(f"  {_YL}Des problèmes HIGH ont été détectés.{_R}\n")
        return 0

    print(f"  {_GR}{_B}✓  MERGE AUTORISÉ{_R} — "
          f"score {total_score:.0f}\n")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Installation du merge hook
# ─────────────────────────────────────────────────────────────────────────────

def install_merge_hook(project_path: Path) -> None:
    """Installe le hook pre-merge-commit dans .git/hooks/."""
    hook_dir  = project_path / ".git" / "hooks"
    hook_file = hook_dir / "pre-merge-commit"

    if not hook_dir.exists():
        print(f"  {_RD}✗  .git/hooks introuvable : {hook_dir}{_R}")
        return

    hook_content = (
        "#!/bin/sh\n"
        "# Code Auditor Smart Git Merge Hook\n"
        f"python \"{_project_root / 'smart_git' / 'git_merge_hook.py'}\" "
        f"--project \"$(pwd)\"\n"
    )
    hook_file.write_text(hook_content, encoding="utf-8")
    try:
        hook_file.chmod(0o755)
    except Exception:
        pass
    print(f"  {_GR}✓  Merge hook installé : {hook_file}{_R}")


def uninstall_merge_hook(project_path: Path) -> None:
    """Supprime le pre-merge-commit hook."""
    hook_file = project_path / ".git" / "hooks" / "pre-merge-commit"
    if (hook_file.exists() and
            "Code Auditor" in hook_file.read_text(encoding="utf-8", errors="replace")):
        hook_file.unlink()
        print(f"  {_GR}✓  Merge hook désinstallé{_R}")
    else:
        print(f"  {_DM}Aucun merge hook Code Auditor trouvé.{_R}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Code Auditor — Pre-merge Hook")
    parser.add_argument("--project", type=str, default=".", help="Chemin du projet git")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--uninstall", action="store_true")
    args = parser.parse_args()

    project_path = Path(args.project).resolve()

    if args.install:
        install_merge_hook(project_path)
    elif args.uninstall:
        uninstall_merge_hook(project_path)
    else:
        sys.exit(run_pre_merge_hook(project_path))
