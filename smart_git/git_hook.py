"""
git_hook.py — Pre-commit hook intelligent (Smart Git System).

Différence fondamentale avec l'ancien hook :
  AVANT : analysait les fichiers stagés avec le LLM à chaque commit
          → 30-45 secondes de délai, quota Gemini consommé au mauvais moment
  
  MAINTENANT : lit le score déjà calculé par GitSessionTracker depuis SQLite
               → < 1 seconde, 0 appel LLM, résultat toujours disponible

Logique de décision :
  1. Les fichiers stagés sont-ils dans le cache SQLite ? (analyse Watch disponible)
  2. Quel est le score de risque pour ces fichiers spécifiquement ?
  3. Appliquer les règles de commit selon le score.

Règles configurables dans config.py :
  GIT_HOOK_STRICT_MODE = True   → bloque si score ≥ 35 (CRITICAL)
  GIT_HOOK_WARN_SCORE  = 15     → avertissement si score ≥ 15 (WARN)

Installation du hook :
  python git/git_hook.py --install --project C:\\monprojet

Désinstallation :
  python git/git_hook.py --uninstall --project C:\\monprojet

Contournement (si urgence) :
  git commit --no-verify    → court-circuite le hook (toujours possible)
"""
from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

# Ajouter la racine du projet au sys.path pour les imports
_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

# ── Constantes ────────────────────────────────────────────────────────────────
STRICT_MODE     = True    # False → jamais bloquant, warnings seulement
WARN_THRESHOLD  = 15      # score → avertissement (niveau WARN)
BLOCK_THRESHOLD = 35      # score → blocage (niveau CRITICAL, strict mode seulement)

SEVERITY_WEIGHTS = {"CRITICAL": 10, "HIGH": 3, "MEDIUM": 1, "LOW": 0}
WATCHED_EXTENSIONS = {".java", ".py", ".ts", ".js", ".tsx", ".jsx"}

# ── Couleurs ANSI ─────────────────────────────────────────────────────────────
_R  = "\033[0m"
_B  = "\033[1m"
_GR = "\033[92m"
_YL = "\033[93m"
_RD = "\033[91m"
_CY = "\033[96m"
_DM = "\033[2m"


# ─────────────────────────────────────────────────────────────────────────────
# Hook principal
# ─────────────────────────────────────────────────────────────────────────────

def run_pre_commit_hook(project_path: Path) -> int:
    """
    Point d'entrée du pre-commit hook.

    Retourne :
      0 → commit autorisé (propre ou score acceptable)
      1 → commit BLOQUÉ (mode strict + score CRITICAL)

    La logique est :
      1. Récupérer les fichiers stagés (git diff --cached)
      2. Pour chaque fichier, lire l'analyse depuis SQLite
      3. Calculer le score de risque global
      4. Afficher le rapport et décider
    """
    from smart_git.git_diff_parser import get_staged_files, is_git_repo

    print(f"\n  {_CY}{_B}Code Auditor — Vérification pre-commit{_R}\n")

    # Vérification : repo git valide ?
    if not is_git_repo(project_path):
        print(f"  {_DM}Pas un dépôt Git — analyse ignorée.{_R}\n")
        return 0

    # Étape 1 : fichiers stagés
    staged = get_staged_files(project_path)
    code_files = [
        f for f in staged
        if f["status"] != "D"
        and Path(f["path"]).suffix.lower() in WATCHED_EXTENSIONS
    ]

    if not code_files:
        print(f"  {_GR}✓  Aucun fichier de code dans ce commit.{_R}\n")
        return 0

    # Étape 2 : lire le cache SQLite
    cache_db = project_path.parent / "code_auditor" / "data" / "cache" / "analysis_cache.db"
    if not cache_db.exists():
        # Essai à l'emplacement standard (si lancé depuis la racine du projet)
        cache_db = Path(_project_root) / "data" / "cache" / "analysis_cache.db"

    analyses = _read_analyses_from_cache(cache_db, project_path, code_files)

    # Étape 3 : calculer le score et afficher le rapport
    score, file_reports = _calculate_commit_score(code_files, analyses)

    # Étape 4 : afficher et décider
    return _render_and_decide(score, file_reports, code_files, analyses)


def _read_analyses_from_cache(
    cache_db: Path,
    project_path: Path,
    files: list,
) -> dict:
    """
    Lit les analyses depuis SQLite pour les fichiers stagés.
    Retourne {file_path_str: analysis_text} — None si absent du cache.
    """
    result = {}
    if not cache_db.exists():
        return result

    try:
        conn = sqlite3.connect(f"file:{cache_db}?mode=ro", uri=True)
        for file_info in files:
            abs_path = str(project_path / file_info["path"])
            row = conn.execute(
                "SELECT analysis_text FROM file_cache WHERE file_path = ?",
                (abs_path,),
            ).fetchone()
            result[file_info["path"]] = row[0] if row and row[0] else None
        conn.close()
    except Exception:
        pass
    return result


def _calculate_commit_score(files: list, analyses: dict) -> tuple:
    """
    Calcule le score de risque du commit et prépare les rapports par fichier.

    Retourne (score_total, file_reports)
    file_reports = [{path, bugs_critical, bugs_high, bugs_medium, score, analyzed}, ...]
    """
    total_score  = 0.0
    file_reports = []

    for file_info in files:
        path          = file_info["path"]
        analysis_text = analyses.get(path)
        analyzed      = analysis_text is not None

        if not analyzed:
            file_reports.append({
                "path": path, "bugs_critical": 0, "bugs_high": 0,
                "bugs_medium": 0, "score": 0.0, "analyzed": False,
            })
            continue

        c = len(re.findall(r"\[CRITICAL\]|severity.*?CRITICAL", analysis_text, re.I))
        h = len(re.findall(r"\[HIGH\]|severity.*?HIGH",     analysis_text, re.I))
        m = len(re.findall(r"\[MEDIUM\]|severity.*?MEDIUM", analysis_text, re.I))

        file_score = c * SEVERITY_WEIGHTS["CRITICAL"] + h * SEVERITY_WEIGHTS["HIGH"] + m
        total_score += file_score
        file_reports.append({
            "path": path, "bugs_critical": c, "bugs_high": h,
            "bugs_medium": m, "score": file_score, "analyzed": True,
        })

    return total_score, file_reports


def _render_and_decide(
    score: float,
    file_reports: list,
    code_files: list,
    analyses: dict,
) -> int:
    """
    Affiche le rapport de commit et retourne le code de sortie (0 ou 1).
    """
    nb_analyzed    = sum(1 for r in file_reports if r["analyzed"])
    nb_unanalyzed  = len(file_reports) - nb_analyzed

    print(f"  {_B}{len(code_files)} fichier(s) analysé(s){_R}  ·  Score de risque : {_B}{score:.0f}{_R}")

    # ── Tableau par fichier ───────────────────────────────────────────────────
    print(f"  {'─' * 58}")
    print(f"  {'Fichier':<30}  {'C':>3}  {'H':>3}  {'M':>3}  {'Score':>6}")
    print(f"  {'─' * 58}")

    for r in sorted(file_reports, key=lambda x: x["score"], reverse=True):
        name = r["path"].split("/")[-1][:30]
        if not r["analyzed"]:
            print(f"  {name:<30}  {_DM}{'—':>3}  {'—':>3}  {'—':>3}  {'?':>6}  [pas d'analyse]{_R}")
        else:
            c_col = f"{_RD}{r['bugs_critical']}{_R}" if r['bugs_critical'] else str(r['bugs_critical'])
            h_col = f"{_YL}{r['bugs_high']}{_R}"     if r['bugs_high']     else str(r['bugs_high'])
            print(f"  {name:<30}  {c_col:>3}  {h_col:>3}  {r['bugs_medium']:>3}  {r['score']:>6.0f}")

    print(f"  {'─' * 58}")
    print(f"  {'TOTAL':<30}  {_B}{'':>3}  {'':>3}  {'':>3}  {score:>6.0f}{_R}")
    print()

    # ── Fichiers sans analyse ─────────────────────────────────────────────────
    if nb_unanalyzed > 0:
        print(f"  {_YL}⚠  {nb_unanalyzed} fichier(s) sans analyse Watch récente.{_R}")
        print(f"  {_DM}  Conseil : lancer 'python main.py watch <projet>' et sauvegarder ces fichiers.{_R}")
        print()

    # ── Décision ─────────────────────────────────────────────────────────────
    if score == 0:
        print(f"  {_GR}{_B}✓  COMMIT AUTORISÉ — aucun problème détecté.{_R}\n")
        return 0

    if score < WARN_THRESHOLD:
        print(f"  {_GR}✓  COMMIT AUTORISÉ{_R} — {_DM}problèmes mineurs (score {score:.0f} < {WARN_THRESHOLD}).{_R}\n")
        return 0

    if score < BLOCK_THRESHOLD:
        print(f"  {_YL}{_B}⚠  COMMIT AVEC AVERTISSEMENT{_R} — score {score:.0f}")
        print(f"  {_YL}Des problèmes HIGH ont été détectés. Vérifiez les corrections proposées par Watch.{_R}")
        print(f"  {_DM}  Pour annuler : git reset HEAD (unstage){_R}\n")
        return 0   # avertissement mais pas bloquant

    # Score ≥ BLOCK_THRESHOLD
    if STRICT_MODE:
        print(f"  {_RD}{_B}✗  COMMIT BLOQUÉ — score {score:.0f} ≥ {BLOCK_THRESHOLD} (mode strict){_R}")
        print(f"  {_RD}Des bugs CRITICAL sont présents dans les fichiers stagés.{_R}")
        print(f"  {_RD}Appliquez les corrections proposées par le mode Watch avant de commiter.{_R}")
        print(f"  {_DM}  Pour forcer quand même : git commit --no-verify{_R}\n")
        return 1   # exit 1 → git annule le commit

    # Mode non-strict : toujours 0, mais avertissement fort
    print(f"  {_YL}{_B}⚠  COMMIT AVEC ALERTE{_R} — score {score:.0f} (mode non-strict)")
    print(f"  {_YL}Des bugs CRITICAL sont présents. Le commit est autorisé mais risqué.{_R}\n")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Installation / désinstallation du hook
# ─────────────────────────────────────────────────────────────────────────────

def install_hook(project_path: Path, strict: bool = True) -> None:
    """
    Installe le script pre-commit dans .git/hooks/pre-commit.
    Écrase l'ancien hook s'il existe.
    """
    hook_dir  = project_path / ".git" / "hooks"
    hook_file = hook_dir / "pre-commit"

    if not hook_dir.exists():
        print(f"  {_RD}✗  Dossier .git/hooks introuvable : {hook_dir}{_R}")
        return

    strict_flag = "--strict" if strict else ""
    hook_content = (
        "#!/bin/sh\n"
        f"# Code Auditor Smart Git Hook — généré automatiquement\n"
        f"python \"{_project_root / 'smart_git' / 'git_hook.py'}\" "
        f"--project \"$(pwd)\" {strict_flag}\n"
    )
    hook_file.write_text(hook_content, encoding="utf-8")
    try:
        hook_file.chmod(0o755)
    except Exception:
        pass   # Windows : chmod non nécessaire
    print(f"  {_GR}✓  Hook installé : {hook_file}{_R}")
    print(f"  {_DM}  Mode strict : {'activé' if strict else 'désactivé'}{_R}")


def uninstall_hook(project_path: Path) -> None:
    """Supprime le pre-commit hook installé par Code Auditor."""
    hook_file = project_path / ".git" / "hooks" / "pre-commit"
    if hook_file.exists() and "Code Auditor" in hook_file.read_text(encoding="utf-8", errors="replace"):
        hook_file.unlink()
        print(f"  {_GR}✓  Hook désinstallé : {hook_file}{_R}")
    else:
        print(f"  {_DM}Aucun hook Code Auditor trouvé à désinstaller.{_R}")


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Code Auditor — Smart Pre-commit Hook")
    parser.add_argument("--project",    type=str, default=".", help="Chemin du projet git")
    parser.add_argument("--install",    action="store_true",   help="Installer le hook")
    parser.add_argument("--uninstall",  action="store_true",   help="Désinstaller le hook")
    parser.add_argument("--strict",     action="store_true",   help="Mode strict (bloque si CRITICAL)")
    parser.add_argument("--no-strict",  action="store_true",   help="Mode non-strict (avertit seulement)")
    args = parser.parse_args()

    project_path = Path(args.project).resolve()

    if args.install:
        install_hook(project_path, strict=not args.no_strict)
    elif args.uninstall:
        uninstall_hook(project_path)
    else:
        sys.exit(run_pre_commit_hook(project_path))