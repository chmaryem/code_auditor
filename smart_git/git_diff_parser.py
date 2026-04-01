"""
git_diff_parser.py — Interface complète avec Git via subprocess.

Fonctions existantes (inchangées) :
  get_changed_files()       → fichiers d'un commit donné
  get_staged_files()        → fichiers en staging (pre-commit)
  get_diff_content()        → diff textuel d'un fichier dans un commit
  get_current_commit_hash() → hash court du HEAD
  get_commit_message()      → message du dernier commit
  is_git_repo()             → vérifie si le dossier est un repo git

Nouvelles fonctions (Smart Git System) :
  get_uncommitted_files()   → fichiers modifiés depuis le dernier commit
  get_session_stats()       → stats globales de la session (lignes, temps)
  get_last_commit_time()    → timestamp unix du dernier commit
  get_file_at_commit()      → contenu d'un fichier à un commit précis
  get_merge_base()          → ancêtre commun entre deux branches
  get_branch_commits()      → commits exclusifs à une branche
  get_branch_diff_files()   → fichiers modifiés dans une branche vs base
"""
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional


# Helper interne


def _run_git(args: list, cwd: Path = None) -> Optional[str]:
    """
    Exécute une commande git et retourne stdout.
    Retourne None si git est absent ou si la commande échoue.
    Ne lève jamais d'exception — toutes les fonctions sont défensives.
    """
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd else None,
        )
        return result.stdout if result.returncode == 0 else None
    except FileNotFoundError:
        print("  Git n'est pas installé ou introuvable dans le PATH.")
        return None
    except Exception:
        return None


def _extract_int(text: str, pattern: str) -> int:
    """Extrait un entier depuis une chaîne via regex. Retourne 0 si absent."""
    m = re.search(pattern, text)
    return int(m.group(1)) if m else 0



# Fonctions existantes — inchangées


def get_changed_files(commit: str = "HEAD", project_path: Path = None) -> List[Dict[str, str]]:
    """
    Retourne les fichiers modifiés dans un commit donné.
    Statuts possibles : M (modified), A (added), D (deleted).

    Essaie d'abord 'git diff' (commit classique) puis 'git show'
    (fonctionne aussi pour le premier commit du repo sans parent).
    """
    output = _run_git(["diff", "--name-status", f"{commit}~1", commit], cwd=project_path)
    if output is None:
        output = _run_git(["show", "--name-status", "--format=", commit], cwd=project_path)
    if not output:
        return []
    files = []
    for line in output.strip().splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            files.append({"path": parts[-1], "status": parts[0][0]})
    return files


def get_staged_files(project_path: Path = None) -> List[Dict[str, str]]:
    """
    Retourne les fichiers en staging zone (après git add).
    Utilisé par le pre-commit hook.
    """
    output = _run_git(["diff", "--cached", "--name-status"], cwd=project_path)
    if not output:
        return []
    files = []
    for line in output.strip().splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            files.append({"path": parts[-1], "status": parts[0][0]})
    return files


def get_diff_content(file_path: str, commit: str = "HEAD", project_path: Path = None) -> str:
    """
    Retourne le diff textuel (unified diff) d'un fichier dans un commit.
    Injecter uniquement les lignes changées dans le prompt LLM
    économise ~80% des tokens vs envoyer le fichier entier.
    """
    output = _run_git(["diff", f"{commit}~1", commit, "--", file_path], cwd=project_path)
    return output or ""


def get_current_commit_hash(project_path: Path = None) -> str:
    """Hash court (7 chars) du commit HEAD."""
    output = _run_git(["rev-parse", "--short", "HEAD"], cwd=project_path)
    return output.strip() if output else "unknown"


def get_commit_message(commit: str = "HEAD", project_path: Path = None) -> str:
    """Première ligne du message du commit (subject line)."""
    output = _run_git(["log", "-1", "--pretty=%s", commit], cwd=project_path)
    return output.strip() if output else ""


def is_git_repo(project_path: Path) -> bool:
    """Vérifie si project_path est un dépôt git valide."""
    return _run_git(["rev-parse", "--git-dir"], cwd=project_path) is not None



# Nouvelles fonctions — Smart Git System


def get_uncommitted_files(project_path: Path = None) -> List[Dict[str, str]]:
    """
    Retourne TOUS les fichiers modifiés depuis le dernier commit,
    qu'ils soient stagés (git add) ou non.

    C'est la fonction centrale du GitSessionTracker :
    elle donne exactement l'ensemble des fichiers "à risque"
    dans la session de développement courante.

    

    Pourquoi deux git diff ?
      'git diff HEAD'          → fichiers modifiés mais PAS encore stagés
      'git diff --cached HEAD' → fichiers stagés (en attente de commit)
    On fusionne les deux pour avoir la vue complète de la session.
    """
    unstaged = _run_git(["diff", "HEAD", "--name-status"], cwd=project_path) or ""
    staged   = _run_git(["diff", "--cached", "--name-status"], cwd=project_path) or ""

    seen: Dict[str, Dict] = {}

    def _parse(raw: str, is_staged: bool):
        for line in raw.strip().splitlines():
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                path   = parts[-1]
                status = parts[0][0]
                if path not in seen:
                    seen[path] = {"path": path, "status": status, "staged": is_staged}
                else:
                    seen[path]["staged"] = seen[path]["staged"] or is_staged

    _parse(unstaged, False)
    _parse(staged, True)
    return list(seen.values())


def get_session_stats(project_path: Path = None) -> Dict:
    """
    Retourne les statistiques globales de la session de développement courante.

    Données renvoyées :
        files_changed        : nombre de fichiers modifiés depuis le dernier commit
        lines_added          : lignes ajoutées dans la session
        lines_removed        : lignes supprimées dans la session
        minutes_since_commit : temps écoulé depuis le dernier commit (en minutes)
        last_commit_msg      : message du dernier commit (contexte pour le rapport)

    Pourquoi ces métriques ?
    Un développeur avec 15 fichiers modifiés et 800 lignes changées depuis 4h
    est dans une situation très différente de quelqu'un avec 2 fichiers et 30
    lignes depuis 20 minutes. Le score de risque tient compte des deux dimensions.
    """
    stat_out = _run_git(["diff", "HEAD", "--shortstat"], cwd=project_path) or ""
    # Format typique : " 3 files changed, 45 insertions(+), 12 deletions(-)"
    files_changed = _extract_int(stat_out, r"(\d+) file")
    lines_added   = _extract_int(stat_out, r"(\d+) insertion")
    lines_removed = _extract_int(stat_out, r"(\d+) deletion")

    last_commit_ts  = get_last_commit_time(project_path)
    minutes_elapsed = int((time.time() - last_commit_ts) / 60) if last_commit_ts else 0
    last_msg        = get_commit_message("HEAD", project_path)

    return {
        "files_changed":        files_changed,
        "lines_added":          lines_added,
        "lines_removed":        lines_removed,
        "minutes_since_commit": minutes_elapsed,
        "last_commit_msg":      last_msg,
    }


def get_last_commit_time(project_path: Path = None) -> Optional[float]:
    """
    Retourne le timestamp unix (float) du dernier commit sur la branche courante.
    Retourne None si pas de commit (nouveau repo vide).

    Utilisé pour calculer 'minutes_since_commit' et pour pondérer le score
    de risque selon le temps écoulé : un bug non corrigé depuis 4h pèse
    plus qu'un bug apparu il y a 10 minutes.
    """
    output = _run_git(["log", "-1", "--format=%ct"], cwd=project_path)
    if output and output.strip().isdigit():
        return float(output.strip())
    return None


def get_file_at_commit(file_path: str, commit: str = "HEAD",
                       project_path: Path = None) -> Optional[str]:
    """
    Retourne le contenu complet d'un fichier tel qu'il existait à un commit précis.

    Utilisé par GitBranchAnalyzer pour analyser le code source d'un fichier
    dans le contexte d'une branche, sans dépendre de l'état actuel du
    working directory.

    Retourne None si le fichier n'existait pas à ce commit (ex: fichier ajouté
    après ce commit, ou fichier supprimé avant).
    """
    return _run_git(["show", f"{commit}:{file_path}"], cwd=project_path)


def get_merge_base(branch: str, base: str = "main",
                   project_path: Path = None) -> Optional[str]:
    """
    Retourne le hash du commit ancêtre commun entre 'branch' et 'base'.
    C'est le point de divergence — l'instant où la branche a été créée.

    Pourquoi c'est important :
      git diff main..feature  → inclut aussi les commits de main mergés dans feature
      git diff <merge-base>..feature → UNIQUEMENT les changements propres à feature

    Essaie 'main' puis 'master' en fallback (conventions différentes selon les repos).
    """
    output = _run_git(["merge-base", branch, base], cwd=project_path)
    if output is None and base == "main":
        output = _run_git(["merge-base", branch, "master"], cwd=project_path)
    return output.strip() if output else None


def get_branch_commits(branch: str = "HEAD", base: str = "main",
                       project_path: Path = None) -> List[Dict]:
    """
    Retourne la liste des commits présents dans 'branch' mais absents de 'base'.
    Ce sont les commits exclusifs à la branche feature.

    Format :
        [{"hash": "abc1234", "message": "...", "author": "...", "date": "..."}, ...]

    Utilisé par GitBranchAnalyzer pour construire la timeline de la branche.
    """
    merge_base = get_merge_base(branch, base, project_path)
    if not merge_base:
        return []

    fmt    = "%h\t%s\t%an\t%ci"
    output = _run_git(
        ["log", f"{merge_base}..{branch}", f"--format={fmt}"],
        cwd=project_path,
    )
    if not output:
        return []

    commits = []
    for line in output.strip().splitlines():
        parts = line.split("\t", 3)
        if len(parts) >= 1 and parts[0]:
            commits.append({
                "hash":    parts[0],
                "message": parts[1] if len(parts) > 1 else "",
                "author":  parts[2] if len(parts) > 2 else "",
                "date":    parts[3] if len(parts) > 3 else "",
            })
    return commits


def get_branch_diff_files(branch: str = "HEAD", base: str = "main",
                          project_path: Path = None) -> List[Dict[str, str]]:
    """
    Retourne les fichiers modifiés dans 'branch' par rapport à 'base',
    en utilisant le merge-base comme point de référence.

    C'est la liste exacte des fichiers à analyser pour un rapport de branche.
    Format : [{"path": "...", "status": "M/A/D"}, ...]
    """
    merge_base = get_merge_base(branch, base, project_path)
    if merge_base:
        output = _run_git(["diff", f"{merge_base}..{branch}", "--name-status"],
                          cwd=project_path)
    else:
        # Fallback si merge-base indisponible
        output = _run_git(["diff", f"{base}...{branch}", "--name-status"],
                          cwd=project_path)

    if not output:
        return []

    files = []
    for line in output.strip().splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            files.append({"path": parts[-1], "status": parts[0][0]})
    return files