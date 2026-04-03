"""
git_hook.py — Pre-commit hook intelligent (Smart Git System).

Nouveauté v2 : analyse des dépendants
  Le hook ne bloque plus seulement sur les bugs des fichiers stagés.
  Il analyse aussi les fichiers qui DÉPENDENT des fichiers stagés pour
  détecter les incompatibilités introduites par le commit (ex : signature
  d'une méthode publique changée → tous ses appelants peuvent casser).

Deux sources d'information :
  Source A — Cache SQLite (Watch)
    Les fichiers déjà analysés en mode Watch ont leur résultat dans SQLite.
    Le hook les lit directement — 0 appel LLM, quasi instantané.

  Source B — Analyse LLM à la volée
    Si un dépendant n'est pas dans le cache, le hook lance une analyse
    LLM ciblée sur la compatibilité.
    Limité à MAX_LIVE_ANALYSES appels pour préserver le quota Gemini.

Logique de score :
  Score fichiers stagés      (100 % du poids)  — bugs dans le code commité
  Score fichiers dépendants  ( 60 % du poids)  — impact cascade du commit
  Score final = score_staged + score_deps * DEP_WEIGHT

Installation du hook :
  python git/git_hook.py --install --project C:\\monprojet

Désinstallation :
  python git/git_hook.py --uninstall --project C:\\monprojet

Contournement (urgence) :
  git commit --no-verify
"""

from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

# ── Seuils et poids ───────────────────────────────────────────────────────────

STRICT_MODE     = True
WARN_THRESHOLD  = 15
BLOCK_THRESHOLD = 35

# Poids appliqué au score des dépendants dans le score final
# 0.6 = les bugs dans les dépendants comptent à 60 % vs les bugs stagés
DEP_WEIGHT = 0.6

# Nombre max d'analyses LLM lancées en live pour les dépendants non-cachés
# Mettre à 0 pour désactiver l'analyse live et ne compter que le cache.
MAX_LIVE_ANALYSES = 2

# Nombre max de dépendants inspectés par fichier stagé
MAX_DEPS_PER_FILE = 3

SEVERITY_WEIGHTS   = {"CRITICAL": 10, "HIGH": 3, "MEDIUM": 1, "LOW": 0}
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
# Résolution des dépendants d'un fichier
# ─────────────────────────────────────────────────────────────────────────────

def _find_dependents(file_path: str, project_path: Path, cache_db: Path) -> List[str]:
    """
    Retourne les chemins absolus des fichiers qui IMPORTENT file_path.

    Stratégie en deux passes :
      Passe 1 — Index ProjectIndexer (cache SQLite .codeaudit/project_context.db)
        Si l'Orchestrator a déjà tourné, les imports résolus sont dans le cache.
        On interroge directement la colonne "files" du snapshot.

      Passe 2 — Recherche textuelle rapide (fallback)
        Si le cache n'est pas disponible, on cherche le nom du fichier dans
        les imports des autres fichiers du projet via regex.
    """
    abs_path  = str(Path(file_path).resolve())
    file_name = Path(abs_path).stem

    # ── Passe 1 : cache ProjectIndexer ───────────────────────────────────────
    try:
        indexer_db = project_path / ".codeaudit" / "project_context.db"
        if indexer_db.exists():
            conn = sqlite3.connect(f"file:{indexer_db}?mode=ro", uri=True)
            row  = conn.execute(
                "SELECT files FROM project_snapshot ORDER BY id DESC LIMIT 1"
            ).fetchone()
            conn.close()

            if row and row[0]:
                import json
                files_data = json.loads(row[0])
                dependents = []
                for fp, info in files_data.items():
                    if not isinstance(info, dict):
                        continue
                    imports = info.get("imports", [])
                    for imp in imports:
                        imp_base = imp.split(".")[-1] if "." in imp else imp
                        if (file_name.lower() in imp_base.lower()
                                or imp_base.lower() in file_name.lower()):
                            if (fp != abs_path
                                    and Path(fp).suffix.lower() in WATCHED_EXTENSIONS):
                                dependents.append(fp)
                                break
                if dependents:
                    return dependents[:MAX_DEPS_PER_FILE]
    except Exception:
        pass

    # ── Passe 2 : recherche textuelle ─────────────────────────────────────────
    dependents = []
    excluded   = {"__pycache__", ".git", "node_modules", ".venv", "venv",
                  "dist", "build", "target"}

    for candidate in project_path.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in WATCHED_EXTENSIONS:
            continue
        if any(ex in candidate.parts for ex in excluded):
            continue
        if str(candidate.resolve()) == abs_path:
            continue
        try:
            content = candidate.read_text(encoding="utf-8", errors="replace")
            patterns = [
                rf'\bimport\s+.*{re.escape(file_name)}\b',
                rf'\bfrom\s+.*{re.escape(file_name)}\s+import\b',
                rf'import\s+["\'].*{re.escape(file_name)}["\']',
                rf'require\s*\(["\'].*{re.escape(file_name)}["\']',
            ]
            for pat in patterns:
                if re.search(pat, content, re.IGNORECASE):
                    dependents.append(str(candidate.resolve()))
                    break
        except Exception:
            pass

        if len(dependents) >= MAX_DEPS_PER_FILE:
            break

    return dependents


# ─────────────────────────────────────────────────────────────────────────────
# Lecture du cache SQLite
# ─────────────────────────────────────────────────────────────────────────────

def _read_analysis_from_cache(abs_path: str, cache_db: Path) -> Optional[str]:
    """Lit une analyse depuis le cache Watch SQLite. Retourne None si absente."""
    if not cache_db.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{cache_db}?mode=ro", uri=True)
        row  = conn.execute(
            "SELECT analysis_text FROM file_cache WHERE file_path = ?",
            (abs_path,),
        ).fetchone()
        conn.close()
        return row[0] if row and row[0] else None
    except Exception:
        return None


def _read_analyses_from_cache(
    cache_db: Path,
    project_path: Path,
    files: list,
) -> dict:
    """Lit les analyses de plusieurs fichiers depuis le cache. Retourne {path: text}."""
    result = {}
    if not cache_db.exists():
        return result
    try:
        conn = sqlite3.connect(f"file:{cache_db}?mode=ro", uri=True)
        for file_info in files:
            abs_path = str((project_path / file_info["path"]).resolve())
            row = conn.execute(
                "SELECT analysis_text FROM file_cache WHERE file_path = ?",
                (abs_path,),
            ).fetchone()
            result[file_info["path"]] = row[0] if row and row[0] else None
        conn.close()
    except Exception:
        pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Analyse LLM à la volée (dépendant non-caché)
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_dependent_live(
    dep_path:          str,
    changed_file_name: str,
    cache_db:          Path,
) -> Optional[str]:
    """
    Lance une analyse LLM ciblée compatibilité sur un dépendant non-caché.

    La question posée au LLM est volontairement restreinte :
      "Ce fichier dépendant compile-t-il toujours après le changement de
       <changed_file_name> ? Y a-t-il des appels cassés, des imports manquants ?"
    Pas d'audit exhaustif — juste la vérification de compatibilité.
    """
    path = Path(dep_path)
    if not path.exists():
        return None
    try:
        code = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    try:
        from services.llm_service import assistant_agent

        context = {
            "file_path":          dep_path,
            "language":           path.suffix.lstrip("."),
            "post_solution_mode": True,
            "post_solution_hint": (
                f"{changed_file_name} was just modified and committed. "
                f"Check ONLY whether THIS file still compiles and calls the right "
                f"methods/signatures. Use block_fix for any incompatibility found. "
                f"If everything is compatible, say so with 0 fix blocks."
            ),
            "upstream_change": (
                f"IMPORTANT: {changed_file_name} was just committed. "
                f"Verify that THIS file still compiles and works correctly with it. "
                f"Check: broken imports, wrong method calls, signature mismatches."
            ),
        }
        result        = assistant_agent.analyze_code_with_rag(code=code, context=context)
        analysis_text = result.get("analysis", "")

        # Sauvegarder dans le cache pour les prochains commits
        if analysis_text and cache_db.exists():
            try:
                import hashlib
                from datetime import datetime
                conn         = sqlite3.connect(str(cache_db), check_same_thread=False)
                content_hash = hashlib.sha256(code.encode()).hexdigest()
                mtime        = path.stat().st_mtime
                conn.execute(
                    """INSERT OR REPLACE INTO file_cache
                       (file_path, content_hash, last_modified,
                        analysis_text, relevant_knowledge,
                        dependencies, dependents, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (dep_path, content_hash, mtime,
                     analysis_text, "[]", "[]", "[]",
                     datetime.now().isoformat()),
                )
                conn.commit()
                conn.close()
            except Exception:
                pass

        return analysis_text
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Score helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score_from_text(text: str) -> Tuple[int, int, int, float]:
    """Extrait (critical, high, medium, score) depuis un texte d'analyse."""
    c = len(re.findall(
        r"\[CRITICAL\]|severity.*?CRITICAL|\*\*SEVERITY\*\*.*?CRITICAL",
        text, re.I))
    h = len(re.findall(
        r"\[HIGH\]|severity.*?HIGH|\*\*SEVERITY\*\*.*?HIGH",
        text, re.I))
    m = len(re.findall(
        r"\[MEDIUM\]|severity.*?MEDIUM|\*\*SEVERITY\*\*.*?MEDIUM",
        text, re.I))
    score = (c * SEVERITY_WEIGHTS["CRITICAL"]
             + h * SEVERITY_WEIGHTS["HIGH"]
             + m * SEVERITY_WEIGHTS["MEDIUM"])
    return c, h, m, score


def _calculate_staged_score(files: list, analyses: dict) -> Tuple[float, list]:
    """Calcule le score et les rapports pour les fichiers stagés."""
    total_score  = 0.0
    file_reports = []

    for file_info in files:
        path          = file_info["path"]
        analysis_text = analyses.get(path)

        if not analysis_text:
            file_reports.append({
                "path": path, "bugs_critical": 0, "bugs_high": 0,
                "bugs_medium": 0, "score": 0.0, "analyzed": False,
                "kind": "staged",
            })
            continue

        c, h, m, file_score = _score_from_text(analysis_text)
        total_score += file_score
        file_reports.append({
            "path": path, "bugs_critical": c, "bugs_high": h,
            "bugs_medium": m, "score": file_score, "analyzed": True,
            "kind": "staged",
        })

    return total_score, file_reports


# ─────────────────────────────────────────────────────────────────────────────
# Hook principal
# ─────────────────────────────────────────────────────────────────────────────

def run_pre_commit_hook(project_path: Path) -> int:
    """
    Pipeline complet du pre-commit hook :

      Étape 1  : Lire les fichiers stagés
      Étape 2  : Localiser le cache SQLite Watch
      Étape 3  : Score des fichiers stagés (depuis cache)
      Étape 4  : Trouver les dépendants de chaque fichier stagé
      Étape 5  : Score des dépendants (cache → fallback LLM live)
      Étape 6  : Score final pondéré + affichage + décision
    """
    from smart_git.git_diff_parser import get_staged_files, is_git_repo

    print(f"\n  {_CY}{_B}Code Auditor — Vérification pre-commit{_R}\n")

    if not is_git_repo(project_path):
        print(f"  {_DM}Pas un dépôt Git — analyse ignorée.{_R}\n")
        return 0

    # ── Étape 1 : fichiers stagés ─────────────────────────────────────────────
    staged = get_staged_files(project_path)
    code_files = [
        f for f in staged
        if f["status"] != "D"
        and Path(f["path"]).suffix.lower() in WATCHED_EXTENSIONS
    ]

    if not code_files:
        print(f"  {_GR}✓  Aucun fichier de code dans ce commit.{_R}\n")
        return 0

    # ── Étape 2 : localiser le cache ──────────────────────────────────────────
    cache_db = project_path.parent / "code_auditor" / "data" / "cache" / "analysis_cache.db"
    if not cache_db.exists():
        cache_db = Path(_project_root) / "data" / "cache" / "analysis_cache.db"

    # ── Étape 3 : score des fichiers stagés ───────────────────────────────────
    staged_analyses              = _read_analyses_from_cache(cache_db, project_path, code_files)
    staged_score, staged_reports = _calculate_staged_score(code_files, staged_analyses)

    # ── Étape 4+5 : dépendants ────────────────────────────────────────────────
    dep_reports: List[dict] = []
    seen_deps:   set        = set()
    live_count              = 0

    staged_abs_paths = {
        str((project_path / f["path"]).resolve())
        for f in code_files
    }

    for file_info in code_files:
        staged_abs   = str((project_path / file_info["path"]).resolve())
        changed_name = Path(file_info["path"]).name

        print(f"  {_DM}Recherche dépendants de {changed_name}...{_R}")
        dependents = _find_dependents(staged_abs, project_path, cache_db)

        if not dependents:
            print(f"  {_DM}  → aucun dépendant trouvé{_R}")
            continue

        dep_names = ", ".join(Path(d).name for d in dependents)
        print(f"  {_DM}  → {len(dependents)} dépendant(s) : {dep_names}{_R}")

        for dep_abs in dependents:
            # Ne pas analyser un fichier déjà stagé
            if dep_abs in staged_abs_paths or dep_abs in seen_deps:
                continue
            seen_deps.add(dep_abs)

            dep_name = Path(dep_abs).name

            # Source A : cache SQLite
            analysis_text = _read_analysis_from_cache(dep_abs, cache_db)
            source        = "cache"

            # Source B : analyse LLM live si cache vide et quota disponible
            if not analysis_text and live_count < MAX_LIVE_ANALYSES:
                print(f"  {_DM}  → Analyse live de {dep_name} (non en cache)...{_R}")
                analysis_text = _analyze_dependent_live(dep_abs, changed_name, cache_db)
                source        = "live"
                live_count   += 1

            if not analysis_text:
                dep_reports.append({
                    "name": dep_name, "bugs_critical": 0, "bugs_high": 0,
                    "bugs_medium": 0, "score": 0.0, "analyzed": False,
                    "source": "none", "caused_by": changed_name,
                })
                continue

            c, h, m, dep_score = _score_from_text(analysis_text)
            dep_reports.append({
                "name": dep_name, "bugs_critical": c, "bugs_high": h,
                "bugs_medium": m, "score": dep_score, "analyzed": True,
                "source": source, "caused_by": changed_name,
            })

    # ── Étape 6 : score final pondéré ─────────────────────────────────────────
    dep_score_raw = sum(r["score"] for r in dep_reports if r["analyzed"])
    final_score   = staged_score + dep_score_raw * DEP_WEIGHT

    return _render_and_decide(
        staged_score   = staged_score,
        dep_score_raw  = dep_score_raw,
        final_score    = final_score,
        staged_reports = staged_reports,
        dep_reports    = dep_reports,
        code_files     = code_files,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Affichage + décision
# ─────────────────────────────────────────────────────────────────────────────

def _render_and_decide(
    staged_score:   float,
    dep_score_raw:  float,
    final_score:    float,
    staged_reports: list,
    dep_reports:    list,
    code_files:     list,
) -> int:

    nb_staged     = len(staged_reports)
    nb_unanalyzed = sum(1 for r in staged_reports if not r["analyzed"])
    W = 60

    print()
    print(f"  {_B}Fichiers stagés : {nb_staged}{_R}  ·  "
          f"Dépendants inspectés : {len(dep_reports)}")
    print(f"  Score stagés : {_B}{staged_score:.0f}{_R}  ·  "
          f"Score dépendants (×{DEP_WEIGHT}) : {_B}{dep_score_raw * DEP_WEIGHT:.0f}{_R}  ·  "
          f"Score final : {_B}{final_score:.0f}{_R}")
    print()

    # ── Tableau fichiers stagés ───────────────────────────────────────────────
    print(f"  {_B}Fichiers stagés :{_R}")
    print(f"  {'─' * W}")
    print(f"  {'Fichier':<30}  {'C':>3}  {'H':>3}  {'M':>3}  {'Score':>6}")
    print(f"  {'─' * W}")

    for r in sorted(staged_reports, key=lambda x: x["score"], reverse=True):
        name = r["path"].split("/")[-1][:30]
        if not r["analyzed"]:
            print(f"  {name:<30}  {_DM}{'—':>3}  {'—':>3}  {'—':>3}  {'?':>6}  "
                  f"[pas d'analyse]{_R}")
        else:
            c_col = f"{_RD}{r['bugs_critical']}{_R}" if r["bugs_critical"] else str(r["bugs_critical"])
            h_col = f"{_YL}{r['bugs_high']}{_R}"     if r["bugs_high"]     else str(r["bugs_high"])
            print(f"  {name:<30}  {c_col:>3}  {h_col:>3}  {r['bugs_medium']:>3}  "
                  f"{r['score']:>6.0f}")

    print(f"  {'─' * W}")
    print()

    # ── Tableau dépendants ────────────────────────────────────────────────────
    if dep_reports:
        print(f"  {_B}Dépendants impactés :{_R}")
        print(f"  {'─' * W}")
        print(f"  {'Fichier':<22}  {'Causé par':<16}  {'C':>3}  {'H':>3}  "
              f"{'Score×{:.1f}'.format(DEP_WEIGHT):>8}  Src")
        print(f"  {'─' * W}")

        for r in sorted(dep_reports, key=lambda x: x["score"], reverse=True):
            name      = r["name"][:22]
            caused_by = r.get("caused_by", "")[:16]
            src_tag   = (f"{_DM}[{r['source']}]{_R}" if r["analyzed"]
                         else f"{_DM}[—]{_R}")

            if not r["analyzed"]:
                print(f"  {name:<22}  {caused_by:<16}  {_DM}{'—':>3}  {'—':>3}  "
                      f"{'?':>8}{_R}  {src_tag}")
            else:
                c_col    = (f"{_RD}{r['bugs_critical']}{_R}"
                            if r["bugs_critical"] else str(r["bugs_critical"]))
                h_col    = (f"{_YL}{r['bugs_high']}{_R}"
                            if r["bugs_high"] else str(r["bugs_high"]))
                weighted = r["score"] * DEP_WEIGHT
                print(f"  {name:<22}  {caused_by:<16}  {c_col:>3}  {h_col:>3}  "
                      f"{weighted:>8.1f}  {src_tag}")

        print(f"  {'─' * W}")
        print(f"  {_DM}Src : cache = analyse Watch existante  ·  "
              f"live = analyse lancée maintenant{_R}")
        print()

    # ── Avertissements ────────────────────────────────────────────────────────
    if nb_unanalyzed > 0:
        print(f"  {_YL}⚠  {nb_unanalyzed} fichier(s) stagé(s) sans analyse Watch récente.{_R}")
        print(f"  {_DM}  Conseil : lancer 'python main.py watch <projet>' "
              f"et sauvegarder ces fichiers.{_R}")
        print()

    nb_deps_unanalyzed = sum(1 for r in dep_reports if not r["analyzed"])
    if nb_deps_unanalyzed > 0:
        print(f"  {_YL}⚠  {nb_deps_unanalyzed} dépendant(s) non-analysé(s) "
              f"(quota LLM live atteint).{_R}")
        print(f"  {_DM}  Augmentez MAX_LIVE_ANALYSES ou lancez le mode Watch "
              f"pour les pré-analyser.{_R}")
        print()

    # ── Décision ─────────────────────────────────────────────────────────────
    if final_score == 0:
        print(f"  {_GR}{_B}✓  COMMIT AUTORISÉ — aucun problème détecté.{_R}\n")
        return 0

    if final_score < WARN_THRESHOLD:
        print(f"  {_GR}✓  COMMIT AUTORISÉ{_R} — "
              f"{_DM}problèmes mineurs (score {final_score:.0f} < {WARN_THRESHOLD}).{_R}\n")
        return 0

    if final_score < BLOCK_THRESHOLD:
        print(f"  {_YL}{_B}⚠  COMMIT AVEC AVERTISSEMENT{_R} — score {final_score:.0f}")
        print(f"  {_YL}Des problèmes HIGH ont été détectés dans les fichiers "
              f"stagés ou leurs dépendants.{_R}")
        if dep_score_raw * DEP_WEIGHT > 0:
            print(f"  {_DM}  dont {dep_score_raw * DEP_WEIGHT:.0f} pts provenant "
                  f"des dépendants.{_R}")
        print(f"  {_DM}  Pour annuler : git reset HEAD (unstage){_R}\n")
        return 0

    # Score ≥ BLOCK_THRESHOLD
    staged_crits = sum(r["bugs_critical"] for r in staged_reports if r["analyzed"])
    dep_crits    = sum(r["bugs_critical"] for r in dep_reports    if r["analyzed"])

    if STRICT_MODE:
        print(f"  {_RD}{_B}✗  COMMIT BLOQUÉ — score {final_score:.0f} ≥ {BLOCK_THRESHOLD}{_R}")

        if staged_crits > 0:
            print(f"  {_RD}  • {staged_crits} bug(s) CRITICAL dans les fichiers stagés.{_R}")

        if dep_crits > 0:
            affected = [r["name"] for r in dep_reports if r["bugs_critical"] > 0]
            print(f"  {_RD}  • {dep_crits} bug(s) CRITICAL dans les dépendants : "
                  f"{', '.join(affected[:4])}{_R}")
            print(f"  {_YL}  Ces fichiers cassent à cause de votre commit — "
                  f"corrigez-les aussi avant de commiter.{_R}")

        print(f"  {_RD}Appliquez les corrections du mode Watch avant de commiter.{_R}")
        print(f"  {_DM}  Pour forcer quand même : git commit --no-verify{_R}\n")
        return 1

    # Mode non-strict
    print(f"  {_YL}{_B}⚠  COMMIT AVEC ALERTE{_R} — score {final_score:.0f} (mode non-strict)")
    print(f"  {_YL}Des bugs CRITICAL sont présents. Le commit est autorisé mais risqué.{_R}\n")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Installation / désinstallation
# ─────────────────────────────────────────────────────────────────────────────

def install_hook(project_path: Path, strict: bool = True) -> None:
    """Installe le script pre-commit dans .git/hooks/pre-commit."""
    hook_dir  = project_path / ".git" / "hooks"
    hook_file = hook_dir / "pre-commit"

    if not hook_dir.exists():
        print(f"  {_RD}✗  Dossier .git/hooks introuvable : {hook_dir}{_R}")
        return

    strict_flag = "--strict" if strict else ""
    hook_content = (
        "#!/bin/sh\n"
        "# Code Auditor Smart Git Hook — généré automatiquement\n"
        f"python \"{_project_root / 'smart_git' / 'git_hook.py'}\" "
        f"--project \"$(pwd)\" {strict_flag}\n"
    )
    hook_file.write_text(hook_content, encoding="utf-8")
    try:
        hook_file.chmod(0o755)
    except Exception:
        pass
    print(f"  {_GR}✓  Hook installé : {hook_file}{_R}")
    print(f"  {_DM}  Mode strict       : {'activé' if strict else 'désactivé'}{_R}")
    print(f"  {_DM}  Analyse dépendants: activée "
          f"(MAX_LIVE_ANALYSES={MAX_LIVE_ANALYSES}, "
          f"MAX_DEPS_PER_FILE={MAX_DEPS_PER_FILE}){_R}")


def uninstall_hook(project_path: Path) -> None:
    """Supprime le pre-commit hook installé par Code Auditor."""
    hook_file = project_path / ".git" / "hooks" / "pre-commit"
    if (hook_file.exists()
            and "Code Auditor" in hook_file.read_text(encoding="utf-8", errors="replace")):
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
    parser.add_argument("--project",   type=str, default=".", help="Chemin du projet git")
    parser.add_argument("--install",   action="store_true",   help="Installer le hook")
    parser.add_argument("--uninstall", action="store_true",   help="Désinstaller le hook")
    parser.add_argument("--strict",    action="store_true",   help="Mode strict (bloque si CRITICAL)")
    parser.add_argument("--no-strict", action="store_true",   help="Mode non-strict (avertit seulement)")
    args = parser.parse_args()

    project_path = Path(args.project).resolve()

    if args.install:
        install_hook(project_path, strict=not args.no_strict)
    elif args.uninstall:
        uninstall_hook(project_path)
    else:
        sys.exit(run_pre_commit_hook(project_path))