"""
git_hook.py — Pre-commit hook (v3 — fixes score counting + cache freshness).

Fixes vs v2 :
  FIX 1 — Structured block counting (Limitation 2)
    Replaces noisy regex score with _count_severity_from_blocks()
    → reads SEVERITY field inside ---FIX START--- blocks only
    → "// NOT CRITICAL" and KB rule text no longer inflate the score

  FIX 2 — Cache freshness check (Limitation 1)
    Hook now compares content_hash from SQLite against SHA256 of the
    current staged file before trusting the cached analysis.
    Stale = file changed since last Watch analysis → marked as unanalyzed.

  FIX 3 — episode_memory in hook (Limitation 4)
    Reads recurring patterns from episode_memory table and displays
    them with occurrence count + first-seen date.

  FIX 4 — Commit-to-commit score delta (Limitation 5)
    Computes score_before from the last commit's cached analyses,
    subtracts score_staged, shows ↓ 68 → 12.

Everything else (dependency analysis, live LLM fallback, strict mode)
is preserved from v2.
"""

from __future__ import annotations

import hashlib
import re
from services.mcp_redis_service import get_mcp_redis, key_hash, KEY_PREFIX
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

# ── Seuils ───────────────────────────────────────────────────────────────────
STRICT_MODE     = True
WARN_THRESHOLD  = 15
BLOCK_THRESHOLD = 35
DEP_WEIGHT      = 0.6
MAX_LIVE_ANALYSES  = 2
MAX_DEPS_PER_FILE  = 3

SEVERITY_WEIGHTS   = {"CRITICAL": 10, "HIGH": 3, "MEDIUM": 1, "LOW": 0}
WATCHED_EXTENSIONS = {".java", ".py", ".ts", ".js", ".tsx", ".jsx"}


_R  = "\033[0m"
_B  = "\033[1m"
_GR = "\033[92m"
_YL = "\033[93m"
_RD = "\033[91m"
_CY = "\033[96m"
_DM = "\033[2m"


def _count_severity_from_blocks(text: str) -> Tuple[int, int, int, float]:
    """
    Compte les sévérités depuis l'analyse LLM.

    Stratégie à 2 niveaux :
      1. Blocs structurés ---FIX START--- avec **SEVERITY**: X (précis, pas de faux positifs)
      2. Fallback texte libre : compte les mentions CRITICAL/HIGH/MEDIUM dans le texte
         quand aucun bloc structuré n'est trouvé (pour les modèles qui ne suivent pas le format)

    Le fallback utilise des patterns contextuels pour éviter de compter les mentions
    dans les sections KB rules ou knowledge context.
    """
    counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    parts = re.split(r'-{3,}\s*FIX START\s*-{3,}', text, flags=re.IGNORECASE)

    for part in parts[1:]:
        end = re.search(r'-{3,}\s*FIX END\s*-{3,}', part, re.IGNORECASE)
        if end:
            part = part[:end.start()]
        sev_match = re.search(
            r'\*\*SEVERITY\*\*\s*:?\s*(\w+)', part, re.IGNORECASE)
        if sev_match:
            sev = sev_match.group(1).upper()
            if sev in counts:
                counts[sev] += 1

    # ── Fallback : comptage texte libre si aucun bloc structuré ────────────
    # Certains modèles (MiniMax, Qwen) ne génèrent pas de blocs ---FIX START---
    # mais écrivent "CRITICAL - SQL injection" ou "CRITICAL: dataSource undeclared"
    if sum(counts.values()) == 0 and text:
        # Patterns qui indiquent une vraie issue (pas juste une mention dans les rules)
        # Ex: "CRITICAL - SQL injection", "CRITICAL: undeclared", "[CRITICAL]"
        #     "Issues Found:\nCRITICAL - ..."
        critical_patterns = [
            r'(?:^|\n)\s*(?:\*\*)?CRITICAL(?:\*\*)?\s*[-:–—]',        # CRITICAL - xxx ou CRITICAL: xxx
            r'\[CRITICAL\]',                                           # [CRITICAL]
            r'Severity\s*:\s*CRITICAL',                                # Severity: CRITICAL
            r'CRITICAL\s+(?:error|bug|issue|vulnerability|security)',   # CRITICAL error/bug/...
        ]
        high_patterns = [
            r'(?:^|\n)\s*(?:\*\*)?HIGH(?:\*\*)?\s*[-:–—]',
            r'\[HIGH\]',
            r'Severity\s*:\s*HIGH',
            r'HIGH\s+(?:error|bug|issue|vulnerability|security|performance)',
        ]
        medium_patterns = [
            r'(?:^|\n)\s*(?:\*\*)?MEDIUM(?:\*\*)?\s*[-:–—]',
            r'\[MEDIUM\]',
            r'Severity\s*:\s*MEDIUM',
            r'MEDIUM\s+(?:error|bug|issue|vulnerability|warning)',
        ]

        for pattern in critical_patterns:
            counts["CRITICAL"] += len(re.findall(pattern, text, re.IGNORECASE))
        for pattern in high_patterns:
            counts["HIGH"] += len(re.findall(pattern, text, re.IGNORECASE))
        for pattern in medium_patterns:
            counts["MEDIUM"] += len(re.findall(pattern, text, re.IGNORECASE))

    score = (
        counts["CRITICAL"] * SEVERITY_WEIGHTS["CRITICAL"] +
        counts["HIGH"]     * SEVERITY_WEIGHTS["HIGH"]     +
        counts["MEDIUM"]   * SEVERITY_WEIGHTS["MEDIUM"]
    )
    return counts["CRITICAL"], counts["HIGH"], counts["MEDIUM"], score


def _compute_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""


def _read_analysis_fresh(abs_path: str, cache_db: Path = None) -> Optional[str]:
    """Lit l'analyse depuis Redis MCP avec vérification de fraîcheur."""
    try:
        redis = get_mcp_redis()
        redis_key = f"{KEY_PREFIX}fc:{key_hash(abs_path)}"
        data = redis.hgetall(redis_key)
        if not data or not data.get("analysis_text"):
            return None

        cached_hash = data.get("content_hash", "")
        current_hash = _compute_file_hash(abs_path)
        if not current_hash or current_hash != cached_hash:
            return None

        return data["analysis_text"]
    except Exception:
        return None

def _get_recurring_patterns(abs_path: str, cache_db: Path = None, min_count: int = 2) -> List[dict]:
    """Lit les patterns récurrents depuis Redis MCP."""
    try:
        redis = get_mcp_redis()
        scan_pattern = f"{KEY_PREFIX}em:{key_hash(abs_path)}:*"
        keys = redis.scan_keys(scan_pattern)
        results = []
        for k in keys:
            data = redis.hgetall(k)
            count = int(data.get("occurrence_count", "0"))
            if count >= min_count:
                results.append({
                    "pattern":    data.get("pattern_type", ""),
                    "severity":   data.get("severity", "MEDIUM"),
                    "count":      count,
                    "first_seen": data.get("first_seen", ""),
                    "in_kb":      data.get("promoted_to_kb", "0") == "1",
                })
        results.sort(key=lambda x: x["count"], reverse=True)
        return results[:5]
    except Exception:
        return []


def _compute_score_before(
    staged_files: list,
    project_path: Path,
    cache_db: Path = None,
) -> float:
    """Calcule le score avant commit en lisant depuis Redis MCP."""
    import subprocess

    score_before = 0.0
    try:
        redis = get_mcp_redis()
        for file_info in staged_files:
            abs_path = str((project_path / file_info["path"]).resolve())

            prev_result = subprocess.run(
                ["git", "show", f"HEAD:{file_info['path']}"],
                capture_output=True, cwd=str(project_path)
            )
            if prev_result.returncode != 0:
                continue

            prev_hash = hashlib.sha256(prev_result.stdout).hexdigest()

            # Vérifier si le hash correspond au cache
            redis_key = f"{KEY_PREFIX}fc:{key_hash(abs_path)}"
            data = redis.hgetall(redis_key)
            if data and data.get("content_hash") == prev_hash and data.get("analysis_text"):
                _, _, _, file_score = _count_severity_from_blocks(data["analysis_text"])
                score_before += file_score
    except Exception:
        pass

    return score_before

def _find_dependents(file_path: str, project_path: Path, cache_db: Path = None) -> List[str]:

    abs_path  = str(Path(file_path).resolve())
    file_name = Path(abs_path).stem

    try:
        import json
        redis = get_mcp_redis()
        ps_key = f"{KEY_PREFIX}ps:{key_hash(str(project_path))}"
        raw = redis.get(ps_key)
        if raw:
            snapshot = json.loads(raw)
            files_data = snapshot.get("files", {})
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


def _read_analyses_from_cache(
    cache_db: Path, project_path: Path, files: list
) -> dict:
    """Reads fresh analyses for multiple files (with freshness check)."""
    result = {}
    for file_info in files:
        abs_path = str((project_path / file_info["path"]).resolve())
        result[file_info["path"]] = _read_analysis_fresh(abs_path, cache_db)
    return result

def _analyze_dependent_live(
    dep_path:          str,
    changed_file_name: str,
    cache_db:          Path,
) -> Optional[str]:
    """
    Lance une analyse LLM ciblée compatibilité sur un dépendant non-caché.
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
                f"{changed_file_name} was just committed. "
                "Check ONLY whether THIS file still compiles and calls the right "
                "methods/signatures. Use block_fix for any incompatibility found."
            ),
            "upstream_change": (
                f"IMPORTANT: {changed_file_name} was just committed. "
                "Verify that THIS file still compiles and works correctly with it."
            ),
        }
        result        = assistant_agent.analyze_code_with_rag(code=code, context=context)
        analysis_text = result.get("analysis", "")

        # Save to cache for next commits via Redis MCP
        if analysis_text:
            try:
                from datetime import datetime
                redis = get_mcp_redis()
                content_hash = _compute_file_hash(dep_path)
                mtime = path.stat().st_mtime
                redis_key = f"{KEY_PREFIX}fc:{key_hash(dep_path)}"
                redis.hset_dict(redis_key, {
                    "file_path":          dep_path,
                    "content_hash":       content_hash,
                    "last_modified":      str(mtime),
                    "analysis_text":      analysis_text,
                    "relevant_knowledge": "[]",
                    "dependencies":       "[]",
                    "dependents":         "[]",
                    "updated_at":         datetime.now().isoformat(),
                })
            except Exception:
                pass

        return analysis_text
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Hook principal
# ─────────────────────────────────────────────────────────────────────────────

def run_pre_commit_hook(project_path: Path) -> int:
    from smart_git.git_diff_parser import get_staged_files, is_git_repo

    print(f"\n  {_CY}{_B}Code Auditor — Pre-commit check (v3){_R}\n")

    if not is_git_repo(project_path):
        print(f"  {_DM}Pas un dépôt Git — analyse ignorée.{_R}\n")
        return 0

    staged = get_staged_files(project_path)
    code_files = [
        f for f in staged
        if f["status"] != "D"
        and Path(f["path"]).suffix.lower() in WATCHED_EXTENSIONS
    ]

    if not code_files:
        print(f"  {_GR}✓  Aucun fichier de code dans ce commit.{_R}\n")
        return 0

    # Locate cache DB
    cache_db = project_path.parent / "code_auditor" / "data" / "cache" / "analysis_cache.db"
    if not cache_db.exists():
        cache_db = Path(_project_root) / "data" / "cache" / "analysis_cache.db"

    # ── FIX 4 : score_before ─────────────────────────────────────────────────
    score_before = _compute_score_before(code_files, project_path, cache_db)

    # ── Staged files score (FIX 1 + FIX 2) ────────────────────────────────────
    staged_analyses = _read_analyses_from_cache(cache_db, project_path, code_files)
    staged_score    = 0.0
    staged_reports  = []

    for file_info in code_files:
        path     = file_info["path"]
        text     = staged_analyses.get(path)
        abs_path = str((project_path / path).resolve())

        # FIX 3: recurring patterns from episode_memory
        recurring = _get_recurring_patterns(abs_path, cache_db, min_count=2)

        if not text:
            staged_reports.append({
                "path": path, "bugs_critical": 0, "bugs_high": 0,
                "bugs_medium": 0, "score": 0.0, "analyzed": False,
                "stale": True, "recurring": recurring,
            })
            continue

        c, h, m, file_score = _count_severity_from_blocks(text)  # FIX 1
        staged_score += file_score
        staged_reports.append({
            "path": path, "bugs_critical": c, "bugs_high": h,
            "bugs_medium": m, "score": file_score, "analyzed": True,
            "stale": False, "recurring": recurring,
        })

    # ── Dépendants ────────────────────────────────────────────────────────────
    dep_reports:  List[dict] = []
    seen_deps:    set        = set()
    live_count              = 0
    staged_abs_paths = {
        str((project_path / f["path"]).resolve())
        for f in code_files
    }

    for file_info in code_files:
        staged_abs   = str((project_path / file_info["path"]).resolve())
        changed_name = Path(file_info["path"]).name
        dependents   = _find_dependents(staged_abs, project_path, cache_db)

        for dep_abs in dependents:
            if dep_abs in staged_abs_paths or dep_abs in seen_deps:
                continue
            seen_deps.add(dep_abs)
            dep_name = Path(dep_abs).name

            text   = _read_analysis_fresh(dep_abs, cache_db)  # FIX 2
            source = "cache"

            if not text and live_count < MAX_LIVE_ANALYSES:
                text   = _analyze_dependent_live(dep_abs, changed_name, cache_db)
                source = "live"
                live_count += 1

            if not text:
                dep_reports.append({
                    "name": dep_name, "bugs_critical": 0, "bugs_high": 0,
                    "bugs_medium": 0, "score": 0.0, "analyzed": False,
                    "source": "none", "caused_by": changed_name,
                })
                continue

            c, h, m, dep_score = _count_severity_from_blocks(text)  # FIX 1
            dep_reports.append({
                "name": dep_name, "bugs_critical": c, "bugs_high": h,
                "bugs_medium": m, "score": dep_score, "analyzed": True,
                "source": source, "caused_by": changed_name,
            })

    dep_score_raw = sum(r["score"] for r in dep_reports if r["analyzed"])
    final_score   = staged_score + dep_score_raw * DEP_WEIGHT

    return _render_and_decide(
        staged_score   = staged_score,
        dep_score_raw  = dep_score_raw,
        final_score    = final_score,
        score_before   = score_before,        # FIX 4
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
    score_before:   float,       # FIX 4
    staged_reports: list,
    dep_reports:    list,
    code_files:     list,
) -> int:

    nb_staged     = len(staged_reports)
    nb_unanalyzed = sum(1 for r in staged_reports if not r["analyzed"])
    nb_stale      = sum(1 for r in staged_reports if r.get("stale"))
    W = 64

    print()
    print(f"  {_B}Fichiers stagés : {nb_staged}{_R}  ·  "
          f"Dépendants inspectés : {len(dep_reports)}")

    # FIX 4: score delta
    delta = final_score - score_before
    delta_str = (f"{_GR}↓ {abs(delta):.0f}{_R}" if delta < 0
                 else f"{_RD}↑ {abs(delta):.0f}{_R}" if delta > 0
                 else f"{_DM}→ inchangé{_R}")
    print(f"  Score : {_B}{score_before:.0f}{_R} → {_B}{final_score:.0f}{_R}  {delta_str}")
    print(f"  (stagés {staged_score:.0f} + deps×{DEP_WEIGHT} {dep_score_raw * DEP_WEIGHT:.0f})")
    print()

    # ── Tableau fichiers stagés ───────────────────────────────────────────────
    print(f"  {_B}Fichiers stagés :{_R}")
    print(f"  {'─' * W}")
    print(f"  {'Fichier':<30}  {'C':>3}  {'H':>3}  {'M':>3}  {'Score':>6}  Status")
    print(f"  {'─' * W}")

    for r in sorted(staged_reports, key=lambda x: x["score"], reverse=True):
        name = r["path"].split("/")[-1][:30]
        if not r["analyzed"]:
            tag = f"{_YL}[stale — relancer Watch]{_R}" if r.get("stale") else f"{_DM}[pas d'analyse]{_R}"
            print(f"  {name:<30}  {'—':>3}  {'—':>3}  {'—':>3}  {'?':>6}  {tag}")
        else:
            c_col = f"{_RD}{r['bugs_critical']}{_R}" if r["bugs_critical"] else "0"
            h_col = f"{_YL}{r['bugs_high']}{_R}"     if r["bugs_high"]     else "0"
            print(f"  {name:<30}  {c_col:>3}  {h_col:>3}  {r['bugs_medium']:>3}  "
                  f"{r['score']:>6.0f}")

        # FIX 3: show recurring patterns from episode_memory
        for p in r.get("recurring", [])[:2]:
            col = _RD if p["severity"] == "CRITICAL" else _YL
            kb  = f" {_DM}[KB]{_R}" if p["in_kb"] else ""
            days_ago = ""
            try:
                from datetime import datetime
                first = datetime.fromisoformat(p["first_seen"])
                delta_days = (datetime.now() - first).days
                if delta_days > 0:
                    days_ago = f" depuis {delta_days}j"
            except Exception:
                pass
            print(f"    {col}⟳ {p['pattern']} ×{p['count']}{kb}{days_ago}{_R}"
                  f"  {_DM}(cross-session){_R}")

    print(f"  {'─' * W}")
    print()

    # ── Tableau dépendants ────────────────────────────────────────────────────
    if dep_reports:
        print(f"  {_B}Dépendants impactés :{_R}")
        print(f"  {'─' * W}")
        for r in sorted(dep_reports, key=lambda x: x["score"], reverse=True):
            name      = r["name"][:22]
            caused_by = r.get("caused_by", "")[:16]
            src_tag   = (f"{_DM}[{r['source']}]{_R}" if r["analyzed"]
                         else f"{_DM}[—]{_R}")
            if not r["analyzed"]:
                print(f"  {name:<22}  {caused_by:<16}  {_DM}—{_R}  {src_tag}")
            else:
                c_col = f"{_RD}{r['bugs_critical']}{_R}" if r["bugs_critical"] else "0"
                h_col = f"{_YL}{r['bugs_high']}{_R}" if r["bugs_high"] else "0"
                print(f"  {name:<22}  {caused_by:<16}  {c_col}C  {h_col}H  "
                      f"×{DEP_WEIGHT}={r['score']*DEP_WEIGHT:.0f}  {src_tag}")
        print(f"  {'─' * W}")
        print()

    # Stale warning (FIX 2)
    if nb_stale > 0:
        print(f"  {_YL}⚠  {nb_stale} fichier(s) avec cache obsolète "
              f"(modifié depuis la dernière analyse Watch).{_R}")
        print(f"  {_DM}  Lancez 'python main.py watch <projet>' "
              f"et sauvegardez ces fichiers.{_R}")
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
              f"stagés ou leurs dépendants.{_R}\n")
        return 0

    staged_crits = sum(r["bugs_critical"] for r in staged_reports if r["analyzed"])
    dep_crits    = sum(r["bugs_critical"] for r in dep_reports    if r["analyzed"])

    if STRICT_MODE:
        print(f"  {_RD}{_B}✗  COMMIT BLOQUÉ — score {final_score:.0f} ≥ {BLOCK_THRESHOLD}{_R}")
        if staged_crits:
            print(f"  {_RD}  • {staged_crits} CRITICAL dans les fichiers stagés.{_R}")
        if dep_crits:
            affected = [r["name"] for r in dep_reports if r["bugs_critical"] > 0]
            print(f"  {_RD}  • {dep_crits} CRITICAL dans les dépendants : "
                  f"{', '.join(affected[:4])}{_R}")
        print(f"  {_DM}  Pour forcer : git commit --no-verify{_R}\n")
        return 1

    print(f"  {_YL}{_B}⚠  COMMIT AVEC ALERTE{_R} — score {final_score:.0f} (mode non-strict)\n")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Installation / désinstallation
# ─────────────────────────────────────────────────────────────────────────────

def install_hook(project_path: Path, strict: bool = True) -> None:
    hook_dir  = project_path / ".git" / "hooks"

    if not hook_dir.exists():
        print(f"  {_RD}✗  .git/hooks introuvable : {hook_dir}{_R}")
        return

    # Utiliser le Python du venv (pas le système)
    venv_python = _project_root / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        venv_python = _project_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = Path(sys.executable)  # fallback
    python_cmd = str(venv_python)

    # ── Pre-commit hook (quality gate) ────────────────────────────────────
    hook_file = hook_dir / "pre-commit"
    strict_flag = "--strict" if strict else ""
    hook_content = (
        "#!/bin/sh\n"
        "# Code Auditor Smart Git Hook v3\n"
        f"\"{python_cmd}\" \"{_project_root / 'smart_git' / 'git_hook.py'}\" "
        f"--project \"$(pwd)\" {strict_flag}\n"
    )
    hook_file.write_text(hook_content, encoding="utf-8")
    try:
        hook_file.chmod(0o755)
    except Exception:
        pass
    print(f"  {_GR}✓  Pre-commit hook installé{_R}")

    # ── Prepare-commit-msg hook (auto commit message) ─────────────────────
    msg_hook_file = hook_dir / "prepare-commit-msg"
    msg_hook_content = (
        "#!/bin/sh\n"
        "# Code Auditor — Auto commit message generator\n"
        f"\"{python_cmd}\" \"{_project_root / 'smart_git' / 'git_commit_msg.py'}\" "
        f"--project \"$(pwd)\" --msg-file \"$1\"\n"
    )
    msg_hook_file.write_text(msg_hook_content, encoding="utf-8")
    try:
        msg_hook_file.chmod(0o755)
    except Exception:
        pass
    print(f"  {_GR}✓  Commit message hook installé{_R}")

    print(f"  {_DM}  Python : {python_cmd}{_R}")
    print(f"  {_DM}  Mode strict : {'activé' if strict else 'désactivé'}{_R}")
    print(f"  {_DM}  Auto-commit message : activé (Conventional Commits){_R}")



def uninstall_hook(project_path: Path) -> None:
    hook_dir = project_path / ".git" / "hooks"

    # Pre-commit
    hook_file = hook_dir / "pre-commit"
    if (hook_file.exists() and
            "Code Auditor" in hook_file.read_text(encoding="utf-8", errors="replace")):
        hook_file.unlink()
        print(f"  {_GR}✓  Pre-commit hook désinstallé{_R}")
    else:
        print(f"  {_DM}Aucun pre-commit hook Code Auditor trouvé.{_R}")

    # Prepare-commit-msg
    msg_hook = hook_dir / "prepare-commit-msg"
    if (msg_hook.exists() and
            "Code Auditor" in msg_hook.read_text(encoding="utf-8", errors="replace")):
        msg_hook.unlink()
        print(f"  {_GR}✓  Commit message hook désinstallé{_R}")



# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Code Auditor — Smart Pre-commit Hook v3")
    parser.add_argument("--project",   type=str, default=".",   help="Chemin du projet git")
    parser.add_argument("--install",   action="store_true",     help="Installer le hook")
    parser.add_argument("--uninstall", action="store_true",     help="Désinstaller le hook")
    parser.add_argument("--strict",    action="store_true",     help="Mode strict")
    parser.add_argument("--no-strict", action="store_true",     help="Mode non-strict")
    args = parser.parse_args()

    project_path = Path(args.project).resolve()

    if args.install:
        install_hook(project_path, strict=not args.no_strict)
    elif args.uninstall:
        uninstall_hook(project_path)
    else:
        sys.exit(run_pre_commit_hook(project_path))