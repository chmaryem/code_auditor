"""
git_report.py — Formatage et affichage des rapports Smart Git.

Trois types de rapports :
  1. session_report()  → état de la session courante (appelé par GitNotifier)
  2. commit_report()   → résumé d'un commit analysé
  3. branch_report()   → rapport complet de branche avec verdict de merge

Principes de formatage :
  - Largeur fixe 64 caractères (compatible 80-col terminal)
  - Même palette ANSI que console_renderer.py
  - Verdict de merge affiché en grand, facile à lire d'un coup d'oeil
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from smart_git.git_session_tracker import SessionSnapshot
    from smart_git.git_branch_analyzer  import BranchReport

# ── Couleurs ANSI ─────────────────────────────────────────────────────────────
_R  = "\033[0m"
_B  = "\033[1m"
_DM = "\033[2m"
_GR = "\033[92m"
_YL = "\033[93m"
_RD = "\033[91m"
_CY = "\033[96m"

_W  = 64   # largeur fixe des rapports


# ─────────────────────────────────────────────────────────────────────────────
# Rapport de session
# ─────────────────────────────────────────────────────────────────────────────

def session_report(snapshot: "SessionSnapshot") -> str:
    """
    Génère un rapport texte lisible de la session courante.
    Utilisé par 'python main.py git status'.
    Retourne le texte formaté (peut aussi être print()é directement).
    """
    lines = []
    lines.append(f"\n{'─' * _W}")
    lines.append(f"  Session Watch — {datetime.now().strftime('%H:%M:%S')}")
    lines.append(f"  Score de risque : {_B}{snapshot.score}{_R}  ·  "
                 f"Niveau : {_level_badge(snapshot.level)}")
    lines.append(f"  Depuis le dernier commit : {snapshot.minutes_since_commit} min  "
                 f"·  Multiplicateur temps : ×{snapshot.time_multiplier}")
    lines.append(f"{'─' * _W}")

    if not snapshot.files_at_risk and not snapshot.files_unanalyzed:
        lines.append(f"  {_GR}✓  Aucun problème détecté dans les fichiers modifiés.{_R}")
    else:
        if snapshot.files_at_risk:
            lines.append(f"\n  {'Fichier':<30}  {'C':>3} {'H':>3} {'M':>3}  Score")
            lines.append(f"  {'─' * 50}")
            for fr in snapshot.files_at_risk:
                name = fr.path.split("/")[-1][:30]
                c = f"{_RD}{fr.bugs_critical}{_R}" if fr.bugs_critical else "0"
                h = f"{_YL}{fr.bugs_high}{_R}"     if fr.bugs_high     else "0"
                lines.append(f"  {name:<30}  {c:>3} {h:>3} {fr.bugs_medium:>3}  {fr.score:.0f}")

        if snapshot.files_unanalyzed:
            lines.append(f"\n  {_DM}{len(snapshot.files_unanalyzed)} fichier(s) modifié(s) sans analyse Watch.{_R}")

    lines.append(f"{'─' * _W}\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Rapport de commit
# ─────────────────────────────────────────────────────────────────────────────

def commit_report(commit_hash: str, files_analyzed: list, score_before: float = 0) -> str:
    """
    Résumé d'un commit analysé.
    files_analyzed : liste de FileAnalysis produits par GitBranchAnalyzer.
    """
    total_critical = sum(f.get("bugs_critical", 0) for f in files_analyzed)
    total_high     = sum(f.get("bugs_high",     0) for f in files_analyzed)
    total_score    = sum(f.get("score",         0) for f in files_analyzed)

    lines = []
    lines.append(f"\n{'═' * _W}")
    lines.append(f"  Commit {commit_hash}  —  {len(files_analyzed)} fichier(s) analysé(s)")
    lines.append(f"  Score de risque : {total_score:.0f}")
    lines.append(f"{'─' * _W}")

    for f in sorted(files_analyzed, key=lambda x: x.get("score", 0), reverse=True):
        name = f.get("path", "?").split("/")[-1]
        c = f.get("bugs_critical", 0)
        h = f.get("bugs_high",     0)
        m = f.get("bugs_medium",   0)
        indicator = f"{_RD}●{_R}" if c else (f"{_YL}●{_R}" if h else f"{_GR}●{_R}")
        lines.append(f"  {indicator}  {name:<32}  C:{c} H:{h} M:{m}")

    lines.append(f"{'═' * _W}\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Rapport de branche
# ─────────────────────────────────────────────────────────────────────────────

def branch_report(report: "BranchReport") -> str:
    """
    Rapport complet de la branche avec verdict de merge.
    C'est le rapport le plus important — affiché à la fin de l'analyse de branche.
    """
    lines = []
    lines.append(f"\n{'═' * _W}")
    lines.append(f"  Rapport de branche : {report.branch}  →  {report.base}")
    lines.append(f"  Merge-base : {report.merge_base_hash[:8] if report.merge_base_hash != 'unknown' else '—'}")
    lines.append(f"  Commits dans la branche : {len(report.commits)}")
    lines.append(f"  Fichiers modifiés : {len(report.files)}")
    lines.append(f"{'─' * _W}")

    # Timeline des commits
    if report.commits:
        lines.append(f"\n  Commits de la branche :")
        for c in report.commits[:8]:
            msg = c.get("message", "")[:40]
            lines.append(f"    {_DM}{c.get('hash','?')[:7]}{_R}  {msg}")
        if len(report.commits) > 8:
            lines.append(f"    {_DM}... +{len(report.commits)-8} commit(s){_R}")

    # Fichiers propres
    if report.files_clean:
        lines.append(f"\n  {_GR}Fichiers propres ({len(report.files_clean)}) :{_R}")
        for f in report.files_clean[:5]:
            lines.append(f"    {_GR}✓{_R}  {f.path.split('/')[-1]}")
        if len(report.files_clean) > 5:
            lines.append(f"    {_DM}... +{len(report.files_clean)-5}{_R}")

    # Fichiers avec problèmes
    if report.files_with_issues:
        lines.append(f"\n  Fichiers avec problèmes ({len(report.files_with_issues)}) :")
        lines.append(f"  {'─' * 50}")
        lines.append(f"  {'Fichier':<30}  {'C':>3} {'H':>3} {'M':>3}")
        lines.append(f"  {'─' * 50}")
        for f in report.files_with_issues:
            name = f.path.split("/")[-1][:30]
            c = f"{_RD}{f.bugs_critical}{_R}" if f.bugs_critical else "0"
            h = f"{_YL}{f.bugs_high}{_R}"     if f.bugs_high     else "0"
            lines.append(f"  {name:<30}  {c:>3} {h:>3} {f.bugs_medium:>3}")

    # Conflits potentiels
    if report.conflict_risks:
        lines.append(f"\n  {_YL}Risques de conflit ({len(report.conflict_risks)}) :{_R}")
        for c in report.conflict_risks:
            lines.append(f"    {_YL}⚡{_R}  {c}")

    # Verdict
    lines.append(f"\n{'═' * _W}")
    verdict_line = _verdict_banner(report.verdict)
    lines.append(verdict_line)
    lines.append(f"\n  {report.recommendation}")
    lines.append(f"{'═' * _W}\n")

    return "\n".join(lines)


def save_branch_report_json(report: "BranchReport", output_dir: Path) -> Path:
    """
    Sauvegarde le rapport en JSON dans data/git_reports/.
    Utile pour intégration CI/CD ou archivage.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = report.branch.replace("/", "_").replace("\\", "_")
    out_path  = output_dir / f"branch_{safe_name}_{ts}.json"

    data = {
        "branch":          report.branch,
        "base":            report.base,
        "merge_base_hash": report.merge_base_hash,
        "generated_at":    datetime.now().isoformat(),
        "verdict":         report.verdict,
        "recommendation":  report.recommendation,
        "total_score":     report.total_score,
        "commits":         report.commits,
        "conflict_risks":  report.conflict_risks,
        "files": [
            {
                "path":          f.path,
                "status":        f.status,
                "bugs_critical": f.bugs_critical,
                "bugs_high":     f.bugs_high,
                "bugs_medium":   f.bugs_medium,
                "score":         f.score,
                "from_cache":    f.from_cache,
            }
            for f in report.files
        ],
    }

    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers d'affichage
# ─────────────────────────────────────────────────────────────────────────────

def _level_badge(level: str) -> str:
    badges = {
        "CLEAN":    f"{_GR}CLEAN{_R}",
        "WATCH":    f"{_CY}WATCH{_R}",
        "WARN":     f"{_YL}WARN{_R}",
        "CRITICAL": f"{_RD}{_B}CRITICAL{_R}",
    }
    return badges.get(level, level)


def _verdict_banner(verdict: str) -> str:
    banners = {
        "MERGE_OK":      f"  {_GR}{_B}  ✓  MERGE OK — branche prête à intégrer.{_R}",
        "MERGE_WARN":    f"  {_YL}{_B}  ⚠  MERGE AVEC ATTENTION — vérifications recommandées.{_R}",
        "MERGE_BLOCKED": f"  {_RD}{_B}  ✗  MERGE BLOQUÉ — corrections requises avant merge.{_R}",
    }
    return banners.get(verdict, f"  {verdict}")