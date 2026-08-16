
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from smart_git.git_secret_scanner import scan_staged_secrets
from smart_git.git_commit_linter import lint_commit_message, lint_staged_commit
from smart_git.git_test_impact import analyze_test_impact_staged
from smart_git.git_diff_parser import (
    is_git_repo,
    get_staged_files,
    get_session_stats,
    detect_conflict_files,
)



WEIGHT_SECRETS  = 35   # bloquant — sécurité avant tout
WEIGHT_CONFLICTS = 25  # bloquant — un commit sur état en conflit est dangereux
WEIGHT_LINT     = 20   # qualité du message de commit
WEIGHT_IMPACT   = 20   # couverture de tests des fichiers modifiés


@dataclass
class CheckResult:
    """État normalisé d'un check unitaire pour l'affichage cockpit."""
    id:          str                       # secrets | conflicts | lint | impact | session
    state:       str                       # ok | warn | error | skipped
    summary:     str                       # texte court affiché dans la ligne
    blocking:    bool = False              # contribue au badge / verdict BLOCKED
    detail:      Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommitReadinessReport:
    """Verdict agrégé de préparation au commit."""
    score:        int = 100
    verdict:      str = "READY"            # READY | WARN | BLOCKED
    blockers:     List[str] = field(default_factory=list)
    warnings:     List[str] = field(default_factory=list)
    checks:       Dict[str, CheckResult] = field(default_factory=dict)
    branch:       str = ""
    staged_files: List[Dict[str, str]] = field(default_factory=list)
    lines_added:  int = 0
    lines_removed: int = 0
    error:        Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score":         self.score,
            "verdict":       self.verdict,
            "blockers":      self.blockers,
            "warnings":      self.warnings,
            "branch":        self.branch,
            "staged_files":  self.staged_files,
            "staged_count":  len(self.staged_files),
            "lines_added":   self.lines_added,
            "lines_removed": self.lines_removed,
            "checks": {
                cid: {
                    "id":       c.id,
                    "state":    c.state,
                    "summary":  c.summary,
                    "blocking": c.blocking,
                    "detail":   c.detail,
                }
                for cid, c in self.checks.items()
            },
            "success": self.success,
            "error":   self.error,
        }


# ── Helpers de branche ─────────────────────────────────────────────────────────

def _current_branch(project_path: Path) -> str:
    import subprocess
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=str(project_path),
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


# ── Agrégateur principal ───────────────────────────────────────────────────────

def evaluate_commit_readiness(
    project_path: Path,
    commit_message: str = "",
) -> CommitReadinessReport:
    """
    Exécute et agrège les cinq checks locaux en un verdict de commit-readiness.

    Args:
        project_path:   racine du dépôt git local.
        commit_message: message à valider. Vide → lit .git/COMMIT_EDITMSG si présent,
                        sinon le check lint est marqué 'skipped' (non pénalisant).
    """
    report = CommitReadinessReport()

    if not is_git_repo(project_path):
        report.error = "not a git repository"
        report.verdict = "BLOCKED"
        report.score = 0
        return report

    report.branch = _current_branch(project_path)

    # Contexte : fichiers stagés + stats de session
    try:
        report.staged_files = get_staged_files(project_path)
        stats = get_session_stats(project_path)
        report.lines_added   = int(stats.get("lines_added", 0))
        report.lines_removed = int(stats.get("lines_removed", 0))
    except Exception:
        report.staged_files = []

    score = 0
    earned_secrets = earned_conflicts = earned_lint = earned_impact = False

    # ── 1. Secrets (BLOQUANT, 35 pts) ──────────────────────────────────────────
    try:
        sec = scan_staged_secrets(project_path)
        if not sec.success:
            report.checks["secrets"] = CheckResult(
                "secrets", "warn",
                f"scan unavailable ({sec.error or 'error'})",
            )
        elif sec.has_secrets:
            n = len(sec.findings)
            report.checks["secrets"] = CheckResult(
                "secrets", "error",
                f"{n} secret{'s' if n != 1 else ''} in staged files",
                blocking=True,
                detail={"count": n, "scanned": sec.scanned_files},
            )
            report.blockers.append(
                f"{n} secret{'s' if n != 1 else ''} detected in staged files"
            )
        else:
            earned_secrets = True
            report.checks["secrets"] = CheckResult(
                "secrets", "ok",
                f"0 in staged · {sec.scanned_files} scanned",
                detail={"count": 0, "scanned": sec.scanned_files},
            )
    except Exception as e:  # pragma: no cover - defensive
        report.checks["secrets"] = CheckResult("secrets", "warn", f"error: {e}")

    # ── 2. Conflicts (BLOQUANT, 25 pts) ────────────────────────────────────────
    try:
        conflict_files = detect_conflict_files(project_path)
        if conflict_files:
            n = len(conflict_files)
            report.checks["conflicts"] = CheckResult(
                "conflicts", "error",
                f"{n} file{'s' if n != 1 else ''} in conflict",
                blocking=True,
                detail={"files": conflict_files},
            )
            report.blockers.append(
                f"{n} unresolved merge conflict{'s' if n != 1 else ''}"
            )
        else:
            earned_conflicts = True
            report.checks["conflicts"] = CheckResult(
                "conflicts", "ok", "no conflict risk",
            )
    except Exception as e:  # pragma: no cover - defensive
        report.checks["conflicts"] = CheckResult("conflicts", "warn", f"error: {e}")

    # ── 3. Commit lint (AVERTISSEMENT, 20 pts) ─────────────────────────────────
    try:
        msg = commit_message.strip()
        editmsg = project_path / ".git" / "COMMIT_EDITMSG"
        if not msg and editmsg.exists():
            lint = lint_staged_commit(project_path)
            has_msg = bool(lint.original_message.strip())
        elif msg:
            lint = lint_commit_message(msg)
            has_msg = True
        else:
            lint = None
            has_msg = False

        if not has_msg or lint is None:
            # Pas de message à valider → check neutre (n'enlève pas de points).
            earned_lint = True
            report.checks["lint"] = CheckResult(
                "lint", "skipped", "no message yet",
            )
        elif lint.is_valid and not lint.violations:
            earned_lint = True
            report.checks["lint"] = CheckResult(
                "lint", "ok", f"score {lint.score}/100",
                detail={"score": lint.score},
            )
        elif lint.is_valid:
            # Warnings seulement → points partiels.
            n = len(lint.violations)
            report.checks["lint"] = CheckResult(
                "lint", "warn",
                f"score {lint.score}/100 · {n} suggestion{'s' if n != 1 else ''}",
                detail={"score": lint.score, "violations": n},
            )
            report.warnings.append(
                f"commit message has {n} style suggestion{'s' if n != 1 else ''}"
            )
            score += int(WEIGHT_LINT * 0.5)
        else:
            # Erreurs de format → bloque la qualité (non bloquant pour le commit
            # mais pénalise lourdement le score).
            report.checks["lint"] = CheckResult(
                "lint", "error",
                f"invalid format · score {lint.score}/100",
                detail={"score": lint.score,
                        "suggested": lint.suggested_message},
            )
            report.warnings.append(
                "commit message does not follow Conventional Commits"
            )
    except Exception as e:  # pragma: no cover - defensive
        earned_lint = True
        report.checks["lint"] = CheckResult("lint", "warn", f"error: {e}")

    # ── 4. Test impact (AVERTISSEMENT, 20 pts) ─────────────────────────────────
    try:
        impact = analyze_test_impact_staged(project_path)
        if not impact.success:
            earned_impact = True
            report.checks["impact"] = CheckResult(
                "impact", "warn", f"analysis unavailable",
            )
        elif impact.total_files == 0:
            # Aucun fichier source stagé → rien à couvrir.
            earned_impact = True
            report.checks["impact"] = CheckResult(
                "impact", "skipped", "no source files staged",
            )
        elif not impact.has_gaps:
            earned_impact = True
            pct = round(impact.coverage_ratio * 100)
            report.checks["impact"] = CheckResult(
                "impact", "ok", f"{pct}% covered",
                detail={"coverage": pct, "uncovered": 0},
            )
        else:
            pct = round(impact.coverage_ratio * 100)
            report.checks["impact"] = CheckResult(
                "impact", "warn",
                f"{pct}% covered · {impact.uncovered_files} gap"
                f"{'s' if impact.uncovered_files != 1 else ''}",
                detail={"coverage": pct, "uncovered": impact.uncovered_files},
            )
            report.warnings.append(
                f"{impact.uncovered_files} changed file"
                f"{'s' if impact.uncovered_files != 1 else ''} without tests"
            )
            # Points proportionnels à la couverture.
            score += int(WEIGHT_IMPACT * impact.coverage_ratio)
    except Exception as e:  # pragma: no cover - defensive
        earned_impact = True
        report.checks["impact"] = CheckResult("impact", "warn", f"error: {e}")

    # ── 5. Session (INFORMATIF — ne pénalise pas le score) ─────────────────────
    try:
        staged_n = len(report.staged_files)
        report.checks["session"] = CheckResult(
            "session", "ok" if staged_n else "skipped",
            (
                f"{staged_n} staged · +{report.lines_added}/-{report.lines_removed}"
                if staged_n else "nothing staged"
            ),
            detail={"staged": staged_n},
        )
    except Exception:
        pass

    # ── Score final ────────────────────────────────────────────────────────────
    if earned_secrets:
        score += WEIGHT_SECRETS
    if earned_conflicts:
        score += WEIGHT_CONFLICTS
    if earned_lint:
        score += WEIGHT_LINT
    if earned_impact:
        score += WEIGHT_IMPACT

    report.score = max(0, min(100, score))

    # ── Verdict ────────────────────────────────────────────────────────────────
    if report.blockers:
        report.verdict = "BLOCKED"
    elif report.warnings or report.score < 80:
        report.verdict = "WARN"
    else:
        report.verdict = "READY"

    return report
