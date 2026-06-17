"""
git_commit_linter.py — F3 : Validateur de messages de commit (Conventional Commits).

Spécification : https://www.conventionalcommits.org/en/v1.0.0/
Format attendu : type(scope): description [BREAKING CHANGE]

Rôle :
  Valide le format et la qualité du message de commit avant push.
  Retourne un rapport structuré avec les violations et suggestions.

Architecture :
  - Validation 100% locale (pas de LLM) — 0 token
  - Intégré dans git_hook.py (WARN) et accessible via le LangGraph
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ── Constantes Conventional Commits ──────────────────────────────────────────

VALID_TYPES = {
    "feat":     "Nouvelle fonctionnalité",
    "fix":      "Correction de bug",
    "docs":     "Documentation seulement",
    "style":    "Formatage, ponctuation (pas de logique)",
    "refactor": "Refactoring sans fix ni feature",
    "perf":     "Amélioration de performances",
    "test":     "Ajout ou correction de tests",
    "build":    "Système de build, dépendances",
    "ci":       "Scripts CI/CD",
    "chore":    "Maintenance, tooling",
    "revert":   "Revert d'un commit précédent",
    "wip":      "Work in progress (non-bloquant)",
}

CONVENTIONAL_COMMIT_RE = re.compile(
    r"^(?P<type>[a-z]+)"          # type : feat, fix, docs …
    r"(?:\((?P<scope>[^)]+)\))?"  # scope optionnel : (auth)
    r"(?P<breaking>!)?"           # breaking change marqueur
    r":\s"                        # : + espace obligatoire
    r"(?P<desc>.+)$",             # description
    re.MULTILINE,
)

# Longueurs recommandées
MAX_HEADER_LEN  = 72
MIN_DESC_LEN    = 10
MAX_SCOPE_LEN   = 30

# Patterns qui indiquent un message de mauvaise qualité
BAD_MESSAGE_PATTERNS = [
    (r"^(wip|tmp|temp|test|fix|update|changes?|stuff|misc|various)\.?$", "Message trop générique"),
    (r"^\.", "Message commençant par un point"),
    (r"[.!?]$", "Description ne doit pas se terminer par de la ponctuation"),
    (r"^[A-Z]", "La description doit commencer en minuscule (après le type:)"),
]


# ── Structures de données ─────────────────────────────────────────────────────

@dataclass
class LintViolation:
    """Une violation de règle dans le message de commit."""
    rule:        str
    severity:    str       # ERROR | WARN | INFO
    message:     str
    suggestion:  str = ""


@dataclass
class CommitLintReport:
    """Rapport de validation d'un message de commit."""
    original_message: str
    violations:       List[LintViolation] = field(default_factory=list)
    suggested_message: str = ""
    score:            int = 100    # 100 = parfait, 0 = complètement invalide
    error:            Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def is_valid(self) -> bool:
        return not any(v.severity == "ERROR" for v in self.violations)

    @property
    def has_warnings(self) -> bool:
        return any(v.severity == "WARN" for v in self.violations)

    @property
    def errors(self) -> List[LintViolation]:
        return [v for v in self.violations if v.severity == "ERROR"]

    @property
    def warnings(self) -> List[LintViolation]:
        return [v for v in self.violations if v.severity == "WARN"]


# ── Linter ────────────────────────────────────────────────────────────────────

class GitCommitLinter:
    """
    Valide un message de commit selon la spécification Conventional Commits.

    Règles (par sévérité) :
      ERROR : format non conforme, type invalide, description vide
      WARN  : header trop long, scope mal formaté, mauvaise capitalisation
      INFO  : suggestions de style
    """

    def lint(self, message: str) -> CommitLintReport:
        """Valide un message de commit et retourne le rapport."""
        report = CommitLintReport(original_message=message)

        if not message or not message.strip():
            report.violations.append(LintViolation(
                rule="empty-message",
                severity="ERROR",
                message="Le message de commit est vide.",
                suggestion="feat: description de votre changement",
            ))
            report.score = 0
            return report

        # Séparer le header des body/footer
        lines = message.strip().splitlines()
        header = lines[0].strip()

        # Règle 1 : Format Conventional Commits
        match = CONVENTIONAL_COMMIT_RE.match(header)
        if not match:
            report.violations.append(LintViolation(
                rule="invalid-format",
                severity="ERROR",
                message=f"Le header '{header}' ne respecte pas le format Conventional Commits.",
                suggestion="feat(scope): description en minuscule",
            ))
            report.score -= 50
        else:
            commit_type = match.group("type")
            scope       = match.group("scope") or ""
            breaking    = match.group("breaking") or ""
            desc        = match.group("desc") or ""

            # Règle 2 : Type valide
            if commit_type not in VALID_TYPES:
                close = self._closest_type(commit_type)
                report.violations.append(LintViolation(
                    rule="invalid-type",
                    severity="ERROR",
                    message=f"Type '{commit_type}' invalide.",
                    suggestion=f"Types valides : {', '.join(sorted(VALID_TYPES))}. Voulez-vous dire '{close}' ?",
                ))
                report.score -= 30

            # Règle 3 : Description non vide
            if not desc.strip():
                report.violations.append(LintViolation(
                    rule="empty-description",
                    severity="ERROR",
                    message="La description après ':' est vide.",
                    suggestion=f"{commit_type}: description de votre changement",
                ))
                report.score -= 30

            # Règle 4 : Description assez longue
            elif len(desc.strip()) < MIN_DESC_LEN:
                report.violations.append(LintViolation(
                    rule="short-description",
                    severity="WARN",
                    message=f"Description trop courte ({len(desc.strip())} chars, min {MIN_DESC_LEN}).",
                    suggestion="Décrivez brièvement ce qui change et pourquoi.",
                ))
                report.score -= 10

            # Règle 5 : Description ne commence pas en majuscule
            elif desc and desc[0].isupper():
                report.violations.append(LintViolation(
                    rule="uppercase-description",
                    severity="WARN",
                    message=f"La description commence par une majuscule '{desc[0]}' (convention: minuscule).",
                    suggestion=f"{commit_type}({scope}): {desc[0].lower()}{desc[1:]}".replace("()", ""),
                ))
                report.score -= 5

            # Règle 6 : Patterns de mauvaise qualité
            for pattern, reason in BAD_MESSAGE_PATTERNS:
                if re.search(pattern, desc.strip()):
                    report.violations.append(LintViolation(
                        rule="low-quality-message",
                        severity="WARN",
                        message=f"{reason}: '{desc.strip()}'",
                        suggestion="Soyez plus précis sur ce que vous avez changé.",
                    ))
                    report.score -= 10
                    break

            # Règle 7 : Scope mal formaté (espaces, majuscules)
            if scope:
                if " " in scope:
                    report.violations.append(LintViolation(
                        rule="scope-has-spaces",
                        severity="WARN",
                        message=f"Le scope '{scope}' contient des espaces.",
                        suggestion=f"Utilisez des tirets: '{scope.replace(' ', '-')}'",
                    ))
                    report.score -= 5
                elif scope != scope.lower():
                    report.violations.append(LintViolation(
                        rule="scope-uppercase",
                        severity="WARN",
                        message=f"Le scope '{scope}' contient des majuscules.",
                        suggestion=f"Scope en minuscule : '{scope.lower()}'",
                    ))
                    report.score -= 5
                elif len(scope) > MAX_SCOPE_LEN:
                    report.violations.append(LintViolation(
                        rule="scope-too-long",
                        severity="WARN",
                        message=f"Le scope est trop long ({len(scope)} chars, max {MAX_SCOPE_LEN}).",
                    ))
                    report.score -= 5

        # Règle 8 : Header trop long
        if len(header) > MAX_HEADER_LEN:
            report.violations.append(LintViolation(
                rule="header-too-long",
                severity="WARN",
                message=f"Header trop long ({len(header)} chars, max {MAX_HEADER_LEN}).",
                suggestion="Raccourcissez le header, déplacez les détails dans le body.",
            ))
            report.score -= 10

        # Règle 9 : Ligne vide entre header et body
        if len(lines) > 1 and lines[1].strip():
            report.violations.append(LintViolation(
                rule="missing-blank-line",
                severity="WARN",
                message="Ligne vide manquante entre le header et le body.",
                suggestion="Ajoutez une ligne vide après le header.",
            ))
            report.score -= 5

        # Suggérer un message corrigé si des ERRORs
        if report.errors:
            report.suggested_message = self._suggest_correction(message)

        report.score = max(0, report.score)
        return report

    def lint_from_git(self, project_path: Path) -> CommitLintReport:
        """Lit le message du dernier commit stagé (COMMIT_EDITMSG) et le valide."""
        editmsg = project_path / ".git" / "COMMIT_EDITMSG"
        if editmsg.exists():
            message = editmsg.read_text(encoding="utf-8", errors="replace").strip()
            # Supprimer les lignes de commentaire git
            message = "\n".join(l for l in message.splitlines() if not l.startswith("#"))
            return self.lint(message.strip())

        # Fallback : dernier commit
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--pretty=format:%s"],
                capture_output=True, text=True, cwd=str(project_path),
            )
            return self.lint(result.stdout.strip())
        except Exception as e:
            r = CommitLintReport(original_message="")
            r.error = str(e)
            return r

    def _closest_type(self, t: str) -> str:
        """Trouve le type valide le plus proche par distance de Levenshtein simple."""
        def dist(a: str, b: str) -> int:
            if len(a) > len(b): a, b = b, a
            return min(
                abs(len(a) - len(b)),
                sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b)),
            )
        return min(VALID_TYPES, key=lambda v: dist(t, v))

    def _suggest_correction(self, original: str) -> str:
        """Génère un message de commit corrigé basique."""
        lines = original.strip().splitlines()
        header = lines[0].strip() if lines else ""

        # Essayer d'extraire la description brute
        # Format connu invalide : "Added stuff" → "feat: added stuff"
        header_lower = header.lower()
        if header_lower.startswith("add"):
            return f"feat: {header_lower}"
        elif header_lower.startswith(("fix", "bug")):
            return f"fix: {header_lower}"
        elif header_lower.startswith("update"):
            return f"chore: {header_lower}"
        elif header_lower.startswith(("doc", "readme")):
            return f"docs: {header_lower}"

        return f"feat: {header_lower}"


# ── Point d'entrée public ─────────────────────────────────────────────────────

_linter = GitCommitLinter()


def lint_commit_message(message: str) -> CommitLintReport:
    """Valide un message de commit explicite."""
    return _linter.lint(message)


def lint_staged_commit(project_path: Path) -> CommitLintReport:
    """Valide le message du commit en cours (COMMIT_EDITMSG)."""
    return _linter.lint_from_git(project_path)
