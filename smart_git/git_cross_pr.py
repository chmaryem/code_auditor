"""
git_cross_pr.py — F6 : Détection de conflits cross-PR.

Rôle :
  Détecte les fichiers modifiés simultanément dans plusieurs PR ouvertes.
  Un fichier touché par 2+ PRs en parallèle est un candidat au conflit
  lors du merge séquentiel.

Architecture :
  - Nécessite GitHub API (via MCPGitHubService)
  - Accessible via le LangGraph (intent: cross_pr_conflicts)
  - Résultat structuré avec niveau de risque par fichier

Niveaux de risque :
  HIGH   : fichier modifié dans 3+ PRs
  MEDIUM : fichier modifié dans 2 PRs
  LOW    : fichier modifié dans 1 PR mais PR de longue durée (> 7 jours)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


# ── Structures de données ─────────────────────────────────────────────────────

@dataclass
class PRFileConflict:
    """Un fichier partagé entre plusieurs PRs ouvertes."""
    file_path:   str
    pr_numbers:  List[int]
    pr_titles:   List[str]
    risk_level:  str       # HIGH | MEDIUM | LOW
    note:        str = ""


@dataclass
class CrossPRReport:
    """Rapport de détection de conflits cross-PR."""
    repo:            str
    total_open_prs:  int = 0
    conflicts:       List[PRFileConflict] = field(default_factory=list)
    pr_summaries:    List[Dict] = field(default_factory=list)
    error:           Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def has_conflicts(self) -> bool:
        return len(self.conflicts) > 0

    @property
    def high_risk_count(self) -> int:
        return sum(1 for c in self.conflicts if c.risk_level == "HIGH")

    @property
    def medium_risk_count(self) -> int:
        return sum(1 for c in self.conflicts if c.risk_level == "MEDIUM")


# ── Analyseur ─────────────────────────────────────────────────────────────────

class GitCrossPRAnalyzer:
    """
    Détecte les conflits potentiels entre PRs ouvertes.

    Algorithme :
      1. Lister toutes les PRs ouvertes via GitHub API
      2. Pour chaque PR, récupérer la liste des fichiers modifiés
      3. Construire un index inversé : fichier → [pr_number, ...]
      4. Signaler les fichiers présents dans 2+ PRs
    """

    def analyze(
        self,
        owner: str,
        repo: str,
        github_service,
        base_branch: str = "main",
        current_pr: Optional[int] = None,
    ) -> CrossPRReport:
        """
        Lance l'analyse cross-PR.

        Args:
            owner:           Organisation/utilisateur GitHub
            repo:            Nom du dépôt
            github_service:  Instance de MCPGitHubService
            base_branch:     Branche de base (filtre les PRs)
            current_pr:      PR courante à exclure de l'analyse (optionnel)
        """
        report = CrossPRReport(repo=f"{owner}/{repo}")

        try:
            # Récupérer toutes les PRs ouvertes
            prs = github_service.list_open_prs(owner, repo, base=base_branch)
            if not prs:
                return report

            # Filtrer la PR courante
            if current_pr:
                prs = [pr for pr in prs if pr.get("number") != current_pr]

            report.total_open_prs = len(prs)

            # Construire l'index : fichier → liste de (pr_number, pr_title, pr_created_at)
            file_index: Dict[str, List[Dict]] = {}

            for pr in prs:
                pr_number = pr.get("number", 0)
                pr_title  = pr.get("title", "")
                pr_created = pr.get("created_at", "")

                # Récupérer les fichiers de la PR
                files = github_service.get_pr_files(owner, repo, pr_number)
                if not files:
                    continue

                pr_summary = {
                    "number":   pr_number,
                    "title":    pr_title,
                    "files":    len(files),
                    "created":  pr_created,
                    "age_days": self._age_days(pr_created),
                    "url":      pr.get("html_url", ""),
                    "author":   pr.get("user", {}).get("login", ""),
                }
                report.pr_summaries.append(pr_summary)

                for f in files:
                    fname = f.get("filename", "")
                    if not fname:
                        continue
                    file_index.setdefault(fname, []).append({
                        "pr_number": pr_number,
                        "pr_title":  pr_title,
                        "pr_age":    self._age_days(pr_created),
                    })

            # Déterminer les conflits
            for file_path, pr_list in file_index.items():
                if len(pr_list) >= 3:
                    risk = "HIGH"
                    note = f"Fichier modifié dans {len(pr_list)} PRs simultanées — risque de conflit élevé"
                elif len(pr_list) == 2:
                    risk = "MEDIUM"
                    note = f"Fichier modifié dans 2 PRs — merge séquentiel requis"
                elif len(pr_list) == 1 and pr_list[0]["pr_age"] > 7:
                    risk = "LOW"
                    note = f"PR ancienne ({pr_list[0]['pr_age']} jours) — risque de divergence"
                else:
                    continue

                report.conflicts.append(PRFileConflict(
                    file_path  = file_path,
                    pr_numbers = [p["pr_number"] for p in pr_list],
                    pr_titles  = [p["pr_title"]  for p in pr_list],
                    risk_level = risk,
                    note       = note,
                ))

            # Trier par risque puis par chemin
            risk_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            report.conflicts.sort(key=lambda c: (risk_order[c.risk_level], c.file_path))

        except Exception as e:
            report.error = str(e)

        return report

    def _age_days(self, iso_date: str) -> int:
        """Calcule l'âge d'une date ISO en jours."""
        if not iso_date:
            return 0
        try:
            dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
            now = datetime.now(tz=timezone.utc)
            return (now - dt).days
        except Exception:
            return 0


# ── Point d'entrée public ─────────────────────────────────────────────────────

_analyzer = GitCrossPRAnalyzer()


def detect_cross_pr_conflicts(
    owner: str,
    repo: str,
    github_service,
    base_branch: str = "main",
    current_pr: Optional[int] = None,
) -> CrossPRReport:
    """Détecte les conflits cross-PR pour un dépôt."""
    return _analyzer.analyze(owner, repo, github_service, base_branch, current_pr)
