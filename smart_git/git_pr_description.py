"""
git_pr_description.py — F7 : Générateur automatique de description de PR.

Rôle :
  Génère une description professionnelle de Pull Request à partir de :
    - Les commits de la branche (titres + corps)
    - Le diff résumé (fichiers changés, +/- lignes)
    - Les métadonnées de la PR (titre, base, head)

Architecture :
  - LLM requis (résumé contextuel) — utilise le cache Redis avant appel
  - Accessible via le LangGraph (intent: pr_description)
  - Retourne un markdown structuré (Summary, Changes, Testing, Breaking)

Template de sortie :
  ## Summary
  ## Changes
  ## Testing
  ## Breaking Changes (si applicable)
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Structures de données ─────────────────────────────────────────────────────

@dataclass
class PRDescriptionReport:
    """Résultat de la génération de description de PR."""
    title:            str
    description:      str = ""  # markdown généré
    commits_used:     int = 0
    files_changed:    int = 0
    additions:        int = 0
    deletions:        int = 0
    from_cache:       bool = False
    error:            Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.description)


# ── Générateur ────────────────────────────────────────────────────────────────

class GitPRDescriptionGenerator:
    """
    Génère une description de PR structurée via LLM.

    Flux :
      1. Collecter commits (git log) + diff summary (git diff --stat)
      2. Construire le contexte (prompt compact)
      3. Vérifier le cache Redis (évite les appels LLM répétés)
      4. Appeler le LLM si cache miss
      5. Retourner le markdown formaté
    """

    def generate_from_local(
        self,
        project_path: Path,
        branch: str,
        base: str = "main",
        pr_title: str = "",
    ) -> PRDescriptionReport:
        """Génère une description depuis le dépôt local."""
        report = PRDescriptionReport(title=pr_title or branch)

        try:
            # Collecter les données
            commits  = self._get_commits(project_path, branch, base)
            diff_stat = self._get_diff_stat(project_path, branch, base)
            file_list = self._get_changed_files(project_path, branch, base)

            report.commits_used  = len(commits)
            report.files_changed = diff_stat.get("files", 0)
            report.additions     = diff_stat.get("additions", 0)
            report.deletions     = diff_stat.get("deletions", 0)

            if not commits and not file_list:
                report.error = "Aucun commit ni fichier modifié trouvé."
                return report

            # Cache Redis
            cache_key = self._build_cache_key(branch, base, commits)
            cached = self._read_cache(cache_key)
            if cached:
                report.description = cached
                report.from_cache  = True
                return report

            # Générer via LLM
            context = self._build_llm_context(
                branch, base, commits, file_list, diff_stat, pr_title
            )
            description = self._call_llm(context)
            if description:
                report.description = description
                self._write_cache(cache_key, description)
            else:
                # Fallback : description basique sans LLM
                report.description = self._fallback_description(
                    branch, commits, file_list, diff_stat
                )

        except Exception as e:
            logger.error("PR description generation error: %s", e)
            report.error = str(e)

        return report

    def generate_from_github(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        github_service,
        pr_data: Optional[Dict] = None,
    ) -> PRDescriptionReport:
        """Génère une description depuis les données GitHub."""
        report = PRDescriptionReport(title=f"PR #{pr_number}")

        try:
            from smart_git.pr_context_cache import fetch_pr_info_cached, fetch_pr_files_cached

            if not pr_data:
                # cache partagé ~90s — cf. pr_context_cache.py, Axe b
                pr_data = fetch_pr_info_cached(github_service, owner, repo, pr_number)

            pr_title = pr_data.get("title", f"PR #{pr_number}")
            report.title = pr_title

            # Commits via API (github_service = GitHubClient from code_mode_client)
            commits_raw = github_service.get_pr_commits(owner, repo, pr_number) or []
            commits = [
                {
                    "hash":    c.get("sha", "")[:7],
                    "message": c.get("commit", {}).get("message", "").splitlines()[0],
                    "author":  c.get("commit", {}).get("author", {}).get("name", ""),
                }
                for c in commits_raw
            ]

            # Fichiers via API (cache partagé ~90s)
            files_raw = fetch_pr_files_cached(github_service, owner, repo, pr_number)
            file_list = [f.get("filename", "") for f in files_raw]
            additions = sum(f.get("additions", 0) for f in files_raw)
            deletions = sum(f.get("deletions", 0) for f in files_raw)

            report.commits_used  = len(commits)
            report.files_changed = len(file_list)
            report.additions     = additions
            report.deletions     = deletions

            # Cache
            cache_key = self._build_cache_key(
                f"pr-{pr_number}", "", [c["message"] for c in commits]
            )
            cached = self._read_cache(cache_key)
            if cached:
                report.description = cached
                report.from_cache  = True
                return report

            diff_stat = {"files": len(file_list), "additions": additions, "deletions": deletions}
            context = self._build_llm_context(
                f"PR #{pr_number}", "main", commits, file_list, diff_stat, pr_title
            )
            description = self._call_llm(context)
            if description:
                report.description = description
                self._write_cache(cache_key, description)
            else:
                report.description = self._fallback_description(
                    pr_title, commits, file_list, diff_stat
                )

        except Exception as e:
            logger.error("PR description GitHub generation error: %s", e)
            report.error = str(e)

        return report

    # ── Collecte des données git ──────────────────────────────────────────────

    def _get_commits(self, project_path: Path, branch: str, base: str) -> List[Dict]:
        """Récupère les commits exclusifs à la branche."""
        try:
            result = subprocess.run(
                ["git", "log", f"{base}..{branch}", "--pretty=format:%h|%s|%an", "--no-merges"],
                capture_output=True, text=True, cwd=str(project_path),
            )
            commits = []
            for line in result.stdout.strip().splitlines():
                parts = line.split("|", 2)
                if len(parts) == 3:
                    commits.append({"hash": parts[0], "message": parts[1], "author": parts[2]})
            return commits
        except Exception:
            return []

    def _get_diff_stat(self, project_path: Path, branch: str, base: str) -> Dict:
        """Récupère les statistiques du diff."""
        try:
            result = subprocess.run(
                ["git", "diff", f"{base}...{branch}", "--shortstat"],
                capture_output=True, text=True, cwd=str(project_path),
            )
            text = result.stdout.strip()
            files = int(m.group(1)) if (m := __import__("re").search(r"(\d+) file", text)) else 0
            adds  = int(m.group(1)) if (m := __import__("re").search(r"(\d+) insertion", text)) else 0
            dels  = int(m.group(1)) if (m := __import__("re").search(r"(\d+) deletion", text)) else 0
            return {"files": files, "additions": adds, "deletions": dels}
        except Exception:
            return {"files": 0, "additions": 0, "deletions": 0}

    def _get_changed_files(self, project_path: Path, branch: str, base: str) -> List[str]:
        """Liste les fichiers changés."""
        try:
            result = subprocess.run(
                ["git", "diff", f"{base}...{branch}", "--name-only"],
                capture_output=True, text=True, cwd=str(project_path),
            )
            return [f.strip() for f in result.stdout.strip().splitlines() if f.strip()]
        except Exception:
            return []

    # ── LLM ──────────────────────────────────────────────────────────────────

    def _build_llm_context(
        self,
        branch: str,
        base: str,
        commits: List[Dict],
        file_list: List[str],
        diff_stat: Dict,
        pr_title: str,
    ) -> str:
        """Construit le prompt pour le LLM."""
        commit_lines = "\n".join(
            f"- [{c['hash']}] {c['message']} (by {c['author']})"
            for c in commits[:20]  # limiter à 20 commits
        )
        file_lines = "\n".join(f"- {f}" for f in file_list[:30])

        return f"""Generate a professional GitHub Pull Request description in markdown format.

PR Title: {pr_title or branch}
Branch: {branch} → {base}

Commits ({len(commits)} total):
{commit_lines}

Files changed ({diff_stat.get('files', 0)}):
{file_lines}
Stats: +{diff_stat.get('additions', 0)} / -{diff_stat.get('deletions', 0)} lines

Generate the following sections:
1. ## Summary (2-3 sentences describing what this PR does and why)
2. ## Changes (bulleted list of key changes grouped by feature/area)
3. ## Testing (how to test this change)
4. ## Breaking Changes (only if any, otherwise omit this section)

Be specific and concise. Use the commit messages and file names to infer the purpose.
Output only the markdown, no preamble."""

    def _call_llm(self, context: str) -> Optional[str]:
        """Appelle le LLM pour générer la description."""
        try:
            from services.llm_service import assistant_agent
            result = assistant_agent.chat(
                message    = context,
                session_id = "pr_description_gen",
                context    = {},
            )
            return result.get("response", "") or result.get("content", "")
        except Exception as e:
            logger.warning("LLM call failed for PR description: %s", e)
            return None

    # ── Cache Redis ───────────────────────────────────────────────────────────

    def _build_cache_key(self, branch: str, base: str, commits: List) -> str:
        """Construit une clé de cache basée sur les commits."""
        import hashlib
        content = f"{branch}:{base}:" + "|".join(
            (c if isinstance(c, str) else c.get("message", "")) for c in commits[:10]
        )
        return "pr_desc:" + hashlib.sha256(content.encode()).hexdigest()[:16]

    def _read_cache(self, key: str) -> Optional[str]:
        try:
            from services.mcp_redis_service import get_mcp_redis
            redis = get_mcp_redis()
            return redis.get(key) or None
        except Exception:
            return None

    def _write_cache(self, key: str, value: str) -> None:
        try:
            from services.mcp_redis_service import get_mcp_redis
            redis = get_mcp_redis()
            redis.set(key, value, ex=3600 * 6)  # TTL 6h
        except Exception:
            pass

    # ── Fallback sans LLM ─────────────────────────────────────────────────────

    def _fallback_description(
        self,
        branch: str,
        commits: List[Dict],
        file_list: List[str],
        diff_stat: Dict,
    ) -> str:
        """Génère une description basique sans LLM."""
        commit_lines = "\n".join(f"- {c['message']}" for c in commits[:15])
        file_lines   = "\n".join(f"- `{f}`" for f in file_list[:20])
        stats = f"+{diff_stat.get('additions', 0)} / -{diff_stat.get('deletions', 0)}"

        return f"""## Summary

This PR introduces changes on branch `{branch}`.

## Changes

### Commits
{commit_lines or '- No commits found'}

### Files Modified ({diff_stat.get('files', 0)} files, {stats})
{file_lines or '- No files found'}

## Testing

- Review the modified files listed above
- Run existing test suite to verify no regressions
"""


# ── Point d'entrée public ─────────────────────────────────────────────────────

_generator = GitPRDescriptionGenerator()


def generate_pr_description_local(
    project_path: Path,
    branch: str,
    base: str = "main",
    pr_title: str = "",
) -> PRDescriptionReport:
    """Génère une description de PR depuis le dépôt local."""
    return _generator.generate_from_local(project_path, branch, base, pr_title)


def generate_pr_description_github(
    owner: str,
    repo: str,
    pr_number: int,
    github_service,
    pr_data: Optional[Dict] = None,
) -> PRDescriptionReport:
    """Génère une description de PR depuis les données GitHub."""
    return _generator.generate_from_github(owner, repo, pr_number, github_service, pr_data)
