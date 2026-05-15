"""
cd_rollback_advisor.py — Rollback intelligence for failed deployments.

Compares a failed deployment against the last successful one, generates
a specific rollback command with the exact image tag, and assesses the
rollback risk.

NOTE: This module only SUGGESTS rollbacks — it never auto-triggers one.
      The suggestion is posted as a PR comment or console notification.

Usage:
    advisor = CDRollbackAdvisor()
    advice = advisor.advise(
        repo="owner/repo",
        owner="owner",
        environment="production",
        failed_deploy_id="deploy-12345-abc",
    )
    if advice:
        print(advice.command)
        print(advice.risk_level)
"""
from __future__ import annotations

import logging
import os
import json
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RollbackAdvice:
    """Rollback recommendation produced by CDRollbackAdvisor."""
    repo:           str
    environment:    str
    failed_version: str
    target_version: str          # Version to roll back TO
    target_sha:     str
    command:        str          # Exact rollback command to run
    risk_level:     str          # "LOW" | "MEDIUM" | "HIGH"
    risk_reasons:   List[str]    = field(default_factory=list)
    diff_files:     int          = 0
    commits_apart:  int          = 0
    comment_body:   str          = ""

    def to_pr_comment(self) -> str:
        """Format as a GitHub PR comment."""
        risk_icon = {
            "LOW":    "🟢",
            "MEDIUM": "🟡",
            "HIGH":   "🔴",
        }.get(self.risk_level, "❓")

        lines = [
            "## 🔄 Code Auditor — Rollback Recommendation",
            f"",
            f"**Environment:** `{self.environment}` | "
            f"**Risk:** {risk_icon} `{self.risk_level}`",
            f"",
            f"| Field | Value |",
            f"|---|---|",
            f"| Failed version | `{self.failed_version}` |",
            f"| Rollback to | `{self.target_version}` |",
            f"| Target SHA | `{self.target_sha[:8]}` |",
            f"| Files changed | {self.diff_files} |",
            f"| Commits apart | {self.commits_apart} |",
            f"",
            f"**Rollback command:**",
            f"```bash",
            f"{self.command}",
            f"```",
        ]

        if self.risk_reasons:
            lines += ["", "**Risk factors:**"]
            for r in self.risk_reasons:
                lines.append(f"- {r}")

        lines += [
            "",
            "> ⚠️ *Review before running. Code Auditor does not auto-rollback.*",
        ]
        return "\n".join(lines)


class CDRollbackAdvisor:
    """
    Advises on rollback strategy after a failed deployment.

    Algorithm:
      1. Fetch the last successful deployment from CDDeployTracker
      2. Compute GitHub commit diff (files changed, commits apart)
      3. Assess risk: LOW / MEDIUM / HIGH
      4. Generate exact rollback command using docker compose with specific tag
      5. Store advice in Redis for audit trail
    """

    def __init__(self) -> None:
        self._token = (
            os.environ.get("GITHUB_TOKEN")
            or os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def advise(
        self,
        repo:             str,
        owner:            str,
        environment:      str,
        failed_deploy_id: str = "",
        failed_version:   str = "",
        failed_sha:       str = "",
    ) -> Optional[RollbackAdvice]:
        """
        Generates a rollback recommendation.

        Args:
            repo:             "{owner}/{repo}"
            owner:            Repository owner
            environment:      Target environment ("production", "staging", etc.)
            failed_deploy_id: The deploy_id that failed (from CDDeployTracker)
            failed_version:   Docker image version that failed (alternative to deploy_id)
            failed_sha:       Commit SHA that failed

        Returns:
            RollbackAdvice or None if no previous successful deploy found
        """
        from ci_cd.cd_deploy_tracker import CDDeployTracker
        tracker = CDDeployTracker()

        # Get the last successful deploy to roll back to
        last_ok = tracker.get_last_successful_deploy(repo, environment)
        if not last_ok:
            logger.warning(
                "[CDRollback] No successful deploy found for %s/%s",
                repo, environment
            )
            return None

        repo_name     = repo.split("/")[-1] if "/" in repo else repo
        target_version = last_ok.get("version", "latest")
        target_sha     = last_ok.get("commit_sha", "")

        # If failed_version not given, fetch from tracker
        if not failed_version and failed_deploy_id:
            failed_rec = tracker.get_deploy(failed_deploy_id)
            failed_version = failed_rec.get("version", "unknown")
            if not failed_sha:
                failed_sha = failed_rec.get("commit_sha", "")

        # Compute diff between failed and last good commit
        diff_files, commits_apart = self._compute_diff(
            owner, repo_name, target_sha, failed_sha
        )

        # Assess rollback risk
        risk_level, risk_reasons = self._assess_risk(
            owner, repo_name, target_sha, failed_sha,
            diff_files, commits_apart
        )

        # Build rollback command
        command = self._build_rollback_command(
            repo_name=repo_name,
            target_version=target_version,
            environment=environment,
        )

        advice = RollbackAdvice(
            repo=repo,
            environment=environment,
            failed_version=failed_version,
            target_version=target_version,
            target_sha=target_sha,
            command=command,
            risk_level=risk_level,
            risk_reasons=risk_reasons,
            diff_files=diff_files,
            commits_apart=commits_apart,
        )
        advice.comment_body = advice.to_pr_comment()

        # Store in Redis for audit
        self._store_advice(advice, failed_deploy_id)

        logger.info(
            "[CDRollback] Advice generated — %s/%s risk=%s target=%s",
            repo, environment, risk_level, target_version
        )
        return advice

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _gh_get(self, url: str) -> Any:
        if not self._token:
            return {}
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"token {self._token}")
        req.add_header("Accept", "application/vnd.github.v3+json")
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode())

    def _compute_diff(
        self,
        owner: str,
        repo: str,
        base_sha: str,
        head_sha: str,
    ) -> tuple[int, int]:
        """
        Returns (files_changed, commits_apart) between two SHAs.
        """
        if not base_sha or not head_sha or base_sha == head_sha:
            return 0, 0
        try:
            data = self._gh_get(
                f"https://api.github.com/repos/{owner}/{repo}"
                f"/compare/{base_sha}...{head_sha}"
            )
            return len(data.get("files", [])), data.get("ahead_by", 0)
        except Exception as e:
            logger.debug("[CDRollback] diff compute error: %s", e)
            return 0, 0

    def _assess_risk(
        self,
        owner:          str,
        repo:           str,
        target_sha:     str,
        failed_sha:     str,
        diff_files:     int,
        commits_apart:  int,
    ) -> tuple[str, List[str]]:
        """
        Assesses rollback risk as LOW / MEDIUM / HIGH.

        Risk criteria:
          HIGH   → migration files in diff, or > 50 files changed
          MEDIUM → > 15 files changed, or > 5 commits apart
          LOW    → small, clean diff
        """
        reasons: List[str] = []

        # Check for migration files in the diff
        has_migrations = False
        if target_sha and failed_sha:
            try:
                data = self._gh_get(
                    f"https://api.github.com/repos/{owner}/{repo}"
                    f"/compare/{target_sha}...{failed_sha}"
                )
                from ci_cd.cd_release_scorer import MIGRATION_PATTERNS
                for f in data.get("files", []):
                    fname = f.get("filename", "").lower()
                    if any(p in fname for p in MIGRATION_PATTERNS):
                        has_migrations = True
                        reasons.append(
                            f"Migration/infra file in rollback diff: {f['filename']}"
                        )
                        break
            except Exception:
                pass

        if has_migrations or diff_files > 50:
            if has_migrations:
                reasons.append("Database migrations may not be reversible")
            if diff_files > 50:
                reasons.append(f"Large rollback scope: {diff_files} files")
            return "HIGH", reasons

        if diff_files > 15 or commits_apart > 5:
            if diff_files > 15:
                reasons.append(f"{diff_files} files would be reverted")
            if commits_apart > 5:
                reasons.append(f"{commits_apart} commits would be reverted")
            return "MEDIUM", reasons

        return "LOW", reasons

    def _build_rollback_command(
        self,
        repo_name:    str,
        target_version: str,
        environment:  str,
    ) -> str:
        """
        Builds the exact shell command to execute the rollback.
        Assumes docker compose deployment.
        """
        # Clean repo name for docker image tag (same logic as workflow_generator)
        image_name = repo_name.rstrip("-").lower()

        lines = [
            f"# Rollback {repo_name} ({environment}) to {target_version}",
            f"# Run on your deploy server in the app directory",
            f"",
            f"# Option 1 — Docker Compose (recommended)",
            f"docker pull ${{DOCKERHUB_USERNAME:-your-login}}/{image_name}:{target_version}",
            f"IMAGE=${{DOCKERHUB_USERNAME:-your-login}}/{image_name}:{target_version} \\",
            f"  docker compose up -d --remove-orphans",
            f"docker image prune -f",
            f"",
            f"# Option 2 — If using a direct docker run",
            f"docker stop {image_name} 2>/dev/null || true",
            f"docker run -d --name {image_name} \\",
            f"  ${{DOCKERHUB_USERNAME:-your-login}}/{image_name}:{target_version}",
        ]
        return "\n".join(lines)

    def _store_advice(
        self,
        advice: RollbackAdvice,
        failed_deploy_id: str,
    ) -> None:
        """Stores rollback advice in Redis for audit trail."""
        try:
            from services.mcp_redis_service import get_mcp_redis
            redis = get_mcp_redis()
            key = f"cd:rollback_advice:{advice.repo}:{advice.environment}"
            redis.hset_dict(key, {
                "generated_at":   str(int(time.time())),
                "failed_deploy":  failed_deploy_id,
                "failed_version": advice.failed_version,
                "target_version": advice.target_version,
                "target_sha":     advice.target_sha,
                "risk_level":     advice.risk_level,
                "diff_files":     str(advice.diff_files),
                "commits_apart":  str(advice.commits_apart),
            }, expire_seconds=86400 * 30)
        except Exception as e:
            logger.debug("[CDRollback] store_advice Redis error: %s", e)
