"""
cd_release_scorer.py — Pre-deploy Release Readiness Score.

Aggregates signals from CI, SonarCloud, GitHub PR, and file-change risk
into a single weighted score with a DEPLOY_OK / DEPLOY_WARN / DEPLOY_BLOCKED verdict.

Usage:
    scorer = CDReleaseScorer()
    report = scorer.score(
        repo="owner/repo",
        owner="owner",
        commit_sha="abc123",
        run_id="123456789",
        project_key="owner_repo",
        pr_number=42,
    )
    print(report.verdict)    # "DEPLOY_OK" | "DEPLOY_WARN" | "DEPLOY_BLOCKED"
    print(report.score)      # 0.0 – 100.0
"""
from __future__ import annotations

import logging
import os
import json
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Score weights (total = 100 points) ────────────────────────────────────────

WEIGHTS = {
    "ci_build":        25,   # build-test passed
    "sonar_gate":      20,   # SonarCloud Quality Gate OK
    "security_scans":  20,   # CodeQL + Trivy clean
    "pr_approval":     15,   # At least 1 PR approval
    "test_coverage":   10,   # Coverage >= threshold
    "changed_files":    5,   # Low risk file changes
    "migration_risk":   5,   # No DB migrations / config changes
}

THRESHOLDS = {
    "deploy_ok":      85,    # score >= 85 → DEPLOY_OK
    "deploy_warn":    60,    # score >= 60 → DEPLOY_WARN
    # below 60        → DEPLOY_BLOCKED
}

MIGRATION_PATTERNS = (
    ".sql", ".migration", "migrate", "alembic", "flyway",
    "schema", "changelog", "liquibase", ".env.prod",
    "docker-compose.prod", "k8s/", "kubernetes/", "helm/",
)


@dataclass
class ReleaseReadinessReport:
    """Full release readiness assessment."""
    repo:          str
    commit_sha:    str
    score:         float                 # 0–100
    verdict:       str                   # "DEPLOY_OK" | "DEPLOY_WARN" | "DEPLOY_BLOCKED"
    blocking_reasons: List[str]          = field(default_factory=list)
    warnings:         List[str]          = field(default_factory=list)
    component_scores: Dict[str, float]   = field(default_factory=dict)
    details:          Dict[str, Any]     = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Formats as a GitHub PR comment section."""
        icon = {
            "DEPLOY_OK":      "✅",
            "DEPLOY_WARN":    "⚠️",
            "DEPLOY_BLOCKED": "🚫",
        }.get(self.verdict, "❓")

        lines = [
            f"## {icon} Release Readiness — `{self.verdict}`",
            f"",
            f"**Score:** `{self.score:.0f}/100`  |  **Commit:** `{self.commit_sha[:8]}`",
            f"",
            f"| Component | Score | Weight |",
            f"|---|---|---|",
        ]
        for name, w in WEIGHTS.items():
            s = self.component_scores.get(name, 0)
            bar = "🟢" if s >= w * 0.8 else ("🟡" if s >= w * 0.4 else "🔴")
            lines.append(f"| {name.replace('_', ' ').title()} | {bar} {s:.1f}/{w} | {w}% |")

        if self.blocking_reasons:
            lines += ["", "**🚫 Blocking reasons:**"]
            for r in self.blocking_reasons:
                lines.append(f"- {r}")

        if self.warnings:
            lines += ["", "**⚠️ Warnings:**"]
            for w in self.warnings:
                lines.append(f"- {w}")

        return "\n".join(lines)


class CDReleaseScorer:
    """
    Computes a release readiness score by aggregating:
      1. CI build status (from GitHub API)
      2. SonarCloud Quality Gate
      3. Security scans (CodeQL + Trivy)
      4. PR approval count
      5. Test coverage
      6. Changed file count and risk
      7. Migration file detection
    """

    def __init__(self) -> None:
        self._token = (
            os.environ.get("GITHUB_TOKEN")
            or os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def score(
        self,
        repo:        str,
        owner:       str,
        commit_sha:  str,
        run_id:      str       = "",
        project_key: str       = "",
        pr_number:   Optional[int] = None,
        environment: str       = "production",
    ) -> ReleaseReadinessReport:
        """
        Computes the full release readiness score.

        Args:
            repo:        "{owner}/{repo}"  (also used as repo_name if needed)
            owner:       Repository owner
            commit_sha:  HEAD commit SHA to assess
            run_id:      GitHub Actions run ID (for artifact fetch)
            project_key: SonarCloud project key
            pr_number:   Associated PR (for approval check)
            environment: Target environment label

        Returns:
            ReleaseReadinessReport
        """
        repo_name = repo.split("/")[-1] if "/" in repo else repo
        components: Dict[str, float] = {}
        blocking:   List[str]        = []
        warnings:   List[str]        = []
        details:    Dict[str, Any]   = {}

        # 1 ── CI Build (25 pts) ──────────────────────────────────────────────
        ci_score, ci_detail = self._score_ci_build(
            owner, repo_name, commit_sha, run_id
        )
        components["ci_build"] = ci_score
        details["ci"] = ci_detail
        if ci_score < WEIGHTS["ci_build"] * 0.5:
            blocking.append(
                f"CI build failed or not found "
                f"(score {ci_score:.0f}/{WEIGHTS['ci_build']})"
            )

        # 2 ── SonarCloud Quality Gate (20 pts) ───────────────────────────────
        sonar_score, sonar_detail = self._score_sonar_gate(project_key)
        components["sonar_gate"] = sonar_score
        details["sonar"] = sonar_detail
        if sonar_score < WEIGHTS["sonar_gate"] * 0.5:
            blocking.append(
                f"SonarCloud Quality Gate FAILED "
                f"(score {sonar_score:.0f}/{WEIGHTS['sonar_gate']})"
            )
        elif sonar_score < WEIGHTS["sonar_gate"] * 0.8:
            warnings.append("SonarCloud Quality Gate is in WARNING state")

        # 3 ── Security Scans (20 pts) ─────────────────────────────────────────
        sec_score, sec_detail = self._score_security(
            owner, repo_name, commit_sha, run_id
        )
        components["security_scans"] = sec_score
        details["security"] = sec_detail
        if sec_score < WEIGHTS["security_scans"] * 0.3:
            blocking.append(
                f"Critical security vulnerabilities detected "
                f"(score {sec_score:.0f}/{WEIGHTS['security_scans']})"
            )
        elif sec_score < WEIGHTS["security_scans"] * 0.7:
            warnings.append(
                f"High severity security issues found "
                f"(score {sec_score:.0f}/{WEIGHTS['security_scans']})"
            )

        # 4 ── PR Approval (15 pts) ────────────────────────────────────────────
        if pr_number:
            pr_score, pr_detail = self._score_pr_approval(
                owner, repo_name, pr_number
            )
            components["pr_approval"] = pr_score
            details["pr"] = pr_detail
            if pr_score < WEIGHTS["pr_approval"] * 0.5:
                warnings.append(
                    f"PR #{pr_number} has no approvals yet "
                    f"(score {pr_score:.0f}/{WEIGHTS['pr_approval']})"
                )
        else:
            # No PR = direct push to main; give partial credit
            components["pr_approval"] = WEIGHTS["pr_approval"] * 0.6
            details["pr"] = {"note": "direct push, no PR found"}
            warnings.append("No PR found — direct push to main branch")

        # 5 ── Test Coverage (10 pts) ──────────────────────────────────────────
        cov_score, cov_detail = self._score_coverage(sonar_detail)
        components["test_coverage"] = cov_score
        details["coverage"] = cov_detail
        if cov_score < WEIGHTS["test_coverage"] * 0.5:
            warnings.append(
                f"Test coverage is low: {cov_detail.get('coverage', 'N/A')}%"
            )

        # 6 ── Changed Files Risk (5 pts) ──────────────────────────────────────
        chg_score, chg_detail = self._score_changed_files(
            owner, repo_name, commit_sha
        )
        components["changed_files"] = chg_score
        details["changed_files"] = chg_detail
        if chg_score < WEIGHTS["changed_files"] * 0.5:
            warnings.append(
                f"Large changeset: {chg_detail.get('count', '?')} files changed"
            )

        # 7 ── Migration Risk (5 pts) ──────────────────────────────────────────
        mig_score, mig_detail = self._score_migration_risk(
            owner, repo_name, commit_sha
        )
        components["migration_risk"] = mig_score
        details["migrations"] = mig_detail
        if mig_score < WEIGHTS["migration_risk"] * 0.5:
            blocking.append(
                f"Migration files detected — manual review required: "
                f"{', '.join(mig_detail.get('files', [])[:3])}"
            )

        # ── Compute total score ────────────────────────────────────────────────
        total = sum(components.values())

        if blocking:
            verdict = "DEPLOY_BLOCKED"
        elif total >= THRESHOLDS["deploy_ok"]:
            verdict = "DEPLOY_OK"
        elif total >= THRESHOLDS["deploy_warn"]:
            verdict = "DEPLOY_WARN"
        else:
            verdict = "DEPLOY_BLOCKED"

        logger.info(
            "[CDScorer] %s@%s → %s (%.0f/100) blocks=%d warns=%d",
            repo, commit_sha[:8], verdict, total,
            len(blocking), len(warnings)
        )

        return ReleaseReadinessReport(
            repo=repo,
            commit_sha=commit_sha,
            score=round(total, 1),
            verdict=verdict,
            blocking_reasons=blocking,
            warnings=warnings,
            component_scores=components,
            details=details,
        )

    # ── Component scorers ──────────────────────────────────────────────────────

    def _gh_get(self, url: str) -> Any:
        if not self._token:
            raise RuntimeError("GITHUB_TOKEN manquant")
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"token {self._token}")
        req.add_header("Accept", "application/vnd.github.v3+json")
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode())

    def _score_ci_build(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
        run_id: str,
    ) -> tuple[float, Dict[str, Any]]:
        """Score based on GitHub commit status checks."""
        max_pts = WEIGHTS["ci_build"]
        try:
            data = self._gh_get(
                f"https://api.github.com/repos/{owner}/{repo}"
                f"/commits/{commit_sha}/check-runs"
            )
            runs = data.get("check_runs", [])
            if not runs:
                return max_pts * 0.5, {"note": "no check runs found", "count": 0}

            passed  = sum(1 for r in runs if r.get("conclusion") == "success")
            failed  = sum(1 for r in runs if r.get("conclusion") == "failure")
            pending = sum(1 for r in runs if r.get("status") != "completed")
            total   = len(runs)

            if failed > 0:
                score = max_pts * max(0, (passed - failed) / total)
            elif pending > 0:
                score = max_pts * 0.6
            else:
                score = max_pts * (passed / total) if total else max_pts * 0.5

            return round(score, 2), {
                "total": total, "passed": passed,
                "failed": failed, "pending": pending
            }
        except Exception as e:
            logger.debug("[CDScorer] CI build score error: %s", e)
            return max_pts * 0.5, {"error": str(e)}

    def _score_sonar_gate(
        self,
        project_key: str,
    ) -> tuple[float, Dict[str, Any]]:
        """Score based on SonarCloud Quality Gate status."""
        max_pts = WEIGHTS["sonar_gate"]
        if not project_key:
            return max_pts * 0.7, {"note": "no project_key configured"}
        try:
            from services.mcp_sonarqube_service import get_mcp_sonarqube
            sonar = get_mcp_sonarqube()
            gate    = sonar.get_quality_gate_status(project_key)
            metrics = sonar.get_project_metrics(project_key)
            status  = gate.get("status", "NONE")
            score   = {
                "OK":   max_pts,
                "WARN": max_pts * 0.6,
                "ERROR": 0.0,
                "NONE": max_pts * 0.5,
            }.get(status, max_pts * 0.5)
            return round(score, 2), {
                "status":   status,
                "coverage": metrics.get("coverage", "N/A"),
                "bugs":     metrics.get("bugs", 0),
                "vulns":    metrics.get("vulnerabilities", 0),
            }
        except Exception as e:
            logger.debug("[CDScorer] Sonar score error: %s", e)
            return max_pts * 0.6, {"error": str(e)}

    def _score_security(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
        run_id: str,
    ) -> tuple[float, Dict[str, Any]]:
        """Score based on CodeQL alerts and Trivy CVEs."""
        max_pts = WEIGHTS["security_scans"]
        codeql_critical = 0
        codeql_high     = 0
        trivy_critical  = 0
        trivy_high      = 0

        try:
            from langchain_agents.tools.ci_tools import (
                tool_fetch_code_scanning_alerts,
                tool_fetch_trivy_artifact,
            )
            alerts = tool_fetch_code_scanning_alerts.invoke({
                "owner": owner, "repo": repo, "ref": commit_sha,
                "tool_name": "CodeQL", "severity_filter": "critical,high,error",
            })
            codeql_critical = sum(1 for a in alerts if a.get("severity") in ("critical", "error"))
            codeql_high     = sum(1 for a in alerts if a.get("severity") == "high")

            if run_id:
                trivy = tool_fetch_trivy_artifact.invoke({
                    "owner": owner, "repo": repo, "run_id": run_id
                })
                trivy_critical = len(trivy.get("critical", []))
                trivy_high     = len(trivy.get("high", []))
        except Exception as e:
            logger.debug("[CDScorer] Security score fetch error: %s", e)

        # Penalty: critical = -8pts each, high = -3pts each
        penalty = (codeql_critical + trivy_critical) * 8 + \
                  (codeql_high     + trivy_high)     * 3
        score = max(0.0, float(max_pts) - penalty)

        return round(score, 2), {
            "codeql_critical": codeql_critical,
            "codeql_high":     codeql_high,
            "trivy_critical":  trivy_critical,
            "trivy_high":      trivy_high,
        }

    def _score_pr_approval(
        self,
        owner: str,
        repo: str,
        pr_number: int,
    ) -> tuple[float, Dict[str, Any]]:
        """Score based on PR review approvals."""
        max_pts = WEIGHTS["pr_approval"]
        try:
            data = self._gh_get(
                f"https://api.github.com/repos/{owner}/{repo}"
                f"/pulls/{pr_number}/reviews"
            )
            approvals = [r for r in data if r.get("state") == "APPROVED"]
            changes   = [r for r in data if r.get("state") == "CHANGES_REQUESTED"]

            if changes:
                score = 0.0   # Changes requested → blocked
            elif len(approvals) >= 2:
                score = max_pts        # 2+ approvals
            elif len(approvals) == 1:
                score = max_pts * 0.7  # 1 approval
            else:
                score = max_pts * 0.2  # No approval

            return round(score, 2), {
                "approvals": len(approvals),
                "changes_requested": len(changes),
            }
        except Exception as e:
            logger.debug("[CDScorer] PR approval score error: %s", e)
            return max_pts * 0.5, {"error": str(e)}

    def _score_coverage(
        self,
        sonar_detail: Dict[str, Any],
    ) -> tuple[float, Dict[str, Any]]:
        """Score based on test coverage from SonarCloud metrics."""
        max_pts = WEIGHTS["test_coverage"]
        try:
            cov_raw = sonar_detail.get("coverage", "0")
            cov     = float(str(cov_raw).replace("%", "")) if cov_raw != "N/A" else 0.0
            if cov >= 80:
                score = max_pts
            elif cov >= 70:
                score = max_pts * 0.8
            elif cov >= 60:
                score = max_pts * 0.5
            elif cov >= 50:
                score = max_pts * 0.3
            else:
                score = 0.0
            return round(score, 2), {"coverage": cov}
        except Exception:
            return max_pts * 0.5, {"coverage": "N/A"}

    def _score_changed_files(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
    ) -> tuple[float, Dict[str, Any]]:
        """Score based on number of changed files (large PRs = higher risk)."""
        max_pts = WEIGHTS["changed_files"]
        try:
            data  = self._gh_get(
                f"https://api.github.com/repos/{owner}/{repo}"
                f"/commits/{commit_sha}"
            )
            count = len(data.get("files", []))
            if count <= 10:
                score = max_pts
            elif count <= 30:
                score = max_pts * 0.7
            elif count <= 60:
                score = max_pts * 0.4
            else:
                score = max_pts * 0.1
            return round(score, 2), {"count": count}
        except Exception as e:
            logger.debug("[CDScorer] Changed files score error: %s", e)
            return max_pts * 0.7, {"error": str(e)}

    def _score_migration_risk(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
    ) -> tuple[float, Dict[str, Any]]:
        """Score based on detection of migration / infra files in the changeset."""
        max_pts = WEIGHTS["migration_risk"]
        try:
            data  = self._gh_get(
                f"https://api.github.com/repos/{owner}/{repo}"
                f"/commits/{commit_sha}"
            )
            files  = [f["filename"] for f in data.get("files", [])]
            risky  = [
                f for f in files
                if any(p in f.lower() for p in MIGRATION_PATTERNS)
            ]
            if not risky:
                return float(max_pts), {"files": [], "risk": "none"}
            elif len(risky) <= 2:
                return max_pts * 0.3, {"files": risky, "risk": "medium"}
            else:
                return 0.0, {"files": risky, "risk": "high"}
        except Exception as e:
            logger.debug("[CDScorer] Migration risk score error: %s", e)
            return max_pts * 0.8, {"error": str(e)}
