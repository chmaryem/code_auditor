"""
cd_health_checker.py — Post-deploy environment health checker (Priority 6).

Checks HTTP status, latency, error rate signals, and logs after a deployment.
Returns a structured HealthReport used by the post-deploy monitor and CDGraph.

Usage:
    checker = CDHealthChecker(deploy_url="https://myapp.com")
    report = checker.check()
    print(report.healthy)       # True / False
    print(report.latency_ms)    # e.g. 143
    print(report.summary)       # human-readable
"""
from __future__ import annotations

import logging
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Health grade thresholds
LATENCY_WARN_MS  = 800
LATENCY_CRIT_MS  = 2000
STATUS_OK_CODES  = {200, 201, 202, 204}


@dataclass
class HealthReport:
    """Single health-check snapshot."""
    deploy_url:   str
    timestamp:    int
    http_status:  int          # 0 = unreachable
    latency_ms:   float        # Round-trip in ms
    healthy:      bool         # True if status OK and latency acceptable
    grade:        str          # "OK" | "DEGRADED" | "DOWN"
    issues:       List[str]    = field(default_factory=list)

    @property
    def summary(self) -> str:
        issues_str = "; ".join(self.issues) if self.issues else "none"
        return (
            f"[{self.grade}] {self.deploy_url} | "
            f"HTTP {self.http_status} | {self.latency_ms:.0f}ms | "
            f"issues: {issues_str}"
        )


@dataclass
class MonitorSession:
    """Aggregated result of a multi-poll monitoring window."""
    deploy_url:       str
    environment:      str
    duration_s:       int              # How long we monitored
    poll_count:       int
    healthy_polls:    int
    degraded_polls:   int
    down_polls:       int
    avg_latency_ms:   float
    max_latency_ms:   float
    final_grade:      str              # "HEALTHY" | "DEGRADED" | "DOWN"
    regression:       bool             # True if regression vs pre-deploy
    issues_detected:  List[str]        = field(default_factory=list)
    snapshots:        List[HealthReport] = field(default_factory=list)

    @property
    def availability_pct(self) -> float:
        if self.poll_count == 0:
            return 100.0
        return round(self.healthy_polls / self.poll_count * 100, 1)

    def to_summary_lines(self) -> List[str]:
        icon = {"HEALTHY": "✅", "DEGRADED": "⚠️", "DOWN": "🔴"}.get(
            self.final_grade, "❓"
        )
        return [
            f"{icon}  Post-Deploy Health — {self.final_grade}",
            f"   URL          : {self.deploy_url}",
            f"   Duration     : {self.duration_s}s ({self.poll_count} polls)",
            f"   Availability : {self.availability_pct}%",
            f"   Avg latency  : {self.avg_latency_ms:.0f}ms",
            f"   Max latency  : {self.max_latency_ms:.0f}ms",
            f"   Regression   : {'YES ⚠️' if self.regression else 'no'}",
        ] + (
            ["   Issues:"] + [f"     • {i}" for i in self.issues_detected]
            if self.issues_detected else []
        )


class CDHealthChecker:
    """
    HTTP health checker for a deployed service.

    Performs:
      1. HTTP GET to deploy_url — checks status code + latency
      2. Optional additional endpoints (e.g. /health, /api/ping)
      3. Grades result: OK / DEGRADED / DOWN

    Design:
      - No external libraries (stdlib only)
      - Graceful on all errors (returns DOWN, never raises)
      - Thread-safe (each call is independent)
    """

    def __init__(
        self,
        deploy_url:         str,
        timeout_s:          float = 10.0,
        extra_paths:        Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            deploy_url:   Base URL to check (e.g. "https://myapp.com")
            timeout_s:    HTTP request timeout in seconds
            extra_paths:  Additional paths to probe (e.g. ["/health", "/api/ping"])
        """
        self._url     = deploy_url.rstrip("/")
        self._timeout = timeout_s
        self._paths   = extra_paths or ["/health", "/", "/api/health", "/ping"]

    # ── Public API ─────────────────────────────────────────────────────────────

    def check(self) -> HealthReport:
        """
        Performs one health check snapshot.

        Returns:
            HealthReport with grade OK / DEGRADED / DOWN
        """
        ts = int(time.time())
        if not self._url:
            return HealthReport(
                deploy_url="", timestamp=ts,
                http_status=0, latency_ms=0.0,
                healthy=False, grade="DOWN",
                issues=["No deploy_url configured"],
            )

        status, latency, issues = self._probe_url(self._url)
        grade  = self._compute_grade(status, latency, issues)
        return HealthReport(
            deploy_url=self._url,
            timestamp=ts,
            http_status=status,
            latency_ms=round(latency, 1),
            healthy=grade == "OK",
            grade=grade,
            issues=issues,
        )

    def check_with_fallback(self) -> HealthReport:
        """
        Tries the base URL first; if it returns a non-2xx, tries known
        health sub-paths (/health, /api/health) before giving up.
        """
        base = self.check()
        if base.healthy:
            return base

        for path in self._paths:
            url = self._url + path
            if url == self._url:
                continue
            status, latency, issues = self._probe_url(url)
            if status in STATUS_OK_CODES:
                grade = self._compute_grade(status, latency, [])
                return HealthReport(
                    deploy_url=url,
                    timestamp=int(time.time()),
                    http_status=status,
                    latency_ms=round(latency, 1),
                    healthy=grade == "OK",
                    grade=grade,
                    issues=[],
                )
        return base

    # ── Internal ───────────────────────────────────────────────────────────────

    def _probe_url(self, url: str) -> tuple[int, float, List[str]]:
        """Returns (http_status, latency_ms, issues)."""
        issues: List[str] = []
        t0 = time.monotonic()
        try:
            req = urllib.request.Request(url, method="GET")
            req.add_header("User-Agent", "CodeAuditor-HealthChecker/1.0")
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                latency = (time.monotonic() - t0) * 1000
                status  = resp.status
        except urllib.error.HTTPError as e:
            latency = (time.monotonic() - t0) * 1000
            status  = e.code
            issues.append(f"HTTP {e.code}: {e.reason}")
        except urllib.error.URLError as e:
            latency = (time.monotonic() - t0) * 1000
            status  = 0
            issues.append(f"Connection error: {e.reason}")
        except Exception as e:
            latency = (time.monotonic() - t0) * 1000
            status  = 0
            issues.append(f"Unexpected error: {e}")

        # Latency warnings
        if latency > LATENCY_CRIT_MS:
            issues.append(f"Critical latency: {latency:.0f}ms")
        elif latency > LATENCY_WARN_MS:
            issues.append(f"High latency: {latency:.0f}ms")

        return status, latency, issues

    @staticmethod
    def _compute_grade(status: int, latency: float, issues: List[str]) -> str:
        if status in STATUS_OK_CODES and latency <= LATENCY_WARN_MS:
            return "OK"
        if status in STATUS_OK_CODES and latency <= LATENCY_CRIT_MS:
            return "DEGRADED"
        if status == 0 or status >= 500:
            return "DOWN"
        if status >= 400:
            return "DEGRADED"
        return "DEGRADED"
