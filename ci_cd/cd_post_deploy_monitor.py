"""
cd_post_deploy_monitor.py — Async post-deploy monitoring window (Priority 4).

Runs a 5–15 minute monitoring window after a deployment:
  - Polls the health endpoint every N seconds
  - Detects regressions (DOWN or DEGRADED after an OK baseline)
  - Stores the MonitorSession in Redis
  - Fires a notification callback on regression

Design: asyncio-native — runs as a background task alongside the CIPoller.

Usage (in CDGraph / caller):
    monitor = CDPostDeployMonitor(
        deploy_url="https://myapp.com",
        repo="owner/repo",
        deploy_id="deploy-123",
        environment="production",
        on_regression=my_callback,   # optional async callable
    )
    # Fire-and-forget background task:
    asyncio.create_task(monitor.run())

    # Or blocking (for CDGraph node):
    session = await monitor.run()
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from ci_cd.cd_health_checker import CDHealthChecker, HealthReport, MonitorSession

logger = logging.getLogger(__name__)

# Default monitoring parameters
DEFAULT_DURATION_S  = 600     # 10-minute window
DEFAULT_POLL_EVERY  = 60      # poll every 60 seconds
MIN_DURATION_S      = 300     # 5 minutes minimum
MAX_DURATION_S      = 900     # 15 minutes maximum

# Regression triggers
REGRESSION_DOWN_THRESHOLD      = 2   # 2+ consecutive DOWN polls = regression
REGRESSION_DEGRADED_THRESHOLD  = 3   # 3+ consecutive DEGRADED polls = regression


class CDPostDeployMonitor:
    """
    Async post-deploy monitor.

    Polls `deploy_url` for `duration_s` seconds at `poll_every_s` intervals.
    On regression, calls `on_regression(session)` if provided.
    Stores the MonitorSession result in Redis under:
      cd:monitor:{deploy_id}
    """

    def __init__(
        self,
        deploy_url:    str,
        repo:          str,
        deploy_id:     str,
        environment:   str      = "production",
        duration_s:    int      = DEFAULT_DURATION_S,
        poll_every_s:  int      = DEFAULT_POLL_EVERY,
        on_regression: Optional[Callable[["MonitorSession"], None]] = None,
    ) -> None:
        self._url          = deploy_url
        self._repo         = repo
        self._deploy_id    = deploy_id
        self._environment  = environment
        self._duration     = max(MIN_DURATION_S, min(duration_s, MAX_DURATION_S))
        self._poll_every   = max(15, poll_every_s)
        self._on_regression = on_regression
        self._checker      = CDHealthChecker(deploy_url)

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run(self) -> MonitorSession:
        """
        Runs the monitoring window asynchronously.

        Returns:
            MonitorSession with aggregated health data.
        """
        logger.info(
            "[CDMonitor] Starting %ds window for %s (%s)",
            self._duration, self._repo, self._environment
        )

        snapshots:          List[HealthReport] = []
        issues_all:         List[str]          = []
        consecutive_down    = 0
        consecutive_degraded = 0
        regression_detected = False
        baseline_ok         = False    # True once we see first OK
        start               = time.monotonic()
        deadline            = start + self._duration

        while time.monotonic() < deadline:
            # Run the health check in a thread (blocking I/O)
            loop    = asyncio.get_event_loop()
            report  = await loop.run_in_executor(None, self._checker.check_with_fallback)
            snapshots.append(report)

            logger.debug("[CDMonitor] %s", report.summary)

            # Track baseline
            if report.healthy and not baseline_ok:
                baseline_ok = True
                logger.info("[CDMonitor] Baseline established — service is UP")

            # Count consecutive failures
            if report.grade == "DOWN":
                consecutive_down    += 1
                consecutive_degraded = 0
            elif report.grade == "DEGRADED":
                consecutive_degraded += 1
                consecutive_down      = 0
            else:
                consecutive_down      = 0
                consecutive_degraded  = 0

            # Collect issues
            for issue in report.issues:
                if issue not in issues_all:
                    issues_all.append(issue)

            # Regression detection (only after baseline is established)
            if baseline_ok and not regression_detected:
                if consecutive_down >= REGRESSION_DOWN_THRESHOLD:
                    regression_detected = True
                    logger.warning(
                        "[CDMonitor] REGRESSION: %d consecutive DOWN polls on %s",
                        consecutive_down, self._repo
                    )
                elif consecutive_degraded >= REGRESSION_DEGRADED_THRESHOLD:
                    regression_detected = True
                    logger.warning(
                        "[CDMonitor] REGRESSION: %d consecutive DEGRADED polls on %s",
                        consecutive_degraded, self._repo
                    )

            # Sleep until next poll (or until deadline)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(self._poll_every, remaining))

        # ── Aggregate results ─────────────────────────────────────────────────
        session = self._aggregate(snapshots, issues_all, regression_detected)

        # Store in Redis
        self._store(session)

        # Fire regression callback
        if regression_detected and self._on_regression:
            try:
                if asyncio.iscoroutinefunction(self._on_regression):
                    await self._on_regression(session)
                else:
                    self._on_regression(session)
            except Exception as e:
                logger.error("[CDMonitor] on_regression callback failed: %s", e)

        logger.info(
            "[CDMonitor] Complete — %s final_grade=%s availability=%.1f%%",
            self._repo, session.final_grade, session.availability_pct
        )
        return session

    # ── Aggregation ────────────────────────────────────────────────────────────

    def _aggregate(
        self,
        snapshots:   List[HealthReport],
        issues:      List[str],
        regression:  bool,
    ) -> MonitorSession:
        if not snapshots:
            return MonitorSession(
                deploy_url=self._url, environment=self._environment,
                duration_s=self._duration, poll_count=0,
                healthy_polls=0, degraded_polls=0, down_polls=0,
                avg_latency_ms=0.0, max_latency_ms=0.0,
                final_grade="DOWN", regression=False,
                issues_detected=["No polls completed"],
            )

        healthy  = sum(1 for s in snapshots if s.grade == "OK")
        degraded = sum(1 for s in snapshots if s.grade == "DEGRADED")
        down     = sum(1 for s in snapshots if s.grade == "DOWN")
        total    = len(snapshots)

        latencies    = [s.latency_ms for s in snapshots if s.latency_ms > 0]
        avg_latency  = sum(latencies) / len(latencies) if latencies else 0.0
        max_latency  = max(latencies) if latencies else 0.0

        # Final grade
        availability = healthy / total if total else 0.0
        if availability >= 0.95 and avg_latency <= LATENCY_WARN_MS:
            final_grade = "HEALTHY"
        elif availability >= 0.7 or (down == 0 and degraded <= total * 0.3):
            final_grade = "DEGRADED"
        else:
            final_grade = "DOWN"

        if regression:
            final_grade = min(final_grade, "DEGRADED")   # At least DEGRADED on regression

        elapsed = int(
            (snapshots[-1].timestamp - snapshots[0].timestamp)
            if len(snapshots) > 1 else self._duration
        )

        return MonitorSession(
            deploy_url=self._url,
            environment=self._environment,
            duration_s=elapsed or self._duration,
            poll_count=total,
            healthy_polls=healthy,
            degraded_polls=degraded,
            down_polls=down,
            avg_latency_ms=round(avg_latency, 1),
            max_latency_ms=round(max_latency, 1),
            final_grade=final_grade,
            regression=regression,
            issues_detected=issues[:10],
            snapshots=snapshots,
        )

    def _store(self, session: MonitorSession) -> None:
        """Persists the monitor session in Redis."""
        try:
            from services.mcp_redis_service import get_mcp_redis
            redis = get_mcp_redis()
            redis.hset_dict(
                f"cd:monitor:{self._deploy_id}",
                {
                    "deploy_id":       self._deploy_id,
                    "repo":            self._repo,
                    "environment":     self._environment,
                    "deploy_url":      self._url,
                    "final_grade":     session.final_grade,
                    "availability":    str(session.availability_pct),
                    "avg_latency_ms":  str(session.avg_latency_ms),
                    "max_latency_ms":  str(session.max_latency_ms),
                    "poll_count":      str(session.poll_count),
                    "healthy_polls":   str(session.healthy_polls),
                    "down_polls":      str(session.down_polls),
                    "regression":      "true" if session.regression else "false",
                    "issues":          " | ".join(session.issues_detected[:5]),
                    "recorded_at":     str(int(time.time())),
                },
                expire_seconds=86400 * 30,
            )
        except Exception as e:
            logger.debug("[CDMonitor] Redis store failed: %s", e)


# ── Standalone synchronous wrapper for use inside CDGraph nodes ───────────────

def run_post_deploy_monitor_sync(
    deploy_url:   str,
    repo:         str,
    deploy_id:    str,
    environment:  str = "production",
    duration_s:   int = DEFAULT_DURATION_S,
    poll_every_s: int = DEFAULT_POLL_EVERY,
) -> MonitorSession:
    """
    Synchronous wrapper around CDPostDeployMonitor.run() for use in
    LangGraph nodes that are not async.

    Runs the monitor in a new event loop if one isn't available.
    """
    monitor = CDPostDeployMonitor(
        deploy_url=deploy_url,
        repo=repo,
        deploy_id=deploy_id,
        environment=environment,
        duration_s=duration_s,
        poll_every_s=poll_every_s,
    )
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an existing loop — schedule as task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(asyncio.run, monitor.run())
                return future.result(timeout=duration_s + 30)
        else:
            return loop.run_until_complete(monitor.run())
    except Exception as e:
        logger.error("[CDMonitor] run_sync failed: %s", e)
        return MonitorSession(
            deploy_url=deploy_url, environment=environment,
            duration_s=duration_s, poll_count=0,
            healthy_polls=0, degraded_polls=0, down_polls=0,
            avg_latency_ms=0.0, max_latency_ms=0.0,
            final_grade="DOWN", regression=False,
            issues_detected=[f"Monitor error: {e}"],
        )


# Re-export for convenience
from ci_cd.cd_health_checker import LATENCY_WARN_MS  # noqa: F401
