"""
lc_cd_notifier.py — CD Notification Agent.

Mirrors the structure of lc_ci_notifier.py for the CD pipeline.
Renders rich terminal output for:
  - Pre-deploy readiness reports (DEPLOY_OK / DEPLOY_WARN / DEPLOY_BLOCKED)
  - Post-deploy health outcomes (HEALTHY / DEGRADED / DOWN)
  - Rollback advice summaries

4 Pillars:
  LLM     : None (0 tokens)
  Tools   : None
  Memory  : None
  Planning: Grade-based routing (BLOCKED → WARN → OK)
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ANSI
_G = "\033[92m"   # Green
_R = "\033[91m"   # Red
_Y = "\033[93m"   # Yellow
_C = "\033[96m"   # Cyan
_M = "\033[95m"   # Magenta
_B = "\033[1m"    # Bold
_E = "\033[0m"    # Reset
_W = 72            # Panel width


class LCCDNotifier:
    """
    Rich terminal notifier for the CDGraph.

    Usage:
        notifier = LCCDNotifier()
        notifier.notify(cd_state)        # auto-routes by verdict/grade
        notifier.notify_readiness(state) # pre-deploy readiness only
        notifier.notify_health(state)    # post-deploy health only
    """

    def __init__(self, print_lock: Optional[threading.Lock] = None) -> None:
        self._lock = print_lock or threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def notify(self, state: Dict[str, Any]) -> str:
        """
        Auto-routes notification based on CDState content.

        Returns:
            Notification level: "BLOCKED" | "WARN" | "OK"
        """
        level = self._determine_level(state)
        with self._lock:
            if level == "BLOCKED":
                self._print_blocked(state)
            elif level == "WARN":
                self._print_warn(state)
            else:
                self._print_ok(state)
        return level

    def notify_readiness(self, state: Dict[str, Any]) -> None:
        """Prints only the pre-deploy readiness section."""
        with self._lock:
            verdict = state.get("readiness_verdict", "DEPLOY_WARN")
            score   = state.get("readiness_score", 0.0)
            self._print_readiness_block(verdict, score, state)

    def notify_health(self, state: Dict[str, Any]) -> None:
        """Prints only the post-deploy health section."""
        with self._lock:
            self._print_health_block(state)

    # ── Level determination ────────────────────────────────────────────────────

    @staticmethod
    def _determine_level(state: Dict[str, Any]) -> str:
        verdict = state.get("readiness_verdict", "")
        health  = state.get("monitor_grade", "HEALTHY")
        deploy  = state.get("deploy_status", "")

        if verdict == "DEPLOY_BLOCKED" or deploy == "failed":
            return "BLOCKED"
        if verdict == "DEPLOY_WARN" or health in ("DEGRADED", "DOWN"):
            return "WARN"
        return "OK"

    # ── Header helper ──────────────────────────────────────────────────────────

    def _header(self, icon: str, label: str, color: str,
                 repo: str, environment: str) -> None:
        print(f"\n{color}{_B}{'═' * _W}{_E}")
        print(f"{color}{_B}  {icon}  CD INTELLIGENCE — {label}{_E}")
        print(f"{color}{_B}{'─' * _W}{_E}")
        print(f"{color}  Repo        : {repo}{_E}")
        print(f"{color}  Environment : {environment}{_E}")

    # ── Print methods ──────────────────────────────────────────────────────────

    def _print_blocked(self, state: Dict[str, Any]) -> None:
        repo    = state.get("repo", "unknown")
        env     = state.get("environment", "production")
        verdict = state.get("readiness_verdict", "DEPLOY_BLOCKED")
        score   = state.get("readiness_score", 0.0)
        reasons = state.get("readiness_blocking_reasons", [])
        deploy  = state.get("deploy_status", "")
        rollback = state.get("rollback_command", "")

        self._header("🚫", "DEPLOY BLOCKED", _R, repo, env)

        if deploy == "failed":
            print(f"\n{_R}{_B}  DEPLOYMENT FAILED{_E}")
            reason = state.get("deploy_failure_reason", "")
            if reason:
                print(f"{_R}  Reason : {reason[:120]}{_E}")
        else:
            self._print_readiness_block(verdict, score, state)

        if reasons:
            print(f"\n{_R}{_B}  Blocking reasons:{_E}")
            for r in reasons:
                print(f"{_R}    ✗ {r}{_E}")

        # Rollback advice
        if rollback:
            print(f"\n{_Y}{_B}  Rollback Command:{_E}")
            for line in rollback.strip().splitlines()[:6]:
                print(f"  {_Y}  {line}{_E}")
            risk = state.get("rollback_risk", "")
            if risk:
                risk_c = _R if risk == "HIGH" else (_Y if risk == "MEDIUM" else _G)
                print(f"\n{risk_c}  Rollback risk: {risk}{_E}")

        self._print_health_block(state)
        print(f"{_R}{_B}{'═' * _W}{_E}\n")

    def _print_warn(self, state: Dict[str, Any]) -> None:
        repo  = state.get("repo", "unknown")
        env   = state.get("environment", "production")
        score = state.get("readiness_score", 0.0)
        warns = state.get("readiness_warnings", [])

        self._header("⚠️ ", "DEPLOY WARNING", _Y, repo, env)

        if score:
            print(f"\n{_Y}  Readiness score : {score:.0f}/100{_E}")

        if warns:
            print(f"\n{_Y}{_B}  Warnings:{_E}")
            for w in warns:
                print(f"{_Y}    ⚠ {w}{_E}")

        self._print_health_block(state)
        print(f"{_Y}{_B}{'═' * _W}{_E}\n")

    def _print_ok(self, state: Dict[str, Any]) -> None:
        repo    = state.get("repo", "unknown")
        env     = state.get("environment", "production")
        score   = state.get("readiness_score", 0.0)
        version = state.get("version", "")
        sha     = state.get("commit_sha", "")[:8]

        self._header("✅", "DEPLOY SUCCESS", _G, repo, env)

        print(f"\n{_G}  Version     : {version or sha}{_E}")
        if score:
            print(f"{_G}  Readiness   : {score:.0f}/100{_E}")

        self._print_health_block(state)

        if state.get("comment_posted"):
            pr = state.get("pr_number")
            print(f"\n{_G}  ✓ Report posted to PR #{pr}{_E}")

        print(f"{_G}{_B}{'═' * _W}{_E}\n")

    # ── Shared blocks ──────────────────────────────────────────────────────────

    def _print_readiness_block(
        self,
        verdict: str,
        score:   float,
        state:   Dict[str, Any],
    ) -> None:
        icon_map = {
            "DEPLOY_OK":      f"{_G}✅ DEPLOY_OK{_E}",
            "DEPLOY_WARN":    f"{_Y}⚠️  DEPLOY_WARN{_E}",
            "DEPLOY_BLOCKED": f"{_R}🚫 DEPLOY_BLOCKED{_E}",
        }
        components = state.get("readiness_components", {})
        print(f"\n{_B}  Release Readiness:{_E}")
        print(f"    Verdict : {icon_map.get(verdict, verdict)}")
        print(f"    Score   : {score:.0f}/100")
        if components:
            for name, s in components.items():
                bar = "🟢" if s >= 15 else ("🟡" if s >= 7 else "🔴")
                label = name.replace("_", " ").title()
                print(f"    {bar}  {label:<20} {s:.1f}")

    def _print_health_block(self, state: Dict[str, Any]) -> None:
        grade       = state.get("monitor_grade", "")
        avail       = state.get("monitor_availability", "")
        avg_lat     = state.get("monitor_avg_latency_ms", "")
        regression  = state.get("monitor_regression", False)
        issues      = state.get("monitor_issues", [])

        if not grade:
            return

        grade_icon = {"HEALTHY": f"{_G}✅ HEALTHY{_E}",
                      "DEGRADED": f"{_Y}⚠️  DEGRADED{_E}",
                      "DOWN":     f"{_R}🔴 DOWN{_E}"}.get(grade, grade)

        print(f"\n{_B}  Post-Deploy Health:{_E}")
        print(f"    Grade        : {grade_icon}")
        if avail:
            print(f"    Availability : {avail}%")
        if avg_lat:
            print(f"    Avg latency  : {avg_lat}ms")
        if regression:
            print(f"    {_R}⚠ Regression detected after baseline{_E}")
        if issues:
            print(f"    Issues:")
            for i in issues[:4]:
                print(f"      • {i}")
