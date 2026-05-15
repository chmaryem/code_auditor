"""
lc_ci_notifier.py — CI/CD Notification Agent (LangChain).

4 Pillars :
  LLM     : None (0 token)
  Tools   : None (console output)
  Memory  : None
  Planning: Niveaux INFO / WARN / URGENT basés sur severity + gate status

Même patron que LCTestProposalNotifier.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ANSI colors
_G = "\033[92m"   # Green
_R = "\033[91m"   # Red
_Y = "\033[93m"   # Yellow
_C = "\033[96m"   # Cyan
_M = "\033[95m"   # Magenta
_B = "\033[1m"    # Bold
_E = "\033[0m"    # Reset


class LCCINotifier:
    """
    Notificateur CI/CD pour le CIGraph.

    Niveaux :
      INFO   — Run réussi, coverage stable, gate OK
      WARN   — Coverage baisse, CVE MEDIUM, run annulé
      URGENT — Stage fail, CVE CRITICAL/BLOCKER, Quality Gate FAILED

    Usage :
        notifier = LCCINotifier(print_lock=lock)
        notifier.notify(ci_state)
    """

    def __init__(self, print_lock: Optional[threading.Lock] = None):
        self._lock = print_lock or threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def notify(self, state: Dict[str, Any]) -> str:
        """
        Affiche la notification CI selon le niveau.

        Args:
            state: CIState dict (ou sous-ensemble)

        Returns:
            Niveau affiché : "INFO" | "WARN" | "URGENT"
        """
        level = self._determine_level(state)
        with self._lock:
            if level == "URGENT":
                self._print_urgent(state)
            elif level == "WARN":
                self._print_warn(state)
            else:
                self._print_info(state)
        return level

    def notify_success(self, repo: str, run_id: str, metrics: Dict[str, Any] = None) -> None:
        """Notification simplifiée pour un run réussi."""
        with self._lock:
            self._print_info({
                "repo": repo,
                "run_id": run_id,
                "outcome": "success",
                "sonar_metrics": metrics or {},
                "sonar_gate": {"status": "OK"},
            })

    # ── Level determination ───────────────────────────────────────────────────

    @staticmethod
    def _determine_level(state: Dict[str, Any]) -> str:
        """
        Détermine le niveau de notification.

        URGENT si :
          - outcome == "failure"
          - Quality Gate FAILED (status == "ERROR")
          - Issues CRITICAL ou BLOCKER présentes

        WARN si :
          - outcome == "cancelled"
          - Quality Gate WARN
          - Coverage < 70%
          - CVE MEDIUM

        INFO sinon.
        """
        outcome = state.get("outcome", "success")
        gate_status = state.get("sonar_gate", {}).get("status", "OK")
        severity = state.get("severity", "INFO")
        issues = state.get("sonar_issues", [])
        metrics = state.get("sonar_metrics", {})

        # URGENT
        if outcome == "failure":
            return "URGENT"
        if gate_status == "ERROR":
            return "URGENT"
        if severity == "URGENT":
            return "URGENT"
        if any(i.get("severity") in ("BLOCKER", "CRITICAL") for i in issues):
            return "URGENT"

        # WARN
        if outcome == "cancelled":
            return "WARN"
        if gate_status == "WARN":
            return "WARN"
        if severity == "WARN":
            return "WARN"

        try:
            coverage = float(metrics.get("coverage", "100"))
            if coverage < 70.0:
                return "WARN"
        except (ValueError, TypeError):
            pass

        return "INFO"

    # ── Print methods ─────────────────────────────────────────────────────────

    def _print_header(self, icon: str, label: str, color: str, repo: str, run_id: str) -> None:
        width = 70
        print(f"\n{color}{_B}{'═' * width}{_E}")
        print(f"{color}{_B}  {icon}  CI/CD INTELLIGENCE — {label}{_E}")
        print(f"{color}{_B}{'─' * width}{_E}")
        print(f"{color}  Repo  : {repo}{_E}")
        print(f"{color}  Run   : {run_id}{_E}")

    def _print_sonar_summary(self, state: Dict[str, Any]) -> None:
        gate = state.get("sonar_gate", {})
        metrics = state.get("sonar_metrics", {})
        issues = state.get("sonar_issues", [])

        gate_status = gate.get("status", "N/A")
        gate_color = _G if gate_status == "OK" else (_R if gate_status == "ERROR" else _Y)

        print(f"\n{_B}  SonarCloud :{_E}")
        print(f"    Quality Gate  : {gate_color}{_B}{gate_status}{_E}")

        if metrics:
            cov = metrics.get("coverage", "N/A")
            bugs = metrics.get("bugs", "N/A")
            vulns = metrics.get("vulnerabilities", "N/A")
            print(f"    Coverage      : {cov}%")
            print(f"    Bugs          : {bugs}")
            print(f"    Vulnerabilities: {vulns}")

        if issues:
            print(f"\n  {_R}{_B}Issues critiques ({len(issues)}) :{_E}")
            for i in issues[:5]:
                print(f"    {_R}• [{i.get('severity', '?')}] {i.get('message', '')[:80]}{_E}")
                if i.get("component"):
                    print(f"      └─ {i['component']}")

    def _print_urgent(self, state: Dict[str, Any]) -> None:
        repo = state.get("repo", "unknown")
        run_id = state.get("run_id", "?")
        stage = state.get("stage_failed", "")
        failure_type = state.get("failure_type", "unknown")
        root_cause = state.get("root_cause", "")
        fix = state.get("suggested_fix", "")
        pr = state.get("pr_number")

        self._print_header("🚨", "FAILURE URGENTE", _R, repo, run_id)

        print(f"\n{_R}{_B}  ÉCHEC DÉTECTÉ{_E}")
        if stage:
            print(f"{_R}  Stage échoué  : {stage}{_E}")
        print(f"{_R}  Type d'erreur : {failure_type.upper()}{_E}")

        if pr:
            print(f"\n{_B}  PR ciblée     : #{pr}{_E}")

        self._print_sonar_summary(state)

        # Analyse LLM
        if root_cause:
            print(f"\n{_B}  Cause racine :{_E}")
            for line in root_cause.strip().splitlines()[:8]:
                print(f"    {line}")

        if fix:
            print(f"\n{_Y}{_B}  Fix suggéré :{_E}")
            for line in fix.strip().splitlines()[:6]:
                print(f"  {_Y}  {line}{_E}")

        # Similar fixes
        similar = state.get("similar_fixes", [])
        if similar and similar[0].get("fix_applied"):
            best = similar[0]
            print(f"\n{_C}  Fix similaire (vu {best['count']}x) :{_E}")
            print(f"  {_C}  {best['fix_applied'][:150]}{_E}")

        if state.get("comment_posted"):
            print(f"\n{_G}  ✓ Commentaire posté sur la PR #{pr}{_E}")

        print(f"{_R}{_B}{'═' * 70}{_E}\n")

    def _print_warn(self, state: Dict[str, Any]) -> None:
        repo = state.get("repo", "unknown")
        run_id = state.get("run_id", "?")
        gate = state.get("sonar_gate", {})

        self._print_header("⚠️ ", "ATTENTION REQUISE", _Y, repo, run_id)

        outcome = state.get("outcome", "")
        if outcome == "cancelled":
            print(f"\n{_Y}  Run annulé (cancelled){_E}")
        else:
            gate_status = gate.get("status", "")
            if gate_status == "WARN":
                print(f"\n{_Y}  Quality Gate en avertissement{_E}")
                for cond in gate.get("conditions", [])[:5]:
                    if cond.get("status") != "OK":
                        print(f"    • {cond.get('metric', '?')} : "
                              f"{cond.get('actualValue', '?')} "
                              f"(seuil: {cond.get('errorThreshold', '?')})")

        self._print_sonar_summary(state)
        print(f"{_Y}{_B}{'═' * 70}{_E}\n")

    def _print_info(self, state: Dict[str, Any]) -> None:
        repo = state.get("repo", "unknown")
        run_id = state.get("run_id", "?")

        self._print_header("✅", "PIPELINE OK", _G, repo, run_id)

        self._print_sonar_summary(state)

        indexed = state.get("indexed", False)
        if indexed:
            print(f"\n{_G}  ✓ Run indexé dans Redis{_E}")

        print(f"{_G}{_B}{'═' * 70}{_E}\n")
