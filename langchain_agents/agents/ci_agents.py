"""
ci_agents.py — Real tool-calling agents for the agentic CIGraph.

Each specialist is an LCBaseAgent: an LLM that autonomously decides which of
its bound tools to call to accomplish its mission. They reuse the existing
LangChain @tool objects from ci_tools.py — now genuinely selected by the LLM
instead of being called in a fixed order.

Roster:
  - CISupervisor         : LLM router — decides which specialists to activate
  - Diagnostic Agent     : finds the root cause (logs + Sonar + memory)
  - Security Triage Agent: correlates CodeQL + Trivy + Sonar, judges severity
  - Remediation Agent    : proposes a concrete fix, may memorize it

Orchestrated by ci_agent_graph.py (supervisor pattern). Deterministic
prefetch (logs download, classification) stays in the graph — only the
reasoning steps are agents (hybrid approach).
"""
from __future__ import annotations

import logging
from typing import List

from langchain_agents.agents.base_agent import LCBaseAgent

logger = logging.getLogger(__name__)

# Vocabulary of specialists the supervisor may route to.
CI_SPECIALISTS = ("diagnostic", "security_triage", "remediation")


# ── Specialist agents ─────────────────────────────────────────────────────────

def _ci_tools():
    """Imported lazily so importing this module never eagerly pulls heavy deps."""
    from langchain_agents.tools.ci_tools import (
        tool_fetch_run_logs,
        tool_fetch_job_logs,
        tool_sonar_quality_gate,
        tool_sonar_get_issues,
        tool_search_similar_failure,
        tool_store_ci_fix,
        tool_fetch_code_scanning_alerts,
        tool_fetch_trivy_artifact,
    )
    return {
        "fetch_run_logs": tool_fetch_run_logs,
        "fetch_job_logs": tool_fetch_job_logs,
        "sonar_quality_gate": tool_sonar_quality_gate,
        "sonar_get_issues": tool_sonar_get_issues,
        "search_similar_failure": tool_search_similar_failure,
        "store_ci_fix": tool_store_ci_fix,
        "code_scanning_alerts": tool_fetch_code_scanning_alerts,
        "trivy_artifact": tool_fetch_trivy_artifact,
    }


_DIAGNOSTIC_PROMPT = """\
You are a CI Diagnostic Agent, an expert DevOps engineer.
Your mission: find the ROOT CAUSE of a failed GitHub Actions pipeline run.

You have tools to:
  - fetch the failed job's logs (prefer fetch_job_logs when a job_id is given,
    otherwise fetch_run_logs for the whole run),
  - read the SonarCloud quality gate and its critical issues,
  - search past similar failures in memory.

Call the tools you actually need — don't guess. Once you understand the cause,
STOP calling tools and answer EXACTLY in this format:

ROOT CAUSE:
<2-3 sentences>

SUGGESTED FIX:
<concrete commands or code>
"""

_SECURITY_TRIAGE_PROMPT = """\
You are a CI Security Triage Agent.
Your mission: judge whether the pipeline's security findings should BLOCK the
merge, using CodeQL alerts, Trivy CVEs, and SonarCloud vulnerabilities.

Use your tools to gather the real findings (don't assume counts). Distinguish
"scanner ran and found nothing" from "scanner did not run". Then answer:

SECURITY VERDICT: BLOCK | WARN | PASS
KEY FINDINGS:
<short bullet list of the most severe real issues, with counts>
RECOMMENDATION:
<1-2 sentences>
"""

_REMEDIATION_PROMPT = """\
You are a CI Remediation Agent.
Given a diagnosed root cause, produce a CONCRETE, minimal fix (commands, config,
or code). If the fix is reliable and generic, you may store it in memory with
store_ci_fix so future identical failures are resolved instantly. Check memory
first with search_similar_failure to avoid duplicating a known fix.

Answer:

FIX:
<concrete steps or code>
MEMORIZED: yes|no
"""


# Module-level singletons (agents are stateless across runs; the LLM is lazy).
_diagnostic_agent: LCBaseAgent | None = None
_security_triage_agent: LCBaseAgent | None = None
_remediation_agent: LCBaseAgent | None = None


def get_diagnostic_agent() -> LCBaseAgent:
    global _diagnostic_agent
    if _diagnostic_agent is None:
        t = _ci_tools()
        _diagnostic_agent = LCBaseAgent(
            name="ci_diagnostic",
            system_prompt=_DIAGNOSTIC_PROMPT,
            tools=[
                t["fetch_job_logs"], t["fetch_run_logs"],
                t["sonar_quality_gate"], t["sonar_get_issues"],
                t["search_similar_failure"],
            ],
        )
    return _diagnostic_agent


def get_security_triage_agent() -> LCBaseAgent:
    global _security_triage_agent
    if _security_triage_agent is None:
        t = _ci_tools()
        _security_triage_agent = LCBaseAgent(
            name="ci_security_triage",
            system_prompt=_SECURITY_TRIAGE_PROMPT,
            tools=[
                t["code_scanning_alerts"], t["trivy_artifact"],
                t["sonar_get_issues"],
            ],
        )
    return _security_triage_agent


def get_remediation_agent() -> LCBaseAgent:
    global _remediation_agent
    if _remediation_agent is None:
        t = _ci_tools()
        _remediation_agent = LCBaseAgent(
            name="ci_remediation",
            system_prompt=_REMEDIATION_PROMPT,
            tools=[t["search_similar_failure"], t["store_ci_fix"]],
        )
    return _remediation_agent


# ── Supervisor (LLM router) ───────────────────────────────────────────────────

_SUPERVISOR_PROMPT = """\
You are the CI Supervisor orchestrating specialist agents for a GitHub Actions
run. Based on the run outcome and the classified failure, decide WHICH
specialists to activate.

Available specialists:
  - diagnostic      : root-cause analysis of a build/test/generic failure
  - security_triage : judge CodeQL/Trivy/Sonar security findings
  - remediation     : propose (and maybe memorize) a concrete fix

Rules of thumb:
  - A successful run needs no specialist.
  - A security-type failure → security_triage (+ diagnostic if useful).
  - A build/test/docker failure → diagnostic → remediation.

Respond with ONLY a comma-separated list from
[diagnostic, security_triage, remediation], or the word "none".
"""


class CISupervisor:
    """LLM router deciding which specialists run. Falls back to a rule-based
    default if the LLM is unavailable or its answer can't be parsed."""

    name = "ci_supervisor"

    def route(self, outcome: str, failure_type: str, logs_snippet: str = "") -> List[str]:
        # Success / non-failure → no reasoning needed.
        if outcome in ("success", "cancelled", "retrying"):
            return []

        decision = self._llm_route(outcome, failure_type, logs_snippet)
        if decision is not None:
            return decision
        return self._rule_route(failure_type)

    def _llm_route(self, outcome, failure_type, logs_snippet) -> List[str] | None:
        try:
            from services.llm_factory import invoke_with_fallback
            task = (
                f"Run outcome: {outcome}\n"
                f"Classified failure type: {failure_type}\n"
                f"Logs (excerpt): {logs_snippet[:600]}\n\n"
                "Which specialists?"
            )
            raw = invoke_with_fallback(
                f"{_SUPERVISOR_PROMPT}\n\n{task}",
                label="ci_supervisor",
                max_tokens=64,
            )
            low = (raw or "").lower()
            if "none" in low:
                return []
            picked = [s for s in CI_SPECIALISTS if s in low]
            return picked or None
        except Exception as e:
            logger.debug("[CISupervisor] LLM route failed: %s", e)
            return None

    @staticmethod
    def _rule_route(failure_type: str) -> List[str]:
        if failure_type == "security":
            return ["security_triage", "diagnostic"]
        # build / test / docker / unknown
        return ["diagnostic", "remediation"]


_supervisor: CISupervisor | None = None


def get_ci_supervisor() -> CISupervisor:
    global _supervisor
    if _supervisor is None:
        _supervisor = CISupervisor()
    return _supervisor
