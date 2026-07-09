"""
cd_agents.py — Real tool-calling agents for the agentic CDGraph.

Same pattern as ci_agents.py: each specialist is an LCBaseAgent whose LLM
autonomously calls the existing cd_tools.py @tool objects. The structured
deployment fields the dashboard needs (readiness score, health grade) are still
produced by the reused deterministic nodes; these agents add the REASONING
(failure analysis, rollback decision) and the supervisor decides routing.

Roster:
  - CDSupervisor          : LLM router — deploy healthy vs failed
  - Deploy Analysis Agent : reasons on a failed deploy (root cause)
  - Rollback Agent        : calls advise_rollback, judges the rollback decision
"""
from __future__ import annotations

import logging
from typing import List

from langchain_agents.agents.base_agent import LCBaseAgent

logger = logging.getLogger(__name__)

CD_SPECIALISTS = ("deploy_analysis", "rollback")


def _cd_tools():
    from langchain_agents.tools.cd_tools import (
        tool_get_env_state,
        tool_get_last_successful_deploy,
        tool_advise_rollback,
        tool_get_deploy_stats,
    )
    return {
        "get_env_state": tool_get_env_state,
        "get_last_successful_deploy": tool_get_last_successful_deploy,
        "advise_rollback": tool_advise_rollback,
        "get_deploy_stats": tool_get_deploy_stats,
    }


_DEPLOY_ANALYSIS_PROMPT = """\
You are a CD Deploy Analysis Agent, an expert SRE.
A deployment (or its pipeline) has failed or is unhealthy. The health grade and
issues are already provided in the Context — DO NOT try to probe any URL. Use
your tools only to read the recorded environment state and deploy history, then
explain what went wrong. Don't invent URLs or data. Answer EXACTLY as:

ROOT CAUSE:
<2-3 sentences>

SUGGESTED FIX:
<concrete remediation steps>
"""

_ROLLBACK_PROMPT = """\
You are a CD Rollback Agent.
Decide whether rolling back is advisable after a failed deployment. Use
advise_rollback (it compares against the last successful deploy and computes a
risk level) and get_last_successful_deploy. If no prior successful deploy
exists, rollback is NOT possible — say so. Answer EXACTLY as:

ROLLBACK DECISION: RECOMMENDED | NOT_RECOMMENDED | NOT_POSSIBLE
RISK: LOW | MEDIUM | HIGH | N/A
REASONING:
<1-3 sentences>
"""


_deploy_analysis_agent: LCBaseAgent | None = None
_rollback_agent: LCBaseAgent | None = None


def get_deploy_analysis_agent() -> LCBaseAgent:
    global _deploy_analysis_agent
    if _deploy_analysis_agent is None:
        t = _cd_tools()
        # No check_health here: health is already computed by the deterministic
        # monitor_health node and passed as context. Giving the agent a URL
        # prober invited hallucinated deploy_urls + tool-schema 400 errors.
        _deploy_analysis_agent = LCBaseAgent(
            name="cd_deploy_analysis",
            system_prompt=_DEPLOY_ANALYSIS_PROMPT,
            tools=[t["get_env_state"], t["get_deploy_stats"]],
        )
    return _deploy_analysis_agent


def get_rollback_agent() -> LCBaseAgent:
    global _rollback_agent
    if _rollback_agent is None:
        t = _cd_tools()
        _rollback_agent = LCBaseAgent(
            name="cd_rollback",
            system_prompt=_ROLLBACK_PROMPT,
            tools=[t["advise_rollback"], t["get_last_successful_deploy"]],
        )
    return _rollback_agent


# ── Supervisor (LLM router) ───────────────────────────────────────────────────

_SUPERVISOR_PROMPT = """\
You are the CD Supervisor. Given the deployment outcome, decide which
specialists to activate.

Specialists:
  - deploy_analysis : root-cause a failed/unhealthy deployment
  - rollback        : evaluate whether to roll back

Rules:
  - Healthy deploy (grade HEALTHY, conclusion success) → none.
  - Failed/unhealthy deploy → deploy_analysis, rollback.

Respond with ONLY a comma-separated list from
[deploy_analysis, rollback], or the word "none".
"""


class CDSupervisor:
    name = "cd_supervisor"

    def route(self, monitor_grade: str, run_conclusion: str, deploy_status: str) -> List[str]:
        healthy = (
            monitor_grade == "HEALTHY"
            and run_conclusion == "success"
            and deploy_status != "failed"
        )
        if healthy:
            return []

        decision = self._llm_route(monitor_grade, run_conclusion, deploy_status)
        if decision is not None:
            return decision
        # Rule fallback: any non-healthy deploy → full analysis + rollback eval.
        return ["deploy_analysis", "rollback"]

    def _llm_route(self, grade, conclusion, status) -> List[str] | None:
        try:
            from services.llm_factory import invoke_with_fallback
            task = (
                f"Health grade: {grade}\n"
                f"Run conclusion: {conclusion}\n"
                f"Deploy status: {status}\n\nWhich specialists?"
            )
            raw = invoke_with_fallback(
                f"{_SUPERVISOR_PROMPT}\n\n{task}",
                label="cd_supervisor",
                max_tokens=48,
            )
            low = (raw or "").lower()
            if "none" in low:
                return []
            picked = [s for s in CD_SPECIALISTS if s in low]
            return picked or None
        except Exception as e:
            logger.debug("[CDSupervisor] LLM route failed: %s", e)
            return None


_cd_supervisor: CDSupervisor | None = None


def get_cd_supervisor() -> CDSupervisor:
    global _cd_supervisor
    if _cd_supervisor is None:
        _cd_supervisor = CDSupervisor()
    return _cd_supervisor
