
from __future__ import annotations

import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


def agentic_enabled() -> bool:
    """Agentic multi-agent graphs are the default. Only an explicit
    CI_CD_AGENTIC=0/false/no/off forces the deterministic graphs."""
    return os.environ.get("CI_CD_AGENTIC", "").strip().lower() not in ("0", "false", "no", "off")


def run_ci_graph(**kwargs) -> Dict[str, Any]:
    """Dispatch a CI run to the agentic or deterministic CIGraph."""
    if agentic_enabled():
        try:
            from langchain_agents.graphs.ci_agent_graph import invoke_ci_agent_run
            logger.info("[dispatch] CI → agentic graph")
            return invoke_ci_agent_run(**kwargs)
        except Exception as e:
            logger.error("[dispatch] agentic CI failed (%s) — falling back to deterministic", e)
    from langchain_agents.graphs.ci_graph import invoke_ci_run
    return invoke_ci_run(**kwargs)


def run_cd_graph(**kwargs) -> Dict[str, Any]:
    """Dispatch a CD run to the agentic or deterministic CDGraph."""
    if agentic_enabled():
        try:
            from langchain_agents.graphs.cd_agent_graph import invoke_cd_agent_run
            logger.info("[dispatch] CD → agentic graph")
            return invoke_cd_agent_run(**kwargs)
        except Exception as e:
            logger.error("[dispatch] agentic CD failed (%s) — falling back to deterministic", e)
    from langchain_agents.graphs.cd_graph import invoke_cd_run
    return invoke_cd_run(**kwargs)
