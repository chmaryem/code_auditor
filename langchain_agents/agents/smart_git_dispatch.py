"""
smart_git_dispatch.py — thin entry-point into the Smart Git LangGraph.

Free-text callers (today: the ChatAgent, via node_git_question in
chat_graph.py) need intent classification first. This function builds the
initial SmartGitState and invokes the compiled multi-agent graph
(langchain_agents/graphs/smart_git_graph.py):

    router → [route_by_intent] → {working_copy | pr | conflict} → synthesize

It keeps the exact same signature and return shape as before (a state dict
containing `response`), so chat_graph.py's node_git_question is unchanged.

Endpoints (api/git_router.py) and the CLI (main.py) already know their intent
and call the domain layer (smart_git/*.py) directly — they do not use this.
"""
from __future__ import annotations

import time
from typing import Any, Dict


async def adispatch_smart_git_message(
    message: str,
    project_path: str = ".",
    repo: str = "",
    owner: str = "",
    pr_number: int = 0,
    branch: str = "HEAD",
    base: str = "main",
    session_id: str = "",
) -> Dict[str, Any]:
    """
    Classify `message` and run the matching Smart Git agent(s) via the graph.
    Returns the final graph state (contains `response`) — same shape as before.
    """
    from langchain_agents.graphs.smart_git_graph import get_smart_git_graph

    start = time.time()
    initial: Dict[str, Any] = {
        "user_message": message,
        "project_path": project_path,
        "repo": repo,
        "owner": owner,
        "pr_number": int(pr_number or 0),
        "branch": branch,
        "base": base,
        "session_id": session_id,
        "actions": [],
        "warnings": [],
        "errors": [],
        "stats": {},
    }

    final: Dict[str, Any] = await get_smart_git_graph().ainvoke(initial)

    final.setdefault("stats", {})
    final["stats"]["elapsed"] = round(time.time() - start, 2)
    return final
