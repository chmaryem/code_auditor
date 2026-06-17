"""
api/hardening_router.py — Agentic file-hardening endpoints.

Routes:
  POST /api/harden/start    → start a hardening session (streaming SSE)
  POST /api/harden/continue → resume after dev Keep/Reject/Skip (streaming SSE)

Both endpoints return text/event-stream.
Each SSE event is a JSON object:

  {"type": "hardening_analyze",  "iteration": 1, "issues": [...], "score": 60}
  {"type": "hardening_plan",     "iteration": 1, "issue": {...}, "try_num": 1}
  {"type": "hardening_fix",      "iteration": 1, "has_fix": true}
  {"type": "hardening_verify",   "iteration": 1, "tier1": "pass", "tier2": true}
  {"type": "hardening_decide",   "iteration": 1, "verdict": "accepted", "patch": {...}}
  {"type": "hardening_awaiting", "patch": {...}}    ← step_by_step gate
  {"type": "hardening_done",     "summary": {...}}  ← loop finished
  {"type": "error",              "content": "..."}  ← on exception
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

hardening_router = APIRouter(tags=["hardening"])

# In-memory session store (keyed by session_id).
# Production-grade: move to Redis using your existing MCP Redis infrastructure.
_sessions: Dict[str, Any] = {}


# ─── Request models ───────────────────────────────────────────────────────────

class HardenStartRequest(BaseModel):
    file_path:      str
    project_path:   str
    language:       str  = "python"
    goal_score:     int  = Field(90,  ge=50,  le=100)
    max_iterations: int  = Field(8,   ge=1,   le=20)
    mode:           str  = "step_by_step"
    session_id:     str  = ""


class HardenContinueRequest(BaseModel):
    session_id: str
    decision:   Literal["keep", "reject", "skip"]


# ─── SSE helpers ──────────────────────────────────────────────────────────────

def _sse(event: Dict[str, Any]) -> str:
    """Format a dict as an SSE data line (same pattern as chat_router)."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _streaming_response(generator) -> StreamingResponse:
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "Connection":       "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─── Endpoints ────────────────────────────────────────────────────────────────

@hardening_router.post("/harden/start", summary="Start an agentic hardening session")
async def harden_start(req: HardenStartRequest):
    """
    Launch the HardeningGraph on a file.

    In step_by_step mode the stream pauses after each accepted fix with a
    hardening_awaiting event — the plugin shows the fix inline and waits for
    the developer's Keep/Reject/Skip before calling /harden/continue.
    """
    from langchain_agents.graphs.hardening_graph import ainvoke_hardening

    # Reuse or create session_id
    session_id = req.session_id or _make_session_id(req.file_path)

    async def event_generator():
        queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()

        def broadcast_cb(event: Dict[str, Any]):
            """Synchronous callback → pushes into async queue."""
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

        # Drain the queue while the graph is running
        async def drain():
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield _sse(event)
                # hardening_awaiting → pause here until /continue is called
                if event.get("type") == "hardening_awaiting":
                    return

        try:
            graph_task = asyncio.create_task(
                ainvoke_hardening(
                    file_path      = req.file_path,
                    project_path   = req.project_path,
                    language       = req.language,
                    goal_score     = req.goal_score,
                    max_iterations = req.max_iterations,
                    mode           = req.mode,
                    broadcast_cb   = broadcast_cb,
                )
            )

            # Stream events while graph runs
            async for chunk in drain():
                yield chunk

            # Wait for graph to finish if it hasn't paused
            state = await graph_task

            # Store state for step_by_step continuation
            _sessions[session_id] = state

            final_status = state.get("status", "done")
            if final_status == "awaiting":
                # Paused — emit awaiting event for plugin
                last_patch = (state.get("staged_patches") or [{}])[-1]
                yield _sse({
                    "type":       "hardening_awaiting",
                    "session_id": session_id,
                    "patch":      last_patch,
                    "iteration":  state.get("iteration", 0),
                })
            else:
                yield _sse(_build_done_event(state, session_id))

        except asyncio.CancelledError:
            logger.info("Hardening stream cancelled — client disconnected")
            raise
        except Exception as e:
            logger.exception("Hardening start error: %s", e)
            yield _sse({"type": "error", "content": str(e)})
        finally:
            await queue.put(None)   # signal drain to stop

    return _streaming_response(event_generator())


@hardening_router.post("/harden/continue", summary="Resume hardening after dev decision")
async def harden_continue(req: HardenContinueRequest):
    """
    Resume a paused step_by_step session after the developer decides
    Keep / Reject / Skip on the last proposed fix.
    """
    from langchain_agents.graphs.hardening_graph import ainvoke_continue

    state = _sessions.get(req.session_id)
    if state is None:
        async def not_found():
            yield _sse({"type": "error", "content": f"Session '{req.session_id}' not found or expired"})
        return _streaming_response(not_found())

    async def event_generator():
        queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()

        def broadcast_cb(event: Dict[str, Any]):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

        # Re-inject broadcast into state
        state["_broadcast"] = broadcast_cb

        async def drain():
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield _sse(event)
                if event.get("type") == "hardening_awaiting":
                    return

        try:
            graph_task = asyncio.create_task(
                ainvoke_continue(state, req.decision)
            )

            async for chunk in drain():
                yield chunk

            new_state = await graph_task
            _sessions[req.session_id] = new_state

            final_status = new_state.get("status", "done")
            if final_status == "awaiting":
                last_patch = (new_state.get("staged_patches") or [{}])[-1]
                yield _sse({
                    "type":       "hardening_awaiting",
                    "session_id": req.session_id,
                    "patch":      last_patch,
                    "iteration":  new_state.get("iteration", 0),
                })
            else:
                yield _sse(_build_done_event(new_state, req.session_id))
                _sessions.pop(req.session_id, None)   # clean up

        except asyncio.CancelledError:
            logger.info("Hardening continue stream cancelled")
            raise
        except Exception as e:
            logger.exception("Hardening continue error: %s", e)
            yield _sse({"type": "error", "content": str(e)})
        finally:
            await queue.put(None)

    return _streaming_response(event_generator())


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_session_id(file_path: str) -> str:
    import hashlib, time
    raw = f"{file_path}:{time.time()}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _build_done_event(state: Any, session_id: str) -> Dict[str, Any]:
    staged  = state.get("staged_patches", [])
    initial = state.get("score_initial", 0)
    final   = state.get("score", 0)
    return {
        "type":           "hardening_done",
        "session_id":     session_id,
        "status":         state.get("status", "done"),
        "done_reason":    state.get("done_reason", ""),
        "score_before":   initial,
        "score_after":    final,
        "score_delta":    final - initial,
        "patches_staged": len(staged),
        "patches":        staged,
        "iterations":     state.get("iteration", 0),
        "attempts":       len(state.get("attempts", [])),
    }
