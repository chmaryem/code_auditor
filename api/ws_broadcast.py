"""
api/ws_broadcast.py — Shared WebSocket broadcast singleton.

Provides a single ConnectionManager instance and a thread-safe broadcast
helper so any router (ci_router, git_router…) can emit events without
touching server.py internals.

Usage in a background thread:
    from api.ws_broadcast import broadcast_from_thread
    broadcast_from_thread({"type": "history_update", "module": "cicd"})

Usage in an async route (await directly):
    from api.ws_broadcast import manager
    await manager.broadcast({"type": "..."})
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from api.websocket_manager import ConnectionManager

logger = logging.getLogger(__name__)

manager: ConnectionManager = ConnectionManager()
_loop: asyncio.AbstractEventLoop | None = None


def set_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Called once at server startup to capture the running event loop."""
    global _loop
    _loop = loop


def broadcast_from_thread(event: Dict[str, Any]) -> None:
    """
    Thread-safe fire-and-forget broadcast.
    Safe to call from any daemon background thread.
    """
    if _loop is None or not _loop.is_running():
        logger.debug("ws_broadcast: no event loop, skipping broadcast of %s", event.get("type"))
        return
    asyncio.run_coroutine_threadsafe(manager.broadcast(event), _loop)
