"""
chat_memory_service.py — Redis conversation memory for ChatAgent.

Phase 1:
  - store per-session turns in Redis
  - keep only the last N turns
  - store lightweight metadata
  - graceful fallback if Redis/MCP is not available

Redis keys:
  ca:chat:{session_id}:history  -> JSON list of turns
  ca:chat:{session_id}:meta     -> JSON metadata
  ca:chat:sessions              -> sorted set index when Redis supports it
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ChatMemoryService:
    """Small Redis-backed chat history service."""

    HISTORY_TTL_SECONDS = 24 * 3600
    MAX_TURNS = 12

    def __init__(self, max_turns: int = MAX_TURNS, ttl_seconds: int = HISTORY_TTL_SECONDS):
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self._fallback: Dict[str, List[Dict[str, Any]]] = {}

    # ── Keys ────────────────────────────────────────────────────────────────

    @staticmethod
    def new_session_id() -> str:
        return uuid.uuid4().hex[:12]

    @staticmethod
    def history_key(session_id: str) -> str:
        return f"ca:chat:{session_id}:history"

    @staticmethod
    def meta_key(session_id: str) -> str:
        return f"ca:chat:{session_id}:meta"

    def _redis(self):
        try:
            from services.mcp_redis_service import get_mcp_redis
            return get_mcp_redis()
        except Exception as e:
            logger.debug("ChatMemoryService Redis unavailable: %s", e)
            return None

    # ── Public API ──────────────────────────────────────────────────────────

    def load_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Return recent conversation turns for a session."""
        if not session_id:
            return []

        redis = self._redis()
        key = self.history_key(session_id)

        if redis:
            try:
                raw = redis.get(key)
                if not raw:
                    return []
                data = json.loads(raw)
                if isinstance(data, list):
                    return data[-self.max_turns:]
            except Exception as e:
                logger.debug("load_history Redis error: %s", e)

        return self._fallback.get(session_id, [])[-self.max_turns:]

    def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one message turn to the session history."""
        if not session_id:
            return

        turn = {
            "role": role,
            "content": content,
            "ts": int(time.time()),
            "metadata": metadata or {},
        }

        history = self.load_history(session_id)
        history.append(turn)
        history = history[-self.max_turns:]

        redis = self._redis()
        key = self.history_key(session_id)

        if redis:
            try:
                redis.set(
                    key,
                    json.dumps(history, ensure_ascii=False, default=str),
                    expire_seconds=self.ttl_seconds,
                )
                return
            except Exception as e:
                logger.debug("append_turn Redis error: %s", e)

        self._fallback[session_id] = history

    def save_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save user + assistant messages as one exchange."""
        self.append_turn(session_id, "user", user_message, metadata)
        self.append_turn(session_id, "assistant", assistant_response, metadata)

    def set_meta(self, session_id: str, meta: Dict[str, Any]) -> None:
        """Store lightweight session metadata."""
        if not session_id:
            return
        payload = {**meta, "updated_at": int(time.time())}
        redis = self._redis()
        if redis:
            try:
                redis.set(
                    self.meta_key(session_id),
                    json.dumps(payload, ensure_ascii=False, default=str),
                    expire_seconds=self.ttl_seconds,
                )
            except Exception as e:
                logger.debug("set_meta Redis error: %s", e)

    def get_meta(self, session_id: str) -> Dict[str, Any]:
        """Read session metadata."""
        redis = self._redis()
        if not redis:
            return {}
        try:
            raw = redis.get(self.meta_key(session_id))
            return json.loads(raw) if raw else {}
        except Exception:
            return {}


chat_memory_service = ChatMemoryService()
