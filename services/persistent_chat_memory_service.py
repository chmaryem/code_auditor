"""
services/persistent_chat_memory_service.py — Dual-write chat memory.

Write path (save_exchange):
  1. Write to PostgreSQL  → durable, survives logout and server restart
  2. Write to Redis cache → fast reads during active session (TTL: 1h)

Read path (load_history):
  1. Try Redis first     → fast, hot data
  2. Fall back to PG     → cold start, after logout, Redis eviction

Session lifecycle:
  - Logout: Redis key expires/is removed; PostgreSQL row stays forever
  - Login : Redis cache is rebuilt lazily on first read from PG
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from database.repositories.conversation_repo import ConversationRepo
from database.repositories.user_repo import UserRepo
from services.chat_memory_service import ChatMemoryService, _make_title, _project_hash

logger = logging.getLogger(__name__)

# Hot-cache TTL for active sessions (shorter than old 24h value — PG is the source of truth)
_CACHE_TTL = 3600  # 1 hour


class PersistentChatMemoryService:
    """
    Drop-in replacement / companion for ChatMemoryService that adds PostgreSQL
    durability. All Redis logic is delegated to the existing ChatMemoryService.
    """

    def __init__(self, redis_memory: Optional[ChatMemoryService] = None) -> None:
        self._redis = redis_memory or ChatMemoryService(max_turns=50, ttl_seconds=_CACHE_TTL)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get_redis(self):
        try:
            from services.mcp_redis_service import get_mcp_redis
            return get_mcp_redis()
        except Exception:
            return None

    def _redis_history_key(self, session_id: str) -> str:
        return f"ca:chat:{session_id}:history"

    def _prime_redis_cache(self, session_id: str, messages: list) -> None:
        """Write PG messages back into Redis so next read is fast."""
        redis = self._get_redis()
        if not redis or not messages:
            return
        turns = [
            {
                "role": m.role,
                "content": m.content,
                "ts": int(m.created_at.timestamp()),
                "metadata": m.metadata_ or {},
            }
            for m in messages
        ]
        try:
            redis.set(
                self._redis_history_key(session_id),
                json.dumps(turns, ensure_ascii=False, default=str),
                expire_seconds=_CACHE_TTL,
            )
        except Exception as e:
            logger.debug("_prime_redis_cache error: %s", e)

    # ── Public API ───────────────────────────────────────────────────────────

    async def load_history(
        self,
        session_id: str,
        db: AsyncSession,
        max_turns: int = 12,
    ) -> List[Dict[str, Any]]:
        """
        Return recent turns for a session.
        Redis is tried first; on miss, PostgreSQL is queried and Redis is primed.
        """
        # 1 — try Redis (fast path)
        redis_turns = self._redis.load_history(session_id)
        if redis_turns:
            return redis_turns[-max_turns:]

        # 2 — fall back to PostgreSQL
        conv_repo = ConversationRepo(db)
        conv = await conv_repo.get_by_session_id(session_id)
        if not conv:
            return []

        messages = await conv_repo.load_messages(conv.id, limit=max_turns * 2)
        if not messages:
            return []

        # Prime Redis cache so next call is fast
        self._prime_redis_cache(session_id, messages[-max_turns:])

        return [
            {
                "role": m.role,
                "content": m.content,
                "ts": int(m.created_at.timestamp()),
                "metadata": m.metadata_ or {},
            }
            for m in messages[-max_turns:]
        ]

    async def save_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        user_id: str,
        db: AsyncSession,
        metadata: Optional[Dict[str, Any]] = None,
        project_path: str = "",
        project_id: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> None:
        """
        Persist one user↔assistant exchange.

        Write order:
          1. PostgreSQL (durable)
          2. Redis     (hot cache)

        If PostgreSQL fails: log warning, still write Redis so the session
        keeps working — the data will be missing from history after restart.
        If Redis fails: silent (PG is the source of truth, Redis is cache).
        """
        title = _make_title(user_message)

        # 1 — PostgreSQL write (primary)
        try:
            conv_repo = ConversationRepo(db)
            await conv_repo.save_exchange(
                session_id=session_id,
                user_id=user_id,
                user_message=user_message,
                assistant_response=assistant_response,
                metadata=metadata,
                project_id=project_id,
                title=title,
                intent=intent,
            )
        except Exception as exc:
            logger.warning(
                "PersistentChatMemory: PostgreSQL write failed for session %s: %s",
                session_id,
                exc,
            )

        # 2 — Redis write (cache)
        try:
            self._redis.save_exchange(
                session_id=session_id,
                user_message=user_message,
                assistant_response=assistant_response,
                metadata=metadata,
                project_path=project_path,
            )
        except Exception as exc:
            logger.debug("PersistentChatMemory: Redis cache write failed: %s", exc)

    async def list_sessions(
        self,
        user_id: str,
        db: AsyncSession,
        project_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        List conversations from PostgreSQL (authoritative).
        Redis index is a fast supplement but PG is the source of truth here.
        """
        conv_repo = ConversationRepo(db)
        convs = await conv_repo.list_for_user(user_id, project_id=project_id, limit=limit)
        return [
            {
                "session_id": c.session_id,
                "title": c.title,
                "updated_at": int(c.updated_at.timestamp()),
                "turn_count": c.turn_count,
                "intent": c.intent,
            }
            for c in convs
        ]

    def invalidate_redis_cache(self, session_id: str) -> None:
        """
        Called on logout to remove the Redis cache for a session.
        PostgreSQL data is NOT touched — history survives.
        """
        redis = self._get_redis()
        if redis:
            try:
                redis.delete(self._redis_history_key(session_id))
                redis.delete(f"ca:chat:{session_id}:meta")
            except Exception as e:
                logger.debug("invalidate_redis_cache error: %s", e)


# Module-level singleton — injected into lc_chat_agent via dependency
persistent_chat_memory = PersistentChatMemoryService()
