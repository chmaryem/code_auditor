"""
services/persistent_chat_memory_service.py — Dual-layer chat memory.

Architecture:
  PostgreSQL = source de vérité durable (conversations + messages).
  Redis      = cache chaud pour la session active (TTL 1 h).

Write path (save_exchange / save_exchange_sync):
  1. PostgreSQL → durabilité garantie, survit au logout et au redémarrage
  2. Redis      → lecture rapide pendant la session active

Read path (load_history / list_sessions):
  1. Redis first → fast path (cache chaud)
  2. PG fallback → cold start, après logout, expiration TTL Redis

Session lifecycle:
  - Logout : clés Redis expirées/supprimées ; lignes PG intactes pour toujours
  - Login  : cache Redis reconstruit paresseusement au premier read depuis PG
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

    # ── Sync API (background threads uniquement) ─────────────────────────────

    def save_exchange_sync(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        project_path: str = "",
        intent: Optional[str] = None,
    ) -> None:
        """
        Persiste un échange user↔assistant depuis un thread background.

        Utilise SyncSessionLocal (psycopg2) — ne jamais appeler depuis un
        contexte asyncio.  Résout l'email via auth.store pour retrouver
        l'utilisateur PG correspondant au Redis user_id.

        Ordre d'écriture :
          1. PostgreSQL (durable) — le cache Redis est inutile si PG échoue
          2. Redis (cache chaud)
        """
        from database.connection import SyncSessionLocal
        from database.models import Conversation, Message, User
        from datetime import datetime, timezone
        from sqlalchemy import select, update as sa_update

        def _now() -> datetime:
            return datetime.now(timezone.utc)

        # ── Résolution email depuis le store Redis auth ────────────────────
        user_email = ""
        try:
            from auth.store import get_user as _get_auth_user
            auth_user = _get_auth_user(user_id) or {}
            user_email = auth_user.get("email", "")
        except Exception as exc:
            logger.debug("save_exchange_sync: cannot resolve email for user_id=%s: %s", user_id, exc)

        if not user_email or user_email == "local@localhost":
            logger.debug(
                "save_exchange_sync: skipping PG write — no valid email "
                "(user_id=%s, session=%s)", user_id, session_id,
            )
            # Écriture Redis uniquement pour ne pas perdre la session active
            self._write_redis_cache(session_id, user_message, assistant_response, metadata, project_path)
            return

        title = _make_title(user_message)

        # ── 1. PostgreSQL (source de vérité) ──────────────────────────────
        try:
            with SyncSessionLocal() as db:
                # Upsert utilisateur PG
                pg_user = db.execute(
                    select(User).where(User.email == user_email)
                ).scalar_one_or_none()
                if not pg_user:
                    pg_user = User(
                        email=user_email,
                        name=user_email.split("@")[0],
                        role="Developer",
                    )
                    db.add(pg_user)
                    db.flush()

                # Get-or-create conversation
                conv = db.execute(
                    select(Conversation).where(Conversation.session_id == session_id)
                ).scalar_one_or_none()
                if not conv:
                    conv = Conversation(
                        session_id=session_id,
                        user_id=pg_user.id,
                        title=title,
                        intent=intent,
                        created_at=_now(),
                        updated_at=_now(),
                    )
                    db.add(conv)
                    db.flush()

                # Append messages
                db.add(Message(
                    conversation_id=conv.id,
                    role="user",
                    content=user_message,
                    metadata_=metadata,
                    created_at=_now(),
                ))
                db.add(Message(
                    conversation_id=conv.id,
                    role="assistant",
                    content=assistant_response,
                    metadata_=metadata,
                    created_at=_now(),
                ))

                # Bump conversation metadata
                new_title = title if (not conv.title or conv.title == "New conversation") else conv.title
                db.execute(
                    sa_update(Conversation)
                    .where(Conversation.id == conv.id)
                    .values(
                        turn_count=conv.turn_count + 2,
                        updated_at=_now(),
                        title=new_title,
                        intent=intent or conv.intent,
                    )
                )
                db.commit()
        except Exception as exc:
            logger.warning(
                "save_exchange_sync: PG write failed (session=%s): %s",
                session_id, exc,
            )

        # ── 2. Redis (cache chaud) ────────────────────────────────────────
        self._write_redis_cache(session_id, user_message, assistant_response, metadata, project_path)

    def _write_redis_cache(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]],
        project_path: str,
    ) -> None:
        """Délégation vers ChatMemoryService pour le cache Redis."""
        try:
            self._redis.save_exchange(
                session_id=session_id,
                user_message=user_message,
                assistant_response=assistant_response,
                metadata=metadata,
                project_path=project_path,
            )
        except Exception as exc:
            logger.debug("_write_redis_cache failed (session=%s): %s", session_id, exc)

    # ── Invalidation ─────────────────────────────────────────────────────────

    def invalidate_redis_cache(self, session_id: str) -> None:
        """
        Supprime les clés Redis pour une session (appelé au logout).
        Les données PostgreSQL ne sont PAS touchées — l'historique survit.
        """
        redis = self._get_redis()
        if redis:
            try:
                redis.delete(self._redis_history_key(session_id))
                redis.delete(f"ca:chat:{session_id}:meta")
            except Exception as exc:
                logger.debug("invalidate_redis_cache error: %s", exc)


# Singleton module-level — partagé par chat_router et node_memory_save
persistent_chat_memory = PersistentChatMemoryService()
