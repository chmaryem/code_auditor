from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
import uuid

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from database.models import ExtensionChatMessage, ExtensionChatSession


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return uuid.uuid4().hex


class ExtensionChatRepo:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # ── Session upsert (create or refresh title/updated_at) ───────────────────

    async def upsert_session(
        self,
        user_id: str,
        session_id: str,
        title: str = "New chat",
        workspace: Optional[str] = None,
    ) -> ExtensionChatSession:
        result = await self.db.execute(
            select(ExtensionChatSession).where(
                ExtensionChatSession.session_id == session_id,
                ExtensionChatSession.user_id == user_id,
            )
        )
        session = result.scalar_one_or_none()

        if session:
            session.title      = title
            session.updated_at = _now()
            if workspace:
                session.workspace = workspace
            await self.db.flush()
            return session

        session = ExtensionChatSession(
            id         = _uuid(),
            user_id    = user_id,
            session_id = session_id,
            title      = title,
            workspace  = workspace,
            created_at = _now(),
            updated_at = _now(),
        )
        self.db.add(session)
        await self.db.flush()
        return session

    # ── Replace all messages for a session (full snapshot sync) ──────────────

    async def replace_messages(
        self,
        session_pk: str,
        messages: List[dict],
    ) -> None:
        """Delete existing messages then insert the new snapshot."""
        from sqlalchemy import delete as sa_delete
        await self.db.execute(
            sa_delete(ExtensionChatMessage).where(
                ExtensionChatMessage.session_id_fk == session_pk
            )
        )
        for m in messages:
            self.db.add(ExtensionChatMessage(
                id           = _uuid(),
                session_id_fk = session_pk,
                role         = m.get("role", "user"),
                content      = m.get("text", ""),
                context_file = m.get("contextFile"),
                ts           = m.get("timestamp"),
                created_at   = _now(),
            ))
        await self.db.flush()

    # ── List sessions for a user ──────────────────────────────────────────────

    async def list_sessions(
        self,
        user_id: str,
        limit: int = 30,
    ) -> List[ExtensionChatSession]:
        result = await self.db.execute(
            select(ExtensionChatSession)
            .where(ExtensionChatSession.user_id == user_id)
            .options(selectinload(ExtensionChatSession.messages))
            .order_by(desc(ExtensionChatSession.updated_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    # ── Get one session with messages ─────────────────────────────────────────

    async def get_session(
        self,
        user_id: str,
        session_id: str,
    ) -> Optional[ExtensionChatSession]:
        result = await self.db.execute(
            select(ExtensionChatSession)
            .where(
                ExtensionChatSession.session_id == session_id,
                ExtensionChatSession.user_id    == user_id,
            )
            .options(selectinload(ExtensionChatSession.messages))
        )
        return result.scalar_one_or_none()

    # ── Append a single message (efficient — no full-replace) ────────────────

    async def append_message(
        self,
        session_pk:   str,
        user_id:      str,
        session_id:   str,
        title:        str,
        workspace:    Optional[str],
        message:      dict,
    ) -> None:
        """Upsert session metadata and append one new message."""
        await self.upsert_session(
            user_id    = user_id,
            session_id = session_id,
            title      = title,
            workspace  = workspace,
        )
        self.db.add(ExtensionChatMessage(
            id            = _uuid(),
            session_id_fk = session_pk,
            role          = message.get("role", "user"),
            content       = message.get("text", ""),
            context_file  = message.get("contextFile"),
            ts            = message.get("timestamp"),
            created_at    = _now(),
        ))
        await self.db.flush()

    # ── Rename a session ──────────────────────────────────────────────────────

    async def rename_session(self, user_id: str, session_id: str, title: str) -> bool:
        session = await self.get_session(user_id, session_id)
        if not session:
            return False
        session.title      = title
        session.updated_at = _now()
        await self.db.flush()
        return True

    # ── Delete a session (cascade removes messages) ───────────────────────────

    async def delete_session(self, user_id: str, session_id: str) -> bool:
        session = await self.get_session(user_id, session_id)
        if not session:
            return False
        await self.db.delete(session)
        await self.db.flush()
        return True
