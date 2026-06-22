from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from database.models import Conversation, Message


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ConversationRepo:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # ── Conversations ────────────────────────────────────────────────────────

    async def get_by_session_id(self, session_id: str) -> Optional[Conversation]:
        result = await self.db.execute(
            select(Conversation).where(Conversation.session_id == session_id)
        )
        return result.scalar_one_or_none()

    async def get_or_create(
        self,
        session_id: str,
        user_id: str,
        project_id: Optional[str] = None,
        title: str = "New conversation",
        intent: Optional[str] = None,
    ) -> Conversation:
        conv = await self.get_by_session_id(session_id)
        if conv:
            return conv

        conv = Conversation(
            session_id=session_id,
            user_id=user_id,
            project_id=project_id,
            title=title,
            intent=intent,
            created_at=_now(),
            updated_at=_now(),
        )
        self.db.add(conv)
        await self.db.flush()
        return conv

    async def update_meta(
        self,
        session_id: str,
        title: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> None:
        values: Dict[str, Any] = {"updated_at": _now()}
        if title is not None:
            values["title"] = title
        if intent is not None:
            values["intent"] = intent
        await self.db.execute(
            update(Conversation)
            .where(Conversation.session_id == session_id)
            .values(**values)
        )

    async def list_for_user(
        self,
        user_id: str,
        project_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Conversation]:
        q = select(Conversation).where(Conversation.user_id == user_id)
        if project_id:
            q = q.where(Conversation.project_id == project_id)
        q = q.order_by(desc(Conversation.updated_at)).limit(limit)
        result = await self.db.execute(q)
        return list(result.scalars().all())

    # ── Messages ─────────────────────────────────────────────────────────────

    async def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            metadata_=metadata,
            created_at=_now(),
        )
        self.db.add(msg)
        # bump conversation turn_count + updated_at
        await self.db.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(
                turn_count=Conversation.turn_count + 1,
                updated_at=_now(),
            )
        )
        await self.db.flush()
        return msg

    async def load_messages(
        self,
        conversation_id: str,
        limit: int = 50,
    ) -> List[Message]:
        result = await self.db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def save_exchange(
        self,
        session_id: str,
        user_id: str,
        user_message: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None,
        title: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> Conversation:
        """
        Persist one complete user↔assistant exchange.
        Creates the conversation row if it does not exist yet.
        This is the primary write path called after every chat turn.
        """
        conv = await self.get_or_create(
            session_id=session_id,
            user_id=user_id,
            project_id=project_id,
            title=title or "New conversation",
            intent=intent,
        )

        # Update title/intent if provided
        if title or intent:
            await self.update_meta(session_id, title=title, intent=intent)

        await self.append_message(conv.id, "user", user_message, metadata)
        await self.append_message(conv.id, "assistant", assistant_response, metadata)
        return conv
