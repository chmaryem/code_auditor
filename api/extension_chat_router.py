"""
api/extension_chat_router.py — VS Code extension chat history endpoints.

These endpoints are intentionally separate from /api/history/conversations
(which belongs to the dashboard + Redis-coupled Conversation model).

Endpoints:
  POST   /extension/chat/sync                  → upsert session + replace messages
  GET    /extension/chat/sessions              → list user's sessions (most recent first)
  GET    /extension/chat/sessions/{session_id} → get session + messages
  DELETE /extension/chat/sessions/{session_id} → delete session
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from auth.security import Principal, get_current_user
from database.connection import get_db
from database.models import ExtensionChatMessage
from database.repositories.extension_chat_repo import ExtensionChatRepo

logger = logging.getLogger(__name__)


async def _resolve_pg_user_id(principal: Principal, db: AsyncSession) -> str:
    """Return the PostgreSQL users.id for the current principal.

    principal.id is the Redis-generated auth ID (JWT sub), which never
    appears in the PG users table. We resolve the real PG id via email,
    creating the row on first use so the FK constraint is always satisfied.
    """
    from database.repositories.user_repo import UserRepo
    repo = UserRepo(db)
    user = await repo.get_by_email(principal.email)
    if user:
        return user.id
    # First time this user hits a PG-backed endpoint: mirror them now.
    user = await repo.upsert(email=principal.email, role=principal.role)
    return user.id


extension_chat_router = APIRouter(
    prefix="/extension/chat",
    tags=["Extension Chat"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class MessageIn(BaseModel):
    role:        str
    text:        str
    timestamp:   Optional[int]  = None
    contextFile: Optional[str]  = None


class SyncRequest(BaseModel):
    session_id: str              = Field(..., min_length=1, max_length=64)
    title:      str              = Field("New chat", max_length=300)
    workspace:  Optional[str]   = Field(None, max_length=500)
    messages:   List[MessageIn] = Field(default_factory=list)


class SyncResponse(BaseModel):
    session_id:    str
    message_count: int


class SessionOut(BaseModel):
    session_id:    str
    title:         str
    workspace:     Optional[str]
    message_count: int
    created_at:    str
    updated_at:    str


class MessageOut(BaseModel):
    role:         str
    text:         str
    timestamp:    Optional[int]
    context_file: Optional[str]


class SessionDetailOut(BaseModel):
    session_id: str
    title:      str
    workspace:  Optional[str]
    messages:   List[MessageOut]
    created_at: str
    updated_at: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@extension_chat_router.post("/sync", response_model=SyncResponse)
async def sync_session(
    body:      SyncRequest,
    principal: Principal  = Depends(get_current_user),
    db:        AsyncSession = Depends(get_db),
) -> SyncResponse:
    """Upsert a chat session and replace all its messages (full snapshot)."""
    try:
        pg_user_id = await _resolve_pg_user_id(principal, db)
        repo    = ExtensionChatRepo(db)
        session = await repo.upsert_session(
            user_id    = pg_user_id,
            session_id = body.session_id,
            title      = body.title or "New chat",
            workspace  = body.workspace,
        )
        await repo.replace_messages(
            session_pk = session.id,
            messages   = [m.model_dump() for m in body.messages],
        )
        await db.commit()
    except Exception as exc:
        await db.rollback()
        logger.warning("sync_session: DB unavailable (%s) — skipping persist", exc)
    return SyncResponse(session_id=body.session_id, message_count=len(body.messages))


@extension_chat_router.get("/sessions", response_model=List[SessionOut])
async def list_sessions(
    principal: Principal    = Depends(get_current_user),
    db:        AsyncSession = Depends(get_db),
) -> List[SessionOut]:
    """List the user's extension chat sessions, most recent first."""
    try:
        pg_user_id = await _resolve_pg_user_id(principal, db)
        repo     = ExtensionChatRepo(db)
        sessions = await repo.list_sessions(pg_user_id)
    except Exception as exc:
        await db.rollback()
        logger.warning("list_sessions: DB unavailable (%s) — returning []", exc)
        return []

    out = []
    for s in sessions:
        msg_count = len(s.messages) if s.messages is not None else 0
        out.append(SessionOut(
            session_id    = s.session_id,
            title         = s.title,
            workspace     = s.workspace,
            message_count = msg_count,
            created_at    = s.created_at.isoformat(),
            updated_at    = s.updated_at.isoformat(),
        ))
    return out


@extension_chat_router.get("/sessions/{session_id}", response_model=SessionDetailOut)
async def get_session(
    session_id: str,
    principal:  Principal    = Depends(get_current_user),
    db:         AsyncSession = Depends(get_db),
) -> SessionDetailOut:
    """Return a session with all its messages."""
    try:
        pg_user_id = await _resolve_pg_user_id(principal, db)
        repo    = ExtensionChatRepo(db)
        session = await repo.get_session(pg_user_id, session_id)
    except Exception as exc:
        await db.rollback()
        logger.warning("get_session: DB unavailable (%s)", exc)
        raise HTTPException(status_code=503, detail="Database temporarily unavailable")
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = [
        MessageOut(
            role         = m.role,
            text         = m.content,
            timestamp    = m.ts,
            context_file = m.context_file,
        )
        for m in (session.messages or [])
    ]
    return SessionDetailOut(
        session_id = session.session_id,
        title      = session.title,
        workspace  = session.workspace,
        messages   = messages,
        created_at = session.created_at.isoformat(),
        updated_at = session.updated_at.isoformat(),
    )


class AppendRequest(BaseModel):
    session_id: str            = Field(..., min_length=1, max_length=64)
    title:      str            = Field("New chat", max_length=300)
    workspace:  Optional[str] = Field(None, max_length=500)
    message:    MessageIn


@extension_chat_router.post("/messages", response_model=SyncResponse)
async def append_message(
    body:      AppendRequest,
    principal: Principal    = Depends(get_current_user),
    db:        AsyncSession = Depends(get_db),
) -> SyncResponse:
    """Upsert session + append a single message. Efficient alternative to full sync."""
    try:
        pg_user_id = await _resolve_pg_user_id(principal, db)
        repo    = ExtensionChatRepo(db)
        session = await repo.upsert_session(
            user_id    = pg_user_id,
            session_id = body.session_id,
            title      = body.title,
            workspace  = body.workspace,
        )
        await repo.append_message(
            session_pk = session.id,
            user_id    = pg_user_id,
            session_id = body.session_id,
            title      = body.title,
            workspace  = body.workspace,
            message    = body.message.model_dump(),
        )
        from sqlalchemy import func, select as sa_select
        count_result = await db.scalar(
            sa_select(func.count()).select_from(ExtensionChatMessage)
            .where(ExtensionChatMessage.session_id_fk == session.id)
        )
        await db.commit()
        return SyncResponse(session_id=body.session_id, message_count=count_result or 0)
    except Exception as exc:
        await db.rollback()
        logger.warning("append_message: DB unavailable (%s) — skipping persist", exc)
        return SyncResponse(session_id=body.session_id, message_count=0)


class RenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)


@extension_chat_router.patch("/sessions/{session_id}", response_model=SyncResponse)
async def rename_session(
    session_id: str,
    body:       RenameRequest,
    principal:  Principal    = Depends(get_current_user),
    db:         AsyncSession = Depends(get_db),
) -> SyncResponse:
    """Rename a session title."""
    try:
        pg_user_id = await _resolve_pg_user_id(principal, db)
        repo    = ExtensionChatRepo(db)
        success = await repo.rename_session(pg_user_id, session_id, body.title)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        await db.commit()
    except HTTPException:
        raise
    except Exception as exc:
        await db.rollback()
        logger.warning("rename_session: DB unavailable (%s) — skipping", exc)
    return SyncResponse(session_id=session_id, message_count=0)


@extension_chat_router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    principal:  Principal    = Depends(get_current_user),
    db:         AsyncSession = Depends(get_db),
) -> None:
    """Delete a session and all its messages."""
    try:
        pg_user_id = await _resolve_pg_user_id(principal, db)
        repo    = ExtensionChatRepo(db)
        deleted = await repo.delete_session(pg_user_id, session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        await db.commit()
    except HTTPException:
        raise
    except Exception as exc:
        await db.rollback()
        logger.warning("delete_session: DB unavailable (%s) — skipping", exc)
