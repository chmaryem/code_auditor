"""
api/history_router.py — History & Developer Memory endpoints.

Endpoints:
  GET  /history/summary              → stats cards for the dashboard header
  GET  /history/timeline             → paginated unified event timeline
  GET  /history/conversations        → ChatAgent conversation list
  GET  /history/cicd                 → CI/CD analysis history
  GET  /history/git                  → Smart Git / PR review history
  GET  /history/fixes                → applied dashboard fixes
  GET  /history/memory               → project memory facts
  GET  /history/events/{id}          → event detail (resolves source object)
  POST /history/events               → create a new history event (used by modules)
  POST /history/memory/facts         → upsert a project memory fact
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.security import Principal, get_current_user
from database.connection import get_db
from database.models import (
    CICDReport,
    Conversation,
    DashboardFix,
    GitReport,
    HistoryEvent,
    Project,
    ProjectMemoryFact,
)
from database.repositories.history_repo import (
    DashboardFixRepo,
    HistoryEventRepo,
    ProjectMemoryFactRepo,
)
from services import history_cache_service as cache

logger = logging.getLogger(__name__)

history_router = APIRouter(prefix="/history", tags=["History"])


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class HistoryEventOut(BaseModel):
    id: str
    event_type: str
    source_module: str
    source_id: Optional[str]
    title: str
    summary: Optional[str]
    severity: str
    status: str
    metadata: Optional[Dict[str, Any]]
    created_at: str

    @classmethod
    def from_orm(cls, e: HistoryEvent) -> "HistoryEventOut":
        return cls(
            id=e.id,
            event_type=e.event_type,
            source_module=e.source_module,
            source_id=e.source_id,
            title=e.title,
            summary=e.summary,
            severity=e.severity,
            status=e.status,
            metadata=e.metadata_,
            created_at=e.created_at.isoformat(),
        )


class CreateEventRequest(BaseModel):
    event_type: str = Field(..., description="e.g. cicd_analyzed, git_pr_reviewed")
    source_module: str = Field(..., description="chat | cicd | smart_git | fixes")
    title: str
    project_id: Optional[str] = None
    source_id: Optional[str] = None
    summary: Optional[str] = None
    severity: str = "info"
    status: str = "completed"
    metadata: Optional[Dict[str, Any]] = None


class CreateMemoryFactRequest(BaseModel):
    fact_type: str = Field(..., description="tech_stack | recurring_issue | convention | team_preference")
    content: str
    project_id: Optional[str] = None
    confidence: float = 1.0
    source_module: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _isoformat_or_none(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


async def _get_pg_user_id(email: str, db: AsyncSession) -> Optional[str]:
    """Resolve principal email to PostgreSQL users.id (≠ Redis auth ID)."""
    from database.models import User
    result = await db.execute(select(User.id).where(User.email == email))
    return result.scalar_one_or_none()


async def _resolve_active_repo(principal: Principal, db: AsyncSession) -> Optional[str]:
    """Return owner/repo from users.active_github_repo."""
    from database.models import User
    result = await db.execute(
        select(User.active_github_repo).where(User.email == principal.email)
    )
    return result.scalar_one_or_none()


# ── GET /history/summary ──────────────────────────────────────────────────────

@history_router.get("/summary")
async def get_summary(
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = await _get_pg_user_id(principal.email, db)
    if not user_id:
        return {
            "conversations_count": 0, "cicd_analyses_count": 0,
            "git_reviews_count": 0, "fixes_count": 0,
            "open_issues_count": 0, "last_activity": None, "active_repository": None,
        }

    cached = cache.get_summary(user_id)
    if cached:
        return cached

    repo = HistoryEventRepo(db)
    data = await repo.get_summary(user_id)
    data["active_repository"] = await _resolve_active_repo(principal, db)

    cache.set_summary(user_id, data)
    return data


# ── GET /history/timeline ─────────────────────────────────────────────────────

@history_router.get("/timeline")
async def get_timeline(
    module: Optional[str]   = Query(None, description="all | chat | cicd | smart_git | fixes"),
    status: Optional[str]   = Query(None, description="all | completed | failed | pending"),
    severity: Optional[str] = Query(None),
    q: Optional[str]        = Query(None, description="Full-text search in title and summary"),
    date_from: Optional[str] = Query(None, description="ISO datetime lower bound"),
    date_to: Optional[str]   = Query(None, description="ISO datetime upper bound"),
    limit: int               = Query(30, ge=1, le=100),
    cursor: Optional[str]    = Query(None, description="ISO datetime — exclusive upper bound for pagination"),
    principal: Principal     = Depends(get_current_user),
    db: AsyncSession         = Depends(get_db),
):
    user_id = await _get_pg_user_id(principal.email, db)
    if not user_id:
        return {"events": [], "next_cursor": None, "has_more": False}

    params: Dict[str, Any] = {
        "module": module, "status": status, "severity": severity,
        "q": q, "date_from": date_from, "date_to": date_to,
        "limit": limit, "cursor": cursor,
    }

    cached = cache.get_timeline(user_id, params)
    if cached:
        return cached

    def _parse_dt(s: Optional[str]) -> Optional[datetime]:
        if not s:
            return None
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None

    repo = HistoryEventRepo(db)
    events = await repo.list_for_user(
        user_id=user_id,
        source_module=module if module and module != "all" else None,
        status=status if status and status != "all" else None,
        severity=severity if severity and severity != "all" else None,
        search=q or None,
        date_from=_parse_dt(date_from),
        date_to=_parse_dt(date_to),
        limit=limit + 1,
        cursor=_parse_dt(cursor),
    )

    has_more = len(events) > limit
    if has_more:
        events = events[:limit]

    next_cursor = events[-1].created_at.isoformat() if has_more and events else None

    result = {
        "events": [HistoryEventOut.from_orm(e).model_dump() for e in events],
        "next_cursor": next_cursor,
        "has_more": has_more,
    }

    cache.set_timeline(user_id, params, result)
    return result


# ── GET /history/conversations ────────────────────────────────────────────────

@history_router.get("/conversations")
async def get_conversations(
    limit: int            = Query(30, ge=1, le=100),
    cursor: Optional[str] = Query(None),
    principal: Principal  = Depends(get_current_user),
    db: AsyncSession      = Depends(get_db),
):
    user_id = await _get_pg_user_id(principal.email, db)
    if not user_id:
        return {"conversations": [], "next_cursor": None, "has_more": False}

    q = select(Conversation).where(
        Conversation.user_id == user_id,
        Conversation.deleted_at.is_(None),
    )
    if cursor:
        try:
            ts = datetime.fromisoformat(cursor.replace("Z", "+00:00"))
            q = q.where(Conversation.updated_at < ts)
        except ValueError:
            pass

    q = q.order_by(desc(Conversation.updated_at)).limit(limit + 1)
    result = await db.execute(q)
    rows = list(result.scalars().all())

    has_more = len(rows) > limit
    if has_more:
        rows = rows[:limit]

    return {
        "conversations": [
            {
                "id": c.id,
                "session_id": c.session_id,
                "title": c.title,
                "intent": c.intent,
                "turn_count": c.turn_count,
                "created_at": _isoformat_or_none(c.created_at),
                "updated_at": _isoformat_or_none(c.updated_at),
            }
            for c in rows
        ],
        "next_cursor": rows[-1].updated_at.isoformat() if has_more and rows else None,
        "has_more": has_more,
    }


# ── GET /history/cicd ─────────────────────────────────────────────────────────

@history_router.get("/cicd")
async def get_cicd_history(
    outcome: Optional[str] = Query(None, description="success | failure"),
    limit: int             = Query(30, ge=1, le=100),
    cursor: Optional[str]  = Query(None),
    principal: Principal   = Depends(get_current_user),
    db: AsyncSession       = Depends(get_db),
):
    pg_user_id = await _get_pg_user_id(principal.email, db)
    if not pg_user_id:
        return {"analyses": [], "next_cursor": None, "has_more": False}

    owned_ids_result = await db.execute(
        select(Project.id).where(Project.owner_id == pg_user_id)
    )
    owned_ids = [row[0] for row in owned_ids_result.all()]

    if not owned_ids:
        return {"analyses": [], "next_cursor": None, "has_more": False}

    q = select(CICDReport).where(CICDReport.project_id.in_(owned_ids))
    if outcome:
        q = q.where(CICDReport.outcome == outcome)
    if cursor:
        try:
            ts = datetime.fromisoformat(cursor.replace("Z", "+00:00"))
            q = q.where(CICDReport.created_at < ts)
        except ValueError:
            pass
    q = q.order_by(desc(CICDReport.created_at)).limit(limit + 1)
    result = await db.execute(q)
    rows = list(result.scalars().all())

    has_more = len(rows) > limit
    if has_more:
        rows = rows[:limit]

    return {
        "analyses": [
            {
                "id": r.id,
                "repo": r.repo,
                "outcome": r.outcome,
                "failure_type": r.failure_type,
                "stage_failed": r.stage_failed,
                "root_cause": r.root_cause,
                "pr_number": r.pr_number,
                "elapsed_seconds": r.elapsed_seconds,
                "created_at": _isoformat_or_none(r.created_at),
            }
            for r in rows
        ],
        "next_cursor": rows[-1].created_at.isoformat() if has_more and rows else None,
        "has_more": has_more,
    }


# ── GET /history/git ──────────────────────────────────────────────────────────

@history_router.get("/git")
async def get_git_history(
    report_type: Optional[str] = Query(None, description="branch | pr_review | commit_lint | secret_scan | test_impact"),
    limit: int                 = Query(30, ge=1, le=100),
    cursor: Optional[str]      = Query(None),
    principal: Principal       = Depends(get_current_user),
    db: AsyncSession           = Depends(get_db),
):
    pg_user_id = await _get_pg_user_id(principal.email, db)
    if not pg_user_id:
        return {"reviews": [], "next_cursor": None, "has_more": False}

    owned_ids_result = await db.execute(
        select(Project.id).where(Project.owner_id == pg_user_id)
    )
    owned_ids = [row[0] for row in owned_ids_result.all()]

    if not owned_ids:
        return {"reviews": [], "next_cursor": None, "has_more": False}

    q = select(GitReport).where(GitReport.project_id.in_(owned_ids))
    if report_type:
        q = q.where(GitReport.report_type == report_type)
    if cursor:
        try:
            ts = datetime.fromisoformat(cursor.replace("Z", "+00:00"))
            q = q.where(GitReport.created_at < ts)
        except ValueError:
            pass
    q = q.order_by(desc(GitReport.created_at)).limit(limit + 1)
    result = await db.execute(q)
    rows = list(result.scalars().all())

    has_more = len(rows) > limit
    if has_more:
        rows = rows[:limit]

    return {
        "reviews": [
            {
                "id": r.id,
                "report_type": r.report_type,
                "branch": r.branch,
                "base_branch": r.base_branch,
                "pr_number": r.pr_number,
                "verdict": r.verdict,
                "total_score": r.total_score,
                "summary": r.summary,
                "created_at": _isoformat_or_none(r.created_at),
            }
            for r in rows
        ],
        "next_cursor": rows[-1].created_at.isoformat() if has_more and rows else None,
        "has_more": has_more,
    }


# ── GET /history/fixes ────────────────────────────────────────────────────────

@history_router.get("/fixes")
async def get_fixes_history(
    limit: int            = Query(30, ge=1, le=100),
    cursor: Optional[str] = Query(None),
    principal: Principal  = Depends(get_current_user),
    db: AsyncSession      = Depends(get_db),
):
    repo = DashboardFixRepo(db)
    cursor_dt = None
    if cursor:
        try:
            cursor_dt = datetime.fromisoformat(cursor.replace("Z", "+00:00"))
        except ValueError:
            pass

    pg_user_id = await _get_pg_user_id(principal.email, db)
    if not pg_user_id:
        return {"fixes": [], "next_cursor": None, "has_more": False}

    rows = await repo.list_for_user(pg_user_id, limit=limit + 1, cursor=cursor_dt)
    has_more = len(rows) > limit
    if has_more:
        rows = rows[:limit]

    return {
        "fixes": [
            {
                "id": f.id,
                "source_module": f.source_module,
                "source_id": f.source_id,
                "title": f.title,
                "description": f.description,
                "status": f.status,
                "files_changed": f.files_changed,
                "applied_at": _isoformat_or_none(f.applied_at),
                "created_at": _isoformat_or_none(f.created_at),
            }
            for f in rows
        ],
        "next_cursor": rows[-1].created_at.isoformat() if has_more and rows else None,
        "has_more": has_more,
    }


# ── GET /history/memory ───────────────────────────────────────────────────────

@history_router.get("/memory")
async def get_memory(
    fact_type: Optional[str]  = Query(None),
    project_id: Optional[str] = Query(None),
    principal: Principal       = Depends(get_current_user),
    db: AsyncSession           = Depends(get_db),
):
    pg_user_id = await _get_pg_user_id(principal.email, db)
    if not pg_user_id:
        return {"facts": []}

    repo = ProjectMemoryFactRepo(db)
    facts = await repo.list_for_user(
        user_id=pg_user_id,
        fact_type=fact_type,
        project_id=project_id,
    )
    return {
        "facts": [
            {
                "id": f.id,
                "fact_type": f.fact_type,
                "content": f.content,
                "confidence": f.confidence,
                "source_module": f.source_module,
                "project_id": f.project_id,
                "created_at": _isoformat_or_none(f.created_at),
                "updated_at": _isoformat_or_none(f.updated_at),
            }
            for f in facts
        ]
    }


# ── GET /history/events/{id} ──────────────────────────────────────────────────

@history_router.get("/events/{event_id}")
async def get_event_detail(
    event_id: str,
    principal: Principal = Depends(get_current_user),
    db: AsyncSession     = Depends(get_db),
):
    pg_user_id = await _get_pg_user_id(principal.email, db)
    if not pg_user_id:
        raise HTTPException(404, "Event not found")

    repo = HistoryEventRepo(db)
    event = await repo.get_by_id(event_id, pg_user_id)
    if not event:
        raise HTTPException(404, "Event not found")

    out = HistoryEventOut.from_orm(event).model_dump()

    # Resolve source object for richer detail panel
    source_data: Optional[Dict[str, Any]] = None
    try:
        if event.source_id:
            if event.source_module == "cicd":
                r = await db.execute(
                    select(CICDReport).where(CICDReport.id == event.source_id)
                )
                cicd = r.scalar_one_or_none()
                if cicd:
                    source_data = {
                        "repo": cicd.repo,
                        "outcome": cicd.outcome,
                        "failure_type": cicd.failure_type,
                        "stage_failed": cicd.stage_failed,
                        "root_cause": cicd.root_cause,
                        "pr_number": cicd.pr_number,
                        "elapsed_seconds": cicd.elapsed_seconds,
                    }
            elif event.source_module == "smart_git":
                r = await db.execute(
                    select(GitReport).where(GitReport.id == event.source_id)
                )
                git = r.scalar_one_or_none()
                if git:
                    source_data = {
                        "report_type": git.report_type,
                        "branch": git.branch,
                        "base_branch": git.base_branch,
                        "pr_number": git.pr_number,
                        "verdict": git.verdict,
                        "total_score": git.total_score,
                        "summary": git.summary,
                    }
            elif event.source_module == "chat":
                r = await db.execute(
                    select(Conversation).where(Conversation.id == event.source_id)
                )
                conv = r.scalar_one_or_none()
                if conv:
                    source_data = {
                        "session_id": conv.session_id,
                        "title": conv.title,
                        "turn_count": conv.turn_count,
                        "intent": conv.intent,
                    }
            elif event.source_module == "fixes":
                r = await db.execute(
                    select(DashboardFix).where(DashboardFix.id == event.source_id)
                )
                fix = r.scalar_one_or_none()
                if fix:
                    source_data = {
                        "title": fix.title,
                        "description": fix.description,
                        "status": fix.status,
                        "files_changed": fix.files_changed,
                        "applied_at": _isoformat_or_none(fix.applied_at),
                    }
    except Exception as e:
        logger.warning("Could not resolve source_data for event %s: %s", event_id, e)

    out["source_data"] = source_data
    return out


# ── POST /history/events ──────────────────────────────────────────────────────

@history_router.post("/events", status_code=201)
async def create_event(
    req: CreateEventRequest,
    principal: Principal = Depends(get_current_user),
    db: AsyncSession     = Depends(get_db),
):
    pg_user_id = await _get_pg_user_id(principal.email, db)
    if not pg_user_id:
        raise HTTPException(404, "User not found in DB")

    repo = HistoryEventRepo(db)
    event = await repo.create(
        user_id=pg_user_id,
        event_type=req.event_type,
        source_module=req.source_module,
        title=req.title,
        project_id=req.project_id,
        source_id=req.source_id,
        summary=req.summary,
        severity=req.severity,
        status=req.status,
        metadata=req.metadata,
    )
    await db.commit()
    cache.invalidate(pg_user_id)
    return {"id": event.id, "created_at": event.created_at.isoformat()}


# ── POST /history/memory/facts ────────────────────────────────────────────────

@history_router.post("/memory/facts", status_code=201)
async def upsert_memory_fact(
    req: CreateMemoryFactRequest,
    principal: Principal = Depends(get_current_user),
    db: AsyncSession     = Depends(get_db),
):
    pg_user_id = await _get_pg_user_id(principal.email, db)
    if not pg_user_id:
        raise HTTPException(404, "User not found in DB")

    repo = ProjectMemoryFactRepo(db)
    fact = await repo.upsert(
        user_id=pg_user_id,
        fact_type=req.fact_type,
        content=req.content,
        project_id=req.project_id,
        confidence=req.confidence,
        source_module=req.source_module,
    )
    await db.commit()
    return {"id": fact.id, "updated_at": fact.updated_at.isoformat()}
