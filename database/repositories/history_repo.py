from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import (
    Conversation,
    CICDReport,
    DashboardFix,
    GitReport,
    HistoryEvent,
    Project,
    ProjectMemoryFact,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    import uuid
    return uuid.uuid4().hex


class HistoryEventRepo:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(
        self,
        user_id: str,
        event_type: str,
        source_module: str,
        title: str,
        project_id: Optional[str] = None,
        source_id: Optional[str] = None,
        summary: Optional[str] = None,
        severity: str = "info",
        status: str = "completed",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HistoryEvent:
        event = HistoryEvent(
            id=_uuid(),
            user_id=user_id,
            project_id=project_id,
            event_type=event_type,
            source_module=source_module,
            source_id=source_id,
            title=title,
            summary=summary,
            severity=severity,
            status=status,
            metadata_=metadata,
            created_at=_now(),
        )
        self.db.add(event)
        await self.db.flush()
        return event

    async def list_for_user(
        self,
        user_id: str,
        project_id: Optional[str] = None,
        source_module: Optional[str] = None,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        search: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 50,
        cursor: Optional[datetime] = None,
    ) -> List[HistoryEvent]:
        q = select(HistoryEvent).where(HistoryEvent.user_id == user_id)

        if project_id:
            q = q.where(HistoryEvent.project_id == project_id)
        if source_module and source_module != "all":
            q = q.where(HistoryEvent.source_module == source_module)
        if status and status != "all":
            q = q.where(HistoryEvent.status == status)
        if severity and severity != "all":
            q = q.where(HistoryEvent.severity == severity)
        if search:
            pattern = f"%{search}%"
            from sqlalchemy import or_
            q = q.where(
                or_(
                    HistoryEvent.title.ilike(pattern),
                    HistoryEvent.summary.ilike(pattern),
                )
            )
        if date_from:
            q = q.where(HistoryEvent.created_at >= date_from)
        if date_to:
            q = q.where(HistoryEvent.created_at <= date_to)
        if cursor:
            q = q.where(HistoryEvent.created_at < cursor)

        q = q.order_by(desc(HistoryEvent.created_at)).limit(limit)
        result = await self.db.execute(q)
        return list(result.scalars().all())

    async def get_by_id(self, event_id: str, user_id: str) -> Optional[HistoryEvent]:
        result = await self.db.execute(
            select(HistoryEvent).where(
                HistoryEvent.id == event_id,
                HistoryEvent.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_summary(self, user_id: str) -> Dict[str, Any]:
        """Aggregated stats for the History header cards."""
        from datetime import timedelta

        week_ago = _now() - timedelta(days=7)

        conversations_count = await self.db.scalar(
            select(func.count()).select_from(Conversation).where(
                Conversation.user_id == user_id,
                Conversation.deleted_at.is_(None),
            )
        ) or 0

        conversations_week = await self.db.scalar(
            select(func.count()).select_from(Conversation).where(
                Conversation.user_id == user_id,
                Conversation.deleted_at.is_(None),
                Conversation.created_at >= week_ago,
            )
        ) or 0

        # Project ids owned by this user (needed for joining reports)
        owned_project_ids_result = await self.db.execute(
            select(Project.id).where(Project.owner_id == user_id)
        )
        owned_ids = [row[0] for row in owned_project_ids_result.all()]

        cicd_count = 0
        cicd_week = 0
        git_count = 0
        git_week = 0
        if owned_ids:
            cicd_count = await self.db.scalar(
                select(func.count()).select_from(CICDReport).where(
                    CICDReport.project_id.in_(owned_ids)
                )
            ) or 0
            cicd_week = await self.db.scalar(
                select(func.count()).select_from(CICDReport).where(
                    CICDReport.project_id.in_(owned_ids),
                    CICDReport.created_at >= week_ago,
                )
            ) or 0
            git_count = await self.db.scalar(
                select(func.count()).select_from(GitReport).where(
                    GitReport.project_id.in_(owned_ids)
                )
            ) or 0
            git_week = await self.db.scalar(
                select(func.count()).select_from(GitReport).where(
                    GitReport.project_id.in_(owned_ids),
                    GitReport.created_at >= week_ago,
                )
            ) or 0

        fixes_count = await self.db.scalar(
            select(func.count()).select_from(DashboardFix).where(
                DashboardFix.user_id == user_id
            )
        ) or 0

        fixes_week = await self.db.scalar(
            select(func.count()).select_from(DashboardFix).where(
                DashboardFix.user_id == user_id,
                DashboardFix.created_at >= week_ago,
            )
        ) or 0

        # open_issues = CI/CD failures + blocked/needs-changes git reviews
        cicd_failures = 0
        git_blocked   = 0
        if owned_ids:
            cicd_failures = await self.db.scalar(
                select(func.count()).select_from(CICDReport).where(
                    CICDReport.project_id.in_(owned_ids),
                    CICDReport.outcome == "failure",
                )
            ) or 0
            git_blocked = await self.db.scalar(
                select(func.count()).select_from(GitReport).where(
                    GitReport.project_id.in_(owned_ids),
                    GitReport.verdict.in_(["BLOCKED", "NEEDS_CHANGES"]),
                )
            ) or 0
        open_issues = cicd_failures + git_blocked

        # last_activity = most recent timestamp across all domain tables
        conv_max: Optional[datetime] = await self.db.scalar(
            select(func.max(Conversation.updated_at)).where(
                Conversation.user_id == user_id,
                Conversation.deleted_at.is_(None),
            )
        )
        cicd_max: Optional[datetime] = None
        git_max:  Optional[datetime] = None
        if owned_ids:
            cicd_max = await self.db.scalar(
                select(func.max(CICDReport.created_at)).where(
                    CICDReport.project_id.in_(owned_ids)
                )
            )
            git_max = await self.db.scalar(
                select(func.max(GitReport.created_at)).where(
                    GitReport.project_id.in_(owned_ids)
                )
            )
        event_max: Optional[datetime] = await self.db.scalar(
            select(func.max(HistoryEvent.created_at)).where(
                HistoryEvent.user_id == user_id
            )
        )
        candidates = [t for t in [conv_max, cicd_max, git_max, event_max] if t is not None]
        last_ts = max(candidates) if candidates else None
        last_activity = last_ts.isoformat() if last_ts else None

        return {
            "conversations_count": conversations_count,
            "cicd_analyses_count": cicd_count,
            "git_reviews_count": git_count,
            "fixes_count": fixes_count,
            "open_issues_count": open_issues,
            "last_activity": last_activity,
            "conversations_this_week": conversations_week,
            "cicd_this_week": cicd_week,
            "git_reviews_this_week": git_week,
            "fixes_this_week": fixes_week,
        }


class DashboardFixRepo:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create(
        self,
        user_id: str,
        source_module: str,
        title: str,
        project_id: Optional[str] = None,
        source_id: Optional[str] = None,
        description: Optional[str] = None,
        files_changed: Optional[list] = None,
        status: str = "applied",
    ) -> DashboardFix:
        now = _now()
        fix = DashboardFix(
            id=_uuid(),
            user_id=user_id,
            project_id=project_id,
            source_module=source_module,
            source_id=source_id,
            title=title,
            description=description,
            status=status,
            files_changed=files_changed,
            applied_at=now if status == "applied" else None,
            created_at=now,
        )
        self.db.add(fix)
        await self.db.flush()
        return fix

    async def list_for_user(
        self,
        user_id: str,
        limit: int = 50,
        cursor: Optional[datetime] = None,
    ) -> List[DashboardFix]:
        q = select(DashboardFix).where(DashboardFix.user_id == user_id)
        if cursor:
            q = q.where(DashboardFix.created_at < cursor)
        q = q.order_by(desc(DashboardFix.created_at)).limit(limit)
        result = await self.db.execute(q)
        return list(result.scalars().all())


class ProjectMemoryFactRepo:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def upsert(
        self,
        user_id: str,
        fact_type: str,
        content: str,
        project_id: Optional[str] = None,
        confidence: float = 1.0,
        source_module: Optional[str] = None,
    ) -> ProjectMemoryFact:
        result = await self.db.execute(
            select(ProjectMemoryFact).where(
                ProjectMemoryFact.user_id == user_id,
                ProjectMemoryFact.fact_type == fact_type,
                ProjectMemoryFact.content == content,
            )
        )
        fact = result.scalar_one_or_none()
        if fact:
            fact.confidence = confidence
            fact.updated_at = _now()
            await self.db.flush()
            return fact

        fact = ProjectMemoryFact(
            id=_uuid(),
            user_id=user_id,
            project_id=project_id,
            fact_type=fact_type,
            content=content,
            confidence=confidence,
            source_module=source_module,
            created_at=_now(),
            updated_at=_now(),
        )
        self.db.add(fact)
        await self.db.flush()
        return fact

    async def list_for_user(
        self,
        user_id: str,
        fact_type: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[ProjectMemoryFact]:
        q = select(ProjectMemoryFact).where(ProjectMemoryFact.user_id == user_id)
        if fact_type:
            q = q.where(ProjectMemoryFact.fact_type == fact_type)
        if project_id:
            q = q.where(ProjectMemoryFact.project_id == project_id)
        q = q.order_by(desc(ProjectMemoryFact.updated_at))
        result = await self.db.execute(q)
        return list(result.scalars().all())
