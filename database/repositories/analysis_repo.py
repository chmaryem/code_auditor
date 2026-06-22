from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import AnalysisRun, CICDReport, GitReport, TestGenerationRun


def _now() -> datetime:
    return datetime.now(timezone.utc)


class AnalysisRunRepo:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def save(
        self,
        project_id: str,
        file_path: str,
        content_hash: str,
        language: Optional[str] = None,
        score: Optional[int] = None,
        issues: Optional[list] = None,
        fixes: Optional[list] = None,
        strategy: Optional[str] = None,
        source: str = "watch",
    ) -> AnalysisRun:
        run = AnalysisRun(
            project_id=project_id,
            file_path=file_path,
            content_hash=content_hash,
            language=language,
            score=score,
            issues=issues,
            fixes=fixes,
            strategy=strategy,
            source=source,
            created_at=_now(),
        )
        self.db.add(run)
        await self.db.flush()
        return run

    async def latest_for_file(
        self, project_id: str, file_path: str
    ) -> Optional[AnalysisRun]:
        result = await self.db.execute(
            select(AnalysisRun)
            .where(
                AnalysisRun.project_id == project_id,
                AnalysisRun.file_path == file_path,
            )
            .order_by(desc(AnalysisRun.created_at))
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def history_for_project(
        self, project_id: str, limit: int = 100
    ) -> List[AnalysisRun]:
        result = await self.db.execute(
            select(AnalysisRun)
            .where(AnalysisRun.project_id == project_id)
            .order_by(desc(AnalysisRun.created_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_content_hash(
        self, project_id: str, content_hash: str
    ) -> Optional[AnalysisRun]:
        """Used by cache-hit path: if same content_hash exists, skip re-analysis."""
        result = await self.db.execute(
            select(AnalysisRun)
            .where(
                AnalysisRun.project_id == project_id,
                AnalysisRun.content_hash == content_hash,
            )
            .order_by(desc(AnalysisRun.created_at))
            .limit(1)
        )
        return result.scalar_one_or_none()


class GitReportRepo:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def save(
        self,
        project_id: str,
        report_type: str,
        branch: Optional[str] = None,
        base_branch: Optional[str] = None,
        pr_number: Optional[int] = None,
        verdict: Optional[str] = None,
        total_score: Optional[int] = None,
        summary: Optional[str] = None,
        raw_data: Optional[Dict[str, Any]] = None,
    ) -> GitReport:
        report = GitReport(
            project_id=project_id,
            report_type=report_type,
            branch=branch,
            base_branch=base_branch,
            pr_number=pr_number,
            verdict=verdict,
            total_score=total_score,
            summary=summary,
            raw_data=raw_data,
            created_at=_now(),
        )
        self.db.add(report)
        await self.db.flush()
        return report

    async def history_for_project(
        self,
        project_id: str,
        report_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[GitReport]:
        q = select(GitReport).where(GitReport.project_id == project_id)
        if report_type:
            q = q.where(GitReport.report_type == report_type)
        q = q.order_by(desc(GitReport.created_at)).limit(limit)
        result = await self.db.execute(q)
        return list(result.scalars().all())


class CICDReportRepo:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def save(
        self,
        project_id: str,
        repo: Optional[str] = None,
        outcome: Optional[str] = None,
        failure_type: Optional[str] = None,
        stage_failed: Optional[str] = None,
        root_cause: Optional[str] = None,
        pr_number: Optional[int] = None,
        elapsed_seconds: Optional[float] = None,
        raw_data: Optional[Dict[str, Any]] = None,
    ) -> CICDReport:
        report = CICDReport(
            project_id=project_id,
            repo=repo,
            outcome=outcome,
            failure_type=failure_type,
            stage_failed=stage_failed,
            root_cause=root_cause,
            pr_number=pr_number,
            elapsed_seconds=elapsed_seconds,
            raw_data=raw_data,
            created_at=_now(),
        )
        self.db.add(report)
        await self.db.flush()
        return report

    async def history_for_project(
        self, project_id: str, limit: int = 50
    ) -> List[CICDReport]:
        result = await self.db.execute(
            select(CICDReport)
            .where(CICDReport.project_id == project_id)
            .order_by(desc(CICDReport.created_at))
            .limit(limit)
        )
        return list(result.scalars().all())
