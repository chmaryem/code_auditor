from __future__ import annotations
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import Project, User


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _path_hash(local_path: str) -> str:
    resolved = str(Path(local_path or ".").resolve()).lower()
    return hashlib.sha256(resolved.encode()).hexdigest()[:12]


class UserRepo:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_by_id(self, user_id: str) -> Optional[User]:
        result = await self.db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[User]:
        result = await self.db.execute(
            select(User).where(User.email == email.strip().lower())
        )
        return result.scalar_one_or_none()

    async def upsert(
        self,
        email: str,
        name: Optional[str] = None,
        role: str = "Developer",
    ) -> User:
        email = email.strip().lower()
        user = await self.get_by_email(email)
        now = _now()

        if user:
            user.last_login_at = now
            if name:
                user.name = name
            await self.db.flush()
            return user

        user = User(
            email=email,
            name=name or email.split("@", 1)[0],
            role=role,
            is_active=True,
            created_at=now,
            last_login_at=now,
        )
        self.db.add(user)
        await self.db.flush() 
        return user

    async def deactivate(self, user_id: str) -> None:
        await self.db.execute(
            update(User).where(User.id == user_id).values(is_active=False)
        )

    async def get_or_create_project(
        self,
        owner_id: str,
        local_path: str,
        name: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Project:
        ph = _path_hash(local_path)
        result = await self.db.execute(
            select(Project).where(
                Project.owner_id == owner_id,
                Project.path_hash == ph,
            )
        )
        project = result.scalar_one_or_none()

        if project:
            project.last_active_at = _now()
            if language and not project.language:
                project.language = language
            await self.db.flush()
            return project

        project = Project(
            owner_id=owner_id,
            name=name or Path(local_path).name,
            path_hash=ph,
            local_path=local_path,
            language=language,
            created_at=_now(),
            last_active_at=_now(),
        )
        self.db.add(project)
        await self.db.flush()
        return project

    async def get_project_by_path_hash(
        self, owner_id: str, path_hash: str
    ) -> Optional[Project]:
        result = await self.db.execute(
            select(Project).where(
                Project.owner_id == owner_id,
                Project.path_hash == path_hash,
            )
        )
        return result.scalar_one_or_none()


    async def set_active_repo_by_email(self, email: str, github_repo: str) -> None:
        """Persist owner/repo selection for a user (looked up by email)."""
        await self.db.execute(
            update(User)
            .where(User.email == email.strip().lower())
            .values(active_github_repo=github_repo.strip() or None)
        )

    async def get_active_repo_by_email(self, email: str) -> Optional[str]:
        """Return the stored owner/repo string for a user, or None."""
        result = await self.db.execute(
            select(User.active_github_repo).where(
                User.email == email.strip().lower()
            )
        )
        return result.scalar_one_or_none()

    async def set_github_token_by_email(self, email: str, token: Optional[str]) -> None:
        """Persist a GitHub OAuth access token for a user."""
        await self.db.execute(
            update(User)
            .where(User.email == email.strip().lower())
            .values(github_token=token)
        )

    async def get_github_token_by_email(self, email: str) -> Optional[str]:
        """Return the stored GitHub OAuth token for a user, or None."""
        result = await self.db.execute(
            select(User.github_token).where(
                User.email == email.strip().lower()
            )
        )
        return result.scalar_one_or_none()
