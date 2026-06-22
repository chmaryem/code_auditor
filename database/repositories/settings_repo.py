"""
database/repositories/settings_repo.py — Settings Center persistence layer.

All settings tables follow an upsert pattern (one row per user).
Audit logging is appended on every mutation.
Secrets are never stored here — only metadata / status flags.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Type, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import (
    AISettings,
    CICDSettings,
    GitSettings,
    SecuritySettings,
    SettingsAuditLog,
    UserSettings,
    WorkspaceSettings,
)

_SENSITIVE_KEYS = frozenset(
    {"github_token", "sonar_token", "docker_password", "ghcr_token", "deploy_token"}
)

T = TypeVar(
    "T",
    UserSettings,
    WorkspaceSettings,
    AISettings,
    GitSettings,
    CICDSettings,
    SecuritySettings,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return uuid.uuid4().hex


def _mask(key: str, value: Any) -> Any:
    """Redact sensitive values before writing to audit log."""
    if key.lower() in _SENSITIVE_KEYS:
        return "••••••"
    return value


class SettingsRepo:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # ── Generic upsert ────────────────────────────────────────────────────────

    async def _upsert(
        self,
        model_cls: Type[T],
        user_id: str,
        scope: str,
        updates: Dict[str, Any],
    ) -> T:
        result = await self.db.execute(
            select(model_cls).where(model_cls.user_id == user_id)  # type: ignore[attr-defined]
        )
        obj = result.scalar_one_or_none()

        if obj is None:
            obj = model_cls(id=_uuid(), user_id=user_id)  # type: ignore[call-arg]
            self.db.add(obj)

        old_snapshot: Dict[str, Any] = {}
        for key, value in updates.items():
            if hasattr(obj, key):
                old_snapshot[key] = _mask(key, getattr(obj, key))
                setattr(obj, key, value)

        obj.updated_at = _now()  # type: ignore[attr-defined]
        await self.db.flush()

        await self._audit(user_id, scope, old_snapshot, {k: _mask(k, v) for k, v in updates.items()})
        return obj

    async def _audit(
        self,
        user_id: str,
        scope: str,
        old: Dict[str, Any],
        new: Dict[str, Any],
    ) -> None:
        for key in new:
            entry = SettingsAuditLog(
                id=_uuid(),
                user_id=user_id,
                setting_scope=scope,
                setting_key=key,
                old_value={"v": old.get(key)},
                new_value={"v": new.get(key)},
                changed_at=_now(),
            )
            self.db.add(entry)
        await self.db.flush()

    # ── User settings ─────────────────────────────────────────────────────────

    async def get_user_settings(self, user_id: str) -> Optional[UserSettings]:
        result = await self.db.execute(
            select(UserSettings).where(UserSettings.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def upsert_user_settings(
        self, user_id: str, updates: Dict[str, Any]
    ) -> UserSettings:
        return await self._upsert(UserSettings, user_id, "user", updates)

    # ── Workspace settings ────────────────────────────────────────────────────

    async def get_workspace_settings(self, user_id: str) -> Optional[WorkspaceSettings]:
        result = await self.db.execute(
            select(WorkspaceSettings).where(WorkspaceSettings.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def upsert_workspace_settings(
        self, user_id: str, updates: Dict[str, Any]
    ) -> WorkspaceSettings:
        return await self._upsert(WorkspaceSettings, user_id, "workspace", updates)

    # ── AI settings ───────────────────────────────────────────────────────────

    async def get_ai_settings(self, user_id: str) -> Optional[AISettings]:
        result = await self.db.execute(
            select(AISettings).where(AISettings.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def upsert_ai_settings(
        self, user_id: str, updates: Dict[str, Any]
    ) -> AISettings:
        return await self._upsert(AISettings, user_id, "ai", updates)

    # ── Git settings ──────────────────────────────────────────────────────────

    async def get_git_settings(self, user_id: str) -> Optional[GitSettings]:
        result = await self.db.execute(
            select(GitSettings).where(GitSettings.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def upsert_git_settings(
        self, user_id: str, updates: Dict[str, Any]
    ) -> GitSettings:
        return await self._upsert(GitSettings, user_id, "git", updates)

    # ── CI/CD settings ────────────────────────────────────────────────────────

    async def get_cicd_settings(self, user_id: str) -> Optional[CICDSettings]:
        result = await self.db.execute(
            select(CICDSettings).where(CICDSettings.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def upsert_cicd_settings(
        self, user_id: str, updates: Dict[str, Any]
    ) -> CICDSettings:
        return await self._upsert(CICDSettings, user_id, "cicd", updates)

    # ── Security settings ─────────────────────────────────────────────────────

    async def get_security_settings(self, user_id: str) -> Optional[SecuritySettings]:
        result = await self.db.execute(
            select(SecuritySettings).where(SecuritySettings.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def upsert_security_settings(
        self, user_id: str, updates: Dict[str, Any]
    ) -> SecuritySettings:
        return await self._upsert(SecuritySettings, user_id, "security", updates)

    # ── Reset helpers ─────────────────────────────────────────────────────────

    async def reset_user_settings(self, user_id: str) -> UserSettings:
        return await self._upsert(
            UserSettings,
            user_id,
            "user",
            {
                "theme": "dark",
                "language": "en",
                "density": "comfortable",
                "default_landing_page": "assistant",
                "show_ai_context_panel": True,
                "show_sidebar_labels": True,
                "show_cicd_in_sidebar": True,
                "notifications": {"enabled": True, "analysis": True, "cicd": True, "git": True},
            },
        )

    async def update_cicd_secret_status(
        self,
        user_id: str,
        name: str,
        status: str,
        updated_at: Optional[datetime] = None,
    ) -> None:
        """Record a secret's status in required_secrets JSONB.
        Never stores the raw value — only a status flag and ISO timestamp.
        """
        result = await self.db.execute(
            select(CICDSettings).where(CICDSettings.user_id == user_id)
        )
        obj = result.scalar_one_or_none()

        if obj is None:
            obj = CICDSettings(id=_uuid(), user_id=user_id, required_secrets={})
            self.db.add(obj)

        ts = (updated_at or _now()).isoformat()
        current = dict(obj.required_secrets or {})
        current[name] = {"status": status, "updated_at": ts}
        # Reassign to ensure SQLAlchemy detects the JSONB mutation
        obj.required_secrets = current
        obj.updated_at = _now()
        await self.db.flush()

        await self._audit(
            user_id,
            "cicd.secrets",
            {name: "[previous]"},
            {name: f"[{status}]"},
        )

    async def reset_ai_settings(self, user_id: str) -> AISettings:
        return await self._upsert(
            AISettings,
            user_id,
            "ai",
            {
                "ai_mode": "balanced",
                "response_style": "detailed",
                "use_repository_context": True,
                "use_conversation_memory": True,
                "use_rag": True,
                "use_cicd_context": False,
                "use_smart_git_context": False,
                "streaming_enabled": True,
                "proactive_enabled": False,
                "temperature": 0.3,
                "max_context_size": 8000,
            },
        )
