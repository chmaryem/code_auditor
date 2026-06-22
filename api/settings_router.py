"""
api/settings_router.py — Settings Center API (v1).

All endpoints require JWT authentication via the shared _auth dependency.
Secrets are NEVER returned in responses.
user_id is always resolved from the JWT principal — never trusted from the client.

Endpoints:
  GET  /api/settings                  → all user settings (aggregated)
  GET  /api/settings/status           → live system status
  PATCH /api/settings/user            → UI preferences
  PATCH /api/settings/workspace       → workspace / environment settings
  PATCH /api/settings/ai              → AI/ChatAgent behavior
  PATCH /api/settings/git             → Git integration
  PATCH /api/settings/cicd            → CI/CD configuration
  PATCH /api/settings/security        → session & auth preferences
  POST /api/settings/test/backend     → ping backend
  POST /api/settings/test/postgres    → ping PostgreSQL
  POST /api/settings/test/redis       → ping Redis
  POST /api/settings/test/github      → validate GitHub token
  POST /api/settings/cache/clear      → clear user Redis keys
  POST /api/settings/reset/user       → reset UI prefs to defaults
  POST /api/settings/reset/ai         → reset AI settings to defaults
  GET  /api/settings/export           → export settings JSON (no secrets)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession

from auth import get_current_user
from auth.security import Principal
from database.connection import get_db
from database.repositories.settings_repo import SettingsRepo
from database.repositories.user_repo import UserRepo

logger = logging.getLogger(__name__)

settings_router = APIRouter(prefix="/settings", tags=["settings"])


# ── Redis rate-limit helper ───────────────────────────────────────────────────

_RATE_LIMIT_TTL = 60   # seconds window
_RATE_LIMIT_MAX = 5    # max test calls per window per user per type


def _check_rate_limit(user_id: str, test_type: str) -> None:
    """Block if the user has exceeded test connection rate limit."""
    try:
        from services.mcp_redis_service import get_mcp_redis
        redis = get_mcp_redis()
        key = f"ca:ratelimit:settings_test:{user_id}:{test_type}"
        count_raw = redis.get(key)
        count = int(count_raw) if count_raw and str(count_raw).isdigit() else 0
        if count >= _RATE_LIMIT_MAX:
            raise HTTPException(
                429,
                detail=f"Rate limit exceeded. Max {_RATE_LIMIT_MAX} test requests per minute.",
            )
        redis.set(key, str(count + 1), ex=_RATE_LIMIT_TTL)
    except HTTPException:
        raise
    except Exception:
        pass  # Redis unavailable — allow through, don't block users


def _clear_settings_cache(user_id: str) -> int:
    """Delete all ca:settings:* Redis keys for a user. Returns count deleted."""
    try:
        from services.mcp_redis_service import get_mcp_redis
        redis = get_mcp_redis()
        pattern = f"ca:settings:*:{user_id}"
        keys = redis.keys(pattern)
        if keys:
            redis.delete(*keys)
        return len(keys) if keys else 0
    except Exception:
        return 0


# ── Pydantic response schemas ─────────────────────────────────────────────────

class UserSettingsOut(BaseModel):
    theme: str
    language: str
    density: str
    default_landing_page: str
    show_ai_context_panel: bool
    show_sidebar_labels: bool
    show_cicd_in_sidebar: bool
    notifications: Dict[str, Any]
    updated_at: Optional[datetime]


class WorkspaceSettingsOut(BaseModel):
    environment: str
    backend_url: Optional[str]
    websocket_url: Optional[str]
    default_branch: Optional[str]
    updated_at: Optional[datetime]


class AISettingsOut(BaseModel):
    ai_mode: str
    response_style: str
    use_repository_context: bool
    use_conversation_memory: bool
    use_rag: bool
    use_cicd_context: bool
    use_smart_git_context: bool
    streaming_enabled: bool
    proactive_enabled: bool
    temperature: float
    max_context_size: int
    updated_at: Optional[datetime]


class GitSettingsOut(BaseModel):
    provider: str
    default_branch: str
    safe_mode: bool
    pr_analysis_mode: str
    auto_refresh_prs: bool
    active_repo: Optional[str]
    updated_at: Optional[datetime]


class CICDSettingsOut(BaseModel):
    provider: str
    pipeline_template: str
    quality_gate_threshold: int
    auto_generate_pipeline: bool
    open_pr_after_generation: bool
    sonar_organization: Optional[str]
    sonar_project_key: Optional[str]
    docker_registry: str
    deploy_provider: str
    required_secrets: Dict[str, Any]
    updated_at: Optional[datetime]


class SecuritySettingsOut(BaseModel):
    auth_mode: str
    session_timeout: int
    auto_logout_enabled: bool
    mask_sensitive_values: bool
    updated_at: Optional[datetime]


class AllSettingsOut(BaseModel):
    user: UserSettingsOut
    workspace: WorkspaceSettingsOut
    ai: AISettingsOut
    git: GitSettingsOut
    cicd: CICDSettingsOut
    security: SecuritySettingsOut


class ComponentStatus(BaseModel):
    ok: bool
    label: str
    detail: Optional[str] = None
    latency_ms: Optional[float] = None


class SystemStatusOut(BaseModel):
    backend: ComponentStatus
    postgres: ComponentStatus
    redis: ComponentStatus
    github: ComponentStatus
    ai: ComponentStatus
    cicd: ComponentStatus
    active_repo: Optional[str]
    checked_at: datetime


class TestResult(BaseModel):
    ok: bool
    latency_ms: Optional[float] = None
    detail: Optional[str] = None


class SavedOut(BaseModel):
    saved: bool
    updated_at: datetime


# ── Pydantic input schemas with strict validation ─────────────────────────────

class PatchUserSettings(BaseModel):
    theme: Optional[str] = Field(None, pattern="^(dark|light|system)$")
    language: Optional[str] = Field(None, pattern="^(en|fr)$")
    density: Optional[str] = Field(None, pattern="^(comfortable|compact)$")
    default_landing_page: Optional[str] = Field(
        None, pattern="^(assistant|cicd|git|history|watch)$"
    )
    show_ai_context_panel: Optional[bool] = None
    show_sidebar_labels: Optional[bool] = None
    show_cicd_in_sidebar: Optional[bool] = None
    notifications: Optional[Dict[str, Any]] = None


class PatchWorkspaceSettings(BaseModel):
    environment: Optional[str] = Field(None, pattern="^(local|staging|production)$")
    backend_url: Optional[str] = None
    websocket_url: Optional[str] = None
    default_branch: Optional[str] = None


class PatchAISettings(BaseModel):
    ai_mode: Optional[str] = Field(
        None, pattern="^(fast|balanced|deep|strict)$"
    )
    response_style: Optional[str] = Field(
        None, pattern="^(concise|detailed|professional|step_by_step)$"
    )
    use_repository_context: Optional[bool] = None
    use_conversation_memory: Optional[bool] = None
    use_rag: Optional[bool] = None
    use_cicd_context: Optional[bool] = None
    use_smart_git_context: Optional[bool] = None
    streaming_enabled: Optional[bool] = None
    proactive_enabled: Optional[bool] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_context_size: Optional[int] = Field(None, ge=1000, le=32000)


class PatchGitSettings(BaseModel):
    default_branch: Optional[str] = None
    safe_mode: Optional[bool] = None
    pr_analysis_mode: Optional[str] = Field(None, pattern="^(fast|deep)$")
    auto_refresh_prs: Optional[bool] = None


class PatchCICDSettings(BaseModel):
    provider: Optional[str] = Field(
        None, pattern="^(github_actions|gitlab_ci|jenkins)$"
    )
    pipeline_template: Optional[str] = Field(
        None, pattern="^(minimal|standard|full)$"
    )
    quality_gate_threshold: Optional[int] = Field(None, ge=0, le=100)
    auto_generate_pipeline: Optional[bool] = None
    open_pr_after_generation: Optional[bool] = None
    sonar_organization: Optional[str] = None
    sonar_project_key: Optional[str] = None
    docker_registry: Optional[str] = Field(None, pattern="^(dockerhub|ghcr)$")
    deploy_provider: Optional[str] = Field(
        None, pattern="^(none|render|railway|vercel|custom)$"
    )


class PatchSecuritySettings(BaseModel):
    session_timeout: Optional[int] = Field(None, ge=5, le=10080)
    auto_logout_enabled: Optional[bool] = None
    mask_sensitive_values: Optional[bool] = None


# ── Default builders (used when no row exists yet) ────────────────────────────

_DEFAULT_USER = UserSettingsOut(
    theme="dark", language="en", density="comfortable",
    default_landing_page="assistant", show_ai_context_panel=True,
    show_sidebar_labels=True, show_cicd_in_sidebar=True,
    notifications={"enabled": True, "analysis": True, "cicd": True, "git": True},
    updated_at=None,
)
_DEFAULT_WORKSPACE = WorkspaceSettingsOut(
    environment="local", backend_url="http://127.0.0.1:8765",
    websocket_url="ws://127.0.0.1:8765/ws",
    default_branch="main", updated_at=None,
)
_DEFAULT_AI = AISettingsOut(
    ai_mode="balanced", response_style="detailed",
    use_repository_context=True, use_conversation_memory=True, use_rag=True,
    use_cicd_context=False, use_smart_git_context=False,
    streaming_enabled=True, proactive_enabled=False,
    temperature=0.3, max_context_size=8000, updated_at=None,
)
_DEFAULT_GIT = GitSettingsOut(
    provider="github", default_branch="main", safe_mode=True,
    pr_analysis_mode="fast", auto_refresh_prs=False,
    active_repo=None, updated_at=None,
)
_DEFAULT_CICD = CICDSettingsOut(
    provider="github_actions", pipeline_template="standard",
    quality_gate_threshold=80, auto_generate_pipeline=False,
    open_pr_after_generation=False, sonar_organization=None,
    sonar_project_key=None, docker_registry="dockerhub",
    deploy_provider="none", required_secrets={}, updated_at=None,
)
_DEFAULT_SECURITY = SecuritySettingsOut(
    auth_mode="jwt_otp", session_timeout=1440,
    auto_logout_enabled=False, mask_sensitive_values=True, updated_at=None,
)


# ── Helper: resolve PG user.id from Principal ─────────────────────────────────

async def _resolve_user_id(principal: Principal, db: AsyncSession) -> str:
    user = await UserRepo(db).get_by_email(principal.email)
    if not user:
        raise HTTPException(404, "User not found.")
    return user.id


# ── GET /api/settings ─────────────────────────────────────────────────────────

@settings_router.get("", response_model=AllSettingsOut)
async def get_all_settings(
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AllSettingsOut:
    async with db.begin():
        user_id = await _resolve_user_id(principal, db)
        repo = SettingsRepo(db)

        us = await repo.get_user_settings(user_id)
        ws = await repo.get_workspace_settings(user_id)
        ai = await repo.get_ai_settings(user_id)
        gs = await repo.get_git_settings(user_id)
        cs = await repo.get_cicd_settings(user_id)
        ss = await repo.get_security_settings(user_id)
        active_repo = await UserRepo(db).get_active_repo_by_email(principal.email)

    return AllSettingsOut(
        user=UserSettingsOut(
            theme=us.theme if us else _DEFAULT_USER.theme,
            language=us.language if us else _DEFAULT_USER.language,
            density=us.density if us else _DEFAULT_USER.density,
            default_landing_page=us.default_landing_page if us else _DEFAULT_USER.default_landing_page,
            show_ai_context_panel=us.show_ai_context_panel if us else _DEFAULT_USER.show_ai_context_panel,
            show_sidebar_labels=us.show_sidebar_labels if us else _DEFAULT_USER.show_sidebar_labels,
            show_cicd_in_sidebar=us.show_cicd_in_sidebar if us else _DEFAULT_USER.show_cicd_in_sidebar,
            notifications=us.notifications if us else _DEFAULT_USER.notifications,
            updated_at=us.updated_at if us else None,
        ),
        workspace=WorkspaceSettingsOut(
            environment=ws.environment if ws else _DEFAULT_WORKSPACE.environment,
            backend_url=ws.backend_url if ws else _DEFAULT_WORKSPACE.backend_url,
            websocket_url=ws.websocket_url if ws else _DEFAULT_WORKSPACE.websocket_url,
            default_branch=ws.default_branch if ws else _DEFAULT_WORKSPACE.default_branch,
            updated_at=ws.updated_at if ws else None,
        ),
        ai=AISettingsOut(
            ai_mode=ai.ai_mode if ai else _DEFAULT_AI.ai_mode,
            response_style=ai.response_style if ai else _DEFAULT_AI.response_style,
            use_repository_context=ai.use_repository_context if ai else _DEFAULT_AI.use_repository_context,
            use_conversation_memory=ai.use_conversation_memory if ai else _DEFAULT_AI.use_conversation_memory,
            use_rag=ai.use_rag if ai else _DEFAULT_AI.use_rag,
            use_cicd_context=ai.use_cicd_context if ai else _DEFAULT_AI.use_cicd_context,
            use_smart_git_context=ai.use_smart_git_context if ai else _DEFAULT_AI.use_smart_git_context,
            streaming_enabled=ai.streaming_enabled if ai else _DEFAULT_AI.streaming_enabled,
            proactive_enabled=ai.proactive_enabled if ai else _DEFAULT_AI.proactive_enabled,
            temperature=ai.temperature if ai else _DEFAULT_AI.temperature,
            max_context_size=ai.max_context_size if ai else _DEFAULT_AI.max_context_size,
            updated_at=ai.updated_at if ai else None,
        ),
        git=GitSettingsOut(
            provider=gs.provider if gs else _DEFAULT_GIT.provider,
            default_branch=gs.default_branch if gs else _DEFAULT_GIT.default_branch,
            safe_mode=gs.safe_mode if gs else _DEFAULT_GIT.safe_mode,
            pr_analysis_mode=gs.pr_analysis_mode if gs else _DEFAULT_GIT.pr_analysis_mode,
            auto_refresh_prs=gs.auto_refresh_prs if gs else _DEFAULT_GIT.auto_refresh_prs,
            active_repo=active_repo,
            updated_at=gs.updated_at if gs else None,
        ),
        cicd=CICDSettingsOut(
            provider=cs.provider if cs else _DEFAULT_CICD.provider,
            pipeline_template=cs.pipeline_template if cs else _DEFAULT_CICD.pipeline_template,
            quality_gate_threshold=cs.quality_gate_threshold if cs else _DEFAULT_CICD.quality_gate_threshold,
            auto_generate_pipeline=cs.auto_generate_pipeline if cs else _DEFAULT_CICD.auto_generate_pipeline,
            open_pr_after_generation=cs.open_pr_after_generation if cs else _DEFAULT_CICD.open_pr_after_generation,
            sonar_organization=cs.sonar_organization if cs else None,
            sonar_project_key=cs.sonar_project_key if cs else None,
            docker_registry=cs.docker_registry if cs else _DEFAULT_CICD.docker_registry,
            deploy_provider=cs.deploy_provider if cs else _DEFAULT_CICD.deploy_provider,
            required_secrets=cs.required_secrets if cs else {},
            updated_at=cs.updated_at if cs else None,
        ),
        security=SecuritySettingsOut(
            auth_mode="jwt_otp",
            session_timeout=ss.session_timeout if ss else _DEFAULT_SECURITY.session_timeout,
            auto_logout_enabled=ss.auto_logout_enabled if ss else _DEFAULT_SECURITY.auto_logout_enabled,
            mask_sensitive_values=ss.mask_sensitive_values if ss else _DEFAULT_SECURITY.mask_sensitive_values,
            updated_at=ss.updated_at if ss else None,
        ),
    )


# ── GET /api/settings/status ──────────────────────────────────────────────────

@settings_router.get("/status", response_model=SystemStatusOut)
async def get_system_status(
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SystemStatusOut:
    now = datetime.now(timezone.utc)

    # Backend
    backend_ok = ComponentStatus(ok=True, label="Connected", detail="v8.0.0", latency_ms=0.0)

    # PostgreSQL
    pg_ok: bool = False
    pg_latency: float = 0.0
    pg_detail: str = "Error"
    try:
        t0 = time.perf_counter()
        async with db.begin():
            from sqlalchemy import text
            await db.execute(text("SELECT 1"))
        pg_latency = round((time.perf_counter() - t0) * 1000, 1)
        pg_ok = True
        pg_detail = "Connected"
    except Exception as exc:
        pg_detail = str(exc)[:120]

    # Redis
    redis_ok: bool = False
    redis_latency: float = 0.0
    redis_detail: str = "Not connected"
    try:
        from services.mcp_redis_service import get_mcp_redis
        t0 = time.perf_counter()
        get_mcp_redis().ping()
        redis_latency = round((time.perf_counter() - t0) * 1000, 1)
        redis_ok = True
        redis_detail = "Connected"
    except Exception as exc:
        redis_detail = str(exc)[:80]

    # GitHub
    github_connected: bool = False
    active_repo: Optional[str] = None
    try:
        async with db.begin():
            token = await UserRepo(db).get_github_token_by_email(principal.email)
            active_repo = await UserRepo(db).get_active_repo_by_email(principal.email)
        github_connected = bool(token)
    except Exception:
        pass

    # AI service (check backend services)
    ai_ok: bool = False
    ai_detail: str = "Unknown"
    try:
        from api.server import _watch_services
        ai_ok = "rag_system" in _watch_services
        ai_detail = "Ready" if ai_ok else "Initializing"
    except Exception:
        ai_detail = "Unavailable"

    # CI/CD
    cicd_detail: str = "No pipeline declared"
    cicd_ok: bool = False
    try:
        async with db.begin():
            from database.repositories.settings_repo import SettingsRepo
            user_id = await _resolve_user_id(principal, db)
            cs = await SettingsRepo(db).get_cicd_settings(user_id)
            if cs and cs.provider != "none":
                cicd_ok = True
                cicd_detail = f"{cs.provider} · {cs.pipeline_template}"
    except Exception:
        pass

    return SystemStatusOut(
        backend=backend_ok,
        postgres=ComponentStatus(ok=pg_ok, label="Connected" if pg_ok else "Error", detail=pg_detail, latency_ms=pg_latency),
        redis=ComponentStatus(ok=redis_ok, label="Connected" if redis_ok else "Error", detail=redis_detail, latency_ms=redis_latency),
        github=ComponentStatus(ok=github_connected, label="Connected" if github_connected else "Not connected", detail=active_repo),
        ai=ComponentStatus(ok=ai_ok, label="Ready" if ai_ok else "Initializing", detail=ai_detail),
        cicd=ComponentStatus(ok=cicd_ok, label="Configured" if cicd_ok else "Not configured", detail=cicd_detail),
        active_repo=active_repo,
        checked_at=now,
    )


# ── PATCH endpoints ───────────────────────────────────────────────────────────

@settings_router.patch("/user", response_model=SavedOut)
async def patch_user_settings(
    body: PatchUserSettings,
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SavedOut:
    updates = body.dict(exclude_none=True)
    if not updates:
        raise HTTPException(422, "No fields to update.")
    async with db.begin():
        user_id = await _resolve_user_id(principal, db)
        obj = await SettingsRepo(db).upsert_user_settings(user_id, updates)
    _clear_settings_cache(user_id)
    return SavedOut(saved=True, updated_at=obj.updated_at)


@settings_router.patch("/workspace", response_model=SavedOut)
async def patch_workspace_settings(
    body: PatchWorkspaceSettings,
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SavedOut:
    updates = body.dict(exclude_none=True)
    if not updates:
        raise HTTPException(422, "No fields to update.")
    async with db.begin():
        user_id = await _resolve_user_id(principal, db)
        obj = await SettingsRepo(db).upsert_workspace_settings(user_id, updates)
    _clear_settings_cache(user_id)
    return SavedOut(saved=True, updated_at=obj.updated_at)


@settings_router.patch("/ai", response_model=SavedOut)
async def patch_ai_settings(
    body: PatchAISettings,
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SavedOut:
    updates = body.dict(exclude_none=True)
    if not updates:
        raise HTTPException(422, "No fields to update.")
    async with db.begin():
        user_id = await _resolve_user_id(principal, db)
        obj = await SettingsRepo(db).upsert_ai_settings(user_id, updates)
    _clear_settings_cache(user_id)
    return SavedOut(saved=True, updated_at=obj.updated_at)


@settings_router.patch("/git", response_model=SavedOut)
async def patch_git_settings(
    body: PatchGitSettings,
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SavedOut:
    updates = body.dict(exclude_none=True)
    if not updates:
        raise HTTPException(422, "No fields to update.")
    async with db.begin():
        user_id = await _resolve_user_id(principal, db)
        obj = await SettingsRepo(db).upsert_git_settings(user_id, updates)
    _clear_settings_cache(user_id)
    return SavedOut(saved=True, updated_at=obj.updated_at)


@settings_router.patch("/cicd", response_model=SavedOut)
async def patch_cicd_settings(
    body: PatchCICDSettings,
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SavedOut:
    updates = body.dict(exclude_none=True)
    if not updates:
        raise HTTPException(422, "No fields to update.")
    async with db.begin():
        user_id = await _resolve_user_id(principal, db)
        obj = await SettingsRepo(db).upsert_cicd_settings(user_id, updates)
    _clear_settings_cache(user_id)
    return SavedOut(saved=True, updated_at=obj.updated_at)


@settings_router.patch("/security", response_model=SavedOut)
async def patch_security_settings(
    body: PatchSecuritySettings,
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SavedOut:
    updates = body.dict(exclude_none=True)
    if not updates:
        raise HTTPException(422, "No fields to update.")
    async with db.begin():
        user_id = await _resolve_user_id(principal, db)
        obj = await SettingsRepo(db).upsert_security_settings(user_id, updates)
    _clear_settings_cache(user_id)
    return SavedOut(saved=True, updated_at=obj.updated_at)


# ── Test connection endpoints ─────────────────────────────────────────────────

@settings_router.post("/test/backend", response_model=TestResult)
async def test_backend(principal: Principal = Depends(get_current_user)) -> TestResult:
    _check_rate_limit(principal.id, "backend")
    t0 = time.perf_counter()
    latency = round((time.perf_counter() - t0) * 1000, 1)
    return TestResult(ok=True, latency_ms=latency, detail="Backend API v8.0.0 responding.")


@settings_router.post("/test/postgres", response_model=TestResult)
async def test_postgres(
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> TestResult:
    _check_rate_limit(principal.id, "postgres")
    try:
        from sqlalchemy import text
        t0 = time.perf_counter()
        async with db.begin():
            await db.execute(text("SELECT 1"))
        latency = round((time.perf_counter() - t0) * 1000, 1)
        return TestResult(ok=True, latency_ms=latency, detail="PostgreSQL connection healthy.")
    except Exception as exc:
        return TestResult(ok=False, detail=str(exc)[:200])


@settings_router.post("/test/redis", response_model=TestResult)
async def test_redis(principal: Principal = Depends(get_current_user)) -> TestResult:
    _check_rate_limit(principal.id, "redis")
    try:
        from services.mcp_redis_service import get_mcp_redis
        t0 = time.perf_counter()
        get_mcp_redis().ping()
        latency = round((time.perf_counter() - t0) * 1000, 1)
        return TestResult(ok=True, latency_ms=latency, detail="Redis PING successful.")
    except Exception as exc:
        return TestResult(ok=False, detail=str(exc)[:200])


@settings_router.post("/test/github", response_model=TestResult)
async def test_github(
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> TestResult:
    _check_rate_limit(principal.id, "github")
    try:
        async with db.begin():
            token = await UserRepo(db).get_github_token_by_email(principal.email)
        if not token:
            return TestResult(ok=False, detail="GitHub account not connected.")
        req = urllib.request.Request("https://api.github.com/user")
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        latency = round((time.perf_counter() - t0) * 1000, 1)
        login = data.get("login", "unknown")
        return TestResult(ok=True, latency_ms=latency, detail=f"GitHub token valid · @{login}")
    except Exception as exc:
        return TestResult(ok=False, detail=str(exc)[:200])


# ── Cache & maintenance ───────────────────────────────────────────────────────

@settings_router.post("/cache/clear")
async def clear_cache(principal: Principal = Depends(get_current_user)) -> Dict[str, Any]:
    count = _clear_settings_cache(principal.id)
    return {"cleared": count, "message": f"Cleared {count} cached settings keys."}


# ── Reset endpoints ───────────────────────────────────────────────────────────

@settings_router.post("/reset/user", response_model=SavedOut)
async def reset_user_settings(
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SavedOut:
    async with db.begin():
        user_id = await _resolve_user_id(principal, db)
        obj = await SettingsRepo(db).reset_user_settings(user_id)
    _clear_settings_cache(user_id)
    return SavedOut(saved=True, updated_at=obj.updated_at)


@settings_router.post("/reset/ai", response_model=SavedOut)
async def reset_ai_settings(
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SavedOut:
    async with db.begin():
        user_id = await _resolve_user_id(principal, db)
        obj = await SettingsRepo(db).reset_ai_settings(user_id)
    _clear_settings_cache(user_id)
    return SavedOut(saved=True, updated_at=obj.updated_at)


# ── CI/CD Secrets — GitHub API integration ───────────────────────────────────

_ALLOWED_SECRET_NAMES = frozenset({
    "SONAR_TOKEN",
    "DOCKER_USERNAME",
    "DOCKER_PASSWORD",
    "GHCR_TOKEN",
    "DEPLOY_TOKEN",
    "CODECOV_TOKEN",
})


class _NaClMissingError(RuntimeError):
    pass


def _encrypt_secret_for_github(public_key_b64: str, secret_value: str) -> str:
    """Encrypt a secret using the repo's LibSodium public key.
    Required by GitHub Secrets API — raw values are rejected.
    """
    try:
        import base64
        from nacl.public import PublicKey, SealedBox
    except ImportError:
        raise _NaClMissingError(
            "PyNaCl is not installed. Run: pip install PyNaCl"
        )
    import base64 as _b64
    public_key_bytes = _b64.b64decode(public_key_b64)
    encrypted = SealedBox(PublicKey(public_key_bytes)).encrypt(secret_value.encode("utf-8"))
    return _b64.b64encode(encrypted).decode("utf-8")


def _github_push_secret_sync(
    token: str, owner: str, repo: str, name: str, value: str
) -> None:
    """Push a secret to GitHub via REST API. Blocking — run in thread pool."""
    pub_key_req = urllib.request.Request(
        f"https://api.github.com/repos/{owner}/{repo}/actions/public-key"
    )
    pub_key_req.add_header("Authorization", f"Bearer {token}")
    pub_key_req.add_header("Accept", "application/vnd.github+json")
    pub_key_req.add_header("X-GitHub-Api-Version", "2022-11-28")

    with urllib.request.urlopen(pub_key_req, timeout=10) as resp:
        key_data = json.loads(resp.read().decode())

    encrypted_value = _encrypt_secret_for_github(key_data["key"], value)

    payload = json.dumps({
        "encrypted_value": encrypted_value,
        "key_id": key_data["key_id"],
    }).encode()
    put_req = urllib.request.Request(
        f"https://api.github.com/repos/{owner}/{repo}/actions/secrets/{name}",
        data=payload,
        method="PUT",
    )
    put_req.add_header("Authorization", f"Bearer {token}")
    put_req.add_header("Accept", "application/vnd.github+json")
    put_req.add_header("X-GitHub-Api-Version", "2022-11-28")
    put_req.add_header("Content-Type", "application/json")

    with urllib.request.urlopen(put_req, timeout=10):
        pass  # 201 Created or 204 No Content — both are success


def _github_delete_secret_sync(
    token: str, owner: str, repo: str, name: str
) -> None:
    """Delete a secret from GitHub via REST API. Blocking — run in thread pool."""
    del_req = urllib.request.Request(
        f"https://api.github.com/repos/{owner}/{repo}/actions/secrets/{name}",
        method="DELETE",
    )
    del_req.add_header("Authorization", f"Bearer {token}")
    del_req.add_header("Accept", "application/vnd.github+json")
    del_req.add_header("X-GitHub-Api-Version", "2022-11-28")

    try:
        with urllib.request.urlopen(del_req, timeout=10):
            pass
    except urllib.error.HTTPError as exc:
        if exc.code != 404:  # 404 = already gone, treat as success
            raise


class SetSecretIn(BaseModel):
    name: str = Field(
        ...,
        pattern=(
            "^(SONAR_TOKEN|DOCKER_USERNAME|DOCKER_PASSWORD"
            "|GHCR_TOKEN|DEPLOY_TOKEN|CODECOV_TOKEN)$"
        ),
    )
    value: str = Field(..., min_length=1, max_length=65536)


class SecretStatusOut(BaseModel):
    name: str
    status: str  # "set" | "not_set"
    updated_at: Optional[datetime] = None


@settings_router.get("/cicd/secrets", response_model=List[SecretStatusOut])
async def get_cicd_secrets(
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> List[SecretStatusOut]:
    async with db.begin():
        user_id = await _resolve_user_id(principal, db)
        cs = await SettingsRepo(db).get_cicd_settings(user_id)

    stored: Dict[str, Any] = (cs.required_secrets or {}) if cs else {}
    return [
        SecretStatusOut(
            name=name,
            status=stored.get(name, {}).get("status", "not_set"),
            updated_at=stored.get(name, {}).get("updated_at"),
        )
        for name in sorted(_ALLOWED_SECRET_NAMES)
    ]


@settings_router.post("/cicd/secrets", response_model=SecretStatusOut)
async def set_cicd_secret(
    body: SetSecretIn,
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SecretStatusOut:
    _check_rate_limit(principal.id, "secret_set")

    async with db.begin():
        user_id = await _resolve_user_id(principal, db)
        token = await UserRepo(db).get_github_token_by_email(principal.email)
        active_repo = await UserRepo(db).get_active_repo_by_email(principal.email)

    if not token:
        raise HTTPException(400, "GitHub account not connected. Connect GitHub first.")
    if not active_repo or "/" not in active_repo:
        raise HTTPException(400, "No active repository selected.")

    owner, repo = active_repo.split("/", 1)

    try:
        await asyncio.to_thread(
            _github_push_secret_sync, token, owner, repo, body.name, body.value
        )
    except _NaClMissingError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        raise HTTPException(502, f"GitHub API error: {str(exc)[:200]}")

    now = datetime.now(timezone.utc)
    async with db.begin():
        uid = await _resolve_user_id(principal, db)
        await SettingsRepo(db).update_cicd_secret_status(uid, body.name, "set", now)

    _clear_settings_cache(principal.id)
    logger.info(
        "Secret %s pushed to %s/%s by %s", body.name, owner, repo, principal.email
    )
    return SecretStatusOut(name=body.name, status="set", updated_at=now)


@settings_router.delete("/cicd/secrets/{name}", response_model=SecretStatusOut)
async def delete_cicd_secret(
    name: str,
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SecretStatusOut:
    if name not in _ALLOWED_SECRET_NAMES:
        raise HTTPException(
            400,
            f"Unknown secret name. Allowed: {', '.join(sorted(_ALLOWED_SECRET_NAMES))}",
        )
    _check_rate_limit(principal.id, "secret_delete")

    async with db.begin():
        user_id = await _resolve_user_id(principal, db)
        token = await UserRepo(db).get_github_token_by_email(principal.email)
        active_repo = await UserRepo(db).get_active_repo_by_email(principal.email)

    if not token:
        raise HTTPException(400, "GitHub account not connected.")
    if not active_repo or "/" not in active_repo:
        raise HTTPException(400, "No active repository selected.")

    owner, repo = active_repo.split("/", 1)

    try:
        await asyncio.to_thread(
            _github_delete_secret_sync, token, owner, repo, name
        )
    except Exception as exc:
        raise HTTPException(502, f"GitHub API error: {str(exc)[:200]}")

    now = datetime.now(timezone.utc)
    async with db.begin():
        uid = await _resolve_user_id(principal, db)
        await SettingsRepo(db).update_cicd_secret_status(uid, name, "not_set", now)

    _clear_settings_cache(principal.id)
    logger.info(
        "Secret %s deleted from %s/%s by %s", name, owner, repo, principal.email
    )
    return SecretStatusOut(name=name, status="not_set", updated_at=now)


# ── Export ────────────────────────────────────────────────────────────────────

@settings_router.get("/export")
async def export_settings(
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    """Return all user settings as a downloadable JSON. Secrets are never included."""
    data = await get_all_settings(principal, db)
    payload = data.dict()
    # Scrub any residual sensitive keys just to be safe
    for scope in payload.values():
        if isinstance(scope, dict):
            for k in list(scope.keys()):
                if any(s in k.lower() for s in ("token", "secret", "password", "key")):
                    scope[k] = "••••••"
    return JSONResponse(
        content=payload,
        headers={
            "Content-Disposition": 'attachment; filename="code_auditor_settings.json"'
        },
    )
