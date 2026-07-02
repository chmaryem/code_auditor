"""
api/issue_actions_router.py — Issue Apply lifecycle endpoints.

Endpoints:
  POST /issue-actions/snapshot              → capture file content before apply
  POST /issue-actions/snapshot/{patch_id}/after → fill content_after post-apply
  POST /issue-actions/event                 → record a lifecycle action
  POST /issue-actions/verify                → re-scan file + compare issues
  POST /issue-actions/undo/{patch_id}       → restore file to content_before
  GET  /issue-actions/history/{issue_id}    → timeline for one issue

Redis usage (read-only for this router — locks managed by the caller):
  issue:apply-lock:{workspace_id}:{file_hash}  — checked, not set here
  issue:rescan-job:{issue_id}                  — set during verify, cleared on done
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from auth.security import Principal, get_current_user
from database.connection import get_db
from database.models import IssueActionEvent, PatchSnapshot, User

logger = logging.getLogger(__name__)

issue_actions_router = APIRouter(prefix="/issue-actions", tags=["Issue Actions"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


async def _resolve_user_id(email: str, db: AsyncSession) -> Optional[str]:
    result = await db.execute(select(User.id).where(User.email == email))
    return result.scalar_one_or_none()


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class CreateSnapshotRequest(BaseModel):
    issue_id:       str = Field(..., description="Stable issue hash from the plugin")
    patch_id:       str = Field(..., description="Client-generated UUID for this patch")
    file_path:      str = Field(..., description="Absolute path to the file")
    content_before: str = Field(..., description="Full file content before the patch")
    workspace_id:   Optional[str] = None


class FillSnapshotAfterRequest(BaseModel):
    content_after: str = Field(..., description="Full file content after the patch")


class RecordEventRequest(BaseModel):
    issue_id:                str
    file_path:               str
    action:                  str  # applied | verified_fixed | verification_failed | undo | ignored | false_positive | rescan_requested | rescan_completed
    patch_id:                Optional[str]  = None
    previous_status:         Optional[str]  = None
    next_status:             Optional[str]  = None
    file_version_before_hash: Optional[str] = None
    patch_summary:           Optional[str]  = None
    workspace_id:            Optional[str]  = None
    metadata:                Optional[Dict[str, Any]] = None


class VerifyRequest(BaseModel):
    issue_id:     str
    file_path:    str
    patch_id:     str
    project_path: str = "."
    workspace_id: Optional[str] = None


class UndoResult(BaseModel):
    status:  str  # undone | failed | file_changed
    message: str


# ── POST /issue-actions/snapshot ──────────────────────────────────────────────

@issue_actions_router.post("/snapshot", status_code=201)
async def create_snapshot(
    req:       CreateSnapshotRequest,
    principal: Principal    = Depends(get_current_user),
    db:        AsyncSession = Depends(get_db),
):
    """
    Capture file content BEFORE applying a patch.
    Must be called before WorkspaceEdit — enables safe undo.
    """
    user_id = await _resolve_user_id(principal.email, db)
    if not user_id:
        raise HTTPException(404, "User not found")

    snap = PatchSnapshot(
        user_id            = user_id,
        workspace_id       = req.workspace_id,
        issue_id           = req.issue_id,
        patch_id           = req.patch_id,
        file_path          = req.file_path,
        content_before     = req.content_before,
        content_hash_before = _sha256(req.content_before),
    )
    db.add(snap)
    await db.commit()
    await db.refresh(snap)
    return {"id": snap.id, "patch_id": snap.patch_id}


# ── POST /issue-actions/snapshot/{patch_id}/after ────────────────────────────

@issue_actions_router.post("/snapshot/{patch_id}/after")
async def fill_snapshot_after(
    patch_id:  str,
    req:       FillSnapshotAfterRequest,
    principal: Principal    = Depends(get_current_user),
    db:        AsyncSession = Depends(get_db),
):
    """Fill content_after + applied_at once the WorkspaceEdit has succeeded."""
    user_id = await _resolve_user_id(principal.email, db)
    if not user_id:
        raise HTTPException(404, "User not found")

    result = await db.execute(
        select(PatchSnapshot).where(
            PatchSnapshot.patch_id == patch_id,
            PatchSnapshot.user_id  == user_id,
        )
    )
    snap = result.scalar_one_or_none()
    if not snap:
        raise HTTPException(404, f"Snapshot not found for patch_id={patch_id}")

    snap.content_after      = req.content_after
    snap.content_hash_after = _sha256(req.content_after)
    snap.applied_at         = _now()
    await db.commit()
    return {"status": "ok"}


# ── POST /issue-actions/event ─────────────────────────────────────────────────

@issue_actions_router.post("/event", status_code=201)
async def record_event(
    req:       RecordEventRequest,
    principal: Principal    = Depends(get_current_user),
    db:        AsyncSession = Depends(get_db),
):
    """Append an action event to the issue timeline."""
    user_id = await _resolve_user_id(principal.email, db)
    if not user_id:
        raise HTTPException(404, "User not found")

    ev = IssueActionEvent(
        user_id                  = user_id,
        workspace_id             = req.workspace_id,
        issue_id                 = req.issue_id,
        patch_id                 = req.patch_id,
        file_path                = req.file_path,
        action                   = req.action,
        previous_status          = req.previous_status,
        next_status              = req.next_status,
        file_version_before_hash = req.file_version_before_hash,
        patch_summary            = req.patch_summary,
        metadata_                = req.metadata,
    )
    db.add(ev)
    await db.commit()
    await db.refresh(ev)
    return {"id": ev.id, "created_at": ev.created_at.isoformat()}


# ── POST /issue-actions/verify ────────────────────────────────────────────────

@issue_actions_router.post("/verify")
async def verify_fix(
    req:       VerifyRequest,
    principal: Principal    = Depends(get_current_user),
    db:        AsyncSession = Depends(get_db),
):
    """
    Re-scan the file via the WatchGraph and check whether the issue_id
    still appears in the new results.

    Returns:
      { status: "fixed" | "open" | "failed", issues: [...], error: str|null }
    """
    user_id = await _resolve_user_id(principal.email, db)
    if not user_id:
        raise HTTPException(404, "User not found")

    fp = Path(req.file_path)
    if not fp.exists():
        return {"status": "failed", "issues": [], "error": f"File not found: {req.file_path}"}

    # Optional Redis lock check — if another verify is already running for this issue, skip
    try:
        from services.mcp_redis_service import get_mcp_redis
        redis = get_mcp_redis()
        lock_key = f"issue:rescan-job:{req.issue_id}"
        existing = redis.get(lock_key)
        if existing:
            return {"status": "failed", "issues": [], "error": "Re-scan already in progress"}
        redis.setex(lock_key, 120, "running")
    except Exception:
        pass  # Redis unavailable — proceed without lock

    try:
        from api.server import _get_project_services
        from langchain_agents.graphs.watch_graph import invoke_watch

        project_path = Path(req.project_path).resolve() if req.project_path else fp.parent
        svc = _get_project_services(project_path)

        result = await asyncio.to_thread(
            invoke_watch,
            file_path    = str(fp),
            project_path = str(project_path),
            project_indexer = svc.get("project_indexer"),
            extractor       = svc.get("extractor"),
            rag_system      = svc.get("rag_system"),
            dep_graph       = svc.get("dep_graph"),
        )

        # Extract issues from ws_events (schema v2.0) or raw result
        ws_events  = result.get("ws_events") or []
        ar_event   = next((e for e in ws_events if e.get("type") == "analysis_result"), {})
        new_issues = ar_event.get("issues") or result.get("issues") or []

        # Build a set of stable IDs for the new issues using the same djb2 hash
        # the plugin uses — filePath|severity|line|title[:60]
        def _djb2(s: str) -> str:
            h = 5381
            for c in s:
                h = ((h << 5) + h + ord(c)) & 0xFFFFFFFF
            return hex(h)[2:]

        new_ids: set[str] = set()
        for iss in new_issues:
            title    = (iss.get("title") or iss.get("message") or "")[:60]
            sev_raw  = str(iss.get("severity", "warning")).upper()
            sev      = "HIGH" if sev_raw in ("ERROR", "HIGH") else sev_raw if sev_raw in ("CRITICAL", "WARNING", "INFO") else "INFO"
            line_str = str(iss.get("line") or "")
            seed     = f"{req.file_path}|{sev}|{line_str}|{title}"
            new_ids.add(f"iss-{_djb2(seed)}")

        issue_gone = req.issue_id not in new_ids
        status     = "fixed" if issue_gone else "open"

        # Record verify event in PG (fire-and-forget on error)
        try:
            ev = IssueActionEvent(
                user_id         = user_id,
                workspace_id    = req.workspace_id,
                issue_id        = req.issue_id,
                patch_id        = req.patch_id,
                file_path       = req.file_path,
                action          = "verified_fixed" if issue_gone else "verification_failed",
                previous_status = "pending_verification",
                next_status     = status,
            )
            db.add(ev)
            await db.commit()
        except Exception as e:
            logger.warning("verify: failed to persist event: %s", e)

        return {"status": status, "issues": new_issues, "error": None}

    except Exception as e:
        logger.exception("verify_fix error for issue %s: %s", req.issue_id, e)

        # Record failure event
        try:
            ev = IssueActionEvent(
                user_id         = user_id,
                issue_id        = req.issue_id,
                patch_id        = req.patch_id,
                file_path       = req.file_path,
                action          = "verification_failed",
                previous_status = "pending_verification",
                next_status     = "failed",
                metadata_       = {"error": str(e)},
            )
            db.add(ev)
            await db.commit()
        except Exception:
            pass

        return {"status": "failed", "issues": [], "error": str(e)}

    finally:
        try:
            from services.mcp_redis_service import get_mcp_redis
            redis = get_mcp_redis()
            redis.delete(f"issue:rescan-job:{req.issue_id}")
        except Exception:
            pass


# ── POST /issue-actions/undo/{patch_id} ──────────────────────────────────────

@issue_actions_router.post("/undo/{patch_id}", response_model=UndoResult)
async def undo_patch(
    patch_id:  str,
    principal: Principal    = Depends(get_current_user),
    db:        AsyncSession = Depends(get_db),
):
    """
    Restore the file to its state before the patch was applied.

    Safety checks:
    1. Verify the snapshot exists and belongs to this user.
    2. Read the current file — if it doesn't match content_hash_after,
       the developer has made changes after the patch → refuse automatic rollback.
    3. Write content_before to disk.
    4. Mark snapshot.reverted_at.
    5. Record undo event in PG.
    """
    user_id = await _resolve_user_id(principal.email, db)
    if not user_id:
        raise HTTPException(404, "User not found")

    result = await db.execute(
        select(PatchSnapshot).where(
            PatchSnapshot.patch_id == patch_id,
            PatchSnapshot.user_id  == user_id,
        )
    )
    snap = result.scalar_one_or_none()
    if not snap:
        raise HTTPException(404, f"Snapshot not found for patch_id={patch_id}")

    if snap.reverted_at:
        return UndoResult(status="failed", message="This patch has already been reverted.")

    fp = Path(snap.file_path)
    if not fp.exists():
        return UndoResult(status="failed", message=f"File not found: {snap.file_path}")

    # Safety: check current file vs content_after hash
    current_content = fp.read_text(encoding="utf-8", errors="replace")
    current_hash    = _sha256(current_content)

    if snap.content_hash_after and current_hash != snap.content_hash_after:
        return UndoResult(
            status  = "file_changed",
            message = "File changed after patch was applied. Manual review required — automatic rollback refused.",
        )

    # Restore content_before
    try:
        fp.write_text(snap.content_before, encoding="utf-8")
    except OSError as e:
        return UndoResult(status="failed", message=f"Could not write file: {e}")

    # Mark snapshot as reverted
    snap.reverted_at = _now()
    await db.commit()

    # Record undo event
    try:
        ev = IssueActionEvent(
            user_id         = user_id,
            workspace_id    = snap.workspace_id,
            issue_id        = snap.issue_id,
            patch_id        = patch_id,
            file_path       = snap.file_path,
            action          = "undo",
            previous_status = "fixed",
            next_status     = "open",
        )
        db.add(ev)
        await db.commit()
    except Exception as e:
        logger.warning("undo: event record failed: %s", e)

    return UndoResult(status="undone", message="Patch reverted successfully. Re-scan recommended.")


# ── GET /issue-actions/history/{issue_id} ────────────────────────────────────

@issue_actions_router.get("/history/{issue_id}")
async def get_history(
    issue_id:  str,
    principal: Principal    = Depends(get_current_user),
    db:        AsyncSession = Depends(get_db),
):
    """Return the chronological action timeline for a single issue."""
    user_id = await _resolve_user_id(principal.email, db)
    if not user_id:
        return {"events": []}

    q = (
        select(IssueActionEvent)
        .where(
            IssueActionEvent.user_id  == user_id,
            IssueActionEvent.issue_id == issue_id,
        )
        .order_by(IssueActionEvent.created_at)
        .limit(100)
    )
    rows = list((await db.execute(q)).scalars().all())

    return {
        "issue_id": issue_id,
        "events": [
            {
                "id":              ev.id,
                "action":          ev.action,
                "previous_status": ev.previous_status,
                "next_status":     ev.next_status,
                "patch_id":        ev.patch_id,
                "patch_summary":   ev.patch_summary,
                "created_at":      ev.created_at.isoformat(),
            }
            for ev in rows
        ],
    }
