"""
api/git_router.py — FastAPI router for SmartGitGraph.

Endpoints:
  POST /git/status           → git session snapshot + risks
  POST /git/branch           → branch readiness vs base
  POST /git/commit-msg       → generate conventional commit message
  POST /git/conflicts        → dry-run conflict resolution
  POST /git/pr/review        → PR review via GitHub API
  POST /git/pr/readiness     → merge readiness check
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

git_router = APIRouter(prefix="/git", tags=["SmartGit"])


# ── Pydantic models ───────────────────────────────────────────────────────────

class GitSessionRequest(BaseModel):
    project_path: str = Field(..., description="Absolute path to git project")
    session_id:   str = Field("", description="Optional session ID")

class GitBranchRequest(BaseModel):
    project_path: str = Field(..., description="Absolute path to git project")
    branch:       str = Field("HEAD", description="Feature branch to analyze")
    base:         str = Field("main", description="Base branch")
    session_id:   str = Field("", description="Optional session ID")

class GitCommitMsgRequest(BaseModel):
    project_path: str = Field(..., description="Absolute path to git project")
    session_id:   str = Field("", description="Optional session ID")

class GitConflictRequest(BaseModel):
    project_path: str = Field(..., description="Absolute path to git project")
    session_id:   str = Field("", description="Optional session ID")

class GitPRRequest(BaseModel):
    owner:      str = Field(..., description="GitHub owner")
    repo:       str = Field(..., description="GitHub repo name")
    pr_number:  int = Field(..., description="Pull request number")
    session_id: str = Field("", description="Optional session ID")

class GitResponse(BaseModel):
    response:         str             = ""
    intent:           str             = ""
    confidence:       float           = 0.0
    safe_mode:        bool            = True
    elapsed_seconds:  float           = 0.0
    session_snapshot: Dict[str, Any]  = Field(default_factory=dict)
    branch_report:    Dict[str, Any]  = Field(default_factory=dict)
    commit_message:   str             = ""
    changes:          Dict[str, Any]  = Field(default_factory=dict)
    conflict_report:  Dict[str, Any]  = Field(default_factory=dict)
    pr_report:        Dict[str, Any]  = Field(default_factory=dict)
    readiness_report: Dict[str, Any]  = Field(default_factory=dict)
    errors:           List[str]       = Field(default_factory=list)


# ── Helper ────────────────────────────────────────────────────────────────────

async def _invoke(message: str, **kwargs) -> Dict[str, Any]:
    from langchain_agents.graphs.smart_git_graph import ainvoke_smart_git
    return await ainvoke_smart_git(message=message, **kwargs)


def _to_response(result: Dict[str, Any], elapsed: float) -> GitResponse:
    return GitResponse(
        response         = result.get("response", ""),
        intent           = result.get("intent", ""),
        confidence       = float(result.get("confidence", 0.0)),
        safe_mode        = bool(result.get("safe_mode", True)),
        elapsed_seconds  = elapsed,
        session_snapshot = result.get("session_snapshot") or {},
        branch_report    = result.get("branch_report") or {},
        commit_message   = result.get("commit_message", ""),
        changes          = result.get("changes") or {},
        conflict_report  = result.get("conflict_report") or {},
        pr_report        = result.get("pr_report") or {},
        readiness_report = result.get("readiness_report") or {},
        errors           = result.get("errors") or [],
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@git_router.post("/status", response_model=GitResponse, summary="Git session status")
async def git_status(req: GitSessionRequest):
    """
    Retourne le snapshot de session Git (bugs accumulés, score de risque,
    fichiers non commités). Utilise SmartGitGraph → node_session.
    """
    project_path = Path(req.project_path)
    if not project_path.exists():
        raise HTTPException(404, f"Projet introuvable : {project_path}")

    t0 = time.time()
    try:
        result = await _invoke(
            message="git status",
            project_path=str(project_path),
            session_id=req.session_id,
        )
    except Exception as e:
        logger.exception("git/status error")
        raise HTTPException(500, f"SmartGitGraph error: {e}")

    return _to_response(result, round(time.time() - t0, 2))


@git_router.post("/branch", response_model=GitResponse, summary="Branch readiness")
async def git_branch(req: GitBranchRequest):
    """
    Analyse une branche feature vs sa base et retourne un verdict de merge
    (criticality, uncommitted bugs, divergence). SmartGitGraph → node_branch.
    """
    project_path = Path(req.project_path)
    if not project_path.exists():
        raise HTTPException(404, f"Projet introuvable : {project_path}")

    t0 = time.time()
    try:
        result = await _invoke(
            message=f"branch readiness {req.branch} vs {req.base}",
            project_path=str(project_path),
            branch=req.branch,
            base=req.base,
            session_id=req.session_id,
        )
    except Exception as e:
        logger.exception("git/branch error")
        raise HTTPException(500, f"SmartGitGraph error: {e}")

    return _to_response(result, round(time.time() - t0, 2))


@git_router.post("/commit-msg", response_model=GitResponse, summary="Generate commit message")
async def git_commit_msg(req: GitCommitMsgRequest):
    """
    Génère un message de commit Conventional Commits basé sur les diffs
    actuels. SmartGitGraph → node_diff (intent=commit_message).
    """
    project_path = Path(req.project_path)
    if not project_path.exists():
        raise HTTPException(404, f"Projet introuvable : {project_path}")

    t0 = time.time()
    try:
        result = await _invoke(
            message="generate commit message",
            project_path=str(project_path),
            session_id=req.session_id,
        )
    except Exception as e:
        logger.exception("git/commit-msg error")
        raise HTTPException(500, f"SmartGitGraph error: {e}")

    return _to_response(result, round(time.time() - t0, 2))


@git_router.post("/conflicts", response_model=GitResponse, summary="Dry-run conflict resolution")
async def git_conflicts(req: GitConflictRequest):
    """
    Analyse les conflits de merge locaux et propose une résolution (dry-run,
    sans écriture sur disque). SmartGitGraph → node_conflict.
    """
    project_path = Path(req.project_path)
    if not project_path.exists():
        raise HTTPException(404, f"Projet introuvable : {project_path}")

    t0 = time.time()
    try:
        result = await _invoke(
            message="resolve conflicts dry run",
            project_path=str(project_path),
            session_id=req.session_id,
        )
    except Exception as e:
        logger.exception("git/conflicts error")
        raise HTTPException(500, f"SmartGitGraph error: {e}")

    return _to_response(result, round(time.time() - t0, 2))


@git_router.post("/pr/review", response_model=GitResponse, summary="PR review")
async def git_pr_review(req: GitPRRequest):
    """
    Lance une revue de Pull Request via SmartGitGraph → node_pr.
    Analyse le code de la feature branch + posts un review sur GitHub.
    """
    t0 = time.time()
    try:
        result = await _invoke(
            message=f"review PR #{req.pr_number}",
            owner=req.owner,
            repo=req.repo,
            pr_number=req.pr_number,
            session_id=req.session_id,
        )
    except Exception as e:
        logger.exception("git/pr/review error")
        raise HTTPException(500, f"SmartGitGraph error: {e}")

    return _to_response(result, round(time.time() - t0, 2))


@git_router.post("/pr/readiness", response_model=GitResponse, summary="PR merge readiness")
async def git_pr_readiness(req: GitPRRequest):
    """
    Vérifie si une PR est prête à merger (CI checks, reviews, conflicts).
    SmartGitGraph → node_pr (intent=pr_readiness). Ne merge JAMAIS.
    """
    t0 = time.time()
    try:
        result = await _invoke(
            message=f"is PR #{req.pr_number} ready to merge?",
            owner=req.owner,
            repo=req.repo,
            pr_number=req.pr_number,
            session_id=req.session_id,
        )
    except Exception as e:
        logger.exception("git/pr/readiness error")
        raise HTTPException(500, f"SmartGitGraph error: {e}")

    return _to_response(result, round(time.time() - t0, 2))
