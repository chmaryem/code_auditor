"""
api/ci_router.py — FastAPI router for CIGraph + CDGraph.

Endpoints:
  POST /ci/analyze          → analyze a specific CI run (CIGraph)
  POST /ci/poll/start       → start background CI polling
  POST /ci/poll/stop        → stop CI polling
  GET  /ci/poll/status      → polling status
  POST /cd/score            → Release Readiness Score (CDGraph)
  GET  /cd/status           → current deploy status for an environment
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

ci_router = APIRouter(tags=["CI/CD"])

# Shared polling state
_ci_poller_thread: Optional[threading.Thread] = None
_ci_poller_stop   = threading.Event()


# ── Pydantic models ───────────────────────────────────────────────────────────

class CIAnalyzeRequest(BaseModel):
    repo:        str = Field(...,  description="owner/repo")
    run_id:      str = Field("",  description="GitHub Actions run ID (auto-detected if empty)")
    pr_number:   Optional[int] = Field(None, description="PR number to comment on")
    branch:      str = Field("",  description="Branch filter for run auto-detection")
    project_key: str = Field("",  description="SonarCloud project key (optional)")

class CIAnalyzeResponse(BaseModel):
    outcome:            str  = ""
    failure_type:       str  = ""
    stage_failed:       str  = ""
    root_cause:         str  = ""
    pr_number:          Optional[int] = None
    comment_posted:     bool = False
    notification_level: str  = ""
    elapsed_seconds:    float = 0.0

class CIPollStartRequest(BaseModel):
    repo:        str = Field(...,  description="owner/repo")
    interval:    int = Field(120,  description="Polling interval in seconds")
    branch:      str = Field("",  description="Branch filter")
    project_key: str = Field("",  description="SonarCloud project key")

class CDScoreRequest(BaseModel):
    repo:        str           = Field(..., description="owner/repo")
    sha:         str           = Field("",  description="Commit SHA (HEAD if empty)")
    pr_number:   Optional[int] = Field(None)
    project_key: str           = Field("")
    environment: str           = Field("production")
    run_id:      str           = Field("")

class CDScoreResponse(BaseModel):
    verdict:          str              = ""
    score:            float            = 0.0
    component_scores: Dict[str, float] = Field(default_factory=dict)
    blocking_reasons: List[str]        = Field(default_factory=list)
    warnings:         List[str]        = Field(default_factory=list)
    elapsed_seconds:  float            = 0.0

class CDStatusRequest(BaseModel):
    repo:        str = Field(..., description="owner/repo")
    environment: str = Field("production")


# ── CI Analyze ────────────────────────────────────────────────────────────────

@ci_router.post("/ci/analyze", response_model=CIAnalyzeResponse, summary="Analyze a CI run")
async def ci_analyze(req: CIAnalyzeRequest):
    """
    Lance une analyse d'un run GitHub Actions via le CIGraph IA.
    Si run_id est vide, détecte automatiquement le dernier run du repo.
    Poste un commentaire structuré sur la PR si pr_number est fourni.
    """
    import os
    import urllib.request
    import urllib.parse
    import json as _json
    from dotenv import load_dotenv

    load_dotenv(override=True)

    parts     = req.repo.split("/")
    owner     = parts[0]
    repo_name = parts[-1]
    run_id    = req.run_id
    token     = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")

    if not token:
        raise HTTPException(401, "GITHUB_TOKEN manquant dans .env")

    # Auto-detect run_id
    if not run_id:
        def _fetch(branch_filter=""):
            params = {"status": "completed", "per_page": "20"}
            if branch_filter:
                params["branch"] = branch_filter
            url = (
                f"https://api.github.com/repos/{owner}/{repo_name}/actions/runs"
                f"?{urllib.parse.urlencode(params)}"
            )
            r = urllib.request.Request(url)
            r.add_header("Authorization", f"token {token}")
            r.add_header("Accept", "application/vnd.github.v3+json")
            with urllib.request.urlopen(r, timeout=15) as resp:
                return _json.loads(resp.read().decode()).get("workflow_runs", [])

        try:
            runs = _fetch(req.branch) or _fetch("")
            if not runs:
                raise HTTPException(404, f"Aucun run CI trouvé pour {req.repo}")
            run = runs[0]
            if req.pr_number:
                for r in runs:
                    if any(p.get("number") == req.pr_number for p in r.get("pull_requests", [])):
                        run = r
                        break
            run_id = str(run["id"])
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Erreur détection run : {e}")

    t0 = time.time()
    try:
        from langchain_agents.graphs.ci_graph import invoke_ci_run
        result = await asyncio.to_thread(
            invoke_ci_run,
            run_id=run_id,
            repo=req.repo,
            owner=owner,
            project_key=req.project_key,
            pr_number=req.pr_number,
            pr_branch=req.branch,
            run_conclusion="",
        )
    except Exception as e:
        logger.exception("ci/analyze error")
        raise HTTPException(500, f"CIGraph error: {e}")

    return CIAnalyzeResponse(
        outcome            = result.get("outcome", ""),
        failure_type       = result.get("failure_type", ""),
        stage_failed       = result.get("stage_failed", "") or "",
        root_cause         = result.get("root_cause", "") or "",
        pr_number          = result.get("pr_number"),
        comment_posted     = bool(result.get("comment_posted")),
        notification_level = result.get("notification_level", ""),
        elapsed_seconds    = round(time.time() - t0, 2),
    )


# ── CI Polling ────────────────────────────────────────────────────────────────

@ci_router.post("/ci/poll/start", summary="Start CI polling")
async def ci_poll_start(req: CIPollStartRequest):
    """
    Démarre la boucle de polling CI en arrière-plan.
    Surveille les GitHub Actions runs et lance le CIGraph à chaque run terminé.
    """
    global _ci_poller_thread, _ci_poller_stop

    if _ci_poller_thread and _ci_poller_thread.is_alive():
        return {"status": "already_running", "repo": req.repo}

    _ci_poller_stop.clear()

    def _run():
        from ci_cd.ci_poller import CIPoller
        poller = CIPoller(
            repo=req.repo,
            interval=req.interval,
            branch=req.branch,
            project_key=req.project_key,
        )
        try:
            poller.run(stop_event=_ci_poller_stop)
        except Exception as e:
            logger.error("CI poller error: %s", e)

    _ci_poller_thread = threading.Thread(target=_run, daemon=True)
    _ci_poller_thread.start()

    return {"status": "started", "repo": req.repo, "interval": req.interval}


@ci_router.post("/ci/poll/stop", summary="Stop CI polling")
async def ci_poll_stop():
    """Arrête la boucle de polling CI."""
    global _ci_poller_thread
    _ci_poller_stop.set()
    if _ci_poller_thread:
        _ci_poller_thread.join(timeout=5)
        _ci_poller_thread = None
    return {"status": "stopped"}


@ci_router.get("/ci/poll/status", summary="CI polling status")
async def ci_poll_status():
    """Retourne l'état du poller CI."""
    running = bool(_ci_poller_thread and _ci_poller_thread.is_alive())
    return {"running": running, "stop_requested": _ci_poller_stop.is_set()}


# ── CD Score ──────────────────────────────────────────────────────────────────

@ci_router.post("/cd/score", response_model=CDScoreResponse, summary="Release Readiness Score")
async def cd_score(req: CDScoreRequest):
    """
    Calcule le Release Readiness Score avant un déploiement.
    Agrège CI, SonarCloud, sécurité, PR approvals, risk fichiers.
    Verdict : DEPLOY_OK | DEPLOY_WARN | DEPLOY_BLOCKED
    """
    parts    = req.repo.split("/")
    owner    = parts[0]

    t0 = time.time()
    try:
        from ci_cd.cd_release_scorer import CDReleaseScorer
        scorer = CDReleaseScorer()
        report = await asyncio.to_thread(
            scorer.score,
            repo=req.repo,
            owner=owner,
            commit_sha=req.sha,
            run_id=req.run_id,
            project_key=req.project_key,
            pr_number=req.pr_number,
            environment=req.environment,
        )
    except Exception as e:
        logger.exception("cd/score error")
        raise HTTPException(500, f"CDReleaseScorer error: {e}")

    return CDScoreResponse(
        verdict          = report.verdict,
        score            = report.score,
        component_scores = report.component_scores,
        blocking_reasons = report.blocking_reasons,
        warnings         = report.warnings,
        elapsed_seconds  = round(time.time() - t0, 2),
    )


# ── CD Status ─────────────────────────────────────────────────────────────────

@ci_router.post("/cd/status", summary="CD environment status")
async def cd_status(req: CDStatusRequest):
    """
    Retourne l'état courant d'un environnement de déploiement :
    dernier déploiement réussi, taux de succès, déploiements récents.
    """
    try:
        from ci_cd.cd_deploy_tracker import CDDeployTracker
        import datetime

        tracker = CDDeployTracker()
        state   = await asyncio.to_thread(tracker.get_env_state,   req.repo, req.environment)
        last_ok = await asyncio.to_thread(tracker.get_last_successful_deploy, req.repo, req.environment)
        stats   = await asyncio.to_thread(tracker.get_deploy_stats, req.repo, req.environment)
        recent  = await asyncio.to_thread(tracker.get_recent_deploys, req.repo, limit=5, env=req.environment)

        def _fmt(ts):
            if not ts:
                return None
            return datetime.datetime.fromtimestamp(ts).isoformat()

        return {
            "repo":        req.repo,
            "environment": req.environment,
            "current":     state or {},
            "last_success": {
                "version":    last_ok.get("version")    if last_ok else None,
                "commit_sha": last_ok.get("commit_sha") if last_ok else None,
                "at":         _fmt(last_ok.get("success_at")) if last_ok else None,
            } if last_ok else None,
            "stats": {
                "total":           stats.get("total", 0),
                "success_rate":    stats.get("success_rate", 0),
                "avg_duration_s":  stats.get("avg_duration_s", 0),
            },
            "recent": [
                {
                    "status":  d.get("status"),
                    "version": d.get("version"),
                    "at":      _fmt(d.get("started_at")),
                }
                for d in recent
            ],
        }
    except Exception as e:
        logger.exception("cd/status error")
        raise HTTPException(500, f"CDDeployTracker error: {e}")
