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
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from auth.security import Principal, get_current_user
from database.connection import get_db

logger = logging.getLogger(__name__)

ci_router = APIRouter(tags=["CI/CD"])


# ── helpers REST GitHub ────────────────────────────────────────────────────────

def _gh(token: str, method: str, url: str, data: dict | None = None):
    """Appel GitHub REST API minimal via urllib."""
    import json as _json
    headers = {
        "Authorization": f"token {token}",
        "Accept":        "application/vnd.github.v3+json",
        "Content-Type":  "application/json",
    }
    body = _json.dumps(data).encode() if data else None
    req  = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return _json.loads(resp.read().decode())

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
    branch:      str           = Field("",  description="Branch to match PRs against (optional)")
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


# ── PostgreSQL background save helper ────────────────────────────────────────

def _pg_save_cicd_report(
    user_email: str,
    github_slug: str,
    outcome: str = "",
    failure_type: str = "",
    stage_failed: str = "",
    root_cause: str = "",
    pr_number: Optional[int] = None,
    elapsed_seconds: float = 0.0,
    raw_data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Background-thread write to PostgreSQL (Supabase).
    Sync psycopg2 — no asyncio, no event-loop conflict.
    Guard: skipped for local/anonymous dev mode.
    """
    if not user_email or user_email == "local@localhost":
        return

    def _write() -> None:
        try:
            import hashlib
            from database.connection import SyncSessionLocal
            from database.models import User, Project, CICDReport
            from sqlalchemy import select

            path_hash = hashlib.sha256(github_slug.lower().encode()).hexdigest()[:12]
            p_name    = github_slug.split("/")[-1] or "unknown"

            with SyncSessionLocal() as db:
                # 1. upsert user by email
                pg_user = db.execute(
                    select(User).where(User.email == user_email)
                ).scalar_one_or_none()
                if not pg_user:
                    pg_user = User(
                        email=user_email,
                        name=user_email.split("@")[0],
                        role="Developer",
                    )
                    db.add(pg_user)
                    db.flush()

                # 2. get or create project row
                project = db.execute(
                    select(Project).where(
                        Project.owner_id  == pg_user.id,
                        Project.path_hash == path_hash,
                    )
                ).scalar_one_or_none()
                if not project:
                    project = Project(
                        owner_id   = pg_user.id,
                        name       = p_name,
                        path_hash  = path_hash,
                        local_path = github_slug,
                    )
                    db.add(project)
                    db.flush()

                # 3. persist cicd_report
                import uuid as _uuid_mod
                from datetime import datetime, timezone
                report_id = _uuid_mod.uuid4().hex
                cicd_row = CICDReport(
                    id              = report_id,
                    project_id      = project.id,
                    repo            = github_slug,
                    outcome         = outcome or None,
                    failure_type    = failure_type or None,
                    stage_failed    = stage_failed or None,
                    root_cause      = root_cause[:4000] if root_cause else None,
                    pr_number       = pr_number,
                    elapsed_seconds = elapsed_seconds,
                    raw_data        = raw_data or {},
                )
                db.add(cicd_row)
                db.flush()

                # 4. emit history event
                from database.models import HistoryEvent
                _now = datetime.now(timezone.utc)
                _severity = "error" if outcome == "failure" else "info"
                _title = f"CI/CD analysis — {github_slug}"
                _summary = (
                    f"Stage {stage_failed} failed: {root_cause[:120]}"
                    if outcome == "failure" and stage_failed
                    else f"Pipeline {outcome or 'analyzed'} — {github_slug}"
                )
                db.add(HistoryEvent(
                    id            = _uuid_mod.uuid4().hex,
                    user_id       = pg_user.id,
                    project_id    = project.id,
                    event_type    = "cicd_analyzed",
                    source_module = "cicd",
                    source_id     = report_id,
                    title         = _title,
                    summary       = _summary[:300],
                    severity      = _severity,
                    status        = "completed",
                    metadata_     = {"repo": github_slug, "outcome": outcome},
                    created_at    = _now,
                ))
                db.commit()

                # 5. invalidate redis cache + notify WS clients
                try:
                    from services import history_cache_service as _hcs
                    _hcs.invalidate(pg_user.id)
                except Exception:
                    pass
                try:
                    from api.ws_broadcast import broadcast_from_thread
                    broadcast_from_thread({"type": "history_update", "module": "cicd"})
                except Exception:
                    pass

        except Exception as _exc:
            logger.debug("ci_router: PG save failed: %s", _exc)

    threading.Thread(target=_write, daemon=True).start()


# ── CI Analyze ────────────────────────────────────────────────────────────────

@ci_router.post("/ci/analyze", response_model=CIAnalyzeResponse, summary="Analyze a CI run")
async def ci_analyze(req: CIAnalyzeRequest, user: Principal = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """
    Lance une analyse d'un run GitHub Actions via le CIGraph IA.
    Si run_id est vide, détecte automatiquement le dernier run du repo.
    Poste un commentaire structuré sur la PR si pr_number est fourni.
    Utilise le token OAuth GitHub du développeur connecté (pas GITHUB_TOKEN env var).
    """
    import json as _json
    from database.repositories.user_repo import UserRepo

    parts     = req.repo.split("/")
    owner     = parts[0]
    repo_name = parts[-1]
    run_id    = req.run_id

    async with db.begin():
        token = await UserRepo(db).get_github_token_by_email(user.email)

    if not token:
        raise HTTPException(403, "GitHub account not connected. Connect via /api/auth/github/start.")

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

    elapsed = round(time.time() - t0, 2)

    _pg_save_cicd_report(
        user_email      = user.email,
        github_slug     = req.repo,
        outcome         = result.get("outcome", ""),
        failure_type    = result.get("failure_type", ""),
        stage_failed    = result.get("stage_failed", "") or "",
        root_cause      = result.get("root_cause", "") or "",
        pr_number       = result.get("pr_number"),
        elapsed_seconds = elapsed,
        raw_data        = result,
    )

    return CIAnalyzeResponse(
        outcome            = result.get("outcome", ""),
        failure_type       = result.get("failure_type", ""),
        stage_failed       = result.get("stage_failed", "") or "",
        root_cause         = result.get("root_cause", "") or "",
        pr_number          = result.get("pr_number"),
        comment_posted     = bool(result.get("comment_posted")),
        notification_level = result.get("notification_level", ""),
        elapsed_seconds    = elapsed,
    )


# ── CI Polling ────────────────────────────────────────────────────────────────

@ci_router.post("/ci/poll/start", summary="Start CI polling")
async def ci_poll_start(req: CIPollStartRequest, user: Principal = Depends(get_current_user)):
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
async def ci_poll_stop(user: Principal = Depends(get_current_user)):
    """Arrête la boucle de polling CI."""
    global _ci_poller_thread
    _ci_poller_stop.set()
    if _ci_poller_thread:
        _ci_poller_thread.join(timeout=5)
        _ci_poller_thread = None
    return {"status": "stopped"}


@ci_router.get("/ci/poll/status", summary="CI polling status")
async def ci_poll_status(user: Principal = Depends(get_current_user)):
    """Retourne l'état du poller CI."""
    running = bool(_ci_poller_thread and _ci_poller_thread.is_alive())
    return {"running": running, "stop_requested": _ci_poller_stop.is_set()}


# ── CI: Declare Pipeline (push files + workflow_dispatch) ────────────────────

class CIDeclareRequest(BaseModel):
    repo:      str           = Field(..., description="owner/repo")
    pr_number: Optional[int] = Field(None)
    branch:    str           = Field("main", description="Head branch of the selected PR")
    sha:       str           = Field("", description="Head commit SHA of the selected PR")

class CIDeclareResponse(BaseModel):
    files_pushed:   List[str]     = Field(default_factory=list)
    run_id:         Optional[str] = None
    run_url:        str           = ""
    coherence_ok:   bool          = True
    coherence_notes: List[str]    = Field(default_factory=list)
    replaced:       List[str]     = Field(default_factory=list)
    elapsed_seconds: float        = 0.0


@ci_router.post("/ci/declare", response_model=CIDeclareResponse, summary="Declare CI/CD pipeline on PR branch")
async def ci_declare(
    req: CIDeclareRequest,
    user: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Pour une PR sélectionnée :
      1. Détecte le profil du projet (langage, build system) via GitHub Contents API
      2. Pour chaque fichier (ci.yml, Dockerfile, docker-compose.yml) :
         - S'il existe : vérifie la cohérence avec le profil détecté
         - S'il est incohérent ou absent : génère + pousse sur la branche de la PR
      3. Déclenche le workflow via workflow_dispatch sur la branche de la PR
      4. Retourne le run_id pour le polling temps réel
    Utilise le token OAuth GitHub du développeur connecté (pas GITHUB_TOKEN env var).
    """
    import base64 as _b64
    import json as _json

    from database.repositories.user_repo import UserRepo

    async with db.begin():
        token = await UserRepo(db).get_github_token_by_email(user.email)

    if not token:
        raise HTTPException(403, "GitHub account not connected. Connect via /api/auth/github/start.")

    parts     = req.repo.split("/", 1)
    owner     = parts[0]
    repo_name = parts[1] if len(parts) > 1 else parts[0]
    branch    = req.branch or "main"
    t0        = time.time()

    # ── helpers locaux ────────────────────────────────────────────────────────

    def _fetch_file(path: str) -> str:
        try:
            url  = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{path}?ref={branch}"
            data = _gh(token, "GET", url)
            if data.get("encoding") == "base64":
                return _b64.b64decode(data["content"]).decode("utf-8", errors="replace")
        except Exception:
            pass
        return ""

    def _fetch_file_sha(path: str) -> Optional[str]:
        """Retourne le SHA blob GitHub du fichier (nécessaire pour update)."""
        try:
            url  = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{path}?ref={branch}"
            data = _gh(token, "GET", url)
            return data.get("sha")
        except Exception:
            return None

    def _list_files() -> list:
        try:
            url  = f"https://api.github.com/repos/{owner}/{repo_name}/git/trees/{branch}?recursive=1"
            data = _gh(token, "GET", url)
            return [item["path"] for item in data.get("tree", []) if item.get("type") == "blob"]
        except Exception:
            return []

    def _push_file(path: str, content: str, message: str) -> bool:
        encoded = _b64.b64encode(content.encode()).decode()
        payload: dict = {"message": message, "content": encoded, "branch": branch}
        existing_sha = _fetch_file_sha(path)
        if existing_sha:
            payload["sha"] = existing_sha
        try:
            _gh(token, "PUT",
                f"https://api.github.com/repos/{owner}/{repo_name}/contents/{path}",
                payload)
            return True
        except Exception as e:
            logger.warning("Push failed %s: %s", path, e)
            return False

    # ── 1. Détection du profil projet ─────────────────────────────────────────

    from ci_cd.workflow_generator import (
        detect_project_profile,
        generate_workflow,
        generate_dockerfile,
        generate_docker_compose,
        validate_workflow_strict,
    )

    profile = await asyncio.to_thread(detect_project_profile, _fetch_file, _list_files)

    # ── 2. Cohérence check + push pour chaque fichier ────────────────────────

    FILES = [
        {
            "path":    ".github/workflows/ci.yml",
            "name":    "ci.yml",
            "commit":  "ci(declare): add/update CI/CD pipeline — Code Auditor",
            "generate": lambda: generate_workflow(profile, enable_publish=True, enable_deploy=False),
        },
        {
            "path":    "Dockerfile",
            "name":    "Dockerfile",
            "commit":  "ci(declare): add/update Dockerfile — Code Auditor",
            "generate": lambda: generate_dockerfile(profile),
        },
        {
            "path":    "docker-compose.yml",
            "name":    "docker-compose.yml",
            "commit":  "ci(declare): add/update docker-compose.yml — Code Auditor",
            "generate": lambda: generate_docker_compose(profile),
        },
    ]

    files_pushed:    List[str] = []
    replaced:        List[str] = []
    coherence_notes: List[str] = []

    for spec in FILES:
        existing = await asyncio.to_thread(_fetch_file, spec["path"])
        needs_push = False

        if existing:
            # Cohérence check selon le fichier
            if spec["name"] == "ci.yml":
                ok, errors = validate_workflow_strict(existing)
                # Vérifier aussi que le langage correspond
                lang_hint = profile.language.lower()
                if lang_hint not in ("unknown",) and lang_hint not in existing.lower():
                    errors.append(f"Language '{profile.language}' not found in existing ci.yml")
                if not ok or errors:
                    coherence_notes.extend(errors)
                    needs_push = True
                    replaced.append(spec["name"])
            elif spec["name"] == "Dockerfile":
                # Vérifier que la base image correspond au langage
                lang_images = {
                    "java":       ["eclipse-temurin", "openjdk", "amazoncorretto"],
                    "python":     ["python:"],
                    "javascript": ["node:"],
                    "typescript": ["node:"],
                }
                expected = lang_images.get(profile.language.lower(), [])
                if expected and not any(img in existing for img in expected):
                    coherence_notes.append(
                        f"Dockerfile base image doesn't match detected language '{profile.language}'"
                    )
                    needs_push = True
                    replaced.append(spec["name"])
            elif spec["name"] == "docker-compose.yml":
                # Vérifier que les services correspondent au stack
                lang_services = {"java": "postgres", "python": "redis"}
                expected_svc = lang_services.get(profile.language.lower())
                if expected_svc and expected_svc not in existing:
                    coherence_notes.append(
                        f"docker-compose.yml missing expected service '{expected_svc}' for {profile.language}"
                    )
                    needs_push = True
                    replaced.append(spec["name"])
        else:
            needs_push = True

        if needs_push:
            new_content = await asyncio.to_thread(spec["generate"])
            pushed = await asyncio.to_thread(_push_file, spec["path"], new_content, spec["commit"])
            if pushed:
                files_pushed.append(spec["name"])

    # ── 3. workflow_dispatch pour déclencher le pipeline ─────────────────────

    run_id:  Optional[str] = None
    run_url: str           = ""

    def _dispatch_and_get_run() -> Optional[dict]:
        # Trouver le workflow ci.yml
        try:
            wf_url  = f"https://api.github.com/repos/{owner}/{repo_name}/actions/workflows"
            wf_data = _gh(token, "GET", wf_url)
            workflows = wf_data.get("workflows", [])
            ci_wf = next(
                (w for w in workflows if "ci.yml" in w.get("path", "") or "ci" in w.get("name", "").lower()),
                None
            )
            if not ci_wf:
                return None
            wf_id = ci_wf["id"]
        except Exception as e:
            logger.warning("Cannot find workflow: %s", e)
            return None

        # Déclencher workflow_dispatch
        try:
            _gh(token, "POST",
                f"https://api.github.com/repos/{owner}/{repo_name}/actions/workflows/{wf_id}/dispatches",
                {"ref": branch, "inputs": {}})
        except Exception as e:
            logger.warning("workflow_dispatch failed: %s", e)

        # Attendre 3s puis récupérer le run_id créé
        import time as _t
        _t.sleep(3)
        try:
            runs_url = (
                f"https://api.github.com/repos/{owner}/{repo_name}/actions/runs"
                f"?branch={urllib.parse.quote(branch)}&per_page=5&event=workflow_dispatch"
            )
            runs_data = _gh(token, "GET", runs_url)
            runs = runs_data.get("workflow_runs", [])
            if runs:
                latest = runs[0]
                return {"run_id": str(latest["id"]), "run_url": latest.get("html_url", "")}
        except Exception as e:
            logger.warning("Cannot fetch run after dispatch: %s", e)
        return None

    dispatch_result = await asyncio.to_thread(_dispatch_and_get_run)
    if dispatch_result:
        run_id  = dispatch_result["run_id"]
        run_url = dispatch_result["run_url"]

    return CIDeclareResponse(
        files_pushed    = files_pushed,
        run_id          = run_id,
        run_url         = run_url,
        coherence_ok    = len(coherence_notes) == 0,
        coherence_notes = coherence_notes,
        replaced        = replaced,
        elapsed_seconds = round(time.time() - t0, 2),
    )


# ── CI: Run status (polling temps réel) ───────────────────────────────────────

class CIJobStatus(BaseModel):
    name:       str
    status:     str   # queued | in_progress | completed
    conclusion: Optional[str] = None  # success | failure | skipped | cancelled
    step_id:    str   # mapped to PipelineStepId

class CIRunStatusResponse(BaseModel):
    run_id:     str
    status:     str           # queued | in_progress | completed
    conclusion: Optional[str] = None
    jobs:       List[CIJobStatus] = Field(default_factory=list)
    run_url:    str = ""


@ci_router.get("/ci/run-status", response_model=CIRunStatusResponse, summary="Poll GitHub Actions run status")
async def ci_run_status(
    repo:   str,
    run_id: str,
    user: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Lit l'état en temps réel d'un run GitHub Actions.
    Retourne le statut global du run + chaque job mappé à un PipelineStepId.
    Le frontend poll cet endpoint toutes les 5s pendant que le run tourne.
    """
    import json as _json

    from database.repositories.user_repo import UserRepo

    async with db.begin():
        token = await UserRepo(db).get_github_token_by_email(user.email)

    if not token:
        raise HTTPException(403, "GitHub account not connected.")

    parts     = repo.split("/", 1)
    owner     = parts[0]
    repo_name = parts[1] if len(parts) > 1 else parts[0]

    # Map GitHub job name → PipelineStepId
    def _job_to_step(job_name: str) -> str:
        n = job_name.lower()
        if "build" in n or "test" in n:           return "build"
        if "sonar" in n or "quality" in n:        return "sonar"
        if "codeql" in n or "dep" in n:           return "security"
        if "trivy" in n or "docker" in n:         return "docker"
        if "publish" in n or "push" in n:         return "docker"
        if "deploy" in n or "ssh" in n:           return "deploy"
        return "build"

    def _fetch_run_and_jobs() -> dict:
        # Run global
        run_data = _gh(token, "GET",
            f"https://api.github.com/repos/{owner}/{repo_name}/actions/runs/{run_id}")
        # Jobs du run
        jobs_data = _gh(token, "GET",
            f"https://api.github.com/repos/{owner}/{repo_name}/actions/runs/{run_id}/jobs")
        return {"run": run_data, "jobs": jobs_data.get("jobs", [])}

    try:
        raw = await asyncio.to_thread(_fetch_run_and_jobs)
    except Exception as e:
        raise HTTPException(502, f"GitHub API error: {e}")

    run  = raw["run"]
    jobs = raw["jobs"]

    return CIRunStatusResponse(
        run_id     = run_id,
        status     = run.get("status", "queued"),
        conclusion = run.get("conclusion"),
        run_url    = run.get("html_url", ""),
        jobs       = [
            CIJobStatus(
                name       = j.get("name", ""),
                status     = j.get("status", "queued"),
                conclusion = j.get("conclusion"),
                step_id    = _job_to_step(j.get("name", "")),
            )
            for j in jobs
        ],
    )


# ── CI: list open PRs ─────────────────────────────────────────────────────────

class PrItem(BaseModel):
    number:     int
    title:      str
    branch:     str   # head ref
    author:     str
    updated_at: str
    sha:        str   # head commit SHA

@ci_router.get("/ci/prs", response_model=List[PrItem], summary="List open PRs for a repo")
async def list_open_prs(
    repo: str,
    user: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns open PRs for the given repo using the authenticated user's GitHub token.
    Each developer uses their own OAuth token — not a shared GITHUB_TOKEN.
    """
    import json as _json
    from database.repositories.user_repo import UserRepo

    async with db.begin():
        token = await UserRepo(db).get_github_token_by_email(user.email)

    if not token:
        raise HTTPException(403, "GitHub account not connected. Connect via /api/auth/github/start.")

    parts     = repo.split("/")
    owner     = parts[0]
    repo_name = parts[-1] if len(parts) > 1 else parts[0]

    def _fetch_prs() -> List[dict]:
        url = (
            f"https://api.github.com/repos/{owner}/{repo_name}/pulls"
            f"?state=open&sort=updated&direction=desc&per_page=50"
        )
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"token {token}")
        req.add_header("Accept", "application/vnd.github.v3+json")
        with urllib.request.urlopen(req, timeout=15) as r:
            return _json.loads(r.read().decode())

    try:
        raw = await asyncio.to_thread(_fetch_prs)
    except Exception as e:
        logger.error("list_open_prs error: %s", e)
        raise HTTPException(502, f"GitHub API error: {e}")

    return [
        PrItem(
            number     = p["number"],
            title      = p.get("title", ""),
            branch     = p.get("head", {}).get("ref", ""),
            author     = p.get("user", {}).get("login", ""),
            updated_at = p.get("updated_at", ""),
            sha        = p.get("head", {}).get("sha", ""),
        )
        for p in raw
    ]


# ── CD Score ──────────────────────────────────────────────────────────────────

@ci_router.post("/cd/score", response_model=CDScoreResponse, summary="Release Readiness Score")
async def cd_score(
    req: CDScoreRequest,
    user: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Calcule le Release Readiness Score avant un déploiement.
    Agrège CI, SonarCloud, sécurité, PR approvals, risk fichiers.
    Verdict : DEPLOY_OK | DEPLOY_WARN | DEPLOY_BLOCKED

    Uses the authenticated user's GitHub OAuth token — any developer
    can run the analysis against their own connected account.
    """
    from database.repositories.user_repo import UserRepo

    parts     = req.repo.split("/")
    owner     = parts[0]

    # Resolve the user's own GitHub token (multi-developer support)
    async with db.begin():
        gh_token = await UserRepo(db).get_github_token_by_email(user.email)

    if not gh_token:
        raise HTTPException(403, "GitHub account not connected. Connect via /api/auth/github/start.")

    t0 = time.time()
    try:
        from ci_cd.cd_release_scorer import CDReleaseScorer
        scorer = CDReleaseScorer(token=gh_token)
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

    elapsed = round(time.time() - t0, 2)

    _pg_save_cicd_report(
        user_email      = user.email,
        github_slug     = req.repo,
        outcome         = report.verdict,
        failure_type    = "deploy_blocked" if report.verdict == "DEPLOY_BLOCKED" else "",
        root_cause      = "; ".join(report.blocking_reasons) if report.blocking_reasons else "",
        pr_number       = req.pr_number,
        elapsed_seconds = elapsed,
        raw_data        = {
            "verdict":          report.verdict,
            "score":            report.score,
            "component_scores": report.component_scores,
            "blocking_reasons": report.blocking_reasons,
            "warnings":         report.warnings,
        },
    )

    return CDScoreResponse(
        verdict          = report.verdict,
        score            = report.score,
        component_scores = report.component_scores,
        blocking_reasons = report.blocking_reasons,
        warnings         = report.warnings,
        elapsed_seconds  = elapsed,
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


# ── Generate Files ─────────────────────────────────────────────────────────────

class GeneratedFile(BaseModel):
    name:    str
    path:    str
    content: str
    label:   str
    stage:   str   # "generate" | "docker"

class ProjectDetection(BaseModel):
    language:       str   = "unknown"
    build_system:   str   = "unknown"
    java_version:   str   = "17"
    python_version: str   = "3.11"
    node_version:   str   = "20"
    has_dockerfile: bool  = False

class GenerateFilesRequest(BaseModel):
    repo:   str = Field(..., description="owner/repo")
    branch: str = Field("main", description="Branch to analyze for project detection")

class GenerateFilesResponse(BaseModel):
    files:           List[GeneratedFile]         = Field(default_factory=list)
    project:         Optional[ProjectDetection]  = None
    elapsed_seconds: float                       = 0.0


@ci_router.post("/ci/generate-files", response_model=GenerateFilesResponse,
                summary="Detect project + generate ci.yml, Dockerfile, docker-compose.yml")
async def ci_generate_files(req: GenerateFilesRequest):
    """
    1. Fetches candidate build files from the repo via GitHub Contents API.
    2. Detects the project profile (language, build system, versions).
    3. Generates ci.yml (7-job workflow) + Dockerfile + docker-compose.yml.
    Returns content only — nothing is pushed to the repo yet.
    """
    import os
    import base64 as _b64
    import json as _json
    from dotenv import load_dotenv

    load_dotenv(override=True)

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    if not token:
        raise HTTPException(401, "GITHUB_TOKEN manquant dans .env")

    parts     = req.repo.split("/")
    owner     = parts[0]
    repo_name = parts[-1]
    branch    = req.branch or "main"

    t0 = time.time()

    # ── Detect project profile via GitHub Contents API ────────────────────────

    def _fetch_file(path: str) -> str:
        try:
            url = f"https://api.github.com/repos/{owner}/{repo_name}/contents/{path}?ref={branch}"
            data = _gh(token, "GET", url)
            if data.get("encoding") == "base64":
                return _b64.b64decode(data["content"]).decode("utf-8", errors="replace")
        except Exception:
            pass
        return ""

    def _list_files() -> list[str]:
        try:
            url  = f"https://api.github.com/repos/{owner}/{repo_name}/git/trees/{branch}?recursive=1"
            data = _gh(token, "GET", url)
            return [item["path"] for item in data.get("tree", []) if item.get("type") == "blob"]
        except Exception:
            return []

    try:
        from ci_cd.workflow_generator import (
            detect_project_profile,
            generate_workflow,
            generate_dockerfile,
            generate_docker_compose,
            ProjectProfile,
        )
    except ImportError as e:
        raise HTTPException(500, f"workflow_generator import error: {e}")

    profile = await asyncio.to_thread(
        detect_project_profile, _fetch_file, _list_files
    )

    # ── Generate files ────────────────────────────────────────────────────────

    yaml_content    = generate_workflow(profile, enable_publish=True, enable_deploy=False)
    dockerfile      = generate_dockerfile(profile)
    docker_compose  = generate_docker_compose(profile)

    # Human-readable labels
    lang_label = f"{profile.language.capitalize()}"
    if profile.build_system not in ("unknown", profile.language):
        lang_label += f"/{profile.build_system}"

    files: List[GeneratedFile] = [
        GeneratedFile(
            name    = "ci.yml",
            path    = ".github/workflows/ci.yml",
            content = yaml_content,
            label   = f"7 jobs · {lang_label}",
            stage   = "generate",
        ),
        GeneratedFile(
            name    = "Dockerfile",
            path    = "Dockerfile",
            content = dockerfile,
            label   = f"multi-stage · {lang_label}",
            stage   = "docker",
        ),
        GeneratedFile(
            name    = "docker-compose.yml",
            path    = "docker-compose.yml",
            content = docker_compose,
            label   = _compose_label(profile),
            stage   = "docker",
        ),
    ]

    detection = ProjectDetection(
        language       = profile.language,
        build_system   = profile.build_system,
        java_version   = profile.java_version,
        python_version = profile.python_version,
        node_version   = profile.node_version,
        has_dockerfile = profile.has_dockerfile,
    )

    return GenerateFilesResponse(
        files           = files,
        project         = detection,
        elapsed_seconds = round(time.time() - t0, 2),
    )


def _compose_label(profile) -> str:
    lang = profile.language.lower()
    if lang == "java":
        return "app + postgres"
    if lang == "python":
        return "app + redis"
    return "app"


# ── Open PR ───────────────────────────────────────────────────────────────────

class OpenPRFile(BaseModel):
    path:    str
    content: str
    name:    str = ""
    stage:   str = ""   # "generate" | "docker"

class OpenPRRequest(BaseModel):
    repo:     str            = Field(..., description="owner/repo")
    files:    List[OpenPRFile]
    base:     str            = Field("main",  description="Target branch for the PR")
    branch:   str            = Field("",      description="Feature branch (auto if empty)")
    pr_title: str            = Field("",      description="PR title (auto if empty)")
    pr_body:  str            = Field("",      description="PR body (auto if empty)")

class OpenPRResponse(BaseModel):
    pr_url:         str
    branch:         str
    files_pushed:   int
    elapsed_seconds: float = 0.0


@ci_router.post("/ci/open-pr", response_model=OpenPRResponse,
                summary="Create feature branch, push generated files, open PR")
async def ci_open_pr(req: OpenPRRequest):
    """
    GitOps-compliant file delivery:
      1. Resolve default branch HEAD SHA.
      2. Create feature/ci-cd-setup-{timestamp} branch.
      3. Push each file via GitHub Contents API (create or update).
      4. Open PR against `base` with LLM-friendly description.
    """
    import os
    import base64 as _b64
    import json as _json
    from datetime import datetime
    from dotenv import load_dotenv

    load_dotenv(override=True)

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    if not token:
        raise HTTPException(401, "GITHUB_TOKEN manquant dans .env")

    parts     = req.files  # keep name
    owner, repo_name = req.repo.split("/", 1)
    base      = req.base or "main"
    t0        = time.time()

    # ── 1. Get HEAD SHA of base branch ───────────────────────────────────────
    try:
        ref_data = _gh(token, "GET",
            f"https://api.github.com/repos/{owner}/{repo_name}/git/refs/heads/{base}")
        head_sha = ref_data["object"]["sha"]
    except Exception as e:
        raise HTTPException(500, f"Impossible de lire la branche '{base}': {e}")

    # ── 2. Create feature branch ──────────────────────────────────────────────
    timestamp   = datetime.utcnow().strftime("%Y%m%d-%H%M")
    new_branch  = req.branch or f"feature/ci-cd-setup-{timestamp}"

    try:
        _gh(token, "POST",
            f"https://api.github.com/repos/{owner}/{repo_name}/git/refs",
            {"ref": f"refs/heads/{new_branch}", "sha": head_sha})
    except urllib.error.HTTPError as e:
        if e.code == 422:
            pass   # Branch already exists — reuse it
        else:
            raise HTTPException(500, f"Création branche '{new_branch}' échouée: {e}")

    # ── 3. Push each file in pipeline stage order ─────────────────────────────
    # generate → docker ensures git history mirrors the pipeline execution order.
    _STAGE_ORDER = {"generate": 0, "docker": 1}
    ordered_files = sorted(req.files, key=lambda f: _STAGE_ORDER.get(f.stage, 99))

    _COMMIT_MSGS: dict[str, str] = {
        "ci.yml":           "ci(generate): add .github/workflows/ci.yml — GitHub Actions workflow",
        "Dockerfile":       "ci(docker): add Dockerfile — container build",
        "docker-compose.yml": "ci(docker): add docker-compose.yml — production orchestration",
    }

    pushed = 0
    for f in ordered_files:
        filename = f.name or f.path.split("/")[-1]
        commit_msg = _COMMIT_MSGS.get(filename, f"ci({f.stage or 'pipeline'}): add {filename}")
        encoded = _b64.b64encode(f.content.encode()).decode()
        payload: dict = {
            "message": commit_msg,
            "content": encoded,
            "branch":  new_branch,
        }
        # Check if file exists (need SHA for update)
        try:
            existing = _gh(token, "GET",
                f"https://api.github.com/repos/{owner}/{repo_name}/contents/{f.path}?ref={new_branch}")
            payload["sha"] = existing["sha"]
        except Exception:
            pass   # File doesn't exist → create

        try:
            _gh(token, "PUT",
                f"https://api.github.com/repos/{owner}/{repo_name}/contents/{f.path}",
                payload)
            pushed += 1
        except Exception as e:
            logger.warning("Push failed for %s: %s", f.path, e)

    if pushed == 0:
        raise HTTPException(500, "Aucun fichier n'a pu être poussé sur GitHub")

    # ── 4. Open PR ────────────────────────────────────────────────────────────
    pr_title = req.pr_title or f"ci: add Code Auditor CI/CD pipeline ({pushed} files)"
    pr_body  = req.pr_body  or _default_pr_body(req.files, new_branch)

    try:
        pr_data = _gh(token, "POST",
            f"https://api.github.com/repos/{owner}/{repo_name}/pulls",
            {"title": pr_title, "body": pr_body, "head": new_branch, "base": base})
        pr_url = pr_data.get("html_url", "")
    except urllib.error.HTTPError as e:
        body_txt = e.read().decode() if hasattr(e, "read") else str(e)
        if "already exists" in body_txt or e.code == 422:
            # PR already open for this branch — find it
            try:
                prs = _gh(token, "GET",
                    f"https://api.github.com/repos/{owner}/{repo_name}/pulls?head={owner}:{new_branch}&state=open")
                pr_url = prs[0]["html_url"] if prs else ""
            except Exception:
                pr_url = ""
        else:
            raise HTTPException(500, f"Création PR échouée: {e.code} {body_txt[:200]}")

    return OpenPRResponse(
        pr_url          = pr_url,
        branch          = new_branch,
        files_pushed    = pushed,
        elapsed_seconds = round(time.time() - t0, 2),
    )


def _default_pr_body(files: "List[OpenPRFile]", branch: str) -> str:
    file_list = "\n".join(f"- `{f.path}`" for f in files)
    return f"""## CI/CD Pipeline — Code Auditor

This PR was automatically generated by **Code Auditor AI** after analyzing your project.

### Files added
{file_list}

### What was detected
- Language and build system auto-detected from your repository
- Workflow tailored to your stack (Java/Python/Node.js)
- Dockerfile uses multi-stage build for minimal image size
- docker-compose.yml includes appropriate backing services

### Next steps
1. Review each generated file
2. Add required GitHub Secrets (see `ci.yml` header comments)
3. Merge this PR — the pipeline will trigger automatically

> Generated by [Code Auditor](https://github.com/chmaryem/code_auditor) on branch `{branch}`
"""
