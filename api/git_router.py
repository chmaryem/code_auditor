from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import os
import urllib.error
import urllib.parse
import urllib.request

import threading

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from auth.security import Principal, get_current_user
from database.connection import get_db
from database.repositories.user_repo import UserRepo

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

class GitSecretScanRequest(BaseModel):
    project_path: str = Field(..., description="Absolute path to git project")
    session_id:   str = Field("", description="Optional session ID")

class GitCommitLintRequest(BaseModel):
    project_path:   str = Field(..., description="Absolute path to git project")
    commit_message: str = Field("", description="Message to lint (empty = read COMMIT_EDITMSG)")
    session_id:     str = Field("", description="Optional session ID")

class GitTestImpactRequest(BaseModel):
    project_path:  str          = Field(..., description="Absolute path to git project")
    changed_files: List[str]    = Field(default_factory=list, description="Files to analyze (empty = staged)")
    session_id:    str          = Field("", description="Optional session ID")

class GitCommitReadinessRequest(BaseModel):
    project_path:   str = Field(..., description="Absolute path to git project")
    commit_message: str = Field("", description="Message to validate (empty = read COMMIT_EDITMSG)")
    session_id:     str = Field("", description="Optional session ID")

class GitCrossPRRequest(BaseModel):
    owner:      str = Field(..., description="GitHub owner")
    repo:       str = Field(..., description="GitHub repo name")
    base:       str = Field("main", description="Base branch to filter PRs")
    pr_number:  int = Field(0, description="Current PR to exclude (optional)")
    session_id: str = Field("", description="Optional session ID")

class GitPRDescriptionRequest(BaseModel):
    project_path: str = Field("", description="Local git project path (for local mode)")
    owner:        str = Field("", description="GitHub owner (for GitHub mode)")
    repo:         str = Field("", description="GitHub repo (for GitHub mode)")
    pr_number:    int = Field(0, description="PR number (for GitHub mode)")
    branch:       str = Field("HEAD", description="Feature branch")
    base:         str = Field("main", description="Base branch")
    session_id:   str = Field("", description="Optional session ID")

class ResolveConflictsRequest(BaseModel):
    owner:      str = Field(..., description="GitHub owner")
    repo:       str = Field(..., description="GitHub repo name")
    pr_number:  int = Field(..., description="Pull request number to resolve conflicts for")

class ApplySuggestionRequest(BaseModel):
    owner:        str = Field(..., description="GitHub owner")
    repo:         str = Field(..., description="GitHub repo name")
    pr_number:    int = Field(..., description="Pull request number")
    head_ref:     str = Field(..., description="PR head branch (e.g. 'feature/my-branch')")
    file_path:    str = Field(..., description="File path relative to repo root")
    current_code: str = Field(..., description="Exact code snippet to replace")
    fixed_code:   str = Field(..., description="Replacement code")
    hint_line:    int = Field(0,   description="Line number hint for fuzzy search anchoring (0 = no hint)")

class GitHookRequest(BaseModel):
    project_path: str = Field(..., description="Absolute path to git project")

class GitResponse(BaseModel):
    response:           str             = ""
    intent:             str             = ""
    confidence:         float           = 0.0
    safe_mode:          bool            = True
    elapsed_seconds:    float           = 0.0
    session_snapshot:   Dict[str, Any]  = Field(default_factory=dict)
    branch_report:      Dict[str, Any]  = Field(default_factory=dict)
    commit_message:     str             = ""
    changes:            Dict[str, Any]  = Field(default_factory=dict)
    conflict_report:    Dict[str, Any]  = Field(default_factory=dict)
    pr_report:          Dict[str, Any]  = Field(default_factory=dict)
    readiness_report:   Dict[str, Any]  = Field(default_factory=dict)
    secret_scan_report: Dict[str, Any]  = Field(default_factory=dict)
    commit_lint_report: Dict[str, Any]  = Field(default_factory=dict)
    test_impact_report: Dict[str, Any]  = Field(default_factory=dict)
    cross_pr_report:    Dict[str, Any]  = Field(default_factory=dict)
    pr_description:     Dict[str, Any]  = Field(default_factory=dict)
    errors:             List[str]       = Field(default_factory=list)


# ── Git context status ────────────────────────────────────────────────────────

@git_router.get("/status")
async def get_git_status(
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Return the GitHub connection context for Smart Git (repo + token presence).

    Resilient to DB outages: if PostgreSQL is unreachable, fall back to the
    .env PAT (connected=true) so Smart Git keeps working instead of 500-ing.
    """
    repo_str = None
    token    = ""
    try:
        async with db.begin():
            repo_str = await UserRepo(db).get_active_repo_by_email(principal.email)
            token    = await UserRepo(db).get_github_token_by_email(principal.email)
    except Exception as exc:
        import os as _os
        logger.warning("git/status: DB unreachable, falling back to .env PAT: %s", exc)
        token = _os.environ.get("GITHUB_TOKEN") or _os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    owner, repo = (repo_str.split("/", 1) if repo_str and "/" in repo_str else ("", ""))
    return {
        "connected": bool(token),
        "owner":     owner,
        "repo":      repo,
        "full_name": repo_str or "",
    }


# ── Helper ────────────────────────────────────────────────────────────────────

async def _invoke(message: str, **kwargs) -> Dict[str, Any]:
    from langchain_agents.graphs.smart_git_graph import ainvoke_smart_git
    return await ainvoke_smart_git(message=message, **kwargs)


def _pg_save_git_report(
    user_email: str,
    report_type: str,
    raw_data: Dict[str, Any],
    local_path: str = "",
    github_slug: str = "",
    project_name: str = "",
    branch: str = "",
    base_branch: str = "",
    pr_number: int = 0,
    verdict: str = "",
    total_score: Optional[int] = None,
    summary: str = "",
) -> None:
    """
    Background-thread write to PostgreSQL (Supabase).
    Redis cache is never touched — this is purely additive.
    Guard: skipped for local/anonymous dev mode (email == local@localhost).
    """
    if not user_email or user_email == "local@localhost":
        return

    def _write() -> None:
        try:
            import hashlib
            from database.connection import SyncSessionLocal
            from database.models import User, Project, GitReport
            from sqlalchemy import select

            path_key  = local_path or github_slug or "unknown"
            p_name    = project_name or path_key.split("/")[-1] or "unknown"
            path_hash = hashlib.sha256(path_key.lower().encode()).hexdigest()[:12]

            with SyncSessionLocal() as db:
                # 1. upsert user by email → PG-authoritative user.id
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
                        local_path = path_key,
                    )
                    db.add(project)
                    db.flush()

                # 3. persist git report
                import uuid as _uuid_mod
                from datetime import datetime, timezone
                report_id = _uuid_mod.uuid4().hex
                git_row = GitReport(
                    id          = report_id,
                    project_id  = project.id,
                    report_type = report_type,
                    branch      = branch or None,
                    base_branch = base_branch or None,
                    pr_number   = pr_number or None,
                    verdict     = verdict or None,
                    total_score = total_score,
                    summary     = summary[:2000] if summary else None,
                    raw_data    = raw_data,
                )
                db.add(git_row)
                db.flush()

                # 4. emit history event
                from database.models import HistoryEvent
                _now = datetime.now(timezone.utc)
                _type_labels = {
                    "pr_review":    "PR Review",
                    "branch":       "Branch Analysis",
                    "commit_lint":  "Commit Lint",
                    "secret_scan":  "Secret Scan",
                    "test_impact":  "Test Impact",
                }
                _label = _type_labels.get(report_type, report_type.replace("_", " ").title())
                _verdict_sev = {"APPROVED": "info", "BLOCKED": "error", "NEEDS_CHANGES": "warning"}
                _severity = _verdict_sev.get(verdict, "info") if verdict else "info"
                _title = f"{_label} — {github_slug or path_key.split('/')[-1]}"
                _summary_text = summary[:200] if summary else f"Score {total_score}/100" if total_score else ""
                db.add(HistoryEvent(
                    id            = _uuid_mod.uuid4().hex,
                    user_id       = pg_user.id,
                    project_id    = project.id,
                    event_type    = f"git_{report_type}",
                    source_module = "smart_git",
                    source_id     = report_id,
                    title         = _title,
                    summary       = _summary_text[:300],
                    severity      = _severity,
                    status        = "completed",
                    metadata_     = {
                        "report_type": report_type,
                        "verdict":     verdict,
                        "pr_number":   pr_number or None,
                        "branch":      branch or None,
                    },
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
                    broadcast_from_thread({"type": "history_update", "module": "smart_git"})
                except Exception:
                    pass

        except Exception as _exc:
            logger.debug("git_router: PG save failed (%s): %s", report_type, _exc)

    threading.Thread(target=_write, daemon=True).start()


def _to_response(result: Dict[str, Any], elapsed: float) -> GitResponse:
    return GitResponse(
        response            = result.get("response", ""),
        intent              = result.get("intent", ""),
        confidence          = float(result.get("confidence", 0.0)),
        safe_mode           = bool(result.get("safe_mode", True)),
        elapsed_seconds     = elapsed,
        session_snapshot    = result.get("session_snapshot") or {},
        branch_report       = result.get("branch_report") or {},
        commit_message      = result.get("commit_message", ""),
        changes             = result.get("changes") or {},
        conflict_report     = result.get("conflict_report") or {},
        pr_report           = result.get("pr_report") or {},
        readiness_report    = result.get("readiness_report") or {},
        secret_scan_report  = result.get("secret_scan_report") or {},
        commit_lint_report  = result.get("commit_lint_report") or {},
        test_impact_report  = result.get("test_impact_report") or {},
        cross_pr_report     = result.get("cross_pr_report") or {},
        pr_description      = result.get("pr_description") or {},
        errors              = result.get("errors") or [],
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
async def git_branch(req: GitBranchRequest, user: Principal = Depends(get_current_user)):
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
            message=f"is branch {req.branch} ready to merge into {req.base}",
            project_path=str(project_path),
            branch=req.branch,
            base=req.base,
            session_id=req.session_id,
        )
    except Exception as e:
        logger.exception("git/branch error")
        raise HTTPException(500, f"SmartGitGraph error: {e}")

    _pg_save_git_report(
        user_email  = user.email,
        report_type = "branch",
        raw_data    = result,
        local_path  = str(project_path),
        branch      = req.branch,
        base_branch = req.base,
        verdict     = (result.get("branch_report") or {}).get("verdict", ""),
        total_score = (result.get("branch_report") or {}).get("total_score"),
        summary     = result.get("response", "")[:2000],
    )

    return _to_response(result, round(time.time() - t0, 2))


@git_router.post("/commit-msg", response_model=GitResponse, summary="Generate commit message")
async def git_commit_msg(req: GitCommitMsgRequest):
    """
    Génère un message de commit Conventional Commits basé sur les diffs stagés.

    Appel DIRECT de generate_commit_message() (même fonction que le hook Git
    prepare-commit-msg) plutôt que le LangGraph complet → plus rapide, identique
    au comportement du hook. Additif : ne touche ni le graph ni le dashboard.
    """
    project_path = Path(req.project_path)
    if not project_path.exists():
        raise HTTPException(404, f"Projet introuvable : {project_path}")

    t0 = time.time()
    try:
        from smart_git.git_commit_msg import generate_commit_message

        message = await asyncio.to_thread(generate_commit_message, project_path)
        return GitResponse(
            response        = message,
            intent          = "commit_message",
            confidence      = 1.0,
            safe_mode       = True,
            elapsed_seconds = round(time.time() - t0, 2),
            commit_message  = message,
            errors          = [] if message else ["no staged changes or generation failed"],
        )
    except Exception as e:
        logger.exception("git/commit-msg error")
        raise HTTPException(500, f"commit-msg error: {e}")


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
async def git_pr_review(req: GitPRRequest, user: Principal = Depends(get_current_user)):
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

    _pg_save_git_report(
        user_email   = user.email,
        report_type  = "pr_review",
        raw_data     = result,
        github_slug  = f"{req.owner}/{req.repo}",
        project_name = req.repo,
        pr_number    = req.pr_number,
        verdict      = (result.get("pr_report") or {}).get("verdict", ""),
        summary      = result.get("response", "")[:2000],
    )

    return _to_response(result, round(time.time() - t0, 2))


@git_router.post("/pr/readiness", response_model=GitResponse, summary="PR merge readiness")
async def git_pr_readiness(req: GitPRRequest, user: Principal = Depends(get_current_user)):
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

    _ready = (result.get("readiness_report") or {}).get("ready")
    _verdict = "READY" if _ready is True else ("BLOCKED" if _ready is False else "")

    _pg_save_git_report(
        user_email   = user.email,
        report_type  = "pr_review",
        raw_data     = result,
        github_slug  = f"{req.owner}/{req.repo}",
        project_name = req.repo,
        pr_number    = req.pr_number,
        verdict      = _verdict,
        summary      = result.get("response", "")[:2000],
    )

    return _to_response(result, round(time.time() - t0, 2))


# ── Fuzzy replacement engine ─────────────────────────────────────────────────

def _fuzzy_replace(content: str, current_code: str, fixed_code: str, hint_line: int = 0) -> str:
    """
    Multi-strategy code replacement — handles real-world LLM snippet drift.

    Strategy 1 — Exact match          : fastest path, used when possible.
    Strategy 2 — Whitespace-normalized : strips trailing spaces + normalises
                 line endings; preserves original indentation in the output.
    Strategy 3 — Similarity window    : SequenceMatcher sliding window (≥85 %).
                 Search is anchored around hint_line ± 30 lines when provided
                 so a 300-line file doesn't degrade to O(n²).

    Returns the patched file content.
    Raises ValueError with a human-readable message if no match is found.
    """
    import difflib

    # ── Strategy 1 : exact ────────────────────────────────────────────────────
    if current_code in content:
        return content.replace(current_code, fixed_code, 1)

    content_lines = content.splitlines(keepends=True)
    current_lines = current_code.splitlines()
    n = len(current_lines)
    if n == 0:
        raise ValueError("current_code is vide — impossible d'appliquer le fix.")

    def _apply_indent(source_line: str, new_block: str) -> str:
        """Re-indente new_block en copiant le niveau d'indentation de source_line."""
        indent = len(source_line) - len(source_line.lstrip())
        pfx = source_line[:indent]
        out_lines = []
        for ln in new_block.splitlines(keepends=True):
            out_lines.append(pfx + ln.lstrip() if ln.strip() else ln)
        result = "".join(out_lines)
        if not result.endswith("\n"):
            result += "\n"
        return result

    def _reconstruct(i: int) -> str:
        block = _apply_indent(content_lines[i], fixed_code)
        return "".join(content_lines[:i]) + block + "".join(content_lines[i + n:])

    # ── Strategy 2 : trailing-whitespace normalised ───────────────────────────
    cur_stripped = [l.rstrip() for l in current_lines]
    for i in range(len(content_lines) - n + 1):
        window = [content_lines[i + j].rstrip("\n\r").rstrip() for j in range(n)]
        if window == cur_stripped:
            return _reconstruct(i)

    # ── Strategy 3 : fuzzy sliding window (SequenceMatcher ≥ 85 %) ───────────
    cur_tokens = [l.strip() for l in current_lines if l.strip()]
    if not cur_tokens:
        raise ValueError("current_code ne contient que des lignes vides.")

    # Restrict search window around hint_line if provided
    total = len(content_lines)
    if hint_line > 0:
        lo = max(0, hint_line - 30)
        hi = min(total - n, hint_line + 30)
    else:
        lo, hi = 0, total - n

    best_ratio, best_i = 0.0, -1
    for i in range(lo, hi + 1):
        window_tokens = [content_lines[i + j].strip() for j in range(n)]
        ratio = difflib.SequenceMatcher(None, cur_tokens, window_tokens).ratio()
        if ratio > best_ratio:
            best_ratio, best_i = ratio, i

    if best_ratio >= 0.85 and best_i >= 0:
        logger.info(
            "apply-suggestion fuzzy match: similarity=%.0f%% at line %d",
            best_ratio * 100, best_i + 1,
        )
        return _reconstruct(best_i)

    raise ValueError(
        f"current_code introuvable dans le fichier "
        f"(meilleure similarité : {best_ratio:.0%}). "
        "Le fichier a peut-être été modifié depuis la review, "
        "ou l'indentation/les fins de ligne ont changé."
    )


# ── Apply Suggestion ─────────────────────────────────────────────────────────

@git_router.post("/pr/apply-suggestion", summary="Apply a suggested fix to the PR branch")
async def git_apply_suggestion(req: ApplySuggestionRequest):
    """
    Remplace `current_code` par `fixed_code` dans `file_path` sur la branche
    `head_ref` de la PR, puis commit via l'API REST GitHub.

    Flux :
      GET /repos/{owner}/{repo}/contents/{path}?ref={head_ref}  → content (b64) + sha
      str.replace(current_code, fixed_code, 1)
      PUT /repos/{owner}/{repo}/contents/{path}  → {message, content (b64), sha, branch}

    Retourne : {success, sha, commit_url}
    """
    import base64 as _b64

    token = (
        os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
        or os.environ.get("GITHUB_TOKEN", "")
    )
    if not token:
        raise HTTPException(401, "GITHUB_PERSONAL_ACCESS_TOKEN manquant — définissez-le dans .env")

    headers = {
        "Authorization": f"token {token}",
        "Accept":        "application/vnd.github.v3+json",
        "User-Agent":    "code-auditor/1.0",
        "Content-Type":  "application/json",
    }
    base_url = f"https://api.github.com/repos/{req.owner}/{req.repo}/contents/{req.file_path}"

    # 1. Récupérer le contenu actuel + SHA (nécessaire pour le PUT)
    try:
        get_url = f"{base_url}?ref={urllib.parse.quote(req.head_ref, safe='')}"
        get_req = urllib.request.Request(get_url, headers=headers)
        with urllib.request.urlopen(get_req, timeout=15) as resp:
            file_data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:300]
        raise HTTPException(exc.code, f"GitHub API (get file): {body}")
    except Exception as exc:
        raise HTTPException(500, f"GitHub REST error (get file): {exc}")

    file_sha = file_data.get("sha", "")
    raw_content = file_data.get("content", "")
    encoding   = file_data.get("encoding", "base64")

    # Décoder le contenu
    try:
        if encoding == "base64":
            original = _b64.b64decode(raw_content.replace("\n", "")).decode("utf-8", errors="replace")
        else:
            original = raw_content
    except Exception as exc:
        raise HTTPException(500, f"Impossible de décoder le fichier: {exc}")

    # 2. Appliquer le remplacement (fuzzy multi-strategy)
    try:
        updated = _fuzzy_replace(original, req.current_code, req.fixed_code, req.hint_line)
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    # 3. Encoder + committer via PUT
    updated_b64 = _b64.b64encode(updated.encode("utf-8")).decode("ascii")
    commit_message = (
        f"fix: apply Code Auditor suggestion in {req.file_path}\n\n"
        f"PR #{req.pr_number} — automated fix via Code Auditor"
    )
    put_payload = json.dumps({
        "message": commit_message,
        "content": updated_b64,
        "sha":     file_sha,
        "branch":  req.head_ref,
    }).encode("utf-8")

    try:
        put_req = urllib.request.Request(base_url, data=put_payload, headers=headers, method="PUT")
        with urllib.request.urlopen(put_req, timeout=20) as resp:
            result = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:400]
        raise HTTPException(exc.code, f"GitHub API (put file): {body}")
    except Exception as exc:
        raise HTTPException(500, f"GitHub REST error (put file): {exc}")

    commit = result.get("commit", {})
    return {
        "success":    True,
        "sha":        commit.get("sha", ""),
        "commit_url": commit.get("html_url", ""),
        "file_path":  req.file_path,
        "branch":     req.head_ref,
    }


# ── F1: Secret Scan ───────────────────────────────────────────────────────────

@git_router.post("/secret-scan", response_model=GitResponse, summary="Scan staged files for secrets")
async def git_secret_scan(req: GitSecretScanRequest, user: Principal = Depends(get_current_user)):
    """
    Scanne les fichiers stagés pour détecter les secrets/credentials.
    Bloquant si un secret est trouvé.
    SmartGitGraph → node_secret_scan.
    """
    project_path = Path(req.project_path)
    if not project_path.exists():
        raise HTTPException(404, f"Projet introuvable : {project_path}")

    t0 = time.time()
    try:
        result = await _invoke(
            message="scan secrets staged files",
            project_path=str(project_path),
            session_id=req.session_id,
        )
    except Exception as e:
        logger.exception("git/secret-scan error")
        raise HTTPException(500, f"SmartGitGraph error: {e}")

    _found = (result.get("secret_scan_report") or {}).get("found", False)
    _pg_save_git_report(
        user_email  = user.email,
        report_type = "secret_scan",
        raw_data    = result,
        local_path  = str(project_path),
        verdict     = "BLOCKED" if _found else "CLEAN",
        summary     = result.get("response", "")[:2000],
    )

    return _to_response(result, round(time.time() - t0, 2))


# ── F3: Commit Lint ───────────────────────────────────────────────────────────

def _commit_lint_to_dict(report) -> Dict[str, Any]:
    """Sérialise un CommitLintReport en dict aligné avec le client extension."""
    return {
        "is_valid":          report.is_valid,
        "score":             report.score,
        "original_message":  report.original_message,
        "suggested_message": report.suggested_message,
        "violations": [
            {
                "rule":       v.rule,
                "severity":   v.severity.lower(),   # error|warn → l'UI attend lowercase
                "message":    v.message,
                "suggestion": v.suggestion,
            }
            for v in report.violations
        ],
        "success": report.success,
        "error":   report.error,
    }


@git_router.post("/commit-lint", response_model=GitResponse, summary="Validate commit message")
async def git_commit_lint(req: GitCommitLintRequest, user: Principal = Depends(get_current_user)):
    """
    Valide un message de commit selon la spécification Conventional Commits.

    Appel DIRECT du linter (100 % local, 0 LLM) — pas le LangGraph.
    """
    project_path = Path(req.project_path)
    if not project_path.exists():
        raise HTTPException(404, f"Projet introuvable : {project_path}")

    t0 = time.time()
    try:
        from smart_git.git_commit_linter import lint_commit_message, lint_staged_commit

        def _run() -> Dict[str, Any]:
            if req.commit_message.strip():
                rep = lint_commit_message(req.commit_message)
            else:
                rep = lint_staged_commit(project_path)
            return _commit_lint_to_dict(rep)

        lint = await asyncio.to_thread(_run)
    except Exception as e:
        logger.exception("git/commit-lint error")
        raise HTTPException(500, f"commit-lint error: {e}")

    _pg_save_git_report(
        user_email  = user.email,
        report_type = "commit_lint",
        raw_data    = {"commit_lint_report": lint},
        local_path  = str(project_path),
        verdict     = "VALID" if lint.get("is_valid") else "INVALID",
        total_score = lint.get("score"),
        summary     = f"score {lint.get('score')}/100 · {len(lint.get('violations', []))} issue(s)",
    )

    return GitResponse(
        response           = f"score {lint.get('score')}/100",
        intent             = "commit_lint",
        confidence         = 1.0,
        safe_mode          = True,
        elapsed_seconds    = round(time.time() - t0, 2),
        commit_lint_report = lint,
        errors             = [lint["error"]] if lint.get("error") else [],
    )


# ── F4: Test Impact ───────────────────────────────────────────────────────────

def _test_impact_to_dict(report) -> Dict[str, Any]:
    """Sérialise un TestImpactReport en dict aligné avec le client extension."""
    return {
        "impacts": [
            {
                "source_file":      i.source_file,
                "test_files":       i.test_files,
                "missing_tests":    i.missing_tests,
                "discovery_method": i.discovery_method,
            }
            for i in report.impacts
        ],
        "total_files":     report.total_files,
        "covered_files":   report.covered_files,
        "uncovered_files": report.uncovered_files,
        "all_test_files":  report.all_test_files,
        "coverage_ratio":  report.coverage_ratio,
        "has_gaps":        report.has_gaps,
        "success":         report.success,
        "error":           report.error,
    }


@git_router.post("/test-impact", response_model=GitResponse, summary="Test impact analysis")
async def git_test_impact(req: GitTestImpactRequest, user: Principal = Depends(get_current_user)):
    """
    Trouve les fichiers de test impactés par les modifications (staged ou liste explicite).

    Appel DIRECT de l'analyseur (100 % local, 0 LLM) — pas le LangGraph.
    """
    project_path = Path(req.project_path)
    if not project_path.exists():
        raise HTTPException(404, f"Projet introuvable : {project_path}")

    t0 = time.time()
    try:
        from smart_git.git_test_impact import (
            analyze_test_impact_staged,
            analyze_test_impact_files,
        )

        def _run() -> Dict[str, Any]:
            if req.changed_files:
                rep = analyze_test_impact_files(req.changed_files, project_path)
            else:
                rep = analyze_test_impact_staged(project_path)
            return _test_impact_to_dict(rep)

        impact = await asyncio.to_thread(_run)
    except Exception as e:
        logger.exception("git/test-impact error")
        raise HTTPException(500, f"test-impact error: {e}")

    _pct = round(impact.get("coverage_ratio", 1.0) * 100)
    _pg_save_git_report(
        user_email  = user.email,
        report_type = "test_impact",
        raw_data    = {"test_impact_report": impact},
        local_path  = str(project_path),
        verdict     = "GAPS" if impact.get("has_gaps") else "COVERED",
        total_score = _pct,
        summary     = f"{_pct}% covered · {impact.get('uncovered_files', 0)} gap(s)",
    )

    return GitResponse(
        response           = f"{_pct}% covered",
        intent             = "test_impact",
        confidence         = 1.0,
        safe_mode          = True,
        elapsed_seconds    = round(time.time() - t0, 2),
        test_impact_report = impact,
        errors             = [impact["error"]] if impact.get("error") else [],
    )


# ── F5: Commit Readiness (aggregate) ──────────────────────────────────────────

@git_router.post("/commit-readiness", response_model=GitResponse, summary="Aggregate commit readiness")
async def git_commit_readiness(req: GitCommitReadinessRequest, user: Principal = Depends(get_current_user)):
    """
    Agrège les cinq checks locaux (secrets, conflicts, lint, test impact, session)
    en un verdict unique : score 0-100 + verdict READY/WARN/BLOCKED + blockers.

    Appel DIRECT des modules smart_git (pas le LangGraph) → rapide, déterministe,
    0 token LLM. Le résultat est exposé dans `readiness_report`.
    """
    project_path = Path(req.project_path)
    if not project_path.exists():
        raise HTTPException(404, f"Projet introuvable : {project_path}")

    t0 = time.time()
    try:
        # Appel synchrone (git CLI + regex) déporté hors de l'event loop.
        from smart_git.git_commit_readiness import evaluate_commit_readiness

        def _run() -> Dict[str, Any]:
            return evaluate_commit_readiness(
                project_path,
                commit_message=req.commit_message,
            ).to_dict()

        readiness = await asyncio.to_thread(_run)
    except Exception as e:
        logger.exception("git/commit-readiness error")
        raise HTTPException(500, f"commit-readiness error: {e}")

    verdict = str(readiness.get("verdict", ""))
    score   = readiness.get("score")
    summary = (
        f"{verdict} · score {score}/100 · "
        f"{len(readiness.get('blockers', []))} blocker(s)"
    )

    _pg_save_git_report(
        user_email  = user.email,
        report_type = "commit_readiness",
        raw_data    = {"readiness_report": readiness},
        local_path  = str(project_path),
        branch      = readiness.get("branch", ""),
        verdict     = verdict,
        total_score = score if isinstance(score, int) else None,
        summary     = summary,
    )

    return GitResponse(
        response         = summary,
        intent           = "commit_readiness",
        confidence       = 1.0,
        safe_mode        = True,
        elapsed_seconds  = round(time.time() - t0, 2),
        readiness_report = readiness,
        errors           = [readiness["error"]] if readiness.get("error") else [],
    )


# ── F6: Cross-PR Conflicts ────────────────────────────────────────────────────

@git_router.post("/pr/cross-conflicts", response_model=GitResponse, summary="Cross-PR conflict detection")
async def git_cross_pr_conflicts(req: GitCrossPRRequest, user: Principal = Depends(get_current_user)):
    """
    Détecte les fichiers modifiés dans plusieurs PRs ouvertes simultanément.
    SmartGitGraph → node_cross_pr.
    """
    t0 = time.time()
    try:
        result = await _invoke(
            message=f"cross pr analysis {req.owner}/{req.repo}",
            owner=req.owner,
            repo=req.repo,
            pr_number=req.pr_number,
            base=req.base,
            session_id=req.session_id,
        )
    except Exception as e:
        logger.exception("git/pr/cross-conflicts error")
        raise HTTPException(500, f"SmartGitGraph error: {e}")

    _pg_save_git_report(
        user_email   = user.email,
        report_type  = "pr_review",
        raw_data     = result,
        github_slug  = f"{req.owner}/{req.repo}",
        project_name = req.repo,
        pr_number    = req.pr_number,
        base_branch  = req.base,
        verdict      = (result.get("cross_pr_report") or {}).get("verdict", ""),
        summary      = result.get("response", "")[:2000],
    )

    return _to_response(result, round(time.time() - t0, 2))


# ── F7: PR Auto-Description ───────────────────────────────────────────────────

@git_router.post("/pr/description", response_model=GitResponse, summary="Generate PR description")
async def git_pr_description(req: GitPRDescriptionRequest, user: Principal = Depends(get_current_user)):
    """
    Génère une description structurée de Pull Request (LLM + cache Redis).
    Mode local (git log) si project_path fourni, sinon GitHub API.
    SmartGitGraph → node_pr_description.
    """
    t0 = time.time()
    try:
        result = await _invoke(
            message=f"generate pr description for branch {req.branch}",
            project_path=req.project_path or ".",
            owner=req.owner,
            repo=req.repo,
            pr_number=req.pr_number,
            branch=req.branch,
            base=req.base,
            session_id=req.session_id,
        )
    except Exception as e:
        logger.exception("git/pr/description error")
        raise HTTPException(500, f"SmartGitGraph error: {e}")

    _pg_save_git_report(
        user_email   = user.email,
        report_type  = "pr_description",
        raw_data     = result,
        local_path   = req.project_path,
        github_slug  = f"{req.owner}/{req.repo}" if not req.project_path else "",
        project_name = req.repo or (Path(req.project_path).name if req.project_path else ""),
        pr_number    = req.pr_number,
        branch       = req.branch,
        base_branch  = req.base,
        summary      = result.get("response", "")[:2000],
    )

    return _to_response(result, round(time.time() - t0, 2))


# ── PR List — dashboard cloud surface (no LLM, direct REST) ──────────────────

@git_router.get("/prs", summary="List open pull requests")
async def git_list_prs(
    owner:    str = Query(..., description="GitHub owner"),
    repo:     str = Query(..., description="GitHub repo"),
    base:     str = Query("",  description="Filter by base branch"),
    per_page: int = Query(30,  description="Results per page (max 100)"),
):
    """
    Retourne les PRs ouvertes via l'API REST GitHub (sans LLM ni MCP).
    Consommé par le dashboard PR Cockpit.
    """
    token = (
        os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
        or os.environ.get("GITHUB_TOKEN", "")
    )
    if not token:
        raise HTTPException(401, "GITHUB_PERSONAL_ACCESS_TOKEN manquant — définissez-le dans .env")

    per_page = max(1, min(per_page, 100))
    url = (
        f"https://api.github.com/repos/{owner}/{repo}/pulls"
        f"?state=open&per_page={per_page}&sort=updated&direction=desc"
    )
    if base:
        url += f"&base={urllib.parse.quote(base)}"

    headers = {
        "Authorization": f"token {token}",
        "Accept":        "application/vnd.github.v3+json",
        "User-Agent":    "code-auditor/1.0",
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            prs: list = json.loads(resp.read().decode())

        return [
            {
                "number":          pr.get("number"),
                "title":           pr.get("title", ""),
                "state":           pr.get("state", "open"),
                "draft":           pr.get("draft", False),
                "head":            pr.get("head", {}).get("ref", ""),
                "base":            pr.get("base", {}).get("ref", "main"),
                "author":          pr.get("user", {}).get("login", ""),
                "avatar_url":      pr.get("user", {}).get("avatar_url", ""),
                "created_at":      pr.get("created_at", ""),
                "updated_at":      pr.get("updated_at", ""),
                "labels":          [lbl.get("name", "") for lbl in pr.get("labels", [])],
                "html_url":        pr.get("html_url", ""),
                "mergeable":       pr.get("mergeable"),
                "mergeable_state": pr.get("mergeable_state", ""),
                "comments":        pr.get("comments", 0),
                "review_comments": pr.get("review_comments", 0),
                "commits":         pr.get("commits", 0),
                "changed_files":   pr.get("changed_files", 0),
            }
            for pr in prs
            if isinstance(pr, dict)
        ]

    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:300] if hasattr(exc, "read") else str(exc)
        raise HTTPException(exc.code, f"GitHub API: {body}")
    except Exception as exc:
        logger.exception("git/prs error")
        raise HTTPException(500, f"GitHub REST error: {exc}")


# ── PR Detail — authoritative single-PR representation ───────────────────────

@git_router.get("/pr/detail", summary="Full single-PR representation (mergeable, commits, files)")
async def git_pr_detail(
    owner:  str = Query(..., description="GitHub owner"),
    repo:   str = Query(..., description="GitHub repo"),
    number: int = Query(..., description="Pull request number"),
):
    """
    Récupère la représentation COMPLÈTE d'une PR via GET /pulls/{number}.

    L'endpoint liste (/git/prs) renvoie une représentation *minimale* de GitHub :
    `mergeable`, `mergeable_state`, `commits`, `changed_files`, `additions`,
    `deletions`, `review_comments` y sont absents (null/0). Ils n'existent que
    sur le GET d'une PR unique. Cet endpoint est la source de vérité consommée
    par tous les onglets (header, Readiness, Review, Resolve).

    `mergeable` est calculé en asynchrone côté GitHub : s'il revient `null`, on
    poll quelques fois (backoff) jusqu'à obtenir une valeur stable.
    """
    token = (
        os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
        or os.environ.get("GITHUB_TOKEN", "")
    )
    if not token:
        raise HTTPException(401, "GITHUB_PERSONAL_ACCESS_TOKEN manquant — définissez-le dans .env")

    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}"
    headers = {
        "Authorization": f"token {token}",
        "Accept":        "application/vnd.github.v3+json",
        "User-Agent":    "code-auditor/1.0",
    }

    pr: Dict[str, Any] = {}
    # GitHub calcule `mergeable` en asynchrone → poll avec backoff (1s,2s,2s).
    delays = [0, 1, 2, 2]
    for attempt, delay in enumerate(delays):
        if delay:
            await asyncio.sleep(delay)
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=15) as resp:
                pr = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:300] if hasattr(exc, "read") else str(exc)
            raise HTTPException(exc.code, f"GitHub API: {body}")
        except Exception as exc:
            logger.exception("git/pr/detail error")
            raise HTTPException(500, f"GitHub REST error: {exc}")

        # mergeable connu → on s'arrête. Sinon on retente (sauf au dernier essai).
        if pr.get("mergeable") is not None or attempt == len(delays) - 1:
            break

    return {
        "number":          pr.get("number", number),
        "title":           pr.get("title", ""),
        "state":           pr.get("state", "open"),
        "draft":           pr.get("draft", False),
        "head":            pr.get("head", {}).get("ref", ""),
        "base":            pr.get("base", {}).get("ref", "main"),
        "author":          pr.get("user", {}).get("login", ""),
        "avatar_url":      pr.get("user", {}).get("avatar_url", ""),
        "created_at":      pr.get("created_at", ""),
        "updated_at":      pr.get("updated_at", ""),
        "labels":          [lbl.get("name", "") for lbl in pr.get("labels", [])],
        "html_url":        pr.get("html_url", ""),
        "mergeable":       pr.get("mergeable"),
        "mergeable_state": pr.get("mergeable_state", ""),
        "comments":        pr.get("comments", 0),
        "review_comments": pr.get("review_comments", 0),
        "commits":         pr.get("commits", 0),
        "changed_files":   pr.get("changed_files", 0),
        "additions":       pr.get("additions", 0),
        "deletions":       pr.get("deletions", 0),
    }


# ── F8: PR Auto Conflict Resolution (SSE) ────────────────────────────────────

@git_router.post(
    "/pr/resolve-conflicts",
    summary="Auto-resolve PR merge conflicts — 3-way diff + RAG + LLM (SSE stream)",
)
async def git_pr_resolve_conflicts(req: ResolveConflictsRequest):
    """
    Résout les conflits d'une PR via pipeline 3-way diff → conservative → LLM + RAG.

    Pipeline :
      1. Détection des conflits (MCP GitHub REST)
      2. Charge le contexte RAG (Redis + ChromaDB, 0 token)
      3. Crée branche auto-resolve/pr-{N}
      4. Résout chaque fichier (3 niveaux, sans LLM si possible)
      5. Pousse les fichiers résolus + RESOLVE_README.md
      6. Ouvre une nouvelle PR auto-resolve → base

    Retourne un text/event-stream avec les événements de progrès :
      detected        → conflits détectés, liste des fichiers
      no_conflicts    → PR déjà mergeable
      resolving_file  → résolution d'un fichier en cours
      file_resolved   → fichier résolu (method, details, lines)
      file_failed     → fichier en échec (reason)
      done            → résolution terminée (resolved, failed, branch, pr_url)
      error           → erreur fatale (message)
    """
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _emit(event: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    async def _run() -> None:
        try:
            from smart_git.conflict_resolution_agent import resolve_pr_conflicts
            await asyncio.to_thread(
                lambda: asyncio.run(
                    resolve_pr_conflicts(req.owner, req.repo, req.pr_number, emit=_emit)
                )
            )
        except Exception as exc:
            logger.exception("git/pr/resolve-conflicts error")
            _emit({"event": "error", "message": str(exc)})
        finally:
            queue.put_nowait(None)  # sentinel — stream ends

    async def _stream():
        task = asyncio.create_task(_run())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        finally:
            task.cancel()

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


# ── F8b: Resolve Diff — compare auto-resolve branch vs base ──────────────────

@git_router.get("/pr/resolve-diff", summary="Diff between auto-resolve branch and base")
async def git_pr_resolve_diff(
    owner:  str = Query(..., description="GitHub owner"),
    repo:   str = Query(..., description="GitHub repo"),
    branch: str = Query(..., description="Resolution branch, e.g. auto-resolve/pr-17"),
    base:   str = Query("main", description="Base branch to compare against"),
):
    """
    Appelle l'API GitHub /compare/{base}...{head} et retourne les fichiers
    modifiés avec leurs patches (unified diff).

    Retourne : [{ filename, status, additions, deletions, patch }]
    """
    token = (
        os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
        or os.environ.get("GITHUB_TOKEN", "")
    )
    if not token:
        raise HTTPException(401, "GITHUB_PERSONAL_ACCESS_TOKEN manquant")

    url = (
        f"https://api.github.com/repos/{owner}/{repo}/compare/"
        f"{urllib.parse.quote(base, safe='')}...{urllib.parse.quote(branch, safe='')}"
    )
    headers = {
        "Authorization": f"token {token}",
        "Accept":        "application/vnd.github.v3+json",
        "User-Agent":    "code-auditor/1.0",
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:300]
        raise HTTPException(exc.code, f"GitHub API: {body}")
    except Exception as exc:
        logger.exception("git/pr/resolve-diff error")
        raise HTTPException(500, f"GitHub REST error: {exc}")

    return [
        {
            "filename":  f.get("filename", ""),
            "status":    f.get("status", "modified"),
            "additions": f.get("additions", 0),
            "deletions": f.get("deletions", 0),
            "patch":     f.get("patch", ""),
        }
        for f in data.get("files", [])
        if isinstance(f, dict)
    ]


# ── Git Hook Management ───────────────────────────────────────────────────────

@git_router.get("/hook/status", summary="Check if Code Auditor git hook is installed")
async def git_hook_status(project_path: str = Query(..., description="Absolute path to git project")):
    """
    Vérifie si le hook pre-commit Code Auditor est installé dans le projet.
    Retourne {"installed": bool} — utilisé par le tab Git pour afficher le statut.
    """
    hook_file = Path(project_path) / ".git" / "hooks" / "pre-commit"
    msg_hook  = Path(project_path) / ".git" / "hooks" / "prepare-commit-msg"
    try:
        installed = (
            hook_file.exists() and
            "Code Auditor" in hook_file.read_text(encoding="utf-8", errors="replace")
        )
    except Exception:
        installed = False
    return {
        "installed":               installed,
        "has_prepare_commit_msg":  msg_hook.exists(),
        "hook_path":               str(hook_file),
    }


@git_router.post("/hook/install", summary="Install Code Auditor pre-commit hook")
async def git_hook_install(req: GitHookRequest):
    """
    Installe le hook pre-commit Code Auditor (strict=True) dans le projet.
    Crée .git/hooks/pre-commit + .git/hooks/prepare-commit-msg.
    """
    project_path = Path(req.project_path)
    if not project_path.exists():
        raise HTTPException(404, f"Projet introuvable : {project_path}")
    if not (project_path / ".git").exists():
        raise HTTPException(400, f"Pas un dépôt Git : {project_path}")

    try:
        from smart_git.git_hook import install_hook
        await asyncio.to_thread(install_hook, project_path, True)
        return {
            "success": True,
            "message": "Git hooks installed (strict mode ON)",
            "project": str(project_path),
        }
    except Exception as exc:
        logger.exception("git/hook/install error")
        raise HTTPException(500, f"Hook installation failed: {exc}")


@git_router.post("/hook/uninstall", summary="Uninstall Code Auditor pre-commit hook")
async def git_hook_uninstall(req: GitHookRequest):
    """
    Désinstalle le hook pre-commit Code Auditor du projet.
    Supprime .git/hooks/pre-commit et .git/hooks/prepare-commit-msg si installés par Code Auditor.
    """
    project_path = Path(req.project_path)
    if not project_path.exists():
        raise HTTPException(404, f"Projet introuvable : {project_path}")

    try:
        from smart_git.git_hook import uninstall_hook
        await asyncio.to_thread(uninstall_hook, project_path)
        return {
            "success": True,
            "message": "Git hooks uninstalled",
            "project": str(project_path),
        }
    except Exception as exc:
        logger.exception("git/hook/uninstall error")
        raise HTTPException(500, f"Hook uninstallation failed: {exc}")
