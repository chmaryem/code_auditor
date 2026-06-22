"""
api/user_router.py — User preferences & GitHub OAuth endpoints.

Routers exported:
  user_router           — protected (registered with _auth dependency)
  github_callback_router — public   (GitHub redirects here, no JWT needed)

Endpoints:
  GET  /api/auth/github/start        Begin GitHub OAuth — returns { url }
  GET  /api/auth/github/callback     GitHub callback (public) — exchanges code, saves token
  GET  /api/user/github-status       { connected: bool }
  GET  /api/user/github-repos?q=     List user's GitHub repos via their OAuth token
  POST /api/user/repo                Save selected repo to PostgreSQL
  GET  /api/user/repo                Restore active repo from PostgreSQL
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from auth import get_current_user
from auth.security import Principal
from database.connection import get_db
from database.repositories.user_repo import UserRepo

logger = logging.getLogger(__name__)

user_router            = APIRouter(tags=["user"])
github_callback_router = APIRouter(tags=["auth"])

# ── OAuth pending states ───────────────────────────────────────────────────────
# Maps state_nonce → (email, created_at_timestamp).
# Pruned on every /start call; TTL = 10 minutes.

_oauth_pending: Dict[str, Tuple[str, float]] = {}
_OAUTH_TTL = 600


def _prune_states() -> None:
    now = time.time()
    stale = [k for k, (_, ts) in _oauth_pending.items() if now - ts > _OAUTH_TTL]
    for k in stale:
        _oauth_pending.pop(k, None)


# ── Config helpers ─────────────────────────────────────────────────────────────

def _env(key: str) -> str:
    """Read env var from environment or .env file."""
    val = os.environ.get(key, "")
    if not val:
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        try:
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f"{key}="):
                        val = line.split("=", 1)[1].strip()
                        break
        except OSError:
            pass
    return val


def _callback_url() -> str:
    return _env("GITHUB_CALLBACK_URL") or "http://localhost:8000/api/auth/github/callback"


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class GithubRepoItem(BaseModel):
    full_name:   str
    name:        str
    description: str
    private:     bool
    language:    Optional[str]
    updated_at:  str
    html_url:    str


class SetRepoRequest(BaseModel):
    repo: str


class ActiveRepoResponse(BaseModel):
    repo: Optional[str]


class GithubStatusResponse(BaseModel):
    connected: bool


# ── GitHub REST helpers ────────────────────────────────────────────────────────

def _github_get(path: str, token: str) -> object:
    url = f"https://api.github.com{path}"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_repos(token: str, q: str = "") -> List[GithubRepoItem]:
    params = {"type": "all", "sort": "updated", "per_page": "100"}
    path = "/user/repos?" + urllib.parse.urlencode(params)
    try:
        data = _github_get(path, token)
    except Exception as exc:
        logger.error("GitHub /user/repos failed: %s", exc)
        raise HTTPException(502, f"GitHub API error: {exc}") from exc

    items = [
        GithubRepoItem(
            full_name=r["full_name"],
            name=r["name"],
            description=r.get("description") or "",
            private=r.get("private", False),
            language=r.get("language"),
            updated_at=r.get("updated_at", ""),
            html_url=r.get("html_url", ""),
        )
        for r in data
        if isinstance(r, dict)
    ]
    if q:
        q_lower = q.lower()
        items = [i for i in items if q_lower in i.full_name.lower()]
    return items


# ── GitHub OAuth — public callback (no _auth dependency) ──────────────────────

@github_callback_router.get("/auth/github/callback", response_class=HTMLResponse)
async def github_oauth_callback(
    code: str = "",
    state: str = "",
    db: AsyncSession = Depends(get_db),
):
    """GitHub redirects here after the user authorizes the OAuth app."""
    _prune_states()

    entry = _oauth_pending.pop(state, None)
    if not entry:
        return _html_page(success=False, error="Invalid or expired OAuth state. Please try again.")

    email, _ = entry
    client_id     = _env("GITHUB_CLIENT_ID")
    client_secret = _env("GITHUB_CLIENT_SECRET")

    if not client_id or not client_secret:
        return _html_page(success=False, error="GitHub OAuth not configured on server.")

    # Exchange authorization code for access token
    try:
        token_req = urllib.request.Request(
            "https://github.com/login/oauth/access_token",
            data=urllib.parse.urlencode({
                "client_id":     client_id,
                "client_secret": client_secret,
                "code":          code,
                "redirect_uri":  _callback_url(),
            }).encode(),
        )
        token_req.add_header("Accept", "application/json")
        with urllib.request.urlopen(token_req, timeout=15) as resp:
            token_data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.error("GitHub token exchange failed: %s", exc)
        return _html_page(success=False, error=f"GitHub token exchange failed: {exc}")

    access_token = token_data.get("access_token", "")
    if not access_token:
        error = token_data.get("error_description", "no access_token in response")
        return _html_page(success=False, error=f"GitHub denied access: {error}")

    async with db.begin():
        await UserRepo(db).set_github_token_by_email(email, access_token)

    logger.info("GitHub OAuth token saved for %s", email)
    return _html_page(success=True)


def _html_page(success: bool, error: str = "") -> HTMLResponse:
    if success:
        content = """
        <div class="icon">&#10003;</div>
        <h1>GitHub Connected!</h1>
        <p>Your GitHub account is now linked to Code Auditor AI.</p>
        <p>Return to VS Code — this window will close automatically.</p>
        <script>setTimeout(() => window.close(), 3000);</script>
        """
    else:
        content = f"""
        <div class="icon err">&#10007;</div>
        <h1>Connection Failed</h1>
        <p>{error}</p>
        <p>Close this tab and try again from VS Code.</p>
        """

    return HTMLResponse(f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>GitHub Connection — Code Auditor AI</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
      background:#0d1117;color:#e6edf3;display:flex;align-items:center;
      justify-content:center;min-height:100vh}}
    .card{{text-align:center;padding:48px 40px;background:#161b22;
      border:1px solid #30363d;border-radius:16px;max-width:420px;width:90%}}
    .icon{{font-size:48px;margin-bottom:20px;color:#3fb950}}
    .icon.err{{color:#f85149}}
    h1{{font-size:22px;font-weight:600;margin-bottom:12px}}
    p{{color:#8b949e;line-height:1.7;margin-top:8px;font-size:14px}}
    .brand{{font-size:12px;color:#30363d;margin-top:32px}}
  </style>
</head>
<body>
  <div class="card">
    {content}
    <p class="brand">Code Auditor AI</p>
  </div>
</body>
</html>""")


# ── Protected endpoints ────────────────────────────────────────────────────────

@user_router.get("/auth/github/start")
async def github_oauth_start(
    principal: Principal = Depends(get_current_user),
):
    """Generate a GitHub OAuth authorization URL for the authenticated user."""
    client_id = _env("GITHUB_CLIENT_ID")
    if not client_id:
        raise HTTPException(503, "GITHUB_CLIENT_ID not configured on server.")

    _prune_states()
    state = secrets.token_urlsafe(24)
    _oauth_pending[state] = (principal.email, time.time())

    params = {
        "client_id":    client_id,
        "redirect_uri": _callback_url(),
        "scope":        "repo read:user",
        "state":        state,
    }
    url = "https://github.com/login/oauth/authorize?" + urllib.parse.urlencode(params)
    return {"url": url}


@user_router.get("/user/github-status", response_model=GithubStatusResponse)
async def get_github_status(
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return whether the user has a GitHub OAuth token stored."""
    async with db.begin():
        token = await UserRepo(db).get_github_token_by_email(principal.email)
    return GithubStatusResponse(connected=bool(token))


@user_router.get("/user/github-repos", response_model=List[GithubRepoItem])
async def list_github_repos(
    q: str = "",
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List repos using the user's OAuth token (requires GitHub to be connected)."""
    async with db.begin():
        token = await UserRepo(db).get_github_token_by_email(principal.email)

    if not token:
        raise HTTPException(
            403,
            "GitHub account not connected. Use GET /api/auth/github/start to authorize.",
        )

    return _fetch_repos(token, q)


@user_router.post("/user/repo")
async def set_active_repo(
    req: SetRepoRequest,
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Persist the user's selected GitHub repo to PostgreSQL."""
    repo = req.repo.strip()
    if repo and "/" not in repo:
        raise HTTPException(422, "repo must be in 'owner/repo' format.")

    async with db.begin():
        await UserRepo(db).set_active_repo_by_email(principal.email, repo)

    return {"repo": repo or None}


@user_router.get("/user/repo", response_model=ActiveRepoResponse)
async def get_active_repo(
    principal: Principal = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the last repo the user selected (or null if none)."""
    async with db.begin():
        repo = await UserRepo(db).get_active_repo_by_email(principal.email)
    return ActiveRepoResponse(repo=repo)
