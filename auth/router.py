from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse

from auth import email_service, service
from auth.schemas import (
    PairingStatusOut,
    PairingTokenOut,
    RedeemIn,
    RefreshIn,
    RequestCodeIn,
    RequestCodeOut,
    TokenOut,
    UserOut,
    VerifyCodeIn,
    VscodeTokenIn,
    VscodeVerifyIn,
)
from auth.security import Principal, get_current_user

auth_router = APIRouter(prefix="/auth", tags=["auth"])


def _cleanup_redis_session_cache(user_id: str) -> None:
    try:
        from services.mcp_redis_service import get_mcp_redis
        redis = get_mcp_redis()
        if redis:
            redis.delete(f"ca:session:{user_id}")
    except Exception:
        pass


async def _pg_sync_user(email: str, name: str, role: str) -> None:

    try:
        from database.connection import AsyncSessionLocal
        from auth.pg_sync import sync_user_to_pg
        async with AsyncSessionLocal() as db:
            await sync_user_to_pg(db, user_id="", email=email, name=name, role=role)
            await db.commit()
    except Exception:
        pass  


def _handle(fn, *args):
    try:
        return fn(*args)
    except service.AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@auth_router.post("/request-code", response_model=RequestCodeOut)
def request_code(body: RequestCodeIn, background: BackgroundTasks):
    try:
        resp, code = service.request_code(body.email)
    except service.AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    if code:
        # Send the email AFTER responding — keeps the request fast.
        background.add_task(email_service.send_otp_email, body.email, code)
    return resp


@auth_router.post("/verify-code", response_model=TokenOut)
def verify_code(body: VerifyCodeIn, background: BackgroundTasks):
    result = _handle(service.verify_code, body.email, body.code)
    # Mirror user to PostgreSQL in the background — async task, non-blocking
    background.add_task(_pg_sync_user, body.email, result.user.name, result.user.role)
    return result


@auth_router.post("/refresh", response_model=TokenOut)
def refresh(body: RefreshIn):
    return _handle(service.refresh, body.refresh_token)


@auth_router.post("/logout")
def logout(body: RefreshIn, background: BackgroundTasks, _user: Principal = Depends(get_current_user)):
    # Revoke refresh token in Redis — permanent data in PostgreSQL is NOT touched
    service.logout(body.refresh_token)
    # Remove any hot Redis chat-cache keys for this user (optional, conservative)
    # PostgreSQL conversations/messages remain intact for future logins
    background.add_task(_cleanup_redis_session_cache, _user.id)
    return {"detail": "Signed out."}


@auth_router.get("/me", response_model=UserOut)
def me(user: Principal = Depends(get_current_user)):
    return _handle(service.get_me, user)


@auth_router.post("/pairing-token", response_model=PairingTokenOut)
def pairing_token(user: Principal = Depends(get_current_user)):
    return _handle(service.issue_pairing_token, user)


@auth_router.post("/pairing/redeem", response_model=TokenOut)
def redeem_pairing(body: RedeemIn):
    return _handle(service.redeem_pairing_token, body.pairing_token)


@auth_router.get("/pairing/{token}/status", response_model=PairingStatusOut)
def pairing_status(token: str, _user: Principal = Depends(get_current_user)):
    """Dashboard polls this after opening the vscode:// deep link to know
    whether the extension has redeemed the token (non-destructive read)."""
    return _handle(service.pairing_status, token)


# ── VS Code OAuth-like PKCE flow ──────────────────────────────────────────────

_VSCODE_LOGIN_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Code Auditor AI — Sign in</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0d1117; color: #e6edf3;
      display: flex; align-items: center; justify-content: center;
      min-height: 100vh; padding: 24px;
    }}
    .card {{
      background: #161b22; border: 1px solid #30363d; border-radius: 12px;
      padding: 40px 48px; width: 100%; max-width: 420px;
    }}
    .logo {{ text-align: center; margin-bottom: 24px; }}
    .logo svg {{ width: 48px; height: 48px; }}
    h1 {{ font-size: 20px; font-weight: 600; text-align: center; margin-bottom: 8px; }}
    p.sub {{ color: #8b949e; font-size: 14px; text-align: center; margin-bottom: 28px; }}
    label {{ display: block; font-size: 13px; font-weight: 500; color: #8b949e; margin-bottom: 6px; }}
    input {{
      width: 100%; padding: 10px 14px; background: #0d1117;
      border: 1px solid #30363d; border-radius: 6px;
      color: #e6edf3; font-size: 15px; outline: none;
      transition: border-color .15s;
    }}
    input:focus {{ border-color: #58a6ff; }}
    .field {{ margin-bottom: 16px; }}
    .otp-hint {{ font-size: 12px; color: #8b949e; margin-top: 4px; }}
    button {{
      width: 100%; padding: 11px; margin-top: 8px;
      background: #238636; color: #fff; border: none;
      border-radius: 6px; font-size: 15px; font-weight: 500;
      cursor: pointer; transition: background .15s;
    }}
    button:hover {{ background: #2ea043; }}
    .step {{ display: none; }}
    .step.active {{ display: block; }}
    .msg {{ font-size: 13px; margin-top: 14px; text-align: center; min-height: 20px; }}
    .msg.error {{ color: #f85149; }}
    .msg.info  {{ color: #58a6ff; }}
    .msg.ok    {{ color: #3fb950; }}
    .divider {{ border: none; border-top: 1px solid #30363d; margin: 20px 0; }}
    .back {{ font-size: 13px; color: #58a6ff; cursor: pointer; background: none;
             border: none; width: auto; padding: 0; margin-top: 12px; }}
  </style>
</head>
<body>
<div class="card">
  <div class="logo">
    <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="24" cy="24" r="22" fill="#1f2937" stroke="#58a6ff" stroke-width="2"/>
      <path d="M16 20c0-4.4 3.6-8 8-8s8 3.6 8 8-3.6 8-8 8-8-3.6-8-8z" fill="#58a6ff" opacity=".3"/>
      <circle cx="24" cy="20" r="4" fill="#58a6ff"/>
      <path d="M10 38c0-5.5 6.3-10 14-10s14 4.5 14 10" stroke="#58a6ff" stroke-width="2" stroke-linecap="round"/>
    </svg>
  </div>
  <h1>Code Auditor AI</h1>
  <p class="sub">Sign in to connect your VS Code extension</p>

  <!-- Step 1: Email -->
  <div id="step-email" class="step active">
    <div class="field">
      <label for="email">Work email</label>
      <input id="email" type="email" placeholder="you@company.com" autocomplete="email" autofocus>
    </div>
    <button id="btn-email">Send verification code</button>
    <p id="msg-email" class="msg"></p>
  </div>

  <!-- Step 2: OTP -->
  <div id="step-otp" class="step">
    <p style="font-size:14px;color:#8b949e;margin-bottom:16px">
      We sent a 6-digit code to <strong id="email-display"></strong>
    </p>
    <div class="field">
      <label for="otp">Verification code</label>
      <input id="otp" type="text" inputmode="numeric" pattern="[0-9]*"
             maxlength="6" placeholder="123456" autocomplete="one-time-code">
      <p class="otp-hint">Check your inbox — the code expires in 10 minutes.</p>
    </div>
    <button id="btn-otp">Sign in</button>
    <p id="msg-otp" class="msg"></p>
    <hr class="divider">
    <button class="back" id="btn-back">&#8592; Use a different email</button>
  </div>

  <!-- Step 3: Success -->
  <div id="step-done" class="step">
    <p class="msg ok" style="font-size:16px;margin-bottom:12px">&#10003; Signed in successfully</p>
    <p style="font-size:14px;color:#8b949e;text-align:center">
      Returning you to VS Code…<br>
      <span style="font-size:12px">(You can close this tab)</span>
    </p>
  </div>
</div>

<script>
  const STATE = "{state}";
  const API   = "";  // same origin

  function show(id) {{
    document.querySelectorAll(".step").forEach(el => el.classList.remove("active"));
    document.getElementById(id).classList.add("active");
  }}
  function setMsg(id, text, kind) {{
    const el = document.getElementById(id);
    el.textContent = text;
    el.className = "msg " + (kind || "");
  }}

  let currentEmail = "";

  document.getElementById("btn-email").addEventListener("click", async () => {{
    const email = document.getElementById("email").value.trim();
    if (!email) {{ setMsg("msg-email", "Please enter your email.", "error"); return; }}
    setMsg("msg-email", "Sending code…", "info");
    try {{
      const r = await fetch(API + "/api/auth/request-code", {{
        method: "POST", headers: {{"Content-Type":"application/json"}},
        body: JSON.stringify({{ email }})
      }});
      const d = await r.json();
      if (!r.ok) {{ setMsg("msg-email", d.detail || "Error sending code.", "error"); return; }}
      currentEmail = email;
      document.getElementById("email-display").textContent = email;
      // Dev mode: pre-fill code if returned
      if (d.dev_code) document.getElementById("otp").value = d.dev_code;
      show("step-otp");
    }} catch {{ setMsg("msg-email", "Network error. Is the backend running?", "error"); }}
  }});

  document.getElementById("btn-otp").addEventListener("click", async () => {{
    const code = document.getElementById("otp").value.trim();
    if (!code) {{ setMsg("msg-otp", "Please enter the code.", "error"); return; }}
    setMsg("msg-otp", "Verifying…", "info");
    try {{
      const r = await fetch(API + "/api/auth/vscode-verify", {{
        method: "POST", headers: {{"Content-Type":"application/json"}},
        body: JSON.stringify({{ email: currentEmail, code, state: STATE }})
      }});
      if (r.redirected) {{
        show("step-done");
        setTimeout(() => {{ window.location.href = r.url; }}, 600);
        return;
      }}
      const d = await r.json();
      if (!r.ok) {{ setMsg("msg-otp", d.detail || "Invalid code.", "error"); return; }}
      // Fallback: response body may contain redirect_url
      if (d.redirect_url) {{
        show("step-done");
        setTimeout(() => {{ window.location.href = d.redirect_url; }}, 600);
      }}
    }} catch {{ setMsg("msg-otp", "Network error.", "error"); }}
  }});

  document.getElementById("btn-back").addEventListener("click", () => {{
    document.getElementById("otp").value = "";
    setMsg("msg-otp", "", "");
    show("step-email");
  }});

  // Submit on Enter
  document.getElementById("email").addEventListener("keydown", e => {{
    if (e.key === "Enter") document.getElementById("btn-email").click();
  }});
  document.getElementById("otp").addEventListener("keydown", e => {{
    if (e.key === "Enter") document.getElementById("btn-otp").click();
  }});
</script>
</body>
</html>
"""


@auth_router.get("/vscode-login", response_class=HTMLResponse)
def vscode_login(
    state: str = Query(..., min_length=1, max_length=128),
    code_challenge: str = Query(..., min_length=32, max_length=128),
    code_challenge_method: str = Query(default="S256"),
):
    """
    Step 1 of the VS Code PKCE flow.
    The extension opens this URL in the browser.  We persist the PKCE state and
    serve the OTP login page.
    """
    try:
        service.initiate_vscode_login(state, code_challenge, code_challenge_method)
    except service.AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    return HTMLResponse(content=_VSCODE_LOGIN_HTML.format(state=state))


@auth_router.post("/vscode-verify")
def vscode_verify(body: VscodeVerifyIn, background: BackgroundTasks):
    """
    Step 2: the browser login form POSTs here.
    On success, redirect the browser to the vscode:// deep link carrying the
    single-use auth_code.
    """
    try:
        auth_code = service.verify_vscode_otp(body.email, body.code, body.state)
    except service.AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    redirect_uri = (
        f"vscode://code-auditor.code-auditor-ai/auth-callback"
        f"?code={auth_code}&state={body.state}"
    )
    # Mirror user to PostgreSQL in background (same as verify-code).
    background.add_task(_pg_sync_user, body.email, body.email.split("@")[0], "Developer")
    # Return JSON so the browser JS can do window.location.href = redirect_url.
    # A 302 redirect to vscode:// is blocked by fetch() CORS policy (non-HTTP scheme).
    return {"redirect_url": redirect_uri}


@auth_router.get("/vscode/pending/{state}")
def vscode_pending_poll(state: str):
    """
    Extension polls this endpoint while waiting for the browser OTP to complete.
    Fallback for dev-mode where the vscode:// deep link may open the wrong window.
    Returns {"status":"pending"} until OTP is verified, then {"status":"ready","code":"..."}.
    The code is single-use (GETDEL) — consumed here, not by the deep-link path.
    """
    from auth import store as _store
    code = _store.pop_vscode_pending(state)
    if not code:
        return {"status": "pending"}
    return {"status": "ready", "code": code}


@auth_router.post("/vscode/token", response_model=TokenOut)
def vscode_token(body: VscodeTokenIn):
    """
    Step 3: the extension exchanges the auth_code + PKCE verifier for a JWT pair.
    This endpoint is called server-to-server (extension → backend), never from the browser.
    """
    try:
        return service.exchange_vscode_token(body.code, body.pkce_verifier)
    except service.AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
