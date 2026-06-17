from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from auth import email_service, service
from auth.schemas import (
    PairingTokenOut,
    RedeemIn,
    RefreshIn,
    RequestCodeIn,
    RequestCodeOut,
    TokenOut,
    UserOut,
    VerifyCodeIn,
)
from auth.security import Principal, get_current_user

auth_router = APIRouter(prefix="/auth", tags=["auth"])


def _cleanup_redis_session_cache(user_id: str) -> None:
    """
    Remove Redis hot-cache keys for a user on logout.
    Conversations and messages in PostgreSQL are NOT touched.
    This is best-effort: if Redis is down, nothing breaks.
    """
    try:
        from services.mcp_redis_service import get_mcp_redis
        redis = get_mcp_redis()
        if redis:
            # Scan and remove ca:session:{user_id} if it exists
            redis.delete(f"ca:session:{user_id}")
    except Exception:
        pass


async def _pg_sync_user(email: str, name: str, role: str) -> None:
    """Mirror a newly-authenticated user into PostgreSQL (best-effort, non-blocking)."""
    try:
        from database.connection import AsyncSessionLocal
        from auth.pg_sync import sync_user_to_pg
        async with AsyncSessionLocal() as db:
            # We don't have the Redis-generated id here; pg_sync upserts by email
            await sync_user_to_pg(db, user_id="", email=email, name=name, role=role)
            await db.commit()
    except Exception:
        pass  # never block auth on a DB write failure


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
