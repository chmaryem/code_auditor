"""
auth/router.py — FastAPI routes for the auth module.

Endpoints are synchronous `def` so FastAPI runs them in its threadpool — the MCP
Redis client and stdlib SMTP are blocking, and this keeps them off the event loop.
Mounted under /api by the server → final paths are /api/auth/*.
"""
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
def verify_code(body: VerifyCodeIn):
    return _handle(service.verify_code, body.email, body.code)


@auth_router.post("/refresh", response_model=TokenOut)
def refresh(body: RefreshIn):
    return _handle(service.refresh, body.refresh_token)


@auth_router.post("/logout")
def logout(body: RefreshIn, _user: Principal = Depends(get_current_user)):
    service.logout(body.refresh_token)
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
