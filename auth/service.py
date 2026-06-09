"""
auth/service.py — Orchestration layer tying together store, security and email.

Raises `AuthError(status_code, detail)` for any user-facing failure; the router
maps these to HTTP responses.
"""
from __future__ import annotations

import logging

from auth import security, store
from auth.config import settings
from auth.schemas import PairingTokenOut, RequestCodeOut, TokenOut, UserOut

logger = logging.getLogger("auth")


class AuthError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


# ── Token helpers ─────────────────────────────────────────────────────────────

def _issue_tokens(user: dict) -> TokenOut:
    access = security.create_access_token(user)
    refresh, jti = security.create_refresh_token(user)
    store.store_refresh(jti, user["id"])
    return TokenOut(
        access_token=access,
        refresh_token=refresh,
        expires_in=settings.access_ttl_sec,
        user=UserOut(id=user["id"], email=user["email"], name=user["name"], role=user["role"]),
    )


# ── Public operations ─────────────────────────────────────────────────────────

def request_code(email: str) -> tuple[RequestCodeOut, str | None]:
    """Validate + store the OTP and return (response, code_to_email).

    The router emails `code_to_email` in a background task so the HTTP request
    returns immediately (no blocking on the SMTP round-trip). In dev mode
    (SMTP unconfigured) the code is surfaced in the response instead and
    code_to_email is None.
    """
    if not settings.is_email_domain_allowed(email):
        allowed = ", ".join(settings.allowed_email_domains)
        raise AuthError(403, f"Please use your work email ({allowed}).")

    if store.in_cooldown(email):
        raise AuthError(429, "A code was just sent. Please wait a moment before requesting another.")

    if store.hit_rate_limit(email):
        raise AuthError(429, "Too many code requests. Please try again later.")

    code = security.generate_otp()
    store.set_otp(email, security.hash_otp(code))

    if settings.smtp_configured:
        return (
            RequestCodeOut(
                detail="Verification code sent to your email.",
                resend_in=settings.otp_resend_cooldown_sec,
            ),
            code,
        )

    # Dev fallback: SMTP not configured → surface the code so the flow stays testable.
    logger.warning("DEV OTP for %s: %s", email, code)
    return (
        RequestCodeOut(
            detail="SMTP not configured — using dev code (also printed to server logs).",
            resend_in=settings.otp_resend_cooldown_sec,
            dev_code=code,
        ),
        None,
    )


def verify_code(email: str, code: str) -> TokenOut:
    otp = store.get_otp(email)
    if not otp:
        raise AuthError(400, "Code expired or never requested. Please request a new one.")

    if security.hash_otp(code) != otp.get("code_hash"):
        attempts = int(otp.get("attempts", 0)) + 1
        if attempts >= settings.otp_max_attempts:
            store.clear_otp(email)
            raise AuthError(429, "Too many invalid attempts. Please request a new code.")
        store.update_otp_attempts(email, otp, attempts)
        left = settings.otp_max_attempts - attempts
        raise AuthError(400, f"Invalid code. {left} attempt(s) left.")

    store.clear_otp(email)
    user = store.upsert_user(email)
    return _issue_tokens(user)


def refresh(refresh_token: str) -> TokenOut:
    try:
        claims = security.decode_token(refresh_token, "refresh")
    except security.TokenError as exc:
        raise AuthError(401, "Session expired. Please sign in again.") from exc

    jti = claims.get("jti", "")
    if not store.is_refresh_valid(jti):
        raise AuthError(401, "Session revoked or expired. Please sign in again.")

    user = store.get_user(claims.get("sub", ""))
    if not user or not user.get("is_active"):
        raise AuthError(401, "Account not found or inactive.")

    store.revoke_refresh(jti)  # rotate: old refresh is consumed
    return _issue_tokens(user)


def logout(refresh_token: str) -> None:
    try:
        claims = security.decode_token(refresh_token, "refresh")
        store.revoke_refresh(claims.get("jti", ""))
    except security.TokenError:
        # Already invalid/expired — nothing to revoke.
        pass


def get_me(principal: security.Principal) -> UserOut:
    user = store.get_user(principal.id)
    if not user:
        raise AuthError(404, "Account not found.")
    return UserOut(id=user["id"], email=user["email"], name=user["name"], role=user["role"])


def issue_pairing_token(principal: security.Principal) -> PairingTokenOut:
    token = store.create_pairing(principal.id)
    return PairingTokenOut(pairing_token=token, expires_in=settings.pairing_ttl_sec)


def redeem_pairing_token(pairing_token: str) -> TokenOut:
    user_id = store.pop_pairing(pairing_token)
    if not user_id:
        raise AuthError(400, "Pairing token is invalid or has expired.")
    user = store.get_user(user_id)
    if not user or not user.get("is_active"):
        raise AuthError(401, "Account not found or inactive.")
    return _issue_tokens(user)
