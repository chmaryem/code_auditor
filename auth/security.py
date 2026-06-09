"""
auth/security.py — JWT issuance/decoding, OTP helpers, and the FastAPI guard.

Stateless access tokens (decode-only on every request). Refresh tokens carry a
`jti` tracked in Redis (see store) so sessions can be revoked.
"""
from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from auth.config import settings


# ── Principal (request identity) ──────────────────────────────────────────────

@dataclass
class Principal:
    id: str
    email: str
    role: str


class TokenError(Exception):
    """Raised when a JWT is invalid, expired, or of the wrong type."""


# ── OTP ───────────────────────────────────────────────────────────────────────

def generate_otp() -> str:
    return "".join(secrets.choice("0123456789") for _ in range(settings.otp_length))


def hash_otp(code: str) -> str:
    # Peppered with the JWT secret so a Redis dump alone cannot reveal codes.
    return hashlib.sha256(f"{settings.jwt_secret}:{code}".encode("utf-8")).hexdigest()


# ── JWT ───────────────────────────────────────────────────────────────────────

def _encode(user: dict, token_type: str, ttl_sec: int, jti: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user["id"],
        "email": user["email"],
        "role": user.get("role", "Developer"),
        "type": token_type,
        "jti": jti,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=ttl_sec)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_access_token(user: dict) -> str:
    return _encode(user, "access", settings.access_ttl_sec, uuid.uuid4().hex)


def create_refresh_token(user: dict) -> tuple[str, str]:
    """Returns (token, jti). The jti must be stored in the Redis allow-list."""
    jti = uuid.uuid4().hex
    return _encode(user, "refresh", settings.refresh_ttl_sec, jti), jti


def decode_token(token: str, expected_type: Optional[str] = None) -> dict:
    try:
        claims = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("Token expired.") from exc
    except jwt.PyJWTError as exc:
        raise TokenError("Invalid token.") from exc

    if expected_type and claims.get("type") != expected_type:
        raise TokenError("Wrong token type.")
    return claims


def _principal_from_claims(claims: dict) -> Principal:
    return Principal(
        id=str(claims.get("sub", "")),
        email=str(claims.get("email", "")),
        role=str(claims.get("role", "Developer")),
    )


# ── FastAPI dependency ────────────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Principal:
    """Guard for protected routes. When AUTH_REQUIRED is false, falls back to a
    local principal so local-only development keeps working."""
    if not settings.auth_required:
        if creds:
            try:
                return _principal_from_claims(decode_token(creds.credentials, "access"))
            except TokenError:
                pass
        return Principal(id="local", email="local@localhost", role="Developer")

    if creds is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        claims = decode_token(creds.credentials, "access")
    except TokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    return _principal_from_claims(claims)


def authenticate_ws(token: str) -> Optional[Principal]:
    """Validate a WebSocket handshake token (passed as ?token=...). Decode-only."""
    if not settings.auth_required:
        return Principal(id="local", email="local@localhost", role="Developer")
    if not token:
        return None
    try:
        return _principal_from_claims(decode_token(token, "access"))
    except TokenError:
        return None
