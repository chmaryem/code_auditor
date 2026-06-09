"""
auth/store.py — Redis-backed storage for the auth module.

Uses a DEDICATED direct redis-py connection (not the shared MCP Redis transport),
so auth requests never contend with the project indexer / cache and stay fast.
It talks to the SAME Redis server; every key lives under the `ca:auth:` namespace.
"""
from __future__ import annotations

import json
import secrets
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import redis

from auth.config import settings

# ── Key namespace ─────────────────────────────────────────────────────────────
_NS = "ca:auth:"

_client: Optional[redis.Redis] = None


def _r() -> redis.Redis:
    """Lazy singleton redis-py client (thread-safe connection pool)."""
    global _client
    if _client is None:
        _client = redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
    return _client


def _norm_email(email: str) -> str:
    return email.strip().lower()


def _user_key(user_id: str) -> str:
    return f"{_NS}user:{user_id}"


def _email_key(email: str) -> str:
    return f"{_NS}email:{_norm_email(email)}"


def _otp_key(email: str) -> str:
    return f"{_NS}otp:{_norm_email(email)}"


def _cooldown_key(email: str) -> str:
    return f"{_NS}otp:cooldown:{_norm_email(email)}"


def _rate_key(email: str) -> str:
    return f"{_NS}otp:rate:{_norm_email(email)}"


def _pairing_key(token: str) -> str:
    return f"{_NS}pairing:{token}"


def _refresh_key(jti: str) -> str:
    return f"{_NS}refresh:{jti}"


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _get_json(key: str) -> Optional[Any]:
    raw = _r().get(key)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


# ── Users ─────────────────────────────────────────────────────────────────────

def get_user(user_id: str) -> Optional[dict]:
    if not user_id:
        return None
    return _get_json(_user_key(user_id))


def get_user_by_email(email: str) -> Optional[dict]:
    uid = _r().get(_email_key(email))
    return get_user(uid) if uid else None


def upsert_user(email: str, name: str | None = None, role: str = "Developer") -> dict:
    """Create the user on first login, otherwise refresh last_login_at."""
    email = _norm_email(email)
    now = datetime.now(timezone.utc).isoformat()
    existing = get_user_by_email(email)

    if existing:
        existing["last_login_at"] = now
        if name:
            existing["name"] = name
        _r().set(_user_key(existing["id"]), json.dumps(existing))
        return existing

    user = {
        "id": uuid.uuid4().hex,
        "email": email,
        "name": name or email.split("@", 1)[0],
        "role": role,
        "is_active": True,
        "created_at": now,
        "last_login_at": now,
    }
    _r().set(_user_key(user["id"]), json.dumps(user))
    _r().set(_email_key(email), user["id"])  # unique-email index
    return user


# ── OTP ───────────────────────────────────────────────────────────────────────

def set_otp(email: str, code_hash: str) -> None:
    payload = json.dumps({"code_hash": code_hash, "attempts": 0, "created_at": time.time()})
    _r().set(_otp_key(email), payload, ex=settings.otp_ttl_sec)
    _r().set(_cooldown_key(email), "1", ex=settings.otp_resend_cooldown_sec)


def get_otp(email: str) -> Optional[dict]:
    return _get_json(_otp_key(email))


def update_otp_attempts(email: str, otp: dict, attempts: int) -> None:
    """Persist the new attempt count while preserving the original TTL window."""
    otp["attempts"] = attempts
    _r().set(_otp_key(email), json.dumps(otp), keepttl=True)


def clear_otp(email: str) -> None:
    _r().delete(_otp_key(email))


def in_cooldown(email: str) -> bool:
    return _r().exists(_cooldown_key(email)) == 1


def hit_rate_limit(email: str) -> bool:
    """Atomic fixed 1-hour window counter. Returns True if over the limit."""
    key = _rate_key(email)
    count = _r().incr(key)
    if count == 1:
        _r().expire(key, 3600)
    return count > settings.otp_rate_per_hour


# ── Pairing tokens (VS Code handoff) ──────────────────────────────────────────

def create_pairing(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    _r().set(_pairing_key(token), user_id, ex=settings.pairing_ttl_sec)
    return token


def pop_pairing(token: str) -> Optional[str]:
    """Single-use redeem: atomic GETDEL (fallback to get+delete on old Redis)."""
    key = _pairing_key(token)
    try:
        return _r().getdel(key)
    except redis.ResponseError:
        uid = _r().get(key)
        if uid:
            _r().delete(key)
        return uid


# ── Refresh-token allow-list (revocable sessions) ─────────────────────────────

def store_refresh(jti: str, user_id: str) -> None:
    _r().set(_refresh_key(jti), user_id, ex=settings.refresh_ttl_sec)


def is_refresh_valid(jti: str) -> bool:
    return _r().exists(_refresh_key(jti)) == 1


def revoke_refresh(jti: str) -> None:
    _r().delete(_refresh_key(jti))
