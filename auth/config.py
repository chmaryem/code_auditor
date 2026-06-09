"""
auth/config.py — Configuration for the self-contained authentication module.

Reads everything from environment variables directly (no dependency on the
shared project `config.py`). Loaded once at import time.
"""
from __future__ import annotations

import logging
import os
import secrets

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("auth")


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _int(value: str | None, default: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


class AuthSettings:
    """All auth configuration, read from env. Self-contained."""

    def __init__(self) -> None:
        self.app_name: str = os.getenv("AUTH_APP_NAME", "Code Auditor AI")
        self.auth_required: bool = _bool(os.getenv("AUTH_REQUIRED"), True)

        # Auth gets its own direct redis-py connection (same server as the MCP
        # Redis, but a dedicated client so it never contends with the indexer).
        self.redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

        # ── JWT ───────────────────────────────────────────────────────────────
        self.jwt_secret: str = os.getenv("AUTH_JWT_SECRET", "")
        self.jwt_algorithm: str = os.getenv("AUTH_JWT_ALGORITHM", "HS256")
        self.access_ttl_min: int = _int(os.getenv("AUTH_ACCESS_TTL_MIN"), 60)
        self.refresh_ttl_days: int = _int(os.getenv("AUTH_REFRESH_TTL_DAYS"), 14)
        self.pairing_ttl_sec: int = _int(os.getenv("AUTH_PAIRING_TTL_SEC"), 90)
        self.jwt_ephemeral: bool = False

        if not self.jwt_secret:
            # Don't crash setup — generate an ephemeral secret so the server runs,
            # but warn loudly: tokens will NOT survive a restart until the operator
            # sets a stable AUTH_JWT_SECRET in .env.
            self.jwt_secret = secrets.token_urlsafe(48)
            self.jwt_ephemeral = True
            logger.warning(
                "AUTH_JWT_SECRET is not set — using a random ephemeral secret. "
                "All sessions will be invalidated on restart. Set AUTH_JWT_SECRET in .env."
            )

        # ── OTP ───────────────────────────────────────────────────────────────
        self.otp_length: int = _int(os.getenv("AUTH_OTP_LENGTH"), 6)
        self.otp_ttl_sec: int = _int(os.getenv("AUTH_OTP_TTL_SEC"), 600)
        self.otp_max_attempts: int = _int(os.getenv("AUTH_OTP_MAX_ATTEMPTS"), 5)
        self.otp_resend_cooldown_sec: int = _int(os.getenv("AUTH_OTP_RESEND_COOLDOWN_SEC"), 60)
        self.otp_rate_per_hour: int = _int(os.getenv("AUTH_OTP_RATE_PER_HOUR"), 5)

        # ── Work-email restriction (empty = allow any domain) ─────────────────
        raw_domains = os.getenv("AUTH_ALLOWED_EMAIL_DOMAINS", "")
        self.allowed_email_domains: list[str] = [
            d.strip().lower() for d in raw_domains.split(",") if d.strip()
        ]

        # ── SMTP (Gmail App Password) ─────────────────────────────────────────
        self.smtp_host: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port: int = _int(os.getenv("SMTP_PORT"), 587)
        self.smtp_user: str = os.getenv("SMTP_USER", "")
        # Gmail shows App Passwords as 4 space-separated groups; the spaces are cosmetic.
        self.smtp_password: str = os.getenv("SMTP_APP_PASSWORD", "").replace(" ", "")
        self.smtp_from: str = os.getenv("SMTP_FROM", "") or self.smtp_user
        self.smtp_from_name: str = os.getenv("SMTP_FROM_NAME", self.app_name)

    @property
    def access_ttl_sec(self) -> int:
        return self.access_ttl_min * 60

    @property
    def refresh_ttl_sec(self) -> int:
        return self.refresh_ttl_days * 86400

    @property
    def smtp_configured(self) -> bool:
        return bool(self.smtp_user and self.smtp_password)

    def is_email_domain_allowed(self, email: str) -> bool:
        if not self.allowed_email_domains:
            return True
        domain = email.rsplit("@", 1)[-1].lower()
        return domain in self.allowed_email_domains


settings = AuthSettings()
