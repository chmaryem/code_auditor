from __future__ import annotations

import re

from pydantic import BaseModel, field_validator

_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def _validate_email(value: str) -> str:
    value = (value or "").strip().lower()
    if not _EMAIL_RE.match(value):
        raise ValueError("Invalid email address.")
    return value


class RequestCodeIn(BaseModel):
    email: str

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        return _validate_email(v)


class VerifyCodeIn(BaseModel):
    email: str
    code: str

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        return _validate_email(v)

    @field_validator("code")
    @classmethod
    def _code(cls, v: str) -> str:
        v = (v or "").strip()
        if not v.isdigit():
            raise ValueError("Code must be numeric.")
        return v


class RefreshIn(BaseModel):
    refresh_token: str


class RedeemIn(BaseModel):
    pairing_token: str


class UserOut(BaseModel):
    id: str
    email: str
    name: str
    role: str


class TokenOut(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserOut


class PairingTokenOut(BaseModel):
    pairing_token: str
    expires_in: int


class PairingStatusOut(BaseModel):
    status: str  # "pending" | "redeemed" | "expired"


class RequestCodeOut(BaseModel):
    detail: str
    resend_in: int
    dev_code: str | None = None


# ── VS Code OAuth-like PKCE flow ──────────────────────────────────────────────

class VscodeVerifyIn(BaseModel):
    """Form submission from the browser login page."""
    email: str
    code: str
    state: str

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        return _validate_email(v)

    @field_validator("code")
    @classmethod
    def _code(cls, v: str) -> str:
        v = (v or "").strip()
        if not v.isdigit():
            raise ValueError("Code must be numeric.")
        return v


class VscodeTokenIn(BaseModel):
    """Token exchange — extension posts this after receiving the vscode:// callback."""
    code: str
    state: str
    pkce_verifier: str
    device_id: str | None = None
    device_name: str | None = None
