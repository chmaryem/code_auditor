from __future__ import annotations

import base64
import hashlib
import logging
import secrets

from auth import security, store
from config import config as _cfg
settings = _cfg.auth
from auth.schemas import PairingStatusOut, PairingTokenOut, RequestCodeOut, TokenOut, UserOut

logger = logging.getLogger("auth")


class AuthError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


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


def request_code(email: str) -> tuple[RequestCodeOut, str | None]:
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


def pairing_status(pairing_token: str) -> PairingStatusOut:
    return PairingStatusOut(status=store.pairing_status(pairing_token))


def redeem_pairing_token(pairing_token: str) -> TokenOut:
    user_id = store.pop_pairing(pairing_token)
    if not user_id:
        raise AuthError(400, "Pairing token is invalid or has expired.")
    user = store.get_user(user_id)
    if not user or not user.get("is_active"):
        raise AuthError(401, "Account not found or inactive.")
    return _issue_tokens(user)


# ── VS Code OAuth-like PKCE flow ──────────────────────────────────────────────

def _verify_pkce(verifier: str, challenge: str) -> bool:
    """RFC 7636 S256: BASE64URL(SHA256(verifier)) must equal the stored challenge."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    computed = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return computed == challenge


def initiate_vscode_login(state: str, code_challenge: str, code_challenge_method: str) -> None:
    """Validate and persist the PKCE state sent by the extension."""
    if not state or len(state) > 128:
        raise AuthError(400, "Invalid state parameter.")
    if code_challenge_method != "S256":
        raise AuthError(400, "Only S256 code_challenge_method is supported.")
    if not code_challenge or len(code_challenge) < 32:
        raise AuthError(400, "Invalid code_challenge.")
    store.store_vscode_state(state, code_challenge)


def verify_vscode_otp(email: str, code: str, state: str) -> str:
    """Verify OTP, generate a single-use auth_code, return it (caller handles redirect).

    Order matters:
      1. Peek state (non-destructive) so OTP failures leave state intact for retry.
      2. Verify OTP hash directly — avoids the verify_code() helper which issues a
         full JWT pair that would be immediately discarded.
      3. Only after OTP passes, atomically consume the state (pop = GETDEL).
    """
    # Step 1: check state exists without consuming it (OTP failure must allow retry).
    if not store.get_vscode_state(state):
        raise AuthError(400, "State is invalid or has expired. Please restart the sign-in flow.")

    # Step 2: verify OTP hash directly (no JWT issuance).
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

    # Step 3: OTP valid — consume OTP and state atomically.
    store.clear_otp(email)
    state_data = store.pop_vscode_state(state)
    if not state_data:
        # Extremely rare: concurrent request consumed state between peek and pop.
        raise AuthError(400, "State already used. Please restart the sign-in flow.")

    user = store.upsert_user(email)
    auth_code = secrets.token_urlsafe(32)
    store.store_vscode_code(auth_code, user["id"], state_data["pkce_challenge"])
    # Fallback for dev-mode: extension polls this key when deep link doesn't fire.
    store.store_vscode_pending(state, auth_code)
    return auth_code


def exchange_vscode_token(code: str, pkce_verifier: str) -> TokenOut:
    """Exchange single-use auth_code + PKCE verifier for a JWT pair."""
    code_data = store.pop_vscode_code(code)
    if not code_data:
        raise AuthError(400, "Authorization code is invalid or has expired.")

    if not _verify_pkce(pkce_verifier, code_data["pkce_challenge"]):
        raise AuthError(400, "PKCE verification failed. Code verifier does not match.")

    user = store.get_user(code_data["user_id"])
    if not user or not user.get("is_active"):
        raise AuthError(401, "Account not found or inactive.")

    return _issue_tokens(user)
