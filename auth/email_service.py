"""
auth/email_service.py — Sends OTP emails via Gmail SMTP using the stdlib only.

Called from synchronous (threadpooled) FastAPI endpoints, so blocking SMTP I/O
here does not stall the event loop.
"""
from __future__ import annotations

import logging
import smtplib
import ssl
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr

from config import config as _cfg
settings = _cfg.auth

logger = logging.getLogger("auth")

_MAX_ATTEMPTS = 3
_RETRY_DELAY_SEC = 2


def _build_message(to_email: str, code: str) -> MIMEMultipart:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"{settings.app_name} — your sign-in code: {code}"
    msg["From"] = formataddr((settings.smtp_from_name, settings.smtp_from))
    msg["To"] = to_email

    ttl_min = max(1, settings.otp_ttl_sec // 60)
    text = (
        f"Your {settings.app_name} verification code is: {code}\n\n"
        f"It expires in {ttl_min} minutes. If you didn't request it, ignore this email."
    )
    html = f"""\
<div style="font-family:Inter,Segoe UI,Arial,sans-serif;background:#0B1020;padding:32px;color:#E5E7EB">
  <div style="max-width:480px;margin:0 auto;background:#111827;border:1px solid #2B3454;border-radius:18px;padding:28px">
    <div style="font-size:13px;letter-spacing:.18em;text-transform:uppercase;color:#7C3AED;font-weight:700">
      {settings.app_name}
    </div>
    <h1 style="margin:12px 0 4px;font-size:20px;color:#fff">Your sign-in code</h1>
    <p style="margin:0 0 20px;color:#94a3b8;font-size:14px">Enter this code to access your dashboard.</p>
    <div style="font-size:34px;font-weight:800;letter-spacing:10px;color:#38BDF8;
                background:#0B1020;border:1px solid #2B3454;border-radius:14px;padding:18px;text-align:center">
      {code}
    </div>
    <p style="margin:20px 0 0;color:#64748b;font-size:12px">
      Expires in {ttl_min} minutes. If you didn't request this, you can safely ignore it.
    </p>
  </div>
</div>"""

    msg.attach(MIMEText(text, "plain", "utf-8"))
    msg.attach(MIMEText(html, "html", "utf-8"))
    return msg


def _send_once(to_email: str, msg: MIMEMultipart) -> None:
    context = ssl.create_default_context()
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as server:
        server.ehlo()
        server.starttls(context=context)
        server.login(settings.smtp_user, settings.smtp_password)
        server.sendmail(settings.smtp_from, [to_email], msg.as_string())


def send_otp_email(to_email: str, code: str) -> bool:
    """Returns True if the email was sent, False if SMTP is not configured.

    Retries transient SMTP/network failures (Gmail occasionally drops the
    connection under load) before giving up. On final failure, lifts the
    resend cooldown so the user isn't locked out by an email that never
    arrived, and re-raises so the caller's logs still capture the failure.
    """
    if not settings.smtp_configured:
        logger.warning("SMTP not configured — skipping real email send for %s.", to_email)
        return False

    msg = _build_message(to_email, code)

    last_exc: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            _send_once(to_email, msg)
            logger.info("OTP email sent to %s (attempt %d).", to_email, attempt)
            return True
        except (smtplib.SMTPException, OSError) as exc:
            last_exc = exc
            logger.warning(
                "OTP email attempt %d/%d to %s failed: %s",
                attempt, _MAX_ATTEMPTS, to_email, exc,
            )
            if attempt < _MAX_ATTEMPTS:
                time.sleep(_RETRY_DELAY_SEC)

    logger.error("OTP email to %s failed after %d attempts — lifting resend cooldown.",
                 to_email, _MAX_ATTEMPTS)
    from auth import store
    store.clear_cooldown(to_email)
    raise last_exc
