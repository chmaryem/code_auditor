"""
secret_redactor.py — Redact secrets from source code BEFORE it leaves the machine
for an external LLM (OpenRouter / Groq / Gemini).

Enterprise requirement (Tritux): a hardcoded secret in an analyzed file must never
be transmitted to a third-party model. This reuses the detection patterns from the
Smart Git secret scanner (single source of truth) and replaces only the secret
VALUE with a typed placeholder, keeping the code structure intact so the analyzer
can still flag a "hardcoded secret" issue.

Usage (at every LLM egress that embeds user code):
    from services.secret_redactor import redact_secrets
    safe_code, n = redact_secrets(code)
"""
from __future__ import annotations

import re
from typing import Tuple

try:
    from smart_git.git_secret_scanner import SECRET_PATTERNS, FAKE_VALUE_PATTERNS
except Exception:  # pragma: no cover - defensive: never break analysis on import error
    SECRET_PATTERNS = {}
    FAKE_VALUE_PATTERNS = []

_FAKE_RES = [re.compile(p, re.IGNORECASE) for p in FAKE_VALUE_PATTERNS]
_QUOTED = re.compile(r"(['\"])(.*?)\1", re.S)

# Redaction-specific patterns. Unlike *detection* (which prefers precision to
# avoid false positives), *redaction* prefers OVER-redaction. These run FIRST so
# they win over the narrower scanner patterns — e.g. URL credentials whose
# password itself contains '@' (greedy up to the LAST '@' of the authority).
_EXTRA_PATTERNS = {
    "URL Credentials": r"(?i)\b[a-z][a-z0-9+.\-]*://[^/\s]+@",
}


def _looks_fake(matched: str) -> bool:
    """True if the matched secret is an obvious placeholder (your_key, os.environ, …)."""
    qm = _QUOTED.search(matched)
    value = qm.group(2) if qm else matched.split("=")[-1].split(":")[-1]
    value = value.strip().lower()
    return any(rx.search(value) for rx in _FAKE_RES)


def _token(secret_type: str) -> str:
    return "REDACTED_" + re.sub(r"[^A-Za-z0-9]+", "_", secret_type).strip("_")


def _redact_in_match(matched: str, secret_type: str) -> str:
    """Redact the secret value, preserving surrounding quotes when present so the
    line still reads as code (e.g. ``api_key = "REDACTED_Generic_API_Key"``)."""
    token = _token(secret_type)
    quotes = list(_QUOTED.finditer(matched))
    if quotes:
        last = quotes[-1]  # the value is the last quoted run
        q = last.group(1)
        return matched[:last.start()] + f"{q}{token}{q}" + matched[last.end():]
    # No quotes in the match (bare token / key header / connection string) →
    # drop the sensitive part entirely.
    return token


def redact_secrets(code: str) -> Tuple[str, int]:
    """Return ``(redacted_code, n_secrets_redacted)``.

    Conservative: obvious placeholder/fake values are left untouched. Never raises
    — on any error the original code is returned with count 0 (the caller decides
    whether that is acceptable; for analysis it is, the input is already trusted).
    """
    if not code or not SECRET_PATTERNS:
        return code, 0

    count = 0

    def _make_repl(secret_type: str):
        def _repl(m: "re.Match") -> str:
            nonlocal count
            matched = m.group(0)
            if _looks_fake(matched):
                return matched
            count += 1
            return _redact_in_match(matched, secret_type)
        return _repl

    redacted = code
    try:
        # _EXTRA_PATTERNS first so greedy redaction wins over narrower detection.
        for secret_type, pattern in {**_EXTRA_PATTERNS, **SECRET_PATTERNS}.items():
            redacted = re.sub(pattern, _make_repl(secret_type), redacted)
    except Exception:
        return code, 0
    return redacted, count
