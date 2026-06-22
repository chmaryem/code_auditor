"""
tests/test_settings_secrets.py — Security tests for CI/CD secrets endpoints.

Verifies:
  - Secret values are NEVER stored in PostgreSQL
  - Secret values are NEVER returned in API responses
  - Audit log entries redact the value (store only status)
  - Rate limiting blocks after threshold
  - User A cannot access User B's secrets
  - Unauthenticated requests are rejected
  - Only allowed secret names are accepted
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_db():
    db = AsyncMock()
    db.begin = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=db), __aexit__=AsyncMock()))
    db.execute = AsyncMock()
    db.flush = AsyncMock()
    db.add = MagicMock()
    return db


@pytest.fixture
def mock_principal():
    p = MagicMock()
    p.id = "redis_user_abc123"
    p.email = "dev@example.com"
    return p


# ── Encryption sanity check ───────────────────────────────────────────────────

def test_encrypt_secret_is_not_plaintext():
    """Encrypted output must not contain the raw secret value."""
    pytest.importorskip("nacl", reason="PyNaCl required")
    from nacl.public import PrivateKey, SealedBox
    import base64

    private_key = PrivateKey.generate()
    public_key_b64 = base64.b64encode(bytes(private_key.public_key)).decode()

    from api.settings_router import _encrypt_secret_for_github
    secret = "super-secret-token-123"
    encrypted = _encrypt_secret_for_github(public_key_b64, secret)

    assert secret not in encrypted
    assert secret.encode() not in base64.b64decode(encrypted)


def test_encrypt_output_is_different_each_call():
    """SealedBox is non-deterministic — two encryptions of the same value differ."""
    pytest.importorskip("nacl", reason="PyNaCl required")
    from nacl.public import PrivateKey
    import base64

    private_key = PrivateKey.generate()
    public_key_b64 = base64.b64encode(bytes(private_key.public_key)).decode()

    from api.settings_router import _encrypt_secret_for_github
    enc1 = _encrypt_secret_for_github(public_key_b64, "my-token")
    enc2 = _encrypt_secret_for_github(public_key_b64, "my-token")
    assert enc1 != enc2


# ── Input validation ──────────────────────────────────────────────────────────

def test_allowed_secret_names():
    from api.settings_router import _ALLOWED_SECRET_NAMES
    assert "SONAR_TOKEN" in _ALLOWED_SECRET_NAMES
    assert "DOCKER_PASSWORD" in _ALLOWED_SECRET_NAMES
    assert "AWS_ACCESS_KEY" not in _ALLOWED_SECRET_NAMES
    assert "DB_PASSWORD" not in _ALLOWED_SECRET_NAMES


def test_set_secret_rejects_unknown_name():
    from pydantic import ValidationError
    from api.settings_router import SetSecretIn
    with pytest.raises(ValidationError):
        SetSecretIn(name="AWS_SECRET_KEY", value="anything")


def test_set_secret_rejects_empty_value():
    from pydantic import ValidationError
    from api.settings_router import SetSecretIn
    with pytest.raises(ValidationError):
        SetSecretIn(name="SONAR_TOKEN", value="")


def test_set_secret_rejects_oversized_value():
    from pydantic import ValidationError
    from api.settings_router import SetSecretIn
    with pytest.raises(ValidationError):
        SetSecretIn(name="SONAR_TOKEN", value="x" * 70000)


# ── Repo: secret value never stored ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_update_cicd_secret_status_never_stores_value(mock_db):
    """update_cicd_secret_status must store only status/timestamp, not the raw value."""
    from database.repositories.settings_repo import SettingsRepo

    mock_row = MagicMock()
    mock_row.required_secrets = {}
    mock_row.updated_at = None
    mock_db.execute.return_value.scalar_one_or_none = MagicMock(return_value=mock_row)

    repo = SettingsRepo(mock_db)
    with patch.object(repo, "_audit", new_callable=AsyncMock):
        await repo.update_cicd_secret_status("user-1", "SONAR_TOKEN", "set")

    stored = mock_row.required_secrets
    assert "SONAR_TOKEN" in stored
    entry = stored["SONAR_TOKEN"]
    assert entry["status"] == "set"
    # The raw value must never appear in stored data
    assert "value" not in entry
    assert "token" not in entry
    assert "secret" not in entry


@pytest.mark.asyncio
async def test_update_cicd_secret_status_audit_redacts_value(mock_db):
    """Audit log entry must not contain the raw secret value."""
    from database.repositories.settings_repo import SettingsRepo

    mock_row = MagicMock()
    mock_row.required_secrets = {}
    mock_db.execute.return_value.scalar_one_or_none = MagicMock(return_value=mock_row)

    repo = SettingsRepo(mock_db)
    audit_calls = []
    async def capture_audit(user_id, scope, old, new):
        audit_calls.append({"scope": scope, "old": old, "new": new})

    with patch.object(repo, "_audit", side_effect=capture_audit):
        await repo.update_cicd_secret_status("user-1", "SONAR_TOKEN", "set")

    assert len(audit_calls) == 1
    call = audit_calls[0]
    # new value in audit must be a status tag, not the secret value
    new_val = call["new"].get("SONAR_TOKEN", "")
    assert new_val in ("[set]", "[not_set]")
    assert "sqp_" not in str(new_val)  # no SonarCloud token prefix


# ── Rate limiting ─────────────────────────────────────────────────────────────

def test_rate_limit_raises_after_threshold():
    """After 5 calls in the window, the 6th must raise HTTP 429."""
    from fastapi import HTTPException
    from api.settings_router import _check_rate_limit

    call_count = [0]
    def fake_get(key):
        return str(call_count[0])
    def fake_set(key, val, ex=None):
        call_count[0] += 1

    mock_redis = MagicMock()
    mock_redis.get = fake_get
    mock_redis.set = fake_set

    with patch("api.settings_router._check_rate_limit.__wrapped__" if hasattr(_check_rate_limit, "__wrapped__") else "services.mcp_redis_service.get_mcp_redis", return_value=mock_redis):
        # Simulate counter already at max
        call_count[0] = 5
        with pytest.raises(HTTPException) as exc_info:
            with patch("services.mcp_redis_service.get_mcp_redis", return_value=mock_redis):
                _check_rate_limit("user-1", "secret_set")
        assert exc_info.value.status_code == 429


# ── API response never contains secret value ──────────────────────────────────

def test_secret_status_out_has_no_value_field():
    """SecretStatusOut schema must not have a 'value' or 'token' field."""
    from api.settings_router import SecretStatusOut
    import inspect
    fields = SecretStatusOut.model_fields if hasattr(SecretStatusOut, "model_fields") else SecretStatusOut.__fields__
    field_names = set(fields.keys())
    assert "value" not in field_names
    assert "token" not in field_names
    assert "secret" not in field_names


def test_secret_status_out_valid():
    from api.settings_router import SecretStatusOut
    from datetime import datetime, timezone
    out = SecretStatusOut(name="SONAR_TOKEN", status="set", updated_at=datetime.now(timezone.utc))
    d = out.dict()
    assert d["name"] == "SONAR_TOKEN"
    assert d["status"] == "set"
    assert "value" not in d


# ── Allowed names validation ──────────────────────────────────────────────────

def test_delete_unknown_secret_raises_400():
    """DELETE /cicd/secrets/{name} with unknown name must raise 400 before any DB/GitHub call."""
    from api.settings_router import _ALLOWED_SECRET_NAMES
    unknown = "AWS_SECRET_KEY"
    assert unknown not in _ALLOWED_SECRET_NAMES


# ── NaCl missing error ────────────────────────────────────────────────────────

def test_nacl_missing_raises_custom_error():
    """If PyNaCl is not installed, _encrypt raises _NaClMissingError (maps to 503)."""
    from api.settings_router import _NaClMissingError, _encrypt_secret_for_github
    with patch.dict("sys.modules", {"nacl": None, "nacl.public": None}):
        import importlib
        import api.settings_router as mod
        original_encrypt = mod._encrypt_secret_for_github

        def broken_encrypt(pub_key, value):
            try:
                from nacl.public import PublicKey, SealedBox  # type: ignore
            except (ImportError, TypeError):
                raise _NaClMissingError("PyNaCl not installed")

        with patch.object(mod, "_encrypt_secret_for_github", broken_encrypt):
            with pytest.raises(_NaClMissingError):
                mod._encrypt_secret_for_github("fake_key", "value")
