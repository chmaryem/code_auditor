"""
auth/pg_sync.py — Async PostgreSQL mirror for auth entities.

The auth module uses a synchronous redis-py client (fast, no event-loop overhead).
This module provides async functions that are called from FastAPI endpoints (which
run in the async context) to mirror durable data into PostgreSQL.

Pattern:
  1. auth/store.py (sync) writes to Redis — always fast, always primary for session data
  2. auth/pg_sync.py (async) mirrors user records to PostgreSQL — durable, survives Redis flush

Callers must run inside a request context that has a DB session available.
"""
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from database.repositories.user_repo import UserRepo

logger = logging.getLogger("auth.pg_sync")


async def sync_user_to_pg(
    db: AsyncSession,
    user_id: str,
    email: str,
    name: str,
    role: str = "Developer",
) -> None:
    """
    Mirror a user from Redis into PostgreSQL.
    Safe to call multiple times (upserts by email).
    Called after every successful OTP verification.
    """
    try:
        repo = UserRepo(db)
        user = await repo.upsert(email=email, name=name, role=role)
        # Align the PG id with the Redis id if the user was just created.
        # If the user already exists in PG (different id), keep PG id — both
        # ids will work because auth tokens carry the Redis-generated id.
        # A future migration can unify these if needed.
        if user.id != user_id:
            logger.debug(
                "auth pg_sync: Redis id %s ≠ PG id %s for %s — PG id is authoritative",
                user_id, user.id, email,
            )
    except Exception as exc:
        # Never block authentication on a DB write failure
        logger.warning("auth pg_sync: failed to mirror user %s to PG: %s", email, exc)


async def sync_login_timestamp(db: AsyncSession, email: str) -> None:
    """Update last_login_at in PostgreSQL on every successful login."""
    try:
        repo = UserRepo(db)
        await repo.upsert(email=email)  # upsert already refreshes last_login_at
    except Exception as exc:
        logger.debug("auth pg_sync: login timestamp update failed for %s: %s", email, exc)
