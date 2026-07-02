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
        if user.id != user_id:
            logger.debug(
                "auth pg_sync: Redis id %s ≠ PG id %s for %s — PG id is authoritative",
                user_id, user.id, email,
            )
    except Exception as exc:
     
        logger.warning("auth pg_sync: failed to mirror user %s to PG: %s", email, exc)


async def sync_login_timestamp(db: AsyncSession, email: str) -> None:
    try:
        repo = UserRepo(db)
        await repo.upsert(email=email) 
    except Exception as exc:
        logger.debug("auth pg_sync: login timestamp update failed for %s: %s", email, exc)
