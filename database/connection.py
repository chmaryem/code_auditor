"""
database/connection.py — SQLAlchemy 2.x async engine + session factory.

Usage in FastAPI endpoints:
    from database import get_db
    from sqlalchemy.ext.asyncio import AsyncSession

    @router.get("/...")
    async def handler(db: AsyncSession = Depends(get_db)):
        ...
"""
from __future__ import annotations

import logging
from typing import AsyncGenerator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config import config as _cfg

logger = logging.getLogger(__name__)

_db = _cfg.database

# ── Engine ────────────────────────────────────────────────────────────────────

engine = create_async_engine(
    _db.url,
    pool_size=_db.pool_size,
    max_overflow=_db.max_overflow,
    pool_timeout=_db.pool_timeout,
    pool_pre_ping=True,
    echo=_db.echo,
    future=True,
)

# ── Session factories ─────────────────────────────────────────────────────────

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Sync engine — used ONLY in background threads (no asyncio, no loop conflicts).
sync_engine = create_engine(
    _db.sync_url,
    pool_size=2,
    max_overflow=5,
    pool_timeout=30,
    pool_pre_ping=True,
    echo=_db.echo,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


# ── Base class for all ORM models ─────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── FastAPI dependency ────────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ── Startup helper ────────────────────────────────────────────────────────────

async def init_db() -> None:
    """Create all tables that do not exist yet (safe to call on every startup)."""
    from database import models  # noqa: F401 — registers all ORM classes
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ready.")


async def close_db() -> None:
    await engine.dispose()
    logger.info("Database connection pool closed.")
