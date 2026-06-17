"""
migrations/env.py — Alembic environment configuration.

Supports both offline (SQL script) and online (direct DB) migration modes.
Uses the synchronous psycopg2 driver because Alembic doesn't support asyncpg natively.
"""
from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# ── Import all models so Alembic can detect schema changes ────────────────────
from database.connection import Base
import database.models  # noqa: F401 — registers all ORM classes with Base.metadata

# ── Alembic Config object ─────────────────────────────────────────────────────
config = context.config

# Override sqlalchemy.url from env var if set
db_url = os.getenv(
    "DATABASE_SYNC_URL",
    "postgresql+psycopg2://codeauditor:codeauditor@localhost:5432/codeauditor",
)
config.set_main_option("sqlalchemy.url", db_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Generate migration SQL without a live DB connection."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live database."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
