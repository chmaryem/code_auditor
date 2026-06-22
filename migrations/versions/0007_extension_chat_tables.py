"""Add extension_chat_sessions and extension_chat_messages tables.

Revision ID: 0007
Revises: 0006
Create Date: 2026-06-19 00:00:00
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on:    Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── extension_chat_sessions ───────────────────────────────────────────────
    op.create_table(
        "extension_chat_sessions",
        sa.Column("id",         sa.String(32), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(32),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("session_id", sa.String(32),  nullable=False, unique=True),
        sa.Column("title",      sa.String(300), nullable=False, server_default="New chat"),
        sa.Column("workspace",  sa.String(500), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index(
        "ix_ext_chat_sessions_user_id",
        "extension_chat_sessions",
        ["user_id"],
    )
    op.create_index(
        "ix_ext_chat_sessions_session_id",
        "extension_chat_sessions",
        ["session_id"],
        unique=True,
    )
    op.create_index(
        "ix_ext_chat_sessions_user_updated",
        "extension_chat_sessions",
        ["user_id", "updated_at"],
    )

    # ── extension_chat_messages ───────────────────────────────────────────────
    op.create_table(
        "extension_chat_messages",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "session_fk",
            sa.String(32),
            sa.ForeignKey("extension_chat_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role",         sa.String(20),  nullable=False),
        sa.Column("content",      sa.Text,        nullable=False),
        sa.Column("context_file", sa.String(500), nullable=True),
        sa.Column("ts",           sa.BigInteger,  nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index(
        "ix_ext_chat_messages_session_created",
        "extension_chat_messages",
        ["session_fk", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_ext_chat_messages_session_created", table_name="extension_chat_messages")
    op.drop_table("extension_chat_messages")
    op.drop_index("ix_ext_chat_sessions_user_updated",  table_name="extension_chat_sessions")
    op.drop_index("ix_ext_chat_sessions_session_id",    table_name="extension_chat_sessions")
    op.drop_index("ix_ext_chat_sessions_user_id",       table_name="extension_chat_sessions")
    op.drop_table("extension_chat_sessions")
