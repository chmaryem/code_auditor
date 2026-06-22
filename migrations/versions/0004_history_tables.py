"""Add history_events, dashboard_fixes, project_memory_facts tables.
Extend projects with github_repo_url / active_branch / last_activity_at.
Add deleted_at to conversations.

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-18 00:00:00
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Extend projects ───────────────────────────────────────────────────────
    op.add_column("projects", sa.Column("github_repo_url", sa.String(500), nullable=True))
    op.add_column("projects", sa.Column("active_branch", sa.String(300), nullable=True))
    op.add_column(
        "projects",
        sa.Column(
            "last_activity_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )

    # ── Soft-delete on conversations ──────────────────────────────────────────
    op.add_column(
        "conversations",
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )

    # ── history_events ────────────────────────────────────────────────────────
    op.create_table(
        "history_events",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(32),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "project_id",
            sa.String(32),
            sa.ForeignKey("projects.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("source_module", sa.String(30), nullable=False),
        sa.Column("source_id", sa.String(32), nullable=True),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("summary", sa.Text, nullable=True),
        sa.Column("severity", sa.String(20), nullable=False, server_default="info"),
        sa.Column("status", sa.String(30), nullable=False, server_default="completed"),
        sa.Column("metadata_", JSONB, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index(
        "ix_history_events_user_created",
        "history_events",
        ["user_id", "created_at"],
    )
    op.create_index(
        "ix_history_events_project",
        "history_events",
        ["project_id", "created_at"],
    )
    op.create_index(
        "ix_history_events_module",
        "history_events",
        ["source_module"],
    )

    # ── dashboard_fixes ───────────────────────────────────────────────────────
    op.create_table(
        "dashboard_fixes",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(32),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "project_id",
            sa.String(32),
            sa.ForeignKey("projects.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("source_module", sa.String(30), nullable=False),
        sa.Column("source_id", sa.String(32), nullable=True),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="applied"),
        sa.Column("files_changed", JSONB, nullable=True),
        sa.Column("applied_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index(
        "ix_dashboard_fixes_user_created",
        "dashboard_fixes",
        ["user_id", "created_at"],
    )

    # ── project_memory_facts ──────────────────────────────────────────────────
    op.create_table(
        "project_memory_facts",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(32),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "project_id",
            sa.String(32),
            sa.ForeignKey("projects.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("fact_type", sa.String(50), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("source_module", sa.String(30), nullable=True),
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
        "ix_project_memory_facts_user",
        "project_memory_facts",
        ["user_id"],
    )


def downgrade() -> None:
    op.drop_table("project_memory_facts")
    op.drop_table("dashboard_fixes")
    op.drop_index("ix_history_events_module", table_name="history_events")
    op.drop_index("ix_history_events_project", table_name="history_events")
    op.drop_index("ix_history_events_user_created", table_name="history_events")
    op.drop_table("history_events")
    op.drop_column("conversations", "deleted_at")
    op.drop_column("projects", "last_activity_at")
    op.drop_column("projects", "active_branch")
    op.drop_column("projects", "github_repo_url")
