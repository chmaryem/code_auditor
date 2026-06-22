"""Add Settings Center tables: user_settings, workspace_settings, ai_settings,
git_settings, cicd_settings, security_settings, settings_audit_log.

Revision ID: 0006
Revises: 0005
Create Date: 2026-06-19
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from alembic import op

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── user_settings ─────────────────────────────────────────────────────────
    op.create_table(
        "user_settings",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(32),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
            index=True,
        ),
        sa.Column("theme", sa.String(20), nullable=False, server_default="dark"),
        sa.Column("language", sa.String(10), nullable=False, server_default="en"),
        sa.Column("density", sa.String(20), nullable=False, server_default="comfortable"),
        sa.Column(
            "default_landing_page",
            sa.String(50),
            nullable=False,
            server_default="assistant",
        ),
        sa.Column(
            "show_ai_context_panel", sa.Boolean, nullable=False, server_default="true"
        ),
        sa.Column(
            "show_sidebar_labels", sa.Boolean, nullable=False, server_default="true"
        ),
        sa.Column(
            "show_cicd_in_sidebar", sa.Boolean, nullable=False, server_default="true"
        ),
        sa.Column(
            "notifications",
            JSONB,
            nullable=False,
            server_default='{"enabled": true, "analysis": true, "cicd": true, "git": true}',
        ),
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

    # ── workspace_settings ────────────────────────────────────────────────────
    op.create_table(
        "workspace_settings",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(32),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
            index=True,
        ),
        sa.Column(
            "repository_id",
            sa.String(32),
            sa.ForeignKey("projects.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "environment", sa.String(20), nullable=False, server_default="local"
        ),
        sa.Column("backend_url", sa.String(500), nullable=True),
        sa.Column("websocket_url", sa.String(500), nullable=True),
        sa.Column("default_branch", sa.String(300), nullable=True),
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

    # ── ai_settings ───────────────────────────────────────────────────────────
    op.create_table(
        "ai_settings",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(32),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
            index=True,
        ),
        sa.Column(
            "ai_mode", sa.String(30), nullable=False, server_default="balanced"
        ),
        sa.Column(
            "response_style", sa.String(30), nullable=False, server_default="detailed"
        ),
        sa.Column(
            "use_repository_context",
            sa.Boolean,
            nullable=False,
            server_default="true",
        ),
        sa.Column(
            "use_conversation_memory",
            sa.Boolean,
            nullable=False,
            server_default="true",
        ),
        sa.Column("use_rag", sa.Boolean, nullable=False, server_default="true"),
        sa.Column(
            "use_cicd_context", sa.Boolean, nullable=False, server_default="false"
        ),
        sa.Column(
            "use_smart_git_context",
            sa.Boolean,
            nullable=False,
            server_default="false",
        ),
        sa.Column(
            "streaming_enabled", sa.Boolean, nullable=False, server_default="true"
        ),
        sa.Column(
            "proactive_enabled", sa.Boolean, nullable=False, server_default="false"
        ),
        sa.Column("temperature", sa.Float, nullable=False, server_default="0.3"),
        sa.Column(
            "max_context_size", sa.Integer, nullable=False, server_default="8000"
        ),
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

    # ── git_settings ──────────────────────────────────────────────────────────
    op.create_table(
        "git_settings",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(32),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
            index=True,
        ),
        sa.Column(
            "provider", sa.String(30), nullable=False, server_default="github"
        ),
        sa.Column(
            "default_branch", sa.String(300), nullable=False, server_default="main"
        ),
        sa.Column("safe_mode", sa.Boolean, nullable=False, server_default="true"),
        sa.Column(
            "pr_analysis_mode", sa.String(20), nullable=False, server_default="fast"
        ),
        sa.Column(
            "auto_refresh_prs", sa.Boolean, nullable=False, server_default="false"
        ),
        sa.Column("permissions", JSONB, nullable=False, server_default="{}"),
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

    # ── cicd_settings ─────────────────────────────────────────────────────────
    op.create_table(
        "cicd_settings",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(32),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
            index=True,
        ),
        sa.Column(
            "repository_id",
            sa.String(32),
            sa.ForeignKey("projects.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "provider",
            sa.String(30),
            nullable=False,
            server_default="github_actions",
        ),
        sa.Column(
            "pipeline_template",
            sa.String(50),
            nullable=False,
            server_default="standard",
        ),
        sa.Column(
            "quality_gate_threshold",
            sa.Integer,
            nullable=False,
            server_default="80",
        ),
        sa.Column(
            "auto_generate_pipeline",
            sa.Boolean,
            nullable=False,
            server_default="false",
        ),
        sa.Column(
            "open_pr_after_generation",
            sa.Boolean,
            nullable=False,
            server_default="false",
        ),
        sa.Column("sonar_organization", sa.String(200), nullable=True),
        sa.Column("sonar_project_key", sa.String(200), nullable=True),
        sa.Column(
            "docker_registry",
            sa.String(30),
            nullable=False,
            server_default="dockerhub",
        ),
        sa.Column(
            "deploy_provider", sa.String(30), nullable=False, server_default="none"
        ),
        sa.Column("required_secrets", JSONB, nullable=False, server_default="{}"),
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

    # ── security_settings ─────────────────────────────────────────────────────
    op.create_table(
        "security_settings",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(32),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
            index=True,
        ),
        sa.Column(
            "session_timeout", sa.Integer, nullable=False, server_default="1440"
        ),
        sa.Column(
            "auto_logout_enabled", sa.Boolean, nullable=False, server_default="false"
        ),
        sa.Column(
            "mask_sensitive_values",
            sa.Boolean,
            nullable=False,
            server_default="true",
        ),
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

    # ── settings_audit_log ────────────────────────────────────────────────────
    op.create_table(
        "settings_audit_log",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(32),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
        sa.Column("setting_scope", sa.String(50), nullable=False),
        sa.Column("setting_key", sa.String(200), nullable=False),
        sa.Column("old_value", JSONB, nullable=True),
        sa.Column("new_value", JSONB, nullable=True),
        sa.Column(
            "changed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index(
        "ix_settings_audit_user_changed",
        "settings_audit_log",
        ["user_id", "changed_at"],
    )
    op.create_index(
        "ix_settings_audit_scope",
        "settings_audit_log",
        ["setting_scope", "changed_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_settings_audit_scope", table_name="settings_audit_log")
    op.drop_index("ix_settings_audit_user_changed", table_name="settings_audit_log")
    op.drop_table("settings_audit_log")
    op.drop_table("security_settings")
    op.drop_table("cicd_settings")
    op.drop_table("git_settings")
    op.drop_table("ai_settings")
    op.drop_table("workspace_settings")
    op.drop_table("user_settings")
