"""Initial schema — all persistent tables.

Revision ID: 0001
Revises:
Create Date: 2026-06-17 00:00:00
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── users ────────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("email", sa.String(254), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("role", sa.String(50), nullable=False, server_default="Developer"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    # ── projects ─────────────────────────────────────────────────────────────
    op.create_table(
        "projects",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("owner_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("path_hash", sa.String(12), nullable=False),
        sa.Column("local_path", sa.String(1000), nullable=True),
        sa.Column("language", sa.String(50), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_active_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_projects_owner_id", "projects", ["owner_id"])
    op.create_index("ix_projects_path_hash", "projects", ["path_hash"])
    op.create_unique_constraint("uq_project_owner_path", "projects", ["owner_id", "path_hash"])

    # ── conversations ─────────────────────────────────────────────────────────
    op.create_table(
        "conversations",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("session_id", sa.String(12), nullable=False),
        sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("project_id", sa.String(32), sa.ForeignKey("projects.id", ondelete="SET NULL"), nullable=True),
        sa.Column("title", sa.String(200), nullable=False, server_default="New conversation"),
        sa.Column("intent", sa.String(100), nullable=True),
        sa.Column("turn_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_conversations_session_id", "conversations", ["session_id"], unique=True)
    op.create_index("ix_conversations_user_id", "conversations", ["user_id"])
    op.create_index("ix_conversations_project_id", "conversations", ["project_id"])
    op.create_index("ix_conversations_user_updated", "conversations", ["user_id", "updated_at"])

    # ── messages ──────────────────────────────────────────────────────────────
    op.create_table(
        "messages",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("conversation_id", sa.String(32), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"])
    op.create_index("ix_messages_created_at", "messages", ["created_at"])
    op.create_index("ix_messages_conversation_created", "messages", ["conversation_id", "created_at"])

    # ── analysis_runs ──────────────────────────────────────────────────────────
    op.create_table(
        "analysis_runs",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("project_id", sa.String(32), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("file_path", sa.String(1000), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("language", sa.String(50), nullable=True),
        sa.Column("score", sa.Integer(), nullable=True),
        sa.Column("issues", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("fixes", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("strategy", sa.String(50), nullable=True),
        sa.Column("source", sa.String(20), nullable=False, server_default="watch"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_analysis_runs_project_id", "analysis_runs", ["project_id"])
    op.create_index("ix_analysis_runs_created_at", "analysis_runs", ["created_at"])
    op.create_index("ix_analysis_runs_project_file", "analysis_runs", ["project_id", "file_path"])
    op.create_index("ix_analysis_runs_content_hash", "analysis_runs", ["content_hash"])

    # ── cicd_reports ───────────────────────────────────────────────────────────
    op.create_table(
        "cicd_reports",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("project_id", sa.String(32), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("repo", sa.String(500), nullable=True),
        sa.Column("outcome", sa.String(50), nullable=True),
        sa.Column("failure_type", sa.String(100), nullable=True),
        sa.Column("stage_failed", sa.String(200), nullable=True),
        sa.Column("root_cause", sa.Text(), nullable=True),
        sa.Column("pr_number", sa.Integer(), nullable=True),
        sa.Column("elapsed_seconds", sa.Float(), nullable=True),
        sa.Column("raw_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_cicd_reports_project_id", "cicd_reports", ["project_id"])
    op.create_index("ix_cicd_reports_created_at", "cicd_reports", ["created_at"])

    # ── git_reports ────────────────────────────────────────────────────────────
    op.create_table(
        "git_reports",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("project_id", sa.String(32), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("report_type", sa.String(30), nullable=False),
        sa.Column("branch", sa.String(300), nullable=True),
        sa.Column("base_branch", sa.String(300), nullable=True),
        sa.Column("pr_number", sa.Integer(), nullable=True),
        sa.Column("verdict", sa.String(50), nullable=True),
        sa.Column("total_score", sa.Integer(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("raw_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_git_reports_project_id", "git_reports", ["project_id"])
    op.create_index("ix_git_reports_created_at", "git_reports", ["created_at"])
    op.create_index("ix_git_reports_project_type", "git_reports", ["project_id", "report_type"])

    # ── test_generation_runs ───────────────────────────────────────────────────
    op.create_table(
        "test_generation_runs",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("project_id", sa.String(32), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("file_path", sa.String(1000), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("mode", sa.String(10), nullable=False, server_default="inc"),
        sa.Column("framework", sa.String(50), nullable=True),
        sa.Column("test_code", sa.Text(), nullable=True),
        sa.Column("validated", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("rag_docs_used", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_test_generation_runs_project_id", "test_generation_runs", ["project_id"])
    op.create_index("ix_test_generation_runs_created_at", "test_generation_runs", ["created_at"])
    op.create_index("ix_test_runs_project_file", "test_generation_runs", ["project_id", "file_path"])

    # ── kb_entries ─────────────────────────────────────────────────────────────
    op.create_table(
        "kb_entries",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("language", sa.String(50), nullable=False),
        sa.Column("pattern_name", sa.String(200), nullable=False),
        sa.Column("severity", sa.String(20), nullable=False, server_default="MEDIUM"),
        sa.Column("problem", sa.Text(), nullable=True),
        sa.Column("solution", sa.Text(), nullable=True),
        sa.Column("file_path", sa.String(500), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="approved"),
        sa.Column("frequency", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("auto_promoted", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("approved_by", sa.String(32), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_kb_entries_language", "kb_entries", ["language"])
    op.create_index("ix_kb_language_status", "kb_entries", ["language", "status"])
    op.create_unique_constraint("uq_kb_lang_pattern", "kb_entries", ["language", "pattern_name"])

    # ── audit_logs ─────────────────────────────────────────────────────────────
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("entity_type", sa.String(50), nullable=True),
        sa.Column("entity_id", sa.String(32), nullable=True),
        sa.Column("details", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_audit_logs_user_id", "audit_logs", ["user_id"])
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"])
    op.create_index("ix_audit_logs_user_created", "audit_logs", ["user_id", "created_at"])
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"])

    # ── notifications ──────────────────────────────────────────────────────────
    op.create_table(
        "notifications",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("user_id", sa.String(32), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("type", sa.String(50), nullable=False),
        sa.Column("title", sa.String(300), nullable=False),
        sa.Column("body", sa.Text(), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("is_read", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_notifications_user_id", "notifications", ["user_id"])
    op.create_index("ix_notifications_is_read", "notifications", ["is_read"])
    op.create_index("ix_notifications_created_at", "notifications", ["created_at"])
    op.create_index("ix_notifications_user_unread", "notifications", ["user_id", "is_read", "created_at"])


def downgrade() -> None:
    op.drop_table("notifications")
    op.drop_table("audit_logs")
    op.drop_table("kb_entries")
    op.drop_table("test_generation_runs")
    op.drop_table("git_reports")
    op.drop_table("cicd_reports")
    op.drop_table("analysis_runs")
    op.drop_table("messages")
    op.drop_table("conversations")
    op.drop_table("projects")
    op.drop_table("users")
