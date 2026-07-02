
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.connection import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return uuid.uuid4().hex


class User(Base):
   

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(254), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    role: Mapped[str] = mapped_column(
        String(50), nullable=False, default="Developer"
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    active_github_repo: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True
    )
    github_token: Mapped[Optional[str]] = mapped_column(
        String(500), nullable=True
    )

    # relationships
    projects: Mapped[list["Project"]] = relationship(
        "Project", back_populates="owner", cascade="all, delete-orphan"
    )
    conversations: Mapped[list["Conversation"]] = relationship(
        "Conversation", back_populates="user", cascade="all, delete-orphan"
    )
    audit_logs: Mapped[list["AuditLog"]] = relationship(
        "AuditLog", back_populates="user"
    )
    notifications: Mapped[list["Notification"]] = relationship(
        "Notification", back_populates="user", cascade="all, delete-orphan"
    )



class Project(Base):
    """Registered project / repository root."""

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    owner_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    path_hash: Mapped[str] = mapped_column(
        String(12), nullable=False
    )  # SHA-256[:12] of resolved path — matches ca:chat:idx: key
    local_path: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    language: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    github_repo_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    active_branch: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    last_active_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_activity_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        UniqueConstraint("owner_id", "path_hash", name="uq_project_owner_path"),
        Index("ix_projects_path_hash", "path_hash"),
    )

    owner: Mapped["User"] = relationship("User", back_populates="projects")
    conversations: Mapped[list["Conversation"]] = relationship(
        "Conversation", back_populates="project", cascade="all, delete-orphan"
    )
    analysis_runs: Mapped[list["AnalysisRun"]] = relationship(
        "AnalysisRun", back_populates="project", cascade="all, delete-orphan"
    )
    cicd_reports: Mapped[list["CICDReport"]] = relationship(
        "CICDReport", back_populates="project", cascade="all, delete-orphan"
    )
    git_reports: Mapped[list["GitReport"]] = relationship(
        "GitReport", back_populates="project", cascade="all, delete-orphan"
    )
    test_runs: Mapped[list["TestGenerationRun"]] = relationship(
        "TestGenerationRun", back_populates="project", cascade="all, delete-orphan"
    )



class Conversation(Base):
    """
    Chat session. Maps 1:1 to a Redis session_id.
    After logout, the row stays — history survives.
    """

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(
        String(32), primary_key=True, default=_uuid
    )
    session_id: Mapped[str] = mapped_column(
        String(12), nullable=False, unique=True, index=True
    )  # matches ca:chat:{session_id} Redis key suffix
    user_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    project_id: Mapped[Optional[str]] = mapped_column(
        String(32), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True, index=True
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="New conversation")
    intent: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    turn_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, onupdate=_now
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index("ix_conversations_user_updated", "user_id", "updated_at"),
    )

    user: Mapped["User"] = relationship("User", back_populates="conversations")
    project: Mapped[Optional["Project"]] = relationship(
        "Project", back_populates="conversations"
    )
    messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan",
        order_by="Message.created_at",
    )



class Message(Base):
    """Individual turn in a conversation."""

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # "user" | "assistant" | "system"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        "metadata", JSONB, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, index=True
    )

    __table_args__ = (
        Index("ix_messages_conversation_created", "conversation_id", "created_at"),
    )

    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="messages"
    )


# ── analysis_runs ─────────────────────────────────────────────────────────────

class AnalysisRun(Base):
    """
    Watch-mode or on-demand file/project analysis result.
    Stores the structured output from lc_analysis_agent.
    """

    __tablename__ = "analysis_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    language: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    issues: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    fixes: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    strategy: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    source: Mapped[str] = mapped_column(
        String(20), nullable=False, default="watch"
    )  # "watch" | "ondemand" | "hardening"
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, index=True
    )

    __table_args__ = (
        Index("ix_analysis_runs_project_file", "project_id", "file_path"),
        Index("ix_analysis_runs_content_hash", "content_hash"),
    )

    project: Mapped["Project"] = relationship("Project", back_populates="analysis_runs")


class CICDReport(Base):
    """Result of one CI/CD pipeline analysis (CIGraph output)."""

    __tablename__ = "cicd_reports"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    repo: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    outcome: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    failure_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    stage_failed: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    root_cause: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    pr_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    elapsed_seconds: Mapped[Optional[float]] = mapped_column(nullable=True)
    raw_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, index=True
    )

    project: Mapped["Project"] = relationship("Project", back_populates="cicd_reports")


class GitReport(Base):
    """Branch analysis or PR review result from SmartGitGraph."""

    __tablename__ = "git_reports"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    report_type: Mapped[str] = mapped_column(
        String(30), nullable=False
    )  # "branch" | "pr_review" | "commit_lint" | "secret_scan" | "test_impact"
    branch: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    base_branch: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    pr_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    verdict: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    total_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    raw_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, index=True
    )

    __table_args__ = (
        Index("ix_git_reports_project_type", "project_id", "report_type"),
    )

    project: Mapped["Project"] = relationship("Project", back_populates="git_reports")


class TestGenerationRun(Base):
    """Record of a test generation job (incremental or full)."""

    __tablename__ = "test_generation_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    mode: Mapped[str] = mapped_column(String(10), nullable=False, default="inc")
    framework: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    test_code: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    validated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rag_docs_used: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, index=True
    )

    __table_args__ = (
        Index("ix_test_runs_project_file", "project_id", "file_path"),
    )

    project: Mapped["Project"] = relationship("Project", back_populates="test_runs")


class HistoryEvent(Base):
    """
    Unified timeline index for all dashboard actions.
    One row per significant event (chat started, CI/CD analyzed, PR reviewed, fix applied).
    Lightweight — keeps only title/summary/severity; detail is fetched via source_id.
    """

    __tablename__ = "history_events"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    project_id: Mapped[Optional[str]] = mapped_column(
        String(32), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True
    )
    event_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # chat_started | cicd_analyzed | git_pr_reviewed | fix_applied
    source_module: Mapped[str] = mapped_column(
        String(30), nullable=False
    )  # chat | cicd | smart_git | fixes
    source_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    severity: Mapped[str] = mapped_column(String(20), nullable=False, default="info")
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="completed")
    metadata_: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        "metadata_", JSONB, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, index=True
    )

    __table_args__ = (
        Index("ix_history_events_user_created", "user_id", "created_at"),
        Index("ix_history_events_project", "project_id", "created_at"),
        Index("ix_history_events_module", "source_module"),
    )


class DashboardFix(Base):
    """A fix suggestion applied by the developer from within the dashboard."""

    __tablename__ = "dashboard_fixes"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    project_id: Mapped[Optional[str]] = mapped_column(
        String(32), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True
    )
    source_module: Mapped[str] = mapped_column(String(30), nullable=False)
    source_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="applied")
    files_changed: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    applied_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, index=True
    )

    __table_args__ = (
        Index("ix_dashboard_fixes_user_created", "user_id", "created_at"),
    )


class ProjectMemoryFact(Base):
    """Persistent, project-scoped facts learned from dashboard activity."""

    __tablename__ = "project_memory_facts"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    project_id: Mapped[Optional[str]] = mapped_column(
        String(32), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True
    )
    fact_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # tech_stack | recurring_issue | convention | team_preference
    content: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(nullable=False, default=1.0)
    source_module: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, onupdate=_now
    )

    __table_args__ = (
        Index("ix_project_memory_facts_user", "user_id"),
    )


class KBEntry(Base):
    """
    Knowledge Base rule that has been approved and is active.
    The actual markdown lives on disk; this table tracks metadata,
    approval history, and enables fast lookup by language/pattern.
    """

    __tablename__ = "kb_entries"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    language: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    pattern_name: Mapped[str] = mapped_column(String(200), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False, default="MEDIUM")
    problem: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    solution: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    file_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="approved"
    )  # "pending" | "approved" | "rejected"
    frequency: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    auto_promoted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    approved_by: Mapped[Optional[str]] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    approved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        UniqueConstraint("language", "pattern_name", name="uq_kb_lang_pattern"),
        Index("ix_kb_language_status", "language", "status"),
    )


# ── audit_logs ────────────────────────────────────────────────────────────────

class AuditLog(Base):
    """
    Append-only compliance log. Never deleted or updated.
    Retention: minimum 1 year (enforced externally / by archival job).
    """

    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[Optional[str]] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    entity_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    details: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, index=True
    )

    __table_args__ = (
        Index("ix_audit_logs_user_created", "user_id", "created_at"),
        Index("ix_audit_logs_action", "action"),
    )

    user: Mapped[Optional["User"]] = relationship("User", back_populates="audit_logs")


class Notification(Base):
    """
    User-facing notifications (analysis complete, KB promotion, etc.).
    Soft-deleted by setting is_read=True; hard-deleted after 90 days by cleanup job.
    """

    __tablename__ = "notifications"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True
    )
    type: Mapped[str] = mapped_column(String(50), nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    body: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    payload: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    is_read: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, index=True
    )

    __table_args__ = (
        Index("ix_notifications_user_unread", "user_id", "is_read", "created_at"),
    )

    user: Mapped["User"] = relationship("User", back_populates="notifications")


# ── Settings Center ───────────────────────────────────────────────────────────

class UserSettings(Base):
    """UI and display preferences per user."""

    __tablename__ = "user_settings"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, unique=True, index=True,
    )
    theme: Mapped[str] = mapped_column(String(20), nullable=False, default="dark")
    language: Mapped[str] = mapped_column(String(10), nullable=False, default="en")
    density: Mapped[str] = mapped_column(String(20), nullable=False, default="comfortable")
    default_landing_page: Mapped[str] = mapped_column(String(50), nullable=False, default="assistant")
    show_ai_context_panel: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    show_sidebar_labels: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    show_cicd_in_sidebar: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    notifications: Mapped[Dict[str, Any]] = mapped_column(
        JSONB, nullable=False,
        default=lambda: {"enabled": True, "analysis": True, "cicd": True, "git": True},
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now, onupdate=_now)


class WorkspaceSettings(Base):
    """Workspace/environment-level settings per user."""

    __tablename__ = "workspace_settings"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, unique=True, index=True,
    )
    repository_id: Mapped[Optional[str]] = mapped_column(
        String(32), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True,
    )
    environment: Mapped[str] = mapped_column(String(20), nullable=False, default="local")
    backend_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    websocket_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    default_branch: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now, onupdate=_now)


class AISettings(Base):
    """ChatAgent / LLM behavior preferences per user."""

    __tablename__ = "ai_settings"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, unique=True, index=True,
    )
    ai_mode: Mapped[str] = mapped_column(String(30), nullable=False, default="balanced")
    response_style: Mapped[str] = mapped_column(String(30), nullable=False, default="detailed")
    use_repository_context: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    use_conversation_memory: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    use_rag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    use_cicd_context: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    use_smart_git_context: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    streaming_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    proactive_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    temperature: Mapped[float] = mapped_column(nullable=False, default=0.3)
    max_context_size: Mapped[int] = mapped_column(Integer, nullable=False, default=8000)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now, onupdate=_now)


class GitSettings(Base):
    """Git/GitHub integration preferences per user."""

    __tablename__ = "git_settings"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, unique=True, index=True,
    )
    provider: Mapped[str] = mapped_column(String(30), nullable=False, default="github")
    default_branch: Mapped[str] = mapped_column(String(300), nullable=False, default="main")
    safe_mode: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    pr_analysis_mode: Mapped[str] = mapped_column(String(20), nullable=False, default="fast")
    auto_refresh_prs: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    permissions: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now, onupdate=_now)


class CICDSettings(Base):
    """CI/CD pipeline preferences per user."""

    __tablename__ = "cicd_settings"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, unique=True, index=True,
    )
    repository_id: Mapped[Optional[str]] = mapped_column(
        String(32), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True,
    )
    provider: Mapped[str] = mapped_column(String(30), nullable=False, default="github_actions")
    pipeline_template: Mapped[str] = mapped_column(String(50), nullable=False, default="standard")
    quality_gate_threshold: Mapped[int] = mapped_column(Integer, nullable=False, default=80)
    auto_generate_pipeline: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    open_pr_after_generation: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    sonar_organization: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    sonar_project_key: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    docker_registry: Mapped[str] = mapped_column(String(30), nullable=False, default="dockerhub")
    deploy_provider: Mapped[str] = mapped_column(String(30), nullable=False, default="none")
    required_secrets: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now, onupdate=_now)


class SecuritySettings(Base):
    """Session and authentication preferences per user."""

    __tablename__ = "security_settings"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, unique=True, index=True,
    )
    session_timeout: Mapped[int] = mapped_column(Integer, nullable=False, default=1440)
    auto_logout_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    mask_sensitive_values: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_now, onupdate=_now)


class SettingsAuditLog(Base):
    """Append-only log of all settings changes. Never deleted."""

    __tablename__ = "settings_audit_log"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[Optional[str]] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True,
    )
    setting_scope: Mapped[str] = mapped_column(String(50), nullable=False)
    setting_key: Mapped[str] = mapped_column(String(200), nullable=False)
    old_value: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    new_value: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, index=True
    )

    __table_args__ = (
        Index("ix_settings_audit_user_changed", "user_id", "changed_at"),
        Index("ix_settings_audit_scope", "setting_scope", "changed_at"),
    )


# ── Extension chat (VS Code panel) — tables séparées du dashboard ────────────

class ExtensionChatSession(Base):
    """
    One conversation from the VS Code chat panel.
    Separate from `conversations` (dashboard + Redis-coupled).
    session_id  = short random hex generated by the extension.
    workspace   = workspace folder name (human label, not a FK).
    """

    __tablename__ = "extension_chat_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    session_id: Mapped[str] = mapped_column(
        String(32), nullable=False, unique=True, index=True
    )
    title: Mapped[str] = mapped_column(String(300), nullable=False, default="New chat")
    workspace: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, onupdate=_now
    )

    __table_args__ = (
        Index("ix_ext_chat_sessions_user_updated", "user_id", "updated_at"),
    )

    user: Mapped["User"] = relationship("User")
    messages: Mapped[list["ExtensionChatMessage"]] = relationship(
        "ExtensionChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ExtensionChatMessage.created_at",
    )


class ExtensionChatMessage(Base):
    """Individual turn in an extension chat session."""

    __tablename__ = "extension_chat_messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id_fk: Mapped[str] = mapped_column(
        "session_fk",
        String(32),
        ForeignKey("extension_chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)   # user | assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    context_file: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    ts: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)  # client epoch ms
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, index=True
    )

    __table_args__ = (
        Index("ix_ext_chat_messages_session_created", "session_fk", "created_at"),
    )

    session: Mapped["ExtensionChatSession"] = relationship(
        "ExtensionChatSession", back_populates="messages"
    )


# ── Issue actions & patch snapshots ───────────────────────────────────────────

class IssueActionEvent(Base):
    """
    Append-only log of every developer action on a VS Code issue
    (apply, verify, undo, ignore, false_positive, rescan).
    Source of truth for the issue History panel.
    """

    __tablename__ = "issue_action_events"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    workspace_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    issue_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    patch_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    action: Mapped[str] = mapped_column(String(30), nullable=False)
    previous_status: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    next_status: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    file_version_before_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    patch_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_: Mapped[Optional[Dict[str, Any]]] = mapped_column("metadata", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, index=True,
    )

    __table_args__ = (
        Index("ix_issue_action_events_user_issue", "user_id", "issue_id"),
        Index("ix_issue_action_events_created", "user_id", "created_at"),
    )


class PatchSnapshot(Base):
    """
    Full file content captured before and after a patch is applied.
    Enables safe undo: restore content_before without touching other changes.
    Created BEFORE apply — content_after filled in after WorkspaceEdit succeeds.
    """

    __tablename__ = "patch_snapshots"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    workspace_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    issue_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    patch_id: Mapped[str] = mapped_column(String(32), nullable=False, unique=True, index=True)
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    content_before: Mapped[str] = mapped_column(Text, nullable=False)
    content_after: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    content_hash_before: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    content_hash_after: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    applied_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    reverted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, index=True,
    )

    __table_args__ = (
        Index("ix_patch_snapshots_issue", "user_id", "issue_id"),
    )
