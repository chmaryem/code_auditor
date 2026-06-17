
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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    last_active_at: Mapped[Optional[datetime]] = mapped_column(
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
