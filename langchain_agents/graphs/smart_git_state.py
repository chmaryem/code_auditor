"""
smart_git_state.py — Shared state for SmartGitGraph.

This state is used by the LangGraph multi-agent Smart Git system.

Goal:
  Wrap existing Smart Git services/hooks into a clean multi-agent graph
  callable by:
    - ChatAgent
    - FastAPI
    - VS Code plugin
    - CLI
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class SmartGitState(TypedDict, total=False):
    # ── Input ────────────────────────────────────────────────────────────────
    user_message: str
    project_path: str
    repo: str
    owner: str
    pr_number: int
    branch: str
    base: str
    session_id: str

    # ── Decision ─────────────────────────────────────────────────────────────
    intent: str
    confidence: float
    selected_agents: List[str]
    needs_confirmation: bool
    safe_mode: bool
    reason: str

    # ── Git facts ────────────────────────────────────────────────────────────
    changed_files: List[Dict[str, Any]]
    staged_files: List[Dict[str, Any]]
    changes: Dict[str, Any]
    session_snapshot: Dict[str, Any]
    branch_report: Dict[str, Any]
    pr_report: Dict[str, Any]
    readiness_report: Dict[str, Any]
    conflict_report: Dict[str, Any]
    commit_message: str

    # ── Output ───────────────────────────────────────────────────────────────
    response: str
    actions: List[Dict[str, Any]]
    warnings: List[str]
    errors: List[str]
    stats: Dict[str, Any]