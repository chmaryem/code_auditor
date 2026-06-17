"""
state.py — Shared state definitions for LangGraph graphs.

Each graph uses a TypedDict state that flows through all nodes.
This is the ONLY place where state schemas are defined.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class WatchState(TypedDict, total=False):
    """State for the WatchGraph pipeline."""

    file_path: str
    project_path: str
    code: str
    old_content: str
    content_hash: str
    language: str
    change_info: Dict[str, Any]
    parsed: Dict[str, Any]
    neighborhood: Dict[str, Any]
    rag_docs: List[Dict[str, Any]]
    rag_scores: List[float]
    patterns: List[str]
    test_gap: Optional[Dict[str, Any]]
    git_session: Optional[Dict[str, Any]]
    context: Dict[str, Any]
    analysis: Dict[str, Any]
    strategy: str
    learning_result: Dict[str, Any]
    skip_reason: Optional[str]
    dependents_to_analyze: List[str]
    post_solution_mode: bool
    stats: Dict[str, Any]

    # ── Structured output for VS Code plugin ────────────────────────────────
    # Parsed issues extracted from raw LLM text (line, severity, rule, …)
    issues: List[Dict[str, Any]]
    # Ready-to-broadcast WebSocket events (analysis_result, dependency_impact,
    # test_gap, git_recommendation, known_issue)
    ws_events: List[Dict[str, Any]]

    _project_indexer: Any
    _extractor: Any
    _rag_system: Any
    _dep_graph: Any
    _cache: Any
    _print_lock: Any
    _learning_agent: Any
    _file_counter: Any
    # Thread-safe callback(event: dict) -> None used to push WS events to the
    # plugin *incrementally* (primary result first, dependents as they finish)
    # instead of batching everything at the end of the pipeline.
    _ws_broadcast: Any
    # Set True by node_emit_ws_events when it has already pushed events through
    # _ws_broadcast, so the caller does not re-broadcast (avoids duplicates).
    ws_broadcasted: bool


class CIState(TypedDict, total=False):
    """State for the CIGraph pipeline."""

    run_id: str
    repo: str
    owner: str
    project_key: str
    pr_number: Optional[int]
    head_sha: str
    pr_branch: str
    logs: str
    stage_failed: Optional[str]
    run_conclusion: str
    run_duration_seconds: int
    outcome: str
    failure_type: str
    severity: str
    sonar_gate: Dict[str, Any]
    sonar_metrics: Dict[str, Any]
    sonar_issues: List[Dict[str, Any]]
    similar_fixes: List[Dict[str, Any]]
    confidence: float
    root_cause: Optional[str]
    suggested_fix: Optional[str]
    comment_posted: bool
    indexed: bool
    notification_level: str
    codeql_alerts: List[Dict[str, Any]]
    trivy_report: Dict[str, Any]
    auto_retried: bool


class CDState(TypedDict, total=False):
    """State for the CDGraph pipeline."""

    run_id: str
    repo: str
    owner: str
    project_key: str
    pr_number: Optional[int]
    head_sha: str
    pr_branch: str
    environment: str
    deploy_id: str
    version: str
    deploy_url: str
    run_conclusion: str
    deploy_status: str
    readiness_score: float
    readiness_verdict: str
    readiness_blocking_reasons: List[str]
    readiness_warnings: List[str]
    readiness_components: Dict[str, float]
    readiness_markdown: str
    env_state: Dict[str, Any]
    last_ok_deploy: Dict[str, Any]
    monitor_grade: str
    monitor_availability: float
    monitor_avg_latency_ms: float
    monitor_regression: bool
    monitor_issues: List[str]
    deploy_failure_reason: str
    deploy_suggested_fix: str
    rollback_available: bool
    rollback_command: str
    rollback_risk: str
    rollback_risk_reasons: List[str]
    rollback_comment_body: str
    comment_posted: bool
    indexed: bool
    notification_level: str


class ChatState(TypedDict, total=False):
    """
    State for the ChatGraph pipeline.

    ChatGraph now supports:
      - Memory-aware decision routing
      - Fast explain path
      - Contextual Q&A path
      - Code generation path
      - SSE streaming path
    """

    # ── Input ────────────────────────────────────────────────────────────────
    session_id: str
    user_id: str          # authenticated user id (empty = anonymous / dev mode)
    user_message: str
    project_path: str
    target_file: Optional[str]
    target_lang: str

    # ── IDE Cursor context (VS Code extension → API) ─────────────────────────
    cursor_line: int             # 0 = unknown
    active_function: str         # name of function/method under cursor
    selected_text: str           # highlighted code (empty if none)
    visible_range: List[int]     # [start_line, end_line] of visible editor area

    # ── Git context snapshot (injected if available) ─────────────────────────
    git_context: Dict[str, Any]  # session_snapshot from GitSessionTracker
    ci_context: Dict[str, Any]   # last CI run summary (injected if available)
    project_state_context: Dict[str, Any]  # git_risk/secrets/test_gaps/security_quality/ci_readiness snapshot

    # ── Intent routing ──────────────────────────────────────────────────────
    intent: str
    intent_params: Dict[str, Any]

    # ── Decision Agent / Orchestration ─────────────────────────────────────
    decision_plan: Dict[str, Any]
    context_level: str        # "fast" | "context" | "deep"
    selected_agents: List[str]
    needs_rag: bool
    needs_git: bool
    needs_ci: bool
    needs_generation: bool
    needs_tests: bool

    # ── Loaded context ──────────────────────────────────────────────────────
    file_code: str
    file_analysis: Dict[str, Any]
    dependencies: List[str]
    dependents: List[str]
    rag_docs: List[Dict[str, Any]]
    rag_scores: List[float]
    project_summary: Dict[str, Any]

    # ── Phase 2 — Code generation ────────────────────────────────────────────
    generation_target: str
    generation_language: str
    generated_code: str
    generation_valid: bool
    generation_errors: List[str]
    project_patterns: Dict[str, Any]
    apply_to_disk: bool

    # ── Conversation memory ─────────────────────────────────────────────────
    history: List[Dict[str, Any]]
    memory_key: str
    file_cache: Dict[str, Any]       # per-invocation cache; Redis session cache can be added later

    # ── Output ──────────────────────────────────────────────────────────────
    response: str
    formatted_response: str
    context_sources: List[str]  # which project_state_context sections were available
    code_blocks: List[Dict[str, Any]]
    suggested_files: List[str]
    stats: Dict[str, Any]

    # ── Injected services ───────────────────────────────────────────────────
    _rag_system: Any
    _cache: Any
    _indexer: Any
    _dep_graph: Any
