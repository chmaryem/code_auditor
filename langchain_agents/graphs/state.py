"""
state.py — Shared state definitions for LangGraph graphs.

Each graph uses a TypedDict state that flows through all nodes.
This is the ONLY place where state schemas are defined.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


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
    # Numeric GitHub Actions job id — when set, node_fetch_run fetches logs for
    # exactly this job (tool_fetch_job_logs) instead of guessing the failed
    # stage from the whole run's zip archive. Populated when the dashboard's
    # "Analyze root cause" is clicked on a specific issue/job.
    job_id: str
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
    # Rendered PR-comment markdown, always populated even when dry_run=True
    # and nothing is actually posted — lets the dashboard preview it.
    report_markdown: str
    # When True: no Redis persistence (CDDeployTracker) and no PR comment is
    # posted. Used for on-demand dashboard previews of the CDGraph, since
    # workflow_dispatch (how the dashboard always triggers pipelines) never
    # satisfies the publish/deploy jobs' `event_name == 'push'` condition —
    # meaning the graph's real trigger path (CIPoller reacting to an actual
    # publish/deploy job) is otherwise unreachable from the dashboard.
    dry_run: bool


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

    # ── Dashboard context (sent by the React webview) ─────────────────────────
    active_module: str        # current dashboard module: cicd | git | chat | analyze | tests
    branch: str               # current git branch
    active_repository: str    # name of the active repository (from dashboard)

    # ── Scope / périmètre du chat ─────────────────────────────────────────────
    scope: str                          # "extension" (défaut, analyse projet complète) | "dashboard" (restreint)
    attached_files: List[Dict[str, Any]]  # [{path, content}] fournis par le dev via le bouton d'ajout (dashboard)
    dashboard_needs_attachment: bool    # True → question code en mode dashboard sans fichier attaché (nudge)

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
    active_agents: List[str]  # agents blackboard réellement activés (SMA — Phase 2)
    tools_called: List[str]   # outils appelés lors de l'escalade ReAct (SMA — Phase 4)
    quality_score: float      # garde-fou qualité déterministe 0-1 (SMA — Phase 6)
    quality_flags: List[str]  # signaux qualité (empty_or_failed | not_grounded | thin)
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

    # ── Télémétrie (migration SMA — Phase 0.2) ───────────────────────────────
    # Compteur d'appels LLM de PREMIER PLAN pour ce tour. Reducer additif : chaque
    # nœud qui appelle le LLM retourne {"llm_calls": n}, LangGraph somme le tout.
    # Sert à chiffrer le gain tokens "avant/après" (exposé dans l'événement `done`).
    llm_calls: Annotated[int, operator.add]

    # ── AI Settings (loaded from PostgreSQL per user, defaults used if absent) ──
    ai_temperature: float        # 0.0–1.0, default 0.3
    ai_mode: str                 # "fast" | "balanced" | "deep" | "strict"
    ai_response_style: str       # "concise" | "detailed" | "professional" | "step_by_step"
    ai_use_rag: bool             # enable/disable RAG retrieval
    ai_use_memory: bool          # enable/disable conversation memory
    ai_max_context: int          # max chars fed to LLM (default 8000)
    ai_streaming: bool           # SSE streaming toggle

    # ── Injected services ───────────────────────────────────────────────────
    _rag_system: Any
    _cache: Any
    _indexer: Any
    _dep_graph: Any


class SmartGitState(TypedDict, total=False):
    """
    State for the SmartGitGraph pipeline (multi-agent Smart Git orchestration).

    Flow: node_router (RouterAgent, LLM) → [route_by_intent] → one worker node
    (WorkingCopyAgent | PRAgent | ConflictAgent) → node_synthesize (join) → END.
    Worker nodes select their tool(s) by `intent` and write their report field;
    node_synthesize reads whatever report is present and produces `response`.
    """

    # ── Input ────────────────────────────────────────────────────────────────
    user_message: str
    project_path: str
    owner: str
    repo: str
    pr_number: int
    branch: str
    base: str
    session_id: str

    # ── Decision (written by node_router) ────────────────────────────────────
    intent: str
    confidence: float
    branch_name: str            # branch explicitly named in the message (LLM-extracted)
    selected_agents: List[str]
    needs_confirmation: bool
    safe_mode: bool
    reason: str

    # ── Worker report fields (each worker writes the one matching its intent) ─
    session_snapshot: Dict[str, Any]
    changes: Dict[str, Any]
    commit_message: str
    branch_report: Dict[str, Any]
    readiness_report: Dict[str, Any]
    pr_report: Dict[str, Any]
    pr_description: Dict[str, Any]
    conflict_report: Dict[str, Any]
    secret_scan_report: Dict[str, Any]
    commit_lint_report: Dict[str, Any]
    test_impact_report: Dict[str, Any]
    cross_pr_report: Dict[str, Any]
    repo_overview_report: Dict[str, Any]

    # ── Output (written by node_synthesize) ──────────────────────────────────
    response: str
    actions: List[Any]
    warnings: List[Any]
    errors: List[Any]
    stats: Dict[str, Any]
