"""
state.py — Shared state definitions for LangGraph graphs.

Each graph uses a TypedDict state that flows through all nodes.
This is the ONLY place where state schemas are defined.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class WatchState(TypedDict, total=False):
    """
    State for the WatchGraph pipeline (15 nodes).

    This state flows through every node. Each node reads what it needs
    and writes its outputs. LangGraph manages the state transitions.
    """

    # ── Input ────────────────────────────────────────────────────────────────
    file_path: str                   # Absolute path to the file being analyzed
    project_path: str                # Project root directory

    # ── CodeAgent outputs ────────────────────────────────────────────────────
    code: str                        # Current file content
    old_content: str                 # Previous file content (for diff)
    content_hash: str                # SHA-256 hash of current content
    language: str                    # Detected programming language
    change_info: Dict[str, Any]      # Change analysis result (score, type, etc.)
    parsed: Dict[str, Any]           # AST parse result (entities, imports, etc.)

    # ── RetrieverAgent outputs ───────────────────────────────────────────────
    neighborhood: Dict[str, Any]     # Dependency graph neighborhood
    rag_docs: List[Dict[str, Any]]   # RAG documents (serialized)
    rag_scores: List[float]          # RAG similarity scores
    patterns: List[str]              # Detected KG patterns

    # ── TestGapAgent outputs ─────────────────────────────────────────────────
    test_gap: Optional[Dict[str, Any]] # Test gap status (None = no gap or test file)

    # ── GitSessionAgent outputs ───────────────────────────────────────────────
    git_session: Optional[Dict[str, Any]] # Git session status from Redis (None = not checked)

    # ── AnalysisAgent outputs ────────────────────────────────────────────────
    context: Dict[str, Any]          # Enriched context for LLM prompt
    analysis: Dict[str, Any]         # Full analysis result from LLM
    strategy: str                    # Repair strategy chosen by LLM

    # ── LearningAgent outputs ────────────────────────────────────────────────
    learning_result: Dict[str, Any]  # Patterns recorded, rules promoted

    # ── Graph control ────────────────────────────────────────────────────────
    skip_reason: Optional[str]       # None = continue, str = skip with reason
    dependents_to_analyze: List[str] # Files impacted by changes
    post_solution_mode: bool         # True if file was already fixed

    # ── Stats ────────────────────────────────────────────────────────────────
    stats: Dict[str, Any]            # Timing, node counts, etc.

    # ── Shared services (injected at graph.invoke) ───────────────────────────
    # These are service references passed through the state.
    # They are NOT serialized — they are live object references.
    _project_indexer: Any            # ProjectCodeIndexer instance
    _extractor: Any                  # DependencyExtractor instance
    _rag_system: Any                 # CodeRAGSystemAPI instance
    _dep_graph: Any                  # nx.DiGraph — dependency graph
    _cache: Any                      # CacheService instance
    _print_lock: Any                 # threading.Lock for console output
    _learning_agent: Any             # Legacy LearningAgent instance
    _file_counter: Any               # Shared dict for session stats


# ── CI Graph State ─────────────────────────────────────────────────────────────

class CIState(TypedDict, total=False):
    """
    State for the CIGraph pipeline (CI/CD Intelligence).

    Triggered by polling GitHub Actions runs. Flows through:
    fetch_run → classify_failure → sonar_mcp_query → search_similar
    → analyze_root_cause → generate_fix → post_pr_comment
    → index_result → notify → END
    """

    # ── Input (from polling) ─────────────────────────────────────────────────
    run_id: str                      # GitHub Actions workflow run ID
    repo: str                        # "{owner}/{repo}"
    owner: str                       # Repository owner
    project_key: str                 # SonarCloud project key (ex: "chmaryem_myapp")
    pr_number: Optional[int]         # PR number (None if push to branch, not PR)
    head_sha: str                    # HEAD commit SHA of the run
    pr_branch: str                   # Branche source de la PR (ex: 'feature/test-hook')

    # ── Run data ────────────────────────────────────────────────────────────
    logs: str                        # Raw workflow run logs (truncated)
    stage_failed: Optional[str]      # Name of the failed stage/job
    run_conclusion: str              # "success" | "failure" | "cancelled" | "skipped"
    run_duration_seconds: int        # Run duration in seconds

    # ── Classification ───────────────────────────────────────────────────────
    outcome: str                     # "success" | "failure" | "cancelled"
    failure_type: str                # "build" | "test" | "security" | "docker" | "unknown"
    severity: str                    # "URGENT" | "WARN" | "INFO"

    # ── SonarCloud MCP outputs ───────────────────────────────────────────────
    sonar_gate: Dict[str, Any]       # Quality Gate {status, conditions}
    sonar_metrics: Dict[str, Any]    # coverage, bugs, vulnerabilities, etc.
    sonar_issues: List[Dict[str, Any]]  # Critical/Blocker issues

    # ── Similarity search ────────────────────────────────────────────────────
    similar_fixes: List[Dict[str, Any]]  # Redis similar failures + their fixes
    confidence: float                # 0.0–1.0 — confidence of similar fix match

    # ── LLM Analysis ─────────────────────────────────────────────────────────
    root_cause: Optional[str]        # LLM root cause analysis
    suggested_fix: Optional[str]     # LLM suggested fix

    # ── Actions taken ────────────────────────────────────────────────────────
    comment_posted: bool             # True if PR comment was posted
    indexed: bool                    # True if run was indexed in Redis
    notification_level: str          # "URGENT" | "WARN" | "INFO" — final notification

    # ── Security Intelligence (T1+T2) ────────────────────────────────────────
    codeql_alerts: List[Dict[str, Any]]  # CodeQL alerts [{rule_id, severity, description, location_path}]
    trivy_report: Dict[str, Any]         # Trivy CVEs {critical:[...], high:[...], total:N, source:str}
    auto_retried: bool               # True if flaky test auto-retry was triggered (T6)


# ── CD Graph State ─────────────────────────────────────────────────────────────

class CDState(TypedDict, total=False):
    """
    State for the CDGraph pipeline (Continuous Deployment Intelligence).

    Triggered when a deploy job completes (success OR failure) in GitHub Actions.
    Flows through:
      fetch_deployment → pre_deploy_risk_score → check_environment
      → monitor_health → analyze_failure → suggest_rollback
      → post_deploy_report → index_result → notify → END
    """

    # ── Input (from CIPoller or webhook) ─────────────────────────────────────
    run_id:      str                   # GitHub Actions run ID (deploy job)
    repo:        str                   # "{owner}/{repo}"
    owner:       str                   # Repository owner
    project_key: str                   # SonarCloud key (for readiness scoring)
    pr_number:   Optional[int]         # PR number (for readiness + commenting)
    head_sha:    str                   # HEAD commit SHA
    pr_branch:   str                   # Source branch
    environment: str                   # "production" | "staging" | "dev"

    # ── Deployment metadata ──────────────────────────────────────────────────
    deploy_id:      str                # CDDeployTracker deploy_id
    version:        str                # Docker image tag / semver being deployed
    deploy_url:     str                # URL to health-check after deploy
    run_conclusion: str                # "success" | "failure" | "cancelled"
    deploy_status:  str                # "deploying" | "success" | "failed"

    # ── Pre-deploy Readiness Score ───────────────────────────────────────────
    readiness_score:            float                # 0–100
    readiness_verdict:          str                  # "DEPLOY_OK"|"DEPLOY_WARN"|"DEPLOY_BLOCKED"
    readiness_blocking_reasons: List[str]
    readiness_warnings:         List[str]
    readiness_components:       Dict[str, float]     # per-component scores
    readiness_markdown:         str                  # formatted PR comment block

    # ── Environment check ────────────────────────────────────────────────────
    env_state:    Dict[str, Any]       # Current env from CDDeployTracker
    last_ok_deploy: Dict[str, Any]     # Last successful deploy record

    # ── Post-deploy health monitor ───────────────────────────────────────────
    monitor_grade:          str        # "HEALTHY" | "DEGRADED" | "DOWN"
    monitor_availability:   float      # Availability % during monitor window
    monitor_avg_latency_ms: float
    monitor_regression:     bool       # True if regression detected
    monitor_issues:         List[str]  # Issues found during monitoring

    # ── Failure analysis (LLM) ───────────────────────────────────────────────
    deploy_failure_reason:  str        # LLM root cause of deploy failure
    deploy_suggested_fix:   str        # LLM suggested remediation

    # ── Rollback advisory ────────────────────────────────────────────────────
    rollback_available:  bool          # True if rollback target found
    rollback_command:    str           # Exact rollback shell command
    rollback_risk:       str           # "LOW" | "MEDIUM" | "HIGH"
    rollback_risk_reasons: List[str]
    rollback_comment_body: str         # Formatted PR comment for rollback

    # ── Actions taken ────────────────────────────────────────────────────────
    comment_posted:    bool            # True if report posted to PR
    indexed:           bool            # True if recorded in Redis
    notification_level: str            # "OK" | "WARN" | "BLOCKED"

# ── Chat Graph State ───────────────────────────────────────────────────────────

class ChatState(TypedDict, total=False):
    """
    State for the ChatGraph pipeline.

    Phase 1: Q&A / explain
      user_message → intent_router → load_memory → load_file_context
      → rag_retrieve → answer_question → memory_save → format_response

    Phase 2: Code generation
      intent_router → ... → [generate_completion | generate_class]
      → validate_generated → memory_save → format_response
    """

    # ── Input ────────────────────────────────────────────────────────────────
    session_id: str
    user_message: str
    project_path: str
    target_file: Optional[str]
    target_lang: str

    # ── Intent routing ──────────────────────────────────────────────────────
    intent: str          # "question" | "explain" | "complete_fn" | "new_class"
    intent_params: Dict[str, Any]

    # ── Loaded context ──────────────────────────────────────────────────────
    file_code: str
    file_analysis: Dict[str, Any]
    dependencies: List[str]
    dependents: List[str]
    rag_docs: List[Dict[str, Any]]
    rag_scores: List[float]
    project_summary: Dict[str, Any]

    # ── Phase 2 — Code generation ────────────────────────────────────────────
    generation_target: str           # function name or class name to generate
    generation_language: str         # "java" | "python" | "javascript" | "typescript"
    generated_code: str              # raw generated code from LLM
    generation_valid: bool           # passed syntax validation
    generation_errors: List[str]     # validation error messages
    project_patterns: Dict[str, Any] # detected naming conventions + existing classes
    apply_to_disk: bool              # True if user wants to write file directly

    # ── Conversation memory ─────────────────────────────────────────────────
    history: List[Dict[str, Any]]
    memory_key: str

    # ── Output ──────────────────────────────────────────────────────────────
    response: str
    formatted_response: str
    code_blocks: List[Dict[str, Any]]
    suggested_files: List[str]

    # ── Injected services ───────────────────────────────────────────────────
    _rag_system: Any
    _cache: Any
    _indexer: Any
    _dep_graph: Any

