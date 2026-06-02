"""
api/models.py — Modèles Pydantic pour l'API FastAPI.

Tous les types request/response sont centralisés ici.
Le serveur (server.py) et l'orchestrateur importent ces types.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "8.0.0"
    services: Dict[str, object] = Field(
        default_factory=dict,
        description="État des services: redis, chromadb, llm + uptime",
    )


# ── Analyze File ──────────────────────────────────────────────────────────────

class AnalyzeFileRequest(BaseModel):
    file_path: str = Field(..., description="Chemin absolu du fichier à analyser")
    project_path: str = Field(
        ".", description="Chemin racine du projet (défaut: dossier courant)"
    )


class IssueDiagnostic(BaseModel):
    """Un problème détecté — format compatible LSP Diagnostic."""
    severity: str = Field(..., description="CRITICAL | HIGH | MEDIUM | LOW")
    message: str = Field(..., description="Description du problème")
    line: Optional[int] = Field(None, description="Numéro de ligne (1-indexed)")
    column: Optional[int] = Field(None, description="Numéro de colonne")
    source: str = Field("code-auditor", description="Source du diagnostic")
    code_snippet: str = Field("", description="Extrait du code problématique")
    suggestion: Optional[str] = Field(None, description="Explication / conseil")


class FixSuggestion(BaseModel):
    """Correction proposée par le LLM."""
    location: str = Field("", description="Emplacement (méthode, ligne)")
    current_code: str = Field("", description="Code actuel (avant fix)")
    fixed_code: str = Field("", description="Code corrigé (après fix)")
    explanation: str = Field("", description="Pourquoi cette correction")


class AnalysisResultResponse(BaseModel):
    """Résultat complet d'une analyse de fichier."""
    file_path: str
    language: str = "unknown"
    score: int = Field(0, description="Score d'importance du changement (0-100)")
    strategy: str = Field(
        "block_fix",
        description="Stratégie choisie: full_class | targeted_methods | block_fix",
    )
    issues: List[IssueDiagnostic] = Field(default_factory=list)
    fixes: List[FixSuggestion] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    rag_docs_used: int = Field(0, description="Nombre de documents RAG consultés")
    raw_analysis: str = Field("", description="Texte brut de l'analyse LLM")


# ── Analyze Project ───────────────────────────────────────────────────────────

class AnalyzeProjectRequest(BaseModel):
    project_path: str = Field(..., description="Chemin racine du projet")
    max_files: int = Field(10, description="Nombre max de fichiers à analyser")


class ProjectAnalysisResponse(BaseModel):
    project_path: str
    files_analyzed: int = 0
    entry_points: List[str] = Field(default_factory=list)
    circular_dependencies: List[List[str]] = Field(default_factory=list)
    orphaned_modules: List[str] = Field(default_factory=list)
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    refactoring_plan: str = ""
    file_results: Dict[str, AnalysisResultResponse] = Field(default_factory=dict)


# ── Watch Mode ────────────────────────────────────────────────────────────────

class WatchStartRequest(BaseModel):
    project_path: str = Field(..., description="Chemin du projet à surveiller")


class WatchStatusResponse(BaseModel):
    is_running: bool = False
    project_path: str = ""
    files_processed: int = 0
    stats: Dict[str, Any] = Field(default_factory=dict)


# ── Git ───────────────────────────────────────────────────────────────────────

class GitStatusRequest(BaseModel):
    project_path: str = Field(..., description="Chemin du projet Git")


class GitBranchRequest(BaseModel):
    project_path: str
    branch: str = Field("HEAD", description="Branche à analyser")
    base: str = Field("main", description="Branche de base pour comparaison")


# ── Generate Tests ────────────────────────────────────────────────────────────

class GenerateTestsRequest(BaseModel):
    file_path: str = Field(..., description="Fichier source à tester")
    project_path: str = Field("", description="Racine du projet (optionnel)")
    write: bool = Field(False, description="Écrire le fichier de test sur disque")


class GenerateTestsResponse(BaseModel):
    test_file: str = ""
    test_code: str = ""
    framework: str = ""
    rag_docs_used: int = 0
    validated: bool = False
    error: str = ""


# ── WebSocket Events ─────────────────────────────────────────────────────────
# These are the events broadcast via /ws to the VS Code plugin.
# The plugin receives a JSON object with a "type" discriminator field.

class WSEvent(BaseModel):
    """Generic WebSocket message envelope (used when type is unknown)."""
    type: str = Field(
        ...,
        description="Event type: analysis_result | dependency_impact | test_gap | "
                    "git_recommendation | known_issue | watch_event | file_deleted | error",
    )
    data: Dict[str, Any] = Field(default_factory=dict)


# ── Typed events (informational — actual WS payloads are plain dicts) ─────────

class WSIssue(BaseModel):
    """A single code issue detected by the LLM — maps to vscode.Diagnostic."""
    title:      str           = ""
    message:    str           = ""
    line:       Optional[int] = None
    severity:   str           = "warning"    # critical | error | warning | info
    rule:       str           = "code-auditor"
    suggestion: str           = ""


class WSAnalysisResultEvent(BaseModel):
    """
    type: analysis_result
    Sent when a file is analysed. Plugin uses this to populate the Problems tab.
    """
    type:      str           = "analysis_result"
    file_path: str           = ""
    language:  str           = "unknown"
    severity:  str           = "info"        # worst-case across all issues
    strategy:  str           = "block_fix"
    issues:    List[WSIssue] = Field(default_factory=list)


class WSDependencyImpactEvent(BaseModel):
    """
    type: dependency_impact
    Sent when the changed file has dependents that may be affected.
    """
    type:           str       = "dependency_impact"
    file_path:      str       = ""
    impacted_files: List[str] = Field(default_factory=list)
    risk:           str       = "low"        # low | medium | high
    reason:         str       = ""


class WSTestGapEvent(BaseModel):
    """
    type: test_gap
    Sent when the changed source file has missing or outdated tests.
    """
    type:               str       = "test_gap"
    file_path:          str       = ""
    severity:           str       = "warning"
    missing_tests:      bool      = True
    related_test_files: List[str] = Field(default_factory=list)
    recommendation:     str       = ""


class WSGitRecommendationEvent(BaseModel):
    """
    type: git_recommendation
    Sent after each analysis to advise whether a commit is safe.
    """
    type:                     str       = "git_recommendation"
    status:                   str       = "not_recommended"   # recommended | not_recommended
    should_commit:            bool      = False
    message:                  str       = ""
    modified_files:           int       = 0
    staged_files:             int       = 0
    risk:                     str       = "medium"            # low | medium | high
    blocking_reasons:         List[str] = Field(default_factory=list)
    suggested_commit_message: str       = ""


class WSKnownIssueEvent(BaseModel):
    """
    type: known_issue
    Sent when a Redis pattern matches the current issue (seen before).
    """
    type:          str   = "known_issue"
    file_path:     str   = ""
    issue_title:   str   = ""
    similarity:    float = 0.0
    previous_fix:  str   = ""
    seen_count:    int   = 0


# ── WS Event Schema v2.0 ──────────────────────────────────────────────────────
# These models define the plugin-ready contract for analysis_result events.
# They replace the v1 WSAnalysisResultEvent + WSIssue + WSFix models.
# The server broadcasts WSAnalysisResultEventV2 (or a plain dict equivalent).

class WSDiffHunk(BaseModel):
    """
    A targeted diff hunk — tells the plugin exactly which lines changed.

    The plugin uses this to show a precise diff in a vscode.diff() panel
    and to apply the fix with ``WorkspaceEdit.replace()`` at the correct range.
    """
    start_line:     int        = Field(..., description="1-indexed start line in original file")
    end_line:       int        = Field(..., description="1-indexed end line in original file (inclusive)")
    original_lines: List[str]  = Field(default_factory=list, description="Lines removed / replaced")
    new_lines:      List[str]  = Field(default_factory=list, description="Lines inserted / replacing")
    context:        str        = Field("", description="Human-readable description of the hunk")


class WSIssueV2(BaseModel):
    """
    A single code issue — v2.0, fully compatible with vscode.Diagnostic.

    Key additions vs WSIssue v1:
      - id            : UUID for cross-referencing with WSFixV2.issue_id
      - column        : precise squiggle positioning
      - end_line/end_column : full diagnostic range
      - fix_available : quick boolean for CodeLens rendering
      - fix_id        : direct reference to the associated fix
    """
    id:          str           = Field(default="", description="UUID — unique per issue")
    title:       str           = ""
    message:     str           = ""
    line:        Optional[int] = None
    column:      Optional[int] = None
    end_line:    Optional[int] = None
    end_column:  Optional[int] = None
    severity:    str           = "warning"     # critical | error | warning | info
    rule:        str           = "code-auditor"
    suggestion:  str           = ""
    fix_available: bool        = False
    fix_id:      Optional[str] = None          # references WSFixV2.id


class WSFixV2(BaseModel):
    """
    A suggested fix — v2.0 with diff_hunks and unique IDs.

    Key additions vs FixSuggestion v1:
      - id          : UUID for cross-referencing with WSIssueV2.fix_id
      - issue_id    : back-reference to the related issue
      - diff_hunks  : targeted line-level diff (replaces raw code blobs for display)
      - apply_mode  : replace_snippet | replace_method | full_file
      - current_code / fixed_code : kept for backward-compat but truncated for full_file
    """
    id:           str              = Field(default="", description="UUID — unique per fix")
    issue_id:     Optional[str]    = None
    title:        str              = ""
    apply_mode:   str              = "replace_snippet"   # replace_snippet | replace_method | full_file
    file_path:    str              = ""
    line:         Optional[int]    = None
    diff_hunks:   List[WSDiffHunk] = Field(default_factory=list)
    # Kept for backward-compat; truncated to 300 chars for full_file strategy
    current_code: str              = ""
    fixed_code:   str              = ""
    explanation:  str              = ""
    language:     str              = ""


class WSAnalysisResultEventV2(BaseModel):
    """
    type: analysis_result  (schema_version: 2.0)

    Plugin-ready analysis result event.

    New fields vs v1:
      - schema_version  : "2.0" — allows plugin to detect and handle upgrades
      - request_id      : UUID correlating the file-change trigger to this event
      - file_name       : pre-parsed basename (avoids parsing in the plugin)
      - strategy_reason : short human explanation of the strategy choice
      - change_score    : 0-100 significance score from the change filter
      - elapsed_ms      : end-to-end pipeline latency in milliseconds
      - analyzed_at     : ISO 8601 UTC timestamp
      - rag_docs_used   : number of RAG documents injected into the LLM prompt
      - fixes           : List[WSFixV2] with diff_hunks
    """
    type:            str            = "analysis_result"
    schema_version:  str            = "2.0"
    request_id:      str            = Field(default="", description="UUID for event correlation")
    file_path:       str            = ""
    file_name:       str            = Field(default="", description="Basename of file_path")
    language:        str            = "unknown"
    severity:        str            = "info"         # worst-case severity across all issues
    strategy:        str            = "block_fix"
    strategy_reason: str            = ""
    change_score:    int            = Field(0, ge=0, le=100)
    issues:          List[WSIssueV2] = Field(default_factory=list)
    fixes:           List[WSFixV2]   = Field(default_factory=list)
    elapsed_ms:      int            = 0
    analyzed_at:     str            = ""
    rag_docs_used:   int            = 0
