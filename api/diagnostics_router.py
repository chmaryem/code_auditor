"""
api/diagnostics_router.py — Diagnostics API for VS Code plugin.

Endpoints:
  GET  /api/diagnostics              — diagnostics for one file (from cache)
  POST /api/diagnostics/batch        — diagnostics for multiple files at once
  GET  /api/diagnostics/patterns     — recurring patterns from PatternMemory
  POST /api/feedback                 — inline completion / analysis feedback

These endpoints are designed so the VS Code extension can:
  1. Load diagnostics for the active file WITHOUT triggering a full re-analysis
  2. Show inline squiggles + Problems panel items immediately from cache
  3. Collect accept/reject feedback to improve inline completions over time
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

diagnostics_router = APIRouter(prefix="/diagnostics", tags=["Diagnostics"])


# ── Pydantic models ──────────────────────────────────────────────────────────

class DiagnosticRange(BaseModel):
    start_line: int = 0
    start_char: int = 0
    end_line:   int = 0
    end_char:   int = 0


class DiagnosticItem(BaseModel):
    """VS Code–compatible diagnostic item."""
    message:   str   = ""
    severity:  int   = 1        # 0=Error, 1=Warning, 2=Info, 3=Hint
    code:      str   = "code-auditor"
    source:    str   = "Code Auditor"
    line:      Optional[int] = None
    range:     Optional[DiagnosticRange] = None
    suggestion: str  = ""
    rule:      str   = ""


class FileDiagnosticsResponse(BaseModel):
    file_path:   str  = ""
    language:    str  = "unknown"
    diagnostics: List[DiagnosticItem] = Field(default_factory=list)
    from_cache:  bool = False
    analyzed_at: float = 0.0
    elapsed_ms:  int  = 0


class BatchDiagnosticsRequest(BaseModel):
    files:        List[str] = Field(..., description="Absolute file paths")
    project_path: str       = Field(".", description="Project root")
    analyze_if_missing: bool = Field(
        False,
        description="If true, trigger lightweight AST analysis for uncached files"
    )


class BatchDiagnosticsResponse(BaseModel):
    results:      List[FileDiagnosticsResponse] = Field(default_factory=list)
    total_issues: int  = 0
    elapsed_ms:   int  = 0


class PatternItem(BaseModel):
    pattern:  str = ""
    language: str = ""
    count:    int = 0
    last_seen: float = 0.0


class FeedbackRequest(BaseModel):
    type:          str = Field(..., description="'accepted' | 'rejected' | 'modified'")
    file_path:     str = Field("",  description="File where the completion occurred")
    language:      str = Field("",  description="Programming language")
    completion_id: str = Field("",  description="Cache key of the completion")
    completion:    str = Field("",  description="The original completion text")
    modified_to:   str = Field("",  description="What the user actually typed (if modified)")
    elapsed_ms:    int = Field(0,   description="Time from suggestion to accept/reject")


# ── Helpers ──────────────────────────────────────────────────────────────────

_SEVERITY_MAP = {
    "critical": 0,
    "error":    0,
    "warning":  1,
    "warn":     1,
    "info":     2,
    "hint":     3,
    "low":      3,
}


def _to_vscode_severity(sev: str) -> int:
    return _SEVERITY_MAP.get(str(sev).lower(), 1)


def _get_cached_analysis(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve the latest analysis result for a file from the AnalysisCacheMemory.

    Priority:
      1. AnalysisCacheMemory.get_cached() — the primary LangGraph analysis cache
      2. Redis direct lookup with key pattern ca:analysis:{hash}
    """
    # 1. AnalysisCacheMemory (primary — used by node_llm_analyze)
    try:
        from langchain_agents.memory.redis_memory import AnalysisCacheMemory
        cache = AnalysisCacheMemory()
        cached = cache.get_cached(file_path)
        if cached:
            return cached
    except Exception as e:
        logger.debug("AnalysisCacheMemory lookup failed for %s: %s", file_path, e)

    # 2. Direct Redis scan as fallback
    try:
        import json
        from services.mcp_redis_service import get_mcp_redis, key_hash
        redis = get_mcp_redis()
        key   = f"ca:analysis:{key_hash(file_path)}"
        data  = redis.hgetall(key)
        if data and "result" in data:
            raw = data["result"]
            return json.loads(raw) if isinstance(raw, (str, bytes)) else None
    except Exception:
        pass

    return None


def _build_diagnostics_from_analysis(
    analysis: Dict[str, Any],
    file_path: str,
) -> List[DiagnosticItem]:
    """
    Convert cached LLM analysis to VS Code diagnostic items.
    Reuses the same parser as node_emit_ws_events in watch_graph.
    """
    try:
        from langchain_agents.graphs.watch_graph import _parse_issues_from_llm
        raw_text = analysis.get("analysis", "") if isinstance(analysis, dict) else str(analysis)
        raw_issues = _parse_issues_from_llm(raw_text, file_path)
    except Exception as e:
        logger.debug("_parse_issues_from_llm failed: %s", e)
        raw_issues = []

    # Also check if ws_events were stored with explicit issues
    ws_issues = analysis.get("issues", []) if isinstance(analysis, dict) else []

    all_issues = raw_issues + [i for i in ws_issues if i not in raw_issues]

    items = []
    for issue in all_issues[:30]:
        line = issue.get("line")
        sev  = issue.get("severity", "warning")
        items.append(DiagnosticItem(
            message    = issue.get("message", issue.get("title", "")),
            severity   = _to_vscode_severity(sev),
            code       = issue.get("rule", "code-auditor"),
            source     = "Code Auditor",
            line       = line,
            range      = DiagnosticRange(
                start_line = max(0, (line or 1) - 1),
                start_char = 0,
                end_line   = max(0, (line or 1) - 1),
                end_char   = 999,
            ) if line else None,
            suggestion = issue.get("suggestion", ""),
            rule       = issue.get("rule", ""),
        ))

    return items


def _lightweight_ast_check(file_path: str) -> List[DiagnosticItem]:
    """
    Quick AST-only checks (no LLM) for uncached files.
    Only catches syntax errors and obvious issues.
    """
    items = []
    try:
        import ast
        fp = Path(file_path)
        if fp.suffix != ".py":
            return []

        source = fp.read_text(encoding="utf-8", errors="replace")
        try:
            ast.parse(source)
        except SyntaxError as e:
            items.append(DiagnosticItem(
                message  = f"Syntax error: {e.msg}",
                severity = 0,  # Error
                code     = "syntax.error",
                source   = "Code Auditor",
                line     = e.lineno,
                range    = DiagnosticRange(
                    start_line = max(0, (e.lineno or 1) - 1),
                    start_char = e.offset or 0,
                    end_line   = max(0, (e.lineno or 1) - 1),
                    end_char   = 999,
                ) if e.lineno else None,
                suggestion = "Fix the syntax error at this location.",
            ))
    except Exception as e:
        logger.debug("AST check failed for %s: %s", file_path, e)

    return items


# ── Endpoints ────────────────────────────────────────────────────────────────

@diagnostics_router.get(
    "",
    response_model=FileDiagnosticsResponse,
    summary="Get diagnostics for a file (from cache)",
)
async def get_diagnostics(
    file: str  = Query(..., description="Absolute file path"),
    project: str = Query(".", description="Project root"),
    analyze_if_missing: bool = Query(False, description="Trigger AST check if not in cache"),
):
    """
    Return cached diagnostics for a file without triggering a full LLM re-analysis.

    VS Code extension uses this to populate the Problems panel and inline squiggles
    when it opens a file or needs a refresh.

    Priority:
      1. LangGraph analysis cache → structured issues
      2. Redis cache (latest watch event)
      3. Lightweight AST check (syntax only, if analyze_if_missing=True)
      4. Empty list (no error)
    """
    t0 = time.time()
    file_path = file

    if not Path(file_path).exists():
        raise HTTPException(404, f"File not found: {file_path}")

    from langchain_agents.agents.lc_code_agent import lc_code_agent
    language = lc_code_agent.detect_language(Path(file_path))

    # 1. Try cache
    cached = _get_cached_analysis(file_path)
    if cached:
        diagnostics = _build_diagnostics_from_analysis(cached, file_path)
        return FileDiagnosticsResponse(
            file_path    = file_path,
            language     = language,
            diagnostics  = diagnostics,
            from_cache   = True,
            analyzed_at  = cached.get("timestamp", 0.0) if isinstance(cached, dict) else 0.0,
            elapsed_ms   = round((time.time() - t0) * 1000),
        )

    # 2. AST-only check if requested
    if analyze_if_missing:
        diagnostics = _lightweight_ast_check(file_path)
        return FileDiagnosticsResponse(
            file_path   = file_path,
            language    = language,
            diagnostics = diagnostics,
            from_cache  = False,
            elapsed_ms  = round((time.time() - t0) * 1000),
        )

    # 3. Empty (not yet analyzed)
    return FileDiagnosticsResponse(
        file_path  = file_path,
        language   = language,
        diagnostics = [],
        from_cache = False,
        elapsed_ms = round((time.time() - t0) * 1000),
    )


@diagnostics_router.post(
    "/batch",
    response_model=BatchDiagnosticsResponse,
    summary="Get diagnostics for multiple files at once",
)
async def get_diagnostics_batch(req: BatchDiagnosticsRequest):
    """
    Batch diagnostics query — load issues for an entire list of files.

    Called at plugin startup to pre-populate the Problems panel for all
    recently modified files.
    """
    import asyncio
    t0 = time.time()
    results = []

    async def _check_one(fp: str) -> FileDiagnosticsResponse:
        file_path = fp
        try:
            if not Path(file_path).exists():
                return FileDiagnosticsResponse(file_path=fp, diagnostics=[])

            from langchain_agents.agents.lc_code_agent import lc_code_agent
            language = lc_code_agent.detect_language(Path(file_path))

            cached = _get_cached_analysis(file_path)
            if cached:
                diags = _build_diagnostics_from_analysis(cached, file_path)
                return FileDiagnosticsResponse(
                    file_path   = fp,
                    language    = language,
                    diagnostics = diags,
                    from_cache  = True,
                )

            if req.analyze_if_missing:
                diags = _lightweight_ast_check(file_path)
                return FileDiagnosticsResponse(
                    file_path   = fp,
                    language    = language,
                    diagnostics = diags,
                )

            return FileDiagnosticsResponse(file_path=fp, language=language)
        except Exception as e:
            logger.debug("batch diagnostics error for %s: %s", fp, e)
            return FileDiagnosticsResponse(file_path=fp)

    tasks = [_check_one(fp) for fp in req.files[:50]]  # cap at 50 files
    results = await asyncio.gather(*tasks)

    total_issues = sum(len(r.diagnostics) for r in results)

    return BatchDiagnosticsResponse(
        results      = list(results),
        total_issues = total_issues,
        elapsed_ms   = round((time.time() - t0) * 1000),
    )


@diagnostics_router.get(
    "/patterns",
    summary="Get recurring code patterns from Pattern Memory",
)
async def get_patterns(
    language: str = Query("", description="Filter by language (empty = all)"),
    min_count: int = Query(2, description="Minimum occurrence count"),
    limit: int = Query(20, description="Max patterns to return"),
):
    """
    Return the top recurring patterns detected by the LearningAgent.
    Displayed in the VS Code 'Patterns' panel to help developers
    understand the most common issues in their project.
    """
    try:
        from langchain_agents.memory.redis_memory import PatternMemory
        pm = PatternMemory()

        if language:
            # Single language query
            patterns = pm.get_top_patterns(language, n=limit)
        else:
            # Aggregate across known languages
            _LANGS = ["python", "typescript", "javascript", "java", "go", "rust", "csharp"]
            seen: dict = {}
            for lang in _LANGS:
                for p in pm.get_top_patterns(lang, n=limit):
                    key = p["pattern"]
                    if key not in seen:
                        seen[key] = {"pattern": key, "language": lang, "count": p["count"]}
                    else:
                        seen[key]["count"] += p["count"]
            patterns = sorted(seen.values(), key=lambda x: x["count"], reverse=True)[:limit]

        filtered = [p for p in patterns if p.get("count", 0) >= min_count]
        return {
            "patterns": filtered,
            "total":    len(filtered),
            "language": language or "all",
        }
    except Exception as e:
        logger.warning("get_patterns error: %s", e)
        return {"patterns": [], "total": 0, "language": language or "all"}


# ── Feedback router ───────────────────────────────────────────────────────────

feedback_router = APIRouter(prefix="/feedback", tags=["Feedback"])


@feedback_router.post("", summary="Submit accept/reject feedback for completions")
async def submit_feedback(req: FeedbackRequest):
    """
    Collect developer feedback on inline completions and analysis suggestions.

    This drives the self-improvement loop:
      - accepted  → reinforce this completion pattern in PatternMemory
      - rejected  → mark pattern as negative, reduce confidence
      - modified  → store the corrected version for future training
    """
    t0 = time.time()

    try:
        from langchain_agents.memory.redis_memory import PatternMemory
        pm = PatternMemory()
        lang = req.language or "unknown"

        if req.type == "accepted" and req.completion:
            pm.record_pattern(
                language     = lang,
                pattern_name = f"accepted:{req.completion[:150]}",
            )

        elif req.type == "rejected" and req.completion:
            pm.record_pattern(
                language     = lang,
                pattern_name = f"rejected:{req.completion[:150]}",
            )

        elif req.type == "modified" and req.modified_to:
            # Store the corrected version as a positive pattern
            pm.record_pattern(
                language     = lang,
                pattern_name = f"accepted:{req.modified_to[:150]}",
            )

    except Exception as e:
        logger.debug("feedback storage error: %s", e)

    return {
        "status":     "recorded",
        "type":       req.type,
        "elapsed_ms": round((time.time() - t0) * 1000),
    }
