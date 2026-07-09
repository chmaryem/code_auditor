"""
api/mcp_router.py — MCP (Model Context Protocol) Router.

Exposes the backend multi-agent capabilities as MCP-compatible tools
so external MCP clients (Cursor, Claude Desktop, other IDE extensions)
can call them directly.

Endpoints:
  GET  /mcp/tools/list    — list all available tools + schemas
  POST /mcp/tools/call    — call a tool by name with arguments
  GET  /mcp/resources     — list available resources (project files, cache)

Tool catalog:
  analyze_file       — Full WatchGraph analysis of a file
  get_diagnostics    — Get cached diagnostics for a file
  generate_tests     — Generate unit tests for a file
  inline_complete    — Cursor-aware inline code completion
  explain_code       — Explain selected code (ChatGraph)
  get_git_status     — Smart Git session status
  get_ci_status      — Latest CI pipeline status
  search_codebase    — Semantic code search (RAG)
  get_patterns       — Recurring patterns from Pattern Memory
  apply_fix          — Apply a generated fix to a file

MCP Protocol Reference: https://modelcontextprotocol.io/
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

mcp_router = APIRouter(prefix="/mcp", tags=["MCP"])


# ── Tool Registry ─────────────────────────────────────────────────────────────

MCP_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "analyze_file",
        "description": "Run a full multi-agent analysis on a source file. Detects bugs, security issues, test gaps, and dependency impacts. Uses LLM + RAG + cache.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path":    {"type": "string", "description": "Absolute path to the file"},
                "project_path": {"type": "string", "description": "Project root (optional)", "default": "."},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "get_diagnostics",
        "description": "Retrieve cached diagnostics (issues/warnings) for a file without triggering a new analysis. Fast, < 50ms.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path":    {"type": "string", "description": "Absolute path to the file"},
                "project_path": {"type": "string", "description": "Project root", "default": "."},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "generate_tests",
        "description": "Generate unit tests for a source file using RAG-powered TestGeneratorAgent. Supports pytest, jest, mocha, junit.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path":    {"type": "string", "description": "Absolute path to the file"},
                "project_path": {"type": "string", "description": "Project root", "default": "."},
                "write":        {"type": "boolean", "description": "Write tests to disk", "default": False},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "inline_complete",
        "description": "Cursor-aware inline code completion (Copilot-style). Returns 1-5 lines to insert at the cursor. Cached in Redis (TTL 5min).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prefix_code":  {"type": "string", "description": "Code before the cursor"},
                "suffix_code":  {"type": "string", "description": "Code after the cursor", "default": ""},
                "language":     {"type": "string", "description": "Programming language", "default": "python"},
                "file_path":    {"type": "string", "description": "Current file path", "default": ""},
                "project_path": {"type": "string", "description": "Project root", "default": "."},
            },
            "required": ["prefix_code"],
        },
    },
    {
        "name": "explain_code",
        "description": "Explain what a piece of code does in plain language. Uses the ChatGraph with RAG context from the project.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code":         {"type": "string", "description": "Code to explain"},
                "file_path":    {"type": "string", "description": "File containing the code", "default": ""},
                "project_path": {"type": "string", "description": "Project root", "default": "."},
                "language":     {"type": "string", "description": "Programming language", "default": ""},
            },
            "required": ["code"],
        },
    },
    {
        "name": "get_git_status",
        "description": "Get the Smart Git session status: risk score, uncommitted changes, files at risk, time since last commit.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_path": {"type": "string", "description": "Project root"},
            },
            "required": ["project_path"],
        },
    },
    {
        "name": "get_ci_status",
        "description": "Get the latest CI pipeline status and analysis for a repository.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo":       {"type": "string", "description": "GitHub repo (owner/repo)"},
                "run_id":     {"type": "string", "description": "Specific run ID (optional)", "default": ""},
            },
            "required": ["repo"],
        },
    },
    {
        "name": "search_codebase",
        "description": "Semantic search across the project codebase using RAG. Returns the most relevant code snippets for a query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query":        {"type": "string", "description": "Search query"},
                "project_path": {"type": "string", "description": "Project root"},
                "language":     {"type": "string", "description": "Filter by language", "default": ""},
                "top_k":        {"type": "integer", "description": "Number of results", "default": 5},
            },
            "required": ["query", "project_path"],
        },
    },
    {
        "name": "get_patterns",
        "description": "Get the top recurring code patterns detected by the LearningAgent across the project.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "language":   {"type": "string", "description": "Filter by language", "default": ""},
                "min_count":  {"type": "integer", "description": "Minimum occurrence count", "default": 2},
            },
        },
    },
    {
        "name": "apply_fix",
        "description": "Apply a generated code fix to a function in a file. Returns a diff + VS Code WorkspaceEdit.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path":     {"type": "string", "description": "Target file"},
                "project_path":  {"type": "string", "description": "Project root"},
                "function_name": {"type": "string", "description": "Function to replace"},
                "new_code":      {"type": "string", "description": "New implementation"},
                "language":      {"type": "string", "description": "Language", "default": "python"},
                "write_mode":    {"type": "string", "description": "'dry_run'|'apply'", "default": "dry_run"},
            },
            "required": ["file_path", "project_path", "function_name", "new_code"],
        },
    },
]


# ── MCP Models ────────────────────────────────────────────────────────────────

class MCPCallRequest(BaseModel):
    name:      str            = Field(..., description="Tool name")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")


class MCPCallResponse(BaseModel):
    tool:       str           = ""
    result:     Any           = None
    is_error:   bool          = False
    error:      str           = ""
    elapsed_ms: int           = 0


# ── Tool Implementations ─────────────────────────────────────────────────────

async def _call_analyze_file(args: Dict[str, Any]) -> Any:
    file_path    = args.get("file_path", "")
    project_path = args.get("project_path", ".")

    if not Path(file_path).exists():
        raise ValueError(f"File not found: {file_path}")

    from langchain_agents.graphs.watch_graph import invoke_watch
    result = await asyncio.to_thread(
        invoke_watch,
        file_path    = file_path,
        project_path = project_path,
    )

    analysis = result.get("analysis", {})
    raw_text  = analysis.get("analysis", "") if isinstance(analysis, dict) else str(analysis)
    issues    = result.get("issues", [])

    return {
        "file":      file_path,
        "language":  result.get("language", "unknown"),
        "strategy":  result.get("strategy", "block_fix"),
        "issues":    issues,
        "summary":   raw_text[:1500],
        "skipped":   bool(result.get("skip_reason")),
        "skip_reason": result.get("skip_reason"),
    }


async def _call_get_diagnostics(args: Dict[str, Any]) -> Any:
    file_path = args.get("file_path", "")

    if not Path(file_path).exists():
        raise ValueError(f"File not found: {file_path}")

    # Reuse the diagnostics router logic
    from api.diagnostics_router import _get_cached_analysis, _build_diagnostics_from_analysis
    cached = _get_cached_analysis(file_path)

    if cached:
        from langchain_agents.agents.lc_code_agent import lc_code_agent
        language = lc_code_agent.detect_language(Path(file_path))
        diags = _build_diagnostics_from_analysis(cached, file_path)
        return {
            "file":        file_path,
            "language":    language,
            "diagnostics": [d.model_dump() for d in diags],
            "from_cache":  True,
        }

    return {"file": file_path, "diagnostics": [], "from_cache": False}


async def _call_generate_tests(args: Dict[str, Any]) -> Any:
    file_path    = args.get("file_path", "")
    project_path = args.get("project_path", ".")
    write        = args.get("write", False)

    from agents.test_generator_agent import TestGeneratorAgent
    agent = TestGeneratorAgent(project_path=Path(project_path))
    result = await asyncio.to_thread(agent.generate_for_file, Path(file_path), write)
    return result


async def _call_inline_complete(args: Dict[str, Any]) -> Any:
    from langchain_agents.agents.lc_inline_completion_agent import lc_inline_completion_agent
    result = await asyncio.to_thread(
        lc_inline_completion_agent.complete,
        prefix_code  = args.get("prefix_code", ""),
        suffix_code  = args.get("suffix_code", ""),
        language     = args.get("language", "python"),
        file_path    = args.get("file_path", ""),
        project_path = args.get("project_path", "."),
        use_rag      = True,
    )
    return result


async def _call_explain_code(args: Dict[str, Any]) -> Any:
    from langchain_agents.graphs.chat_graph import ainvoke_chat
    code         = args.get("code", "")
    file_path    = args.get("file_path", "")
    project_path = args.get("project_path", ".")
    language     = args.get("language", "")

    lang_part = f" ({language})" if language else ""
    message   = f"Explain this code{lang_part}:\n\n```\n{code[:2000]}\n```"

    result = await ainvoke_chat(
        message      = message,
        project_path = project_path,
        session_id   = "mcp_explain",
        target_file  = file_path,
    )
    return {
        "explanation": result.get("formatted_response") or result.get("response", ""),
        "intent":      result.get("intent", "explain"),
    }


async def _call_get_git_status(args: Dict[str, Any]) -> Any:
    project_path = args.get("project_path", ".")

    from langchain_agents.tools.git_tools import tool_session_status
    return tool_session_status.invoke({"project_path": project_path})


async def _call_get_ci_status(args: Dict[str, Any]) -> Any:
    repo   = args.get("repo", "")
    run_id = args.get("run_id", "")

    from langchain_agents.tools.ci_tools import tool_fetch_workflow_runs
    runs = await asyncio.to_thread(tool_fetch_workflow_runs.invoke, {"repo": repo, "limit": 1})
    return {"repo": repo, "runs": runs}


async def _call_search_codebase(args: Dict[str, Any]) -> Any:
    query        = args.get("query", "")
    project_path = args.get("project_path", ".")
    language     = args.get("language", "")
    top_k        = args.get("top_k", 5)

    from langchain_agents.tools.chat_tools import tool_chat_rag_retrieve
    result = await asyncio.to_thread(tool_chat_rag_retrieve.invoke, {
        "project_path": project_path,
        "query":        query,
        "target_file":  "",
        "file_code":    "",
        "language":     language,
    })
    docs = result.get("rag_docs", [])[:top_k]
    return {
        "query":   query,
        "results": [{"content": d.get("content", "")[:500], "metadata": d.get("metadata", {})} for d in docs],
        "total":   len(docs),
    }


async def _call_get_patterns(args: Dict[str, Any]) -> Any:
    language  = args.get("language", "")
    min_count = args.get("min_count", 2)

    from langchain_agents.memory.redis_memory import PatternMemory
    pm       = PatternMemory()
    patterns = pm.get_top_patterns(language or None, n=20)
    filtered = [p for p in patterns if p.get("count", 0) >= min_count]
    return {"patterns": filtered, "total": len(filtered)}


async def _call_apply_fix(args: Dict[str, Any]) -> Any:
    from langchain_agents.agents.lc_apply_agent import lc_apply_agent
    result = await asyncio.to_thread(
        lc_apply_agent.apply_function_patch,
        project_path  = args.get("project_path", "."),
        file_path     = args.get("file_path", ""),
        function_name = args.get("function_name", ""),
        new_code      = args.get("new_code", ""),
        language      = args.get("language", "python"),
        write_mode    = args.get("write_mode", "dry_run"),
    )
    return result


# ── Tool dispatch ─────────────────────────────────────────────────────────────

_TOOL_HANDLERS = {
    "analyze_file":    _call_analyze_file,
    "get_diagnostics": _call_get_diagnostics,
    "generate_tests":  _call_generate_tests,
    "inline_complete": _call_inline_complete,
    "explain_code":    _call_explain_code,
    "get_git_status":  _call_get_git_status,
    "get_ci_status":   _call_get_ci_status,
    "search_codebase": _call_search_codebase,
    "get_patterns":    _call_get_patterns,
    "apply_fix":       _call_apply_fix,
}


# ── Endpoints ────────────────────────────────────────────────────────────────

@mcp_router.get("/tools/list", summary="List all available MCP tools")
async def mcp_tools_list():
    """
    Return the catalog of available tools in MCP format.
    Compatible with MCP clients: Cursor, Claude Desktop, Zed, etc.
    """
    return {
        "tools": MCP_TOOLS,
        "total": len(MCP_TOOLS),
        "server": "Code Auditor MCP v1.0",
    }


@mcp_router.post(
    "/tools/call",
    response_model=MCPCallResponse,
    summary="Call an MCP tool by name",
)
async def mcp_tools_call(req: MCPCallRequest):
    """
    Execute an MCP tool call.

    This is the single entry point for all MCP tool invocations.
    External clients (Cursor, Claude Desktop) send:
      { "name": "analyze_file", "arguments": { "file_path": "..." } }

    Returns a structured result or error.
    """
    t0 = time.time()

    handler = _TOOL_HANDLERS.get(req.name)
    if handler is None:
        raise HTTPException(
            400,
            f"Unknown tool: '{req.name}'. Available: {list(_TOOL_HANDLERS.keys())}"
        )

    try:
        result = await handler(req.arguments)
        return MCPCallResponse(
            tool       = req.name,
            result     = result,
            is_error   = False,
            elapsed_ms = round((time.time() - t0) * 1000),
        )
    except Exception as e:
        logger.exception("MCP tool '%s' error: %s", req.name, e)
        return MCPCallResponse(
            tool       = req.name,
            result     = None,
            is_error   = True,
            error      = f"{type(e).__name__}: {e}",
            elapsed_ms = round((time.time() - t0) * 1000),
        )


@mcp_router.get("/resources", summary="List available project resources")
async def mcp_resources(project_path: str = "."):
    """
    Return a list of project resources accessible via MCP.
    Includes source files, test files, and cached analysis results.
    """
    try:
        root = Path(project_path).resolve()
        if not root.exists():
            return {"resources": [], "total": 0}

        source_files = []
        for ext in ("*.py", "*.ts", "*.js", "*.java", "*.go", "*.rs", "*.cs"):
            source_files.extend(root.rglob(ext))

        # Filter out venv, node_modules, .git
        _SKIP = {".venv", "venv", "node_modules", ".git", "__pycache__", "dist", "build"}
        filtered = [
            f for f in source_files
            if not any(part in _SKIP for part in f.parts)
        ][:200]

        resources = [
            {
                "uri":      f"file://{f}",
                "name":     str(f.relative_to(root)),
                "mimeType": "text/x-" + f.suffix.lstrip("."),
            }
            for f in filtered
        ]

        return {
            "resources": resources,
            "total":     len(resources),
            "project":   str(root),
        }
    except Exception as e:
        logger.warning("mcp_resources error: %s", e)
        return {"resources": [], "total": 0, "error": str(e)}
