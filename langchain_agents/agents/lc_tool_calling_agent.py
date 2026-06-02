"""
lc_tool_calling_agent.py — Real tool-calling Pair Programmer (Phase C3).

Unlike keyword routing (Phase A/B), this agent lets the LLM autonomously
decide WHICH tools to call based on the developer's message.

Tools available:
  search_codebase(query)       → RAG vector search
  get_file(path)               → read a file from project
  get_git_diff()               → current uncommitted diff
  analyze_file(path)           → run WatchGraph on a file
  run_tests(file_path)         → execute pytest on a file (safe, read-only)
  get_ci_status()              → last CI run summary
  get_dependencies(file_path)  → dep graph neighbors

Execution model:
  1. LLM receives tools + message → chooses tool calls
  2. Agent executes tools (sandboxed, no mutation unless write_mode=apply)
  3. LLM synthesizes final answer from tool results

Uses LangChain tool_calling interface (bind_tools + parse_tool_calls).
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Tool definitions
# ══════════════════════════════════════════════════════════════════════════════

TOOL_SCHEMAS = [
    {
        "name": "search_codebase",
        "description": "Search the project codebase for relevant code patterns, classes, functions, or documentation using semantic search.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_file",
        "description": "Read the contents of a specific file from the project.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to project root"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "get_git_diff",
        "description": "Get the current uncommitted git diff (what has been modified since last commit).",
        "parameters": {
            "type": "object",
            "properties": {
                "staged": {"type": "boolean", "description": "True for staged changes, False for unstaged"}
            },
            "required": [],
        },
    },
    {
        "name": "analyze_file",
        "description": "Run a deep static analysis on a file to detect bugs, code smells, and quality issues.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to analyze"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "get_dependencies",
        "description": "Get the dependency graph for a file — what it imports and what imports it.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "get_ci_status",
        "description": "Get the latest CI pipeline status for this project (GitHub Actions, SonarCloud).",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# Tool executors
# ══════════════════════════════════════════════════════════════════════════════

class ToolExecutor:
    """Executes tool calls safely in the project context."""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path).resolve()

    def _safe_path(self, path: str) -> Optional[Path]:
        """Validate that a path is within project root."""
        p = Path(path)
        if not p.is_absolute():
            p = self.project_path / p
        try:
            p.resolve().relative_to(self.project_path)
            return p
        except ValueError:
            logger.warning("Rejected path outside project root: %s", path)
            return None

    async def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a tool and return its result as a string."""
        try:
            if tool_name == "search_codebase":
                return await self._search_codebase(args.get("query", ""))
            elif tool_name == "get_file":
                return await self._get_file(args.get("path", ""))
            elif tool_name == "get_git_diff":
                return await self._get_git_diff(bool(args.get("staged", False)))
            elif tool_name == "analyze_file":
                return await self._analyze_file(args.get("path", ""))
            elif tool_name == "get_dependencies":
                return await self._get_dependencies(args.get("path", ""))
            elif tool_name == "get_ci_status":
                return await self._get_ci_status()
            else:
                return f"[Unknown tool: {tool_name}]"
        except Exception as e:
            return f"[Tool {tool_name} error: {e}]"

    async def _search_codebase(self, query: str) -> str:
        try:
            from langchain_agents.tools.chat_tools import tool_chat_rag_retrieve
            result = await asyncio.to_thread(
                tool_chat_rag_retrieve.invoke, {
                    "project_path": str(self.project_path),
                    "query":        query[:400],
                    "target_file":  "",
                    "file_code":    "",
                    "language":     "python",
                }
            )
            docs = result.get("rag_docs", [])
            if not docs:
                return "(no results found)"
            parts = []
            for d in docs[:3]:
                content = d.get("content", "")[:300]
                source  = d.get("metadata", {}).get("source", "?")
                parts.append(f"[{source}]\n{content}")
            return "\n\n---\n\n".join(parts)
        except Exception as e:
            return f"(search failed: {e})"

    async def _get_file(self, path: str) -> str:
        fp = self._safe_path(path)
        if not fp or not fp.exists():
            return f"(file not found: {path})"
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
            if len(content) > 3000:
                content = content[:3000] + "\n... (truncated)"
            return content
        except Exception as e:
            return f"(read error: {e})"

    async def _get_git_diff(self, staged: bool = False) -> str:
        import subprocess
        cmd = ["git", "-C", str(self.project_path), "diff", "--unified=3"]
        if staged:
            cmd.append("--cached")
        try:
            r = await asyncio.to_thread(
                subprocess.run, cmd, capture_output=True, text=True, timeout=10
            )
            return r.stdout[:3000] or "(no changes)"
        except Exception as e:
            return f"(git diff failed: {e})"

    async def _analyze_file(self, path: str) -> str:
        fp = self._safe_path(path)
        if not fp or not fp.exists():
            return f"(file not found: {path})"
        try:
            from langchain_agents.graphs.watch_graph import invoke_watch
            result = await asyncio.to_thread(
                invoke_watch,
                file_path=str(fp),
                project_path=str(self.project_path),
            )
            analysis = result.get("analysis", {})
            raw = analysis.get("analysis", "") if isinstance(analysis, dict) else str(analysis)
            return raw[:2000] if raw else "(no analysis)"
        except Exception as e:
            return f"(analyze failed: {e})"

    async def _get_dependencies(self, path: str) -> str:
        fp = self._safe_path(path)
        if not fp:
            return "(invalid path)"
        try:
            from services.graph_service import dependency_builder
            dep_graph = dependency_builder.build_from_project(self.project_path)
            node_key = str(fp.relative_to(self.project_path))
            imports   = list(dep_graph.successors(node_key))[:10]
            importers = list(dep_graph.predecessors(node_key))[:10]
            return (
                f"File: {node_key}\n"
                f"Imports ({len(imports)}): {', '.join(imports) or 'none'}\n"
                f"Used by ({len(importers)}): {', '.join(importers) or 'none'}"
            )
        except Exception as e:
            return f"(dep graph failed: {e})"

    async def _get_ci_status(self) -> str:
        try:
            from services.mcp_redis_service import get_mcp_redis
            redis = get_mcp_redis()
            keys = redis.keys("ci:result:*")
            if not keys:
                return "(no CI results cached)"
            # Get most recent
            key = sorted(keys)[-1]
            data = redis.get(key)
            if not data:
                return "(no CI data)"
            import json as _json
            result = _json.loads(data)
            return (
                f"Outcome: {result.get('outcome', '?')}\n"
                f"Stage failed: {result.get('stage_failed', 'none')}\n"
                f"Root cause: {(result.get('root_cause') or '')[:200]}"
            )
        except Exception as e:
            return f"(ci status failed: {e})"


# ══════════════════════════════════════════════════════════════════════════════
# Main agent
# ══════════════════════════════════════════════════════════════════════════════

class LCToolCallingAgent:
    """
    LLM-driven pair programmer with real tool calls.

    Execution loop:
      1. Build prompt with tools + message + history
      2. LLM call → may return tool_calls
      3. Execute each tool call
      4. Feed results back to LLM
      5. Repeat up to max_iterations
      6. Return final text response
    """

    MAX_ITERATIONS = 4

    def __init__(self):
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            try:
                from langchain_agents.agents.lc_analysis_agent import _build_llm_with_fallback
                raw_llm = _build_llm_with_fallback()
                # Try to bind tools (OpenAI/Gemini function calling)
                self._llm = raw_llm.bind_tools(
                    [{"type": "function", "function": s} for s in TOOL_SCHEMAS]
                )
            except Exception as e:
                logger.debug("bind_tools failed: %s", e)
                self._llm = None
        return self._llm

    async def arun(
        self,
        message: str,
        project_path: str,
        session_id: str = "",
        history: Optional[List[Dict[str, Any]]] = None,
        semantic_context: Optional[List[str]] = None,
        target_file: str = "",
    ) -> Dict[str, Any]:
        """
        Async tool-calling pair programmer execution.

        Returns: {response, tools_called, iterations}
        """
        executor = ToolExecutor(project_path)
        history = history or []
        tools_called: List[str] = []

        # Build system prompt
        semantic_ctx = ""
        if semantic_context:
            semantic_ctx = "\n\nUser context from previous sessions:\n" + "\n".join(
                f"- {f}" for f in semantic_context[:5]
            )

        system = (
            "You are an expert pair programmer embedded in a VS Code extension. "
            "You have access to tools to explore the codebase. "
            "Use tools when you need specific information — don't guess. "
            "Be concise and actionable."
            f"{semantic_ctx}"
        )

        if target_file:
            system += f"\n\nCurrently focused file: {target_file}"

        # Build messages
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

        messages = [SystemMessage(content=system)]
        for turn in history[-4:]:
            role    = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=message))

        # Agentic loop
        iterations = 0
        final_response = ""

        llm = self.llm
        if llm is None:
            # Fallback: direct answer without tools
            try:
                from services.llm_factory import invoke_with_fallback
                final_response = invoke_with_fallback(
                    f"{system}\n\nUser: {message}",
                    label="tool_calling_fallback",
                    max_tokens=512,
                )
            except Exception as e:
                final_response = f"(Tool-calling unavailable: {e})"
            return {"response": final_response, "tools_called": [], "iterations": 0}

        while iterations < self.MAX_ITERATIONS:
            iterations += 1

            try:
                ai_msg = await asyncio.to_thread(llm.invoke, messages)
            except Exception as e:
                logger.warning("LLM invoke failed: %s", e)
                final_response = f"(LLM error: {e})"
                break

            # Check for tool calls
            tool_calls = getattr(ai_msg, "tool_calls", []) or []

            if not tool_calls:
                # Final answer
                final_response = getattr(ai_msg, "content", "") or str(ai_msg)
                break

            # Execute tool calls
            messages.append(ai_msg)
            for tc in tool_calls:
                tool_name = tc.get("name") or (tc.get("function", {}).get("name", ""))
                args_raw  = tc.get("args") or tc.get("function", {}).get("arguments", "{}")
                if isinstance(args_raw, str):
                    try:
                        args = json.loads(args_raw)
                    except Exception:
                        args = {}
                else:
                    args = args_raw or {}

                logger.info("Tool call: %s(%s)", tool_name, list(args.keys()))
                tools_called.append(tool_name)

                tool_result = await executor.execute(tool_name, args)

                messages.append(ToolMessage(
                    content=str(tool_result)[:2000],
                    tool_call_id=tc.get("id", f"call_{iterations}_{tool_name}"),
                ))

        if not final_response:
            final_response = "(No response generated)"

        return {
            "response":    final_response,
            "tools_called": tools_called,
            "iterations":  iterations,
        }


lc_tool_calling_agent = LCToolCallingAgent()
