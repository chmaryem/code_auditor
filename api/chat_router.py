"""
api/chat_router.py — FastAPI router for ChatAgent (Phase 1 + Phase 2).

Endpoints:
  POST /chat              — Q&A / explain (Phase 1)
  POST /chat/complete     — Complete a function body (Phase 2)
  POST /chat/generate     — Generate a new class/file (Phase 2)
  GET  /chat/history/{id} — Load session conversation history
  DELETE /chat/history/{id} — Clear session history

All endpoints delegate to invoke_chat() from chat_graph.py.
The VS Code extension plugin calls these endpoints from the Chat Webview panel.

Usage (standalone):
    from api.chat_router import chat_router
    app.include_router(chat_router, prefix="/api")
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

chat_router = APIRouter(prefix="/chat", tags=["ChatAgent"])


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Phase 1 — Q&A / explain request."""
    message: str = Field(..., description="Developer question or request")
    project_path: str = Field(".", description="Project root (absolute or relative)")
    session_id: str = Field("", description="Session ID for conversation continuity")
    target_file: str = Field("", description="Optional file to focus on (path or filename)")


class CompletionRequest(BaseModel):
    """Phase 2 — Function completion request."""
    function_name: str = Field(
        ..., description="Name of the function to complete, e.g. 'findByEmail'"
    )
    file_path: str = Field(
        "", description="File containing the function, e.g. 'services/user_service.py'"
    )
    project_path: str = Field(".", description="Project root")
    session_id: str = Field("", description="Session ID")
    language: str = Field("", description="Language override (java|python|javascript|typescript)")


class GenerateClassRequest(BaseModel):
    """Phase 2 — New class/file generation request."""
    class_name: str = Field(
        ..., description="PascalCase class name to generate, e.g. 'NotificationService'"
    )
    language: str = Field("python", description="Target language")
    project_path: str = Field(".", description="Project root")
    session_id: str = Field("", description="Session ID")
    description: str = Field(
        "", description="Optional natural language description of what the class should do"
    )
    write_to_disk: bool = Field(
        False, description="If true, write the generated file to the suggested path"
    )


class ChatResponse(BaseModel):
    """Unified response for all chat endpoints."""
    response: str = Field("", description="Human-readable answer or generation result")
    session_id: str = Field("", description="Session ID (use to continue conversation)")
    intent: str = Field("question", description="Detected intent: question|explain|complete_fn|new_class")
    target_file: str = Field("", description="File used as context (resolved path)")
    generated_code: str = Field("", description="Raw generated code (Phase 2 only)")
    code_blocks: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Parsed code blocks [{lang, code}] for syntax highlighting in plugin"
    )
    suggested_files: List[str] = Field(
        default_factory=list,
        description="Suggested file paths for generated code"
    )
    generation_valid: bool = Field(True, description="True if generated code passed syntax check")
    generation_errors: List[str] = Field(
        default_factory=list, description="Syntax errors if generation_valid is False"
    )
    elapsed_seconds: float = Field(0.0, description="Total processing time in seconds")


class HistoryEntry(BaseModel):
    role: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionHistoryResponse(BaseModel):
    session_id: str
    turns: List[HistoryEntry] = Field(default_factory=list)
    turn_count: int = 0


# ── Helper ────────────────────────────────────────────────────────────────────

async def _run_chat(
    message: str,
    project_path: str,
    session_id: str,
    target_file: str,
) -> ChatResponse:
    """Run invoke_chat() in a thread and map the result to ChatResponse."""
    from langchain_agents.graphs.chat_graph import invoke_chat

    start = time.time()
    try:
        result: Dict[str, Any] = await asyncio.to_thread(
            invoke_chat,
            message=message,
            project_path=project_path,
            session_id=session_id,
            target_file=target_file,
        )
    except Exception as e:
        logger.exception("invoke_chat failed: %s", e)
        raise HTTPException(status_code=500, detail=f"ChatAgent error: {e}")

    elapsed = round(time.time() - start, 2)

    return ChatResponse(
        response=result.get("formatted_response") or result.get("response", ""),
        session_id=result.get("session_id", session_id),
        intent=result.get("intent", "question"),
        target_file=result.get("target_file", ""),
        generated_code=result.get("generated_code", ""),
        code_blocks=result.get("code_blocks") or [],
        suggested_files=result.get("suggested_files") or [],
        generation_valid=result.get("generation_valid", True),
        generation_errors=result.get("generation_errors") or [],
        elapsed_seconds=elapsed,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@chat_router.post("", response_model=ChatResponse, summary="Q&A / Explain (Phase 1)")
async def chat(req: ChatRequest):
    """
    Project-aware Q&A and code explanation.

    The agent automatically:
    - Detects intent (question / explain)
    - Loads file context if a filename or ClassName.method is mentioned
    - Retrieves RAG documents from the project knowledge base
    - Answers grounded in project code, cached analysis, and dependencies

    **Examples:**
    - `"What does CacheService.get_cached_analysis do?"`
    - `"Explain the dependency graph in services/dep_graph.py"`
    - `"What tests are missing in analysis_agent.py?"`
    """
    return await _run_chat(
        message=req.message,
        project_path=req.project_path,
        session_id=req.session_id,
        target_file=req.target_file,
    )


@chat_router.post("/complete", response_model=ChatResponse, summary="Complete a function (Phase 2)")
async def complete_function(req: CompletionRequest):
    """
    Generate the body of an incomplete/stub function.

    The agent:
    - Finds the function in the project (by name + file)
    - Detects project conventions (naming, imports, style)
    - Retrieves relevant RAG documents
    - Generates only the function body (never rewrites other code)
    - Validates Python syntax (or brace balance for Java/JS/TS)

    **Examples:**
    - function_name: `"findByEmail"`, file_path: `"services/user_service.py"`
    - function_name: `"processQueue"`, file_path: `"UserService.java"`
    """
    # Build a natural language message that the intent router understands
    file_part = f" in {req.file_path}" if req.file_path else ""
    message = f"complete {req.function_name}{file_part}"

    return await _run_chat(
        message=message,
        project_path=req.project_path,
        session_id=req.session_id,
        target_file=req.file_path,
    )


@chat_router.post("/generate", response_model=ChatResponse, summary="Generate new class/file (Phase 2)")
async def generate_class(req: GenerateClassRequest):
    """
    Generate a complete new class following project conventions.

    The agent:
    - Scans existing classes to infer naming/import/style conventions
    - Generates a complete, compilable class with docstrings
    - Includes a unit test skeleton
    - Suggests the correct file path based on project structure
    - Optionally writes the file to disk (`write_to_disk: true`)

    **Examples:**
    - class_name: `"NotificationService"`, language: `"python"`
    - class_name: `"ProductRepository"`, language: `"java"`, description: `"CRUD for Product entity"`
    """
    lang_part = f" in {req.language}" if req.language else ""
    desc_part = f" — {req.description}" if req.description else ""
    message = f"create a {req.class_name} class{lang_part}{desc_part}"

    response = await _run_chat(
        message=message,
        project_path=req.project_path,
        session_id=req.session_id,
        target_file="",
    )

    # Optionally write generated file to disk
    if req.write_to_disk and response.generated_code and response.suggested_files:
        from pathlib import Path
        out_path = Path(response.suggested_files[0])
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(response.generated_code, encoding="utf-8")
            logger.info("Generated file written: %s", out_path)
        except Exception as e:
            logger.warning("write_to_disk failed: %s", e)

    return response


@chat_router.get(
    "/history/{session_id}",
    response_model=SessionHistoryResponse,
    summary="Get conversation history",
)
async def get_history(session_id: str):
    """
    Retrieve the conversation history for a session.

    Used by the VS Code plugin to restore a previous chat session
    when the developer reopens the panel.
    """
    try:
        from services.chat_memory_service import chat_memory_service
        history = await asyncio.to_thread(chat_memory_service.load_history, session_id)
    except Exception as e:
        logger.warning("load_history failed: %s", e)
        history = []

    turns = [
        HistoryEntry(
            role=h.get("role", "user"),
            content=h.get("content", ""),
            metadata=h.get("metadata", {}),
        )
        for h in history
    ]

    return SessionHistoryResponse(
        session_id=session_id,
        turns=turns,
        turn_count=len(turns),
    )


@chat_router.delete(
    "/history/{session_id}",
    summary="Clear conversation history",
)
async def clear_history(session_id: str):
    """
    Clear the conversation history for a session (Redis + local fallback).
    """
    try:
        from services.chat_memory_service import chat_memory_service
        await asyncio.to_thread(chat_memory_service.clear_session, session_id)
        return {"status": "cleared", "session_id": session_id}
    except Exception as e:
        logger.warning("clear_session failed: %s", e)
        return {"status": "error", "detail": str(e)}
