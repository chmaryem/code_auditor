"""
api/chat_router.py — FastAPI router for ChatAgent.

Endpoints:
  POST /chat                 — Q&A / explain
  POST /chat/stream          — Q&A / explain streaming via SSE
  POST /chat/complete        — Complete a function body
  POST /chat/generate        — Generate a new class/file
  GET  /chat/history/{id}    — Load session conversation history
  DELETE /chat/history/{id}  — Clear session history

Usage:
    from api.chat_router import chat_router
    app.include_router(chat_router, prefix="/api")

Final URL examples:
    POST /api/chat
    POST /api/chat/stream
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from auth.security import Principal, get_current_user

logger = logging.getLogger(__name__)

chat_router = APIRouter(prefix="/chat", tags=["ChatAgent"])


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Q&A / explain request."""
    message:           str = Field(..., description="Developer question or request")
    project_path:      str = Field(".", description="Project root (absolute or relative)")
    session_id:        str = Field("", description="Session ID for conversation continuity")
    target_file:       str = Field("", description="Optional file to focus on (path or filename)")
    # IDE cursor context (sent by VS Code extension)
    cursor_line:       int = Field(0,  description="Line number under cursor (0 = unknown)")
    active_function:   str = Field("", description="Function/method name under cursor")
    selected_text:     str = Field("", description="Currently selected code (empty if none)")
    visible_range:     list[int] = Field(default_factory=lambda: [0, 0],
                                         description="[start_line, end_line] of visible editor area")
    # Dashboard context (sent by React webview)
    active_module:     str = Field("", description="Active dashboard module: cicd|git|chat|analyze|tests")
    branch:            str = Field("", description="Current git branch")
    active_repository: str = Field("", description="Name of the active repository")


class CompletionRequest(BaseModel):
    """Function completion request."""
    function_name: str = Field(..., description="Name of the function to complete")
    file_path: str = Field("", description="File containing the function")
    project_path: str = Field(".", description="Project root")
    session_id: str = Field("", description="Session ID")
    language: str = Field("", description="Language override")


class GenerateClassRequest(BaseModel):
    """New class/file generation request."""
    class_name: str = Field(..., description="PascalCase class name to generate")
    language: str = Field("python", description="Target language")
    project_path: str = Field(".", description="Project root")
    session_id: str = Field("", description="Session ID")
    description: str = Field("", description="Optional natural language description")
    write_to_disk: bool = Field(False, description="Trusted local mode only")


class ChatResponse(BaseModel):
    """Unified response for all non-streaming chat endpoints."""
    response: str = ""
    session_id: str = ""
    intent: str = "question"
    context_level: str = "context"
    selected_agents: List[str] = Field(default_factory=list)
    target_file: str = ""
    generated_code: str = ""
    code_blocks: List[Dict[str, str]] = Field(default_factory=list)
    suggested_files: List[str] = Field(default_factory=list)
    generation_valid: bool = True
    generation_errors: List[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    context_sources: List[str] = Field(default_factory=list)


class HistoryEntry(BaseModel):
    role: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionHistoryResponse(BaseModel):
    session_id: str
    turns: List[HistoryEntry] = Field(default_factory=list)
    turn_count: int = 0


class SessionSummary(BaseModel):
    session_id: str
    title: str = "New conversation"
    updated_at: int = 0
    turn_count: int = 0


class SessionListResponse(BaseModel):
    sessions: List[SessionSummary] = Field(default_factory=list)


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _run_chat(
    message: str,
    project_path: str,
    session_id: str,
    target_file: str,
    user_id: str = "",
    cursor_line: int = 0,
    active_function: str = "",
    selected_text: str = "",
    visible_range: list | None = None,
    active_module: str = "",
    branch: str = "",
    active_repository: str = "",
) -> ChatResponse:
    """Run ainvoke_chat() and map the result to ChatResponse."""
    from langchain_agents.graphs.chat_graph import ainvoke_chat

    start = time.time()
    try:
        result: Dict[str, Any] = await ainvoke_chat(
            message=message,
            project_path=project_path,
            session_id=session_id,
            user_id=user_id,
            target_file=target_file,
            cursor_line=cursor_line,
            active_function=active_function,
            selected_text=selected_text,
            visible_range=visible_range or [0, 0],
            active_module=active_module,
            branch=branch,
            active_repository=active_repository,
        )
    except Exception as e:
        logger.exception("ainvoke_chat failed: %s", e)
        raise HTTPException(status_code=500, detail=f"ChatAgent error: {e}")

    elapsed = round(time.time() - start, 2)

    return ChatResponse(
        response=result.get("formatted_response") or result.get("response", ""),
        session_id=result.get("session_id", session_id),
        intent=result.get("intent", "question"),
        context_level=result.get("context_level", "context"),
        selected_agents=result.get("selected_agents") or [],
        target_file=result.get("target_file", ""),
        generated_code=result.get("generated_code", ""),
        code_blocks=result.get("code_blocks") or [],
        suggested_files=result.get("suggested_files") or [],
        generation_valid=result.get("generation_valid", True),
        generation_errors=result.get("generation_errors") or [],
        elapsed_seconds=elapsed,
        context_sources=result.get("context_sources") or [],
    )


def _safe_write_generated_file(project_path: str, suggested_file: str, code: str) -> None:
    """Write generated code only inside project root and never overwrite existing files."""
    root = Path(project_path).resolve()
    out_path = Path(suggested_file).resolve()

    if not str(out_path).startswith(str(root)):
        raise ValueError(f"Refusing to write outside project root: {out_path}")

    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(code, encoding="utf-8")


# ── Endpoints ────────────────────────────────────────────────────────────────

@chat_router.post("", response_model=ChatResponse, summary="Q&A / Explain")
async def chat(req: ChatRequest, user: Principal = Depends(get_current_user)):
    """Project-aware Q&A and code explanation with IDE cursor context."""
    return await _run_chat(
        message           = req.message,
        project_path      = req.project_path,
        session_id        = req.session_id,
        user_id           = user.id,
        target_file       = req.target_file,
        cursor_line       = req.cursor_line,
        active_function   = req.active_function,
        selected_text     = req.selected_text,
        visible_range     = req.visible_range,
        active_module     = req.active_module,
        branch            = req.branch,
        active_repository = req.active_repository,
    )


@chat_router.post("/stream", summary="Stream Q&A / Explain via SSE")
async def chat_stream(req: ChatRequest, user: Principal = Depends(get_current_user)):
    """
    Streaming version of Q&A endpoint.

    Returns Server-Sent Events (SSE):
      data: {"type":"status","content":"..."}
      data: {"type":"plan","intent":"..."}
      data: {"type":"token","content":"..."}
      data: {"type":"done","session_id":"..."}
    """
    from langchain_agents.graphs.chat_graph import stream_chat

    async def event_generator():
        try:
            async for chunk in stream_chat(
                message           = req.message,
                project_path      = req.project_path,
                session_id        = req.session_id,
                user_id           = user.id,
                target_file       = req.target_file,
                cursor_line       = req.cursor_line,
                active_function   = req.active_function,
                selected_text     = req.selected_text,
                visible_range     = req.visible_range,
                active_module     = req.active_module,
                branch            = req.branch,
                active_repository = req.active_repository,
            ):
                yield chunk
        except asyncio.CancelledError:
            logger.info("Client disconnected from chat stream")
            raise
        except Exception as e:
            logger.exception("Stream error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@chat_router.post("/complete", response_model=ChatResponse, summary="Complete a function")
async def complete_function(req: CompletionRequest, user: Principal = Depends(get_current_user)):
    """Generate the body of an incomplete/stub function."""
    file_part = f" in {req.file_path}" if req.file_path else ""
    lang_part = f" using {req.language}" if req.language else ""
    message = f"complete {req.function_name}{file_part}{lang_part}"

    return await _run_chat(
        message=message,
        project_path=req.project_path,
        session_id=req.session_id,
        user_id=user.id,
        target_file=req.file_path,
    )


@chat_router.post("/generate", response_model=ChatResponse, summary="Generate new class/file")
async def generate_class(req: GenerateClassRequest, user: Principal = Depends(get_current_user)):
    """Generate a complete new class following project conventions."""
    lang_part = f" in {req.language}" if req.language else ""
    desc_part = f" — {req.description}" if req.description else ""
    message = f"create a {req.class_name} class{lang_part}{desc_part}"

    response = await _run_chat(
        message=message,
        project_path=req.project_path,
        session_id=req.session_id,
        user_id=user.id,
        target_file="",
    )

    # Trusted local mode only. Plugin should keep write_to_disk=false.
    if req.write_to_disk and response.generated_code and response.suggested_files:
        try:
            _safe_write_generated_file(
                project_path=req.project_path,
                suggested_file=response.suggested_files[0],
                code=response.generated_code,
            )
            logger.info("Generated file written: %s", response.suggested_files[0])
        except Exception as e:
            logger.warning("write_to_disk failed: %s", e)
            response.response += f"\n\n---\n⚠️ Write failed: {e}"

    return response


@chat_router.get(
    "/history/{session_id}",
    response_model=SessionHistoryResponse,
    summary="Get conversation history",
)
async def get_history(
    session_id: str,
    user: Principal = Depends(get_current_user),
):
    """
    Retourne l'historique d'une session.

    Stratégie de lecture :
      1. Redis (cache chaud, TTL 1 h)
      2. PostgreSQL en fallback (source de vérité durable)

    La vérification de propriété se fait côté PG : si la conversation
    n'appartient pas à l'utilisateur connecté, elle n'est pas retournée.
    """
    from services.chat_memory_service import chat_memory_service
    from database.connection import AsyncSessionLocal
    from database.repositories.conversation_repo import ConversationRepo
    from database.repositories.user_repo import UserRepo

    # ── 1. Cache Redis (fast path) ────────────────────────────────────────
    history: List[Dict[str, Any]] = []
    try:
        history = await asyncio.to_thread(chat_memory_service.load_history, session_id)
    except Exception as exc:
        logger.debug("get_history: Redis read failed: %s", exc)

    # ── 2. Fallback PostgreSQL si Redis vide ──────────────────────────────
    if not history:
        try:
            async with AsyncSessionLocal() as db:
                user_repo   = UserRepo(db)
                conv_repo   = ConversationRepo(db)

                pg_user = await user_repo.get_by_email(user.email)
                if pg_user:
                    conv = await conv_repo.get_by_session_id(session_id)
                    if conv and conv.user_id == pg_user.id:
                        messages = await conv_repo.load_messages(conv.id, limit=100)
                        history = [
                            {
                                "role":     m.role,
                                "content":  m.content,
                                "metadata": m.metadata_ or {},
                            }
                            for m in messages
                        ]
        except Exception as exc:
            logger.warning("get_history: PG fallback failed (session=%s): %s", session_id, exc)

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
async def clear_history(
    session_id: str,
    user: Principal = Depends(get_current_user),
):
    """
    Supprime l'historique Redis d'une session et l'invalide dans l'index.
    Les lignes PostgreSQL sont conservées (soft-delete côté cache uniquement).
    L'endpoint vérifie que la session appartient à l'utilisateur connecté.
    """
    from services.chat_memory_service import chat_memory_service
    from services.persistent_chat_memory_service import persistent_chat_memory
    from database.connection import AsyncSessionLocal
    from database.repositories.conversation_repo import ConversationRepo
    from database.repositories.user_repo import UserRepo

    # Vérification de propriété via PostgreSQL
    try:
        async with AsyncSessionLocal() as db:
            pg_user = await UserRepo(db).get_by_email(user.email)
            if pg_user:
                conv = await ConversationRepo(db).get_by_session_id(session_id)
                if conv and conv.user_id != pg_user.id:
                    return {"status": "forbidden", "session_id": session_id}
    except Exception as exc:
        logger.debug("clear_history: ownership check failed: %s", exc)

    # Suppression Redis (source PG intacte)
    try:
        await asyncio.to_thread(chat_memory_service.clear_session, session_id)
        persistent_chat_memory.invalidate_redis_cache(session_id)
        return {"status": "cleared", "session_id": session_id}
    except Exception as exc:
        logger.warning("clear_history failed (session=%s): %s", session_id, exc)
        return {"status": "error", "detail": str(exc)}


@chat_router.get(
    "/sessions",
    response_model=SessionListResponse,
    summary="List past conversations for the authenticated user",
)
async def list_sessions(
    project_path: str = ".",
    limit: int = 50,
    user: Principal = Depends(get_current_user),
):
    """
    Liste les conversations de l'utilisateur connecté, les plus récentes en premier.

    Stratégie :
      1. PostgreSQL (source de vérité, scopé par user_id) — résultats fiables
         même après un logout/login ou une expiration Redis.
      2. Enrichissement Redis pour les sessions dont le cache est encore chaud.

    Le project_path est utilisé comme filtre optionnel.
    """
    from database.connection import AsyncSessionLocal
    from database.repositories.conversation_repo import ConversationRepo
    from database.repositories.user_repo import UserRepo

    sessions: List[Dict[str, Any]] = []

    try:
        async with AsyncSessionLocal() as db:
            pg_user = await UserRepo(db).get_by_email(user.email)
            if pg_user:
                convs = await ConversationRepo(db).list_for_user(
                    user_id=pg_user.id,
                    limit=limit,
                )
                sessions = [
                    {
                        "session_id": c.session_id,
                        "title":      c.title or "New conversation",
                        "updated_at": int(c.updated_at.timestamp()),
                        "turn_count": c.turn_count,
                    }
                    for c in convs
                ]
    except Exception as exc:
        logger.warning("list_sessions: PG read failed (user=%s): %s", user.email, exc)

    # Fallback Redis si PG vide (ex. base non migrée ou premier démarrage)
    if not sessions:
        try:
            from services.chat_memory_service import chat_memory_service
            redis_sessions = await asyncio.to_thread(
                chat_memory_service.list_sessions, project_path, limit
            )
            sessions = redis_sessions
        except Exception as exc:
            logger.debug("list_sessions: Redis fallback also failed: %s", exc)

    return SessionListResponse(
        sessions=[SessionSummary(**s) for s in sessions]
    )


# ══════════════════════════════════════════════════════════════════════════════
# Phase B — New endpoints
# ══════════════════════════════════════════════════════════════════════════════

class InlineCompletionRequest(BaseModel):
    prefix_code:  str  = Field(...,      description="Code before the cursor")
    suffix_code:  str  = Field("",       description="Code after the cursor")
    language:     str  = Field("python", description="Programming language")
    file_path:    str  = Field("",       description="Current file path")
    project_path: str  = Field(".",      description="Project root")
    cursor_line:  int  = Field(0)
    use_rag:      bool = Field(True)


class InlineCompletionResponse(BaseModel):
    completion:  str   = ""
    confidence:  float = 0.0
    source:      str   = "llm"
    elapsed_ms:  int   = 0
    error:       str   = ""


@chat_router.post("/complete/inline", response_model=InlineCompletionResponse,
                  summary="Inline completion (Copilot-style)")
async def complete_inline(req: InlineCompletionRequest):
    """Cursor-aware inline code completion. Cache-first (Redis TTL 5min)."""
    try:
        from langchain_agents.agents.lc_inline_completion_agent import lc_inline_completion_agent
        result = await asyncio.to_thread(
            lc_inline_completion_agent.complete,
            prefix_code=req.prefix_code,
            suffix_code=req.suffix_code,
            language=req.language,
            file_path=req.file_path,
            project_path=req.project_path,
            cursor_line=req.cursor_line,
            use_rag=req.use_rag,
        )
    except Exception as e:
        raise HTTPException(500, f"InlineCompletion error: {e}")
    return InlineCompletionResponse(**result)


@chat_router.post(
    "/complete/inline/stream",
    summary="Streaming inline completion via SSE — 3-tier pipeline",
)
async def complete_inline_stream(req: InlineCompletionRequest):
    """
    Streaming inline code completion — 3-tier pipeline identical to the JSON endpoint.

    Events:
      data: {"type":"cache_hit","completion":"...","source":"cache"|"graph",...}
      data: {"type":"token","content":"..."}
      data: {"type":"done","completion":"...","confidence":0.9,"elapsed_ms":280}
      data: {"type":"error","content":"..."}
    """
    import json as _json
    import time as _time

    async def _event_gen():
        t0       = _time.time()
        language = req.language or "python"

        from langchain_agents.agents.lc_inline_completion_agent import lc_inline_completion_agent

        # ── Tier 1: in-memory cache ───────────────────────────────────────────
        cache_key = lc_inline_completion_agent._cache_key(
            req.prefix_code, req.suffix_code, language
        )
        cached = lc_inline_completion_agent._cache_get(cache_key)
        if cached:
            elapsed = round((_time.time() - t0) * 1000)
            yield f"data: {_json.dumps({'type':'cache_hit','completion':cached,'confidence':0.85,'source':'cache','elapsed_ms':elapsed})}\n\n"
            return

        # ── Tier 2: graph fast path (unambiguous symbol, no LLM) ─────────────
        fast = await asyncio.to_thread(
            lc_inline_completion_agent._try_graph_completion,
            req.file_path, req.prefix_code, req.cursor_line,
        )
        if fast is not None:
            lc_inline_completion_agent._cache_set(cache_key, fast)
            elapsed = round((_time.time() - t0) * 1000)
            yield f"data: {_json.dumps({'type':'cache_hit','completion':fast,'confidence':0.95,'source':'graph','elapsed_ms':elapsed})}\n\n"
            return

        # ── Tier 3: LLM stream with project context ───────────────────────────
        ctx = await asyncio.to_thread(
            lc_inline_completion_agent._build_project_context,
            req.file_path, req.prefix_code, req.cursor_line,
        )
        context_section = (
            f"\nProject context (match style and types):\n{ctx}\n" if ctx else ""
        )

        # Security gate (Phase A): redact before sending to external LLM
        _safe_prefix = req.prefix_code
        _safe_suffix = req.suffix_code
        try:
            from services.secret_redactor import redact_secrets as _redact
            _safe_prefix, _np = _redact(_safe_prefix)
            _safe_suffix, _ns = _redact(_safe_suffix)
        except Exception:
            pass

        import re as _re
        fim_prompt = (
            f"Complete the {language} code at <CURSOR>. Output ONLY the inserted text — "
            f"no explanation, no markdown fences.\n"
            f"Keep it concise (1–5 lines). Must be syntactically valid {language}."
            f"{context_section}\n"
            f"<prefix>{_safe_prefix[-1200:]}</prefix><CURSOR>"
            f"<suffix>{_safe_suffix[:300]}</suffix>\n\n"
            f"Completion:"
        )

        completion_buf = []
        try:
            llm = lc_inline_completion_agent.fast_llm
            if llm is None:
                raise RuntimeError("fast LLM unavailable")
            from langchain_core.messages import HumanMessage

            async for chunk in llm.astream([HumanMessage(content=fim_prompt)]):
                token = getattr(chunk, "content", "")
                if token:
                    token = _re.sub(r"```[a-zA-Z0-9_-]*", "", token)
                    if token:
                        completion_buf.append(token)
                        yield f"data: {_json.dumps({'type':'token','content':token})}\n\n"

        except Exception:
            # Fallback: sync complete() call
            try:
                result = await asyncio.to_thread(
                    lc_inline_completion_agent.complete,
                    prefix_code=req.prefix_code,
                    suffix_code=req.suffix_code,
                    language=language,
                    file_path=req.file_path,
                    project_path=req.project_path,
                    cursor_line=req.cursor_line,
                    use_rag=False,
                )
                text = result.get("completion", "")
                if text:
                    completion_buf = [text]
                    yield f"data: {_json.dumps({'type':'token','content':text})}\n\n"
            except Exception as e2:
                yield f"data: {_json.dumps({'type':'error','content':str(e2)})}\n\n"
                return

        full = "".join(completion_buf).strip()
        full = _re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", full)
        full = _re.sub(r"\n?```$", "", full)

        if full:
            lc_inline_completion_agent._cache_set(cache_key, full)

        elapsed    = round((_time.time() - t0) * 1000)
        confidence = 0.9 if len(full) > 5 else 0.5
        yield f"data: {_json.dumps({'type':'done','completion':full,'confidence':confidence,'source':'llm','elapsed_ms':elapsed})}\n\n"

    return StreamingResponse(
        _event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "Connection":       "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


class ApplyFunctionRequest(BaseModel):
    project_path:  str = Field(...,        description="Project root")
    file_path:     str = Field(...,        description="Target file")
    function_name: str = Field(...,        description="Function to replace")
    new_code:      str = Field(...,        description="New implementation")
    language:      str = Field("python")
    write_mode:    str = Field("dry_run",  description="dry_run | preview | apply")


class ApplyNewFileRequest(BaseModel):
    project_path:   str = Field(...)
    suggested_file: str = Field(...)
    code:           str = Field(...)
    language:       str = Field("python")
    write_mode:     str = Field("dry_run")


class ApplyMultiFileRequest(BaseModel):
    project_path: str                   = Field(...)
    patches:      List[Dict[str, Any]]  = Field(...)
    write_mode:   str                   = Field("dry_run")


class ApplyResponse(BaseModel):
    file_path:      str               = ""
    function_name:  str               = ""
    diff:           str               = ""
    workspace_edit: Dict[str, Any]    = Field(default_factory=dict)
    valid:          bool              = True
    errors:         List[str]         = Field(default_factory=list)
    written:        bool              = False
    mode:           str               = "dry_run"
    error:          str               = ""
    patches:        List[Dict[str, Any]] = Field(default_factory=list)
    all_valid:      bool              = True


@chat_router.post("/apply", response_model=ApplyResponse,
                  summary="Apply generated code to a function (diff + WorkspaceEdit)")
async def apply_function(req: ApplyFunctionRequest):
    """
    Replace a function in a file with generated code.
    Returns unified diff + VS Code WorkspaceEdit JSON.
    write_mode: dry_run (default, safe) | preview | apply (writes to disk).
    """
    try:
        from langchain_agents.agents.lc_apply_agent import lc_apply_agent
        result = await asyncio.to_thread(
            lc_apply_agent.apply_function_patch,
            project_path=req.project_path, file_path=req.file_path,
            function_name=req.function_name, new_code=req.new_code,
            language=req.language, write_mode=req.write_mode,
        )
    except Exception as e:
        raise HTTPException(500, f"ApplyAgent error: {e}")
    if result.get("error"):
        raise HTTPException(400, result["error"])
    return ApplyResponse(**{k: v for k, v in result.items() if k in ApplyResponse.model_fields})


@chat_router.post("/apply/new-file", response_model=ApplyResponse,
                  summary="Apply generated code as a new file")
async def apply_new_file(req: ApplyNewFileRequest):
    """Create a new file with generated code. Never overwrites unless write_mode='overwrite'."""
    try:
        from langchain_agents.agents.lc_apply_agent import lc_apply_agent
        result = await asyncio.to_thread(
            lc_apply_agent.apply_new_file,
            project_path=req.project_path, suggested_file=req.suggested_file,
            code=req.code, language=req.language, write_mode=req.write_mode,
        )
    except Exception as e:
        raise HTTPException(500, f"ApplyAgent error: {e}")
    if result.get("error"):
        raise HTTPException(400, result["error"])
    return ApplyResponse(**{k: v for k, v in result.items() if k in ApplyResponse.model_fields})


@chat_router.post("/apply/multi", response_model=ApplyResponse,
                  summary="Multi-file patch (Phase B3)")
async def apply_multi_file(req: ApplyMultiFileRequest):
    """
    Apply generated patches to multiple files in one operation.
    Patch: {file_path, function_name, new_code, language} or {file_path, code, is_new_file, language}
    """
    try:
        from langchain_agents.agents.lc_apply_agent import lc_apply_agent
        result = await asyncio.to_thread(
            lc_apply_agent.apply_multi_file_patch,
            project_path=req.project_path, patches=req.patches, write_mode=req.write_mode,
        )
    except Exception as e:
        raise HTTPException(500, f"ApplyAgent multi error: {e}")
    return ApplyResponse(all_valid=result.get("all_valid", False),
                         written=result.get("written", False),
                         mode=result.get("mode", "dry_run"),
                         patches=result.get("patches", []))


# ══════════════════════════════════════════════════════════════════════════════
# Phase C — Proactive + Semantic Memory + Pair Programmer
# ══════════════════════════════════════════════════════════════════════════════

class ProactiveRequest(BaseModel):
    project_path:  str = Field(...,  description="Project root")
    session_id:    str = Field("",   description="Current session ID")
    target_file:   str = Field("",   description="Currently open file")
    max_suggestions: int = Field(5,  description="Max suggestions to return")


class ProactiveSuggestion(BaseModel):
    type:     str = ""
    severity: str = "info"
    title:    str = ""
    message:  str = ""
    file:     str = ""
    action:   Optional[str] = None


class ProactiveResponse(BaseModel):
    suggestions:  List[ProactiveSuggestion] = Field(default_factory=list)
    total:        int   = 0
    has_critical: bool  = False
    elapsed_ms:   int   = 0


@chat_router.post("/proactive", response_model=ProactiveResponse,
                  summary="Proactive suggestions (Phase C2)")
async def proactive_suggestions(req: ProactiveRequest):
    """
    Scan the project for proactive suggestions without the developer asking:
      - Uncommitted critical bugs (from GitSessionTracker)
      - Test gaps (modified files without tests)
      - CI risk patterns (files historically correlated with CI failures)
      - Coupling impact (high-fan-in files modified)

    VS Code extension shows these in a side panel or as notifications.
    """
    try:
        from langchain_agents.agents.lc_proactive_agent import proactive_agent
        result = await asyncio.to_thread(
            proactive_agent.scan,
            project_path    = req.project_path,
            session_id      = req.session_id,
            target_file     = req.target_file,
            max_suggestions = req.max_suggestions,
        )
    except Exception as e:
        logger.exception("proactive error: %s", e)
        raise HTTPException(500, f"ProactiveAgent error: {e}")

    suggestions = [
        ProactiveSuggestion(
            type     = s.get("type", ""),
            severity = s.get("severity", "info"),
            title    = s.get("title", ""),
            message  = s.get("message", ""),
            file     = s.get("file", ""),
            action   = s.get("action"),
        )
        for s in result.get("suggestions", [])
    ]

    return ProactiveResponse(
        suggestions  = suggestions,
        total        = result.get("total", 0),
        has_critical = result.get("has_critical", False),
        elapsed_ms   = result.get("elapsed_ms", 0),
    )


class SemanticMemoryRequest(BaseModel):
    session_id:   str = Field(..., description="Session ID")
    query:        str = Field("",  description="Query for semantic recall (empty = profile)")
    action:       str = Field("recall", description="recall | profile | clear")


@chat_router.post("/memory/semantic", summary="Semantic memory operations (Phase C1)")
async def semantic_memory_op(req: SemanticMemoryRequest):
    """
    Manage per-session semantic memory:
      - recall : find relevant past facts for a query (vector search)
      - profile : build a developer profile from all stored facts
      - clear   : delete all semantic memory for this session
    """
    try:
        from langchain_agents.memory.lc_semantic_memory import semantic_memory

        if req.action == "recall":
            facts = await asyncio.to_thread(
                semantic_memory.recall_memory,
                session_id=req.session_id,
                query=req.query or "project",
            )
            return {"action": "recall", "session_id": req.session_id, "facts": facts}

        elif req.action == "profile":
            profile = await asyncio.to_thread(
                semantic_memory.get_profile,
                session_id=req.session_id,
            )
            return {"action": "profile", "session_id": req.session_id, "profile": profile}

        elif req.action == "clear":
            ok = await asyncio.to_thread(semantic_memory.clear_session, req.session_id)
            return {"action": "clear", "session_id": req.session_id, "cleared": ok}

        else:
            raise HTTPException(400, f"Unknown action: {req.action}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"SemanticMemory error: {e}")


class PairProgrammerRequest(BaseModel):
    """Direct tool-calling pair programmer — bypasses the full ChatGraph."""
    message:         str = Field(...,  description="Developer question")
    project_path:    str = Field(".",  description="Project root")
    session_id:      str = Field("",   description="Session ID")
    target_file:     str = Field("",   description="Currently focused file")
    history:         List[Dict[str, Any]] = Field(default_factory=list,
                                                   description="Last N conversation turns")


class PairProgrammerResponse(BaseModel):
    response:     str       = ""
    tools_called: List[str] = Field(default_factory=list)
    iterations:   int       = 0
    elapsed_seconds: float  = 0.0


@chat_router.post("/pair-programmer", response_model=PairProgrammerResponse,
                  summary="Pair programmer with tool-calling (Phase C3)")
async def pair_programmer(req: PairProgrammerRequest):
    """
    Direct pair programmer endpoint that bypasses the full ChatGraph.

    The LLM autonomously decides which tools to use:
      search_codebase | get_file | get_git_diff | analyze_file | get_dependencies | get_ci_status

    Best for deep, multi-step questions about the project.
    Max 4 tool-call iterations.
    """
    import time

    t0 = time.time()
    try:
        from langchain_agents.agents.lc_tool_calling_agent import lc_tool_calling_agent
        from langchain_agents.memory.lc_semantic_memory import semantic_memory

        # Enrich with semantic context
        semantic_ctx = await asyncio.to_thread(
            semantic_memory.recall_memory,
            session_id=req.session_id,
            query=req.message,
        ) if req.session_id else []

        result = await lc_tool_calling_agent.arun(
            message          = req.message,
            project_path     = req.project_path,
            session_id       = req.session_id,
            history          = req.history,
            semantic_context = semantic_ctx,
            target_file      = req.target_file,
        )
    except Exception as e:
        logger.exception("pair-programmer error: %s", e)
        raise HTTPException(500, f"ToolCallingAgent error: {e}")

    return PairProgrammerResponse(
        response        = result.get("response", ""),
        tools_called    = result.get("tools_called", []),
        iterations      = result.get("iterations", 0),
        elapsed_seconds = round(time.time() - t0, 2),
    )
