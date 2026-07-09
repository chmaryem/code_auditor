"""
api/server.py — Code Auditor FastAPI Server (Multi-Agent v8).

Architecture 100% LangGraph :
  - WatchGraph    : watch mode (remplace legacy Orchestrator)
  - SmartGitGraph : git status / branch / conflicts
  - CIGraph       : CI/CD intelligence
  - ChatGraph     : conversational agent (Phase 1 + Phase 2)

Routers :
  /api/chat/*      → chat_router.py     (ChatGraph)
  /api/git/*       → git_router.py      (SmartGitGraph)
  /api/ci/*        → ci_router.py       (CIGraph)
  /api/watch/*     → watch endpoints    (WatchGraph)
  /analyze/file    → WatchGraph one-shot
  /analyze/project → project_analyzer (legacy OK)
  /ws              → WebSocket watch events

Démarrage :
    python -m api.server
    python -m api.server --port 9000 --project /path/to/project
"""
from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.models import (
    AnalyzeFileRequest,
    AnalysisResultResponse,
    AnalyzeProjectRequest,
    ProjectAnalysisResponse,
    ScanProjectTestsRequest,
    ScanProjectTestsResponse,
    WatchStartRequest,
    WatchStatusResponse,
    GitStatusRequest,
    GitBranchRequest,
    GenerateTestsRequest,
    GenerateTestsResponse,
    RunTestsRequest,
    RunTestsResponse,
    HealthResponse,
    WSEvent,
    IssueDiagnostic,
    FixSuggestion,
)
from api.ws_broadcast import manager as _ws_manager
import api.ws_broadcast as _ws_broadcast

logger = logging.getLogger(__name__)

# ── Shared state ──────────────────────────────────────────────────────────────

_server_start_time: float = 0.0
_default_project: Path    = Path(".")

# Multi-agent shared services (injected at startup)
_watch_services: Dict[str, Any] = {}   # project_indexer, extractor, rag_system, dep_graph
_watch_active: Dict[str, Any]   = {}   # project_path → {thread, counter, timers, lock}
_watch_last_events: Dict[str, Any] = {}  # project_path → latest ws_events snapshot

# Per-project service cache. The startup graph/indexer are built on the backend's
# OWN repo (_default_project = "."), so reusing them for a watched project made
# the dependency analysis report bogus files (e.g. config.py/main.py from the
# backend) as "dependents". These are (re)built and scoped to the watched project.
_project_services_cache: Dict[str, Dict[str, Any]] = {}
_project_services_lock = threading.Lock()

# Main asyncio event loop — captured at startup so background threads can broadcast
_main_loop: asyncio.AbstractEventLoop | None = None

# Extension → language mapping for fast pre-pipeline detection (no AST needed)
_EXT_LANGUAGE: Dict[str, str] = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".java": "java",
}

def _ext_to_language(fp: Path) -> str:
    return _EXT_LANGUAGE.get(fp.suffix.lower(), "unknown")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup : initialize multi-agent services in background.
    Shutdown : stop watchers + learning agent.
    """
    global _server_start_time, _watch_services, _main_loop

    # Capture the running asyncio event loop BEFORE any background threads start
    _main_loop = asyncio.get_running_loop()
    _ws_broadcast.set_loop(_main_loop)

    _server_start_time = time.time()
    logger.info("Code Auditor API v8 — démarrage (multi-agent mode)")

    # ── PostgreSQL init ────────────────────────────────────────────────────────
    try:
        from database.connection import init_db
        await init_db()
    except Exception as _db_err:
        logger.warning("Database init failed (running without persistent DB): %s", _db_err)

    def _init_services():
        """Background init — heavy services (embeddings, ChromaDB, dep graph)."""
        global _watch_services

        services: Dict[str, Any] = {}

        # RAG system
        try:
            from services.llm_service import CodeRAGSystemAPI
            services["rag_system"] = CodeRAGSystemAPI()
            logger.info("RAG system initialisé")
        except Exception as e:
            logger.warning("RAG system init failed: %s", e)

        # Project Indexer
        try:
            from services.project_indexer import get_project_index
            services["project_indexer"] = get_project_index(_default_project)
            logger.info("ProjectIndexer initialisé")
        except Exception as e:
            logger.warning("ProjectIndexer init failed: %s", e)

        # Dependency graph
        try:
            from services.graph_service import dependency_builder, DependencyExtractor
            dep_graph = dependency_builder.build_from_project(_default_project)
            services["dep_graph"]  = dep_graph
            services["extractor"] = DependencyExtractor(dep_graph)
            logger.info("DependencyGraph: %d nœuds", dep_graph.number_of_nodes())
        except Exception as e:
            logger.warning("DependencyGraph init failed: %s", e)

        # Knowledge Graph
        try:
            from services.knowledge_graph import knowledge_graph
            if not knowledge_graph._built:
                knowledge_graph.build(project_indexer=services.get("project_indexer"))
            logger.info("KnowledgeGraph initialisé")
        except Exception as e:
            logger.warning("KnowledgeGraph init failed: %s", e)

        # RetrieverAgent
        try:
            from agents.retriever_agent import retriever_agent
            from services.knowledge_graph import knowledge_graph as kg
            rag = services.get("rag_system")
            retriever_agent.initialize(
                graph=services.get("dep_graph"),
                project_indexer=services.get("project_indexer"),
                vector_store=rag.vector_store if rag else None,
                project_code_indexer=services.get("project_indexer"),
                knowledge_graph=kg if kg._built else None,
            )
        except Exception as e:
            logger.warning("RetrieverAgent init failed: %s", e)

        # TestKnowledgeLoader — collection ChromaDB dédiée aux patterns de test (RAG pour génération)
        try:
            from services.test_knowledge_loader import TestKnowledgeLoader
            _rag_svc = services.get("rag_system")
            _embeddings = (
                getattr(_rag_svc, "_embeddings", None)
                or getattr(_rag_svc, "embeddings", None)
            ) if _rag_svc else None
            test_kb = TestKnowledgeLoader(embeddings=_embeddings)
            loaded = test_kb.load()
            services["test_kb"] = test_kb
            logger.info("TestKnowledgeLoader initialisé (%d chunks)", loaded)
        except Exception as e:
            logger.warning("TestKnowledgeLoader init failed: %s", e)

        # LearningAgent
        try:
            from agents.learning_agent import learning_agent as la
            rag = services.get("rag_system")
            from config import config
            # CodeRAGSystemAPI exposes the LLM under different attribute names across
            # versions — probe defensively instead of assuming `_llm` (the missing
            # attribute that was disabling the LearningAgent entirely).
            _rag_llm = None
            if rag is not None:
                _rag_llm = (
                    getattr(rag, "_llm", None)
                    or getattr(rag, "llm", None)
                    or getattr(rag, "assistant_agent", None)
                )
            la.initialize(
                llm=_rag_llm,
                vector_store=getattr(rag, "vector_store", None) if rag else None,
                kb_dir=config.KNOWLEDGE_BASE_DIR,
            )
            la.start()
        except Exception as e:
            logger.warning("LearningAgent init failed: %s", e)

        _watch_services = services
        logger.info("Services multi-agents initialisés (%d)", len(services))

    threading.Thread(target=_init_services, daemon=True).start()

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Arrêt du serveur...")

    # Stop active watchers
    for _proj, ctx in list(_watch_active.items()):
        watcher = ctx.get("watcher")
        if watcher:
            try:
                watcher.stop()
            except Exception:
                pass
        for t in ctx.get("timers", {}).values():
            try:
                t.cancel()
            except Exception:
                pass

    # Stop LearningAgent
    try:
        from agents.learning_agent import learning_agent as la
        la.stop()
    except Exception:
        pass

    # Close PostgreSQL connection pool
    try:
        from database.connection import close_db
        await close_db()
    except Exception:
        pass


def _get_project_services(project_path: Path) -> Dict[str, Any]:
    """Build (and cache) a dependency graph + ProjectIndexer SCOPED to the
    watched project.

    Why: the services created at startup (``_watch_services``) are built on the
    backend's own repository (``_default_project``). Reusing them for a watched
    project made ``neighborhood.predecessors`` return backend files, so the watch
    reported non-existent "dependent files". Here we build a graph for the real
    project and re-wire the shared ``retriever_agent`` so neighborhood lookups are
    grounded in THIS project. Cached per project_path.

    Note: the ``retriever_agent`` singleton is shared, so the last watched project
    wins. In practice only one project is watched at a time.
    """
    key = str(project_path)
    with _project_services_lock:
        cached = _project_services_cache.get(key)
    if cached:
        return cached

    svc: Dict[str, Any] = {}

    try:
        from services.graph_service import dependency_builder, DependencyExtractor
        dep_graph = dependency_builder.build_from_project(project_path)
        svc["dep_graph"] = dep_graph
        svc["extractor"] = DependencyExtractor(dep_graph)
    except Exception as e:
        logger.warning("Project dep graph build failed for %s: %s", key, e)

    try:
        from services.project_indexer import ProjectIndexer
        pidx = ProjectIndexer(project_path)
        pidx.build_index(svc.get("dep_graph"))
        svc["project_indexer"] = pidx
    except Exception as e:
        logger.warning("Project indexer build failed for %s: %s", key, e)

    # Re-wire the shared retriever so get_neighborhood() resolves predecessors
    # from this project's graph instead of the backend repo graph.
    try:
        from agents.retriever_agent import retriever_agent
        from services.knowledge_graph import knowledge_graph as kg
        rag = _watch_services.get("rag_system")
        retriever_agent.initialize(
            graph=svc.get("dep_graph"),
            project_indexer=svc.get("project_indexer"),
            vector_store=getattr(rag, "vector_store", None) if rag else None,
            project_code_indexer=svc.get("project_indexer"),
            knowledge_graph=kg if getattr(kg, "_built", False) else None,
        )
    except Exception as e:
        logger.warning("retriever_agent re-init for project %s failed: %s", key, e)

    # Carry over the shared (project-agnostic) services.
    svc["rag_system"] = _watch_services.get("rag_system")
    svc["test_kb"]    = _watch_services.get("test_kb")

    with _project_services_lock:
        _project_services_cache[key] = svc

    _nodes = svc["dep_graph"].number_of_nodes() if svc.get("dep_graph") else 0
    logger.info("Project-scoped services built for %s (%d graph nodes)", key, _nodes)
    return svc


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Code Auditor API",
    description="REST + WebSocket — 100% multi-agent LangGraph architecture",
    version="8.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # Restricted to local origins only — never wildcard in production.
    # VS Code extension host (Node.js) does not send an Origin header for
    # WebSocket connections, so this restriction only affects webview fetch calls.
    allow_origins=[
        "http://localhost:8765",
        "http://localhost:8000",
        "http://127.0.0.1:8765",
        "http://127.0.0.1:8000",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
from auth import auth_router, authenticate_ws, get_current_user  # noqa: E402
from api.chat_router        import chat_router          # noqa: E402
from api.git_router         import git_router           # noqa: E402
from api.ci_router          import ci_router            # noqa: E402
from api.diagnostics_router import diagnostics_router, feedback_router  # noqa: E402
from api.code_actions_router import code_actions_router  # noqa: E402
from api.mcp_router         import mcp_router            # noqa: E402
from api.hardening_router   import hardening_router      # noqa: E402
from api.user_router        import user_router, github_callback_router  # noqa: E402
from api.history_router          import history_router                   # noqa: E402
from api.settings_router         import settings_router                  # noqa: E402
from api.extension_chat_router   import extension_chat_router            # noqa: E402
from api.issue_actions_router    import issue_actions_router              # noqa: E402

# Authentication endpoints — always public (login lives here).
app.include_router(auth_router,             prefix="/api")
# GitHub OAuth callback is also public — GitHub redirects here without a JWT.
app.include_router(github_callback_router,  prefix="/api")

# Everything else requires a valid access token (bypassed when AUTH_REQUIRED=false).
_auth = [Depends(get_current_user)]
app.include_router(chat_router,        prefix="/api", dependencies=_auth)
app.include_router(git_router,         prefix="/api", dependencies=_auth)
app.include_router(ci_router,          prefix="/api", dependencies=_auth)
app.include_router(diagnostics_router, prefix="/api", dependencies=_auth)
app.include_router(feedback_router,    prefix="/api", dependencies=_auth)
app.include_router(code_actions_router, prefix="/api", dependencies=_auth)
app.include_router(hardening_router,   prefix="/api", dependencies=_auth)
app.include_router(user_router,        prefix="/api", dependencies=_auth)
app.include_router(history_router,     prefix="/api", dependencies=_auth)
app.include_router(settings_router,         prefix="/api", dependencies=_auth)
app.include_router(extension_chat_router,   prefix="/api", dependencies=_auth)
app.include_router(issue_actions_router,    prefix="/api", dependencies=_auth)
app.include_router(mcp_router,              dependencies=_auth)


# ══════════════════════════════════════════════════════════════════════════════
# Health
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
async def health():
    services_ready = {
        "rag_system":       "rag_system"       in _watch_services,
        "project_indexer":  "project_indexer"  in _watch_services,
        "dep_graph":        "dep_graph"         in _watch_services,
        "watch_active":     len(_watch_active) > 0,
    }

    redis_ok = False
    try:
        from services.mcp_redis_service import get_mcp_redis
        get_mcp_redis().ping()
        redis_ok = True
    except Exception:
        pass

    services_ready["redis"] = redis_ok
    ready = services_ready["rag_system"]

    return HealthResponse(
        status="ready" if ready else "initializing",
        version="8.0.0",
        services={
            **services_ready,
            "uptime_seconds": round(time.time() - _server_start_time, 1),
            "watch_projects": list(_watch_active.keys()),
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# Analyze File (WatchGraph one-shot)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/analyze/file", response_model=AnalysisResultResponse, dependencies=[Depends(get_current_user)])
async def analyze_file(req: AnalyzeFileRequest):
    """
    Analyse un fichier unique via le WatchGraph (pipeline complet).
    Remplace l'ancien endpoint basé sur le legacy Orchestrator.
    """
    file_path = Path(req.file_path)
    if not file_path.exists():
        raise HTTPException(404, f"Fichier introuvable : {file_path}")

    project_path = Path(req.project_path).resolve() if req.project_path else file_path.parent

    try:
        from langchain_agents.graphs.watch_graph import invoke_watch
        _svc = _get_project_services(project_path)
        result = await asyncio.to_thread(
            invoke_watch,
            file_path=str(file_path),
            project_path=str(project_path),
            project_indexer=_svc.get("project_indexer"),
            extractor=_svc.get("extractor"),
            rag_system=_svc.get("rag_system"),
            dep_graph=_svc.get("dep_graph"),
        )
    except Exception as e:
        logger.exception("analyze_file error: %s", e)
        raise HTTPException(500, f"Erreur WatchGraph : {e}")

    language = result.get("language", "unknown")
    strategy = result.get("strategy", "block_fix")
    elapsed  = result.get("stats", {}).get("elapsed", 0.0)

    # Map structured issues from node_emit_ws_events (v2.0) → IssueDiagnostic
    raw_issues = result.get("issues") or []
    issues = [
        IssueDiagnostic(
            severity=i.get("severity", "warning"),
            message=i.get("message", i.get("title", "")),
            line=i.get("line"),
            column=i.get("column"),
            source="code-auditor",
            code_snippet="",
            suggestion=i.get("suggestion", ""),
        )
        for i in raw_issues
    ]

    # Extract fixes from ws_events (built by node_emit_ws_events v2.0)
    # Bug fix: was hardcoded to fixes=[] — fixes are now included in the response.
    ws_events = result.get("ws_events") or []
    ar_event  = next((e for e in ws_events if e.get("type") == "analysis_result"), {})
    raw_fixes = ar_event.get("fixes") or []
    fixes = [
        FixSuggestion(
            location=str(f.get("title", f.get("file", ""))),
            current_code=str(f.get("current_code", ""))[:500],
            fixed_code=str(f.get("fixed_code", ""))[:500],
            explanation=str(f.get("explanation", "")),
        )
        for f in raw_fixes[:5]
    ]

    return AnalysisResultResponse(
        file_path=str(file_path),
        language=language,
        score=int(result.get("change_info", {}).get("score", 0)),
        strategy=strategy,
        issues=issues,
        fixes=fixes,
        elapsed_seconds=elapsed,
        rag_docs_used=len(result.get("rag_docs", [])),
        raw_analysis="",  # omit raw LLM text — use WS events instead
    )


# ══════════════════════════════════════════════════════════════════════════════
# Analyze Project
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/analyze/project", response_model=ProjectAnalysisResponse, dependencies=[Depends(get_current_user)])
async def analyze_project(req: AnalyzeProjectRequest):
    """Analyse architecturale d'un projet complet."""
    project_path = Path(req.project_path)
    if not project_path.exists():
        raise HTTPException(404, f"Projet introuvable : {project_path}")

    try:
        from core.project_analyzer import project_analyzer
        result = await asyncio.to_thread(
            project_analyzer.analyze_full_project,
            project_path,
            req.max_files,
        )
    except Exception as e:
        logger.exception("analyze_project error")
        raise HTTPException(500, f"Erreur : {e}")

    structure = result.get("structure_analysis", {})
    return ProjectAnalysisResponse(
        project_path=str(project_path),
        files_analyzed=len(result.get("file_analyses", {})),
        entry_points=structure.get("entry_points", []),
        circular_dependencies=structure.get("circular_dependencies", []),
        orphaned_modules=structure.get("orphaned_modules", []),
        conflicts=result.get("conflicts", []),
        refactoring_plan=result.get("refactoring_plan", ""),
    )


@app.post("/tests/scan-project", response_model=ScanProjectTestsResponse, dependencies=[Depends(get_current_user)])
async def scan_project_tests(req: ScanProjectTestsRequest):
    """
    Balaie tout le projet pour l'état de couverture de tests (testés + gaps).
    Utilisé par l'extension à l'activation pour peupler le panel Tests avec une
    image complète, indépendamment de Watch (qui ne détecte les gaps qu'au fil
    des sauvegardes).
    """
    project_path = Path(req.project_path)
    if not project_path.exists():
        raise HTTPException(404, f"Projet introuvable : {project_path}")

    from services.test_gap_scanner import scan_project_test_gaps
    results = await asyncio.to_thread(scan_project_test_gaps, project_path, req.max_files)

    return ScanProjectTestsResponse(project_path=str(project_path), files=results)


# ══════════════════════════════════════════════════════════════════════════════
# Watch Mode — WatchGraph
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/watch/start", dependencies=[Depends(get_current_user)])
async def watch_start(req: WatchStartRequest):
    """
    Démarre la surveillance en temps réel via WatchGraph.
    Les événements sont broadcastés via WebSocket.
    """
    project_path = Path(req.project_path).resolve()
    proj_key     = str(project_path)

    if not project_path.exists():
        raise HTTPException(404, f"Projet introuvable : {project_path}")

    if proj_key in _watch_active:
        return {"status": "already_running", "project_path": proj_key}

    from langchain_agents.graphs.watch_graph import invoke_watch
    from watchers.file_watcher import FileWatcher

    # Build/scope the dependency graph + indexer to THIS project so the watch
    # never reports backend files as dependents (see _get_project_services).
    proj_services = _get_project_services(project_path)

    print_lock   = threading.Lock()
    file_counter = {"analyzed": 0, "skipped": 0, "skipped_hash": 0, "skipped_minor": 0}
    pending_timers: Dict[str, Any] = {}
    pending_lock = threading.Lock()
    COALESCE_DELAY = 0.8

    def _debounced(fp: Path):
        with pending_lock:
            pending_timers.pop(str(fp), None)

        # ── Phase A : signal immédiat avant le pipeline LLM (< 1ms) ──────────
        _loop = _main_loop
        if _loop and _loop.is_running():
            asyncio.run_coroutine_threadsafe(
                _ws_manager.broadcast({
                    "type":      "analysis_started",
                    "file_path": str(fp),
                    "file_name": fp.name,
                    "language":  _ext_to_language(fp),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }),
                _loop,
            )

        # ── Real-time push callback ──────────────────────────────────────────
        # Passed into the graph so the PRIMARY result is broadcast the instant it
        # is ready (before the dependent cascade), and each dependent streams in
        # as it finishes — instead of batching everything after the whole 297s
        # pipeline. Thread-safe: schedules onto the captured main loop.
        def _ws_push(event: Dict[str, Any]):
            _loop2 = _main_loop
            if _loop2 is not None and _loop2.is_running():
                asyncio.run_coroutine_threadsafe(
                    _ws_manager.broadcast(event), _loop2
                )

        # ── Phase B : pipeline complet (hash → AST → RAG → LLM → learn) ─────
        result = invoke_watch(
            file_path=str(fp),
            project_path=proj_key,
            project_indexer=proj_services.get("project_indexer"),
            extractor=proj_services.get("extractor"),
            rag_system=proj_services.get("rag_system"),
            dep_graph=proj_services.get("dep_graph"),
            print_lock=print_lock,
            file_counter=file_counter,
            ws_broadcast=_ws_push,
        )

        # ── Broadcast all WebSocket events built by node_emit_ws_events ────────
        ws_events = result.get("ws_events") or []

        # Use the pre-captured main event loop — safe to call from any thread
        loop = _main_loop
        if loop is None or not loop.is_running():
            logger.warning("_debounced: main event loop unavailable, WS broadcast skipped")
            return

        # When the graph already streamed events in real time via _ws_push, only
        # persist the snapshot for TreeView init — do NOT re-broadcast (duplicates).
        if result.get("ws_broadcasted"):
            if ws_events:
                _watch_last_events[proj_key] = {
                    "file_path": str(fp),
                    "timestamp": time.time(),
                    "events":    ws_events,
                }
            if result.get("skip_reason"):
                file_counter["skipped"] += 1
            return

        if ws_events:
            # Store latest snapshot for TreeView initialisation
            _watch_last_events[proj_key] = {
                "file_path": str(fp),
                "timestamp": time.time(),
                "events":    ws_events,
            }
            for event in ws_events:
                asyncio.run_coroutine_threadsafe(
                    _ws_manager.broadcast(event),
                    loop,
                )
        else:
            # Fallback: always send at least a basic watch_event
            analysis = result.get("analysis", {})
            raw_text  = analysis.get("analysis", "") if isinstance(analysis, dict) else ""
            asyncio.run_coroutine_threadsafe(
                _ws_manager.broadcast({
                    "type":        "watch_event",
                    "file_path":   str(fp),
                    "file":        fp.name,
                    "strategy":    result.get("strategy", "block_fix"),
                    "language":    result.get("language", "?"),
                    "skip_reason": result.get("skip_reason"),
                    "raw_analysis": raw_text[:2000],
                }),
                loop,
            )

        # diagnostics_update removed (schema v2.0):
        # The analysis_result event (schema_version=2.0) already includes issues[]
        # with full diagnostic data (line, column, end_line, severity, rule, suggestion).
        # WatchController.ts routes analysis_result → applyDiagnostics() directly,
        # making this separate broadcast redundant and costly (double WS message).

        if result.get("skip_reason"):
            file_counter["skipped"] += 1

    def on_change(fp: Path, deleted: bool = False):
        logger.info("[WATCH DEBUG] on_change called: %s deleted=%s", fp, deleted)

        if deleted:
            loop = _main_loop
            if loop is None or not loop.is_running():
                logger.warning("on_change deleted: main event loop unavailable, WS broadcast skipped")
                return

            asyncio.run_coroutine_threadsafe(
                _ws_manager.broadcast({
                    "type": "file_deleted",
                    "file": fp.name,
                    "file_path": str(fp),
                }),
                loop,
            )
            return

        fp_key = str(fp)

        with pending_lock:
            old = pending_timers.get(fp_key)
            if old:
                old.cancel()

            t = threading.Timer(COALESCE_DELAY, _debounced, args=[fp])
            t.daemon = True
            pending_timers[fp_key] = t
            t.start()

    try:
        watcher = FileWatcher(project_path=project_path, callback=on_change)

        watch_method = getattr(watcher, "watch", None) or getattr(watcher, "start", None)
        if watch_method is None:
            raise RuntimeError("FileWatcher has no watch() or start() method")

        t = threading.Thread(target=watch_method, daemon=True)
        t.start()

        _watch_active[proj_key] = {
            "watcher":  watcher,
            "thread":   t,
            "counter":  file_counter,
            "timers":   pending_timers,
            "lock":     pending_lock,
            "started":  time.time(),
        }

        logger.info("Watch started for project: %s", proj_key)
        return {"status": "started", "project_path": proj_key}

    except Exception as e:
        logger.exception("watch_start failed for project %s: %s", proj_key, e)
        raise HTTPException(500, f"watch_start failed: {type(e).__name__}: {e}")


@app.post("/watch/stop", dependencies=[Depends(get_current_user)])
async def watch_stop(req: WatchStartRequest):
    """Arrête la surveillance d'un projet."""
    proj_key = str(Path(req.project_path).resolve())
    ctx = _watch_active.pop(proj_key, None)
    if not ctx:
        return {"status": "not_running"}

    try:
        ctx["watcher"].stop()
    except Exception:
        pass
    for t in ctx.get("timers", {}).values():
        try:
            t.cancel()
        except Exception:
            pass

    return {"status": "stopped", "files_analyzed": ctx["counter"].get("analyzed", 0)}


@app.get("/watch/status", response_model=WatchStatusResponse, dependencies=[Depends(get_current_user)])
async def watch_status():
    """État de tous les watchers actifs."""
    if not _watch_active:
        return WatchStatusResponse(is_running=False)

    total = sum(c["counter"].get("analyzed", 0) for c in _watch_active.values())
    first_proj = next(iter(_watch_active))

    return WatchStatusResponse(
        is_running=True,
        project_path=first_proj,
        files_processed=total,
        stats={
            "projects":  list(_watch_active.keys()),
            "counters":  {p: c["counter"] for p, c in _watch_active.items()},
        },
    )


@app.get("/watch/events/latest", dependencies=[Depends(get_current_user)])
async def watch_events_latest(project_path: str = ""):
    """
    Retourne le dernier snapshot d'événements WS pour un projet.
    Utilisé par le plugin VS Code pour initialiser la TreeView au démarrage.

    Query param: ?project_path=C:/path/to/project  (optionnel)
    """
    if project_path:
        key = str(Path(project_path).resolve())
        snapshot = _watch_last_events.get(key)
        if not snapshot:
            return {"events": [], "project_path": key, "has_data": False}
        return {**snapshot, "has_data": True}

    # No project specified → return latest across all projects
    if not _watch_last_events:
        return {"events": [], "has_data": False}

    latest = max(_watch_last_events.values(), key=lambda s: s["timestamp"])
    return {**latest, "has_data": True}


# ══════════════════════════════════════════════════════════════════════════════
# Generate Tests
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/generate-tests", response_model=GenerateTestsResponse, dependencies=[Depends(get_current_user)])
async def generate_tests(req: GenerateTestsRequest):
    """Génère des tests unitaires via TestGeneratorAgent (RAG)."""
    file_path = Path(req.file_path)

    if not file_path.exists():
        raise HTTPException(404, f"Fichier introuvable : {file_path}")

    try:
        from agents.test_generator_agent import TestGeneratorAgent
        from services.knowledge_graph import knowledge_graph as _kg

        project_path = Path(req.project_path).resolve() if req.project_path else file_path.parent

        # Injecter les composants RAG initialisés au démarrage du serveur
        agent = TestGeneratorAgent(
            project_path=project_path,
            test_kb=_watch_services.get("test_kb"),
            project_code_indexer=_watch_services.get("project_indexer"),
            knowledge_graph=_kg if getattr(_kg, "_built", False) else None,
        )

        result = await asyncio.to_thread(
            agent.generate_for_file,
            file_path,
            req.write,
            req.incremental,
        )

        if not isinstance(result, dict):
            logger.warning("generate_tests returned non-dict result: %r", result)
            result = {
                "error": "Le générateur de tests a retourné une réponse invalide.",
                "test_file": "",
                "test_code": "",
                "framework": "",
                "rag_docs_used": 0,
                "validated": False,
            }

        test_file = result.get("test_file")
        test_code = result.get("test_code")
        framework = result.get("framework")
        rag_docs_used = result.get("rag_docs_used")
        validated = result.get("validated")
        error = result.get("error")
        run_success = result.get("run_success")
        run_error_summary = result.get("run_error_summary")
        cached = result.get("cached")
        review_notes = result.get("review_notes")

        return GenerateTestsResponse(
            test_file=str(test_file or ""),
            test_code=str(test_code or ""),
            framework=str(framework or ""),
            rag_docs_used=int(rag_docs_used or 0),
            validated=bool(validated or False),
            incremental=bool(result.get("incremental") or False),
            error=str(error or ""),
            run_success=run_success if isinstance(run_success, bool) else None,
            run_error_summary=str(run_error_summary) if run_error_summary else None,
            cached=bool(cached or False),
            review_notes=review_notes if isinstance(review_notes, dict) else None,
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("generate_tests error")

        raise HTTPException(
            status_code=500,
            detail=f"Erreur génération tests : {type(e).__name__}: {e}",
        )

# ══════════════════════════════════════════════════════════════════════════════
# Run Tests (existing file — no LLM, no regeneration)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/run-tests", response_model=RunTestsResponse, dependencies=[Depends(get_current_user)])
async def run_tests(req: RunTestsRequest):
    """Exécute un fichier de test existant sans régénérer (pytest/jest/mvn/gradle)."""
    test_path = Path(req.test_file)
    if not test_path.exists():
        raise HTTPException(404, f"Fichier de test introuvable : {test_path}")

    try:
        from services.test_runner import TestRunner

        project_path = Path(req.project_path).resolve() if req.project_path else test_path.parent
        runner = TestRunner(project_path=project_path)

        result = await asyncio.to_thread(runner.run, test_path, req.language)

        return RunTestsResponse(
            test_file=str(test_path),
            run_success=result.success,
            run_error_summary=result.error_summary or None,
            duration_seconds=round(result.duration_seconds, 2),
            error="",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("run_tests error")
        raise HTTPException(status_code=500, detail=f"Erreur exécution tests : {type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Knowledge Base — human approval of learned rules
# ══════════════════════════════════════════════════════════════════════════════

class KBRuleAction(BaseModel):
    language: str
    pattern: str


@app.post("/api/kb/approve", dependencies=[Depends(get_current_user)])
async def kb_approve(req: KBRuleAction):
    """Promote a pending learned rule into the active Knowledge Base."""
    from langchain_agents.agents.lc_learning_agent import lc_learning_agent
    ok = lc_learning_agent.approve_pending(req.language, req.pattern)
    if not ok:
        raise HTTPException(404, f"Règle en attente introuvable : {req.language}/{req.pattern}")
    return {"status": "approved", "language": req.language, "pattern": req.pattern}


@app.post("/api/kb/reject", dependencies=[Depends(get_current_user)])
async def kb_reject(req: KBRuleAction):
    """Discard a pending learned rule (developer rejected it)."""
    from langchain_agents.agents.lc_learning_agent import lc_learning_agent
    lc_learning_agent.reject_pending(req.language, req.pattern)
    return {"status": "rejected", "language": req.language, "pattern": req.pattern}


@app.get("/api/kb/pending", dependencies=[Depends(get_current_user)])
async def kb_list_pending():
    """List all KB rules staged for human approval (pending/ directories)."""
    rules = []
    kb_root = config.KNOWLEDGE_BASE_DIR
    if not kb_root.exists():
        return {"rules": []}
    for lang_dir in sorted(kb_root.iterdir()):
        if not lang_dir.is_dir():
            continue
        pending_dir = lang_dir / "pending"
        if not pending_dir.exists():
            continue
        for rule_file in sorted(pending_dir.glob("*.md")):
            try:
                content = rule_file.read_text(encoding="utf-8")
            except Exception:
                content = ""
            severity = "MEDIUM"
            for s in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
                if s in content.upper():
                    severity = s
                    break
            rules.append({
                "language": lang_dir.name,
                "pattern":  rule_file.stem,
                "severity": severity,
                "preview":  content[:300],
            })
    return {"rules": rules}


# ══════════════════════════════════════════════════════════════════════════════
# Stats
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/stats", dependencies=[Depends(get_current_user)])
async def get_stats():
    total_analyzed = sum(c["counter"].get("analyzed", 0) for c in _watch_active.values())
    total_skipped  = sum(c["counter"].get("skipped",  0) for c in _watch_active.values())
    return {
        "server": {
            "uptime_seconds": round(time.time() - _server_start_time, 1),
            "version":        "8.0.0",
            "ws_clients":     _ws_manager.active_count,
            "watch_projects": len(_watch_active),
        },
        "watch": {
            "total_analyzed": total_analyzed,
            "total_skipped":  total_skipped,
            "projects":       list(_watch_active.keys()),
        },
        "services": {k: True for k in _watch_services},
    }



# ══════════════════════════════════════════════════════════════════════════════
# Legacy Compat — /results stub
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/results", dependencies=[Depends(get_current_user)])
async def get_results_legacy():
    """
    Legacy endpoint kept for plugin backward compatibility.
    The new plugin should use GET /watch/events/latest instead.
    Returns an empty dict so the plugin doesn't get a 404.
    """
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket
# ══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str = Query(default="")):
    """
    WebSocket pour events temps réel (watch + CI + git alerts).

    Client → serveur :
      {"type": "ping"}
      {"type": "analyze_file", "file_path": "..."}

    Serveur → client :
      {"type": "watch_event", "file": "...", "strategy": "..."}
      {"type": "file_deleted", "file": "..."}
      {"type": "ci_event", ...}
    """
    # Authenticate the WebSocket handshake (?token=<access JWT>).
    # Loopback bypass: the extension owns LOCAL and the backend binds to 127.0.0.1,
    # so a WS from the same machine is the developer's own — allow it WITHOUT a token
    # so real-time watch events always reach the local plugin even when the user has
    # not signed into the extension. Remote clients (cloud dashboard) still need a JWT.
    client_host = websocket.client.host if websocket.client else ""
    is_loopback = client_host in ("127.0.0.1", "::1", "localhost")
    if not is_loopback and authenticate_ws(token) is None:
        await websocket.close(code=1008)
        return

    await _ws_manager.connect(websocket)
    try:
        await _ws_manager.send_to(websocket, {
            "type":    "connected",
            "version": "8.0.0",
            "clients": _ws_manager.active_count,
        })

        while True:
            data     = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "ping":
                await _ws_manager.send_to(websocket, {"type": "pong"})

            elif msg_type == "analyze_file":
                fp = data.get("file_path", "")
                if fp:
                    # ── Phase A : feedback immédiat avant le pipeline LLM ─────
                    await _ws_manager.send_to(websocket, {
                        "type":      "analysis_started",
                        "file_path": fp,
                        "file_name": Path(fp).name,
                        "language":  _ext_to_language(Path(fp)),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    })
                    try:
                        from langchain_agents.graphs.watch_graph import invoke_watch
                        result = await asyncio.to_thread(
                            invoke_watch,
                            file_path=fp,
                            project_path=data.get("project_path", "."),
                            project_indexer=_watch_services.get("project_indexer"),
                            extractor=_watch_services.get("extractor"),
                            rag_system=_watch_services.get("rag_system"),
                            dep_graph=_watch_services.get("dep_graph"),
                        )
                        # ── Phase B : ws_events complets schema v2.0 ─────────
                        ws_events = result.get("ws_events") or []
                        if ws_events:
                            for event in ws_events:
                                await _ws_manager.send_to(websocket, event)
                        else:
                            await _ws_manager.send_to(websocket, {
                                "type":     "analysis_result",
                                "file":     Path(fp).name,
                                "strategy": result.get("strategy", "?"),
                                "skipped":  bool(result.get("skip_reason")),
                            })
                    except Exception as e:
                        await _ws_manager.send_to(websocket, {"type": "error", "detail": str(e)})
                else:
                    await _ws_manager.send_to(websocket, {"type": "error", "detail": "file_path requis"})

            else:
                await _ws_manager.send_to(websocket, {"type": "error", "detail": f"Type inconnu: {msg_type}"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug("WebSocket error: %s", e)
    finally:
        await _ws_manager.disconnect(websocket)


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Code Auditor API Server v8")
    parser.add_argument("--host",    default="127.0.0.1")
    parser.add_argument("--port",    type=int, default=8765)
    parser.add_argument("--project", default=".")
    parser.add_argument("--reload",  action="store_true")
    args = parser.parse_args()

    global _default_project
    _default_project = Path(args.project).resolve()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
