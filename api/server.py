"""
api/server.py — Serveur FastAPI pour Code Auditor.

Ce serveur expose toute la puissance du moteur d'analyse via :
  - REST  : endpoints synchrones pour analyse one-shot
  - WebSocket : streaming temps réel pour le watch mode

L'extension VS Code lance ce serveur automatiquement en subprocess
et communique avec lui via HTTP (REST) et WS (WebSocket).

Démarrage :
    python -m api.server                    # port par défaut : 8765
    python -m api.server --port 9000        # port custom
    python -m api.server --project /path    # projet par défaut
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.models import (
    AnalyzeFileRequest,
    AnalysisResultResponse,
    AnalyzeProjectRequest,
    ProjectAnalysisResponse,
    WatchStartRequest,
    WatchStatusResponse,
    GitStatusRequest,
    GitBranchRequest,
    GenerateTestsRequest,
    GenerateTestsResponse,
    HealthResponse,
    WSEvent,
)
from api.websocket_manager import ConnectionManager

logger = logging.getLogger(__name__)

# ── État global du serveur ────────────────────────────────────────────────────
# L'orchestrateur et le watcher sont partagés entre les requêtes.
# Ils sont initialisés au démarrage dans le lifespan.

_orchestrator = None
_file_watcher = None
_ws_manager = ConnectionManager()
_server_start_time: float = 0.0
_default_project: Path = Path(".")


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialise l'orchestrateur au démarrage du serveur.
    L'extension VS Code peut ensuite envoyer des requêtes immédiatement.
    """
    global _orchestrator, _server_start_time

    _server_start_time = time.time()
    logger.info("Démarrage du serveur Code Auditor API...")

    # Initialisation de l'orchestrateur dans un thread séparé
    # (l'initialisation est lourde : embeddings, ChromaDB, Redis MCP)
    from core.orchestrator import Orchestrator

    def _on_result_callback(result: dict):
        """Callback appelé quand le watch mode produit un résultat."""
        asyncio.run_coroutine_threadsafe(
            _ws_manager.broadcast({
                "type": "analysis_result",
                "data": result,
            }),
            asyncio.get_event_loop(),
        )

    _orchestrator = Orchestrator(
        project_path=_default_project,
        on_result=_on_result_callback,
    )

    # Initialisation dans un thread (bloquant, ~10-30s)
    init_thread = threading.Thread(target=_orchestrator.initialize, daemon=True)
    init_thread.start()

    logger.info("Orchestrateur en cours d'initialisation (background)...")

    yield

    # Shutdown
    logger.info("Arrêt du serveur...")
    if _file_watcher:
        _file_watcher.stop()
    if _orchestrator:
        _orchestrator.stop()


# ── App FastAPI ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Code Auditor API",
    description="API REST + WebSocket pour l'analyse intelligente de code",
    version="7.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # VS Code extension
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Chat Agent router (Phase 1 + Phase 2) ─────────────────────────────────────
from api.chat_router import chat_router          # noqa: E402
app.include_router(chat_router, prefix="/api")


# ══════════════════════════════════════════════════════════════════════════════
# REST Endpoints
# ══════════════════════════════════════════════════════════════════════════════


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check — appelé par l'extension VS Code pour savoir
    quand le serveur est prêt à recevoir des requêtes.

    Services vérifiés :
      - orchestrator : pipeline d'analyse initialisé
      - redis        : cache MCP connecté
      - chromadb     : vector store chargé
    """
    orch_ready = _orchestrator is not None and _orchestrator._is_running or (
        hasattr(_orchestrator, "_analysis_agent") and _orchestrator._analysis_agent
    )

    services = {
        "orchestrator": bool(orch_ready),
        "redis": False,
        "chromadb": False,
    }

    # Vérifier Redis
    if _orchestrator and _orchestrator._cache:
        try:
            services["redis"] = True
        except Exception:
            pass

    # Vérifier ChromaDB
    if orch_ready:
        try:
            services["chromadb"] = True
        except Exception:
            pass

    uptime = round(time.time() - _server_start_time, 1) if _server_start_time else 0

    return HealthResponse(
        status="ready" if orch_ready else "initializing",
        version="7.0.0",
        services={**services, "uptime_seconds": uptime},
    )


# ── Analyse de fichier ───────────────────────────────────────────────────────

@app.post("/analyze/file", response_model=AnalysisResultResponse)
async def analyze_file(req: AnalyzeFileRequest):
    """
    Analyse un fichier unique avec le pipeline complet :
    Parse AST → RAG retrieval → LLM analysis → JSON structuré.

    C'est l'équivalent API de `python main.py file <path>`.
    """
    if not _orchestrator:
        raise HTTPException(503, "Serveur en cours d'initialisation")

    file_path = Path(req.file_path)
    if not file_path.exists():
        raise HTTPException(404, f"Fichier introuvable : {file_path}")

    if not file_path.is_file():
        raise HTTPException(400, f"Ce n'est pas un fichier : {file_path}")

    # Exécuter l'analyse dans un thread (le pipeline LLM est bloquant)
    try:
        result = await asyncio.to_thread(
            _orchestrator.analyze_single, file_path
        )
    except Exception as e:
        logger.exception("Erreur analyse %s", file_path)
        raise HTTPException(500, f"Erreur d'analyse : {e}")

    return result


# ── Analyse de projet ─────────────────────────────────────────────────────────

@app.post("/analyze/project", response_model=ProjectAnalysisResponse)
async def analyze_project(req: AnalyzeProjectRequest):
    """
    Analyse architecturale d'un projet complet.
    Identifie les fichiers critiques, les dépendances circulaires,
    les modules orphelins, et génère un plan de refactoring.

    Équivalent API de `python main.py project <path>`.
    """
    if not _orchestrator:
        raise HTTPException(503, "Serveur en cours d'initialisation")

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
        logger.exception("Erreur analyse projet %s", project_path)
        raise HTTPException(500, f"Erreur : {e}")

    # Convertir le résultat interne en format API
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


# ── Watch Mode ────────────────────────────────────────────────────────────────

@app.post("/watch/start")
async def watch_start(req: WatchStartRequest):
    """
    Démarre la surveillance en temps réel d'un projet.
    Les résultats sont envoyés via WebSocket à tous les clients connectés.

    Équivalent API de `python main.py watch <path>`.
    """
    global _file_watcher

    if not _orchestrator:
        raise HTTPException(503, "Serveur en cours d'initialisation")

    if _file_watcher and _file_watcher.is_running:
        return {"status": "already_running", "project_path": str(_file_watcher.project_path)}

    project_path = Path(req.project_path)
    if not project_path.exists():
        raise HTTPException(404, f"Projet introuvable : {project_path}")

    from watchers.file_watcher import FileWatcher

    def on_file_change(file_path: Path, deleted: bool = False):
        """Callback du watcher — déclenche l'analyse via l'orchestrateur."""
        if deleted:
            return
        _orchestrator.handle(
            __import__("core.events", fromlist=["file_changed_event"]).file_changed_event(file_path)
        )

    _file_watcher = FileWatcher(
        project_path=project_path,
        callback=on_file_change,
    )

    # Démarrer dans un thread pour ne pas bloquer FastAPI
    threading.Thread(target=_file_watcher.start, daemon=True).start()

    return {"status": "started", "project_path": str(project_path)}


@app.post("/watch/stop")
async def watch_stop():
    """Arrête la surveillance en temps réel."""
    global _file_watcher

    if not _file_watcher or not _file_watcher.is_running:
        return {"status": "not_running"}

    _file_watcher.stop()
    return {"status": "stopped", "files_processed": _file_watcher.files_processed}


@app.get("/watch/status", response_model=WatchStatusResponse)
async def watch_status():
    """Retourne l'état actuel du watcher."""
    if not _file_watcher:
        return WatchStatusResponse(is_running=False)

    return WatchStatusResponse(
        is_running=_file_watcher.is_running,
        project_path=str(_file_watcher.project_path),
        files_processed=_file_watcher.files_processed,
        stats=_orchestrator.get_stats_dict() if _orchestrator else {},
    )


# ── Git Intelligence ──────────────────────────────────────────────────────────

@app.post("/git/status")
async def git_status(req: GitStatusRequest):
    """
    Retourne le statut de session Git (bugs accumulés, score).
    Équivalent API de `python main.py git-status <path>`.
    """
    project_path = Path(req.project_path)
    if not project_path.exists():
        raise HTTPException(404, f"Projet introuvable : {project_path}")

    try:
        from smart_git.git_session_tracker import GitSessionTracker
        tracker = GitSessionTracker(project_path)
        status = await asyncio.to_thread(tracker.get_session_status)
        return status
    except ImportError:
        raise HTTPException(501, "Module smart_git non disponible")
    except Exception as e:
        raise HTTPException(500, f"Erreur git status : {e}")


@app.post("/git/branch")
async def git_branch(req: GitBranchRequest):
    """
    Analyse une branche Git vs sa base et retourne un verdict de merge.
    Équivalent API de `python main.py git-branch <branch> --base <main>`.
    """
    project_path = Path(req.project_path)
    if not project_path.exists():
        raise HTTPException(404, f"Projet introuvable : {project_path}")

    try:
        from smart_git.git_branch_analyzer import BranchAnalyzer
        analyzer = BranchAnalyzer(project_path)
        result = await asyncio.to_thread(
            analyzer.analyze_branch, req.branch, req.base
        )
        return result
    except ImportError:
        raise HTTPException(501, "Module smart_git non disponible")
    except Exception as e:
        raise HTTPException(500, f"Erreur branch analysis : {e}")


# ── Generate Tests ────────────────────────────────────────────────────────────

@app.post("/generate-tests", response_model=GenerateTestsResponse)
async def generate_tests(req: GenerateTestsRequest):
    """
    Génère des tests unitaires pour un fichier source.
    Utilise le pipeline RAG + TestGeneratorAgent.

    Équivalent API de `python main.py generate-tests <path>`.
    """
    file_path = Path(req.file_path)
    if not file_path.exists():
        raise HTTPException(404, f"Fichier introuvable : {file_path}")

    try:
        from agents.test_generator_agent import test_generator_agent
        result = await asyncio.to_thread(
            test_generator_agent.generate_tests,
            file_path,
            Path(req.project_path) if req.project_path else file_path.parent,
            req.write,
        )
        return GenerateTestsResponse(
            test_file=str(result.get("test_file", "")),
            test_code=result.get("test_code", ""),
            framework=result.get("framework", ""),
            rag_docs_used=result.get("rag_docs_used", 0),
            validated=result.get("validated", False),
        )
    except ImportError:
        raise HTTPException(501, "Module test_generator non disponible")
    except Exception as e:
        logger.exception("Erreur generate-tests %s", file_path)
        raise HTTPException(500, f"Erreur : {e}")


# ── Stats ─────────────────────────────────────────────────────────────────────

@app.get("/stats")
async def get_stats():
    """Retourne les statistiques du serveur et du moteur d'analyse."""
    stats = _orchestrator.get_stats_dict() if _orchestrator else {}
    return {
        "server": {
            "uptime_seconds": round(time.time() - _server_start_time, 1),
            "ws_clients": _ws_manager.active_count,
            "watcher_running": _file_watcher.is_running if _file_watcher else False,
        },
        "engine": stats,
        "results_cached": len(_orchestrator.get_all_results()) if _orchestrator else 0,
    }


# ── Résultats stockés ────────────────────────────────────────────────────────

@app.get("/results")
async def get_all_results():
    """Retourne tous les résultats d'analyse stockés en mémoire."""
    if not _orchestrator:
        return {}
    return _orchestrator.get_all_results()


@app.get("/results/{file_path:path}")
async def get_file_result(file_path: str):
    """Retourne le dernier résultat d'analyse pour un fichier spécifique."""
    if not _orchestrator:
        raise HTTPException(503, "Serveur non initialisé")

    result = _orchestrator.get_last_result(Path(file_path))
    if not result:
        raise HTTPException(404, f"Aucun résultat pour : {file_path}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket
# ══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Point de connexion WebSocket pour le streaming temps réel.

    L'extension VS Code se connecte ici pour recevoir :
      - analysis_result : résultats du watch mode en temps réel
      - progress        : progression de l'initialisation
      - error           : erreurs du pipeline

    Le client peut envoyer :
      - {"type": "ping"}                           → pong
      - {"type": "analyze", "file_path": "..."}    → lance une analyse
      - {"type": "subscribe", "events": [...]}     → filtre les événements
    """
    await _ws_manager.connect(websocket)

    try:
        # Envoyer le statut initial
        await _ws_manager.send_to(websocket, {
            "type": "connected",
            "data": {
                "server_version": "7.0.0",
                "ws_clients": _ws_manager.active_count,
            },
        })

        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "ping":
                await _ws_manager.send_to(websocket, {"type": "pong"})

            elif msg_type == "analyze":
                # Analyse à la demande via WebSocket
                file_path = data.get("file_path")
                if file_path and _orchestrator:
                    result = await asyncio.to_thread(
                        _orchestrator.analyze_single, Path(file_path)
                    )
                    await _ws_manager.send_to(websocket, {
                        "type": "analysis_result",
                        "data": result,
                    })
                else:
                    await _ws_manager.send_to(websocket, {
                        "type": "error",
                        "data": {"message": "file_path requis ou serveur non prêt"},
                    })

            else:
                await _ws_manager.send_to(websocket, {
                    "type": "error",
                    "data": {"message": f"Type inconnu: {msg_type}"},
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug("WebSocket erreur: %s", e)
    finally:
        await _ws_manager.disconnect(websocket)



def main():
    """Lance le serveur uvicorn avec les options CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Code Auditor API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Adresse d'écoute")
    parser.add_argument("--port", type=int, default=8765, help="Port d'écoute")
    parser.add_argument("--project", default=".", help="Projet par défaut")
    parser.add_argument("--reload", action="store_true", help="Hot reload (dev)")
    args = parser.parse_args()

    global _default_project
    _default_project = Path(args.project).resolve()

    # Configurer le logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import uvicorn
    uvicorn.run(
        "api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
