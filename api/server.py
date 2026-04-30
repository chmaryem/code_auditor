
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Models Pydantic pour API
class AnalyzeRequest(BaseModel):
    file_path: str
    project_path: str

class AnalysisResponse(BaseModel):
    file: str
    score: int
    issues: list[dict[str, Any]]
    fixes: list[dict[str, Any]]
    elapsed_ms: int

class HealthResponse(BaseModel):
    status: str
    version: str

# Global orchestrator instance
_orchestrator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize/shutdown orchestrator."""
    global _orchestrator
    from core.orchestrator import Orchestrator
    
    # Lazy init - à remplacer par le vrai projet
    _orchestrator = Orchestrator(Path("."))
    _orchestrator.initialize()
    
    yield
    
    _orchestrator.stop()

app = FastAPI(
    title="Code Auditor API",
    version="6.2.0",
    lifespan=lifespan,
)

# CORS pour extension VS Code
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En prod: spécifier l'origin VS Code
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok", "version": "6.2.0"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(req: AnalyzeRequest):
    """Analyser un fichier et retourner résultat JSON."""
    if not _orchestrator:
        raise HTTPException(503, "Orchestrator not ready")
    
    from core.events import file_changed_event
    from output.json_renderer import JSONRenderer
    
    file_path = Path(req.file_path)
    
    # Créer event et attendre résultat
    event = file_changed_event(file_path)
    
    # TODO: Hook pour capturer le résultat
    # Pour l'instant, mock response
    renderer = JSONRenderer()
    
    return {
        "file": file_path.name,
        "score": 45,
        "issues": [],
        "fixes": [],
        "elapsed_ms": 1200,
    }

@app.post("/watch/start")
async def start_watch(project_path: str):
    """Démarrer surveillance projet."""
    # Implementation...
    return {"status": "watching", "project": project_path}

@app.post("/watch/stop")
async def stop_watch():
    """Arrêter surveillance."""
    return {"status": "stopped"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
