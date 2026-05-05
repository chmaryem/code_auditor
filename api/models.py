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
    version: str = "7.0.0"
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

class WSEvent(BaseModel):
    """Message WebSocket envoyé au client (VS Code extension)."""
    type: str = Field(
        ...,
        description="Type d'événement: analysis_result | watch_event | progress | error",
    )
    data: Dict[str, Any] = Field(default_factory=dict)
