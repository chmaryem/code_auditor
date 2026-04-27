import os
import logging
from pathlib import Path
from typing import List

from pydantic import BaseModel, field_validator, model_validator

logger = logging.getLogger(__name__)


def _detect_optimal_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("Embeddings device: CUDA")
            return "cuda"
        if torch.backends.mps.is_available():
            logger.info("Embeddings device: MPS (Apple Silicon)")
            return "mps"
    except ImportError:
        pass
    logger.info("Embeddings device: CPU")
    return "cpu"



class APIConfig(BaseModel):

    provider:    str   = "openrouter"
    # ── OpenRouter (primary) ──────────────────────────────────────────────────
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model:   str = os.getenv("OPENROUTER_MODEL", "minimax/minimax-m2.5:free")
    # Rotation round-robin entre modèles gratuits (contourne le rate-limit)
   
    # ── Google Gemini (fallback) ──────────────────────────────────────────────
    gemini_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model:   str = os.getenv("GEMINI_MODEL",   "gemini-2.5-flash")
    # ── Commun ────────────────────────────────────────────────────────────────
    temperature: float = 0.0
    max_tokens:  int   = 16384

class RAGConfig(BaseModel):
    embedding_model:     str   = "jinaai/jina-embeddings-v2-base-code"
    embedding_dimension: int   = 768
    embedding_device:    str   = None
    vector_store:        str   = "chromadb"
    distance_metric:     str   = "cosine"
    chunk_size:          int   = 800
    chunk_overlap:       int   = 150
    top_k:               int   = 8
    relevance_threshold: float = 0.45

    @field_validator("embedding_device", mode="before")
    @classmethod
    def auto_detect_device(cls, v: str) -> str:
        return v if v else _detect_optimal_device()

    @model_validator(mode="after")
    def warn_if_wrong_dimension(self) -> "RAGConfig":
        model = self.embedding_model.lower()
        dim   = self.embedding_dimension
        known = {
            "jina-embeddings-v2-base-code":  768,
            "jina-embeddings-v2-small-code": 512,
            "all-minilm-l6-v2":              384,
            "all-minilm-l12-v2":             384,
            "text-embedding-ada-002":        1536,
        }
        for key, expected in known.items():
            if key in model and dim != expected:
                logger.warning(
                    "RAGConfig: modèle '%s' attend %d dims, embedding_dimension=%d",
                    self.embedding_model, expected, dim,
                )
        return self


class AnalysisConfig(BaseModel):
    supported_languages: List[str] = ["python", "javascript", "typescript", "java"]
    max_file_size_mb:    int        = 5
    max_code_chars:      int        = 10_000
    max_knowledge_chars: int        = 2_000
    max_context_chars:   int        = 1_500
    exclude_patterns:    List[str]  = [
        "**/node_modules/**", "**/__pycache__/**", "**/venv/**",
        "**/dist/**", "**/build/**", "**/.git/**", "**/target/**",
    ]
    analysis_depth: str = "medium"


class WatcherConfig(BaseModel):
    enabled:            bool      = True
    debounce_seconds:   float     = 4.0
    analyze_impacted:   bool      = True
    max_impacted_files: int       = 5
    watched_extensions: List[str] = [".py", ".js", ".jsx", ".ts", ".tsx", ".java"]
    excluded_dirs:      List[str] = [
        "node_modules", "__pycache__", "venv", ".git",
        "dist", "build", ".pytest_cache", ".mypy_cache",
        ".vscode", ".idea", "target", "out",
    ]


class Config:
    BASE_DIR           = Path(__file__).parent
    DATA_DIR           = BASE_DIR / "data"
    KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
    VECTOR_STORE_DIR   = DATA_DIR / "vector_store"
    CACHE_DIR          = DATA_DIR / "cache"
    KG_PATH            = DATA_DIR / "knowledge_graph.json"

    for _d in [DATA_DIR, KNOWLEDGE_BASE_DIR, VECTOR_STORE_DIR, CACHE_DIR]:
        _d.mkdir(parents=True, exist_ok=True)

    api      = APIConfig()
    rag      = RAGConfig()
    analysis = AnalysisConfig()
    watcher  = WatcherConfig()

    HOST  = os.getenv("SERVER_HOST", "127.0.0.1")
    PORT  = int(os.getenv("SERVER_PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    API_PREFIX        = "/api/v1"
    CHROMA_COLLECTION = "code_kb_jina_v2"


config = Config()