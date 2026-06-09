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
 
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model:   str = os.getenv("OPENROUTER_MODEL", "qwen/qwen3-coder:free")

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model:   str = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

    gemini_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model:   str = os.getenv("GEMINI_MODEL",   "gemini-2.0-flash")
   
    temperature: float = 0.0
    max_tokens:  int   = 16384

    # ── P2 · Routage par complexité ──────────────────────────────────────────
    # Le Decision Agent classe chaque message du chat en : fast | context | deep
    # (voir langchain_agents/agents/lc_chat_decision_agent.py). Ce mapping choisit
    # QUEL modèle répond à chaque niveau — sans toucher au reste du code.
    #
    # 👉 TU GARDES LA MAIN : change provider/model ci-dessous (ou via .env) quand
    #    tu auras figé ton choix de modèles. C'est le SEUL endroit à éditer.
    #      provider ∈ {"groq", "openrouter", "gemini"}
    #      model    : "" → modèle par défaut du provider
    #                 (groq_model / openrouter_model / gemini_model ci-dessus)
    #
    # ROUTE_BY_COMPLEXITY=false → revient à « un seul modèle pour tout » (ancien
    # comportement, aucun routage).
    route_by_complexity: bool = os.getenv("ROUTE_BY_COMPLEXITY", "true").lower() == "true"

    fast_provider:    str = os.getenv("FAST_PROVIDER",    "groq")        # question simple → rapide / pas cher
    fast_model:       str = os.getenv("FAST_MODEL",       "")
    context_provider: str = os.getenv("CONTEXT_PROVIDER", "openrouter")  # cas normal
    context_model:    str = os.getenv("CONTEXT_MODEL",    "")
    deep_provider:    str = os.getenv("DEEP_PROVIDER",    "openrouter")  # multi-fichiers / complexe → gros modèle
    deep_model:       str = os.getenv("DEEP_MODEL",       "")

    def model_for_level(self, level: str) -> tuple[str, str]:
        """Retourne (provider, model) pour un niveau fast|context|deep.

        model == "" signifie « utiliser le modèle par défaut du provider ».
        """
        table = {
            "fast":    (self.fast_provider,    self.fast_model),
            "context": (self.context_provider, self.context_model),
            "deep":    (self.deep_provider,    self.deep_model),
        }
        return table.get(level, (self.context_provider, self.context_model))


class RAGConfig(BaseModel):
    embedding_model:     str   = "jinaai/jina-embeddings-v2-base-code"
    embedding_dimension: int   = 768
    embedding_device:    str   = None
    vector_store:        str   = "chromadb"
    # Métrique RÉELLE des collections ChromaDB principales : L2 (défaut Chroma,
    # aucun hnsw:space="cosine" n'est passé à leur création). Les embeddings Jina
    # étant normalisés, L2² = 2(1-cos). Les seuils sont calibrés pour L2 :
    #   SystemAwareRAG.THRESHOLD=1.2 · ProjectCodeIndexer=0.75 · relevance_threshold=0.45
    # (La mémoire sémantique du chat est la seule à utiliser explicitement cosine.)
    distance_metric:     str   = "l2"
    chunk_size:          int   = 800
    chunk_overlap:       int   = 150
    top_k:               int   = 8
    relevance_threshold: float = 0.45

    # ── P4 · Compression contextuelle ─────────────────────────────────────────
    # Filtre les docs RAG (anti-doublons + pertinence) AVANT le prompt, sans appel
    # LLM. Réduit les tokens à qualité égale. Mettre RAG_COMPRESSION=false pour
    # désactiver. compression_threshold : 0-1, plus haut = plus strict (jette plus).
    compression_enabled:   bool  = os.getenv("RAG_COMPRESSION", "true").lower() == "true"
    compression_threshold: float = float(os.getenv("RAG_COMPRESSION_THRESHOLD", "0.70"))
    compression_min_keep:  int   = 2

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


class RedisConfig(BaseModel):
    """Configuration pour la connexion Redis via MCP."""
    url:    str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    prefix: str = "ca:"  


class LangGraphConfig(BaseModel):
    """Configuration for LangGraph orchestration + LangSmith tracing."""
    enabled:            bool = os.getenv("USE_LANGGRAPH", "false").lower() == "true"
    langsmith_tracing:  bool = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    langsmith_api_key:  str  = os.getenv("LANGSMITH_API_KEY", "")
    langsmith_project:  str  = os.getenv("LANGSMITH_PROJECT", "code-auditor")
    langsmith_endpoint: str  = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")


class Config:
    BASE_DIR           = Path(__file__).parent
    DATA_DIR           = BASE_DIR / "data"
    KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
    VECTOR_STORE_DIR   = DATA_DIR / "vector_store"
    CACHE_DIR          = DATA_DIR / "cache"
    KG_PATH            = DATA_DIR / "knowledge_graph.json"
    # Racine de persistance ChromaDB pour les collections "directes" (client
    # chromadb natif), p.ex. la mémoire sémantique du chat → data/semantic_memory.
    # Les autres collections (LangChain Chroma) gèrent leur propre persist_directory.
    CHROMA_PERSIST_DIR = DATA_DIR

    for _d in [DATA_DIR, KNOWLEDGE_BASE_DIR, VECTOR_STORE_DIR, CACHE_DIR]:
        _d.mkdir(parents=True, exist_ok=True)

    api       = APIConfig()
    rag       = RAGConfig()
    analysis  = AnalysisConfig()
    watcher   = WatcherConfig()
    redis     = RedisConfig()
    langgraph = LangGraphConfig()

    HOST  = os.getenv("SERVER_HOST", "127.0.0.1")
    PORT  = int(os.getenv("SERVER_PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    API_PREFIX        = "/api/v1"
    CHROMA_COLLECTION = "code_kb_jina_v2"


config = Config()