import os
import logging
import secrets
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, field_validator, model_validator


load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

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
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    kimi_api_key: str = os.getenv("KIMI_API_KEY", "")
    kimi_model:   str = os.getenv("KIMI_MODEL", "kimi-k3")

    gemini_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model:   str = os.getenv("GEMINI_MODEL",   "gemini-2.0-flash")
   
    temperature: float = 0.0
    max_tokens:  int   = 16384

   
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


class ChatConfig(BaseModel):

  
    orchestrator: str = os.getenv("CHAT_ORCHESTRATOR", "blackboard")

   
    router_regex_first: bool = os.getenv("CHAT_ROUTER_REGEX_FIRST", "true").lower() == "true"
  
    router_regex_min_confidence: float = float(os.getenv("CHAT_ROUTER_REGEX_MIN_CONF", "0.85"))

   
   
    semantic_memory_every: int = int(os.getenv("CHAT_SEMANTIC_MEMORY_EVERY", "3"))

   
    decision_cache_enabled: bool = os.getenv("CHAT_DECISION_CACHE", "true").lower() == "true"
    decision_cache_ttl: int = int(os.getenv("CHAT_DECISION_CACHE_TTL", "900"))  # 15 min
    history_turns: int = int(os.getenv("CHAT_HISTORY_TURNS", "8"))
    rag_top_k: int = int(os.getenv("CHAT_RAG_TOPK", "6"))

    quality_guard_enabled: bool = os.getenv("CHAT_QUALITY_GUARD", "true").lower() == "true"


class RAGConfig(BaseModel):
    embedding_model:     str   = "jinaai/jina-embeddings-v2-base-code"
    embedding_dimension: int   = 768
    embedding_device:    str   = None
    vector_store:        str   = "chromadb"
    distance_metric:     str   = "l2"
    chunk_size:          int   = 800
    chunk_overlap:       int   = 150
    top_k:               int   = 8
    relevance_threshold: float = 0.45
    
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


class DatabaseConfig:
    """PostgreSQL connection settings. Async DSN for SQLAlchemy; sync DSN for Alembic."""
    url: str          = os.getenv("DATABASE_URL",      "postgresql+asyncpg://codeauditor:codeauditor@localhost:5432/codeauditor")
    sync_url: str     = os.getenv("DATABASE_SYNC_URL", "postgresql+psycopg2://codeauditor:codeauditor@localhost:5432/codeauditor")
    pool_size: int    = int(os.getenv("DB_POOL_SIZE",    "10"))
    max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    pool_timeout: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    echo: bool        = os.getenv("DB_ECHO", "false").lower() == "true"


class AuthConfig:
    """All authentication settings. Consumed by auth/ sub-package."""

    def __init__(self) -> None:
        self.app_name: str   = os.getenv("AUTH_APP_NAME", "Code Auditor AI")
        self.auth_required: bool = os.getenv("AUTH_REQUIRED", "true").lower() in ("1", "true", "yes", "on")

        # Redis — auth uses a dedicated redis-py client (same server, separate connection)
        self.redis_url: str  = os.getenv("REDIS_URL", "redis://localhost:6379/0")

        # JWT
        self.jwt_secret: str       = os.getenv("AUTH_JWT_SECRET", "")
        self.jwt_algorithm: str    = os.getenv("AUTH_JWT_ALGORITHM", "HS256")
        self.access_ttl_min: int   = int(os.getenv("AUTH_ACCESS_TTL_MIN",  "60"))
        self.refresh_ttl_days: int = int(os.getenv("AUTH_REFRESH_TTL_DAYS", "14"))
        self.pairing_ttl_sec: int  = int(os.getenv("AUTH_PAIRING_TTL_SEC",  "90"))
        self.jwt_ephemeral: bool   = False

        if not self.jwt_secret:
            self.jwt_secret    = secrets.token_urlsafe(48)
            self.jwt_ephemeral = True
            logging.getLogger("auth").warning(
                "AUTH_JWT_SECRET is not set — using an ephemeral secret. "
                "All sessions will be invalidated on restart. Set AUTH_JWT_SECRET in .env."
            )

        # OTP
        self.otp_length: int              = int(os.getenv("AUTH_OTP_LENGTH",            "6"))
        self.otp_ttl_sec: int             = int(os.getenv("AUTH_OTP_TTL_SEC",           "600"))
        self.otp_max_attempts: int        = int(os.getenv("AUTH_OTP_MAX_ATTEMPTS",       "5"))
        self.otp_resend_cooldown_sec: int = int(os.getenv("AUTH_OTP_RESEND_COOLDOWN_SEC","60"))
        self.otp_rate_per_hour: int       = int(os.getenv("AUTH_OTP_RATE_PER_HOUR",      "5"))

        # Email domain allow-list (empty = any domain allowed)
        raw_domains = os.getenv("AUTH_ALLOWED_EMAIL_DOMAINS", "")
        self.allowed_email_domains: list[str] = [
            d.strip().lower() for d in raw_domains.split(",") if d.strip()
        ]

        # SMTP
        self.smtp_host: str      = os.getenv("SMTP_HOST",         "smtp.gmail.com")
        self.smtp_port: int      = int(os.getenv("SMTP_PORT",     "587"))
        self.smtp_user: str      = os.getenv("SMTP_USER",         "")
        self.smtp_password: str  = os.getenv("SMTP_APP_PASSWORD", "").replace(" ", "")
        self.smtp_from: str      = os.getenv("SMTP_FROM",         "") or self.smtp_user
        self.smtp_from_name: str = os.getenv("SMTP_FROM_NAME",    self.app_name)

    @property
    def access_ttl_sec(self) -> int:
        return self.access_ttl_min * 60

    @property
    def refresh_ttl_sec(self) -> int:
        return self.refresh_ttl_days * 86_400

    @property
    def smtp_configured(self) -> bool:
        return bool(self.smtp_user and self.smtp_password)

    def is_email_domain_allowed(self, email: str) -> bool:
        if not self.allowed_email_domains:
            return True
        domain = email.rsplit("@", 1)[-1].lower()
        return domain in self.allowed_email_domains


class Config:
    BASE_DIR           = Path(__file__).parent
    DATA_DIR           = BASE_DIR / "data"
    KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
    VECTOR_STORE_DIR   = DATA_DIR / "vector_store"
    CACHE_DIR          = DATA_DIR / "cache"
    KG_PATH            = DATA_DIR / "knowledge_graph.json"
    
    CHROMA_PERSIST_DIR = DATA_DIR

    for _d in [DATA_DIR, KNOWLEDGE_BASE_DIR, VECTOR_STORE_DIR, CACHE_DIR]:
        _d.mkdir(parents=True, exist_ok=True)

    api       = APIConfig()
    chat      = ChatConfig()
    rag       = RAGConfig()
    analysis  = AnalysisConfig()
    watcher   = WatcherConfig()
    redis     = RedisConfig()
    langgraph = LangGraphConfig()
    database  = DatabaseConfig()
    auth      = AuthConfig()

    HOST  = os.getenv("SERVER_HOST", "127.0.0.1")
    PORT  = int(os.getenv("SERVER_PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    API_PREFIX        = "/api/v1"
    CHROMA_COLLECTION = "code_kb_jina_v2"


config = Config()