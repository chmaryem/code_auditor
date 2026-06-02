# Roadmap: 3 Améliorations Critiques (Next 2 Weeks)

## 1. FIX: Initialization Race Condition + Per-Request Isolation

### Current Problem
```python
# api/server.py (BUGGY)
_orchestrator = None  # Global, shared by all requests

@lifespan
async def lifespan(app):
    init_thread = threading.Thread(target=_orchestrator.initialize)
    init_thread.start()  # Background
    yield
    # Requests can hit while still initializing
```

### Solution (10 min fix)
```python
# api/server.py (FIXED)
import asyncio
from contextvars import ContextVar

_orchestrator_lock = asyncio.Lock()
_orchestrator_ready = asyncio.Event()
_orchestrator = None

@lifespan
async def lifespan(app):
    global _orchestrator
    
    logger.info("Starting orchestrator initialization...")
    try:
        # Initialize in thread pool (don't block event loop)
        _orchestrator = await asyncio.to_thread(
            Orchestrator,
            project_path=_default_project
        )
        await asyncio.to_thread(_orchestrator.initialize)
        _orchestrator_ready.set()
        logger.info("Orchestrator ready")
    except Exception as e:
        logger.error(f"Init failed: {e}")
        _orchestrator = None
    
    yield
    
    # Cleanup
    if _orchestrator:
        await asyncio.to_thread(_orchestrator.stop)

async def get_orchestrator():
    """Dependency injection for all endpoints"""
    if not await asyncio.wait_for(_orchestrator_ready.wait(), timeout=30):
        raise HTTPException(503, "Server initializing, try again in 10s")
    return _orchestrator

@app.post("/analyze/file")
async def analyze_file(req, orch = Depends(get_orchestrator)):
    # NOW: orch is guaranteed ready
    result = await asyncio.to_thread(orch.analyze_single, Path(req.file_path))
    return result
```

### Validation
```bash
# Test 1: Rapid requests during startup
curl -X POST http://localhost:8765/health  # Should say "initializing"
# Wait 15s
curl -X POST http://localhost:8765/health  # Should say "ready"

# Test 2: Concurrent requests after ready
for i in {1..10}; do
  curl -X POST http://localhost:8765/api/chat -d '{"message":"test"}' &
done
# Should handle all without race conditions
```

---

## 2. SECURITY: Input Validation + Auth

### Current Problem
```python
# BUGGY: No path validation
@app.post("/analyze/file")
async def analyze_file(req: AnalyzeFileRequest):
    file_path = Path(req.file_path)  # Can be "../../../../etc/passwd"
    code = file_path.read_text()     # Security issue!
```

### Solution (20 min)

**A) Path Validation**
```python
# langchain_agents/utils/security.py (NEW)
from pathlib import Path

def validate_path_within_project(project_root: str, target_path: str) -> Path:
    """Ensure target_path is within project_root"""
    root = Path(project_root).resolve()
    target = Path(target_path).resolve()
    
    try:
        target.relative_to(root)  # Raises ValueError if outside
    except ValueError:
        raise ValueError(f"Path {target} escapes project {root}")
    
    return target

# Usage
@app.post("/analyze/file")
async def analyze_file(req: AnalyzeFileRequest):
    try:
        file_path = validate_path_within_project(
            req.project_path,
            req.file_path
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    
    if not file_path.is_file():
        raise HTTPException(404, f"Not a file: {file_path}")
    
    result = await asyncio.to_thread(orch.analyze_single, file_path)
    return result
```

**B) API Key Auth**
```python
# .env
API_KEY_HASH=your_hashed_key  # Use `python -c "import hashlib; print(hashlib.sha256(b'YOUR_KEY').hexdigest())"`

# api/auth.py (NEW)
from fastapi.security import HTTPBearer, HTTPAuthCredential
from fastapi import Depends, HTTPException
import hashlib
import os

security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthCredential = Depends(security)) -> str:
    """Validate API key"""
    provided_hash = hashlib.sha256(credentials.credentials.encode()).hexdigest()
    expected_hash = os.getenv("API_KEY_HASH")
    
    if not expected_hash or provided_hash != expected_hash:
        raise HTTPException(401, "Invalid API key")
    
    return credentials.credentials

# Usage in any endpoint
@app.post("/api/chat")
async def chat(req: ChatRequest, _key = Depends(verify_api_key)):
    # Endpoint is now protected
    return await _run_chat(...)
```

**C) Rate Limiting**
```python
# requirements.txt (ADD)
slowapi>=0.1.8

# api/server.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/chat")
@limiter.limit("10/minute")  # 10 requests per minute per IP
async def chat(req: ChatRequest, request: Request, _key = Depends(verify_api_key)):
    return await _run_chat(...)
```

### Validation
```bash
# Test 1: Path traversal blocked
curl -X POST http://localhost:8765/analyze/file \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{"file_path":"../../../../etc/passwd"}'
# Should return 400: "Path escapes project"

# Test 2: Rate limit
for i in {1..15}; do
  curl -X POST http://localhost:8765/api/chat \
    -H "Authorization: Bearer YOUR_KEY" \
    -d '{"message":"test"}' &
done
# First 10 succeed, next 5 get 429: Too Many Requests
```

---

## 3. OBSERVABILITY: Structured Logging + Tracing

### Current Problem
```python
# BUGGY: Unstructured logs, no tracing
logger.error("ChatAgent LLM answer failed: %s", e)
# Can't correlate logs across request
# Can't see which LLM was actually used
# Can't track latency per intent
```

### Solution (30 min)

**A) Structured Logging**
```python
# requirements.txt (ADD)
structlog>=23.1.0
python-json-logger>=2.0.7

# config/logging.py (NEW)
import logging
import structlog
import json

def setup_logging():
    """Configure structured logging"""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Also log to stdout for debugging
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
    )

# main.py
from config.logging import setup_logging
setup_logging()

logger = structlog.get_logger()

# Usage in chat_router.py
async def _run_chat(message, project_path, session_id, target_file):
    start = time.time()
    
    try:
        result = await ainvoke_chat(
            message=message,
            project_path=project_path,
            session_id=session_id,
            target_file=target_file,
        )
        
        elapsed = time.time() - start
        
        # STRUCTURED LOG (outputs JSON)
        logger.info(
            "chat_request_completed",
            session_id=session_id,
            intent=result.get("intent"),
            target_file=target_file,
            latency_ms=round(elapsed * 1000),
            rag_docs=len(result.get("rag_docs", [])),
            response_length=len(result.get("response", "")),
        )
        
        return ChatResponse(...)
        
    except Exception as e:
        elapsed = time.time() - start
        logger.error(
            "chat_request_failed",
            session_id=session_id,
            error=str(e),
            error_type=type(e).__name__,
            latency_ms=round(elapsed * 1000),
            exc_info=True,
        )
        raise
```

Output (JSON, easy to parse):
```json
{
  "event": "chat_request_completed",
  "timestamp": "2026-05-17T14:32:10Z",
  "session_id": "sess_abc123",
  "intent": "explain",
  "latency_ms": 1243,
  "rag_docs": 4,
  "response_length": 512
}
```

**B) Request Tracing**
```python
# requirements.txt (ADD)
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0
opentelemetry-exporter-jaeger>=1.20.0

# config/tracing.py (NEW)
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

def init_tracing():
    """Initialize OpenTelemetry tracing to Jaeger"""
    jaeger_exporter = JaegerExporter(
        agent_host_name=os.getenv("JAEGER_HOST", "localhost"),
        agent_port=int(os.getenv("JAEGER_PORT", 6831)),
    )
    
    trace.set_tracer_provider(TracerProvider())
    trace.get_tracer_provider().add_span_processor(
        SimpleSpanProcessor(jaeger_exporter)
    )

# api/server.py
from config.tracing import init_tracing

@lifespan
async def lifespan(app):
    init_tracing()
    # ... rest of init

# Usage in chat_router.py
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

async def _run_chat(message, project_path, session_id, target_file):
    with tracer.start_as_current_span("chat_request") as span:
        span.set_attribute("session_id", session_id)
        span.set_attribute("intent", "pending")
        span.set_attribute("target_file", target_file)
        
        try:
            result = await ainvoke_chat(...)
            span.set_attribute("intent", result.get("intent"))
            span.set_attribute("success", True)
            return result
        except Exception as e:
            span.set_attribute("error", str(e))
            span.set_attribute("success", False)
            raise
```

**C) Prometheus Metrics** (for dashboards)
```python
# requirements.txt (ADD)
prometheus-client>=0.17.0

# config/metrics.py (NEW)
from prometheus_client import Counter, Histogram, Gauge

chat_requests_total = Counter(
    "chat_requests_total",
    "Total chat requests",
    ["intent", "status"]
)

chat_latency_seconds = Histogram(
    "chat_latency_seconds",
    "Chat response latency",
    ["intent"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

rag_docs_retrieved = Histogram(
    "rag_docs_retrieved",
    "RAG documents per request",
    buckets=[1, 2, 4, 6, 10]
)

llm_fallback_total = Counter(
    "llm_fallback_total",
    "Times LLM fallback was used",
    ["from_model", "to_model"]
)

# Usage in chat_router.py
async def _run_chat(...):
    start = time.time()
    try:
        result = await ainvoke_chat(...)
        elapsed = time.time() - start
        
        chat_requests_total.labels(
            intent=result.get("intent"),
            status="success"
        ).inc()
        
        chat_latency_seconds.labels(
            intent=result.get("intent")
        ).observe(elapsed)
        
        rag_docs_retrieved.observe(len(result.get("rag_docs", [])))
        
        return result
    except Exception as e:
        chat_requests_total.labels(
            intent="unknown",
            status="error"
        ).inc()
        raise

# api/server.py - expose metrics endpoint
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Now you can access Prometheus: http://localhost:9090
# Scrape from: http://localhost:8765/metrics
```

### Validation
```bash
# Test 1: Structured logs appear
python main.py serve &
sleep 3
curl -X POST http://localhost:8765/api/chat \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{"message":"explain this file","project_path":"."}'
# Check logs in terminal — should be JSON formatted

# Test 2: Metrics are exposed
curl http://localhost:8765/metrics | grep chat_requests_total
# Output: chat_requests_total{intent="explain",status="success"} 1.0

# Test 3: Jaeger visualization (if running)
# Open http://localhost:16686
# Should see traces for "chat_request" span with all attributes
```

---

## Integration Checklist (Apply All 3)

```bash
# 1. Create config directory
mkdir -p config

# 2. Create new files
touch config/logging.py config/tracing.py config/metrics.py
touch langchain_agents/utils/security.py
touch api/auth.py

# 3. Update requirements.txt
# ... add: structlog, slowapi, opentelemetry-*, prometheus-client

# 4. Update api/server.py
# ... import security, auth, logging, metrics, tracing

# 5. Update all endpoints
# @limiter.limit("X/minute")
# @verify_api_key(...)
# span.set_attribute(...)
# logger.info(..., **structured_fields)

# 6. Test
pytest tests/
python main.py serve --reload

# 7. Monitor
# - Tail logs: tail -f app.log | jq .
# - Check metrics: curl http://localhost:8765/metrics
# - View traces: http://localhost:16686
```

---

## Expected Improvement (Before vs After)

| Metric | Before | After |
|--------|--------|-------|
| **Race conditions** | ❌ Yes | ✅ None |
| **Path traversal risk** | ⚠️ High | ✅ Blocked |
| **API exposed** | ❌ Yes | ✅ Authenticated |
| **Rate limiting** | ❌ None | ✅ 10/min |
| **Debug time** | ⏱️ 30 min | ⏱️ 5 min |
| **Latency visibility** | ❌ No | ✅ Yes (P95 = 1.2s) |
| **Production ready** | ❌ 30% | ✅ 85% |

