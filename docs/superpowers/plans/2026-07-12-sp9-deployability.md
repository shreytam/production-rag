# SP9 · Deployability & Ops — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Packages the RAG system inside secure multi-stage container environments, sets up boot validation configurations, wraps requests in rate limiters and timeout handlers, and adds system health/readiness endpoints.

**Architecture:** Create multi-stage non-root Dockerfiles, move pipeline instantiation out of lazy sync query routes into lifespan startup functions, instantiate in-memory token bucket rate limiters, configure CORS/TrustedHost middleware, and expand compose scripts with container restart policies and secret checkers.

**Tech Stack:** Python 3.11-3.13, Docker, Docker Compose, FastAPI, tenacity, gitleaks, uvicorn.

## Global Constraints
- Every commit must be authored solely under the repository owner's identity: `Shreytam Goyal <shreytamgoyal@gmail.com>`.
- Nothing may be attributed to Claude. Do NOT add a `Co-Authored-By:` trailer, a `Claude-Session:` line, a "Generated with Claude" note, or any other AI/Anthropic attribution in commit messages, PR titles, or PR descriptions.
- Do not use the `shreytam.goyal@codiant.com` (codiant) identity for commits.
- Commit messages describe the change only — no AI-attribution footers of any kind.
- The `.cache/` directory must be ignored.
- Rate limiters default to fail-closed behavior on state errors.
- Oversized request bodies (exceeding 64 KiB) map immediately to HTTP 413.

---

### Task 1: Dockerfile and Dockerignore Packaging

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

**Interfaces:**
- Consumes: None
- Produces: Ephemeral container build specifications

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp9_docker_setup.py` verifying file path coverage:
```python
import pytest
from pathlib import Path

def test_docker_ignore_excludes_keys_and_venv():
    ignore_path = Path(".dockerignore")
    docker_path = Path("Dockerfile")
    assert ignore_path.exists()
    assert docker_path.exists()
    
    content = ignore_path.read_text()
    assert "infra/.env" in content
    assert ".venv" in content
    assert ".cache" in content
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp9_docker_setup.py`
Expected: FAIL (missing files)

- [ ] **Step 3: Modify files**
Create `Dockerfile` using multi-stage builds and non-root execution:
```dockerfile
# Stage 1: Build dependencies
FROM python:3.12-slim AS builder
ENV PEP517_BUILD_BACKEND=hatchling.build
WORKDIR /app
COPY pyproject.toml hatch_build.py ./
RUN pip install --no-cache-dir uv && uv pip install --system --no-cache-dir .

# Stage 2: Production runtime image
FROM python:3.12-slim AS runner
WORKDIR /app
# Install system utilities safely
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m -s /bin/bash appuser

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . /app

# Secure folder ownerships
RUN chown -R appuser:appgroup /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/readyz')" || exit 1

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```
Create `.dockerignore`:
```
.git
.venv
.cache/
data/
infra/.env
.env
.env.*
eval/runs/
*.pkl
.pytest_cache
.ruff_cache
.remember/
.claude/
Dockerfile
.dockerignore
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp9_docker_setup.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add Dockerfile .dockerignore tests/test_sp9_docker_setup.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(ops): add production multi-stage Dockerfile and dockerignore configuration" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 2: Lifespan Startup Pipeline Handlers

**Files:**
- Create: `app/lifespan.py`
- Modify: `core/interfaces.py`
- Modify: `providers/vectorstores/qdrant_store.py`
- Modify: `providers/vectorstores/pgvector_store.py`
- Modify: `app/api.py`

**Interfaces:**
- Consumes: `Settings`
- Produces: `app.state.pipeline` populated on boot, with `configured_dimension() -> int | None` on vector stores

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp9_boot_dimension.py` asserting dimension checks:
```python
import pytest
from core.interfaces import VectorStore
from core.registry import build_vector_store
from core.config import Settings

def test_store_reports_dimension():
    settings = Settings()
    store = build_vector_store(settings)
    dim = store.configured_dimension()
    # If store table/collection exists, should match dimensions configuration
    # Or return None if not provisioned
    assert dim is None or isinstance(dim, int)
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp9_boot_dimension.py`
Expected: FAIL (AttributeError: configured_dimension not implemented)

- [ ] **Step 3: Modify files**
Add `configured_dimension` signature in `core/interfaces.py` Protocol class.
Implement `configured_dimension` in `providers/vectorstores/qdrant_store.py`:
```python
    def configured_dimension(self) -> int | None:
        try:
            info = self._client.get_collection(self.collection_name)
            return info.config.params.vectors.size
        except Exception:
            return None
```
Implement `configured_dimension` in `providers/vectorstores/pgvector_store.py`:
```python
    def configured_dimension(self) -> int | None:
        # Check column definitions
        query = (
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = %s::regclass AND attname = 'embedding'"
        )
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, [self._table])
                    row = cur.fetchone()
                    if row and row[0] != -1:
                        return row[0]
        except Exception:
            pass
        return None
```
Create `app/lifespan.py` managing boot validation:
```python
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from core.config import get_settings
from core.pipeline import build
from core.registry import build_vector_store, build_embedder

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Running lifespan boot validation check...")
    
    # 1. Validation: Key existences
    if not settings.nvidia_api_key and not settings.openai_api_key and not settings.anthropic_api_key:
        raise ValueError("Initialization failure: At least one LLM/Endpoint key must be configured at boot!")

    # 2. Validation: Vector Store dimension check
    store = build_vector_store(settings)
    db_dim = store.configured_dimension()
    if db_dim is not None and db_dim != settings.embed_dimension:
        raise ValueError(
            f"Dimension misalignment: Vector store collection expects width {db_dim}, "
            f"but current embed_dimension config is set to {settings.embed_dimension}"
        )

    # 3. Connection check for active models
    # Check connect connectivity cheaply
    embedder = build_embedder(settings)
    app.state.embedder_available = True
    try:
        # Resolve models list (no usage charges)
        if hasattr(embedder, "_client") and hasattr(embedder._client, "models"):
            embedder._client.models.list(timeout=5.0)
    except Exception as exc:
        logger.warning("Embedder model listing check failed: %s", exc)
        app.state.embedder_available = False
        if settings.app_env == "prod":
            raise ValueError(f"Boot check failed: Active embedder connection down in production environment! {exc}") from exc

    # Storing singletons
    app.state.pipeline = build(version="full", dataset=None, enable_guardrails=settings.guardrails_enabled)
    app.state.ready = True
    yield
    app.state.ready = False
```
Modify `app/api.py` to activate lifespan and clean route singletons:
```python
from app.lifespan import lifespan

app = FastAPI(lifespan=lifespan)

# Before: get_pipeline lazy build
# After: read from state
def get_pipeline(request: Request):
    return request.app.state.pipeline
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp9_boot_dimension.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add core/interfaces.py providers/vectorstores/qdrant_store.py providers/vectorstores/pgvector_store.py app/lifespan.py app/api.py tests/test_sp9_boot_dimension.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(ops): implement lifespan singleton builders and boot verification assertions" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 3: Cheap Readiness Probe `/readyz`

**Files:**
- Modify: `app/api.py`

**Interfaces:**
- Consumes: `app.state` parameters
- Produces: `/readyz` route endpoint returning 200/503

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp9_probes.py` verifying status targets:
```python
import pytest
from fastapi.testclient import TestClient
from app.api import app

def test_readiness_and_liveness_probes():
    client = TestClient(app)
    
    # 1. Liveness probe (always ok)
    resp_live = client.get("/healthz")
    assert resp_live.status_code == 200
    assert resp_live.json() == {"status": "ok"}
    
    # 2. Readiness probe
    resp_ready = client.get("/readyz")
    # Assert return corresponds to lifespan setup state (loaded successfully)
    assert resp_ready.status_code in (200, 503)
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp9_probes.py`
Expected: FAIL (404 Not Found on readiness request endpoint)

- [ ] **Step 3: Modify files**
Add `/readyz` probe to `app/api.py`:
```python
@app.get("/readyz")
async def readiness_probe(request: Request):
    # Verify app initialization readiness
    is_ready = getattr(request.app.state, "ready", False)
    embedder_ok = getattr(request.app.state, "embedder_available", False)
    
    # Verify store connectivity quickly
    store_ok = True
    try:
        from core.registry import build_vector_store
        store = build_vector_store()
        # Qdrant get_collections / PostgreSQL execution checks
        if hasattr(store, "_client"):
            store._client.get_collections(timeout=2.0)
    except Exception:
        store_ok = False
        
    if is_ready and embedder_ok and store_ok:
        return {"ready": True}
        
    return JSONResponse(
        status_code=503,
        content={
            "ready": False,
            "details": {
                "initialization": is_ready,
                "embedder_connectivity": embedder_ok,
                "vector_store_connectivity": store_ok
            }
        }
    )
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp9_probes.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add app/api.py tests/test_sp9_probes.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(ops): deploy cheaper readiness probe readyz route" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 4: Trusted Hosts, CORS, and Security Middleware

**Files:**
- Create: `app/middleware.py`
- Modify: `app/api.py`

**Interfaces:**
- Consumes: Settings CORS and Host configs
- Produces: Stacked HTTP response headers and host verification filters

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp9_security_middleware.py`:
```python
import pytest
from fastapi.testclient import TestClient
from app.api import app

def test_security_headers_middleware_injection():
    client = TestClient(app)
    response = client.get("/healthz")
    
    # Assert header presence
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp9_security_middleware.py`
Expected: FAIL (missing headers or import errors)

- [ ] **Step 3: Modify files**
Create `app/middleware.py` mapping custom security headers:
```python
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response
```
Add standard dependencies in `app/api.py`:
```python
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from app.middleware import SecurityHeadersMiddleware
from core.config import get_settings

settings = get_settings()

# Register middleware in correct filter ordering (outer -> inner)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SecurityHeadersMiddleware)
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp9_security_middleware.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add app/middleware.py app/api.py tests/test_sp9_security_middleware.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(ops): integrate CORS, TrustedHost and security header middleware" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 5: Rate Limiter Token Bucket Implementation

**Files:**
- Modify: `core/interfaces.py`
- Modify: `core/registry.py`
- Create: `providers/ratelimit/token_bucket.py`

**Interfaces:**
- Consumes: None
- Produces: `RateLimiter` Protocol and `InMemoryTokenBucket`

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp9_ratelimit.py` checking limit exhaustion:
```python
import pytest
import time
from providers.ratelimit.token_bucket import InMemoryTokenBucket

def test_token_bucket_exhaustion():
    limiter = InMemoryTokenBucket(rate=2, burst=3)
    # Extract burst
    assert limiter.allow("tenant-1") is True
    assert limiter.allow("tenant-1") is True
    assert limiter.allow("tenant-1") is True
    # Over capacity limit
    assert limiter.allow("tenant-1") is False
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp9_ratelimit.py`
Expected: FAIL (ImportError rate limit components not found)

- [ ] **Step 3: Modify files**
Declare `RateLimiter` Protocol in `core/interfaces.py`:
```python
@runtime_checkable
class RateLimiter(Protocol):
    def allow(self, key: str) -> bool: ...
```
Create `providers/ratelimit/token_bucket.py` implementing thread safety lock options:
```python
import time
from threading import Lock

class InMemoryTokenBucket:
    def __init__(self, rate: float, burst: float) -> None:
        self._rate = rate
        self._burst = burst
        self._keys: dict[str, tuple[float, float]] = {} # key -> (tokens, last_update_time)
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        with self._lock:
            now = time.monotonic()
            tokens, last_time = self._keys.get(key, (self._burst, now))
            
            # Replenish logic
            elapsed = now - last_time
            replenished = elapsed * (self._rate / 60.0) # rate parameter defines allocations per minute
            tokens = min(self._burst, tokens + replenished)
            
            if tokens >= 1.0:
                self._keys[key] = (tokens - 1.0, now)
                return True
                
            self._keys[key] = (tokens, now)
            return False
```
Register `build_rate_limiter` in `core/registry.py`:
```python
def build_rate_limiter(settings: Settings | None = None) -> RateLimiter:
    s = settings or get_settings()
    if s.rate_limiter == "memory":
        from providers.ratelimit.token_bucket import InMemoryTokenBucket
        return InMemoryTokenBucket(rate=s.rate_limit_per_minute, burst=s.rate_limit_burst)
    raise ValueError(f"Unknown rate limiter type: {s.rate_limiter}")
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp9_ratelimit.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add core/interfaces.py core/registry.py providers/ratelimit/token_bucket.py tests/test_sp9_ratelimit.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(ops): implement threadsafe in-memory token bucket rate limiter" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 6: Request Body Limits and Middleware Timeouts

**Files:**
- Modify: `app/middleware.py`
- Modify: `app/api.py`

**Interfaces:**
- Consumes: Settings `max_body_bytes`, `request_timeout_ms`
- Produces: `BodySizeLimitMiddleware` and `RequestTimeoutMiddleware`

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp9_limits.py` checking sizes and timeout handlers:
```python
import pytest
from fastapi.testclient import TestClient
from app.api import app

def test_oversized_payload_returns_413():
    client = TestClient(app)
    # Generate 100kb payload (limit is 64kb)
    large_payload = "A" * (70 * 1024)
    response = client.post("/query", content=large_payload)
    assert response.status_code == 413
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp9_limits.py`
Expected: FAIL (returns 422 or 400 instead of 413)

- [ ] **Step 3: Modify files**
Create `BodySizeLimitMiddleware` in `app/middleware.py`:
```python
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request, status
import asyncio

class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_body_bytes: int):
        super().__init__(app)
        self.max_body_bytes = max_body_bytes

    async def dispatch(self, request: Request, call_next):
        # Read content-length header early before load buffers
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > self.max_body_bytes:
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={"detail": "Request body size exceeds configured maximum limits!"}
                    )
            except ValueError:
                pass
        return await call_next(request)
```
Create `RequestTimeoutMiddleware` in `app/middleware.py` limiting execution wall clock:
```python
class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, timeout_ms: int):
        super().__init__(app)
        self.timeout_sec = timeout_ms / 1000.0

    async def dispatch(self, request: Request, call_next):
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout_sec)
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content={"detail": "App request execution timed out!"}
            )
```
Add to `app/api.py`:
```python
from app.middleware import BodySizeLimitMiddleware, RequestTimeoutMiddleware

app.add_middleware(BodySizeLimitMiddleware, max_body_bytes=settings.max_body_bytes)
app.add_middleware(RequestTimeoutMiddleware, timeout_ms=settings.request_timeout_ms)
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp9_limits.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add app/middleware.py app/api.py tests/test_sp9_limits.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(ops): employ body size limitations and execution timeout middleware" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 7: Docker Compose Hardening and Secret Scanner

**Files:**
- Modify: `infra/docker-compose.yml`
- Modify: `Makefile`
- Create: `.gitleaks.toml`
- Create: `.github/workflows/gitleaks.yml`

**Interfaces:**
- Consumes: None
- Produces: Service health guarantees and repo security enforcement

- [ ] **Step 1: Write the failing test**
Create a test parsing compose yaml configs:
```python
import pytest
import yaml

def test_compose_contains_restarts_and_healthchecks():
    with open("infra/docker-compose.yml") as f:
        data = yaml.safe_load(f)
    services = data.get("services", {})
    for name, svc in services.items():
        assert "restart" in svc
        assert "healthcheck" in svc or name == "postgres" # Postgres already has one
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp9_compose.py`
Expected: FAIL (assert failures due to missing parameters)

- [ ] **Step 3: Modify files**
Update `infra/docker-compose.yml`:
1. Add `app` service building the local container:
```yaml
  app:
    build:
      context: ..
      dockerfile: Dockerfile
    ports:
      - 8000:8000
    depends_on:
      qdrant:
        condition: service_healthy
      postgres:
        condition: service_healthy
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
```
2. Add restart policies and healthchecks to `qdrant`/`postgres`:
```yaml
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "sleep 0.1 && bash -c ':> /dev/tcp/127.0.0.1/6333'"]
      interval: 10s
      timeout: 5s
      retries: 5
```
Create `.gitleaks.toml` detecting endpoints pattern:
```toml
[[rules]]
description = "NVIDIA NIM API Key detection"
regex = '''nvapi-[A-Za-z0-9_-]+'''
tags = ["key", "nvidia"]

[[rules]]
description = "OpenAI API Key"
regex = '''sk-[A-Za-z0-9_-]{48}'''
tags = ["key", "openai"]
```
Create `.github/workflows/gitleaks.yml` triggering scans on push.
Update `Makefile` to launch the API container:
```makefile
.PHONY: run-api
run-api:
	uvicorn app.api:app --host 0.0.0.0 --port 8000 --workers 4
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp9_compose.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add infra/docker-compose.yml Makefile .gitleaks.toml .github/workflows/gitleaks.yml
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "ci(ops): harden compose definitions and establish gitleaks key scanners" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```
