# SP6 · Resilience & Failure Modes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implements timeouts, retry budgets with Retry-After backoff, query degradation policies, database connection pooling, and sanitised exception handling.

**Architecture:** Create a resilience utility module offering backoff retry policies, apply timeouts and retry rules to NIM, Qdrant, and psycopg layers, upgrade psycopg connection management to use a client-side pool, handle retrieval failures by degrading to sparse or dense fallbacks, and install global FastAPI handlers returning correlation IDs.

**Tech Stack:** Python 3.11-3.13, Pydantic, FastAPI, psycopg-pool, tenacity, httpx, Qdrant-client.

## Global Constraints
- Every commit must be authored solely under the repository owner's identity: `Shreytam Goyal <shreytamgoyal@gmail.com>`.
- Nothing may be attributed to Claude. Do NOT add a `Co-Authored-By:` trailer, a `Claude-Session:` line, a "Generated with Claude" note, or any other AI/Anthropic attribution in commit messages, PR titles, or PR descriptions.
- Do not use the `shreytam.goyal@codiant.com` (codiant) identity for commits.
- Commit messages describe the change only — no AI-attribution footers of any kind.
- The `.cache/` directory must be ignored.
- Safe default interactive boundaries (timeouts <= 15s, retries <= 3 attempts) are enforced.
- Security-critical ACL boundaries must be preserved under degradation: fallbacks only restrict scope, never bypass filters.

---

### Task 1: Configuration Knobs for Resilience

**Files:**
- Modify: `core/config.py`

**Interfaces:**
- Consumes: None
- Produces:
  - Settings: `query_timeout_seconds`, `retry_attempts`, `retry_min_wait_seconds`, `retry_max_wait_seconds`, `gen_query_timeout_seconds`, `gen_query_max_retries`, `pg_pool_min_size`, `pg_pool_max_size`, `pg_connect_timeout_seconds`, `pg_statement_timeout_ms`, `max_request_id_len`, `circuit_breaker_enabled`

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp6_config.py` verifying new resilience knobs:
```python
import pytest
from core.config import Settings

def test_sp6_config_knobs():
    settings = Settings(
        query_timeout_seconds=10.0,
        retry_attempts=4,
        retry_min_wait_seconds=1.0,
        retry_max_wait_seconds=5.0,
        gen_query_timeout_seconds=30.0,
        gen_query_max_retries=1,
        pg_pool_min_size=2,
        pg_pool_max_size=8,
        pg_connect_timeout_seconds=3.0,
        pg_statement_timeout_ms=10000,
        max_request_id_len=64,
        circuit_breaker_enabled=True
    )
    assert settings.query_timeout_seconds == 10.0
    assert settings.retry_attempts == 4
    assert settings.pg_pool_max_size == 8
    assert settings.pg_statement_timeout_ms == 10000
    assert settings.max_request_id_len == 64
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp6_config.py`
Expected: FAIL (ValidationError or AttributeError)

- [ ] **Step 3: Modify files**
Add configuration fields to `Settings` inside `core/config.py`:
```python
    # --- Resilience knobs (interactive query path) ---
    query_timeout_seconds: float = 15.0
    retry_attempts: int = 3
    retry_min_wait_seconds: float = 0.5
    retry_max_wait_seconds: float = 8.0
    
    # --- Generator timeouts (interactive query path override) ---
    gen_query_timeout_seconds: float = 60.0
    gen_query_max_retries: int = 2
    
    # --- pgvector Connection Pooling ---
    pg_pool_min_size: int = 1
    pg_pool_max_size: int = 10
    pg_connect_timeout_seconds: float = 5.0
    pg_statement_timeout_ms: int = 15000
    
    # --- Circuit Breaker & Middleware ---
    circuit_breaker_enabled: bool = False
    max_request_id_len: int = 128
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp6_config.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add core/config.py tests/test_sp6_config.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(resilience): add settings configuration keys for resilience tuning" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 2: Resilience Utilities and Retry Logic

**Files:**
- Create: `core/resilience.py`

**Interfaces:**
- Consumes: None
- Produces: `is_retryable`, `retry_after_wait`, `resilient_retry`, `UpstreamUnavailable`

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp6_resilience_util.py` to verify backoff and classification:
```python
import pytest
import httpx
from core.resilience import is_retryable, retry_after_wait, UpstreamUnavailable

def test_is_retryable_classification():
    # 429 and 5xx must be retryable
    assert is_retryable(httpx.HTTPStatusError("429", request=None, response=httpx.Response(429))) is True
    assert is_retryable(httpx.HTTPStatusError("503", request=None, response=httpx.Response(503))) is True
    # 400 or 401 must NOT be retryable
    assert is_retryable(httpx.HTTPStatusError("400", request=None, response=httpx.Response(400))) is False
    # Timeout/Network must be retryable
    assert is_retryable(httpx.TimeoutException("timeout")) is True

def test_retry_after_wait_uses_header():
    # Create exception carrying Retry-After header
    resp = httpx.Response(429, headers={"Retry-After": "3"})
    exc = httpx.HTTPStatusError("429", request=None, response=resp)
    
    # Mocking tenacity retry state
    class FakeState:
        def __init__(self, outcome):
            self.outcome = outcome
            self.attempt_number = 1
            
    class FakeOutcome:
        def __init__(self, exception):
            self._exception = exception
        def failed(self):
            return True
        def exception(self):
            return self._exception

    state = FakeState(FakeOutcome(exc))
    # Wait is computed
    wait_time = retry_after_wait(state)
    assert wait_time == 3.0
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp6_resilience_util.py`
Expected: FAIL (Module `core/resilience` doesn't exist)

- [ ] **Step 3: Modify files**
Create `core/resilience.py`:
```python
from __future__ import annotations
import email.utils
import time
import math
from typing import Any, Callable
import httpx
import tenacity
from tenacity import wait_exponential, stop_after_attempt, retry_if_exception

class UpstreamUnavailable(Exception):
    """Raised when retries are exhausted or degradation results in hard failures."""
    def __init__(self, stage: str, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.retry_after = retry_after

def is_retryable(exc: BaseException) -> bool:
    """Classifies if exception is retryable (429, 5xx, or network timeout)."""
    # First inspect Qdrant unexpected response wrapper
    qdrant_exc_type = None
    try:
        from qdrant_client.http.exceptions import UnexpectedResponse
        qdrant_exc_type = UnexpectedResponse
    except ImportError:
        pass

    if qdrant_exc_type and isinstance(exc, qdrant_exc_type):
        status = getattr(exc, "status_code", 500)
        return status == 429 or (500 <= status <= 599)

    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or (500 <= status <= 599)
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    
    # Check psycopg connection errors
    try:
        import psycopg
        if isinstance(exc, psycopg.OperationalError):
            return True
    except ImportError:
        pass

    return False

def retry_after_wait(retry_state: tenacity.RetryCallState) -> float:
    """Computes retry delay, reading Retry-After if present."""
    default_wait = wait_exponential(multiplier=0.5, min=0.5, max=8.0)(retry_state)
    
    if not retry_state.outcome or not retry_state.outcome.failed():
        return default_wait
        
    exc = retry_state.outcome.exception()
    if not isinstance(exc, httpx.HTTPStatusError):
        return default_wait
        
    headers = exc.response.headers
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    
    if not retry_after:
        return default_wait
        
    # Check if number of seconds
    try:
        return max(0.1, float(retry_after))
    except ValueError:
        pass
        
    # Check if HTTP date format
    parsed_date = email.utils.parsedate_to_datetime(retry_after)
    if parsed_date:
        diff = (parsed_date - email.utils.utils.datetime.datetime.now(parsed_date.tzinfo)).total_seconds()
        return max(0.5, diff)
        
    return default_wait

def resilient_retry(attempts: int = 3, min_wait: float = 0.5, max_wait: float = 8.0) -> Callable:
    """Decorates calls with resilience rules."""
    def decorator(func: Callable) -> Callable:
        return tenacity.retry(
            retry=retry_if_exception(is_retryable),
            wait=retry_after_wait,
            stop=stop_after_attempt(attempts),
            reraise=True,
        )(func)
    return decorator
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp6_resilience_util.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add core/resilience.py tests/test_sp6_resilience_util.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(resilience): deploy core resilience decision utility functions" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 3: Upstream Client Gating & NIM Reranker

**Files:**
- Modify: `providers/rerankers/nim_rerank.py`
- Modify: `core/registry.py`

**Interfaces:**
- Consumes: `resilient_retry`, `UpstreamUnavailable`
- Produces: `NIMReranker` instance executing retry logic with interactive timeouts

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp6_rerank_resilience.py` mocks checking HTTP failures:
```python
import pytest
import respx
import httpx
from providers.rerankers.nim_rerank import NIMReranker
from core.resilience import UpstreamUnavailable

@respx.mock
def test_nim_rerank_retries_429_then_succeeds():
    reranker = NIMReranker("meta/rerank", "https://api.nvidia.com", "key", timeout=1.0, attempts=3)
    
    # First call returns 429, second lands success
    route = respx.post("https://api.nvidia.com/ranking").side_effect = [
        httpx.Response(429, headers={"Retry-After": "0.1"}),
        httpx.Response(200, json={"rankings": [{"index": 0, "logit": 0.9}]})
    ]
    
    chunks = [type("ScoredChunk", (), {"chunk": type("Chunk", (), {"text": "A"})()})()]
    res = reranker.rerank("Q", chunks, top_n=1)
    
    assert len(res) == 1
    assert res[0].score == 0.9
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp6_rerank_resilience.py`
Expected: FAIL (NIMReranker constructor mismatch or no retry retry on 429)

- [ ] **Step 3: Modify files**
Update constructor and `rerank` in `providers/rerankers/nim_rerank.py` to route through settings timeouts, retry attempts and `resilient_retry`:
```python
from core.resilience import resilient_retry, UpstreamUnavailable

class NIMReranker:
    def __init__(self, model: str, base_url: str, api_key: str, timeout: float = 15.0, attempts: int = 3) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._attempts = attempts

    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_n: int,
    ) -> list[ScoredChunk]:
        # Wrap the API call step so retry runs correctly
        @resilient_retry(attempts=self._attempts)
        def _execute_post():
            payload = {
                "model": self._model,
                "query": {"text": query},
                "passages": [{"text": c.chunk.text} for c in chunks],
            }
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    f"{self._base_url}/ranking",
                    json=payload,
                    headers=headers,
                )
            response.raise_for_status()
            return response.json()

        if not chunks:
            return []

        try:
            data = _execute_post()
        except Exception as exc:
            # Map exception to UpstreamUnavailable
            retry_after_str = None
            if isinstance(exc, httpx.HTTPStatusError):
                retry_after_str = exc.response.headers.get("retry-after")
            raise UpstreamUnavailable(
                stage="rerank",
                message=f"NIM Reranker failed after {self._attempts} attempts: {exc}",
                retry_after=float(retry_after_str) if retry_after_str else None
            ) from exc

        rankings: list[dict] = data.get("rankings", [])
```
Update `core/registry.py::build_reranker`:
```python
    if s.reranker == "nim":
        from providers.rerankers.nim_rerank import NIMReranker

        return NIMReranker(
            s.reranker_nim_model,
            s.reranker_nim_base_url,
            s.reranker_nim_api_key,
            timeout=s.query_timeout_seconds,
            attempts=s.retry_attempts
        )
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp6_rerank_resilience.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add providers/rerankers/nim_rerank.py core/registry.py tests/test_sp6_rerank_resilience.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(resilience): enforce status client level interactive timeouts and retries on NIM rerank calls" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 4: Qdrant client connection timeouts and retries

**Files:**
- Modify: `providers/vectorstores/qdrant_store.py`
- Modify: `core/registry.py`

**Interfaces:**
- Consumes: Settings `query_timeout_seconds`, `retry_attempts`
- Produces: Timeout parameter mappings passed into `QdrantClient` constructor

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp6_qdrant_timeout.py` verifying client setup:
```python
import pytest
from core.registry import build_vector_store
from core.config import Settings
from qdrant_client import QdrantClient

def test_qdrant_build_timeout():
    settings = Settings(vector_store="qdrant", query_timeout_seconds=42.0)
    store = build_vector_store(settings)
    assert getattr(store, "_client", None) is not None
    # QdrantClient constructor maps timeouts (verify attribute)
    assert store._client._runtimes.get("timeout") == 42.0
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp6_qdrant_timeout.py`
Expected: FAIL (no timeout attribute check matches, or client timeout remains default)

- [ ] **Step 3: Modify files**
Update constructor and operations inside `providers/vectorstores/qdrant_store.py` to add timeouts and `resilient_retry`:
```python
from core.resilience import resilient_retry, UpstreamUnavailable
from qdrant_client.http.exceptions import UnexpectedResponse

class QdrantVectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.collection_name = settings.qdrant_collection
        # Supply explicit timeout to constructor
        self._client = QdrantClient(
            url=settings.qdrant_url,
            timeout=settings.query_timeout_seconds
        )
        self._attempts = settings.retry_attempts

    def search(self, embedding: list[float], top_k: int, acl: ACLContext) -> list[ScoredChunk]:
        @resilient_retry(attempts=self._attempts)
        def _search():
            # Real query call logic
            return self._client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                query_filter=qdrant_filter(acl),
                limit=top_k,
            )

        try:
            results = _search()
        except Exception as exc:
            raise UpstreamUnavailable(
                stage="dense",
                message=f"Qdrant search failed after {self._attempts} retries: {exc}"
            ) from exc
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp6_qdrant_timeout.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add providers/vectorstores/qdrant_store.py tests/test_sp6_qdrant_timeout.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(resilience): enforce client timeouts and retries on Qdrant store operations" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 5: pgvector connection pooling & statement timeouts

**Files:**
- Modify: `pyproject.toml`
- Modify: `providers/vectorstores/pgvector_store.py`

**Interfaces:**
- Consumes: `psycopg_pool.ConnectionPool`
- Produces: High-performance connection sharing, per-query statement timeouts, and one-off database migrations

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp6_pg_pool.py` checking connection reuse:
```python
import pytest
from core.config import Settings
from providers.vectorstores.pgvector_store import PgVectorStore

def test_pgvector_uses_pool(monkeypatch):
    # Mock psycopg_pool to assert we utilize client pool
    called_pool = False
    class MockPool:
        def __init__(self, *args, **kwargs):
            nonlocal called_pool
            called_pool = True
        def open(self):
            pass
        def close(self):
            pass
            
    monkeypatch.setattr("psycopg_pool.ConnectionPool", MockPool)
    settings = Settings(vector_store="pgvector")
    store = PgVectorStore(settings)
    assert called_pool is True
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp6_pg_pool.py`
Expected: FAIL (AttributeError psycopg_pool connection pool or missing imports)

- [ ] **Step 3: Modify files**
Add `psycopg-pool` to dependencies list in `pyproject.toml`:
```toml
dependencies = [
    # ...
    "psycopg[binary]>=3.2",
    "psycopg-pool>=3.2",
    "pgvector>=0.3",
]
```
Ensure dependencies are local by running `uv sync` first.
Modify `providers/vectorstores/pgvector_store.py`:
1. Initialize the global `psycopg_pool.ConnectionPool` in constructor:
```python
import psycopg_pool
from pgvector.psycopg import register_vector
from core.resilience import resilient_retry, UpstreamUnavailable

class PgVectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._table = settings.pg_table
        self._attempts = settings.retry_attempts
        
        # Build client conn pool hook
        def configure_conn(conn):
            register_vector(conn)
            # Add hard query duration constraints
            with conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = {settings.pg_statement_timeout_ms}")

        self._pool = psycopg_pool.ConnectionPool(
            conninfo=settings.pg_dsn,
            min_size=settings.pg_pool_min_size,
            max_size=settings.pg_pool_max_size,
            connect_timeout=settings.pg_connect_timeout_seconds,
            configure=configure_conn,
            open=False # Lazy-open
        )
        self._pool.open()
```
2. Move `CREATE EXTENSION` command out of hot checkout path to `ensure_collection`:
```python
    def ensure_collection(self, dimension: int) -> None:
        # Running extension creation only on catalog setup/migrations
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    f"CREATE TABLE IF NOT EXISTS {self._table} ("
                    "id uuid PRIMARY KEY, "
                    "text text, "
                    "embedding vector, "
                    "tenant_id text, "
                    "acl text[]"
                    ")"
                )
```
3. Wrap `search()` using pooled hook connections and retry guards:
```python
    def search(self, embedding: list[float], top_k: int, acl: ACLContext) -> list[ScoredChunk]:
        @resilient_retry(attempts=self._attempts)
        def _search():
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    # Exec query block using pgvector distance operations
                    ...
                    
        try:
            return _search()
        except Exception as exc:
            import psycopg
            if isinstance(exc, psycopg.errors.QueryCanceled):
                raise UpstreamUnavailable(
                    stage="dense",
                    message="database statement canceled due to timeout duration hit!"
                ) from exc
            raise UpstreamUnavailable(
                stage="dense",
                message=f"Postgres DB dense retrieval failed: {exc}"
            ) from exc
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp6_pg_pool.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add pyproject.toml providers/vectorstores/pgvector_store.py tests/test_sp6_pg_pool.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(resilience): deploy psycopg connection pooling and statement timeouts" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 6: Interactive Generator Timeout Overrides

**Files:**
- Modify: `providers/generators/openai_compatible.py`
- Modify: `core/registry.py`

**Interfaces:**
- Consumes: Settings `gen_query_timeout_seconds`, `gen_query_max_retries`
- Produces: Isolated interactive query parameters set separately from long running batch ingestion

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp6_generator_timeout.py`:
```python
import pytest
from core.registry import build_generator
from core.config import Settings

def test_interactive_generator_timeout_assignment():
    settings = Settings(
        gen_query_timeout_seconds=42.0,
        gen_query_max_retries=1
    )
    generator = build_generator(role="gen", settings=settings)
    assert generator._client.timeout == 42.0
    assert generator._client.max_retries == 1
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp6_generator_timeout.py`
Expected: FAIL (Client timeouts use general settings value of 600s or max_retries default is 5)

- [ ] **Step 3: Modify files**
Modify `core/registry.py::build_generator` to apply isolated interactive parameters for `role == "gen"` / `role == "judge"`:
```python
    if provider == "openai":
        from providers.generators.openai_compatible import OpenAICompatibleGenerator

        model = {"gen": s.gen_model, "context": s.context_model, "judge": s.judge_model}[role]
        base_url = {"gen": s.gen_base_url, "context": s.context_base_url, "judge": s.judge_base_url}[role]
        api_key = {"gen": s.gen_api_key, "context": s.context_api_key, "judge": s.judge_api_key}[role]
        
        # Segregate timeout and retry budgets by role
        if role in ("gen", "judge"):
            timeout = s.gen_query_timeout_seconds
            max_retries = s.gen_query_max_retries
        else:
            timeout = s.request_timeout_seconds
            max_retries = s.max_retries

        return OpenAICompatibleGenerator(
            model,
            base_url,
            api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp6_generator_timeout.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add core/registry.py tests/test_sp6_generator_timeout.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(resilience): apply isolated timeout and retry boundaries to interactive generator calls" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 7: Graceful Degradation in Hybrid Retriever

**Files:**
- Modify: `retrieval/hybrid.py`
- Modify: `core/pipeline.py`

**Interfaces:**
- Consumes: `UpstreamUnavailable`
- Produces: `degraded_stages` list payload injected into retrieval outputs mapping downstream telemetry systems

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp6_retrieval_degradation.py` checking mock recovery:
```python
import pytest
from core.resilience import UpstreamUnavailable
from retrieval.hybrid import HybridRetriever

class MockDenseStore:
    def search(self, *args, **kwargs):
        raise UpstreamUnavailable("dense", "dense connection failed")

def test_hybrid_degrades_to_sparse_only(monkeypatch):
    class MockSparseStore:
        def search(self, *args, **kwargs):
            return ["sparse_item"]
            
    retriever = HybridRetriever(
        dense_store=MockDenseStore(),
        sparse_store=MockSparseStore(),
        reranker=None,  # skip rerank
        embedder=None   # skip embed errors
    )
    # Execute query search
    res, degraded = retriever.retrieve_degraded("Q", top_k=5, acl=None)
    assert res == ["sparse_item"]
    assert "dense" in degraded
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp6_retrieval_degradation.py`
Expected: FAIL (AttributeError retrieve_degraded does not exist or unhandled UpstreamUnavailable error bubbles up)

- [ ] **Step 3: Modify files**
Implement per-stage try/except blocks inside `retrieval/hybrid.py`. Convert exceptions to empty lists + populate `degraded_stages`.
Modify `retrieval/hybrid.py`:
```python
from core.resilience import UpstreamUnavailable

class HybridRetriever:
    # Existing init hooks
    ...
    
    def retrieve(self, query: str, top_k: int, acl: ACLContext) -> list[ScoredChunk]:
        res, degraded = self.retrieve_with_telemetry(query, top_k, acl)
        # Store metadata inside the execution pipeline
        return res

    def retrieve_with_telemetry(self, query: str, top_k: int, acl: ACLContext) -> tuple[list[ScoredChunk], list[str]]:
        degraded_stages = []
        qvec = None
        
        # 1. Embedding Stage
        try:
            if self._embedder:
                qvec = self._embedder.embed_query(query)
        except Exception:
            degraded_stages.append("embed")
            
        # 2. Dense Store Stage
        dense_results = []
        if qvec and self._dense:
            try:
                dense_results = self._dense.search(qvec, top_k, acl)
            except UpstreamUnavailable as exc:
                degraded_stages.append("dense")
                
        # 3. Sparse Store Stage
        sparse_results = []
        if self._sparse:
            try:
                sparse_results = self._sparse.search(query, top_k, acl)
            except UpstreamUnavailable as exc:
                degraded_stages.append("sparse")

        # Raise global hard failure if BOTH databases are unavailable
        if not dense_results and not sparse_results:
            raise UpstreamUnavailable(
                stage="retrieval",
                message="Retrieval is completely down: dense and sparse both unreachable!"
            )

        # 4. reciprocal rank fusion
        fused = reciprocal_rank_fusion([dense_results, sparse_results], k=self._rrf_k)

        # 5. Rerank Stage
        if not self._reranker:
            return fused[:top_k], degraded_stages
            
        try:
            reranked = self._reranker.rerank(query, fused[:self._rerank_window], self._top_n)
            return reranked, degraded_stages
        except UpstreamUnavailable:
            degraded_stages.append("rerank")
            # Degrade gracefully to RRF order
            return fused[:self._top_n], degraded_stages
```
Update `core/pipeline.py` and query execution loops to attach `degraded_stages` array into returned output models/traces.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp6_retrieval_degradation.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add retrieval/hybrid.py core/pipeline.py tests/test_sp6_retrieval_degradation.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(resilience): implement per-stage graceful degradation in hybrid retriever" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 8: Global Error Handling & Correlation Middleware

**Files:**
- Create: `app/errors.py`
- Modify: `app/api.py`

**Interfaces:**
- Consumes: None
- Produces: API routing extensions tracking custom errors mapping to 503 HTTP status outputs, with correlation headers attached

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp6_api_handlers.py` checking response codes:
```python
import pytest
from fastapi.testclient import TestClient
from app.api import app

def test_api_retains_correlation_id_on_failures():
    client = TestClient(app)
    # Query with header triggers 503/500 custom handler checks
    response = client.post("/query", json={"query": "Q"}, headers={"X-Request-Id": "test-id-123"})
    # Match response request header
    assert response.headers.get("X-Request-Id") == "test-id-123"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp6_api_handlers.py`
Expected: FAIL (No X-Request-Id header returned in responses or exception handlers output raw 500)

- [ ] **Step 3: Modify files**
Create `app/errors.py`:
```python
import uuid
import logging
import re
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from core.resilience import UpstreamUnavailable
from core.config import get_settings

logger = logging.getLogger(__name__)
REQUEST_ID_REGEX = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

def setup_request_id_middleware(app: FastAPI):
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        settings = get_settings()
        raw_id = request.headers.get("X-Request-Id", "")
        
        # Bounded character validations
        if raw_id and len(raw_id) <= settings.max_request_id_len and REQUEST_ID_REGEX.match(raw_id):
            request_id = raw_id
        else:
            request_id = str(uuid.uuid4())
            
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response

def install_error_handlers(app: FastAPI):
    setup_request_id_middleware(app)
    
    @app.exception_handler(UpstreamUnavailable)
    async def upstream_unavailable_handler(request: Request, exc: UpstreamUnavailable):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error("UpstreamUnavailable stage=[%s] error=[%s] request_id=[%s]", exc.stage, str(exc), request_id)
        
        headers = {"X-Request-Id": request_id}
        if exc.retry_after:
            headers["Retry-After"] = str(int(exc.retry_after))
            
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Upstream service temporarily unavailable", 
                "request_id": request_id, 
                "stage": exc.stage
            },
            headers=headers
        )

    @app.exception_handler(Exception)
    async def catch_all_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception("Unhandled error encountered! request_id=[%s]", request_id)
        
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error occurred", 
                "request_id": request_id
            },
            headers={"X-Request-Id": request_id}
        )
```
Update `app/api.py` to register custom setup routing:
```python
from app.errors import install_error_handlers

app = FastAPI()
install_error_handlers(app)
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp6_api_handlers.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add app/errors.py app/api.py tests/test_sp6_api_handlers.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(resilience): deploy request correlation middleware and global HTTP error handlers" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```
