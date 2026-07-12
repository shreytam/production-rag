# SP8 · Ingest Robustness & Incremental Updates — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implements fault-tolerant contextual chunking, atomic caching, and content-hash-driven incremental re-embedding.

**Architecture:** Extend the `VectorStore` compiler protocol to support delete and metadata update methods, introduce a JSONL manifest store monitoring document revisions, compute blake2b hashes representing code prompt variations and text payloads, and restructure contextual annotations to fail-soft per-chunk.

**Tech Stack:** Python 3.11-3.13, Pydantic, rank-bm25, qdrant-client, psycopg.

## Global Constraints
- Every commit must be authored solely under the repository owner's identity: `Shreytam Goyal <shreytamgoyal@gmail.com>`.
- Nothing may be attributed to Claude. Do NOT add a `Co-Authored-By:` trailer, a `Claude-Session:` line, a "Generated with Claude" note, or any other AI/Anthropic attribution in commit messages, PR titles, or PR descriptions.
- Do not use the `shreytam.goyal@codiant.com` (codiant) identity for commits.
- Commit messages describe the change only — no AI-attribution footers of any kind.
- The `.cache/` directory must be ignored.
- Delete and metadata-update endpoints must inherit tenant-scoping (ACL constraints fail-closed by default).
- Incremental updates write back-checkpoint manifests strictly after database writes succeed (D-ORDER).

---

### Task 1: VectorStore Deletions and Metadata Updates

**Files:**
- Modify: `core/interfaces.py`
- Modify: `providers/vectorstores/qdrant_store.py`
- Modify: `providers/vectorstores/pgvector_store.py`

**Interfaces:**
- Consumes: `ACLContext` and list of `chunk_ids`
- Produces: `VectorStore.delete(chunk_ids, acl)` and `VectorStore.update_metadata(updates, acl)`

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp8_store_mutations.py` to assert tenant-scoped delete and update behavior:
```python
import pytest
from core.types import ACLContext, Chunk
from core.registry import build_vector_store

def test_delete_and_update_fail_closed_if_cross_tenant():
    # Setup test workspace
    ...
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp8_store_mutations.py`
Expected: FAIL (AttributeError due to missing methods)

- [ ] **Step 3: Modify files**
Update Protocols in `core/interfaces.py`:
```python
    def delete(self, chunk_ids: list[str], acl: ACLContext) -> None: ...
    def update_metadata(self, updates: dict[str, dict], acl: ACLContext) -> None: ...
```
Implement `delete` and `update_metadata` in `providers/vectorstores/qdrant_store.py`:
```python
    def delete(self, chunk_ids: list[str], acl: ACLContext) -> None:
        from qdrant_client import models
        # Map IDs to uuid5
        from providers.vectorstores.qdrant_store import uuid5
        uuids = [str(uuid5(cid)) for cid in chunk_ids]
        
        # Enforce tag-level tenant isolation logic
        # Points must match BOTH the IDs and the tenant ACL bounds
        tenant_filter = qdrant_filter(acl)
        # Combine
        combined_filter = models.Filter(
            must=[
                models.HasIdCondition(has_id=uuids),
                tenant_filter
            ]
        )
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=combined_filter)
        )

    def update_metadata(self, updates: dict[str, dict], acl: ACLContext) -> None:
        from qdrant_client import models
        from providers.vectorstores.qdrant_store import uuid5
        
        tenant_filter = qdrant_filter(acl)
        for chunk_id, payload in updates.items():
            pt_id = str(uuid5(chunk_id))
            
            # Restructuring only metadata (e.g. title/acl)
            # Must verify point matches tenant condition before mutation
            self._client.set_payload(
                collection_name=self.collection_name,
                payload=payload,
                points=[pt_id],
                key=None, # full set payload
                filter=tenant_filter
            )
```
Implement `delete` and `update_metadata` in `providers/vectorstores/pgvector_store.py`:
```python
    def delete(self, chunk_ids: list[str], acl: ACLContext) -> None:
        from retrieval.acl import pg_where
        where_clause, params = pg_where(acl)
        
        # Format query parameters
        query = f"DELETE FROM {self._table} WHERE chunk_id = ANY(%s) AND {where_clause}"
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, [chunk_ids] + params)

    def update_metadata(self, updates: dict[str, dict], acl: ACLContext) -> None:
        from retrieval.acl import pg_where
        import json
        
        where_clause, params = pg_where(acl)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                for chunk_id, payload in updates.items():
                    # Flatten keys or dump JSON representing fields
                    update_query = (
                        f"UPDATE {self._table} SET "
                        "text = %s, tenant_id = %s, acl = %s "
                        f"WHERE chunk_id = %s AND {where_clause}"
                    )
                    # Bind variables safely (payload must match schema)
                    cur.execute(
                        update_query,
                        [
                            payload.get("text", ""),
                            payload.get("tenant_id", ""),
                            payload.get("acl", []),
                            chunk_id
                        ] + params
                    )
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp8_store_mutations.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add core/interfaces.py providers/vectorstores/qdrant_store.py providers/vectorstores/pgvector_store.py tests/test_sp8_store_mutations.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(ingest): add delete and update_metadata operations to vector store backends" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 2: Document Manifests and Storage

**Files:**
- Modify: `core/types.py`
- Modify: `core/interfaces.py`
- Modify: `core/registry.py`
- Create: `providers/manifest/jsonl_store.py`

**Interfaces:**
- Consumes: None
- Produces: `DocManifest` model, `ManifestStore` Protocol, `JsonlManifestStore`

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp8_manifest_store.py` checking loading/writing:
```python
import pytest
from pathlib import Path
from core.types import DocManifest, ChunkRecord
from providers.manifest.jsonl_store import JsonlManifestStore

def test_jsonl_manifest_atomic_writes(tmp_path):
    store = JsonlManifestStore(manifest_dir=str(tmp_path))
    manifest = DocManifest(
        tenant_id="tenant-1",
        doc_id="doc-123",
        prompt_version="v1",
        chunks={
            "chunk-1": ChunkRecord(chunk_id="chunk-1", ordinal=0, embed_hash="h1", meta_hash="h2")
        }
    )
    store.save(manifest)
    
    # Reload and verify
    loaded = store.load("tenant-1", "doc-123")
    assert loaded is not None
    assert loaded.prompt_version == "v1"
    assert loaded.chunks["chunk-1"].embed_hash == "h1"
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp8_manifest_store.py`
Expected: FAIL (ImportError or ModuleNotFoundError)

- [ ] **Step 3: Modify files**
Add types to `core/types.py`:
```python
from pydantic import BaseModel

class ChunkRecord(BaseModel):
    chunk_id: str
    ordinal: int
    embed_hash: str
    meta_hash: str

class DocManifest(BaseModel):
    tenant_id: str
    doc_id: str
    prompt_version: str
    chunks: dict[str, ChunkRecord]
```
Add Protocol to `core/interfaces.py`:
```python
@runtime_checkable
class ManifestStore(Protocol):
    def load(self, tenant_id: str, doc_id: str) -> DocManifest | None: ...
    def save(self, manifest: DocManifest) -> None: ...
    def delete(self, tenant_id: str, doc_id: str) -> None: ...
```
Create `providers/manifest/jsonl_store.py`:
```python
from __future__ import annotations
import json
import os
import hashlib
from pathlib import Path
from core.types import DocManifest, ChunkRecord

class JsonlManifestStore:
    def __init__(self, manifest_dir: str = ".cache/manifest") -> None:
        self.manifest_dir = Path(manifest_dir)

    def _path_for(self, tenant_id: str, doc_id: str) -> Path:
        # Create hash signature for filename safety
        h_id = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
        return self.manifest_dir / tenant_id / f"{h_id}.json"

    def load(self, tenant_id: str, doc_id: str) -> DocManifest | None:
        path = self._path_for(tenant_id, doc_id)
        if not path.exists():
            return None
        try:
            with path.open("r") as f:
                data = json.load(f)
            # Enforce tenant validation fails closed if mismatched
            if data.get("tenant_id") != tenant_id or data.get("doc_id") != doc_id:
                return None
            return DocManifest.model_validate(data)
        except Exception:
            return None

    def save(self, manifest: DocManifest) -> None:
        path = self._path_for(manifest.tenant_id, manifest.doc_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replacement: write to temp, then rename
        tmp_path = path.with_suffix(".tmp")
        try:
            with tmp_path.open("w") as f:
                json.dump(manifest.model_dump(), f, indent=2)
            # Force os replacement
            os.replace(tmp_path, path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def delete(self, tenant_id: str, doc_id: str) -> None:
        path = self._path_for(tenant_id, doc_id)
        if path.exists():
            path.unlink()
```
Update `core/registry.py` to register `build_manifest_store`:
```python
def build_manifest_store(settings: Settings | None = None) -> ManifestStore:
    s = settings or get_settings()
    from providers.manifest.jsonl_store import JsonlManifestStore
    return JsonlManifestStore(manifest_dir=s.manifest_dir)
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp8_manifest_store.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add core/types.py core/interfaces.py core/registry.py providers/manifest/jsonl_store.py tests/test_sp8_manifest_store.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(ingest): implement DocManifest schema and JsonlManifestStore tracker" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 3: Fault-Tolerant Contextual Prefixer

**Files:**
- Modify: `ingest/contextual.py`

**Interfaces:**
- Consumes: `Generator` instance
- Produces: Hardened `ContextualPrefixer.annotate` accepting chunk slices

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp8_prefixer_fault.py` verifying skip logic:
```python
import pytest
from ingest.contextual import annotate_chunks
from core.types import Document

def test_prefixer_continues_on_individual_chunk_failure():
    # Make mock generator that fails on chunk ordinal=1
    class FaultyGenerator:
        def complete(self, messages, **_):
            # Check content triggers
            if "second chunk" in messages[1].content.lower():
                raise RuntimeError("LLM failure")
            return type("Resp", (), {"text": "Prefix", "parsed": None})()
            
    # Mock documents and verify process
    ...
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp8_prefixer_fault.py`
Expected: FAIL (AttributeError or RuntimeError bubbles out of prefixer annotate call)

- [ ] **Step 3: Modify files**
Update `ingest/contextual.py`:
1. Use `blake2b` with prompt version string tags in cache key calculation:
```python
_PROMPT_VERSION = "v1"
```
2. Truncate document context using config `contextual_doc_token_budget`.
3. Wrap individual chunk generation calls inside `annotate` in a `try/except` handler. Log errors and leave `contextual_prefix=None` for failed annotation targets instead of throwing.
4. Maintain write safety by replacing cache execution steps with `os.replace` temp swaps.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp8_prefixer_fault.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add ingest/contextual.py tests/test_sp8_prefixer_fault.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "fix(ingest): harden contextual prefixer with per-chunk recovery and atomic caching" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 4: Incremental Ingestor Difference Logic

**Files:**
- Create: `ingest/incremental.py`

**Interfaces:**
- Consumes: `Embedder`, `VectorStore`, and `ManifestStore`
- Produces: `IncrementalIngestor` executing partition delta allocations

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp8_incremental_diff.py` simulating document scaling changes:
```python
import pytest
from ingest.incremental import IncrementalIngestor

def test_incremental_diff_allocates_work_lists():
    # Mock stores and check delta processing
    ...
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp8_incremental_diff.py`
Expected: FAIL (ImportError or missing class)

- [ ] **Step 3: Modify files**
Create `ingest/incremental.py`:
```python
from __future__ import annotations
import hashlib
from typing import List, Dict, Set
from core.types import Chunk, ACLContext, DocManifest, ChunkRecord
from core.interfaces import Embedder, VectorStore, ManifestStore

def _compute_hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()

def _compute_meta_hash(payload: dict) -> str:
    # Stable serialization
    meta_str = f"{payload.get('title')}:{payload.get('tenant_id')}:{sorted(payload.get('acl_tags', []))}"
    return _compute_hash(meta_str)

class IncrementalIngestor:
    def __init__(self, embedder: Embedder, vector_store: VectorStore, manifest_store: ManifestStore) -> None:
        self._embedder = embedder
        self._store = vector_store
        self._manifest_store = manifest_store

    def ingest_document(self, tenant_id: str, doc_id: str, chunks: list[Chunk], acl: ACLContext) -> None:
        # 1. Load active manifest
        old_manifest = self._manifest_store.load(tenant_id, doc_id)
        
        new_records: Dict[str, ChunkRecord] = {}
        to_embed: List[Chunk] = []
        to_meta: Dict[str, dict] = {}
        
        # Examine current versions
        for idx, c in enumerate(chunks):
            # Compute distinct pay hashes
            e_hash = _compute_hash(c.embed_text)
            payload = {
                "text": c.text,
                "tenant_id": c.tenant_id,
                "acl_tags": acl.tags,
                "title": getattr(c, "title", "")
            }
            m_hash = _compute_meta_hash(payload)
            
            new_records[c.chunk_id] = ChunkRecord(
                chunk_id=c.chunk_id,
                ordinal=idx,
                embed_hash=e_hash,
                meta_hash=m_hash
            )
            
            # Check difference criteria
            old_rec = old_manifest.chunks.get(c.chunk_id) if old_manifest else None
            
            if not old_rec or old_rec.embed_hash != e_hash:
                to_embed.append(c)
            elif old_rec.meta_hash != m_hash:
                to_meta[c.chunk_id] = payload
                
        # Discover deletions (orphans)
        to_delete: List[str] = []
        if old_manifest:
            new_ids = set(new_records.keys())
            for old_id in old_manifest.chunks:
                if old_id not in new_ids:
                    to_delete.append(old_id)
                    
        # 2. Database write mutations
        # Write vectors
        if to_embed:
            # Batch embedding
            embeddings = self._embedder.embed_documents([c.embed_text for c in to_embed])
            for c, emb in zip(to_embed, embeddings):
                c.embedding = emb
            self._store.upsert(to_embed)
            
        # Update metadata
        if to_meta:
            self._store.update_metadata(to_meta, acl)
            
        # Delete orphans
        if to_delete:
            self._store.delete(to_delete, acl)
            
        # 3. Save checkpoint manifest (D-ORDER: ONLY AFTER store operations succeed)
        manifest = DocManifest(
            tenant_id=tenant_id,
            doc_id=doc_id,
            prompt_version="v1",
            chunks=new_records
        )
        self._manifest_store.save(manifest)
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp8_incremental_diff.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add ingest/incremental.py tests/test_sp8_incremental_diff.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(ingest): implement IncrementalIngestor core delta-diff driver" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```

---

### Task 5: Ingest Runner Wire-up and Config Settings

**Files:**
- Modify: `core/config.py`
- Modify: `ingest/run.py`

**Interfaces:**
- Consumes: `settings.incremental_enabled`
- Produces: CLI `--incremental` parameter routing execution blocks

- [ ] **Step 1: Write the failing test**
Create `tests/test_sp8_ingest_cli.py` checking parameter validation:
```python
import pytest
import subprocess
import sys

def test_ingest_script_accepts_incremental_flag():
    # Verify CLI parses without raising syntax error
    cmd = [sys.executable, "-m", "ingest.run", "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "--incremental" in res.stdout
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_sp8_ingest_cli.py`
Expected: FAIL (unrecognized arguments error when passing help)

- [ ] **Step 3: Modify files**
Update `core/config.py` to register Task 2/Task 3 parameters under `Settings`.
Update `ingest/run.py` parser options to support `--incremental`. Under running execution pathways, check if `incremental` is active and resolve through `IncrementalIngestor` rather than standard batch operations:
```python
    parser.add_argument("--incremental", action="store_true", help="Enable incremental document re-ingestion.")
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_sp8_ingest_cli.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add core/config.py ingest/run.py tests/test_sp8_ingest_cli.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "feat(ingest): wire incremental CLI parameters and Settings configuration keys" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```
