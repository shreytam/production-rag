# VDB-Decision · Vector Database Selection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Documents vector database choices, provides store isolation parity checks, and modifies reference schemas.

**Architecture:** Create parameterized store contract parity tests ensuring cross-tenant isolation and open visibility boundaries, and update specifications.

**Tech Stack:** Python 3.11-3.13, Pydantic, pytest.

## Global Constraints
- Every commit must be authored solely under the repository owner's identity: `Shreytam Goyal <shreytamgoyal@gmail.com>`.
- Nothing may be attributed to Claude. Do NOT add a `Co-Authored-By:` trailer, a `Claude-Session:` line, a "Generated with Claude" note, or any other AI/Anthropic attribution in commit messages, PR titles, or PR descriptions.
- Do not use the `shreytam.goyal@codiant.com` (codiant) identity for commits.
- Commit messages describe the change only — no AI-attribution footers of any kind.
- The `.cache/` directory must be ignored.
- Store isolation parity tests must execute fully offline.

---

### Task 1: Store Multi-Tenancy Isolation Parity Tests

**Files:**
- Create: `tests/test_vectorstore_parity.py`

**Interfaces:**
- Consumes: `build_vector_store`
- Produces: Parameterized test routines enforcing cross-tenant boundaries

- [ ] **Step 1: Write the tests**
Create `tests/test_vectorstore_parity.py` enforcing strict tenant filter boundaries on all registered stores:
```python
import pytest
from core.types import ACLContext, Chunk
from core.registry import build_vector_store
from core.config import Settings

@pytest.mark.parametrize("store_type", ["qdrant"])
def test_store_isolation_leaks_none(store_type):
    # Retrieve store instances
    settings = Settings(vector_store=store_type)
    store = build_vector_store(settings)
    store.ensure_collection(dimension=16)
    
    # Ingest tenant_b chunk
    c_b = Chunk(
        chunk_id="c_b",
        doc_id="doc_b",
        text="secret contents",
        tenant_id="tenant_b",
        acl_tags=[],
        embedding=[0.1] * 16
    )
    store.upsert([c_b])
    
    # Query with tenant_a ACL; must return 0 results (no leaking)
    acl_a = ACLContext(tenant_id="tenant_a", tags=[])
    res = store.search([0.1] * 16, top_k=5, acl=acl_a)
    assert len(res) == 0
```

- [ ] **Step 2: Run test to verify it passes**
Run: `pytest tests/test_vectorstore_parity.py`
Expected: PASS (Qdrant in-memory should enforce isolation successfully)

- [ ] **Step 3: Commit**
```bash
git add tests/test_vectorstore_parity.py
git -c user.name="Shreytam Goyal" -c user.email="shreytamgoyal@gmail.com" commit -m "test(vdb): add multi-tenant vectorstore isolation parity contract checks" --author="Shreytam Goyal <shreytamgoyal@gmail.com>"
```
