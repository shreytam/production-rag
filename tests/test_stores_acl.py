"""Tests for vector stores and ACL enforcement.

Offline (always run):
  - BM25 ACL isolation: cross-tenant chunks never returned; tag scoping respected.
  - Filter builder unit tests: qdrant_filter, pg_where, acl_predicate.

Live (skipped if server unreachable):
  - QdrantVectorStore round-trip: upsert + search + ACL isolation.
  - PgVectorStore round-trip: upsert + search + ACL isolation.
"""

from __future__ import annotations

import math
import pytest

from core.types import ACLContext, Chunk, RetrievalSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    chunk_id: str,
    tenant_id: str,
    text: str,
    acl_tags: tuple[str, ...] = (),
    embedding: list[float] | None = None,
    dim: int = 4,
) -> Chunk:
    emb = embedding or [0.1] * dim
    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        tenant_id=tenant_id,
        acl_tags=acl_tags,
        text=text,
        embedding=emb,
    )


def _unit_vec(dim: int, index: int) -> list[float]:
    """Unit vector along the `index` axis — cosine similarity = 1 with itself."""
    v = [0.0] * dim
    v[index] = 1.0
    return v


# ---------------------------------------------------------------------------
# BM25 ACL isolation tests (must pass offline)
# ---------------------------------------------------------------------------

class TestBM25ACL:
    """BM25 retriever isolation and tag-scoping tests — fully offline."""

    def _build_retriever(self):
        from providers.sparse.bm25 import BM25Retriever

        chunks = [
            # tenant_a: open chunks
            _make_chunk("a1", "tenant_a", "machine learning neural network deep learning"),
            _make_chunk("a2", "tenant_a", "information retrieval search ranking bm25"),
            # tenant_b: a chunk with text that strongly matches the query
            # (a naive non-isolated impl would return this)
            _make_chunk("b1", "tenant_b", "machine learning neural network deep learning model"),
            _make_chunk("b2", "tenant_b", "vector database embeddings similarity search"),
            # tenant_a: tag-scoped chunk (requires tag "secret")
            _make_chunk("a3", "tenant_a", "machine learning confidential secret project", acl_tags=("secret",)),
        ]
        r = BM25Retriever()
        r.index(chunks)
        return r

    def test_cross_tenant_isolation(self):
        """Searching tenant_a returns ZERO tenant_b chunks."""
        r = self._build_retriever()
        acl = ACLContext(tenant_id="tenant_a")
        results = r.search("machine learning neural network", top_k=10, acl=acl)
        assert results, "Should return at least one result for tenant_a"
        for sc in results:
            assert sc.chunk.tenant_id == "tenant_a", (
                f"Cross-tenant leak: got chunk {sc.chunk.chunk_id} from {sc.chunk.tenant_id}"
            )

    def test_cross_tenant_isolation_reverse(self):
        """Searching tenant_b returns ZERO tenant_a chunks."""
        r = self._build_retriever()
        acl = ACLContext(tenant_id="tenant_b")
        results = r.search("machine learning neural network", top_k=10, acl=acl)
        for sc in results:
            assert sc.chunk.tenant_id == "tenant_b", (
                f"Cross-tenant leak: got chunk {sc.chunk.chunk_id} from {sc.chunk.tenant_id}"
            )

    def test_tag_scoped_chunk_hidden_from_untagged_caller(self):
        """A caller with no tags cannot see a tag-restricted chunk."""
        r = self._build_retriever()
        acl = ACLContext(tenant_id="tenant_a")  # no tags
        results = r.search("machine learning confidential secret project", top_k=10, acl=acl)
        ids = {sc.chunk.chunk_id for sc in results}
        assert "a3" not in ids, "Tag-restricted chunk a3 must not be visible to untagged caller"

    def test_tag_scoped_chunk_visible_to_matching_caller(self):
        """A caller holding the matching tag sees the tag-restricted chunk."""
        r = self._build_retriever()
        acl = ACLContext(tenant_id="tenant_a", acl_tags=("secret",))
        results = r.search("machine learning confidential secret project", top_k=10, acl=acl)
        ids = {sc.chunk.chunk_id for sc in results}
        assert "a3" in ids, "Tag-restricted chunk a3 should be visible to caller with 'secret' tag"

    def test_missing_tenant_returns_empty(self):
        """Querying a tenant with no indexed chunks returns empty list."""
        r = self._build_retriever()
        acl = ACLContext(tenant_id="tenant_nonexistent")
        results = r.search("anything", top_k=5, acl=acl)
        assert results == []

    def test_sparse_source_label(self):
        """Results are labelled SPARSE."""
        r = self._build_retriever()
        acl = ACLContext(tenant_id="tenant_a")
        results = r.search("machine learning", top_k=5, acl=acl)
        for sc in results:
            assert sc.source == RetrievalSource.SPARSE

    def test_rank_is_set(self):
        """Results have 1-indexed rank."""
        r = self._build_retriever()
        acl = ACLContext(tenant_id="tenant_a")
        results = r.search("retrieval search", top_k=5, acl=acl)
        ranks = [sc.rank for sc in results]
        assert ranks == list(range(1, len(results) + 1))


# ---------------------------------------------------------------------------
# Filter builder unit tests (must pass offline)
# ---------------------------------------------------------------------------

class TestQdrantFilterBuilder:
    """Unit tests for qdrant_filter() — no server required."""

    def test_returns_filter_object(self):
        from qdrant_client import models as qm
        from retrieval.acl import qdrant_filter

        acl = ACLContext(tenant_id="tenant_x", acl_tags=("admin",))
        f = qdrant_filter(acl)
        assert isinstance(f, qm.Filter)

    def test_must_contains_tenant_condition(self):
        from qdrant_client import models as qm
        from retrieval.acl import qdrant_filter

        acl = ACLContext(tenant_id="tenant_x")
        f = qdrant_filter(acl)
        assert f.must is not None

        # Find the FieldCondition for tenant_id
        tenant_conds = [
            c for c in f.must
            if isinstance(c, qm.FieldCondition) and c.key == "tenant_id"
        ]
        assert len(tenant_conds) == 1
        assert tenant_conds[0].match.value == "tenant_x"

    def test_visibility_should_contains_acl_open(self):
        """The visibility sub-filter always has an acl_open == True branch."""
        from qdrant_client import models as qm
        from retrieval.acl import qdrant_filter

        acl = ACLContext(tenant_id="tenant_x")
        f = qdrant_filter(acl)

        # The second must-clause is a nested Filter with should
        nested = [c for c in f.must if isinstance(c, qm.Filter)]
        assert nested, "Expected a nested Filter for visibility"
        should_conds = nested[0].should
        open_conds = [
            c for c in should_conds
            if isinstance(c, qm.FieldCondition) and c.key == "acl_open"
        ]
        assert open_conds, "acl_open condition must be in should"
        assert open_conds[0].match.value is True

    def test_no_tag_caller_has_no_tag_overlap_branch(self):
        """No-tag caller gets only the acl_open branch (no MatchAny on acl_tags)."""
        from qdrant_client import models as qm
        from retrieval.acl import qdrant_filter

        acl = ACLContext(tenant_id="tenant_x")  # no tags
        f = qdrant_filter(acl)
        nested = [c for c in f.must if isinstance(c, qm.Filter)]
        should_conds = nested[0].should
        tag_conds = [
            c for c in should_conds
            if isinstance(c, qm.FieldCondition) and c.key == "acl_tags"
        ]
        assert not tag_conds, "No-tag caller should have no tag-overlap branch"

    def test_tagged_caller_has_tag_overlap_branch(self):
        """Tagged caller gets both acl_open AND MatchAny(acl_tags) branches."""
        from qdrant_client import models as qm
        from retrieval.acl import qdrant_filter

        acl = ACLContext(tenant_id="tenant_x", acl_tags=("admin", "reader"))
        f = qdrant_filter(acl)
        nested = [c for c in f.must if isinstance(c, qm.Filter)]
        should_conds = nested[0].should
        tag_conds = [
            c for c in should_conds
            if isinstance(c, qm.FieldCondition) and c.key == "acl_tags"
        ]
        assert tag_conds, "Tagged caller should have tag-overlap branch"
        assert isinstance(tag_conds[0].match, qm.MatchAny)
        assert set(tag_conds[0].match.any) == {"admin", "reader"}


class TestPgWhereBuilder:
    """Unit tests for pg_where() — no server required."""

    def test_returns_tuple(self):
        from retrieval.acl import pg_where

        result = pg_where(ACLContext(tenant_id="t1"))
        assert isinstance(result, tuple) and len(result) == 2

    def test_fragment_content(self):
        from retrieval.acl import pg_where

        fragment, params = pg_where(ACLContext(tenant_id="t1", acl_tags=("tag_a",)))
        assert "tenant_id = %s" in fragment
        assert "cardinality(acl_tags)=0" in fragment
        assert "acl_tags && %s" in fragment

    def test_params_tenant_first(self):
        from retrieval.acl import pg_where

        fragment, params = pg_where(ACLContext(tenant_id="my_tenant", acl_tags=("x",)))
        assert params[0] == "my_tenant"
        assert "x" in params[1]

    def test_params_empty_tags(self):
        from retrieval.acl import pg_where

        fragment, params = pg_where(ACLContext(tenant_id="t"))
        assert params[1] == []

    def test_exact_fragment(self):
        """Fragment matches the documented form exactly."""
        from retrieval.acl import pg_where

        fragment, params = pg_where(ACLContext(tenant_id="t"))
        assert fragment == "tenant_id = %s AND (cardinality(acl_tags)=0 OR acl_tags && %s)"


class TestAclPredicate:
    """Unit tests for acl_predicate() — no server required."""

    def test_allows_same_tenant_open_chunk(self):
        from retrieval.acl import acl_predicate

        acl = ACLContext(tenant_id="t1")
        chunk = _make_chunk("c1", "t1", "hello")
        assert acl_predicate(acl)(chunk) is True

    def test_denies_cross_tenant(self):
        from retrieval.acl import acl_predicate

        acl = ACLContext(tenant_id="t1")
        chunk = _make_chunk("c2", "t2", "hello")
        assert acl_predicate(acl)(chunk) is False

    def test_denies_tag_scoped_no_matching_tag(self):
        from retrieval.acl import acl_predicate

        acl = ACLContext(tenant_id="t1", acl_tags=("reader",))
        chunk = _make_chunk("c3", "t1", "secret", acl_tags=("admin",))
        assert acl_predicate(acl)(chunk) is False

    def test_allows_tag_scoped_matching_tag(self):
        from retrieval.acl import acl_predicate

        acl = ACLContext(tenant_id="t1", acl_tags=("admin", "reader"))
        chunk = _make_chunk("c4", "t1", "secret", acl_tags=("admin",))
        assert acl_predicate(acl)(chunk) is True

    def test_denies_cross_tenant_even_with_matching_tags(self):
        from retrieval.acl import acl_predicate

        acl = ACLContext(tenant_id="t1", acl_tags=("admin",))
        chunk = _make_chunk("c5", "t2", "secret", acl_tags=("admin",))
        assert acl_predicate(acl)(chunk) is False


# ---------------------------------------------------------------------------
# Live store tests (skipped if server unreachable)
# ---------------------------------------------------------------------------

DIM = 4  # tiny embedding dimension for tests


def _dummy_chunks_for_live(tenant_a: str, tenant_b: str) -> list[Chunk]:
    """Two tenants, distinct orthogonal embeddings."""
    return [
        # tenant_a: open chunk
        Chunk(
            chunk_id="live-a1", doc_id="doc-a1",
            tenant_id=tenant_a, text="open chunk tenant a",
            embedding=_unit_vec(DIM, 0),
        ),
        # tenant_a: tag-scoped chunk
        Chunk(
            chunk_id="live-a2", doc_id="doc-a2",
            tenant_id=tenant_a, acl_tags=("secret",), text="secret chunk tenant a",
            embedding=_unit_vec(DIM, 1),
        ),
        # tenant_b: open chunk (must not appear in tenant_a searches)
        Chunk(
            chunk_id="live-b1", doc_id="doc-b1",
            tenant_id=tenant_b, text="open chunk tenant b",
            embedding=_unit_vec(DIM, 0),  # same direction — would rank high without ACL
        ),
    ]


class TestQdrantVectorStoreLive:
    """Round-trip tests against a live Qdrant server."""

    COLLECTION = "test_acl_qdrant"

    @pytest.fixture(autouse=True)
    def skip_if_offline(self, require_live_or_fail):
        """Skip the entire class if Qdrant is unreachable."""
        reachable = True
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url="http://localhost:6333")
            client.get_collections()  # will raise if server is down
        except Exception:
            reachable = False

        require_live_or_fail(reachable, "Qdrant")

    @pytest.fixture
    def store(self):
        from core.config import Settings
        from providers.vectorstores.qdrant_store import QdrantVectorStore

        settings = Settings(qdrant_collection=self.COLLECTION)
        store = QdrantVectorStore(settings)
        store.ensure_collection(dimension=DIM)
        # Clean slate
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url="http://localhost:6333")
            client.delete_collection(self.COLLECTION)
        except Exception:
            pass
        store.ensure_collection(dimension=DIM)
        return store

    def test_upsert_and_search_returns_own_tenant(self, store):
        chunks = _dummy_chunks_for_live("qa", "qb")
        store.upsert(chunks)

        acl = ACLContext(tenant_id="qa")
        results = store.search(_unit_vec(DIM, 0), top_k=10, acl=acl)
        assert results, "Should find at least one chunk"
        for sc in results:
            assert sc.chunk.tenant_id == "qa"

    def test_search_returns_zero_cross_tenant(self, store):
        chunks = _dummy_chunks_for_live("qa", "qb")
        store.upsert(chunks)

        # Query as tenant_a with embedding aligned to tenant_b's open chunk
        acl = ACLContext(tenant_id="qa")
        results = store.search(_unit_vec(DIM, 0), top_k=10, acl=acl)
        ids = {sc.chunk.chunk_id for sc in results}
        assert "live-b1" not in ids, "Cross-tenant chunk must not appear in tenant_a results"

    def test_tag_scoped_hidden_from_untagged_caller(self, store):
        chunks = _dummy_chunks_for_live("qa", "qb")
        store.upsert(chunks)

        acl = ACLContext(tenant_id="qa")  # no tags
        results = store.search(_unit_vec(DIM, 1), top_k=10, acl=acl)
        ids = {sc.chunk.chunk_id for sc in results}
        assert "live-a2" not in ids, "Tag-restricted chunk must not appear for untagged caller"

    def test_tag_scoped_visible_to_tagged_caller(self, store):
        chunks = _dummy_chunks_for_live("qa", "qb")
        store.upsert(chunks)

        acl = ACLContext(tenant_id="qa", acl_tags=("secret",))
        results = store.search(_unit_vec(DIM, 1), top_k=10, acl=acl)
        ids = {sc.chunk.chunk_id for sc in results}
        assert "live-a2" in ids, "Tag-restricted chunk must appear for caller with matching tag"

    def test_count_with_acl(self, store):
        chunks = _dummy_chunks_for_live("qa", "qb")
        store.upsert(chunks)

        count_a = store.count(ACLContext(tenant_id="qa"))
        count_b = store.count(ACLContext(tenant_id="qb"))
        # tenant_a has 1 open chunk visible without tags; tenant_b has 1 open chunk
        assert count_a >= 1
        assert count_b >= 1
        assert count_a + count_b <= store.count()  # total >= per-tenant

    def test_dense_source_label(self, store):
        chunks = _dummy_chunks_for_live("qa", "qb")
        store.upsert(chunks)
        results = store.search(_unit_vec(DIM, 0), top_k=5, acl=ACLContext(tenant_id="qa"))
        for sc in results:
            assert sc.source == RetrievalSource.DENSE

    def _chunk(self, chunk_id: str, *, tenant: str, collection_id: str, text: str) -> Chunk:
        """Build an embedded Chunk with the given collection_id (mirrors _dummy_chunks_for_live)."""
        return Chunk(
            chunk_id=chunk_id,
            doc_id=f"doc-{chunk_id}",
            tenant_id=tenant,
            collection_id=collection_id,
            text=text,
            embedding=self.embed(text),
        )

    def embed(self, text: str) -> list[float]:
        """Deterministic tiny embedding for tests: hash text into a unit-ish vector."""
        v = [0.0] * DIM
        for i, ch in enumerate(text):
            v[i % DIM] += ord(ch)
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def test_collection_scoping(self, store):
        from core.types import ACLContext
        a = self._chunk("cola", tenant="t", collection_id="A", text="shared alpha")
        b = self._chunk("colb", tenant="t", collection_id="B", text="shared beta")
        store.upsert([a, b])
        acl = ACLContext(tenant_id="t")
        hits = store.search(self.embed("shared"), 5, acl, collection_id="A")
        assert {h.chunk.chunk_id for h in hits} == {"cola"}


class TestPgVectorStoreLive:
    """Round-trip tests against a live PostgreSQL + pgvector server."""

    TABLE = "test_acl_pg"

    @pytest.fixture(autouse=True)
    def skip_if_offline(self, require_live_or_fail):
        """Skip the entire class if Postgres is unreachable."""
        reachable = True
        try:
            import psycopg
            conn = psycopg.connect("postgresql://rag:rag@localhost:5432/rag", connect_timeout=2)
            conn.close()
        except Exception:
            reachable = False

        require_live_or_fail(reachable, "Postgres")

    @pytest.fixture
    def store(self):
        from core.config import Settings
        from providers.vectorstores.pgvector_store import PgVectorStore
        import psycopg

        settings = Settings(pg_table=self.TABLE)
        store = PgVectorStore(settings)

        # Drop and recreate for a clean slate
        conn = psycopg.connect(settings.pg_dsn)
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.TABLE}")
        conn.commit()
        conn.close()

        store.ensure_collection(dimension=DIM)
        return store

    def test_upsert_and_search_returns_own_tenant(self, store):
        chunks = _dummy_chunks_for_live("pa", "pb")
        store.upsert(chunks)

        acl = ACLContext(tenant_id="pa")
        results = store.search(_unit_vec(DIM, 0), top_k=10, acl=acl)
        assert results, "Should find at least one chunk"
        for sc in results:
            assert sc.chunk.tenant_id == "pa"

    def test_search_returns_zero_cross_tenant(self, store):
        chunks = _dummy_chunks_for_live("pa", "pb")
        store.upsert(chunks)

        acl = ACLContext(tenant_id="pa")
        results = store.search(_unit_vec(DIM, 0), top_k=10, acl=acl)
        ids = {sc.chunk.chunk_id for sc in results}
        assert "live-b1" not in ids, "Cross-tenant chunk must not appear in tenant_a results"

    def test_tag_scoped_hidden_from_untagged_caller(self, store):
        chunks = _dummy_chunks_for_live("pa", "pb")
        store.upsert(chunks)

        acl = ACLContext(tenant_id="pa")  # no tags
        results = store.search(_unit_vec(DIM, 1), top_k=10, acl=acl)
        ids = {sc.chunk.chunk_id for sc in results}
        assert "live-a2" not in ids

    def test_tag_scoped_visible_to_tagged_caller(self, store):
        chunks = _dummy_chunks_for_live("pa", "pb")
        store.upsert(chunks)

        acl = ACLContext(tenant_id="pa", acl_tags=("secret",))
        results = store.search(_unit_vec(DIM, 1), top_k=10, acl=acl)
        ids = {sc.chunk.chunk_id for sc in results}
        assert "live-a2" in ids

    def test_count_with_acl(self, store):
        chunks = _dummy_chunks_for_live("pa", "pb")
        store.upsert(chunks)

        count_a = store.count(ACLContext(tenant_id="pa"))
        count_total = store.count()
        assert count_a >= 1
        assert count_total >= count_a

    def test_dense_source_label(self, store):
        chunks = _dummy_chunks_for_live("pa", "pb")
        store.upsert(chunks)
        results = store.search(_unit_vec(DIM, 0), top_k=5, acl=ACLContext(tenant_id="pa"))
        for sc in results:
            assert sc.source == RetrievalSource.DENSE
