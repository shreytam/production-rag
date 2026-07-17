from core.pipeline import build
from core.config import Settings
from core.types import ACLContext
from providers.sparse.tenant_store import TenantSparseStore


def test_pipeline_full_resolves_tenant_sparse_store_at_query_time(tmp_path):
    """build(version="full") no longer binds a corpus-pickle sparse index at
    build time (the old empty-sparse HybridIndexError/warning gate is gone).
    Instead its sparse retriever is a TenantSparseStore that is always present
    and resolves the caller's tenant index at query time, returning no hits
    for a tenant that has never been indexed."""
    settings = Settings(
        tenant_sparse_dir=str(tmp_path),
        hybrid_require_sparse=True,
        nvidia_api_key="mock-key",
    )
    pipeline = build(version="full", settings=settings)
    assert pipeline is not None
    assert isinstance(pipeline.retriever.sparse, TenantSparseStore)
    assert pipeline.retriever.sparse.search(
        "anything", top_k=5, acl=ACLContext(tenant_id="unknown-tenant")
    ) == []
