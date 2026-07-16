from core.types import ACLContext, Chunk
from providers.sparse.tenant_store import TenantSparseStore


def _c(cid, tenant, text):
    return Chunk(chunk_id=cid, doc_id=cid.split("::")[0], text=text, tenant_id=tenant)


def test_add_persists_across_instances(tmp_path):
    s1 = TenantSparseStore(index_dir=str(tmp_path))
    s1.add([_c("d1::0", "t1", "alpha beta")])
    # A fresh instance (new process) loads the persisted tenant index.
    s2 = TenantSparseStore(index_dir=str(tmp_path))
    hits = s2.search("alpha", top_k=5, acl=ACLContext(tenant_id="t1"))
    assert [h.chunk.chunk_id for h in hits] == ["d1::0"]


def test_delete_persists(tmp_path):
    s = TenantSparseStore(index_dir=str(tmp_path))
    s.add([_c("d1::0", "t1", "alpha"), _c("d1::1", "t1", "beta")])
    s.delete(["d1::0"], ACLContext(tenant_id="t1"))
    s2 = TenantSparseStore(index_dir=str(tmp_path))
    hits = s2.search("alpha beta", top_k=5, acl=ACLContext(tenant_id="t1"))
    assert [h.chunk.chunk_id for h in hits] == ["d1::1"]


def test_tenants_isolated_on_disk(tmp_path):
    s = TenantSparseStore(index_dir=str(tmp_path))
    s.add([_c("d1::0", "t1", "shared")])
    s.add([_c("d2::0", "t2", "shared")])
    hits = s.search("shared", top_k=5, acl=ACLContext(tenant_id="t1"))
    assert {h.chunk.tenant_id for h in hits} == {"t1"}


def test_malicious_tenant_id_cannot_escape_index_dir(tmp_path):
    """A JWT-derived tenant_id is only non-empty-validated — a signed token with
    a path-traversal tenant_id must not let the store write outside index_dir,
    and search must still work correctly for that same malicious tenant_id."""
    index_dir = tmp_path / "sparse"
    malicious_tenant = "../../evil"
    s = TenantSparseStore(index_dir=str(index_dir))
    s.add([_c("d1::0", malicious_tenant, "alpha beta")])

    resolved_index_dir = index_dir.resolve()
    # Nothing should have been written outside index_dir, including at tmp_path root.
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert resolved_index_dir in path.resolve().parents, (
                f"file escaped index_dir: {path}"
            )

    # Roundtrip search with the same malicious tenant_id still works.
    s2 = TenantSparseStore(index_dir=str(index_dir))
    hits = s2.search("alpha", top_k=5, acl=ACLContext(tenant_id=malicious_tenant))
    assert [h.chunk.chunk_id for h in hits] == ["d1::0"]
