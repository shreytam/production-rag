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
    store = TenantSparseStore(index_dir=str(tmp_path))
    malicious = "../../evil"
    # The resolved on-disk path for a malicious tenant_id stays inside index_dir.
    resolved = store._path(malicious).resolve()
    assert str(resolved).startswith(str(store._dir.resolve()) + "/") or (
        resolved.parent == store._dir.resolve()
    )

    # And a save+search roundtrip with the malicious id still works (and stays contained).
    store.add([_c("d1::0", malicious, "alpha beta")])
    hits = store.search("alpha", top_k=5, acl=ACLContext(tenant_id=malicious))
    assert [h.chunk.chunk_id for h in hits] == ["d1::0"]
