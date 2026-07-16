from core.types import ChunkRecord, DocManifest
from providers.manifest.jsonl_store import JsonlManifestStore


def test_save_and_load_roundtrip(tmp_path):
    store = JsonlManifestStore(manifest_dir=str(tmp_path))
    m = DocManifest(tenant_id="t1", doc_id="d1", prompt_version="v1",
                    chunks={"c1": ChunkRecord(chunk_id="c1", ordinal=0,
                                              embed_hash="h1", meta_hash="h2")})
    store.save(m)
    loaded = store.load("t1", "d1")
    assert loaded is not None
    assert loaded.chunks["c1"].embed_hash == "h1"


def test_load_missing_returns_none(tmp_path):
    store = JsonlManifestStore(manifest_dir=str(tmp_path))
    assert store.load("t1", "nope") is None


def test_tenant_mismatch_fails_closed(tmp_path):
    store = JsonlManifestStore(manifest_dir=str(tmp_path))
    m = DocManifest(tenant_id="t1", doc_id="d1", prompt_version="v1", chunks={})
    store.save(m)
    assert store.load("t2", "d1") is None


def test_malicious_tenant_id_cannot_escape_manifest_dir(tmp_path):
    store = JsonlManifestStore(manifest_dir=str(tmp_path))
    m = DocManifest(tenant_id="../../evil", doc_id="../../x", prompt_version="v1", chunks={})
    store.save(m)
    # Every file written stays under manifest_dir.
    written = list(tmp_path.rglob("*.json"))
    assert written, "manifest should have been written"
    for p in written:
        assert str(p.resolve()).startswith(str(tmp_path.resolve()))
    # Roundtrip with the same (malicious) ids still works.
    assert store.load("../../evil", "../../x") is not None
