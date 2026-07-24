from dataclasses import dataclass

from ingest.worker import IngestDeps, run_delete, run_ingest


class _RecordingCache:
    def __init__(self): self.calls = []
    def lookup(self, **kw): return None
    def store(self, **kw): pass
    def invalidate_document(self, *, tenant_id, collection_id, doc_id):
        self.calls.append((tenant_id, collection_id, doc_id))
        return 1


@dataclass
class _Rec:
    tenant_id: str = "acme"
    collection_id: str = "kb"
    content_type: str = "text/plain"
    filename: str = "f.txt"
    blob_key: str = "acme/f"


class _Registry:
    def __init__(self, rec): self._rec = rec
    def get_privileged(self, _id): return self._rec
    def set_status(self, *a, **k): pass
    def delete(self, *a, **k): pass


class _Ingestor:
    def ingest_document(self, *a, **k): return 3
    def delete_document(self, *a, **k): pass


class _Blobs:
    def get(self, key): return b"hello"
    def delete(self, key): pass


class _Parser:
    def parse(self, *a, **k):
        from core.types import Document
        return [Document(doc_id="d1", text="hello", tenant_id="acme")]


class _Parsers:
    def resolve(self, _ct): return _Parser()


def _deps(caches, settings, collection_id="kb"):
    return IngestDeps(registry=_Registry(_Rec(collection_id=collection_id)), blobs=_Blobs(),
                      parsers=_Parsers(), ingestor=_Ingestor(), settings=settings, caches=caches)


def _settings():
    from core.config import Settings
    return Settings()


def test_run_delete_invalidates_both_tiers():
    a, r = _RecordingCache(), _RecordingCache()
    run_delete(_deps((a, r), _settings()), "d1")
    # doc's own collection partition first, then the unscoped "__none__"
    # partition — unscoped queries can cite docs from any collection.
    assert a.calls == [("acme", "kb", "d1"), ("acme", None, "d1")]
    assert r.calls == [("acme", "kb", "d1"), ("acme", None, "d1")]


def test_run_ingest_invalidates_both_tiers():
    a, r = _RecordingCache(), _RecordingCache()
    run_ingest(_deps((a, r), _settings()), "d1")
    assert a.calls == [("acme", "kb", "d1"), ("acme", None, "d1")]
    assert r.calls == [("acme", "kb", "d1"), ("acme", None, "d1")]


def test_collection_scoped_doc_also_evicts_unscoped_partition():
    # A doc that lives in a specific collection ("kb") can still be cited by
    # an UNSCOPED answer (collection_id=None retrieves across the whole
    # tenant) — that answer is cached under the "__none__" partition. Prove
    # the worker evicts that partition too, not just the doc's own.
    a, r = _RecordingCache(), _RecordingCache()
    run_delete(_deps((a, r), _settings()), "d1")
    scopes_a = {c[1] for c in a.calls}
    scopes_r = {c[1] for c in r.calls}
    assert None in scopes_a
    assert None in scopes_r


def test_unscoped_doc_evicts_only_one_partition_no_duplicate():
    # A doc whose collection_id is already falsy ("") normalizes to the same
    # "__none__" partition as the explicit-None eviction — must not evict
    # twice for the same scope.
    a, r = _RecordingCache(), _RecordingCache()
    run_delete(_deps((a, r), _settings(), collection_id=""), "d1")
    assert a.calls == [("acme", None, "d1")]
    assert r.calls == [("acme", None, "d1")]


def test_no_caches_is_noop():
    # caches=None must not raise
    run_delete(_deps(None, _settings()), "d1")
