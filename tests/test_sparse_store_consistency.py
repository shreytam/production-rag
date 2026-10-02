import json
import multiprocessing as mp

from core.types import ACLContext, Chunk
from providers.sparse.bm25 import BM25Retriever, _tokenize
from providers.sparse.tenant_store import TenantSparseStore

ACL = ACLContext(tenant_id="t1")


def _c(cid, text, tenant="t1"):
    return Chunk(chunk_id=cid, doc_id=cid.split("::")[0], text=text, tenant_id=tenant)


def _ids(store, q):
    return [h.chunk.chunk_id for h in store.search(q, top_k=100, acl=ACL)]


def test_reader_sees_other_instance_add_and_delete(tmp_path):
    api = TenantSparseStore(index_dir=str(tmp_path))
    worker = TenantSparseStore(index_dir=str(tmp_path))
    assert _ids(api, "alpha") == []  # empty result must not be cached forever
    worker.add([_c("d1::0", "alpha")])
    assert _ids(api, "alpha") == ["d1::0"]
    worker.delete(["d1::0"], ACL)
    assert _ids(api, "alpha") == []


def test_two_writers_no_lost_updates(tmp_path):
    a = TenantSparseStore(index_dir=str(tmp_path))
    b = TenantSparseStore(index_dir=str(tmp_path))
    a.add([_c("d1::0", "alpha")])
    b.add([_c("d2::0", "alpha")])
    assert sorted(_ids(a, "alpha")) == ["d1::0", "d2::0"]
    assert sorted(_ids(b, "alpha")) == ["d1::0", "d2::0"]
    a.delete(["d1::0"], ACL)
    assert _ids(b, "alpha") == ["d2::0"]


def _worker(d, i):
    s = TenantSparseStore(index_dir=d)
    for j in range(5):
        s.add([_c(f"p{i}_{j}::0", "alpha")])


def test_concurrent_processes_no_lost_updates(tmp_path):
    ctx = mp.get_context("fork")
    ps = [ctx.Process(target=_worker, args=(str(tmp_path), i)) for i in range(4)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
        assert p.exitcode == 0
    assert len(_ids(TenantSparseStore(index_dir=str(tmp_path)), "alpha")) == 20


def test_zero_score_hits_dropped():
    r = BM25Retriever()
    r.index([_c("d1::0", "alpha beta"), _c("d2::0", "gamma delta"), _c("d3::0", "epsilon")])
    assert [h.chunk.chunk_id for h in r.search("alpha", 5, ACL)] == ["d1::0"]
    assert r.search("zzz", 5, ACL) == []


def test_tokenizer_strips_punctuation_and_is_unicode_aware():
    assert _tokenize("Revenue? (Q3) café-au-lait") == ["revenue", "q3", "café", "au", "lait"]
    r = BM25Retriever()
    r.index([_c("d1::0", "total revenue grew"), _c("d2::0", "other text")])
    assert [h.chunk.chunk_id for h in r.search("revenue?", 5, ACL)] == ["d1::0"]


def test_snapshot_is_json_not_pickle(tmp_path):
    s = TenantSparseStore(index_dir=str(tmp_path))
    s.add([_c("d1::0", "alpha")])
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())[0]["chunk_id"] == "d1::0"
    assert not list(tmp_path.glob("*.pkl"))


def test_real_match_in_tiny_corpus_survives_zero_idf():
    # Single-chunk corpus: Okapi IDF is 0, so score is 0 despite a true match.
    r = BM25Retriever()
    r.index([_c("d1::0", "alpha beta")])
    assert [h.chunk.chunk_id for h in r.search("alpha", 5, ACL)] == ["d1::0"]
