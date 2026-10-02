from cache.semantic_cache import answer_to_payload, scored_to_payload
from core.pipeline import RAGPipeline
from core.types import Answer, Chunk, ScoredChunk, Usage
from tests.cache.fake_cache import FakeSemanticCache


class _Embedder:
    model = "e"
    def embed_query(self, text): return [1.0, 0.0]
    def embed_documents(self, ts): return [[1.0, 0.0] for _ in ts]


class _Retriever:
    def __init__(self):
        self.embedder = _Embedder()
        self.calls = 0
    def retrieve(self, query):
        self.calls += 1
        return [ScoredChunk(chunk=Chunk(chunk_id="c1", doc_id="d1", text="ctx",
                            tenant_id="public"), score=0.9)]


class _Grounded:
    def __init__(self):
        self.calls = 0
    def generate(self, question, scored):
        self.calls += 1
        return Answer(text=f"ans:{question}", refused=False, usage=Usage())


def _settings():
    from core.config import Settings
    return Settings()


def _pipe(answer_cache, retrieval_cache):
    ret, gen = _Retriever(), _Grounded()
    p = RAGPipeline(ret, gen, _settings(), embedder=ret.embedder,
                    answer_cache=answer_cache, retrieval_cache=retrieval_cache)
    return p, ret, gen


def test_answer_hit_skips_retrieval_and_generation():
    ac, rc = FakeSemanticCache(), FakeSemanticCache()
    ac.store(tenant_id="public", collection_id=None, embedding=[1.0, 0.0],
             payload=answer_to_payload(Answer(text="cached", refused=False)),
             doc_ids=["d1"])
    p, ret, gen = _pipe(ac, rc)
    ans = p.answer("hello")
    assert ans.text == "cached"
    assert ret.calls == 0 and gen.calls == 0


def test_retrieval_hit_skips_retrieval_but_generates():
    ac, rc = FakeSemanticCache(), FakeSemanticCache()
    sc = [ScoredChunk(chunk=Chunk(chunk_id="c1", doc_id="d1", text="ctx",
          tenant_id="public"), score=0.9)]
    rc.store(tenant_id="public", collection_id=None, embedding=[1.0, 0.0],
             payload=scored_to_payload(sc), doc_ids=["d1"])
    p, ret, gen = _pipe(ac, rc)
    ans = p.answer("hello")
    assert ret.calls == 0 and gen.calls == 1
    assert ans.text == "ans:hello"


def test_full_miss_runs_both_and_populates_tiers():
    ac, rc = FakeSemanticCache(), FakeSemanticCache()
    p, ret, gen = _pipe(ac, rc)
    p.answer("hello")
    assert ret.calls == 1 and gen.calls == 1
    # both tiers now warm for the same query
    assert rc.lookup(tenant_id="public", collection_id=None, embedding=[1.0, 0.0]) is not None
    assert ac.lookup(tenant_id="public", collection_id=None, embedding=[1.0, 0.0]) is not None


def test_refused_answer_is_not_cached():
    ac, rc = FakeSemanticCache(), FakeSemanticCache()
    ret = _Retriever()
    class _RefusingGen:
        def generate(self, q, s): return Answer(text="no", refused=True, usage=Usage())
    p = RAGPipeline(ret, _RefusingGen(), _settings(), embedder=ret.embedder,
                    answer_cache=ac, retrieval_cache=rc)
    p.answer("hello")
    assert ac.lookup(tenant_id="public", collection_id=None, embedding=[1.0, 0.0]) is None


def test_no_cache_wired_is_a_total_bypass():
    ret, gen = _Retriever(), _Grounded()
    p = RAGPipeline(ret, gen, _settings(), embedder=ret.embedder,
                    answer_cache=None, retrieval_cache=None)
    ans = p.answer("hello")
    assert ret.calls == 1 and gen.calls == 1 and ans.text == "ans:hello"


# --- FIX 2: `core.pipeline.build()` must honor an explicit `enable_cache`
# override even when `settings.cache_enabled` says otherwise. Eval entry
# points rely on `enable_cache=False` to guarantee the cache never confounds
# metrics, regardless of what's set in the environment. Real `build()` is
# exercised end-to-end (not the FakeSemanticCache path above): the component
# builders it calls (`core.registry.build_embedder/build_vector_store/
# build_generator`) are monkeypatched to lightweight stubs so no network/infra
# is touched, and `version="baseline"` avoids the reranker/sparse builders.
# `build_cache()` itself never connects to Redis (see cache/_redisvl_backend.py
# — the redis-vl import is lazy, inside method bodies only), so exercising the
# `enable_cache=True/None` branch stays fully offline and import-isolated too.

class _StubEmbedder:
    model = "stub-embed"
    def embed_query(self, text): return [0.0]
    def embed_documents(self, texts): return [[0.0] for _ in texts]


class _StubVectorStore:
    def search(self, *a, **k): return []


class _StubGenerator:
    model = "stub-gen"


def _patch_component_builders(monkeypatch):
    import core.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "build_embedder", lambda settings=None: _StubEmbedder())
    monkeypatch.setattr(pipeline_mod, "build_vector_store", lambda settings=None: _StubVectorStore())
    monkeypatch.setattr(pipeline_mod, "build_generator",
                        lambda role="gen", settings=None: _StubGenerator())


def test_build_enable_cache_false_wires_no_cache_even_if_settings_enabled(monkeypatch):
    import core.pipeline as pipeline_mod
    from core.config import Settings

    _patch_component_builders(monkeypatch)
    s = Settings(cache_enabled=True)
    pipeline = pipeline_mod.build(version="baseline", settings=s, enable_cache=False)
    assert pipeline.answer_cache is None
    assert pipeline.retrieval_cache is None


def test_build_enable_cache_none_or_true_wires_cache_when_settings_enabled(monkeypatch):
    import core.pipeline as pipeline_mod
    from core.config import Settings

    _patch_component_builders(monkeypatch)
    s = Settings(cache_enabled=True)

    p_none = pipeline_mod.build(version="baseline", settings=s, enable_cache=None)
    assert p_none.answer_cache is not None
    assert p_none.retrieval_cache is not None

    p_true = pipeline_mod.build(version="baseline", settings=s, enable_cache=True)
    assert p_true.answer_cache is not None
    assert p_true.retrieval_cache is not None
