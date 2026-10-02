from core.config import Settings
from core.registry import build_embedder, build_generator, for_query_path


def test_query_timeout_settings_defaults():
    s = Settings()
    assert s.query_request_timeout_seconds == 30.0
    assert s.query_max_retries == 1
    # Ingest-time knobs are untouched.
    assert s.request_timeout_seconds == 600.0
    assert s.max_retries == 5


def test_for_query_path_overrides_only_timeout_and_retries():
    s = Settings()
    q = for_query_path(s)
    assert q.request_timeout_seconds == s.query_request_timeout_seconds
    assert q.max_retries == s.query_max_retries
    assert q.gen_model == s.gen_model
    assert s.request_timeout_seconds == 600.0  # original not mutated


def test_query_path_clients_use_bounded_timeouts():
    q = for_query_path(Settings(gen_api_key="k", embed_api_key="k"))
    gen = build_generator("gen", q)
    assert gen._client.timeout == 30.0
    assert gen._client.max_retries == 1
    emb = build_embedder(q)
    assert emb._client.timeout == 30.0
    assert emb._client.max_retries == 1


def test_ingest_path_clients_keep_generous_timeouts():
    s = Settings(context_api_key="k", embed_api_key="k")
    assert build_generator("context", s)._client.timeout == 600.0
    assert build_embedder(s)._client.max_retries == 5


def test_anthropic_generator_has_timeout():
    from providers.generators.anthropic import AnthropicGenerator

    g = AnthropicGenerator("m", "k", timeout=12.0, max_retries=1)
    assert g._client.timeout == 12.0
    assert g._client.max_retries == 1


def test_anthropic_registry_passes_timeout():
    s = Settings(gen_provider="anthropic", anthropic_api_key="k")
    g = build_generator("gen", for_query_path(s))
    assert g._client.timeout == 30.0


def test_rewriter_redis_client_has_socket_timeouts(monkeypatch):
    import redis

    from providers.rewriter.hybrid_rewriter import HybridQueryRewriter
    from tests._fakes import RecordingGenerator

    seen = {}
    monkeypatch.setattr(redis, "from_url", lambda url, **kw: seen.update(kw) or object())
    HybridQueryRewriter(RecordingGenerator(), "redis://x")._get_client()
    assert 0 < seen["socket_timeout"] <= 2.0
    assert 0 < seen["socket_connect_timeout"] <= 2.0


def test_pipeline_build_uses_query_settings(monkeypatch):
    import core.pipeline as pipeline_mod

    captured = {}

    class _Stub:
        model = "m"
        dimension = 4

        def complete(self, *a, **k):
            raise AssertionError

    def _emb(settings=None):
        captured["embed"] = settings.request_timeout_seconds
        return _Stub()

    def _gen(role="gen", settings=None):
        captured[role] = settings.request_timeout_seconds
        return _Stub()

    monkeypatch.setattr(pipeline_mod, "build_embedder", _emb)
    monkeypatch.setattr(pipeline_mod, "build_vector_store", lambda settings=None: _Stub())
    monkeypatch.setattr(pipeline_mod, "build_generator", _gen)
    pipeline_mod.build(
        version="baseline",
        settings=Settings(rewriter_enabled=True),
        enable_guardrails=False,
        enable_cache=False,
        enable_rewriter=True,
    )
    assert captured == {"embed": 30.0, "gen": 30.0, "context": 30.0}
