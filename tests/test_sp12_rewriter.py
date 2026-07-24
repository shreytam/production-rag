from core.types import ACLContext
from providers.rewriter.hybrid_rewriter import HybridQueryRewriter
from tests._fakes import RecordingGenerator


class FakeRedis:
    """Minimal hgetall-only stand-in; maps key -> {field: value}."""

    def __init__(self, data):
        self._data = data

    def hgetall(self, key):
        return dict(self._data.get(key, {}))


def _acl(t="t1"):
    return ACLContext(tenant_id=t)


def test_synonym_substitution_word_boundary():
    r = FakeRedis({"rewriter:synonyms:t1": {"NYPD": "New York Police Department"}})
    rw = HybridQueryRewriter(RecordingGenerator(), "redis://x", llm_enabled=False, redis_client=r)
    assert rw.rewrite("who leads NYPD?", _acl("t1")) == "who leads New York Police Department?"
    # Substring guard: "NYPDX" must not match.
    assert rw.rewrite("NYPDX status", _acl("t1")) == "NYPDX status"


def test_tenant_synonym_isolation():
    r = FakeRedis({"rewriter:synonyms:t2": {"Jupiter": "Jupiter-Next"}})
    rw = HybridQueryRewriter(RecordingGenerator(), "redis://x", llm_enabled=False, redis_client=r)
    # Tenant t1 has no dictionary — t2's mapping must not leak.
    assert rw.rewrite("Jupiter roadmap", _acl("t1")) == "Jupiter roadmap"
    assert rw.rewrite("Jupiter roadmap", _acl("t2")) == "Jupiter-Next roadmap"


def test_llm_expansion_triggers_only_over_threshold_without_synonym():
    r = FakeRedis({})
    gen = RecordingGenerator(text="expanded descriptive query")
    rw = HybridQueryRewriter(gen, "redis://x", llm_enabled=True, llm_threshold=5, redis_client=r)
    # 3 words < threshold -> no LLM, raw returned.
    assert rw.rewrite("short simple query", _acl()) == "short simple query"
    assert len(gen.calls) == 0
    # 5 words >= threshold, no synonym match -> LLM expansion.
    assert rw.rewrite("please find the sales reports", _acl()) == "expanded descriptive query"
    assert len(gen.calls) == 1


def test_synonym_match_suppresses_llm():
    r = FakeRedis({"rewriter:synonyms:t1": {"reports": "quarterly financial reports"}})
    gen = RecordingGenerator(text="SHOULD NOT RUN")
    rw = HybridQueryRewriter(gen, "redis://x", llm_enabled=True, llm_threshold=3, redis_client=r)
    out = rw.rewrite("please find the sales reports", _acl("t1"))
    assert "quarterly financial reports" in out
    assert len(gen.calls) == 0


def test_fail_soft_on_redis_error():
    class BoomRedis:
        def hgetall(self, key):
            raise RuntimeError("redis down")

    rw = HybridQueryRewriter(
        RecordingGenerator(), "redis://x", llm_enabled=False, redis_client=BoomRedis()
    )
    assert rw.rewrite("hello world", _acl()) == "hello world"  # no raise, raw returned


def test_fail_soft_on_llm_error():
    class BoomGen:
        def complete(self, *a, **k):
            raise RuntimeError("llm down")

    r = FakeRedis({})
    rw = HybridQueryRewriter(BoomGen(), "redis://x", llm_enabled=True, llm_threshold=1, redis_client=r)
    assert rw.rewrite("expand me please now", _acl()) == "expand me please now"
