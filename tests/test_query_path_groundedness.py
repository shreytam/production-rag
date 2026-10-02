import logging
import threading
import time

import pytest

import guardrails.output_groundedness as og
from core.types import Answer, GuardrailAction, LLMResponse, Usage
from generation.metrics import faithfulness
from guardrails.output_groundedness import GroundednessGuardrail
from tests.test_guardrails import FakeGenerator, _resp


def _ctx(**kw):
    return {"contexts": ["ctx"], "question": "Q?", "answer": Answer(text="a"), **kw}


def test_refused_answer_with_contexts_passes_without_judge_call():
    g = GroundednessGuardrail(generator=FakeGenerator([]))  # any call raises
    ans = Answer(text="I cannot answer from the context.", refused=True)
    res = g.check(ans.text, context={"contexts": ["ctx"], "answer": ans})
    assert res.action == GuardrailAction.PASS
    assert not res.metadata.get("groundedness_unverified")


def test_zero_claims_is_unverified_not_blocked():
    gen = FakeGenerator([_resp({"claims": []})])
    res = GroundednessGuardrail(generator=gen).check("a", context=_ctx())
    assert res.action == GuardrailAction.PASS
    assert res.metadata["groundedness_unverified"] is True


def test_judge_parse_failure_is_unverified_not_blocked():
    bad = LLMResponse(text="not json", parsed=None, usage=Usage(), model="fake")
    res = GroundednessGuardrail(generator=FakeGenerator([bad])).check("a", context=_ctx())
    assert res.action == GuardrailAction.PASS
    assert res.metadata["groundedness_unverified"] is True


def test_verdict_parse_failure_is_unverified():
    bad = LLMResponse(text="x", parsed=None, usage=Usage(), model="fake")
    gen = FakeGenerator([_resp({"claims": ["c1", "c2"]}), bad])
    res = GroundednessGuardrail(generator=gen).check("a", context=_ctx())
    assert res.action == GuardrailAction.PASS
    assert res.metadata["groundedness_unverified"] is True


def test_missing_verdicts_count_as_unsupported():
    gen = FakeGenerator([
        _resp({"claims": ["c1", "c2", "c3", "c4"]}),
        _resp({"verdicts": [{"claim": "c1", "supported": True}]}),
    ])
    assert faithfulness("q", "a", ["ctx"], gen) == pytest.approx(0.25)


def test_judge_prompts_frame_content_as_untrusted():
    seen = []

    class Rec(FakeGenerator):
        def complete(self, messages, **kw):
            seen.append(" ".join(m.content for m in messages).lower())
            return super().complete(messages, **kw)

    gen = Rec([
        _resp({"claims": ["c1"]}),
        _resp({"verdicts": [{"claim": "c1", "supported": True}]}),
    ])
    faithfulness("q", "a", ["ctx"], gen)
    assert all("untrusted" in s for s in seen)


def test_saturated_pool_is_unverified_immediately_and_logged(monkeypatch, caplog):
    sem = threading.BoundedSemaphore(1)
    sem.acquire()  # saturated
    monkeypatch.setattr(og, "_GROUNDEDNESS_SLOTS", sem)
    called = []
    monkeypatch.setattr(og, "faithfulness", lambda **kw: called.append(1) or 1.0)
    t0 = time.perf_counter()
    with caplog.at_level(logging.WARNING):
        res = GroundednessGuardrail(generator=object()).check("a", context=_ctx())
    assert time.perf_counter() - t0 < 0.5
    assert res.action == GuardrailAction.PASS
    assert res.metadata["groundedness_unverified"] is True
    assert res.metadata["unverified_reason"] == "saturated"
    assert not called
    assert any("unverified" in r.message.lower() for r in caplog.records)


def test_timeout_clock_excludes_nothing_extra_and_result_returned(monkeypatch):
    monkeypatch.setattr(og, "faithfulness", lambda **kw: (time.sleep(0.3), 1.0)[1])
    g = GroundednessGuardrail(generator=object(), timeout_seconds=2.0)
    res = g.check("a", context=_ctx())
    assert res.action == GuardrailAction.PASS
    assert res.score == 1.0


def test_slot_released_after_completion(monkeypatch):
    sem = threading.BoundedSemaphore(1)
    monkeypatch.setattr(og, "_GROUNDEDNESS_SLOTS", sem)
    monkeypatch.setattr(og, "faithfulness", lambda **kw: 1.0)
    g = GroundednessGuardrail(generator=object())
    for _ in range(3):
        assert g.check("a", context=_ctx()).score == 1.0


def test_slot_held_until_timed_out_worker_finishes(monkeypatch):
    sem = threading.BoundedSemaphore(1)
    monkeypatch.setattr(og, "_GROUNDEDNESS_SLOTS", sem)
    monkeypatch.setattr(og, "faithfulness", lambda **kw: (time.sleep(0.6), 1.0)[1])
    g = GroundednessGuardrail(generator=object(), timeout_seconds=0.1)
    res = g.check("a", context=_ctx())
    assert res.metadata["unverified_reason"] == "timeout"
    # The abandoned thread still occupies its slot -> next call is shed.
    res2 = g.check("a", context=_ctx())
    assert res2.metadata["unverified_reason"] == "saturated"
    time.sleep(0.8)
    assert sem.acquire(blocking=False)  # released once the worker finished
