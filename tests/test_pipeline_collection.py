"""RAGPipeline.run/.answer thread collection_id into the built Query.

Offline construction note: `build()` only supports version in {"baseline", "full"}
and would need real backends/network for "full" (embedder/vector-store/generator
construction). Instead we construct `RAGPipeline` directly from minimal fakes,
mirroring the pattern in tests/test_pipeline_integration.py::_build, and swap in a
recording retriever that captures the `Query` it receives.
"""

from __future__ import annotations

from core.config import get_settings
from core.pipeline import RAGPipeline
from core.types import ACLContext, Answer, Query, Usage
from generation.grounded_generator import GroundedGenerator


class _RecordingRetriever:
    def __init__(self):
        self.q: Query | None = None

    def retrieve(self, query: Query):
        self.q = query
        return []


class _FakeGenerator:
    """Minimal generate() that GroundedGenerator can call with zero real chunks."""

    model = "fake"

    def complete(self, messages, *, response_model=None, **_):
        from core.types import LLMResponse

        return LLMResponse(
            text="no context available.",
            parsed=None,
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            model="fake",
        )


def _build_pipeline() -> tuple[RAGPipeline, _RecordingRetriever]:
    settings = get_settings()
    grounded = GroundedGenerator(_FakeGenerator(), token_budget=settings.context_token_budget)
    rec = _RecordingRetriever()
    pipeline = RAGPipeline(rec, grounded, settings, guardrails=None)
    return pipeline, rec


def test_answer_passes_collection_id_into_query():
    pipeline, rec = _build_pipeline()
    pipeline.answer("hello", ACLContext(tenant_id="t"), collection_id="A")
    assert rec.q is not None
    assert rec.q.collection_id == "A"


def test_run_passes_collection_id_into_query():
    pipeline, rec = _build_pipeline()
    pipeline.run("hello", ACLContext(tenant_id="t"), collection_id="A")
    assert rec.q is not None
    assert rec.q.collection_id == "A"


def test_collection_id_defaults_to_none():
    pipeline, rec = _build_pipeline()
    pipeline.run("hello", ACLContext(tenant_id="t"))
    assert rec.q is not None
    assert rec.q.collection_id is None
