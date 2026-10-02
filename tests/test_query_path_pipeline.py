from core.config import Settings
from core.pipeline import RAGPipeline
from core.types import ACLContext, Answer, Chunk, ScoredChunk, Usage


class _Ret:
    def __init__(self, scored):
        self._scored = scored

    def retrieve(self, q):
        return list(self._scored)


class _Gen:
    def __init__(self):
        self.calls = 0

    def generate(self, question, scored):
        self.calls += 1
        return Answer(text="ok", contexts=list(scored), usage=Usage())


class _SpyGuardrails:
    def __init__(self):
        self.context = None

    def check_input(self, q):
        return []

    def blocked(self, results):
        return False

    def apply_redactions(self, text, results):
        return text

    def check_output(self, ans, context=None):
        self.context = context
        return []


def test_empty_retrieval_refuses_without_calling_generator():
    gen = _Gen()
    p = RAGPipeline(_Ret([]), gen, Settings(), guardrails=None)
    ans = p.answer("anything", ACLContext(tenant_id="t"))
    assert gen.calls == 0
    assert ans.refused is True
    assert "no relevant documents" in ans.text.lower()
    assert ans.citations == []
    assert ans.metadata["retrieved_doc_ids"] == []


def test_judge_contexts_include_contextual_prefix_like_generator():
    chunk = Chunk(chunk_id="c", doc_id="d", text="body", tenant_id="t",
                  contextual_prefix="PREFIX")
    gen, guards = _Gen(), _SpyGuardrails()
    p = RAGPipeline(_Ret([ScoredChunk(chunk=chunk, score=1.0)]), gen, Settings(),
                    guardrails=guards)
    p.answer("q", ACLContext(tenant_id="t"))
    assert guards.context["contexts"] == [chunk.embed_text]
    assert "PREFIX" in guards.context["contexts"][0]
