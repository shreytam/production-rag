from tests._fakes import FakeEmbedder
from eval.generation_metrics import answer_relevancy

class TrackingFakeEmbedder(FakeEmbedder):
    def __init__(self):
        self.query_calls = []
        self.doc_calls = []
    def embed_query(self, text):
        self.query_calls.append(text)
        return super().embed_query(text)
    def embed_documents(self, texts):
        self.doc_calls.append(texts)
        return super().embed_documents(texts)

def test_relevancy_uses_only_query_embeddings():
    class LocalFakeGenerator:
        def complete(self, *args, **keys):
            return type("Resp", (), {"parsed": {"questions": ["Q1", "Q2"]}})()

    embedder = TrackingFakeEmbedder()
    # Run relevancy metric
    answer_relevancy("Question", "Answer", LocalFakeGenerator(), embedder)
    # Check that generated questions were embedded with embed_query, not embed_documents
    assert len(embedder.query_calls) >= 3  # 1 (original) + 2 (generated questions)
    assert len(embedder.doc_calls) == 0
