import pytest
from core.types import Chunk, ScoredChunk, RetrievalSource
from providers.rerankers._common import normalize_candidates

def test_normalize_candidates_drops_bounds_failures():
    c1 = Chunk(chunk_id="c_1", doc_id="d1", text="a", tenant_id="t")
    sc1 = ScoredChunk(chunk=c1, score=0.5)

    # Index 2 is out of range (only len=1)
    scored = [(2, 0.9), (0, 0.8)]
    clean = normalize_candidates([sc1], scored, top_n=2)

    assert len(clean) == 1
    assert clean[0].chunk_id == "c_1"
    assert clean[0].score == 0.8
    assert clean[0].source == RetrievalSource.RERANK
