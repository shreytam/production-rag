from core.types import Chunk, ScoredChunk, RetrievalSource
from core.rrf import reciprocal_rank_fusion

def test_rrf_stable_tie_breaks():
    c1 = Chunk(chunk_id="ch_a", doc_id="d1", text="text1", tenant_id="t1")
    c2 = Chunk(chunk_id="ch_b", doc_id="d1", text="text2", tenant_id="t1")

    sc1 = ScoredChunk(chunk=c1, score=1.0, source=RetrievalSource.DENSE)
    sc2 = ScoredChunk(chunk=c2, score=1.0, source=RetrievalSource.DENSE)

    # Shuffle rankings to simulate order-dependence hazards
    ranking1 = [sc1, sc2]
    ranking2 = [sc2, sc1]

    r1 = reciprocal_rank_fusion([ranking1], k=60)
    r2 = reciprocal_rank_fusion([ranking2], k=60)

    # Outputs must have exactly identical ordering despite different raw orders
    assert [c.chunk_id for c in r1] == [c.chunk_id for c in r2]
    assert r1[0].chunk_id == "ch_a"
    assert r1[1].chunk_id == "ch_b"
