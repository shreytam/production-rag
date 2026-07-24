from cache.semantic_cache import (
    COLLECTION_NONE, answer_from_payload, answer_to_payload, doc_ids_of,
    norm_collection, scored_from_payload, scored_to_payload,
)
from core.types import Answer, Chunk, Citation, ScoredChunk


def _scored():
    return [
        ScoredChunk(chunk=Chunk(chunk_id="c1", doc_id="d1", text="t1", tenant_id="acme"), score=0.9),
        ScoredChunk(chunk=Chunk(chunk_id="c2", doc_id="d1", text="t2", tenant_id="acme"), score=0.8),
        ScoredChunk(chunk=Chunk(chunk_id="c3", doc_id="d2", text="t3", tenant_id="acme"), score=0.7),
    ]


def test_norm_collection():
    assert norm_collection(None) == COLLECTION_NONE
    assert norm_collection("kb") == "kb"


def test_answer_round_trip():
    ans = Answer(text="hi", citations=[Citation(marker="1", chunk_id="c1")], refused=False)
    back = answer_from_payload(answer_to_payload(ans))
    assert back.text == "hi"
    assert back.refused is False
    assert back.citations[0].chunk_id == "c1"


def test_scored_round_trip_and_doc_ids():
    sc = _scored()
    back = scored_from_payload(scored_to_payload(sc))
    assert [s.chunk_id for s in back] == ["c1", "c2", "c3"]
    assert doc_ids_of(sc) == ["d1", "d2"]  # deduped, order-preserving
