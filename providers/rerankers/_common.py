from core.types import RetrievalSource, ScoredChunk


def normalize_candidates(
    candidates: list[ScoredChunk],
    scored: list[tuple[int, float]],
    top_n: int,
) -> list[ScoredChunk]:
    """Map (index, score) reranker output back onto `candidates`.

    Any index outside `candidates`' bounds is dropped rather than raising —
    reranker responses are external input and a malformed/truncated index
    must not corrupt the candidate-to-score mapping. Raw scores are kept
    as-is (no rescaling) so callers see the model's actual output.
    """
    valid = [(idx, score) for idx, score in scored if 0 <= idx < len(candidates)]
    valid.sort(key=lambda pair: pair[1], reverse=True)

    results: list[ScoredChunk] = []
    for rank, (idx, score) in enumerate(valid[:top_n], start=1):
        results.append(
            ScoredChunk(
                chunk=candidates[idx].chunk,
                score=float(score),
                source=RetrievalSource.RERANK,
                rank=rank,
            )
        )
    return results
