import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from core.types import RetrievalSource, ScoredChunk

_MAX_ATTEMPTS = 3


def is_transient(exc: BaseException) -> bool:
    """Retryable: timeouts, network errors, HTTP 429 and 5xx. Other 4xx
    (auth, bad request) are permanent and must not be retried."""
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return False


retry_transient = retry(
    retry=retry_if_exception(is_transient),
    stop=stop_after_attempt(_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    reraise=True,
)


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
