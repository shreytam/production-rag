"""Reciprocal Rank Fusion (RRF).

Fuses an arbitrary number of ranked candidate lists into one ranking using
score(d) = sum_i 1 / (k + rank_i(d)), with rank 1-based. k defaults to 60, the
value from the original Cormack et al. RRF paper and our config default.
"""

from __future__ import annotations

from core.types import RetrievalSource, ScoredChunk


def reciprocal_rank_fusion(
    rankings: list[list[ScoredChunk]], k: int = 60
) -> list[ScoredChunk]:
    """Fuse ranked lists of ScoredChunk into a single RRF-ordered list.

    Chunks are identified by `chunk_id`. The first-seen Chunk object is preserved;
    per-source contributing scores are recorded in `component_scores`.
    """
    fused: dict[str, float] = {}
    keep: dict[str, ScoredChunk] = {}
    components: dict[str, dict[str, float]] = {}

    for ranking in rankings:
        # Sort each ranking to ensure stable rank assignment for tied scores
        sorted_ranking = sorted(ranking, key=lambda sc: (-sc.score, sc.chunk_id))
        for rank, sc in enumerate(sorted_ranking, start=1):
            cid = sc.chunk_id
            contribution = 1.0 / (k + rank)
            fused[cid] = fused.get(cid, 0.0) + contribution
            comp = components.setdefault(cid, {})
            comp[sc.source.value] = sc.score
            if cid not in keep:
                keep[cid] = sc

    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    out: list[ScoredChunk] = []
    for rank, (cid, score) in enumerate(ordered, start=1):
        base = keep[cid]
        out.append(
            ScoredChunk(
                chunk=base.chunk,
                score=score,
                source=RetrievalSource.FUSED,
                rank=rank,
                component_scores=components[cid],
            )
        )
    return out
