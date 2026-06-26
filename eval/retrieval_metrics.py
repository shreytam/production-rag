"""Pure retrieval evaluation metrics.

All functions are side-effect-free and depend only on the standard library and
numpy. They operate over opaque string ids — no knowledge of document internals.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-k retrieved ids that are relevant.

    Returns 0.0 when k == 0.
    """
    if k <= 0:
        return 0.0
    top_k = list(retrieved_ids)[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant ids found in the top-k retrieved results.

    Returns 1.0 when ``relevant_ids`` is empty (vacuously true) and 0.0 when
    k == 0 (nothing retrieved).
    """
    if not relevant_ids:
        return 1.0
    if k <= 0:
        return 0.0
    top_k = set(list(retrieved_ids)[:k])
    hits = len(top_k & relevant_ids)
    return hits / len(relevant_ids)


def mrr(retrieved_ids: Sequence[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank of the first relevant document.

    Returns 0.0 if no relevant document appears in the ranked list.
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at rank k (binary gains).

    DCG  = Σ_{i=1}^{k}  rel_i / log2(i+1)
    IDCG = DCG of the ideal ranking (all relevant docs ranked first).

    Returns 0.0 when k == 0 or relevant_ids is empty.
    """
    if k <= 0 or not relevant_ids:
        return 0.0

    top_k = list(retrieved_ids)[:k]

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(top_k, start=1)
        if doc_id in relevant_ids
    )

    # Ideal: place all relevant docs at the top (up to k)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg
