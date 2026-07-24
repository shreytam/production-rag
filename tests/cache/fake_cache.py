"""In-memory SemanticCache for the offline suite: brute-force cosine, TAG-style
tenant/collection scoping, targeted eviction, injectable-clock TTL. No Redis."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from cache.semantic_cache import norm_collection


@dataclass
class _Entry:
    embedding: list[float]
    payload: dict
    doc_ids: set[str]
    expires_at: float


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class FakeSemanticCache:
    threshold: float = 0.9
    ttl_seconds: int = 3600
    time_fn: Callable[[], float] = time.time
    _store: dict[tuple[str, str], list[_Entry]] = field(default_factory=dict)

    def _key(self, tenant_id: str, collection_id: str | None) -> tuple[str, str]:
        return (tenant_id, norm_collection(collection_id))

    def _live(self, bucket: list[_Entry]) -> list[_Entry]:
        now = self.time_fn()
        return [e for e in bucket if e.expires_at > now]

    def lookup(self, *, tenant_id, collection_id, embedding):
        bucket = self._store.get(self._key(tenant_id, collection_id), [])
        best, best_sim = None, -1.0
        for e in self._live(bucket):
            sim = _cosine(embedding, e.embedding)
            if sim > best_sim:
                best, best_sim = e, sim
        if best is not None and best_sim >= self.threshold:
            return best.payload
        return None

    def store(self, *, tenant_id, collection_id, embedding, payload, doc_ids):
        bucket = self._store.setdefault(self._key(tenant_id, collection_id), [])
        bucket.append(_Entry(
            embedding=list(embedding), payload=payload, doc_ids=set(doc_ids),
            expires_at=self.time_fn() + self.ttl_seconds,
        ))

    def invalidate_document(self, *, tenant_id, collection_id, doc_id) -> int:
        key = self._key(tenant_id, collection_id)
        bucket = self._store.get(key, [])
        keep = [e for e in bucket if doc_id not in e.doc_ids]
        removed = len(bucket) - len(keep)
        self._store[key] = keep
        return removed
