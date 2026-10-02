"""Semantic cache seam: a tenant-scoped, embedding-keyed cache Protocol plus the
payload (de)serialization shared by the answer and retrieval tiers.

NO top-level redis-vl import lives here or anywhere else in this package except
_redisvl_backend.py (lazily). Importing this module needs neither Redis nor the
redis-vl package, so lint and the offline suite stay infra-free.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Protocol, Sequence

from core.types import Answer, ScoredChunk

if TYPE_CHECKING:
    from core.config import Settings

COLLECTION_NONE = "__none__"


def norm_collection(collection_id: str | None) -> str:
    """Map an absent collection to a stable sentinel so TAG scoping is total."""
    return collection_id if collection_id else COLLECTION_NONE


def acl_key(acl_tags: Sequence[str]) -> str:
    """Stable partition key for a caller's ACL tag SET (order/duplicates ignored).
    A cached answer/chunk list is only valid for callers whose tag set is identical
    to the one that produced it. The empty set hashes a distinct marker."""
    tags = sorted(set(acl_tags))
    canon = "tags:" + "\x1f".join(tags) if tags else "no-tags"
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


class SemanticCache(Protocol):
    def lookup(self, *, tenant_id: str, collection_id: str | None,
               acl_tags: Sequence[str], embedding: Sequence[float]) -> dict | None: ...

    def store(self, *, tenant_id: str, collection_id: str | None,
              acl_tags: Sequence[str], embedding: Sequence[float], payload: dict,
              doc_ids: Sequence[str]) -> None: ...

    def invalidate_document(self, *, tenant_id: str, collection_id: str | None,
                            doc_id: str) -> int: ...


def answer_to_payload(ans: Answer) -> dict:
    return {"kind": "answer", "answer": ans.model_dump(mode="json")}


def answer_from_payload(payload: dict) -> Answer:
    return Answer.model_validate(payload["answer"])


def scored_to_payload(scored: list[ScoredChunk]) -> dict:
    return {"kind": "scored", "scored": [s.model_dump(mode="json") for s in scored]}


def scored_from_payload(payload: dict) -> list[ScoredChunk]:
    return [ScoredChunk.model_validate(s) for s in payload["scored"]]


def doc_ids_of(scored: list[ScoredChunk]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for s in scored:
        d = s.chunk.doc_id
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def build_cache(settings: "Settings") -> tuple[SemanticCache, SemanticCache]:
    """Construct the (answer, retrieval) tier pair. Pure constructor — does NOT
    consult cache_enabled (the caller decides whether to build) and does NOT
    connect to Redis (the backend connects lazily on first use)."""
    from cache._redisvl_backend import RedisVLSemanticCache

    answer = RedisVLSemanticCache(index_name="rag_cache_answer_v2", settings=settings)
    retrieval = RedisVLSemanticCache(index_name="rag_cache_retrieval_v2", settings=settings)
    return answer, retrieval
