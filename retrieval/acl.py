"""ACL filter builders — pure functions, no DB connection.

Three flavours matching each store's filtering API:
- qdrant_filter  → qdrant_client.models.Filter  (pre-similarity payload filter)
- acl_predicate  → Callable[[Chunk], bool]       (in-memory candidate filter)

All three faithfully model ACLContext.allows():
  tenant must match AND (chunk has no tags OR caller∩chunk-tags is non-empty)
"""

from __future__ import annotations

from typing import Callable

from qdrant_client import models as qm

from core.types import ACLContext, Chunk


def qdrant_filter(acl: ACLContext, *, collection_id: str | None = None) -> qm.Filter:
    """Return a Qdrant Filter that enforces ACL before similarity scoring.

    Strategy:
      MUST  tenant_id == acl.tenant_id
      MUST  (acl_open == True  OR  acl_tags overlaps caller's tags)
      MUST  collection_id == collection_id   (only when collection_id is not None)

    We model "chunk is open" via the `acl_open` boolean payload flag
    (set at upsert to `not bool(chunk.acl_tags)`).

    The tag-overlap branch is only included when the caller actually
    holds tags; a no-tag caller may only see open chunks, which the
    unconditional `acl_open == True` already covers.

    `collection_id=None` (the default) means "no collection filter" —
    behaviour is unchanged from before this parameter existed.
    """
    # Always: chunk must be open
    visibility_should: list[qm.Condition] = [
        qm.FieldCondition(key="acl_open", match=qm.MatchValue(value=True))
    ]

    # Additionally: caller's tags intersect chunk's tags
    if acl.acl_tags:
        visibility_should.append(
            qm.FieldCondition(
                key="acl_tags",
                match=qm.MatchAny(any=list(acl.acl_tags)),
            )
        )

    must: list[qm.Condition] = [
        qm.FieldCondition(
            key="tenant_id",
            match=qm.MatchValue(value=acl.tenant_id),
        ),
        qm.Filter(should=visibility_should),
    ]
    if collection_id is not None:
        must.append(
            qm.FieldCondition(
                key="collection_id",
                match=qm.MatchValue(value=collection_id),
            )
        )

    return qm.Filter(must=must)


def acl_predicate(acl: ACLContext, *, collection_id: str | None = None) -> Callable[[Chunk], bool]:
    """Return a callable that tests whether `acl` may see a given chunk.

    Delegates entirely to ACLContext.allows() — single source of truth.

    `collection_id=None` (the default) means "no collection filter" —
    behaviour is unchanged from before this parameter existed. When set,
    the chunk's `collection_id` must equal it as well.
    """
    if collection_id is None:
        return lambda chunk: acl.allows(chunk.acl)
    return lambda chunk: acl.allows(chunk.acl) and chunk.collection_id == collection_id
