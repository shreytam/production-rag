"""ACL filter builders — pure functions, no DB connection.

Three flavours matching each store's filtering API:
- qdrant_filter  → qdrant_client.models.Filter  (pre-similarity payload filter)
- pg_where       → (sql_fragment, params)        (pre-similarity WHERE clause)
- acl_predicate  → Callable[[Chunk], bool]       (in-memory candidate filter)

All three faithfully model ACLContext.allows():
  tenant must match AND (chunk has no tags OR caller∩chunk-tags is non-empty)
"""

from __future__ import annotations

from typing import Callable

from qdrant_client import models as qm

from core.types import ACLContext, Chunk


def qdrant_filter(acl: ACLContext) -> qm.Filter:
    """Return a Qdrant Filter that enforces ACL before similarity scoring.

    Strategy:
      MUST  tenant_id == acl.tenant_id
      MUST  (acl_open == True  OR  acl_tags overlaps caller's tags)

    We model "chunk is open" via the `acl_open` boolean payload flag
    (set at upsert to `not bool(chunk.acl_tags)`).

    The tag-overlap branch is only included when the caller actually
    holds tags; a no-tag caller may only see open chunks, which the
    unconditional `acl_open == True` already covers.
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

    return qm.Filter(
        must=[
            qm.FieldCondition(
                key="tenant_id",
                match=qm.MatchValue(value=acl.tenant_id),
            ),
            qm.Filter(should=visibility_should),
        ]
    )


def pg_where(acl: ACLContext) -> tuple[str, list]:
    """Return (sql_fragment, params) for a PostgreSQL WHERE clause.

    Fragment:
        tenant_id = %s AND (cardinality(acl_tags)=0 OR acl_tags && %s)

    `&&` is the array-overlap operator; an empty caller tag array will
    never overlap, so no-tag callers see only cardinality-0 (open) chunks —
    matching allows() semantics exactly.
    """
    fragment = "tenant_id = %s AND (cardinality(acl_tags)=0 OR acl_tags && %s)"
    params: list = [acl.tenant_id, list(acl.acl_tags)]
    return fragment, params


def acl_predicate(acl: ACLContext) -> Callable[[Chunk], bool]:
    """Return a callable that tests whether `acl` may see a given chunk.

    Delegates entirely to ACLContext.allows() — single source of truth.
    """
    return lambda chunk: acl.allows(chunk.acl)
