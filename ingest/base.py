"""Base classes and shared utilities for ingest pipeline dataset adapters.

Tenant/Gold Guarantee
---------------------
The main quality eval runs in ONE tenant scope (`public`). This means every
GoldenItem has tenant_id="public" and all its relevant_doc_ids must be in the
`public` tenant so that Recall@k is never artificially capped by ACL mismatches.

`assign_tenants` does a seeded 90/5/5 split across all doc_ids. However,
`tenant_split_keeping_gold` overrides: gold doc_ids (those referenced by any
GoldenItem) are always forced back to tenant "public", regardless of the
random draw. Distractor docs may land in tenant_a or tenant_b to exercise
multi-tenant filtering without breaking eval.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from core.types import Document, Chunk


class GoldenItem(BaseModel):
    """A single Q&A pair with pointers to the supporting corpus items."""

    question: str
    answer: str
    relevant_doc_ids: list[str]
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    tenant_id: str = "public"


def assign_tenants(
    doc_ids: list[str],
    seed: int = 42,
) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Return a mapping doc_id -> (tenant_id, acl_tags).

    Distribution: ~90% public, ~5% tenant_a, ~5% tenant_b.
    acl_tags are empty (chunk is visible to any caller in the tenant).
    """
    rng = random.Random(seed)
    result: dict[str, tuple[str, tuple[str, ...]]] = {}
    for doc_id in doc_ids:
        r = rng.random()
        if r < 0.90:
            tenant = "public"
        elif r < 0.95:
            tenant = "tenant_a"
        else:
            tenant = "tenant_b"
        result[doc_id] = (tenant, ())
    return result


def tenant_split_keeping_gold(
    doc_ids: list[str],
    golden_items: list[GoldenItem],
    seed: int = 42,
) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Assign tenants while guaranteeing every gold doc stays in its question's tenant.

    Algorithm:
    1. Run assign_tenants to get the base seeded draw.
    2. Collect the set of gold doc_ids referenced by questions whose tenant_id
       is "public" (which is all of them in our setup).
    3. Force every gold doc_id back to "public" so Recall@k is never capped
       by ACL filters during the main eval.

    The guarantee: for every GoldenItem g and every doc_id in g.relevant_doc_ids,
    the assigned tenant must equal g.tenant_id.
    """
    assignment = assign_tenants(doc_ids, seed=seed)

    # Collect gold doc_ids per question tenant
    for item in golden_items:
        for doc_id in item.relevant_doc_ids:
            if doc_id in assignment:
                assignment[doc_id] = (item.tenant_id, assignment[doc_id][1])

    return assignment


class DatasetAdapter(ABC):
    """Abstract base for corpus adapters.

    Subclasses implement load() and build_golden(). All network / HuggingFace
    calls MUST be confined to these methods so the adapter can be imported
    safely in test environments without triggering downloads.
    """

    name: str  # unique corpus name, e.g. "hotpotqa"

    @abstractmethod
    def load(self, limit: int | None = None) -> list[Document]:
        """Materialise the corpus as Documents.

        Parameters
        ----------
        limit:
            If set, return at most this many documents. Used during development
            and testing to keep runs cheap.
        """

    @abstractmethod
    def build_golden(self, limit: int | None = None) -> list[GoldenItem]:
        """Return golden Q/A pairs with pointers into the loaded corpus.

        Must be called after load() so doc_ids are consistent.
        The returned items MUST satisfy the tenant/gold guarantee (all
        relevant_doc_ids share item.tenant_id).
        """
