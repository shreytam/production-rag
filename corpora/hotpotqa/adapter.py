"""HotpotQA corpus adapter.

Loads the HotpotQA "distractor" validation split from HuggingFace
(`hotpotqa/hotpot_qa`, parquet — the legacy bare `hotpot_qa` script id no longer
resolves under datasets>=3). The adapter is **question-driven**: `limit` bounds
the number of QUESTIONS, and every context paragraph for those questions is
ingested as a Document. This guarantees each question's gold supporting
paragraphs are present in the corpus — so Recall@k is never capped by a
doc/question count mismatch. Gold paragraph titles are forced to the `public`
tenant so the ACL split never orphans a gold doc from its question.

Network calls are confined to load() and build_golden() so importing this
module is safe in offline environments.
"""

from __future__ import annotations

from core.types import Document
from ingest.base import DatasetAdapter, GoldenItem, tenant_split_keeping_gold

_REPO = "hotpotqa/hotpot_qa"
_CONFIG = "distractor"
_SPLIT = "validation"


def _supporting_titles(row: dict) -> list[str]:
    """Gold paragraph titles for a row (deduped).

    `row["supporting_facts"]["title"]` is already a list of title strings; the
    set() dedups the (title, sent_id) duplicates that share a title.
    """
    return list(dict.fromkeys(row["supporting_facts"]["title"]))


class HotpotQAAdapter(DatasetAdapter):
    """Adapter for HotpotQA (distractor setting, validation split)."""

    name = "hotpotqa"

    def __init__(self) -> None:
        self._docs: list[Document] | None = None
        self._tenant_map: dict[str, tuple[str, tuple[str, ...]]] | None = None

    def _rows(self, limit: int | None):
        """Load the first `limit` validation rows (questions)."""
        from datasets import load_dataset  # noqa: PLC0415

        split = _SPLIT if limit is None else f"{_SPLIT}[:{limit}]"
        return load_dataset(_REPO, _CONFIG, split=split)

    def load(self, limit: int | None = None) -> list[Document]:
        """Materialise one Document per unique context paragraph across the first
        `limit` questions, then apply the tenant split (gold titles -> public)."""
        rows = self._rows(limit)

        seen_titles: set[str] = set()
        docs: list[Document] = []
        gold_titles: set[str] = set()

        for row in rows:
            gold_titles.update(_supporting_titles(row))
            for title, sentences in zip(
                row["context"]["title"], row["context"]["sentences"]
            ):
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                docs.append(
                    Document(
                        doc_id=title,
                        text=" ".join(sentences),
                        tenant_id="public",  # overridden by the split below
                        title=title,
                        source="hotpotqa:distractor:validation",
                    )
                )

        return self._apply_tenant_split(docs, gold_titles)

    def _apply_tenant_split(
        self, docs: list[Document], gold_titles: set[str]
    ) -> list[Document]:
        """Seeded 90/5/5 tenant split, forcing every gold paragraph to public."""
        doc_ids = [d.doc_id for d in docs]
        # A single synthetic public golden item carrying every gold title is all
        # tenant_split_keeping_gold needs to pin those docs to the public tenant.
        gold_guard = [
            GoldenItem(
                question="",
                answer="",
                relevant_doc_ids=list(gold_titles),
                tenant_id="public",
            )
        ]
        tenant_map = tenant_split_keeping_gold(doc_ids, gold_guard, seed=42)
        self._tenant_map = tenant_map

        result = []
        for doc in docs:
            tid, tags = tenant_map.get(doc.doc_id, ("public", ()))
            result.append(doc.model_copy(update={"tenant_id": tid, "acl_tags": tags}))
        self._docs = result
        return result

    def build_golden(self, limit: int | None = None) -> list[GoldenItem]:
        """One GoldenItem per question over the same first `limit` rows.

        relevant_doc_ids = the supporting-fact titles (gold paragraphs), which
        load() guarantees are present and pinned to the public tenant.
        """
        items: list[GoldenItem] = []
        for row in self._rows(limit):
            items.append(
                GoldenItem(
                    question=row["question"],
                    answer=row["answer"],
                    relevant_doc_ids=_supporting_titles(row),
                    tenant_id="public",
                )
            )
        return items
