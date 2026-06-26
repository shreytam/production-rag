"""FinanceBench corpus adapter.

Loads PatronusAI/financebench from HuggingFace. Each row's evidence text
becomes one Document; the provided question and answer become a GoldenItem.

Network calls are confined to load() and build_golden().
"""

from __future__ import annotations

from ingest.base import DatasetAdapter, GoldenItem, tenant_split_keeping_gold
from core.types import Document


class FinanceBenchAdapter(DatasetAdapter):
    """Adapter for the FinanceBench financial QA benchmark."""

    name = "financebench"

    def __init__(self) -> None:
        self._docs: list[Document] | None = None
        self._raw_rows: list[dict] | None = None

    def load(self, limit: int | None = None) -> list[Document]:
        """Load FinanceBench evidence passages as Documents."""
        from datasets import load_dataset  # noqa: PLC0415

        ds = load_dataset("PatronusAI/financebench", split="train")
        rows = list(ds) if limit is None else list(ds.select(range(min(limit, len(ds)))))
        self._raw_rows = rows

        docs: list[Document] = []
        seen: set[str] = set()
        for i, row in enumerate(rows):
            # Evidence may be a list of strings or a single string
            evidence = row.get("evidence", row.get("context", ""))
            if isinstance(evidence, list):
                evidence_text = "\n\n".join(str(e) for e in evidence)
            else:
                evidence_text = str(evidence)

            company = row.get("company", f"doc_{i}")
            doc_id = f"financebench::{company}::{i}"
            if doc_id in seen:
                continue
            seen.add(doc_id)

            docs.append(
                Document(
                    doc_id=doc_id,
                    text=evidence_text,
                    tenant_id="public",
                    title=company,
                    source="financebench:PatronusAI/financebench:train",
                    metadata={
                        "company": company,
                        "doc_type": row.get("doc_type", ""),
                        "fiscal_year": row.get("fiscal_year_end", ""),
                    },
                )
            )

        # Apply tenant split keeping gold docs in public
        golden = self._build_golden_items(rows, docs)
        doc_ids = [d.doc_id for d in docs]
        tenant_map = tenant_split_keeping_gold(doc_ids, golden, seed=42)
        self._docs = [
            d.model_copy(update={"tenant_id": tenant_map[d.doc_id][0],
                                  "acl_tags": tenant_map[d.doc_id][1]})
            for d in docs
        ]
        return self._docs

    def _build_golden_items(
        self, rows: list[dict], docs: list[Document]
    ) -> list[GoldenItem]:
        items = []
        for doc, row in zip(docs, rows):
            question = row.get("question", "")
            answer = row.get("answer", "")
            if question and answer:
                items.append(
                    GoldenItem(
                        question=question,
                        answer=answer,
                        relevant_doc_ids=[doc.doc_id],
                        tenant_id="public",
                    )
                )
        return items

    def build_golden(self, limit: int | None = None) -> list[GoldenItem]:
        """Return GoldenItems from FinanceBench Q/A pairs."""
        if self._raw_rows is None:
            self.load(limit=limit)
        rows = self._raw_rows
        docs = self._docs
        if limit is not None:
            rows = rows[:limit]
            docs = docs[:limit]
        return self._build_golden_items(rows, docs)
