"""ArXiv corpus adapter.

Tries to load arXiv abstracts from HuggingFace (``ccdv/arxiv-summarization``).
If that dataset is unavailable (network error or missing), falls back to the
bundled ``sample.jsonl`` file (~10 abstracts) so the adapter works offline.

For golden Q/A pairs, synthesizing questions from abstracts requires an LLM.
This is GUARDED: build_golden() requires an injected generator and is not
called by the offline test suite.

~20 human-verified label pairs should be added to sample.jsonl as a separate
``golden.jsonl`` file (manually annotated) for production eval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ingest.base import DatasetAdapter, GoldenItem
from core.types import Document

if TYPE_CHECKING:
    from core.interfaces import Generator

_SAMPLE_JSONL = Path(__file__).parent / "sample.jsonl"


class ArxivAdapter(DatasetAdapter):
    """Adapter for arXiv abstracts.

    Parameters
    ----------
    generator:
        Optional Generator for synthesising Q/A pairs in build_golden().
        Pass None (default) if golden items are not needed.
    """

    name = "arxiv"

    def __init__(self, generator: "Generator | None" = None) -> None:
        self._generator = generator
        self._docs: list[Document] | None = None

    def load(self, limit: int | None = None) -> list[Document]:
        """Load arXiv abstracts.

        Tries HuggingFace first; falls back to sample.jsonl.
        """
        try:
            docs = self._load_from_hf(limit)
        except Exception:
            docs = self._load_from_sample(limit)

        self._docs = docs
        return docs

    def _load_from_hf(self, limit: int | None) -> list[Document]:
        from datasets import load_dataset  # noqa: PLC0415

        ds = load_dataset(
            "ccdv/arxiv-summarization",
            split="test",
            streaming=True,
            trust_remote_code=True,
        )
        docs: list[Document] = []
        for i, row in enumerate(ds):
            if limit is not None and i >= limit:
                break
            doc_id = f"arxiv::{i:06d}"
            docs.append(
                Document(
                    doc_id=doc_id,
                    text=row.get("article", row.get("abstract", "")),
                    tenant_id="public",
                    title=row.get("title", doc_id),
                    source="arxiv:ccdv/arxiv-summarization:test",
                )
            )
        return docs

    def _load_from_sample(self, limit: int | None) -> list[Document]:
        docs: list[Document] = []
        with _SAMPLE_JSONL.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                if limit is not None and i >= limit:
                    break
                row = json.loads(line)
                doc_id = row.get("id", f"arxiv::{i:06d}")
                text = row.get("abstract", row.get("text", ""))
                docs.append(
                    Document(
                        doc_id=doc_id,
                        text=text,
                        tenant_id="public",
                        title=row.get("title"),
                        source="arxiv:sample.jsonl",
                    )
                )
        return docs

    def build_golden(self, limit: int | None = None) -> list[GoldenItem]:
        """Synthesise Q/A pairs from abstracts using an injected LLM.

        REQUIRES: self._generator is set (not None).
        This method is NOT offline-safe and is NOT called by the test suite.
        Produced items should be human-verified before use in production eval
        (~20 verified pairs are sufficient for a meaningful benchmark).
        """
        if self._generator is None:
            raise RuntimeError(
                "ArxivAdapter.build_golden() requires an injected Generator. "
                "Pass generator=<Generator> to the constructor, or use the "
                "pre-annotated golden.jsonl if available."
            )
        docs = self._docs if self._docs is not None else self.load(limit=limit)
        if limit is not None:
            docs = docs[:limit]

        from core.types import ChatMessage  # noqa: PLC0415

        items: list[GoldenItem] = []
        for doc in docs:
            messages = [
                ChatMessage(
                    role="user",
                    content=(
                        "Given this arXiv abstract, generate ONE factual question "
                        "and a short answer (1-2 sentences) grounded in the text.\n\n"
                        f"Abstract:\n{doc.text}\n\n"
                        "Respond as JSON: {\"question\": \"...\", \"answer\": \"...\"}"
                    ),
                )
            ]
            resp = self._generator.complete(messages, max_tokens=256, temperature=0.0)
            try:
                parsed = json.loads(resp.text)
                question = parsed["question"]
                answer = parsed["answer"]
            except (json.JSONDecodeError, KeyError):
                continue

            items.append(
                GoldenItem(
                    question=question,
                    answer=answer,
                    relevant_doc_ids=[doc.doc_id],
                    tenant_id="public",
                )
            )
        return items
