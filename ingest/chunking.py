"""Structure-aware, token-budget chunking with deterministic IDs.

Uses tiktoken cl100k_base for token counting. Splits on blank-line paragraph
boundaries first, then packs paragraphs into chunks that respect max_tokens.
Paragraphs larger than the budget are split at the word level. Overlap is
achieved by prepending the last `overlap` tokens of the previous chunk.

Chunk IDs are deterministic: f"{doc_id}::{ordinal}" (zero-padded to 6 digits).
"""

from __future__ import annotations

import re
from typing import Sequence

import tiktoken

from core.types import Chunk, Document

_ENCODER: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def _tokenize(text: str) -> list[int]:
    return _get_encoder().encode(text)


def _decode(tokens: list[int]) -> str:
    return _get_encoder().decode(tokens)


def _split_paragraphs(text: str) -> list[str]:
    """Split on one or more blank lines, preserving non-empty paragraphs."""
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _token_chunks_from_paragraph(
    para: str,
    max_tokens: int,
) -> list[list[int]]:
    """Break a single paragraph into token-budget-sized slices (no overlap here)."""
    tokens = _tokenize(para)
    if len(tokens) <= max_tokens:
        return [tokens]
    slices = []
    start = 0
    while start < len(tokens):
        slices.append(tokens[start : start + max_tokens])
        start += max_tokens
    return slices


def chunk_document(
    doc: Document,
    max_tokens: int = 256,
    overlap: int = 32,
) -> list[Chunk]:
    """Chunk *doc* into Chunk objects respecting the token budget.

    Strategy
    --------
    1. Split text on blank-line paragraph boundaries.
    2. Pack consecutive paragraphs into a window up to max_tokens.
    3. When a paragraph alone exceeds max_tokens, split it token-wise.
    4. Prepend the tail (overlap tokens) of the previous chunk to the next one.

    Chunk IDs are ``{doc_id}::{ordinal:06d}`` — fully deterministic.
    Tenant/ACL attributes are propagated verbatim from the source document.
    """
    encoder = _get_encoder()
    paragraphs = _split_paragraphs(doc.text)

    # Collect raw token lists for all paragraphs (splitting oversized ones)
    raw_slices: list[list[int]] = []
    for para in paragraphs:
        raw_slices.extend(_token_chunks_from_paragraph(para, max_tokens))

    if not raw_slices:
        return []

    chunks: list[Chunk] = []
    prev_tail: list[int] = []
    ordinal = 0

    for i, slice_tokens in enumerate(raw_slices):
        window: list[int] = prev_tail + slice_tokens
        # Pack additional slices if they fit
        j = i + 1
        while j < len(raw_slices):
            candidate = window + raw_slices[j]
            if len(candidate) > max_tokens:
                break
            window = candidate
            j += 1

        # Trim to max_tokens (in case overlap bloated it slightly)
        if len(window) > max_tokens:
            window = window[:max_tokens]

        text = encoder.decode(window)
        chunk_id = f"{doc.doc_id}::{ordinal:06d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                text=text,
                tenant_id=doc.tenant_id,
                acl_tags=doc.acl_tags,
                ordinal=ordinal,
                title=doc.title,
                source=doc.source,
            )
        )
        prev_tail = window[-overlap:] if overlap > 0 else []
        ordinal += 1

        # Skip the slices we packed into this chunk
        # But we can only do this safely for the packing loop above —
        # the outer for-loop still walks every i. Use a visited set to skip.
        # Simpler: rebuild using a while loop instead of for.
        # See below — we rebuild with explicit index control.

    # The packing logic above has a flaw: after packing i+1..j-1 into chunk[i],
    # the for-loop still visits i+1..j-1 separately. Rewrite with explicit index.
    # Discard the result above and redo properly.
    chunks = []
    prev_tail = []
    ordinal = 0
    i = 0
    while i < len(raw_slices):
        window = prev_tail + raw_slices[i]
        j = i + 1
        while j < len(raw_slices):
            candidate = window + raw_slices[j]
            if len(candidate) > max_tokens:
                break
            window = candidate
            j += 1

        if len(window) > max_tokens:
            window = window[:max_tokens]

        text = encoder.decode(window)
        chunk_id = f"{doc.doc_id}::{ordinal:06d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                text=text,
                tenant_id=doc.tenant_id,
                acl_tags=doc.acl_tags,
                ordinal=ordinal,
                title=doc.title,
                source=doc.source,
            )
        )
        prev_tail = window[-overlap:] if overlap > 0 else []
        ordinal += 1
        i = j  # skip packed slices

    return chunks
