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
    overlap: int = 0,
) -> list[list[int]]:
    """Break a single paragraph into token-budget-sized slices with overlap."""
    tokens = _tokenize(para)
    if len(tokens) <= max_tokens:
        return [tokens]
    slices = []
    start = 0
    step = max_tokens - overlap
    if step <= 0:
        step = 1
    while start < len(tokens):
        slices.append(tokens[start : start + max_tokens])
        if start + max_tokens >= len(tokens):
            break
        start += step
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
    3. When a paragraph alone exceeds max_tokens, split it token-wise with overlap.
    4. Prepend the tail (overlap tokens) of the previous chunk to the next one.

    Chunk IDs are ``{doc_id}::{ordinal:06d}`` — fully deterministic.
    Tenant/ACL attributes are propagated verbatim from the source document.
    """
    encoder = _get_encoder()
    paragraphs = _split_paragraphs(doc.text)

    # Clean and bound overlap
    overlap = min(overlap, max_tokens - 1)
    if overlap < 0:
        overlap = 0

    chunks_tokens: list[list[int]] = []
    current_chunk: list[int] = []

    for para in paragraphs:
        para_tokens = encoder.encode(para)
        if not para_tokens:
            continue

        start_idx = 0
        while start_idx < len(para_tokens):
            space_left = max_tokens - len(current_chunk)
            if space_left <= 0:
                chunks_tokens.append(current_chunk)
                prev_tail = current_chunk[-overlap:] if overlap > 0 else []
                current_chunk = list(prev_tail)
                space_left = max_tokens - len(current_chunk)

            chunk_slice = para_tokens[start_idx : start_idx + space_left]
            current_chunk.extend(chunk_slice)
            start_idx += len(chunk_slice)

    if current_chunk:
        if not chunks_tokens or len(current_chunk) > overlap:
            chunks_tokens.append(current_chunk)

    chunks: list[Chunk] = []
    for ordinal, tokens in enumerate(chunks_tokens):
        text = encoder.decode(tokens)
        chunk_id = f"{doc.doc_id}::{ordinal:06d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                text=text,
                tenant_id=doc.tenant_id,
                acl_tags=doc.acl_tags,
                collection_id=doc.collection_id,
                ordinal=ordinal,
                title=doc.title,
                source=doc.source,
            )
        )
    return chunks
