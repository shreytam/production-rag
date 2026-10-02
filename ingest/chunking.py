"""Structure-aware, token-budget chunking with deterministic IDs.

Uses tiktoken cl100k_base for token counting. Splits on blank-line paragraph
boundaries first, then packs paragraphs into chunks that respect max_tokens.
Paragraphs larger than the budget are split at the word level. Overlap is
achieved by prepending the last `overlap` tokens of the previous chunk.

Chunk IDs are deterministic: f"{doc_id}::{ordinal}" (zero-padded to 6 digits).
"""

from __future__ import annotations

import re

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

    sep_tokens = encoder.encode("\n\n")

    def clean_start(tok: int) -> bool:
        # A token whose first byte is a UTF-8 continuation byte begins in the
        # middle of a multi-byte character; cutting there yields U+FFFD.
        return (encoder.decode_single_token_bytes(tok)[0] & 0xC0) != 0x80

    def aligned_tail(tokens: list[int]) -> list[int]:
        tail = tokens[-overlap:] if overlap > 0 else []
        while tail and not clean_start(tail[0]):
            tail = tail[1:]
        return tail

    chunks_tokens: list[list[int]] = []
    current_chunk: list[int] = []
    fresh = False  # current_chunk holds only an overlap tail (no new content)

    for p_idx, para in enumerate(paragraphs):
        para_tokens = encoder.encode(para)
        if not para_tokens:
            continue
        if p_idx > 0:
            # Without a separator, "end." + "Next" would fuse into "end.Next".
            para_tokens = sep_tokens + para_tokens

        start_idx = 0
        while start_idx < len(para_tokens):
            space_left = max_tokens - len(current_chunk)
            end = min(start_idx + space_left, len(para_tokens))
            if space_left > 0 and end < len(para_tokens):
                while end > start_idx and not clean_start(para_tokens[end]):
                    end -= 1
            if space_left <= 0 or end == start_idx:
                if fresh:
                    # Nothing fits next to the bare overlap tail: take the
                    # whole character even if it overshoots by a few tokens.
                    end = min(start_idx + max(space_left, 1), len(para_tokens))
                    while end < len(para_tokens) and not clean_start(para_tokens[end]):
                        end += 1
                else:
                    chunks_tokens.append(current_chunk)
                    current_chunk = aligned_tail(current_chunk)
                    fresh = True
                    continue

            chunk_slice = para_tokens[start_idx:end]
            current_chunk.extend(chunk_slice)
            start_idx = end
            fresh = False

    if current_chunk and not fresh:
        if not chunks_tokens or len(current_chunk) > overlap:
            chunks_tokens.append(current_chunk)

    chunks: list[Chunk] = []
    texts = [t for t in (encoder.decode(tk).strip() for tk in chunks_tokens) if t]
    for ordinal, text in enumerate(texts):
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
