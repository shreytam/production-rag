"""Token-budgeted context assembly.

Turns ranked chunks into a numbered, citation-anchored context block that fits a
token budget. Markers ([1], [2], ...) let the generator cite, and we keep the
marker->chunk mapping so citations resolve back to real chunk/doc ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from core.types import ScoredChunk


@lru_cache(maxsize=4)
def _encoder(encoding_name: str = "cl100k_base"):
    import tiktoken

    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    return len(_encoder(encoding_name).encode(text))


def resolve_encoding(context_tokenizer: str, gen_model: str) -> str:
    if context_tokenizer != "auto":
        return context_tokenizer
    m = gen_model.lower()
    if "gpt-4o" in m or "o200k" in m:
        return "o200k_base"
    if "gpt-4" in m or "gpt-3.5" in m or "cl100k" in m:
        return "cl100k_base"
    return "cl100k_base"


@dataclass
class AssembledContext:
    text: str
    items: list[tuple[int, ScoredChunk]] = field(default_factory=list)

    @property
    def marker_to_chunk(self) -> dict[int, ScoredChunk]:
        return {n: sc for n, sc in self.items}


def _render(marker: int, sc: ScoredChunk) -> str:
    c = sc.chunk
    head = f"[{marker}]"
    if c.title:
        head += f" {c.title}"
    if c.source:
        head += f" ({c.source})"
    return f"{head}\n{c.embed_text}"


def assemble_context(
    chunks: list[ScoredChunk], token_budget: int, encoding_name: str = "cl100k_base"
) -> AssembledContext:
    """Greedily pack ranked chunks (dedup by chunk_id) until the budget is hit.

    The first chunk is always included even if it alone exceeds the budget, so a
    single long chunk still yields a usable context.
    """
    seen: set[str] = set()
    used: list[tuple[int, ScoredChunk]] = []
    blocks: list[str] = []
    total = 0
    marker = 0

    for sc in chunks:
        if sc.chunk_id in seen:
            continue
        seen.add(sc.chunk_id)
        marker += 1
        block = _render(marker, sc)
        cost = count_tokens(block, encoding_name)
        if used and total + cost > token_budget:
            marker -= 1
            break
        used.append((marker, sc))
        blocks.append(block)
        total += cost

    return AssembledContext(text="\n\n".join(blocks), items=used)
