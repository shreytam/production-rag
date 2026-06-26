"""Throwaway NIM smoke test: validates key + embed compat + dim + structured output.

Run: uv run python scripts/smoke_nim.py
Makes ~3 real API calls. Prints PASS/FAIL per check; never prints the key.
"""

from __future__ import annotations

import sys
import time

from core.config import get_settings
from core.registry import build_embedder, build_generator
from core.types import ChatMessage
from generation.grounded_generator import GeneratedAnswer


def main() -> int:
    s = get_settings()
    ok = True

    # 1) Embeddings: query + passage, check dimension matches config.
    t0 = time.time()
    emb = build_embedder(s)
    qv = emb.embed_query("What is the capital of France?")
    dv = emb.embed_documents(["Paris is the capital of France."])[0]
    dt = time.time() - t0
    dim_ok = len(qv) == s.embed_dimension == len(dv)
    print(f"[{'PASS' if dim_ok else 'FAIL'}] embeddings: model={s.embed_model} "
          f"dim={len(qv)} (expected {s.embed_dimension}) in {dt:.1f}s")
    ok &= dim_ok

    # 2) Structured output: does the gen model honor json_schema (or fall back)?
    t0 = time.time()
    gen = build_generator("gen", s)
    resp = gen.complete(
        [
            ChatMessage(role="system", content="Answer from the context. Cite passage [1]."),
            ChatMessage(role="user", content="Context:\n[1] The 2023 revenue was $10M.\n\nQuestion: What was the 2023 revenue?"),
        ],
        response_model=GeneratedAnswer,
    )
    dt = time.time() - t0
    parsed_ok = resp.parsed is not None and "answer" in (resp.parsed or {})
    print(f"[{'PASS' if parsed_ok else 'FAIL'}] structured gen: model={s.gen_model} "
          f"parsed={resp.parsed} tokens={resp.usage.total_tokens} in {dt:.1f}s")
    if not parsed_ok:
        print(f"       raw text: {resp.text[:200]!r}")
    ok &= parsed_ok

    print("\n=== SMOKE", "PASS ===" if ok else "FAIL ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
