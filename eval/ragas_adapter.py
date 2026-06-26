"""Optional RAGAS-library cross-check for our native generation metrics.

The native metrics in ``eval/generation_metrics.py`` are the spine (they back the
CI gate, run on our own interfaces, and carry no framework lock-in). This adapter
is the *optional* counterpart: it computes the same four metrics with the real
`ragas` library, pointed at the same NIM models, so we can validate that our
native numbers agree with the reference implementation.

It is intentionally NOT imported by the core eval path. Enable it with::

    uv sync --extra ragas        # or: uv run --with ragas ...
    uv run python -m eval.ragas_adapter --dataset hotpotqa --limit 5

Implementation note — the VertexAI stub
----------------------------------------
`ragas` 0.4.x hard-imports ``langchain_community.chat_models.vertexai.ChatVertexAI``
at package import time, but that submodule was removed from langchain-community
>= 0.3 (ChatVertexAI moved to ``langchain-google-vertexai``). We never use Vertex,
so we register a stub module *before* importing ragas. This is the documented
workaround for the known upstream import bug, scoped to this optional module.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import types
from pathlib import Path

from core.config import Settings, get_settings


def _install_vertexai_stub() -> None:
    name = "langchain_community.chat_models.vertexai"
    if name not in sys.modules:
        stub = types.ModuleType(name)
        stub.ChatVertexAI = object  # never used; only satisfies the import
        sys.modules[name] = stub


class RagasScorer:
    """Compute faithfulness / answer_relevancy / context_precision / context_recall
    via the real RAGAS library, backed by NIM (OpenAI-compatible) models."""

    def __init__(self, settings: Settings | None = None) -> None:
        _install_vertexai_stub()
        # The collections metrics drive async (.ascore -> agenerate), which requires
        # an AsyncOpenAI client — a sync client raises "Cannot use agenerate()".
        from openai import AsyncOpenAI
        from ragas.embeddings import embedding_factory
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        s = settings or get_settings()

        # Judge-role LLM for the metric reasoning; embeddings for answer_relevancy.
        judge_client = AsyncOpenAI(
            base_url=s.judge_base_url,
            api_key=s.judge_api_key,
            timeout=s.request_timeout_seconds,
            max_retries=s.max_retries,
        )
        llm = llm_factory(s.judge_model, client=judge_client)

        embed_client = AsyncOpenAI(
            base_url=s.embed_base_url,
            api_key=s.embed_api_key,
            timeout=s.request_timeout_seconds,
            max_retries=s.max_retries,
        )
        # provider="openai" (NIM is OpenAI-compatible); bge-m3 works without NIM's
        # input_type, so a plain OpenAI embeddings call is valid here (unlike the
        # asymmetric nv-embedqa family).
        embeddings = embedding_factory("openai", s.embed_model, client=embed_client)

        self._faithfulness = Faithfulness(llm=llm)
        self._answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
        self._context_precision = ContextPrecision(llm=llm)
        self._context_recall = ContextRecall(llm=llm)

    async def _ascore(self, question, answer, contexts, ground_truth) -> dict:
        async def _safe(coro):
            try:
                r = await coro
                return float(getattr(r, "value", r))
            except Exception as exc:  # noqa: BLE001 — a metric failure shouldn't kill the row
                print(f"[ragas] metric error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                return float("nan")

        faith, ans_rel, ctx_prec, ctx_rec = await asyncio.gather(
            _safe(self._faithfulness.ascore(
                user_input=question, response=answer, retrieved_contexts=contexts)),
            _safe(self._answer_relevancy.ascore(user_input=question, response=answer)),
            _safe(self._context_precision.ascore(
                user_input=question, reference=ground_truth, retrieved_contexts=contexts)),
            _safe(self._context_recall.ascore(
                user_input=question, retrieved_contexts=contexts, reference=ground_truth)),
        )
        return {
            "faithfulness": faith,
            "answer_relevancy": ans_rel,
            "context_precision": ctx_prec,
            "context_recall": ctx_rec,
        }

    def score(self, question: str, answer: str, contexts: list[str], ground_truth: str) -> dict:
        return asyncio.run(self._ascore(question, answer, contexts, ground_truth))


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS-library cross-check over a dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--version", choices=["baseline", "full"], default="baseline")
    parser.add_argument("--limit", type=int, default=5, help="Keep small — RAGAS fans out LLM calls")
    args = parser.parse_args()

    import core.pipeline as pipeline_mod
    from core.types import ACLContext

    data_path = Path("data") / "eval" / f"{args.dataset}.json"
    items = json.loads(data_path.read_text())[: args.limit]

    # Guardrails OFF for eval (same rationale as eval/run_eval.py).
    pipeline = pipeline_mod.build(
        version=args.version, dataset=args.dataset, enable_guardrails=False
    )
    scorer = RagasScorer()

    rows = []
    for i, item in enumerate(items, start=1):
        acl = ACLContext(tenant_id=item.get("tenant_id", "public"))
        result = pipeline.run(item["question"], acl=acl)
        scores = scorer.score(
            item["question"],
            result.get("answer", ""),
            result.get("contexts", []),
            item.get("ground_truth", ""),
        )
        rows.append(scores)
        print(f"[ragas] item {i}/{len(items)}: {scores}", flush=True)

    # Aggregate (ignoring NaNs)
    def _mean(key):
        vals = [r[key] for r in rows if r[key] == r[key]]  # drop NaN
        return sum(vals) / len(vals) if vals else float("nan")

    out_dir = Path("eval") / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}.{args.version}.ragas.json"
    summary = {k: _mean(k) for k in rows[0]} if rows else {}
    out_path.write_text(json.dumps({"n_items": len(rows), "means": summary, "items": rows}, indent=2))
    print(f"[ragas] means: {summary}")
    print(f"[ragas] wrote {out_path}")


if __name__ == "__main__":
    main()
