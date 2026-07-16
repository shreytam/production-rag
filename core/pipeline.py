"""RAGPipeline — wires the swappable components into one query path.

`build(version, dataset)` constructs either the naive **baseline** (dense -> generate)
or the **full** pipeline (hybrid retrieve -> rerank -> grounded generate) entirely
from config via `core.registry`. `run()` returns the plain dict the eval harness
consumes; `answer()` returns the rich `Answer` for the app/guardrails.
"""

from __future__ import annotations

import logging
from typing import Any

from core.config import Settings, get_settings
from core.registry import (
    build_embedder,
    build_generator,
    build_reranker,
    build_vector_store,
)
from core.types import ACLContext, Answer, Query
from generation.grounded_generator import GroundedGenerator
from guardrails.input_injection import scan_for_injection
from guardrails.runner import GuardrailRunner, default_runner
from observability.cost import cost_usd
from observability.langfuse_tracing import Tracer, timed
from retrieval.hybrid import DenseRetriever, HybridRetriever

logger = logging.getLogger(__name__)

DEFAULT_TENANT = "public"


OUTPUT_BLOCK_MESSAGE = (
    "I can't provide an answer that passes the system's safety and grounding "
    "checks for this request."
)


def _dedup(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


class HybridIndexError(Exception):
    """Raised when hybrid retrieval is required but the sparse index is empty/missing."""
    pass


class RAGPipeline:
    def __init__(
        self,
        retriever,
        grounded: GroundedGenerator,
        settings: Settings,
        tracer: Tracer | None = None,
        guardrails: GuardrailRunner | None = None,
    ):
        self.retriever = retriever
        self.grounded = grounded
        self.settings = settings
        self.default_acl = ACLContext(tenant_id=DEFAULT_TENANT)
        # No-op when langfuse is disabled — safe to always hold a tracer.
        self.tracer = tracer or Tracer(settings)
        # None => guardrails disabled (eval path); otherwise enforced per query.
        self.guardrails = guardrails

    def _refused(self, message: str, results: list) -> Answer:
        """Build a well-formed refused Answer (no retrieval/generation ran)."""
        return Answer(
            text=message,
            refused=True,
            metadata={
                "guardrails": {"input": [r.model_dump() for r in results]},
                "retrieved_doc_ids": [],
                "retrieved_chunk_ids": [],
                "stage_latencies_ms": {},
                "cost_usd": 0.0,
            },
        )

    def answer(self, question: str, acl: ACLContext | None = None) -> Answer:
        acl = acl or self.default_acl

        latencies: dict[str, float] = {}
        guard_log: dict[str, list] = {}

        # 1. Apply input guard check BEFORE creating the root observation span
        # to ensure raw unredacted questions never land in default span parameters
        if self.guardrails is not None:
            in_results = self.guardrails.check_input(question)
            guard_log["input"] = [r.model_dump() for r in in_results]
            
            if self.guardrails.blocked(in_results):
                # Trace details of check blocks safely
                with self.tracer.span(
                    "rag.query", question="[BLOCKED]", tenant=acl.tenant_id
                ) as root:
                    reason = "; ".join(r.reason for r in in_results if not r.ok)
                    root.update(
                        output={
                            "refused": True,
                            "blocked_by": "input_guardrail",
                            "reason": reason,
                        }
                    )
                return self._refused(
                    "This request was blocked by an input safety check.", in_results
                )
            # Safe clean value
            question = self.guardrails.apply_redactions(question, in_results)

        # 2. Main observation span initialized with the redacted/clean form
        with self.tracer.span(
            "rag.query", question=question, tenant=acl.tenant_id
        ) as root:
            # We recreate safe tracing records since actual redaction ran
            if self.guardrails is not None and guard_log:
                with self.tracer.span("guardrail.input") as s_in:
                    s_in.update(output={"actions": [r.action.value for r in in_results]})

            q = Query(
                text=question,
                acl=acl,
                top_k=self.settings.retrieve_top_k,
                rerank_top_n=self.settings.rerank_top_n,
            )

            with self.tracer.span("retrieval", top_k=q.top_k) as s_ret:
                scored, ms = timed(self.retriever.retrieve)(q)
                latencies["retrieval_ms"] = ms
                s_ret.update(output={"n_hits": len(scored)})
                suspected = sorted({
                    lbl for sc in scored for lbl in scan_for_injection(sc.chunk.text)
                })
                if suspected:
                    s_ret.update(output={"indirect_injection_suspected": suspected})
                    logger.warning("indirect_injection_suspected: %s", suspected)

            with self.tracer.span("generation", model=self.settings.gen_model) as s_gen:
                ans, ms = timed(self.grounded.generate)(question, scored)
                latencies["generation_ms"] = ms
                s_gen.update(
                    output={"refused": ans.refused, "n_citations": len(ans.citations)},
                    metadata={
                        "prompt_tokens": ans.usage.prompt_tokens,
                        "completion_tokens": ans.usage.completion_tokens,
                    },
                )

            if suspected:
                ans.metadata["indirect_injection_suspected"] = suspected

            # --- Output guardrails: citation / schema / groundedness ---------
            if self.guardrails is not None:
                with self.tracer.span("guardrail.output") as s_out:
                    out_results = self.guardrails.check_output(
                        ans,
                        context={
                            "question": question,
                            "context_chunk_ids": {sc.chunk_id for sc in ans.contexts},
                            "contexts": [sc.chunk.text for sc in ans.contexts],
                            # GeneratedAnswer-shaped dict for the SchemaGuardrail.
                            "candidate": ans.metadata.get("structured_output", {}),
                        },
                    )
                    guard_log["output"] = [r.model_dump() for r in out_results]
                    s_out.update(output={"actions": [r.action.value for r in out_results]})
                    ans.text = self.guardrails.apply_redactions(ans.text, out_results)
                    
                    # Scrub metadata duplicate structured_output answer if redactions took place
                    from core.types import GuardrailAction
                    if any(r.action == GuardrailAction.REDACT for r in out_results):
                        if "structured_output" in ans.metadata and "answer" in ans.metadata["structured_output"]:
                            raw_meta_ans = ans.metadata["structured_output"]["answer"]
                            ans.metadata["structured_output"]["answer"] = self.guardrails.apply_redactions(raw_meta_ans, out_results)

                    if self.guardrails.blocked(out_results):
                        ans.refused = True
                        ans.metadata["blocked_by"] = "output_guardrail"
                        ans.metadata["block_reason"] = "; ".join(
                            r.reason for r in out_results if not r.ok
                        )

            cost = cost_usd(
                self.settings.gen_model,
                ans.usage.prompt_tokens,
                ans.usage.completion_tokens,
            )
            root.update(
                output={
                    "stage_latencies_ms": latencies,
                    "cost_usd": cost,
                    "n_hits": len(scored),
                    "refused": ans.refused,
                }
            )

        # Stash the retrieval set for metric reporting.
        ans.metadata["retrieved_doc_ids"] = _dedup([sc.chunk.doc_id for sc in scored])
        ans.metadata["retrieved_chunk_ids"] = [sc.chunk_id for sc in scored]
        ans.metadata["stage_latencies_ms"] = latencies
        ans.metadata["cost_usd"] = cost
        if guard_log:
            ans.metadata["guardrails"] = guard_log

        # SP2: an output-guardrail BLOCK must surface ONLY a generic refusal —
        # scrub the content AND every metadata copy of it. The block reason stays
        # in the trace/log (set on the root span), never on the returned object.
        if ans.metadata.get("blocked_by") == "output_guardrail":
            ans.text = OUTPUT_BLOCK_MESSAGE
            ans.citations = []
            ans.contexts = []
            ans.metadata["retrieved_doc_ids"] = []
            ans.metadata["retrieved_chunk_ids"] = []
            ans.metadata.pop("structured_output", None)
            ans.metadata.pop("block_reason", None)
            for phase_results in ans.metadata.get("guardrails", {}).values():
                for r in phase_results:
                    r.pop("reason", None)
                    r.pop("payload", None)
                    r.pop("metadata", None)
        return ans

    def run(self, question: str, acl: ACLContext | None = None) -> dict[str, Any]:
        ans = self.answer(question, acl)
        return {
            "answer": ans.text,
            "retrieved_ids": ans.metadata.get("retrieved_doc_ids", []),
            "retrieved_chunk_ids": ans.metadata.get("retrieved_chunk_ids", []),
            "contexts": [sc.chunk.text for sc in ans.contexts],
            "citations": [c.model_dump() for c in ans.citations],
            "usage": ans.usage.model_dump(),
            "refused": ans.refused,
            "answer_obj": ans,
        }


def build(
    version: str = "full",
    corpus: str | None = None,
    settings: Settings | None = None,
    enable_guardrails: bool | None = None,
) -> RAGPipeline:
    """Construct a pipeline version from config.

    ``enable_guardrails``: None => follow ``settings.guardrails_enabled`` (the
    production default, True). The eval entry points pass ``False`` explicitly so
    blocking guards never confound generation metrics or add per-item LLM cost.

    ``corpus``: unused now that ``version="full"`` no longer binds a corpus-pickle
    sparse index at build time. Its sparse retriever is a `TenantSparseStore` that
    resolves the caller's tenant index at QUERY time instead (see
    ``core.registry.build_tenant_sparse_store``). Kept as a parameter for existing
    call sites; callers may drop it in a future cleanup.
    """
    s = settings or get_settings()
    embedder = build_embedder(s)
    store = build_vector_store(s)
    generator = build_generator("gen", s)
    grounded = GroundedGenerator(generator, token_budget=s.context_token_budget, settings=s)

    if version == "baseline":
        retriever = DenseRetriever(embedder, store)
    elif version == "full":
        from core.registry import build_tenant_sparse_store

        sparse = build_tenant_sparse_store(s)
        reranker = build_reranker(s)
        retriever = HybridRetriever(embedder, store, sparse, reranker, rrf_k=s.rrf_k)
    else:
        raise ValueError(f"Unknown pipeline version: {version!r}")

    use_guards = s.guardrails_enabled if enable_guardrails is None else enable_guardrails
    guardrails = default_runner(generator=generator) if use_guards else None

    return RAGPipeline(retriever, grounded, s, guardrails=guardrails)
