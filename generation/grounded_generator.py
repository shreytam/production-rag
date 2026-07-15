"""Grounded, cited generation.

Wraps a `Generator`, assembles a token-budgeted context, asks for a structured
answer (citations as passage numbers), and resolves those numbers back to real
chunk/doc ids. Citation *enforcement* (rejecting uncited claims) lives in the
output guardrails; here we ensure citations are captured and resolvable.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from core.context_assembly import assemble_context
from core.interfaces import Generator
from core.types import Answer, ChatMessage, Citation, ScoredChunk
from generation.prompts import SYSTEM_PROMPT, build_user_prompt

_MARKER_RE = re.compile(r"\[(\d+)\]")


class GeneratedAnswer(BaseModel):
    """Structured-output schema the generator is asked to fill."""

    answer: str = Field(description="The answer text, with [n] citation markers inline.")
    citations: list[int] = Field(
        default_factory=list, description="Passage numbers actually used to support the answer."
    )
    refused: bool = Field(
        default=False, description="True if the context is insufficient to answer."
    )


class GroundedGenerator:
    def __init__(self, generator: Generator, token_budget: int = 4000):
        self.generator = generator
        self.token_budget = token_budget

    def generate(self, question: str, chunks: list[ScoredChunk]) -> Answer:
        ctx = assemble_context(chunks, self.token_budget)
        marker_map = ctx.marker_to_chunk

        messages = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=build_user_prompt(question, ctx.text)),
        ]
        resp = self.generator.complete(messages, response_model=GeneratedAnswer)

        if resp.parsed is not None:
            parsed = GeneratedAnswer.model_validate(resp.parsed)
            answer_text = parsed.answer
            markers = list(parsed.citations)
            refused = parsed.refused
            claimed = sorted(set(int(m) for m in parsed.citations))
        else:
            # Fallback: model didn't honor the schema — scrape markers from text.
            # We cannot distinguish a claimed citation from incidental prose here,
            # so there are no verifiable claims.
            answer_text = resp.text
            markers = [int(m) for m in _MARKER_RE.findall(answer_text)]
            refused = False
            claimed = []

        # Always reconcile with markers actually present in the answer text.
        markers = sorted(set(markers) | {int(m) for m in _MARKER_RE.findall(answer_text)})

        citations: list[Citation] = []
        for n in markers:
            sc = marker_map.get(n)
            if sc is None:
                continue
            citations.append(
                Citation(marker=f"[{n}]", chunk_id=sc.chunk.chunk_id, doc_id=sc.chunk.doc_id)
            )

        answer = Answer(
            text=answer_text,
            citations=citations,
            contexts=[sc for _, sc in ctx.items],
            usage=resp.usage,
            model=resp.model,
            refused=refused,
        )
        # The generator's structured output, in GeneratedAnswer shape, so the
        # SchemaGuardrail can validate the raw output (not the richer Answer).
        answer.metadata["structured_output"] = {
            "answer": answer_text,
            "citations": markers,
            "refused": refused,
        }
        answer.metadata["valid_markers"] = sorted(marker_map.keys())
        answer.metadata["claimed_markers"] = claimed
        return answer
