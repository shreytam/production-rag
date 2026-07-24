"""Two-tiered, tenant-safe query rewriter (SP12).

Tier 1 is a deterministic per-tenant synonym substitution read from Redis
(`rewriter:synonyms:{tenant_id}`). Tier 2 is a fallback LLM query expansion for
longer queries that matched no synonym. The rewriter is *fail-soft*: any Redis or
LLM error degrades to the best-effort query (synonym result if any, else raw) and
never raises into the query path.

Offline-safe: `redis` is imported lazily inside `_get_client`, so importing this
module — and constructing a `HybridQueryRewriter` — needs neither Redis nor the
`redis` package. Tests inject a fake client via `redis_client=`.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from core.interfaces import Generator
from core.types import ACLContext, ChatMessage

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are an expert search assistant. Rewrite the user's query to maximize "
    "retrieval matching. Return a single descriptive search statement. Keep "
    "specialized terms and expand acronyms. Do not answer the query."
)


def _decode(v: Any) -> str:
    return v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)


class HybridQueryRewriter:
    """Synonym substitution with an LLM expansion fallback. See module docstring."""

    def __init__(
        self,
        generator: Generator,
        redis_url: str,
        *,
        llm_enabled: bool = True,
        llm_threshold: int = 5,
        redis_client: Any | None = None,
    ) -> None:
        self._gen = generator
        self._redis_url = redis_url
        self._llm_enabled = llm_enabled
        self._llm_threshold = llm_threshold
        self._client = redis_client  # injected in tests; lazily built otherwise

    def _get_client(self) -> Any | None:
        if self._client is None:
            try:
                import redis  # lazy: keeps this module offline-safe

                self._client = redis.from_url(self._redis_url)
            except Exception as exc:  # pragma: no cover - infra path
                logger.warning("rewriter: redis client init failed: %s", exc)
                return None
        return self._client

    def _synonyms(self, tenant_id: str) -> dict[str, str]:
        client = self._get_client()
        if client is None:
            return {}
        try:
            raw = client.hgetall(f"rewriter:synonyms:{tenant_id}")
        except Exception as exc:
            logger.warning("rewriter: synonym lookup failed: %s", exc)
            return {}
        return {_decode(k): _decode(v) for k, v in (raw or {}).items()}

    def rewrite(self, query: str, acl: ACLContext) -> str:
        rewritten = query
        replaced = False
        syns = self._synonyms(acl.tenant_id)
        # Longer shortcuts first so multi-word phrases win over their substrings.
        for shortcut, full in sorted(syns.items(), key=lambda kv: -len(kv[0])):
            pattern = re.compile(rf"\b{re.escape(shortcut)}\b", re.IGNORECASE)
            if pattern.search(rewritten):
                rewritten = pattern.sub(full, rewritten)
                replaced = True

        if self._llm_enabled and not replaced and len(query.split()) >= self._llm_threshold:
            try:
                resp = self._gen.complete(
                    [
                        ChatMessage(role="system", content=_SYSTEM),
                        ChatMessage(role="user", content=query),
                    ],
                    max_tokens=128,
                    temperature=0.0,
                )
                expanded = (resp.text or "").strip()
                if expanded:
                    rewritten = expanded
            except Exception as exc:
                logger.error("rewriter: LLM expansion failed, using best-effort: %s", exc)

        return rewritten
