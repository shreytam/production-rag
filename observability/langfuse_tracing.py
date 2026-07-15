"""Langfuse-backed tracing with a graceful no-op fallback.

When ``langfuse_enabled`` is False or keys are missing the ``Tracer`` class is
fully functional but silent — no network calls, no imports of the ``langfuse``
package.  The real Langfuse client is imported lazily *only* when tracing is
actually enabled, so offline / test environments work without the package.

Public surface
--------------
Tracer(settings)
    .span(name, **metadata) -> context manager yielding a SpanHandle
        SpanHandle.update(**kwargs)  — attach arbitrary metadata
        SpanHandle.score(name, value) — record a numeric score
    .log_query_trace(name, *, retrieval_hits, usage, stage_latencies, cost_usd)
    .flush()  — flush buffered events (no-op when disabled)

@timed
    Decorator / context manager that measures elapsed milliseconds.
"""

from __future__ import annotations

import time
import functools
from contextlib import contextmanager
from typing import Any, Generator, Callable

from core.config import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timed(fn: Callable) -> Callable:
    """Decorator that calls the wrapped function and returns (result, elapsed_ms).

    Usage::

        @timed
        def my_fn():
            ...

        result, ms = my_fn()
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return result, elapsed_ms
    return wrapper


# ---------------------------------------------------------------------------
# Span handles
# ---------------------------------------------------------------------------

class _NoOpSpan:
    """Silent span — all calls succeed and do nothing."""

    def update(self, **kwargs: Any) -> None:  # noqa: ANN401
        pass

    def score(self, name: str, value: float) -> None:
        pass


class _LangfuseSpan:
    """Thin wrapper around a real Langfuse v3/v4 observation object.

    The v3+ SDK is OpenTelemetry-based: an observation exposes ``.update(...)``
    (output / input / metadata / model params) and ``.score(...)`` to attach a
    numeric score to the current observation.  We never touch the OTel context
    directly — the object yielded by ``start_as_current_observation`` carries it.
    """

    def __init__(self, span: Any) -> None:
        self._span = span

    def update(self, **kwargs: Any) -> None:
        try:
            self._span.update(**kwargs)
        except Exception:  # pragma: no cover
            pass

    def score(self, name: str, value: float) -> None:
        try:
            self._span.score(name=name, value=value, data_type="NUMERIC")
        except Exception:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------

class Tracer:
    """Observability facade — wraps Langfuse when enabled, no-ops otherwise.

    Parameters
    ----------
    settings:
        A ``core.config.Settings`` instance.  The tracer inspects
        ``langfuse_enabled``, ``langfuse_host``, ``langfuse_public_key``, and
        ``langfuse_secret_key``.
    """

    def __init__(self, settings: Settings) -> None:
        self._enabled = (
            settings.langfuse_enabled
            and bool(settings.langfuse_public_key)
            and bool(settings.langfuse_secret_key)
        )
        self._client: Any = None
        self._pii_redactor: Any = None

        if self._enabled:
            # Lazy import — only when actually needed.  Kept inside this branch
            # so a disabled tracer never imports the package (offline/test safe).
            try:
                from langfuse import Langfuse  # noqa: PLC0415
                # v3/v4 constructor still accepts explicit keys + host; it sets
                # up the OTel tracer provider behind the scenes.
                self._client = Langfuse(
                    host=settings.langfuse_host,
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    mask=self._mask_data,
                    sample_rate=settings.langfuse_sample_rate,
                )
            except Exception:
                # If Langfuse fails to initialise, degrade silently
                self._enabled = False

    def _mask_data(self, data: Any) -> Any:
        """Recursively scan and redact PII from traced attributes using PIIRedactor."""
        if isinstance(data, str):
            if self._pii_redactor is None:
                from ingest.pii import PIIRedactor
                self._pii_redactor = PIIRedactor()
            try:
                cleaned, _ = self._pii_redactor.redact(data)
                return cleaned
            except Exception:
                # Fail-closed: return placeholder on mask error
                return "[PII_REDACTION_ERROR]"
                
        if isinstance(data, dict):
            return {k: self._mask_data(v) for k, v in data.items()}
            
        if isinstance(data, list):
            return [self._mask_data(item) for item in data]
            
        return data

    # ------------------------------------------------------------------
    # Context manager: tracer.span("stage", metadata=...)
    # ------------------------------------------------------------------

    @contextmanager
    def span(
        self, name: str, **metadata: Any
    ) -> Generator[_NoOpSpan | _LangfuseSpan, None, None]:
        """Context manager that wraps a pipeline stage.

        Yields a handle whose ``.update(**kwargs)`` / ``.score(name, value)``
        methods attach data to the span.  Always safe to call — falls back to
        no-op when disabled.

        Example::

            with tracer.span("retrieval", query=q.text) as s:
                hits = retrieve(q)
                s.update(n_hits=len(hits))
        """
        if not self._enabled or self._client is None:
            yield _NoOpSpan()
            return

        # v3/v4: a single context manager opens the observation, makes it the
        # current OTel span, and auto-closes it on exit — no manual .end().
        # Only *creating* the span falls back to no-op; once we're inside the
        # `with`, a caller exception must propagate (and still close the span).
        try:
            span_cm = self._client.start_as_current_observation(
                as_type="span", name=name, metadata=metadata or None
            )
        except Exception:  # pragma: no cover — never let tracing break the call
            yield _NoOpSpan()
            return

        with span_cm as span_obj:
            yield _LangfuseSpan(span_obj)

    # ------------------------------------------------------------------
    # Convenience: log a complete query trace
    # ------------------------------------------------------------------

    def log_query_trace(
        self,
        name: str,
        *,
        retrieval_hits: list[dict[str, Any]] | None = None,
        usage: Any | None = None,  # core.types.Usage
        stage_latencies: dict[str, float] | None = None,
        cost_usd: float = 0.0,
    ) -> None:
        """Log a structured trace for a full RAG query.

        Parameters
        ----------
        name:
            Trace name / query label.
        retrieval_hits:
            List of dicts with at least ``chunk_id`` and ``score``.
        usage:
            A ``core.types.Usage`` with token counts.
        stage_latencies:
            Mapping of stage name -> elapsed milliseconds.
        cost_usd:
            Total estimated cost for this query.
        """
        if not self._enabled or self._client is None:
            return

        metadata: dict[str, Any] = {
            "stage_latencies_ms": stage_latencies or {},
            "cost_usd": cost_usd,
        }
        if usage is not None:
            metadata["tokens"] = {
                "prompt": usage.prompt_tokens,
                "completion": usage.completion_tokens,
                "total": usage.total_tokens,
            }
        if retrieval_hits:
            metadata["retrieval"] = {
                "n_hits": len(retrieval_hits),
                "hits": retrieval_hits,
            }

        try:
            # v3/v4 has no bare `trace()`; emit a single observation that opens
            # (and is the root of) a trace carrying the query-level metadata.
            with self._client.start_as_current_observation(
                as_type="span", name=name, metadata=metadata
            ):
                pass
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Flush buffered Langfuse events.  No-op when disabled."""
        if self._enabled and self._client is not None:
            try:
                self._client.flush()
            except Exception:  # pragma: no cover
                pass
