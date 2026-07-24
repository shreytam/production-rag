"""Offline tests for the observability workstream.

All tests run without a Langfuse server.  The no-op tracer path is exercised
exclusively — ``langfuse`` is never imported or required.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# cost.py
# ---------------------------------------------------------------------------

from core.config import Settings
from core.types import Usage
from observability.cost import (
    PRICING,
    cost_usd,
    cost_per_1k_queries,
    update_usage_cost,
)
from observability.dashboard import print_dashboard, summarize_runs
from observability.langfuse_tracing import Tracer, timed


class TestCostUsd:
    def test_known_model(self) -> None:
        model = "meta/llama-3.1-8b-instruct"
        input_rate, output_rate = PRICING[model]
        expected = (1000 / 1000) * input_rate + (500 / 1000) * output_rate
        assert cost_usd(model, 1000, 500) == pytest.approx(expected)

    def test_unknown_model_returns_zero(self) -> None:
        assert cost_usd("no-such-model/v99", 999, 999) == 0.0

    def test_anthropic_model_nonzero(self) -> None:
        # claude-sonnet-4-6 is a paid model — should produce a positive cost
        result = cost_usd("claude-sonnet-4-6", 1000, 1000)
        assert result > 0.0

    def test_nim_free_tier_zero(self) -> None:
        assert cost_usd("meta/llama-3.3-70b-instruct", 10_000, 5_000) == 0.0


class TestCostPer1kQueries:
    def test_normal(self) -> None:
        assert cost_per_1k_queries(1.0, 500) == pytest.approx(2.0)

    def test_zero_queries(self) -> None:
        assert cost_per_1k_queries(5.0, 0) == 0.0


class TestUpdateUsageCost:
    def test_fills_cost_usd(self) -> None:
        usage = Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        updated = update_usage_cost(usage, "claude-sonnet-4-6")
        assert updated.cost_usd > 0.0
        # Original is unchanged
        assert usage.cost_usd == 0.0

    def test_unknown_model_sets_zero(self) -> None:
        usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        updated = update_usage_cost(usage, "ghost/model")
        assert updated.cost_usd == 0.0

    def test_token_counts_preserved(self) -> None:
        usage = Usage(prompt_tokens=200, completion_tokens=100, total_tokens=300)
        updated = update_usage_cost(usage, "meta/llama-3.1-8b-instruct")
        assert updated.prompt_tokens == 200
        assert updated.completion_tokens == 100


# ---------------------------------------------------------------------------
# langfuse_tracing.py — no-op path (langfuse never imported/required)
# ---------------------------------------------------------------------------


def _disabled_settings() -> Settings:
    return Settings(
        langfuse_enabled=False,
        langfuse_public_key="",
        langfuse_secret_key="",
    )


class TestTracerNoOp:
    def test_tracer_constructs(self) -> None:
        tracer = Tracer(_disabled_settings())
        assert tracer is not None

    def test_span_does_not_raise(self) -> None:
        tracer = Tracer(_disabled_settings())
        with tracer.span("retrieval", query="test") as s:
            s.update(n_hits=5)
            s.score("relevance", 0.9)
        # Reaching here means no exception was raised

    def test_log_query_trace_does_not_raise(self) -> None:
        tracer = Tracer(_disabled_settings())
        usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        tracer.log_query_trace(
            "test-query",
            retrieval_hits=[{"chunk_id": "c1", "score": 0.95}],
            usage=usage,
            stage_latencies={"retrieval": 30.0, "generation": 120.0},
            cost_usd=0.001,
        )

    def test_flush_does_not_raise(self) -> None:
        tracer = Tracer(_disabled_settings())
        tracer.flush()

    def test_langfuse_not_imported_when_disabled(self) -> None:
        """langfuse must not appear in sys.modules when tracing is disabled."""
        # Remove langfuse from modules if it somehow ended up there
        sys.modules.pop("langfuse", None)
        _ = Tracer(_disabled_settings())
        assert "langfuse" not in sys.modules, (
            "langfuse was imported even though tracing is disabled"
        )


class TestTracerEnabledMissingKeys:
    """When enabled=True but keys are blank, should still be a no-op."""

    def test_enabled_without_keys_is_noop(self) -> None:
        settings = Settings(
            langfuse_enabled=True,
            langfuse_public_key="",
            langfuse_secret_key="",
        )
        tracer = Tracer(settings)
        with tracer.span("test") as s:
            s.update(x=1)  # must not raise


# ---------------------------------------------------------------------------
# @timed decorator
# ---------------------------------------------------------------------------

class TestTimed:
    def test_returns_positive_elapsed(self) -> None:
        @timed
        def slow_fn():
            time.sleep(0.01)  # 10 ms
            return 42

        result, elapsed_ms = slow_fn()
        assert result == 42
        assert elapsed_ms > 0.0

    def test_elapsed_roughly_correct(self) -> None:
        @timed
        def sleep_fn():
            time.sleep(0.05)

        _, elapsed_ms = sleep_fn()
        # Should be at least 40 ms (generous lower bound for CI jitter)
        assert elapsed_ms >= 40.0


# ---------------------------------------------------------------------------
# dashboard.py
# ---------------------------------------------------------------------------


class TestSummarizeRuns:
    def test_basic_aggregation(self, tmp_path: Path) -> None:
        records = [
            {"latency_ms": 100.0, "cost_usd": 0.001, "metrics": {"faithfulness": 0.8}},
            {"latency_ms": 200.0, "cost_usd": 0.002, "metrics": {"faithfulness": 0.9}},
        ]
        p = tmp_path / "results.json"
        p.write_text(json.dumps(records))

        summary = summarize_runs([p])

        assert summary["n_runs"] == 2
        assert summary["latency_ms"]["mean"] == pytest.approx(150.0)
        assert summary["latency_ms"]["min"] == pytest.approx(100.0)
        assert summary["latency_ms"]["max"] == pytest.approx(200.0)
        assert summary["cost_usd"]["total"] == pytest.approx(0.003)
        assert summary["cost_usd"]["mean"] == pytest.approx(0.0015)
        assert summary["metrics"]["faithfulness"]["mean"] == pytest.approx(0.85)

    def test_missing_fields_tolerated(self, tmp_path: Path) -> None:
        records = [{"latency_ms": 50.0}, {"cost_usd": 0.005}]
        p = tmp_path / "results.json"
        p.write_text(json.dumps(records))

        summary = summarize_runs([p])
        assert summary["n_runs"] == 2
        # latency only from first record
        assert summary["latency_ms"]["mean"] == pytest.approx(50.0)
        # cost only from second record
        assert summary["cost_usd"]["total"] == pytest.approx(0.005)
        assert summary["metrics"] == {}

    def test_multiple_files(self, tmp_path: Path) -> None:
        p1 = tmp_path / "r1.json"
        p1.write_text(json.dumps([{"latency_ms": 100.0}]))
        p2 = tmp_path / "r2.json"
        p2.write_text(json.dumps([{"latency_ms": 300.0}]))

        summary = summarize_runs([p1, p2])
        assert summary["n_runs"] == 2
        assert summary["latency_ms"]["mean"] == pytest.approx(200.0)

    def test_print_dashboard_no_crash(self, tmp_path: Path, capsys) -> None:
        records = [{"latency_ms": 80.0, "cost_usd": 0.0, "metrics": {"answer_relevance": 0.75}}]
        p = tmp_path / "results.json"
        p.write_text(json.dumps(records))

        summary = summarize_runs([p])
        print_dashboard(summary)  # must not raise

        captured = capsys.readouterr()
        assert "RAG Eval Dashboard" in captured.out
        assert "answer_relevance" in captured.out
