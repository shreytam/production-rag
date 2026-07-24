"""SP7 — model cost table completeness and unknown-model visibility.

These tests operationalise two guarantees for ``observability/cost.py``:

1. Every default model identifier configured in ``core.config.Settings`` has a
   corresponding entry in ``PRICING`` (the single hardcoded source of truth for
   model cost), so real runs never silently cost $0 for a model the app
   actually uses by default.
2. ``cost_usd`` never raises for an unknown model (still returns 0.0), but logs
   a warning the first time it sees a given unknown model name, deduped so it
   only warns once per model.
"""

from __future__ import annotations

import logging

from core.config import Settings
from observability.cost import PRICING, cost_usd


# Every Settings field that names a *default* model identifier. Enumerated
# explicitly (rather than introspected) so this test fails loudly if a new
# model-bearing field is added to Settings without a matching PRICING entry.
DEFAULT_MODEL_FIELDS = (
    "embed_model",
    "gen_model",
    "context_model",
    "judge_model",
    "anthropic_model",
    "anthropic_context_model",
    "anthropic_judge_model",
    "reranker_local_model",
    "reranker_nim_model",
)


class TestPricingCompleteness:
    def test_every_default_model_is_priced(self) -> None:
        settings = Settings(_env_file=None)
        missing = [
            field
            for field in DEFAULT_MODEL_FIELDS
            if getattr(settings, field) not in PRICING
        ]
        assert not missing, f"PRICING is missing entries for default models: {missing}"


class TestCostUsdMath:
    def test_known_model_math(self) -> None:
        input_rate, output_rate = PRICING["claude-sonnet-4-6"]
        expected = input_rate + output_rate
        assert cost_usd("claude-sonnet-4-6", 1000, 1000) == expected

    def test_zero_priced_model_returns_zero(self) -> None:
        assert cost_usd("baai/bge-m3", 10_000, 0) == 0.0


class TestUnknownModelWarning:
    def test_unknown_model_returns_zero_and_does_not_raise(self) -> None:
        assert cost_usd("totally/unknown-model-xyz", 100, 100) == 0.0

    def test_unknown_model_warns_once(self, caplog) -> None:
        model = "some/never-before-seen-model"
        with caplog.at_level(logging.WARNING):
            cost_usd(model, 10, 10)
            cost_usd(model, 10, 10)

        matching = [r for r in caplog.records if model in r.getMessage()]
        assert len(matching) == 1
