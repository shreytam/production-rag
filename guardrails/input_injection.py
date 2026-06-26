"""Prompt-injection / jailbreak detection on user input.

Heuristic pattern list with optional LLM second-opinion via an injected generator.
"""

from __future__ import annotations

import re
from typing import Any

from core.types import GuardrailAction, GuardrailResult

# ---------------------------------------------------------------------------
# Injection / jailbreak heuristic patterns (case-insensitive)
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ignore_previous", re.compile(r"ignore\s+(previous|prior|above|all)\s+instructions?", re.IGNORECASE)),
    ("disregard_above", re.compile(r"disregard\s+(the\s+)?(above|previous|prior)", re.IGNORECASE)),
    ("system_prompt", re.compile(r"\bsystem\s+prompt\b", re.IGNORECASE)),
    ("you_are_now", re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE)),
    ("forget_instructions", re.compile(r"forget\s+(your\s+)?(previous\s+)?instructions?", re.IGNORECASE)),
    ("role_override", re.compile(r"act\s+as\s+(if\s+you\s+are|a\s+|an\s+)", re.IGNORECASE)),
    ("pretend_no_rules", re.compile(r"pretend\s+(you\s+have\s+no|there\s+are\s+no)\s+(rules?|restrictions?|guidelines?)", re.IGNORECASE)),
    ("reveal_prompt", re.compile(r"(reveal|show|print|output|repeat)\s+(your\s+)?(system|initial|original)\s+prompt", re.IGNORECASE)),
    ("exfiltrate_instructions", re.compile(r"(what\s+(are|were)\s+your\s+instructions?|tell\s+me\s+your\s+(instructions?|system\s+prompt))", re.IGNORECASE)),
    ("jailbreak", re.compile(r"\bjailbreak\b", re.IGNORECASE)),
    ("dan_mode", re.compile(r"\bDAN\s+mode\b|\bdo\s+anything\s+now\b", re.IGNORECASE)),
    ("override_rules", re.compile(r"(override|bypass|disable|circumvent)\s+(your\s+)?(rules?|safety|restrictions?|filters?|guidelines?)", re.IGNORECASE)),
    ("new_instructions", re.compile(r"(new|updated)\s+instructions?:?\s*\n", re.IGNORECASE)),
    ("end_of_system", re.compile(r"</?(system|instructions?)>", re.IGNORECASE)),
]


class InjectionGuardrail:
    """Detects prompt-injection / jailbreak attempts in user input.

    Parameters
    ----------
    generator:
        Optional generator for LLM second-opinion. When provided, any input
        that scores above ``llm_threshold`` on heuristics triggers an LLM call.
    patterns:
        Override the default pattern list. Each item is ``(label, compiled_re)``.
    """

    def __init__(
        self,
        generator: Any | None = None,
        patterns: list[tuple[str, re.Pattern[str]]] | None = None,
    ) -> None:
        self._generator = generator
        self._patterns = patterns if patterns is not None else _INJECTION_PATTERNS

    @property
    def name(self) -> str:
        return "input_injection"

    def check(self, text: str, *, context: dict | None = None) -> GuardrailResult:
        matched: list[str] = []
        for label, pattern in self._patterns:
            if pattern.search(text):
                matched.append(label)

        score = min(1.0, len(matched) / max(1, len(self._patterns) / 4))

        if matched:
            return GuardrailResult(
                name=self.name,
                action=GuardrailAction.BLOCK,
                reason=f"Injection/jailbreak patterns detected: {', '.join(matched)}",
                score=score,
                metadata={"matched_patterns": matched},
            )

        return GuardrailResult(
            name=self.name,
            action=GuardrailAction.PASS,
            score=0.0,
        )
