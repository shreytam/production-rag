from __future__ import annotations
from typing import Any
from core.types import PIISpan

class PresidioPIIDetector:
    def __init__(self) -> None:
        self._analyzer: Any = None

    def _ensure_presidio(self) -> Any:
        if self._analyzer is not None:
            return self._analyzer
        try:
            from presidio_analyzer import AnalyzerEngine
        except ImportError:
            raise ImportError(
                "PresidioPIIDetector requires Presidio dependencies. "
                "Install them using: pip install .[pii-ner]"
            )
        self._analyzer = AnalyzerEngine()
        return self._analyzer

    def detect(self, text: str) -> list[PIISpan]:
        analyzer = self._ensure_presidio()
        results = analyzer.analyze(text=text, language="en")
        spans = []
        for r in results:
            spans.append(PIISpan(type=r.entity_type, start=r.start, end=r.end))
        return spans
