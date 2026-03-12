from __future__ import annotations


def score_span(text: str) -> float:
    return float(len(str(text or "")))
