"""Shared deterministic toy data for the Markov appendix walkthrough and slide deck."""

from __future__ import annotations

from typing import Dict, List


REGIME_COLORS = {
    "A": "#4C78A8",
    "B": "#F58518",
    "C": "#54A24B",
    "D": "#E45756",
}

REGIME_VOCAB = {
    "A": ["mist", "lake", "reed"],
    "B": ["rust", "ember", "brick"],
    "C": ["fern", "moss", "leaf"],
    "D": ["plum", "rose", "wine"],
}


def lighten_hex(color: str, factor: float = 0.78) -> str:
    color = color.lstrip("#")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    rr = int(round(255 - (255 - r) * factor))
    gg = int(round(255 - (255 - g) * factor))
    bb = int(round(255 - (255 - b) * factor))
    return f"#{rr:02x}{gg:02x}{bb:02x}"


def token_regime_sequence() -> List[str]:
    return ["A", "A", "A", "B", "B", "C", "C", "D", "D", "A", "A", "B"]


def token_words() -> List[str]:
    return [
        "mist",
        "lake",
        "reed",
        "rust",
        "brick",
        "fern",
        "moss",
        "plum",
        "wine",
        "lake",
        "mist",
        "ember",
    ]


def leaf_summaries(regimes: List[str], leaf_size: int = 4) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for i in range(0, len(regimes), leaf_size):
        span = regimes[i : i + leaf_size]
        count = sum(1 for a, b in zip(span[:-1], span[1:]) if a != b)
        out.append(
            {
                "label": f"L{1 + i // leaf_size}",
                "start": i + 1,
                "end": i + leaf_size,
                "count": count,
                "first": span[0],
                "last": span[-1],
            }
        )
    return out


def merge_summary(left: Dict[str, object], right: Dict[str, object], label: str) -> Dict[str, object]:
    correction = 0 if str(left["last"]) == str(right["first"]) else 1
    return {
        "label": label,
        "count": int(left["count"]) + int(right["count"]) + correction,
        "first": str(left["first"]),
        "last": str(right["last"]),
        "correction": correction,
    }


__all__ = [
    "REGIME_COLORS",
    "REGIME_VOCAB",
    "leaf_summaries",
    "lighten_hex",
    "merge_summary",
    "token_regime_sequence",
    "token_words",
]
