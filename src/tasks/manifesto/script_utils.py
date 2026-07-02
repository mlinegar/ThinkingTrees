"""Shared helpers for Manifesto experiment scripts.

These utilities intentionally stay small and script-facing.  Domain logic,
model calls, tree construction, and scoring stay in the individual runners.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from src.tasks.manifesto.span_targets import COMPACT_TARGET_DIMENSIONS


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any, *, default: Any = str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=default) + "\n",
        encoding="utf-8",
    )
    return out


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def append_jsonl(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    append: bool = True,
    default: Any = str,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with out.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, default=default) + "\n")
    return out


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def safe_int(value: Any, *, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def mean(values: Iterable[Any]) -> Optional[float]:
    finite = [value for value in (safe_float(item) for item in values) if value is not None]
    return float(sum(finite) / len(finite)) if finite else None


def parse_csv(value: str | Sequence[str], allowed: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        tokens = [str(item).strip() for item in value]
    else:
        tokens = [part.strip() for part in str(value or "").replace(";", ",").split(",")]
    parsed = tuple(token for token in tokens if token)
    if allowed is not None:
        allowed_set = set(str(item) for item in allowed)
        unknown = [token for token in parsed if token not in allowed_set]
        if unknown:
            raise ValueError(f"unknown values {unknown!r}; allowed: {list(allowed)}")
    return parsed


def parse_int_grid(value: Any, *, name: str = "grid") -> Tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        raw_values = [str(item).strip() for item in value]
    else:
        raw_values = [part.strip() for part in str(value or "").replace(";", ",").split(",")]
    try:
        grid = tuple(int(item) for item in raw_values if item)
    except ValueError as exc:
        raise ValueError(f"{name} must contain integers: {value!r}") from exc
    if not grid:
        raise ValueError(f"{name} must contain at least one integer")
    if any(item <= 0 for item in grid):
        raise ValueError(f"{name} entries must be positive: {grid!r}")
    return grid


def parse_compact_dimensions(value: str | Sequence[str]) -> Tuple[str, ...]:
    tokens = parse_csv(value)
    lowered = {token.lower() for token in tokens}
    if not tokens or "all" in lowered:
        return tuple(COMPACT_TARGET_DIMENSIONS)
    allowed = set(COMPACT_TARGET_DIMENSIONS)
    unknown = [token for token in tokens if token not in allowed]
    if unknown:
        raise ValueError(
            f"unknown target dimensions {unknown!r}; allowed: {list(COMPACT_TARGET_DIMENSIONS)}"
        )
    selected = set(tokens)
    return tuple(dim for dim in COMPACT_TARGET_DIMENSIONS if dim in selected)
