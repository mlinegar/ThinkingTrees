from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence, Tuple


def parse_csv(value: str | Sequence[Any], *, allowed: Optional[Iterable[str]] = None) -> Tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        parts = [str(item).strip() for item in value]
    else:
        parts = [part.strip() for part in str(value or "").replace(";", ",").split(",")]
    parsed = tuple(part for part in parts if part)
    if allowed is not None:
        allowed_values = tuple(str(item) for item in allowed)
        allowed_set = set(allowed_values)
        unknown = [part for part in parsed if part not in allowed_set]
        if unknown:
            raise ValueError(f"unknown values {unknown!r}; allowed: {list(allowed_values)}")
    return parsed


def parse_int_grid(value: Any, *, name: str = "grid") -> Tuple[int, ...]:
    try:
        grid = tuple(int(item) for item in parse_csv(value))
    except ValueError as exc:
        raise ValueError(f"{name} must contain integers: {value!r}") from exc
    if not grid:
        raise ValueError(f"{name} must contain at least one integer")
    if any(item <= 0 for item in grid):
        raise ValueError(f"{name} entries must be positive: {grid!r}")
    return grid


def parse_float_grid(value: Any, *, name: str = "grid") -> Tuple[float, ...]:
    try:
        grid = tuple(float(item) for item in parse_csv(value))
    except ValueError as exc:
        raise ValueError(f"{name} must contain numbers: {value!r}") from exc
    if not grid:
        raise ValueError(f"{name} must contain at least one number")
    if any(not math.isfinite(item) for item in grid):
        raise ValueError(f"{name} entries must be finite: {grid!r}")
    return grid


def parse_token_list(
    value: Any,
    *,
    default: Sequence[Any] = (),
    separators: str = ",;",
) -> list[str]:
    if value is None:
        return [str(item) for item in default]
    if isinstance(value, (list, tuple)):
        parts = [str(item).strip() for item in value]
    else:
        normalized = str(value or "")
        for separator in str(separators):
            normalized = normalized.replace(separator, " ")
        parts = normalized.split()
    parsed = [part for part in parts if part]
    return parsed or [str(item) for item in default]


def parse_int_list(
    value: Any,
    *,
    default: Sequence[int] = (),
    separators: str = ",;",
) -> list[int]:
    return [
        int(item)
        for item in parse_token_list(
            value,
            default=default,
            separators=separators,
        )
    ]


def parse_float_list(
    value: Any,
    *,
    default: Sequence[float] = (),
    separators: str = ",;",
) -> list[float]:
    return [
        float(item)
        for item in parse_token_list(
            value,
            default=default,
            separators=separators,
        )
    ]


def parse_str_list(
    value: Any,
    *,
    default: Sequence[str] = (),
    separators: str = ",;",
) -> list[str]:
    return parse_token_list(value, default=default, separators=separators)


def safe_float(value: Any, *, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if math.isfinite(converted) else default


def safe_int(value: Any, *, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def coerce_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = str(value).strip()
    lowered = text.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"none", "null", ""}:
        return None
    try:
        if all(ch not in text for ch in (".", "e", "E")):
            return int(text)
        return float(text)
    except ValueError:
        return value


def mean(values: Iterable[Any]) -> Optional[float]:
    finite = [item for item in (safe_float(value) for value in values) if item is not None]
    return float(sum(finite) / len(finite)) if finite else None


__all__ = [
    "coerce_scalar",
    "mean",
    "parse_csv",
    "parse_float_list",
    "parse_float_grid",
    "parse_int_list",
    "parse_int_grid",
    "parse_str_list",
    "parse_token_list",
    "safe_float",
    "safe_int",
]
