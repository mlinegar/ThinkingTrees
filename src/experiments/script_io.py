from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Mapping, Optional, Sequence


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_json_object(path: str | Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def write_json(
    path: str | Path,
    payload: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = True,
    ensure_ascii: bool = True,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            json_safe(payload),
            indent=indent,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
        )
        + "\n",
        encoding="utf-8",
    )
    return out


def read_jsonl(path: str | Path, *, skip_bad: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                if skip_bad:
                    continue
                raise
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_jsonl(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    append: bool = False,
    sort_keys: bool = True,
    ensure_ascii: bool = True,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with out.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    json_safe(dict(row)),
                    sort_keys=sort_keys,
                    ensure_ascii=ensure_ascii,
                )
                + "\n"
            )
        handle.flush()
    return out


def append_jsonl(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]] | Mapping[str, Any],
    *,
    sort_keys: bool = True,
    ensure_ascii: bool = True,
) -> Path:
    if isinstance(rows, Mapping):
        payloads: Iterable[Mapping[str, Any]] = (rows,)
    else:
        payloads = rows
    return write_jsonl(
        path,
        payloads,
        append=True,
        sort_keys=sort_keys,
        ensure_ascii=ensure_ascii,
    )


def stable_digest(
    payload: Any,
    *,
    algorithm: str = "sha256",
    length: Optional[int] = None,
) -> str:
    encoded = json.dumps(
        json_safe(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    digest = hashlib.new(str(algorithm), encoded).hexdigest()
    return digest[: int(length)] if length is not None else digest


def stable_hash(
    text: Any,
    *,
    algorithm: str = "blake2b",
    digest_size: int = 16,
    length: Optional[int] = None,
) -> str:
    data = str(text or "").encode("utf-8", errors="ignore")
    if algorithm == "blake2b":
        digest = hashlib.blake2b(data, digest_size=int(digest_size)).hexdigest()
    else:
        digest = hashlib.new(str(algorithm), data).hexdigest()
    return digest[: int(length)] if length is not None else digest


def require_within_chars(text: Any, *, max_chars: int, label: str) -> str:
    rendered = str(text or "")
    if int(max_chars) > 0 and len(rendered) > int(max_chars):
        raise RuntimeError(
            f"no-truncation guard: {label} has {len(rendered)} chars but "
            f"max_chars={max_chars}. Increase the configured context budget "
            "or reduce the input size."
        )
    return rendered


class JsonlCallCache:
    """Thread-safe append-only JSONL cache for successful script calls."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._rows: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for row in read_jsonl(self.path, skip_bad=True):
                key = str(row.get("cache_key") or "")
                if key:
                    self._rows[key] = dict(row)

    def get(self, key: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._rows.get(str(key))
            return dict(row) if row is not None else None

    def put(self, key: str, row: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        payload["cache_key"] = str(key)
        payload.setdefault("created_at", now_iso())
        with self._lock:
            existing = self._rows.get(str(key))
            if existing is not None:
                return dict(existing)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(json_safe(payload), sort_keys=True) + "\n")
                handle.flush()
            self._rows[str(key)] = dict(payload)
            return dict(payload)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            by_kind: dict[str, int] = {}
            for row in self._rows.values():
                kind = str(row.get("kind") or "unknown")
                by_kind[kind] = by_kind.get(kind, 0) + 1
            return {"path": str(self.path), "entries": len(self._rows), "by_kind": by_kind}


__all__ = [
    "JsonlCallCache",
    "append_jsonl",
    "json_safe",
    "now_iso",
    "now_stamp",
    "read_json",
    "read_json_object",
    "read_jsonl",
    "require_within_chars",
    "stable_digest",
    "stable_hash",
    "write_json",
    "write_jsonl",
]
