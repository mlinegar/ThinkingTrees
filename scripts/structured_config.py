from __future__ import annotations

import json
import math
from pathlib import Path
import tomllib
from typing import Any, Mapping


def load_structured_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    suffix = config_path.suffix.lower()
    if suffix == ".toml":
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    elif suffix in {"", ".json"}:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        raise ValueError(
            f"unsupported config format for {config_path}; use .toml or .json"
        )
    if not isinstance(payload, Mapping):
        raise ValueError(f"config at {config_path} must be a mapping/object")
    return dict(payload)


def write_structured_config(path: str | Path, payload: Mapping[str, Any]) -> Path:
    config_path = Path(path).expanduser()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = config_path.suffix.lower()
    if suffix == ".toml":
        text = render_toml(payload)
    else:
        text = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    config_path.write_text(text, encoding="utf-8")
    return config_path


def render_toml(payload: Mapping[str, Any]) -> str:
    lines = _render_mapping(payload, prefix=())
    return "\n".join(lines).rstrip() + "\n"


def _render_mapping(mapping: Mapping[str, Any], *, prefix: tuple[str, ...]) -> list[str]:
    scalar_items: list[tuple[str, Any]] = []
    table_items: list[tuple[str, Mapping[str, Any]]] = []
    array_table_items: list[tuple[str, list[Mapping[str, Any]]]] = []

    for key, value in mapping.items():
        if isinstance(value, Mapping):
            table_items.append((str(key), value))
        elif _is_array_of_tables(value):
            array_table_items.append((str(key), list(value)))
        else:
            scalar_items.append((str(key), value))

    lines: list[str] = []
    if prefix:
        lines.append(f"[{'.'.join(prefix)}]")
    for key, value in scalar_items:
        lines.append(f"{key} = {_format_toml_value(value)}")

    for key, value in table_items:
        if lines:
            lines.append("")
        lines.extend(_render_mapping(value, prefix=(*prefix, key)))

    for key, values in array_table_items:
        for item in values:
            if lines:
                lines.append("")
            lines.append(f"[[{'.'.join((*prefix, key))}]]")
            for item_key, item_value in item.items():
                if isinstance(item_value, Mapping) or _is_array_of_tables(item_value):
                    raise TypeError(
                        "nested tables inside TOML array-of-tables are not supported"
                    )
                lines.append(f"{item_key} = {_format_toml_value(item_value)}")

    return lines


def _is_array_of_tables(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or not value:
        return False
    return all(isinstance(item, Mapping) for item in value)


def _format_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, Path):
        return json.dumps(str(value))
    if value is None:
        raise TypeError("TOML templates do not support null values")
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_toml_value(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value type: {type(value)!r}")
