#!/usr/bin/env python3
"""Summarize completed Markov runs into fair-comparison classes."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _safe_float(value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def _round_or_none(value: object, digits: int) -> float | None:
    out = _safe_float(value)
    if not math.isfinite(out):
        return None
    return round(out, digits)


def _iter_rows(path: Path) -> Iterable[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload.get("rows", []) or []:
        if isinstance(row, dict):
            yield row


def _median(values: List[float]) -> float | None:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return None
    n = len(clean)
    mid = n // 2
    if n % 2 == 1:
        return clean[mid]
    return 0.5 * (clean[mid - 1] + clean[mid])


def _mean(values: List[float]) -> float | None:
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _row_key(row: dict, keys: List[str]) -> Tuple[object, ...]:
    return tuple(row.get(key) for key in keys)


def _build_summary(rows: List[dict]) -> dict:
    geometry_keys = ["fixed_leaf_tokens", "train_docs", "model_family"]
    local_info_keys = ["leaf_label_coverage", "internal_label_coverage", "root_query_rate"]
    total_info_keys = ["mean_queries_per_doc"]

    by_geometry: Dict[Tuple[object, ...], List[dict]] = defaultdict(list)
    by_equal_info: Dict[Tuple[object, ...], List[dict]] = defaultdict(list)
    by_total_info_proxy: Dict[Tuple[object, ...], List[dict]] = defaultdict(list)

    for row in rows:
        geom_key = _row_key(row, geometry_keys)
        by_geometry[geom_key].append(row)

        eq_key = (
            row.get("fixed_leaf_tokens"),
            row.get("train_docs"),
            _round_or_none(row.get("leaf_label_coverage"), 3),
            _round_or_none(row.get("internal_label_coverage"), 3),
            _round_or_none(row.get("root_query_rate"), 3),
        )
        by_equal_info[eq_key].append(row)

        total_key = (
            row.get("fixed_leaf_tokens"),
            row.get("train_docs"),
            _round_or_none(row.get("mean_queries_per_doc"), 3),
        )
        by_total_info_proxy[total_key].append(row)

    geometry_groups = []
    for key, group_rows in sorted(by_geometry.items()):
        geometry_groups.append(
            {
                "geometry_key": {
                    "fixed_leaf_tokens": key[0],
                    "train_docs": key[1],
                    "model_family": key[2],
                },
                "n_rows": len(group_rows),
                "info_signatures": sorted({str(r.get("info_signature")) for r in group_rows}),
                "leaf_label_coverage_values": sorted(
                    {round(v, 3) for v in (_safe_float(r.get("leaf_label_coverage")) for r in group_rows) if math.isfinite(v)}
                ),
                "internal_label_coverage_values": sorted(
                    {round(v, 3) for v in (_safe_float(r.get("internal_label_coverage")) for r in group_rows) if math.isfinite(v)}
                ),
                "root_query_rate_values": sorted(
                    {round(v, 3) for v in (_safe_float(r.get("root_query_rate")) for r in group_rows) if math.isfinite(v)}
                ),
                "median_root_utility_recovery": _median(
                    [_safe_float(r.get("root_utility_recovery")) for r in group_rows]
                ),
                "median_root_mae": _median([_safe_float(r.get("learned_root_mae")) for r in group_rows]),
                "median_merge_mae": _median([_safe_float(r.get("learned_merge_mae")) for r in group_rows]),
            }
        )

    equal_info_groups = []
    for key, group_rows in sorted(by_equal_info.items()):
        equal_info_groups.append(
            {
                "equal_information_key": {
                    "fixed_leaf_tokens": key[0],
                    "train_docs": key[1],
                    "leaf_label_coverage": key[2],
                    "internal_label_coverage": key[3],
                    "root_query_rate": key[4],
                },
                "n_rows": len(group_rows),
                "model_families_present": sorted({str(r.get("model_family")) for r in group_rows}),
                "seeds_present": sorted({int(r.get("seed")) for r in group_rows if str(r.get("seed")).isdigit()}),
                "median_root_utility_recovery": _median(
                    [_safe_float(r.get("root_utility_recovery")) for r in group_rows]
                ),
                "mean_root_mae": _mean([_safe_float(r.get("learned_root_mae")) for r in group_rows]),
                "mean_merge_mae": _mean([_safe_float(r.get("learned_merge_mae")) for r in group_rows]),
                "rows": [str(r.get("source_path")) for r in group_rows],
            }
        )

    total_info_proxy_groups = []
    for key, group_rows in sorted(by_total_info_proxy.items()):
        total_info_proxy_groups.append(
            {
                "total_information_proxy_key": {
                    "fixed_leaf_tokens": key[0],
                    "train_docs": key[1],
                    "mean_queries_per_doc": key[2],
                },
                "n_rows": len(group_rows),
                "info_signatures": sorted({str(r.get("info_signature")) for r in group_rows}),
                "warning": (
                    "These rows have similar total query counts, but may still be unfair to compare if the "
                    "leaf/internal/root allocation differs."
                ),
                "rows": [str(r.get("source_path")) for r in group_rows],
            }
        )

    return {
        "n_rows": len(rows),
        "grouping_policy": {
            "geometry_keys": geometry_keys,
            "equal_information_keys": [
                "fixed_leaf_tokens",
                "train_docs",
                "leaf_label_coverage",
                "internal_label_coverage",
                "root_query_rate",
            ],
            "total_information_proxy_keys": total_info_keys,
        },
        "geometry_groups": geometry_groups,
        "equal_information_groups": equal_info_groups,
        "total_information_proxy_groups": total_info_proxy_groups,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize fair comparison classes for completed Markov rows.")
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rows = list(_iter_rows(args.summary_json))
    report = _build_summary(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Markov Comparison Classes",
        "",
        f"- Completed rows: `{report['n_rows']}`",
        "",
        "## Grouping Policy",
        "",
        f"- Geometry keys: `{report['grouping_policy']['geometry_keys']}`",
        f"- Equal-information keys: `{report['grouping_policy']['equal_information_keys']}`",
        f"- Total-information proxy keys: `{report['grouping_policy']['total_information_proxy_keys']}`",
        "",
        "## Geometry Groups",
        "",
    ]
    for item in report["geometry_groups"]:
        lines.append(f"- geometry={item['geometry_key']} | n_rows={item['n_rows']}")
        lines.append(f"  info_signatures={item['info_signatures']}")
        lines.append(
            f"  median_root_utility_recovery={item['median_root_utility_recovery']} | "
            f"median_root_mae={item['median_root_mae']} | median_merge_mae={item['median_merge_mae']}"
        )
    lines.extend(["", "## Equal-Information Groups", ""])
    for item in report["equal_information_groups"]:
        lines.append(f"- equal_information={item['equal_information_key']} | n_rows={item['n_rows']}")
        lines.append(f"  model_families_present={item['model_families_present']}")
        lines.append(
            f"  median_root_utility_recovery={item['median_root_utility_recovery']} | "
            f"mean_root_mae={item['mean_root_mae']} | mean_merge_mae={item['mean_merge_mae']}"
        )
    lines.extend(["", "## Total-Information Proxy Groups", ""])
    for item in report["total_information_proxy_groups"]:
        lines.append(f"- total_information_proxy={item['total_information_proxy_key']} | n_rows={item['n_rows']}")
        lines.append(f"  info_signatures={item['info_signatures']}")
        lines.append(f"  warning={item['warning']}")
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "summary_json": str(args.summary_json),
                "output_json": str(args.output_json),
                "output_markdown": str(args.output_markdown),
                "n_rows": report["n_rows"],
                "n_geometry_groups": len(report["geometry_groups"]),
                "n_equal_information_groups": len(report["equal_information_groups"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
