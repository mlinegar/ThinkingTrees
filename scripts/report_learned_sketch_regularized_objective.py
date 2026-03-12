#!/usr/bin/env python3
"""Post-hoc frontier report for the learned-sketch regularized objective."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _parse_float_csv(s: str) -> Tuple[float, ...]:
    out = tuple(float(x.strip()) for x in s.split(",") if x.strip())
    if len(out) == 0:
        raise ValueError("expected a non-empty float CSV")
    return out


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return float(default)
    if text.lower() in {"true", "false"}:
        return 1.0 if text.lower() == "true" else 0.0
    return float(text)


def _safe_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return bool(default)


def _normalized_law_shares(
    *,
    leaf_share: float,
    merge_share: float,
    idemp_share: float,
) -> Tuple[float, float, float]:
    total = max(0.0, float(leaf_share)) + max(0.0, float(merge_share)) + max(
        0.0, float(idemp_share)
    )
    if total <= 0.0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (
        max(0.0, float(leaf_share)) / total,
        max(0.0, float(merge_share)) / total,
        max(0.0, float(idemp_share)) / total,
    )


def _load_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve_json_rows(payload: dict, table: str) -> Tuple[List[dict], str]:
    candidates: List[Tuple[str, List[dict]]] = []
    if isinstance(payload.get("raw_rows"), list):
        candidates.append(("raw_rows", payload["raw_rows"]))
    if isinstance(payload.get("rows"), list):
        candidates.append(("rows", payload["rows"]))
    if isinstance(payload.get("aggregated_rows"), list):
        candidates.append(("aggregated_rows", payload["aggregated_rows"]))

    if table == "auto":
        if len(candidates) == 0:
            raise ValueError("JSON payload does not contain rows/raw_rows/aggregated_rows")
        return candidates[0][1], candidates[0][0]

    for name, rows in candidates:
        if name == table:
            return rows, name
    raise ValueError(f"requested table {table!r} not found in JSON payload")


def _load_rows(path: Path, table: str) -> Tuple[List[dict], str, dict]:
    if path.suffix.lower() == ".csv":
        return _load_csv_rows(path), "csv", {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object at top level")
    rows, source_table = _resolve_json_rows(payload, table)
    config = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
    return rows, source_table, config


def _row_has_mean_suffix(row: dict, metric: str) -> bool:
    return f"{metric}_mean" in row


def _metric_value(row: dict, metric: str, *, aggregated: bool, default: float = 0.0) -> float:
    key = f"{metric}_mean" if aggregated else metric
    return _safe_float(row.get(key), default=default)


def _row_value(
    row: dict,
    key: str,
    *,
    config: dict,
    default: float,
) -> float:
    if key in row:
        return _safe_float(row[key], default=default)
    if key in config:
        return _safe_float(config[key], default=default)
    return float(default)


def _evaluate_row(
    row: dict,
    *,
    aggregated: bool,
    config: dict,
    regularizer_weight: float,
    summary_share_override: float | None,
    law_strength_override: float | None,
    law_leaf_share_override: float | None,
    law_merge_share_override: float | None,
    law_idemp_share_override: float | None,
) -> dict:
    if summary_share_override is not None and law_strength_override is not None:
        raise ValueError("summary_share_override and law_strength_override cannot both be set")
    global_error = _metric_value(
        row, "regularized_objective_global_error", aggregated=aggregated
    )
    summary_budget_penalty = _metric_value(
        row, "regularized_objective_summary_budget_penalty", aggregated=aggregated
    )
    law_penalty = _metric_value(row, "regularized_objective_law_penalty", aggregated=aggregated)
    law_scale = _metric_value(
        row, "regularized_objective_law_scale", aggregated=aggregated, default=1.0
    )
    proxy_fraction = (
        _safe_float(row.get("regularized_objective_proxy_fraction"), default=0.0)
        if aggregated
        else (1.0 if _safe_bool(row.get("regularized_objective_uses_proxy_law_penalty")) else 0.0)
    )

    if law_strength_override is not None:
        if not (0.0 <= float(law_strength_override) <= 1.0):
            raise ValueError("law_strength_override must be in [0, 1]")
        summary_share = float(1.0 - float(law_strength_override))
    else:
        summary_share = (
            float(summary_share_override)
            if summary_share_override is not None
            else _row_value(
                row,
                "regularized_objective_summary_share",
                config=config,
                default=0.5,
            )
        )
    summary_share = max(0.0, min(1.0, float(summary_share)))
    law_strength = float(1.0 - summary_share)

    leaf_share = _row_value(
        row,
        "regularized_objective_leaf_share",
        config=config,
        default=1.0 / 3.0,
    )
    merge_share = _row_value(
        row,
        "regularized_objective_merge_share",
        config=config,
        default=1.0 / 3.0,
    )
    idemp_share = _row_value(
        row,
        "regularized_objective_idemp_share",
        config=config,
        default=1.0 / 3.0,
    )
    if (
        law_leaf_share_override is not None
        or law_merge_share_override is not None
        or law_idemp_share_override is not None
    ):
        leaf_share = float(law_leaf_share_override if law_leaf_share_override is not None else leaf_share)
        merge_share = float(
            law_merge_share_override if law_merge_share_override is not None else merge_share
        )
        idemp_share = float(
            law_idemp_share_override if law_idemp_share_override is not None else idemp_share
        )
    leaf_share, merge_share, idemp_share = _normalized_law_shares(
        leaf_share=leaf_share,
        merge_share=merge_share,
        idemp_share=idemp_share,
    )

    law_penalty_recomputed = False
    if proxy_fraction <= 0.0:
        eps_leaf = _metric_value(row, "eps_leaf", aggregated=aggregated)
        eps_merge = _metric_value(row, "eps_merge", aggregated=aggregated)
        eps_idemp = _metric_value(row, "eps_idemp", aggregated=aggregated)
        if (
            law_leaf_share_override is not None
            or law_merge_share_override is not None
            or law_idemp_share_override is not None
        ):
            law_scale = max(1.0, float(law_scale))
            law_penalty = (
                leaf_share * max(0.0, eps_leaf) / law_scale
                + merge_share * max(0.0, eps_merge) / law_scale
                + idemp_share * max(0.0, eps_idemp) / law_scale
            )
            law_penalty_recomputed = True

    combined_regularizer = summary_share * summary_budget_penalty + (1.0 - summary_share) * law_penalty
    total = (1.0 - float(regularizer_weight)) * global_error + float(regularizer_weight) * combined_regularizer

    return {
        "state_dim": int(_safe_float(row.get("state_dim"), default=0.0)),
        "train_size": int(
            _safe_float(row.get("train_size", row.get("train_docs")), default=0.0)
        ),
        "seed": None if "seed" not in row else int(_safe_float(row.get("seed"), default=0.0)),
        "regularizer_weight": float(regularizer_weight),
        "objective_total": float(total),
        "global_error": float(global_error),
        "summary_budget_penalty": float(summary_budget_penalty),
        "law_penalty": float(law_penalty),
        "combined_regularizer": float(combined_regularizer),
        "learned_memory_bits": int(_safe_float(row.get("learned_memory_bits"), default=0.0)),
        "train_total_queries_estimate": int(
            _safe_float(row.get("train_total_queries_estimate"), default=0.0)
        ),
        "summary_share": float(summary_share),
        "law_strength": float(law_strength),
        "leaf_share": float(leaf_share),
        "merge_share": float(merge_share),
        "idemp_share": float(idemp_share),
        "law_scale": float(max(1.0, law_scale)),
        "proxy_fraction": float(proxy_fraction),
        "law_penalty_recomputed": bool(law_penalty_recomputed),
    }


def _aggregate_rows(rows: Sequence[dict]) -> List[dict]:
    grouped: Dict[Tuple[float, float, int, int], List[dict]] = {}
    for row in rows:
        key = (
            float(row["regularizer_weight"]),
            float(row["law_strength"]),
            int(row["state_dim"]),
            int(row["train_size"]),
        )
        grouped.setdefault(key, []).append(row)

    out: List[dict] = []
    for (regularizer_weight, law_strength, state_dim, train_size), grows in sorted(grouped.items()):
        objective_vals = [float(x["objective_total"]) for x in grows]
        global_vals = [float(x["global_error"]) for x in grows]
        regularizer_vals = [float(x["combined_regularizer"]) for x in grows]
        summary_vals = [float(x["summary_budget_penalty"]) for x in grows]
        law_vals = [float(x["law_penalty"]) for x in grows]
        query_vals = [float(x["train_total_queries_estimate"]) for x in grows]
        agg = {
            "regularizer_weight": float(regularizer_weight),
            "law_strength": float(law_strength),
            "state_dim": int(state_dim),
            "train_size": int(train_size),
            "n_rows": int(len(grows)),
            "objective_total_mean": float(statistics.mean(objective_vals)),
            "objective_total_std": float(statistics.pstdev(objective_vals))
            if len(objective_vals) > 1
            else 0.0,
            "objective_total_min": float(min(objective_vals)),
            "objective_total_max": float(max(objective_vals)),
            "global_error_mean": float(statistics.mean(global_vals)),
            "combined_regularizer_mean": float(statistics.mean(regularizer_vals)),
            "summary_budget_penalty_mean": float(statistics.mean(summary_vals)),
            "law_penalty_mean": float(statistics.mean(law_vals)),
            "train_total_queries_estimate_mean": float(statistics.mean(query_vals)),
            "learned_memory_bits": int(grows[0]["learned_memory_bits"]),
            "summary_share": float(statistics.mean([float(x["summary_share"]) for x in grows])),
            "leaf_share": float(statistics.mean([float(x["leaf_share"]) for x in grows])),
            "merge_share": float(statistics.mean([float(x["merge_share"]) for x in grows])),
            "idemp_share": float(statistics.mean([float(x["idemp_share"]) for x in grows])),
            "law_scale_mean": float(statistics.mean([float(x["law_scale"]) for x in grows])),
            "proxy_fraction_mean": float(
                statistics.mean([float(x["proxy_fraction"]) for x in grows])
            ),
            "law_penalty_recomputed_any": bool(
                any(bool(x["law_penalty_recomputed"]) for x in grows)
            ),
        }
        out.append(agg)
    return out


def _best_overall(rows: Sequence[dict]) -> List[dict]:
    out: List[dict] = []
    weights = sorted({float(r["regularizer_weight"]) for r in rows})
    law_strengths = sorted({float(r["law_strength"]) for r in rows})
    for regularizer_weight in weights:
        for law_strength in law_strengths:
            candidates = [
                r
                for r in rows
                if float(r["regularizer_weight"]) == regularizer_weight
                and float(r["law_strength"]) == law_strength
            ]
            if len(candidates) == 0:
                continue
            best = min(
                candidates,
                key=lambda x: (
                    float(x["objective_total_mean"]),
                    float(x["global_error_mean"]),
                    float(x["combined_regularizer_mean"]),
                    int(x["state_dim"]),
                    int(x["train_size"]),
                ),
            )
            out.append(best)
    return out


def _best_by_train_size(rows: Sequence[dict]) -> List[dict]:
    out: List[dict] = []
    weights = sorted({float(r["regularizer_weight"]) for r in rows})
    law_strengths = sorted({float(r["law_strength"]) for r in rows})
    train_sizes = sorted({int(r["train_size"]) for r in rows})
    for regularizer_weight in weights:
        for law_strength in law_strengths:
            for train_size in train_sizes:
                candidates = [
                    r
                    for r in rows
                    if float(r["regularizer_weight"]) == regularizer_weight
                    and float(r["law_strength"]) == law_strength
                    and int(r["train_size"]) == train_size
                ]
                if len(candidates) == 0:
                    continue
                best = min(
                    candidates,
                    key=lambda x: (
                        float(x["objective_total_mean"]),
                        float(x["global_error_mean"]),
                        float(x["combined_regularizer_mean"]),
                        int(x["state_dim"]),
                    ),
                )
                out.append(best)
    return out


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if len(rows) == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute the learned-sketch regularized-objective frontier from a saved "
            "simulation or sampling-sweep artifact."
        )
    )
    parser.add_argument("--input", required=True, help="JSON or CSV artifact path.")
    parser.add_argument(
        "--table",
        choices=["auto", "rows", "raw_rows", "aggregated_rows"],
        default="auto",
        help="Which JSON table to read. Ignored for CSV inputs.",
    )
    parser.add_argument(
        "--regularizer-weights",
        type=str,
        default="0.0,0.25,0.5,0.75,1.0",
        help="CSV lambda values for post-hoc frontier evaluation.",
    )
    parser.add_argument(
        "--summary-share",
        type=float,
        default=None,
        help="Optional override for the summary share inside the regularizer.",
    )
    parser.add_argument(
        "--law-strengths",
        type=str,
        default=None,
        help=(
            "Optional CSV law-strength grid. law_strength = 0 is the legacy "
            "summary-only endpoint; law_strength = 1 is the law-only endpoint."
        ),
    )
    parser.add_argument(
        "--law-leaf-share",
        type=float,
        default=None,
        help="Optional override for the leaf-law share inside the law penalty.",
    )
    parser.add_argument(
        "--law-merge-share",
        type=float,
        default=None,
        help="Optional override for the merge-law share inside the law penalty.",
    )
    parser.add_argument(
        "--law-idemp-share",
        type=float,
        default=None,
        help="Optional override for the idempotence-law share inside the law penalty.",
    )
    parser.add_argument(
        "--json-summary",
        type=str,
        default=None,
        help="Output JSON path. Default: <input stem>_regularized_objective.json",
    )
    parser.add_argument(
        "--csv-summary",
        type=str,
        default=None,
        help="Output CSV path. Default: <input stem>_regularized_objective.csv",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the JSON summary to stdout as well.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input artifact not found: {input_path}")

    regularizer_weights = tuple(
        max(0.0, min(1.0, float(x))) for x in _parse_float_csv(args.regularizer_weights)
    )
    if args.summary_share is not None and args.law_strengths is not None:
        raise ValueError("use either --summary-share or --law-strengths, not both")
    law_strength_grid: Tuple[float | None, ...]
    if args.law_strengths is not None:
        law_strength_grid = tuple(
            max(0.0, min(1.0, float(x))) for x in _parse_float_csv(args.law_strengths)
        )
    else:
        law_strength_grid = (None,)
    rows, source_table, config = _load_rows(input_path, args.table)
    if len(rows) == 0:
        raise ValueError("input artifact contained no rows")

    aggregated = _row_has_mean_suffix(rows[0], "regularized_objective_global_error")
    evaluated_rows: List[dict] = []
    for row in rows:
        for regularizer_weight in regularizer_weights:
            for law_strength_override in law_strength_grid:
                evaluated_rows.append(
                    _evaluate_row(
                        row,
                        aggregated=aggregated,
                        config=config,
                        regularizer_weight=regularizer_weight,
                        summary_share_override=args.summary_share,
                        law_strength_override=law_strength_override,
                        law_leaf_share_override=args.law_leaf_share,
                        law_merge_share_override=args.law_merge_share,
                        law_idemp_share_override=args.law_idemp_share,
                    )
                )

    frontier_rows = _aggregate_rows(evaluated_rows)
    best_overall = _best_overall(frontier_rows)
    best_by_train_size = _best_by_train_size(frontier_rows)

    json_path = (
        Path(args.json_summary)
        if args.json_summary is not None
        else input_path.with_name(f"{input_path.stem}_regularized_objective.json")
    )
    csv_path = (
        Path(args.csv_summary)
        if args.csv_summary is not None
        else input_path.with_name(f"{input_path.stem}_regularized_objective.csv")
    )

    payload = {
        "input_path": str(input_path),
        "source_table": str(source_table),
        "source_rows_are_aggregated": bool(aggregated),
        "n_source_rows": int(len(rows)),
        "n_frontier_rows": int(len(frontier_rows)),
        "regularizer_weights": [float(x) for x in regularizer_weights],
        "summary_share_override": args.summary_share,
        "law_strengths": None
        if args.law_strengths is None
        else [float(x) for x in law_strength_grid if x is not None],
        "law_share_override": {
            "leaf": args.law_leaf_share,
            "merge": args.law_merge_share,
            "idemp": args.law_idemp_share,
        },
        "config": config,
        "best_overall": best_overall,
        "best_by_train_size": best_by_train_size,
        "frontier_rows": frontier_rows,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(csv_path, frontier_rows)

    if args.json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")

    print(
        "lambda | law_strength | best_state_dim | best_train_size | objective_mean | "
        "global_error_mean | regularizer_mean"
    )
    for row in best_overall:
        print(
            f"{float(row['regularizer_weight']):.2f} | "
            f"{float(row['law_strength']):.2f} | "
            f"{int(row['state_dim'])} | {int(row['train_size'])} | "
            f"{float(row['objective_total_mean']):.5f} | "
            f"{float(row['global_error_mean']):.5f} | "
            f"{float(row['combined_regularizer_mean']):.5f}"
        )

    print(f"wrote_json | {json_path}")
    print(f"wrote_csv | {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
