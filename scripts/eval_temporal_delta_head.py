#!/usr/bin/env python3
"""
Evaluate temporal delta-RILE head against a delta=0 baseline.

Consumes `predictions.csv` from `scripts/train_rile_embedding_sketch.py` and
reports MAE on eligible temporal samples only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    rendered = str(value or "").strip().lower()
    return rendered in {"1", "true", "yes", "y", "on"}


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate temporal delta head against delta=0 baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions", type=Path, required=True, help="Path to predictions.csv")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--min-improvement-frac", type=float, default=0.05)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    pred_path = args.predictions
    if not pred_path.is_absolute():
        pred_path = (Path.cwd() / pred_path).resolve()
    if not pred_path.exists():
        raise SystemExit(f"Predictions file not found: {pred_path}")

    rows: List[Dict[str, Any]] = []
    with pred_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))

    selected = []
    for row in rows:
        if str(args.split) != "all" and str(row.get("split", "")).strip().lower() != str(args.split):
            continue
        if not _as_bool(row.get("has_delta_target")):
            continue
        true_delta = _as_float(row.get("true_delta_rile"))
        pred_delta = _as_float(row.get("pred_delta_rile"))
        if not np.isfinite(true_delta) or not np.isfinite(pred_delta):
            continue
        selected.append((true_delta, pred_delta))

    if not selected:
        result = {
            "eligible_samples": 0,
            "status": "NO_ELIGIBLE_SAMPLES",
            "pass": False,
        }
        if args.json_out:
            out_path = args.json_out if args.json_out.is_absolute() else (Path.cwd() / args.json_out).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("No eligible temporal samples found.")
        return 2

    truth = np.asarray([item[0] for item in selected], dtype=np.float32)
    pred = np.asarray([item[1] for item in selected], dtype=np.float32)
    baseline = np.zeros_like(truth)

    mae_model = float(np.mean(np.abs(pred - truth)))
    mae_zero = float(np.mean(np.abs(baseline - truth)))
    improvement = float((mae_zero - mae_model) / max(mae_zero, 1e-12))
    passed = bool(improvement >= float(args.min_improvement_frac))

    result = {
        "eligible_samples": int(truth.size),
        "mae_delta_head": mae_model,
        "mae_delta_zero": mae_zero,
        "relative_improvement": improvement,
        "min_required_improvement": float(args.min_improvement_frac),
        "pass": passed,
    }

    print("")
    print("=" * 70)
    print("Temporal Delta-Head Diagnostic")
    print("=" * 70)
    print(f"Eligible samples:      {result['eligible_samples']}")
    print(f"Delta-head MAE:        {mae_model:.6f}")
    print(f"Zero baseline MAE:     {mae_zero:.6f}")
    print(f"Relative improvement:  {improvement:+.2%}")
    print(f"Gate (>= {float(args.min_improvement_frac):.0%}): {'PASS' if passed else 'FAIL'}")
    print("=" * 70)

    if args.json_out:
        out_path = args.json_out if args.json_out.is_absolute() else (Path.cwd() / args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
