#!/usr/bin/env python3
"""Ceiling-focused plots for the OPS Markov changepoint-count simulation.

This script is intentionally "paper-figure shaped":

Panel A (ceiling + floor):
  - learned root MAE vs budget/labels
  - exact baseline (ceiling; should be ~0)
  - undersupported baseline (approximation-bias floor)

Panel B (merge robustness diagnostic):
  - schedule-spread vs root error for the learned sketch (colored by budget)

Inputs are per-run JSON outputs from:
  `scripts/run_markov_changepoint_ops_count_simulation.py`
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
import statistics
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot ceiling/floor + schedule robustness for Markov OPS-count sims.")
    p.add_argument(
        "--input-glob",
        type=str,
        default="outputs/markov_changepoint_ops_count/**/*seed_*.json",
        help="Glob for per-run JSON outputs.",
    )
    p.add_argument(
        "--x-axis",
        choices=[
            "train_docs",
            "total_queries_estimate",
            "total_queries",
            "internal_labels_total",
            "leaf_labels_total",
            "root_queries_total",
        ],
        default="total_queries_estimate",
        help="X axis for the ceiling panel.",
    )
    p.add_argument(
        "--budget-axis",
        choices=["audit_fraction", "internal_per_leaf", "leaf_query_rate"],
        default="audit_fraction",
        help="Budget axis used for coloring/stratifying curves.",
    )
    p.add_argument(
        "--budgets",
        type=str,
        default="",
        help="Optional comma/space-separated list of budget values to include (exact match within tolerance).",
    )
    p.add_argument(
        "--aggregate",
        choices=["median", "mean"],
        default="median",
        help="How to aggregate across seeds per (x, budget) point.",
    )
    p.add_argument(
        "--band",
        choices=["none", "p10_p90", "p25_p75"],
        default="p10_p90",
        help="Optional quantile band across seeds for the learned curve.",
    )
    p.add_argument("--normalize", action="store_true", help="Normalize error/spread by (max_segments-1).")
    p.add_argument("--log-x", action="store_true")
    p.add_argument(
        "--feature-mode",
        action="append",
        default=[],
        help="Filter to specific feature_mode values (repeatable). Default: include all.",
    )
    p.add_argument(
        "--c3-audit-strategy",
        action="append",
        default=[],
        help="Filter to specific c3_audit_strategy values (repeatable). Default: include all.",
    )
    p.add_argument(
        "--model-family",
        action="append",
        default=[],
        help="Filter to specific model_family values (repeatable). Default: include all.",
    )
    p.add_argument(
        "--leaf-query-rates",
        type=str,
        default="",
        help="Optional comma/space list filter on leaf_query_rate (exact match within tolerance).",
    )
    p.add_argument(
        "--include-root-query",
        type=str,
        choices=["any", "true", "false"],
        default="any",
        help="Optional filter on include_root_query flag in config.",
    )
    p.add_argument(
        "--output-figure",
        type=str,
        default="outputs/markov_changepoint_ops_count_ceilings.png",
        help="Output PNG figure path.",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default="outputs/markov_changepoint_ops_count_ceilings_report.json",
        help="Output JSON report path.",
    )
    return p.parse_args()


def _parse_floats(text: str) -> List[float]:
    out: List[float] = []
    for raw in str(text).replace(",", " ").split():
        if not raw.strip():
            continue
        out.append(float(raw.strip()))
    return out


def _reduce(vals: List[float], *, agg: str) -> float:
    vals2 = [float(x) for x in vals if np.isfinite(float(x))]
    if not vals2:
        return float("nan")
    if agg == "mean":
        return float(np.mean(np.asarray(vals2, dtype=np.float64)))
    if agg == "median":
        return float(statistics.median(vals2))
    raise ValueError(f"unsupported aggregate: {agg!r}")


def _percentile(vals: List[float], q: float) -> float:
    vals2 = np.asarray([float(x) for x in vals if np.isfinite(float(x))], dtype=np.float64)
    if vals2.size == 0:
        return float("nan")
    return float(np.percentile(vals2, q))


def _band_quantiles(kind: str) -> Optional[Tuple[float, float]]:
    if kind == "none":
        return None
    if kind == "p10_p90":
        return (10.0, 90.0)
    if kind == "p25_p75":
        return (25.0, 75.0)
    raise ValueError(f"unsupported band: {kind!r}")


def _matches_any(value: str, allowed: List[str]) -> bool:
    if not allowed:
        return True
    return str(value) in {str(x) for x in allowed}


def _float_in_set(x: float, targets: List[float], *, tol: float = 1e-12) -> bool:
    if not targets:
        return True
    return any(abs(float(x) - float(t)) <= tol for t in targets)


def _collect_rows(files: Iterable[Path]) -> List[dict]:
    rows: List[dict] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {}) or {}
        geom = payload.get("training_geometry", {}) or {}
        metrics = payload.get("metrics", {}) or {}
        if not isinstance(metrics, dict):
            continue

        max_segments = int(cfg.get("max_segments", -1))
        count_scale = float(max(1, max_segments - 1)) if max_segments > 0 else 1.0

        mean_leaves = float(geom.get("mean_leaves", float("nan")))
        mean_internal = float(geom.get("mean_internal_labels", float("nan")))
        internal_per_leaf = (
            float(mean_internal) / float(mean_leaves)
            if np.isfinite(mean_internal) and np.isfinite(mean_leaves) and mean_leaves > 0
            else float("nan")
        )

        row = {
            "path": str(path),
            "seed": int(cfg.get("seed", -1)),
            "train_docs": int(cfg.get("train_docs", -1)),
            "audit_fraction": float(cfg.get("audit_fraction", float("nan"))),
            "internal_per_leaf": float(internal_per_leaf),
            "leaf_query_rate": float(cfg.get("leaf_query_rate", float("nan"))),
            "include_root_query": bool(cfg.get("include_root_query", True)),
            "feature_mode": str(cfg.get("feature_mode", "")),
            "c3_audit_strategy": str(cfg.get("c3_audit_strategy", "")),
            "model_family": str(cfg.get("model_family", "neural")),
            "total_queries_estimate": int(geom.get("total_queries_estimate", -1)),
            # Backward-compatible alias for older callers.
            "total_queries": int(geom.get("total_queries_estimate", -1)),
            "internal_labels_total": int(geom.get("internal_labels_total", -1)),
            "leaf_labels_total": int(geom.get("leaf_labels_total", -1)),
            "root_queries_total": int(geom.get("root_queries_total", -1)),
            "count_scale": float(count_scale),
            "metrics": {},
        }

        for sketch in ("learned", "exact", "undersupported"):
            block = metrics.get(sketch, {})
            if not isinstance(block, dict):
                continue
            row["metrics"][sketch] = {
                "root_mae": float(block.get("root_mae", float("nan"))),
                "schedule_spread_mean": float(block.get("schedule_spread_mean", float("nan"))),
                "merge_mae": float(block.get("merge_mae", float("nan"))),
            }
        if "learned" not in row["metrics"]:
            continue
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    files = [Path(p) for p in sorted(glob.glob(str(args.input_glob), recursive=True))]
    if not files:
        raise ValueError(f"no files matched: {args.input_glob}")

    rows = _collect_rows(files)
    rows = [r for r in rows if _matches_any(r["feature_mode"], list(args.feature_mode))]
    rows = [r for r in rows if _matches_any(r["c3_audit_strategy"], list(args.c3_audit_strategy))]
    rows = [r for r in rows if _matches_any(r["model_family"], list(args.model_family))]
    leaf_q_targets = _parse_floats(str(args.leaf_query_rates))
    if leaf_q_targets:
        rows = [r for r in rows if _float_in_set(float(r.get("leaf_query_rate", float("nan"))), leaf_q_targets)]
    if str(args.include_root_query) != "any":
        want = str(args.include_root_query) == "true"
        rows = [r for r in rows if bool(r.get("include_root_query", True)) is bool(want)]
    rows_pre_budget = list(rows)

    budget_targets = _parse_floats(args.budgets)
    if budget_targets:
        rows = [r for r in rows if _float_in_set(float(r[str(args.budget_axis)]), budget_targets)]
    if not rows:
        raise ValueError("no rows after filters")

    # Normalization for count-scale comparability.
    def _scale(v: float, *, r: dict) -> float:
        if not bool(args.normalize):
            return float(v)
        s = float(r.get("count_scale", 1.0))
        return float(v) / float(s) if np.isfinite(s) and s > 0 else float(v)

    # Compute global baseline lines (median across all filtered runs).
    baseline: Dict[str, Dict[str, float]] = {}
    for sketch in ("exact", "undersupported"):
        vals_root = [_scale(float(r["metrics"].get(sketch, {}).get("root_mae", float("nan"))), r=r) for r in rows]
        vals_spread = [
            _scale(float(r["metrics"].get(sketch, {}).get("schedule_spread_mean", float("nan"))), r=r) for r in rows
        ]
        baseline[sketch] = {
            "root_mae": _reduce(vals_root, agg=str(args.aggregate)),
            "schedule_spread_mean": _reduce(vals_spread, agg=str(args.aggregate)),
        }

    # Diagnostic: do we have explicit full-audit points?
    full_audit_rows = [
        r
        for r in rows_pre_budget
        if np.isfinite(float(r.get("audit_fraction", float("nan"))))
        and abs(float(r["audit_fraction"]) - 1.0) <= 1e-12
    ]
    full_audit_diagnostic = {
        "present": bool(full_audit_rows),
        "n_rows": int(len(full_audit_rows)),
        "learned_root_mae": _reduce(
            [_scale(float(r["metrics"]["learned"]["root_mae"]), r=r) for r in full_audit_rows],
            agg=str(args.aggregate),
        ),
        "exact_root_mae": _reduce(
            [_scale(float(r["metrics"].get("exact", {}).get("root_mae", float("nan"))), r=r) for r in full_audit_rows],
            agg=str(args.aggregate),
        ),
        "undersupported_root_mae": _reduce(
            [
                _scale(float(r["metrics"].get("undersupported", {}).get("root_mae", float("nan"))), r=r)
                for r in full_audit_rows
            ],
            agg=str(args.aggregate),
        ),
    }

    # Build learned curves: group by budget, then by x.
    grouped: Dict[float, Dict[float, List[float]]] = {}
    grouped_spread: Dict[float, Dict[float, List[float]]] = {}
    scatter_points: List[dict] = []
    for r in rows:
        budget = float(r[str(args.budget_axis)])
        x = float(r[str(args.x_axis)])
        y = _scale(float(r["metrics"]["learned"]["root_mae"]), r=r)
        s = _scale(float(r["metrics"]["learned"]["schedule_spread_mean"]), r=r)
        if not (np.isfinite(budget) and np.isfinite(x) and np.isfinite(y) and np.isfinite(s)):
            continue
        grouped.setdefault(budget, {}).setdefault(x, []).append(y)
        grouped_spread.setdefault(budget, {}).setdefault(x, []).append(s)
        scatter_points.append({"budget": budget, "x": x, "root_mae": y, "spread": s})

    if not grouped:
        raise ValueError("no finite learned points for plotting")

    budgets_sorted = sorted(grouped.keys())
    if budget_targets:
        # Preserve user order if given.
        budgets_sorted = [b for b in budget_targets if any(abs(b - bb) <= 1e-12 for bb in budgets_sorted)]

    qband = _band_quantiles(str(args.band))

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.1), constrained_layout=True)
    ax0, ax1 = axes

    # Panel A: ceiling/floor lines + learned curves.
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

    budget_color: Dict[float, str] = {}
    for idx, b in enumerate(budgets_sorted):
        budget_color[b] = colors[idx % len(colors)]

    for b in budgets_sorted:
        by_x = grouped[b]
        xs = sorted(by_x.keys())
        ys = [_reduce(by_x[x], agg=str(args.aggregate)) for x in xs]
        ax0.plot(
            xs,
            ys,
            marker="o",
            linewidth=1.8,
            color=budget_color[b],
            label=f"q_train({args.budget_axis})={b:g}",
        )
        if qband is not None:
            lo = [_percentile(by_x[x], qband[0]) for x in xs]
            hi = [_percentile(by_x[x], qband[1]) for x in xs]
            lo_arr = np.asarray(lo, dtype=np.float64)
            hi_arr = np.asarray(hi, dtype=np.float64)
            ok = np.isfinite(lo_arr) & np.isfinite(hi_arr)
            if np.any(ok):
                ax0.fill_between(
                    np.asarray(xs, dtype=np.float64)[ok],
                    lo_arr[ok],
                    hi_arr[ok],
                    color=budget_color[b],
                    alpha=0.14,
                    linewidth=0,
                )

    ax0.axhline(baseline["exact"]["root_mae"], color="#222222", linestyle=":", linewidth=2.0, label="exact (ceiling)")
    ax0.axhline(
        baseline["undersupported"]["root_mae"],
        color="#444444",
        linestyle="--",
        linewidth=2.0,
        label="undersupported (bias floor)",
    )
    x_labels = {
        "train_docs": "train_docs",
        "total_queries_estimate": "total_queries_estimate",
        "total_queries": "total_queries_estimate",
        "internal_labels_total": "internal_labels_total",
        "leaf_labels_total": "leaf_labels_total",
        "root_queries_total": "root_queries_total",
    }
    ax0.set_xlabel(str(x_labels.get(str(args.x_axis), str(args.x_axis))))
    ax0.set_ylabel("Root MAE" + (" / (max_segments-1)" if bool(args.normalize) else ""))
    ax0.set_title("Ceiling vs Floor: Root Error")
    if not bool(full_audit_diagnostic["present"]):
        ax0.text(
            0.02,
            0.98,
            "No audit_fraction=1.0 runs in input",
            transform=ax0.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#b22222",
        )
    if bool(args.log_x):
        ax0.set_xscale("log")
    ax0.grid(alpha=0.25)
    ax0.legend(frameon=False, fontsize=9)

    # Panel B: schedule spread vs root error for learned.
    for b in budgets_sorted:
        pts = [p for p in scatter_points if abs(float(p["budget"]) - b) <= 1e-12]
        if not pts:
            continue
        ax1.scatter(
            [p["root_mae"] for p in pts],
            [p["spread"] for p in pts],
            s=26,
            alpha=0.50,
            color=budget_color[b],
            label=f"q_train({args.budget_axis})={b:g}",
        )
    ax1.scatter(
        [baseline["exact"]["root_mae"]],
        [baseline["exact"]["schedule_spread_mean"]],
        s=60,
        marker="*",
        color="#222222",
        label="exact",
        zorder=5,
    )
    ax1.scatter(
        [baseline["undersupported"]["root_mae"]],
        [baseline["undersupported"]["schedule_spread_mean"]],
        s=60,
        marker="X",
        color="#444444",
        label="undersupported",
        zorder=5,
    )
    ax1.set_xlabel("Root MAE" + (" / (max_segments-1)" if bool(args.normalize) else ""))
    ax1.set_ylabel("Schedule spread mean" + (" / (max_segments-1)" if bool(args.normalize) else ""))
    ax1.set_title("Robustness Diagnostic: Schedule Dependence")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False, fontsize=9)

    out_fig = Path(args.output_figure)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=170)
    plt.close(fig)

    report = {
        "input_glob": str(args.input_glob),
        "n_files": int(len(files)),
        "n_rows": int(len(rows)),
        "filters": {
            "feature_mode": list(args.feature_mode),
            "c3_audit_strategy": list(args.c3_audit_strategy),
            "model_family": list(args.model_family),
            "leaf_query_rates": leaf_q_targets,
            "include_root_query": str(args.include_root_query),
            "budget_axis": str(args.budget_axis),
            "budgets": budget_targets,
        },
        "x_axis": str(args.x_axis),
        "x_axis_label": str(x_labels.get(str(args.x_axis), str(args.x_axis))),
        "aggregate": str(args.aggregate),
        "band": str(args.band),
        "normalize": bool(args.normalize),
        "stage_qualification": {
            "budget_axis_interpretation": f"q_train controlled by `{str(args.budget_axis)}`",
            "guidance_axis_present": False,
        },
        "baseline": baseline,
        "diagnostics": {"full_audit": full_audit_diagnostic},
        "budgets": budgets_sorted,
        "budget_accounting": {
            str(b): {
                "mean_total_queries": _reduce(
                    [float(r.get("total_queries_estimate", float("nan"))) for r in rows if abs(float(r[str(args.budget_axis)]) - float(b)) <= 1e-12],
                    agg=str(args.aggregate),
                ),
                "mean_total_queries_estimate": _reduce(
                    [float(r.get("total_queries_estimate", float("nan"))) for r in rows if abs(float(r[str(args.budget_axis)]) - float(b)) <= 1e-12],
                    agg=str(args.aggregate),
                ),
                "mean_internal_labels_total": _reduce(
                    [float(r.get("internal_labels_total", float("nan"))) for r in rows if abs(float(r[str(args.budget_axis)]) - float(b)) <= 1e-12],
                    agg=str(args.aggregate),
                ),
                "mean_leaf_labels_total": _reduce(
                    [float(r.get("leaf_labels_total", float("nan"))) for r in rows if abs(float(r[str(args.budget_axis)]) - float(b)) <= 1e-12],
                    agg=str(args.aggregate),
                ),
                "mean_root_queries_total": _reduce(
                    [float(r.get("root_queries_total", float("nan"))) for r in rows if abs(float(r[str(args.budget_axis)]) - float(b)) <= 1e-12],
                    agg=str(args.aggregate),
                ),
            }
            for b in budgets_sorted
        },
        "curves": {
            str(b): {
                "x": [float(x) for x in sorted(grouped[b].keys())],
                "y": [_reduce(grouped[b][x], agg=str(args.aggregate)) for x in sorted(grouped[b].keys())],
            }
            for b in budgets_sorted
        },
        "scatter_points_n": int(len(scatter_points)),
        "output_figure": str(out_fig),
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_figure": str(out_fig), "output_json": str(out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
