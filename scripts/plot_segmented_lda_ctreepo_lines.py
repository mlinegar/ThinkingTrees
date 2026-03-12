#!/usr/bin/env python3
"""Line plots for segmented-LDA C-TreePO sweeps."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Dict, List

np = None
plt = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot segmented-LDA sweep lines from per-run JSON outputs.")
    p.add_argument("--input-glob", type=str, default="outputs/segmented_lda_ctreepo/**/*.json")
    p.add_argument("--topic-phi-estimator", type=str, default="", help="Optional exact filter.")
    p.add_argument("--train-docs", type=int, default=-1, help="Optional exact filter.")
    p.add_argument(
        "--x-axis",
        choices=[
            "oracle_cost_ratio",
            "eval_internal_query_rate",
            "calibration_leaf_query_rate",
            "topic_phi_l2_error_mean",
        ],
        default="oracle_cost_ratio",
    )
    p.add_argument(
        "--metric",
        choices=[
            "budgeted_root_l1_mean",
            "budgeted_root_l2_mean",
            "budgeted_c3_violation_rate",
            "budgeted_c1_violation_rate",
            "decomposition_total_root_l1_mean",
            "decomposition_topic_component_mean",
            "decomposition_calibration_component_mean",
            "decomposition_guidance_component_mean",
            "decomposition_oracle_proxy_component_mean",
            "decomposition_slack_mean",
        ],
        default="budgeted_root_l1_mean",
    )
    p.add_argument("--aggregate", choices=["mean", "median"], default="median")
    p.add_argument("--band", choices=["none", "p10_p90", "std"], default="p10_p90")
    p.add_argument(
        "--group-by",
        choices=["none", "train_docs", "calibration_leaf_query_rate", "topic_phi_estimator"],
        default="train_docs",
    )
    p.add_argument("--log-x", action="store_true")
    p.add_argument("--output-figure", type=str, default="outputs/segmented_lda_ctreepo/lines.png")
    p.add_argument("--output-json", type=str, default="outputs/segmented_lda_ctreepo/lines_report.json")
    return p.parse_args()


def _agg(xs: List[float], kind: str) -> float:
    vals = [float(x) for x in xs if np.isfinite(float(x))]
    if not vals:
        return float("nan")
    if kind == "mean":
        return float(fmean(vals))
    return float(median(vals))


def _quantile(xs: List[float], q: float) -> float:
    vals = np.asarray([float(x) for x in xs if np.isfinite(float(x))], dtype=np.float64)
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, float(q)))


def _extract_row(payload: dict) -> dict:
    cfg = payload.get("config", {}) or {}
    topic_meta = payload.get("topic_meta", {}) or {}
    m = payload.get("metrics", {}) or {}
    d = payload.get("decomposition", {}) or {}
    budgeted = m.get("estimated_calibrated_budgeted", {}) or {}
    oracle = m.get("oracle_tree", {}) or {}

    oracle_q = float(oracle.get("mean_total_queries", float("nan")))
    budget_q = float(budgeted.get("mean_total_queries", float("nan")))
    oracle_cost_ratio = float(budget_q / oracle_q) if np.isfinite(oracle_q) and oracle_q > 0 else float("nan")

    return {
        "topic_phi_estimator": str(cfg.get("topic_phi_estimator", "")),
        "train_docs": int(cfg.get("n_books_train", -1)),
        "calibration_leaf_query_rate": float(cfg.get("calibration_leaf_query_rate", float("nan"))),
        "eval_internal_query_rate": float(cfg.get("eval_internal_query_rate", float("nan"))),
        "topic_phi_l2_error_mean": float(topic_meta.get("topic_phi_l2_error_mean", float("nan"))),
        "oracle_cost_ratio": oracle_cost_ratio,
        "budgeted_root_l1_mean": float(budgeted.get("root_l1_mean", float("nan"))),
        "budgeted_root_l2_mean": float(budgeted.get("root_l2_mean", float("nan"))),
        "budgeted_c3_violation_rate": float(budgeted.get("c3_violation_rate", float("nan"))),
        "budgeted_c1_violation_rate": float(budgeted.get("c1_violation_rate", float("nan"))),
        "decomposition_total_root_l1_mean": float(d.get("total_root_l1_mean", float("nan"))),
        "decomposition_topic_component_mean": float(d.get("topic_component_mean", float("nan"))),
        "decomposition_calibration_component_mean": float(d.get("calibration_component_mean", float("nan"))),
        "decomposition_guidance_component_mean": float(d.get("guidance_component_mean", float("nan"))),
        "decomposition_oracle_proxy_component_mean": float(d.get("oracle_proxy_component_mean", float("nan"))),
        "decomposition_slack_mean": float(d.get("slack_mean", float("nan"))),
    }


def _group_key(row: dict, group_by: str) -> str:
    if group_by == "none":
        return "all"
    if group_by == "train_docs":
        return f"train_{int(row['train_docs'])}"
    if group_by == "calibration_leaf_query_rate":
        return f"cal_{float(row['calibration_leaf_query_rate']):.6g}"
    if group_by == "topic_phi_estimator":
        return str(row["topic_phi_estimator"])
    return "all"


def main() -> int:
    args = parse_args()
    global np, plt
    import numpy as np  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore

    paths = [Path(p) for p in sorted(glob.glob(str(args.input_glob), recursive=True))]
    if not paths:
        raise ValueError(f"no files matched: {args.input_glob}")

    rows = [_extract_row(json.loads(p.read_text(encoding="utf-8"))) for p in paths]
    if args.topic_phi_estimator:
        rows = [r for r in rows if str(r["topic_phi_estimator"]) == str(args.topic_phi_estimator)]
    if int(args.train_docs) >= 0:
        rows = [r for r in rows if int(r["train_docs"]) == int(args.train_docs)]
    if not rows:
        raise ValueError("no rows after filters")

    grouped: Dict[str, Dict[float, List[float]]] = {}
    for r in rows:
        x = float(r[str(args.x_axis)])
        y = float(r[str(args.metric)])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        g = _group_key(r, str(args.group_by))
        grouped.setdefault(g, {}).setdefault(x, []).append(y)
    if not grouped:
        raise ValueError("no finite rows for plotting")

    fig, ax = plt.subplots(figsize=(10.2, 6.2), constrained_layout=True)
    report_groups: Dict[str, dict] = {}

    for g, by_x in sorted(grouped.items(), key=lambda kv: kv[0]):
        xs = sorted(by_x.keys())
        ys = [_agg(by_x[x], str(args.aggregate)) for x in xs]
        ys_lo: List[float] = []
        ys_hi: List[float] = []
        for x in xs:
            vals = by_x[x]
            if str(args.band) == "none":
                lo = hi = float("nan")
            elif str(args.band) == "p10_p90":
                lo = _quantile(vals, 0.10)
                hi = _quantile(vals, 0.90)
            else:  # std
                center = _agg(vals, str(args.aggregate))
                sd = float(pstdev(vals)) if len(vals) > 0 else float("nan")
                lo = center - sd
                hi = center + sd
            ys_lo.append(lo)
            ys_hi.append(hi)

        ax.plot(xs, ys, marker="o", linewidth=1.8, label=g)
        if str(args.band) != "none":
            lo_arr = np.asarray(ys_lo, dtype=np.float64)
            hi_arr = np.asarray(ys_hi, dtype=np.float64)
            ok = np.isfinite(lo_arr) & np.isfinite(hi_arr)
            if np.any(ok):
                xs_ok = np.asarray(xs, dtype=np.float64)[ok]
                ax.fill_between(xs_ok, lo_arr[ok], hi_arr[ok], alpha=0.18)

        report_groups[g] = {
            "x": [float(x) for x in xs],
            "y": [float(y) for y in ys],
            "y_lo": [float(v) for v in ys_lo],
            "y_hi": [float(v) for v in ys_hi],
            "counts": {f"{x:.6g}": int(len(by_x[x])) for x in xs},
        }

    ax.set_xlabel(str(args.x_axis))
    ax.set_ylabel(str(args.metric))
    ax.set_title(
        f"Segmented-LDA Lines | metric={args.metric}, aggregate={args.aggregate}, band={args.band}, group_by={args.group_by}"
    )
    if bool(args.log_x):
        ax.set_xscale("log")
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.legend(loc="best", fontsize=9)

    out_fig = Path(args.output_figure)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=170)
    plt.close(fig)

    report = {
        "input_glob": str(args.input_glob),
        "n_files": int(len(paths)),
        "n_rows_after_filters": int(len(rows)),
        "filters": {
            "topic_phi_estimator": str(args.topic_phi_estimator),
            "train_docs": int(args.train_docs),
        },
        "x_axis": str(args.x_axis),
        "metric": str(args.metric),
        "aggregate": str(args.aggregate),
        "band": str(args.band),
        "group_by": str(args.group_by),
        "groups": report_groups,
        "output_figure": str(out_fig),
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({"output_figure": str(out_fig), "output_json": str(out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
