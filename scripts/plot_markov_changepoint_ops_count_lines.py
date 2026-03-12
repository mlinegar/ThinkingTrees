#!/usr/bin/env python3
"""Plot learning curves for the OPS Markov changepoint-count simulation.

This expects per-run JSON outputs from `run_markov_changepoint_ops_count_simulation.py`.

The goal is to complement the grid heatmap with a view that makes it obvious whether:
  (a) error shrinks with more training docs / more oracle labels, and
  (b) the learned sketch approaches the relevant bias floor (e.g., undersupported baseline).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import subprocess
import statistics
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _weight_suffix(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _with_variant_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}__{suffix}{path.suffix}")


def _write_multi_weight_manifest(
    *,
    output_figure: Path,
    output_json: Path,
    llw_vals: List[float],
    children: List[dict],
    x_axis: str,
    y_axis: str,
    aggregate: str,
    band: str,
    normalize: bool,
) -> None:
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Multiple local_law_weight values detected.",
        "Per-weight curve sets were written as sibling artifacts:",
        "",
    ]
    for child in children:
        lines.append(
            f"llw={float(child['local_law_weight']):g} | figure={child['figure']} | report={child['report']}"
        )
    fig_height = max(4.0, 1.4 + 0.33 * max(1, len(lines)))
    fig, ax = plt.subplots(figsize=(16.0, fig_height), constrained_layout=True)
    ax.axis("off")
    ax.text(
        0.01,
        0.99,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
    )
    fig.suptitle(
        "OPS changepoint-count learning curves | multi local_law_weight manifest"
        + f" | x={x_axis} | y={y_axis} | agg={aggregate} | band={band}"
        + (" | normalized" if normalize else ""),
        fontsize=12,
    )
    fig.savefig(output_figure, dpi=180)
    plt.close(fig)
    manifest = {
        "mode": "multi_local_law_weight",
        "x_axis": str(x_axis),
        "y_axis": str(y_axis),
        "aggregate": str(aggregate),
        "band": str(band),
        "normalize": bool(normalize),
        "local_law_weights": [float(x) for x in llw_vals],
        "children": children,
        "output_figure": str(output_figure),
    }
    output_json.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot OPS changepoint-count learning curves.")
    parser.add_argument(
        "--input-glob",
        type=str,
        default="outputs/markov_changepoint_ops_count/**/*seed_*.json",
        help="Glob for per-run JSON outputs.",
    )
    parser.add_argument(
        "--output-figure",
        type=str,
        default="outputs/markov_changepoint_ops_count_lines.png",
        help="Output PNG figure path.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/markov_changepoint_ops_count_lines_report.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--x-axis",
        type=str,
        choices=[
            "train_docs",
            "total_queries",
            "internal_labels_total",
            "leaf_labels_total",
            "root_queries_total",
        ],
        default="train_docs",
        help="X axis for the learning curves.",
    )
    parser.add_argument(
        "--y-axis",
        type=str,
        choices=["audit_fraction", "internal_per_leaf"],
        default="audit_fraction",
        help="Which budget axis to stratify lines by. 'audit_fraction' matches the sweep knob; "
        "'internal_per_leaf' is the realized label rate after rounding.",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        choices=["median", "mean"],
        default="median",
        help="How to aggregate across seeds for each point in a learning curve.",
    )
    parser.add_argument(
        "--band",
        type=str,
        choices=["none", "p10_p90", "p25_p75"],
        default="p10_p90",
        help="Optional quantile band to shade across seeds.",
    )
    parser.add_argument(
        "--include-flip-baselines",
        action="store_true",
        help="If set, include flip_R1/flip_R2 baseline curves (additional reference families).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="If set, normalize MAE/spread metrics by (max_segments - 1) to put them on ~[0,1].",
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use a log scale for the x-axis.",
    )
    parser.add_argument(
        "--feature-mode",
        action="append",
        default=[],
        help="Filter to a specific feature_mode (repeatable). Default: include all.",
    )
    parser.add_argument(
        "--leaf-query-rate",
        action="append",
        type=float,
        default=[],
        help="Filter to specific leaf_query_rate values (repeatable). Default: include all.",
    )
    parser.add_argument(
        "--root-weight",
        action="append",
        type=float,
        default=[],
        help="Filter to specific root_weight values (repeatable). Default: include all.",
    )
    parser.add_argument(
        "--local-law-weight",
        action="append",
        type=float,
        default=[],
        help="Filter to specific local_law_weight values (repeatable). Required if inputs span multiple local-law settings.",
    )
    parser.add_argument(
        "--schedule-consistency-weight",
        action="append",
        type=float,
        default=[],
        help="Filter to specific schedule_consistency_weight values (repeatable). Default: include all.",
    )
    parser.add_argument(
        "--c3-audit-strategy",
        action="append",
        default=[],
        help="Filter to specific c3_audit_strategy values (repeatable). Default: include all.",
    )
    parser.add_argument(
        "--c3-include-root",
        action="append",
        type=int,
        choices=[0, 1],
        default=[],
        help="Filter to c3_include_root values (repeatable, use 1/0). Default: include all.",
    )
    return parser.parse_args()


def _percentile(vals: List[float], q: float) -> float:
    if not vals:
        return float("nan")
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))


def _reduce(vals: List[float], *, agg: str) -> float:
    if not vals:
        return float("nan")
    if agg == "mean":
        return float(np.mean(np.asarray(vals, dtype=np.float64)))
    if agg == "median":
        return float(statistics.median(vals))
    raise ValueError(f"unsupported aggregate: {agg!r}")


def _collect_rows(files: Iterable[Path]) -> List[dict]:
    rows: List[dict] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        objective = payload.get("objective", {}) or {}
        geom = payload.get("training_geometry", {})
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict) or "learned" not in metrics:
            continue

        train_docs = int(cfg.get("train_docs", -1))
        seed = int(cfg.get("seed", -1))
        feature_mode = str(cfg.get("feature_mode", ""))
        leaf_query_rate = float(cfg.get("leaf_query_rate", float("nan")))
        root_weight = float(cfg.get("root_weight", float("nan")))
        local_law_weight = float(
            objective.get(
                "local_law_weight",
                cfg.get(
                    "local_law_weight",
                    float(cfg.get("leaf_weight", 0.0)) + float(cfg.get("c3_weight", 0.0)),
                ),
            )
        )
        schedule_consistency_weight = float(cfg.get("schedule_consistency_weight", float("nan")))
        c3_audit_strategy = str(cfg.get("c3_audit_strategy", ""))
        c3_include_root = bool(cfg.get("c3_include_root", True))
        tau = float(cfg.get("violation_tau", float("nan")))
        audit_policy = str(cfg.get("audit_policy", ""))
        audit_fraction = float(cfg.get("audit_fraction", float("nan")))
        max_segments = int(cfg.get("max_segments", -1))
        count_scale = float(max(1, max_segments - 1)) if max_segments > 0 else float("nan")

        mean_leaves = float(geom.get("mean_leaves", float("nan")))
        mean_internal = float(geom.get("mean_internal_labels", float("nan")))
        root_queries_total = int(geom.get("root_queries_total", -1))
        leaf_labels_total = int(geom.get("leaf_labels_total", -1))
        internal_labels_total = int(geom.get("internal_labels_total", -1))
        total_queries = int(geom.get("total_queries_estimate", -1))
        if train_docs <= 0 or seed < 0 or not np.isfinite(mean_leaves) or mean_leaves <= 0:
            continue

        internal_per_leaf = float(mean_internal) / float(mean_leaves)

        def _metric(sketch: str, key: str) -> float:
            block = metrics.get(sketch, {})
            if not isinstance(block, dict):
                return float("nan")
            val = float(block.get(key, float("nan")))
            return float(val)

        rows.append(
            {
                "path": str(path),
                "train_docs": int(train_docs),
                "total_queries": int(total_queries),
                "root_queries_total": int(root_queries_total),
                "leaf_labels_total": int(leaf_labels_total),
                "internal_labels_total": int(internal_labels_total),
                "seed": int(seed),
                "feature_mode": str(feature_mode),
                "leaf_query_rate": float(leaf_query_rate),
                "root_weight": float(root_weight),
                "local_law_weight": float(local_law_weight),
                "schedule_consistency_weight": float(schedule_consistency_weight),
                "c3_audit_strategy": str(c3_audit_strategy),
                "c3_include_root": bool(c3_include_root),
                "violation_tau": float(tau),
                "audit_policy": str(audit_policy),
                "audit_fraction": float(audit_fraction),
                "internal_per_leaf": float(internal_per_leaf),
                "count_scale": float(count_scale),
                "metrics": {
                    "exact": {
                        "root_mae": _metric("exact", "root_mae"),
                        "merge_mae": _metric("exact", "merge_mae"),
                        "schedule_spread_mean": _metric("exact", "schedule_spread_mean"),
                    },
                    "undersupported": {
                        "root_mae": _metric("undersupported", "root_mae"),
                        "merge_mae": _metric("undersupported", "merge_mae"),
                        "schedule_spread_mean": _metric("undersupported", "schedule_spread_mean"),
                    },
                    "flip_R1": {
                        "root_mae": _metric("flip_R1", "root_mae"),
                        "merge_mae": _metric("flip_R1", "merge_mae"),
                        "schedule_spread_mean": _metric("flip_R1", "schedule_spread_mean"),
                    },
                    "flip_R2": {
                        "root_mae": _metric("flip_R2", "root_mae"),
                        "merge_mae": _metric("flip_R2", "merge_mae"),
                        "schedule_spread_mean": _metric("flip_R2", "schedule_spread_mean"),
                    },
                    "learned": {
                        "root_mae": _metric("learned", "root_mae"),
                        "merge_mae": _metric("learned", "merge_mae"),
                        "schedule_spread_mean": _metric("learned", "schedule_spread_mean"),
                    },
                },
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    files = [Path(p) for p in sorted(glob.glob(str(args.input_glob), recursive=True))]
    if not files:
        raise ValueError(f"no files matched: {args.input_glob}")

    rows = _collect_rows(files)
    if not rows:
        raise ValueError("no valid learned rows found in inputs")

    feature_modes = sorted({str(r["feature_mode"]) for r in rows})
    if args.feature_mode:
        keep = {str(x) for x in args.feature_mode}
        rows = [r for r in rows if str(r["feature_mode"]) in keep]
        feature_modes = sorted({str(r["feature_mode"]) for r in rows})

    def _float_keep(v: float, allowed: List[float]) -> bool:
        if not allowed:
            return True
        if not np.isfinite(float(v)):
            return False
        for z in allowed:
            if np.isclose(float(v), float(z), atol=1e-12, rtol=1e-9):
                return True
        return False

    if args.leaf_query_rate:
        rows = [r for r in rows if _float_keep(float(r.get("leaf_query_rate", float("nan"))), args.leaf_query_rate)]
    if args.root_weight:
        rows = [r for r in rows if _float_keep(float(r.get("root_weight", float("nan"))), args.root_weight)]
    if args.local_law_weight:
        rows = [r for r in rows if _float_keep(float(r.get("local_law_weight", float("nan"))), args.local_law_weight)]
    if args.schedule_consistency_weight:
        rows = [
            r
            for r in rows
            if _float_keep(
                float(r.get("schedule_consistency_weight", float("nan"))),
                args.schedule_consistency_weight,
            )
        ]
    if args.c3_audit_strategy:
        keep_c3 = {str(x) for x in args.c3_audit_strategy}
        rows = [r for r in rows if str(r.get("c3_audit_strategy", "")) in keep_c3]
    if args.c3_include_root:
        keep_root = {bool(int(x)) for x in args.c3_include_root}
        rows = [r for r in rows if bool(r.get("c3_include_root", True)) in keep_root]
    feature_modes = sorted({str(r["feature_mode"]) for r in rows})
    if not rows:
        raise ValueError("no rows remaining after filtering")
    llw_vals = sorted({float(r.get("local_law_weight", float("nan"))) for r in rows if np.isfinite(float(r.get("local_law_weight", float("nan"))))})
    if len(llw_vals) > 1 and not args.local_law_weight:
        parent_fig = Path(args.output_figure)
        parent_json = Path(args.output_json)
        base_argv = list(sys.argv[1:])
        children: List[dict] = []
        for llw in llw_vals:
            suffix = f"llw_{_weight_suffix(llw)}"
            child_fig = _with_variant_suffix(parent_fig, suffix)
            child_json = _with_variant_suffix(parent_json, suffix)
            cmd = [
                sys.executable,
                __file__,
                *base_argv,
                "--local-law-weight",
                f"{float(llw):.12g}",
                "--output-figure",
                str(child_fig),
                "--output-json",
                str(child_json),
            ]
            subprocess.run(cmd, check=True)
            children.append(
                {
                    "local_law_weight": float(llw),
                    "figure": str(child_fig),
                    "report": str(child_json),
                }
            )
        _write_multi_weight_manifest(
            output_figure=parent_fig,
            output_json=parent_json,
            llw_vals=llw_vals,
            children=children,
            x_axis=str(args.x_axis),
            y_axis=str(args.y_axis),
            aggregate=str(args.aggregate),
            band=str(args.band),
            normalize=bool(args.normalize),
        )
        return 0

    x_axis = str(args.x_axis)
    y_axis = str(args.y_axis)
    x_values = sorted({int(r[x_axis]) for r in rows})
    budgets = sorted({float(r[y_axis]) for r in rows if np.isfinite(float(r[y_axis]))})
    taus = sorted({float(r["violation_tau"]) for r in rows if np.isfinite(float(r["violation_tau"]))})
    tau_label = ""
    if taus:
        if len(taus) == 1:
            tau_label = f" | τ={taus[0]:g}"
        else:
            tau_label = f" | τ∈[{min(taus):g},{max(taus):g}]"

    def _budget_label(x: float) -> str:
        if y_axis == "audit_fraction":
            return f"{x:g}"
        if x < 1.0:
            return f"{x:.3f}".rstrip("0").rstrip(".") + "/leaf"
        return f"{x:.2f}".rstrip("0").rstrip(".") + "/leaf"

    # Report structure (useful for debugging / paper tables).
    report: Dict[str, object] = {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "aggregate": str(args.aggregate),
        "band": str(args.band),
        "normalize": bool(args.normalize),
        "feature_modes": feature_modes,
        "filters": {
            "feature_mode": [str(x) for x in args.feature_mode],
            "leaf_query_rate": [float(x) for x in args.leaf_query_rate],
            "root_weight": [float(x) for x in args.root_weight],
            "local_law_weight": [float(x) for x in args.local_law_weight],
            "schedule_consistency_weight": [float(x) for x in args.schedule_consistency_weight],
            "c3_audit_strategy": [str(x) for x in args.c3_audit_strategy],
            "c3_include_root": [int(x) for x in args.c3_include_root],
        },
        "budgets": budgets,
        "x_values": x_values,
        "tau_values": taus,
        "series": {},
    }

    metrics = ("root_mae", "merge_mae", "schedule_spread_mean")
    metric_titles = {
        "root_mae": "Root MAE",
        "merge_mae": "Merge MAE",
        "schedule_spread_mean": "Schedule spread mean",
    }

    def _maybe_norm(r: dict, v: float) -> float:
        if not args.normalize:
            return float(v)
        scale = float(r.get("count_scale", float("nan")))
        if np.isfinite(scale) and scale > 0:
            return float(v) / float(scale)
        return float(v)

    # Layout: one row per feature_mode, 3 metric columns.
    nrows = int(max(1, len(feature_modes)))
    fig, axes = plt.subplots(nrows, 3, figsize=(14, 4.2 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = np.asarray([axes])

    cmap = plt.get_cmap("viridis")
    colors = {b: cmap(i / max(1, len(budgets) - 1)) for i, b in enumerate(budgets)}

    def _band_quantiles() -> Optional[Tuple[float, float]]:
        if str(args.band) == "none":
            return None
        if str(args.band) == "p10_p90":
            return (10.0, 90.0)
        if str(args.band) == "p25_p75":
            return (25.0, 75.0)
        raise ValueError(f"unsupported band: {args.band!r}")

    band_q = _band_quantiles()

    for r_i, mode in enumerate(feature_modes):
        mode_rows = [r for r in rows if str(r["feature_mode"]) == str(mode)]
        if not mode_rows:
            continue
        for c_i, metric in enumerate(metrics):
            ax = axes[r_i, c_i]

            # Learned curves: one per budget.
            for b in budgets:
                xs: List[int] = []
                ys: List[float] = []
                lo: List[float] = []
                hi: List[float] = []
                for x in x_values:
                    vals = [
                        _maybe_norm(r, float(r["metrics"]["learned"][metric]))
                        for r in mode_rows
                        if int(r[x_axis]) == int(x) and float(r[y_axis]) == float(b)
                        and np.isfinite(float(r["metrics"]["learned"][metric]))
                    ]
                    if not vals:
                        continue
                    xs.append(int(x))
                    ys.append(float(_reduce(vals, agg=str(args.aggregate))))
                    if band_q is not None:
                        lo.append(float(_percentile(vals, band_q[0])))
                        hi.append(float(_percentile(vals, band_q[1])))
                if not xs:
                    continue
                ax.plot(xs, ys, marker="o", linewidth=1.8, color=colors[b], label=_budget_label(b))
                if band_q is not None and len(lo) == len(xs) and len(hi) == len(xs):
                    ax.fill_between(xs, lo, hi, color=colors[b], alpha=0.20, linewidth=0)

            # Baselines: exact and undersupported (plus optional flip families).
            baselines = [
                ("exact", ("--", 1.6), "black"),
                ("undersupported", (":", 1.8), "gray"),
            ]
            if bool(args.include_flip_baselines):
                baselines.extend(
                    [
                        ("flip_R1", ("-.", 1.6), "#c44e52"),
                        ("flip_R2", ("-.", 1.6), "#dd8452"),
                    ]
                )

            # Dedupe budgets per seed/x.
            for sketch, style, color in baselines:
                base_x: List[int] = []
                base_y: List[float] = []
                for x in x_values:
                    # Deduplicate across budgets by taking one value per (seed, x).
                    seen: set[int] = set()
                    vals: List[float] = []
                    for r in mode_rows:
                        if int(r[x_axis]) != int(x):
                            continue
                        seed = int(r["seed"])
                        if seed in seen:
                            continue
                        seen.add(seed)
                        v = float(r["metrics"][sketch][metric])
                        if not np.isfinite(v):
                            continue
                        vals.append(_maybe_norm(r, v))
                    if not vals:
                        continue
                    base_x.append(int(x))
                    base_y.append(float(_reduce(vals, agg=str(args.aggregate))))
                if base_x:
                    ax.plot(
                        base_x,
                        base_y,
                        linestyle=style[0],
                        linewidth=style[1],
                        color=color,
                        label=str(sketch),
                    )

            title_suffix = " / (max_segments-1)" if args.normalize else ""
            if r_i == 0:
                ax.set_title(f"{metric_titles[metric]}{title_suffix} (↓)")
            if c_i == 0:
                ax.set_ylabel(f"feature_mode={mode}")
            if x_axis == "train_docs":
                ax.set_xlabel("train docs")
            elif x_axis == "total_queries":
                ax.set_xlabel("total oracle queries (train)")
            elif x_axis == "internal_labels_total":
                ax.set_xlabel("internal oracle labels (train)")
            elif x_axis == "leaf_labels_total":
                ax.set_xlabel("leaf oracle labels (train)")
            elif x_axis == "root_queries_total":
                ax.set_xlabel("root oracle labels (train)")
            else:
                ax.set_xlabel(str(x_axis))
            if args.log_x:
                ax.set_xscale("log")
            ax.grid(True, alpha=0.25)

            # Build JSON report series (for the learned curves).
            series_key = f"{mode}:{metric}"
            ser: Dict[str, object] = {}
            for b in budgets:
                pts: List[Dict[str, float | int]] = []
                for x in x_values:
                    vals = [
                        _maybe_norm(r, float(r["metrics"]["learned"][metric]))
                        for r in mode_rows
                        if int(r[x_axis]) == int(x) and float(r[y_axis]) == float(b)
                        and np.isfinite(float(r["metrics"]["learned"][metric]))
                    ]
                    if not vals:
                        continue
                    entry: Dict[str, float | int] = {
                        "x": int(x),
                        "center": float(_reduce(vals, agg=str(args.aggregate))),
                    }
                    if band_q is not None:
                        entry["lo"] = float(_percentile(vals, band_q[0]))
                        entry["hi"] = float(_percentile(vals, band_q[1]))
                    pts.append(entry)
                if pts:
                    ser[str(b)] = pts
            report["series"][series_key] = ser

    # Legend: de-duplicate labels across axes.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=min(6, len(by_label)))

    fig.suptitle(
        f"OPS changepoint-count learning curves | x={x_axis} | y={y_axis} | agg={args.aggregate}"
        + (" | normalized" if args.normalize else "")
        + tau_label,
        fontsize=12,
    )

    out_fig = Path(args.output_figure)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({"output_figure": str(out_fig), "output_json": str(out_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
