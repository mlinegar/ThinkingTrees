#!/usr/bin/env python3
"""Plot a train_docs × node-label-budget grid for the OPS changepoint-count simulation.

This expects per-run JSON outputs from `run_markov_changepoint_ops_count_simulation.py`.

Notes on interpretation:
- `merge_violation_rate` depends on `config.violation_tau`. If `violation_tau=0.0` and the learned
  model outputs real-valued counts, the violation rate will typically be ~1.0 and is not
  informative; prefer plotting `merge_mae` in that case.

Paper-facing "honesty" layout:
- Use `--layout honesty` to plot, in one figure:
  - learned sketch performance vs (train_docs × audit budget),
  - an insufficient-sketch baseline ("approximation bias floor"),
  - selection-bias diagnostics (naive vs IPW vs DSL/AIPW).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import subprocess
from statistics import fmean
import statistics
import sys
from typing import Dict, List, Tuple

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
    layout: str,
    aggregate: str,
    y_axis: str,
    normalize: bool,
) -> None:
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Multiple local_law_weight values detected.",
        "Per-weight figures and reports were written as sibling artifacts:",
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
        "OPS changepoint-count grid | multi local_law_weight manifest"
        + f" | layout={layout} | y={y_axis} | agg={aggregate}"
        + (" | normalized" if normalize else ""),
        fontsize=12,
    )
    fig.savefig(output_figure, dpi=180)
    plt.close(fig)
    manifest = {
        "mode": "multi_local_law_weight",
        "layout": str(layout),
        "aggregate": str(aggregate),
        "y_axis": str(y_axis),
        "normalize": bool(normalize),
        "local_law_weights": [float(x) for x in llw_vals],
        "children": children,
        "output_figure": str(output_figure),
    }
    output_json.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot OPS changepoint-count grid.")
    parser.add_argument(
        "--input-glob",
        type=str,
        default="outputs/markov_changepoint_ops_count/**/*seed_*.json",
        help="Glob for per-run JSON outputs.",
    )
    parser.add_argument(
        "--output-figure",
        type=str,
        default="outputs/markov_changepoint_ops_count_grid.png",
        help="Output PNG figure path.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/markov_changepoint_ops_count_grid_report.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--layout",
        type=str,
        choices=["default", "honesty"],
        default="default",
        help="Layout preset. 'default' plots one sketch family (plus optional bias panels). "
        "'honesty' plots learned vs a baseline sketch plus audit-bias panels (paper-facing).",
    )
    parser.add_argument(
        "--y-axis",
        type=str,
        choices=["audit_fraction", "internal_per_leaf"],
        default="audit_fraction",
        help="Which budget axis to plot. 'audit_fraction' matches the sweep knob; "
        "'internal_per_leaf' is the realized label rate after rounding.",
    )
    parser.add_argument(
        "--sketch",
        type=str,
        choices=["learned", "exact", "undersupported", "flip_R1", "flip_R2"],
        default="learned",
        help="Which sketch family to plot for performance panels (default layout).",
    )
    parser.add_argument(
        "--baseline-sketch",
        type=str,
        choices=["exact", "undersupported", "flip_R1", "flip_R2"],
        default="undersupported",
        help="Baseline sketch family for the approximation-bias row (honesty layout).",
    )
    parser.add_argument(
        "--include-bias-panels",
        action="store_true",
        help="If set, add a second row of heatmaps showing absolute bias of naive/IPW/DSL audit estimators. "
        "These diagnostics are computed on the learned-sketch merge-error population.",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        choices=["mean", "median", "p10", "p90"],
        default="mean",
        help="How to aggregate across seeds for each grid cell.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="If set, normalize MAE/spread/bias metrics by (max_segments - 1) to put them on ~[0,1].",
    )
    parser.add_argument(
        "--feature-mode",
        action="append",
        default=[],
        help="Filter to specific feature_mode values (repeatable). Default: include all.",
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


def _collect_runs(files: List[Path]) -> List[dict]:
    runs: List[dict] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        objective = payload.get("objective", {}) or {}
        geom = payload.get("training_geometry", {})
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        diag = payload.get("estimator_diagnostics", {}) or {}
        train_docs = int(cfg.get("train_docs", -1))
        seed = int(cfg.get("seed", -1))
        tau = float(cfg.get("violation_tau", float("nan")))
        audit_policy = str(cfg.get("audit_policy", ""))
        audit_fraction = float(cfg.get("audit_fraction", float("nan")))
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
        max_segments = int(cfg.get("max_segments", -1))
        count_scale = float(max(1, max_segments - 1)) if max_segments > 0 else float("nan")
        mean_leaves = float(geom.get("mean_leaves", float("nan")))
        mean_internal = float(geom.get("mean_internal_labels", float("nan")))
        if train_docs <= 0 or not np.isfinite(mean_leaves) or mean_leaves <= 0:
            continue
        internal_per_leaf = float(mean_internal) / float(mean_leaves)

        by_sketch: Dict[str, Dict[str, float]] = {}
        for name in ("learned", "exact", "undersupported", "flip_R1", "flip_R2"):
            block = metrics.get(str(name))
            if not isinstance(block, dict):
                continue
            by_sketch[str(name)] = {
                "root_mae": float(block.get("root_mae", float("nan"))),
                "leaf_mae": float(block.get("leaf_mae", float("nan"))),
                "merge_mae": float(block.get("merge_mae", float("nan"))),
                "merge_violation_rate": float(block.get("merge_violation_rate", float("nan"))),
                "schedule_spread_mean": float(block.get("schedule_spread_mean", float("nan"))),
            }

        if not by_sketch:
            continue

        runs.append(
            {
                "train_docs": int(train_docs),
                "seed": int(seed),
                "violation_tau": float(tau),
                "audit_policy": str(audit_policy),
                "audit_fraction": float(audit_fraction),
                "feature_mode": str(feature_mode),
                "leaf_query_rate": float(leaf_query_rate),
                "root_weight": float(root_weight),
                "local_law_weight": float(local_law_weight),
                "schedule_consistency_weight": float(schedule_consistency_weight),
                "c3_audit_strategy": str(c3_audit_strategy),
                "c3_include_root": bool(c3_include_root),
                "max_segments": int(max_segments),
                "count_scale": float(count_scale),
                "mean_internal_labels": float(mean_internal),
                "mean_leaves": float(mean_leaves),
                "internal_per_leaf": float(internal_per_leaf),
                "metrics": by_sketch,
                "diag_abs_naive_bias": abs(float(diag.get("naive_bias", float("nan")))),
                "diag_abs_ipw_bias": abs(float(diag.get("ipw_bias", float("nan")))),
                "diag_abs_dsl_bias": abs(float(diag.get("dsl_bias", float("nan")))),
            }
        )
    return runs


def _heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    *,
    xlabels: List[str],
    ylabels: List[str],
    title: str,
    cmap: str = "viridis_r",
) -> None:
    im = ax.imshow(mat, aspect="auto", cmap=cmap, origin="lower")
    ax.set_title(title)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main() -> int:
    args = parse_args()
    layout = str(args.layout)
    if layout == "honesty" and str(args.sketch) != "learned":
        raise ValueError("--layout honesty requires --sketch learned")
    if layout == "default" and bool(args.include_bias_panels) and str(args.sketch) != "learned":
        raise ValueError("--include-bias-panels currently requires --sketch learned (bias diagnostics use learned merges)")
    files = [Path(p) for p in sorted(glob.glob(str(args.input_glob), recursive=True))]
    if not files:
        raise ValueError(f"no files matched: {args.input_glob}")

    runs = _collect_runs(files)
    if not runs:
        raise ValueError("no valid runs found in inputs")

    def _float_keep(v: float, allowed: List[float]) -> bool:
        if not allowed:
            return True
        if not np.isfinite(float(v)):
            return False
        for z in allowed:
            if np.isclose(float(v), float(z), atol=1e-12, rtol=1e-9):
                return True
        return False

    if args.feature_mode:
        keep_modes = {str(x) for x in args.feature_mode}
        runs = [r for r in runs if str(r.get("feature_mode", "")) in keep_modes]
    if args.leaf_query_rate:
        runs = [r for r in runs if _float_keep(float(r.get("leaf_query_rate", float("nan"))), args.leaf_query_rate)]
    if args.root_weight:
        runs = [r for r in runs if _float_keep(float(r.get("root_weight", float("nan"))), args.root_weight)]
    if args.local_law_weight:
        runs = [r for r in runs if _float_keep(float(r.get("local_law_weight", float("nan"))), args.local_law_weight)]
    if args.schedule_consistency_weight:
        runs = [
            r
            for r in runs
            if _float_keep(
                float(r.get("schedule_consistency_weight", float("nan"))),
                args.schedule_consistency_weight,
            )
        ]
    if args.c3_audit_strategy:
        keep_c3 = {str(x) for x in args.c3_audit_strategy}
        runs = [r for r in runs if str(r.get("c3_audit_strategy", "")) in keep_c3]
    if args.c3_include_root:
        keep_root = {bool(int(x)) for x in args.c3_include_root}
        runs = [r for r in runs if bool(r.get("c3_include_root", True)) in keep_root]
    if not runs:
        raise ValueError("no runs remaining after filtering")
    llw_vals = sorted({float(r.get("local_law_weight", float("nan"))) for r in runs if np.isfinite(float(r.get("local_law_weight", float("nan"))))})
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
            layout=str(layout),
            aggregate=str(args.aggregate),
            y_axis=str(args.y_axis),
            normalize=bool(args.normalize),
        )
        return 0

    train_docs_values = sorted({int(r["train_docs"]) for r in runs})
    y_axis = str(args.y_axis)
    if y_axis == "audit_fraction":
        budgets = sorted(
            {float(r["audit_fraction"]) for r in runs if np.isfinite(float(r["audit_fraction"]))}
        )
    else:
        budgets = sorted({float(r["internal_per_leaf"]) for r in runs})
    taus = sorted({float(r["violation_tau"]) for r in runs if np.isfinite(float(r["violation_tau"]))})
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

    xlabels = [str(x) for x in train_docs_values]
    ylabels = []
    for b in budgets:
        label = _budget_label(b)
        if y_axis == "audit_fraction":
            # Optionally append the realized internal labels per doc (after rounding).
            internal_vals = [
                float(r["mean_internal_labels"])
                for r in runs
                if float(r["audit_fraction"]) == float(b) and np.isfinite(float(r["mean_internal_labels"]))
            ]
            if internal_vals:
                # In fraction policy, this is typically an integer like 2,3,5,12,23.
                realized = float(fmean(internal_vals))
                label = f"{label} ({realized:.0f} internal/doc)"
        ylabels.append(label)

    def _reduce(vals: List[float]) -> float:
        if not vals:
            return float("nan")
        agg = str(args.aggregate)
        if agg == "mean":
            return float(fmean(vals))
        if agg == "median":
            return float(statistics.median(vals))
        if agg == "p10":
            return float(np.percentile(np.asarray(vals, dtype=np.float64), 10))
        if agg == "p90":
            return float(np.percentile(np.asarray(vals, dtype=np.float64), 90))
        raise ValueError(f"unsupported aggregate: {agg!r}")

    def _grid(metric: str) -> np.ndarray:
        mat = np.full((len(budgets), len(train_docs_values)), np.nan, dtype=np.float64)
        for bi, b in enumerate(budgets):
            for xi, td in enumerate(train_docs_values):
                if y_axis == "audit_fraction":
                    in_cell = lambda r: float(r["audit_fraction"]) == float(b)  # noqa: E731
                else:
                    in_cell = lambda r: float(r["internal_per_leaf"]) == float(b)  # noqa: E731
                vals: List[float] = []
                for r in runs:
                    if int(r["train_docs"]) != int(td) or not in_cell(r):
                        continue
                    v = float(r.get(metric, float("nan")))
                    if not np.isfinite(v):
                        continue
                    if args.normalize:
                        scale = float(r.get("count_scale", float("nan")))
                        if np.isfinite(scale) and scale > 0:
                            v = v / scale
                    vals.append(float(v))
                if vals:
                    mat[bi, xi] = float(_reduce(vals))
        return mat

    def _sketch_grid(sketch: str, metric: str) -> np.ndarray:
        mat = np.full((len(budgets), len(train_docs_values)), np.nan, dtype=np.float64)
        for bi, b in enumerate(budgets):
            for xi, td in enumerate(train_docs_values):
                if y_axis == "audit_fraction":
                    in_cell = lambda r: float(r["audit_fraction"]) == float(b)  # noqa: E731
                else:
                    in_cell = lambda r: float(r["internal_per_leaf"]) == float(b)  # noqa: E731
                vals: List[float] = []
                for r in runs:
                    if int(r["train_docs"]) != int(td) or not in_cell(r):
                        continue
                    block = r.get("metrics", {}).get(str(sketch), {})
                    if not isinstance(block, dict):
                        continue
                    v = float(block.get(metric, float("nan")))
                    if not np.isfinite(v):
                        continue
                    if args.normalize:
                        scale = float(r.get("count_scale", float("nan")))
                        if np.isfinite(scale) and scale > 0:
                            v = v / scale
                    vals.append(float(v))
                if vals:
                    mat[bi, xi] = float(_reduce(vals))
        return mat

    main_sketch = str(args.sketch)
    baseline_sketch = str(args.baseline_sketch)
    include_bias_panels = bool(args.include_bias_panels) or (layout == "honesty")

    # Validate sketch availability.
    if layout == "default":
        if not any(main_sketch in (r.get("metrics") or {}) for r in runs):
            raise ValueError(f"no rows found in inputs for sketch={main_sketch!r}")
    else:
        if not any("learned" in (r.get("metrics") or {}) for r in runs):
            raise ValueError("no learned rows found in inputs")
        if not any(baseline_sketch in (r.get("metrics") or {}) for r in runs):
            raise ValueError(f"no rows found in inputs for baseline_sketch={baseline_sketch!r}")

    if layout == "honesty":
        perf_metrics = ("root_mae", "leaf_mae", "merge_mae")
    else:
        perf_metrics = ("root_mae", "merge_mae", "schedule_spread_mean")

    metric_suffix = " / (max_segments-1)" if args.normalize else ""
    nrows = 1
    if layout == "honesty":
        nrows = 3
    elif include_bias_panels:
        nrows = 2

    if nrows == 1:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
        axes = np.asarray([axes])
    elif nrows == 2:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)
    else:
        fig, axes = plt.subplots(3, 3, figsize=(14, 12.0), constrained_layout=True)

    perf_axes_main = axes[0]
    perf_axes_base = axes[1] if layout == "honesty" else None
    bias_axes = axes[-1] if include_bias_panels else None

    # Main performance row.
    for j, metric in enumerate(perf_metrics):
        title = {
            "root_mae": f"{main_sketch} | Root MAE{metric_suffix} (↓)",
            "leaf_mae": f"{main_sketch} | Leaf MAE{metric_suffix} (↓)",
            "merge_mae": f"{main_sketch} | Merge MAE{metric_suffix} (↓)",
            "schedule_spread_mean": f"{main_sketch} | Schedule spread mean{metric_suffix} (↓)",
        }.get(metric, f"{main_sketch} | {metric}{metric_suffix}")
        _heatmap(
            perf_axes_main[j],
            _sketch_grid(main_sketch, metric),
            xlabels=xlabels,
            ylabels=ylabels,
            title=title,
            cmap="viridis_r",
        )

    # Baseline performance row (approximation-bias floor).
    if perf_axes_base is not None:
        for j, metric in enumerate(perf_metrics):
            title = {
                "root_mae": f"{baseline_sketch} | Root MAE{metric_suffix} (↓)",
                "leaf_mae": f"{baseline_sketch} | Leaf MAE{metric_suffix} (↓)",
                "merge_mae": f"{baseline_sketch} | Merge MAE{metric_suffix} (↓)",
                "schedule_spread_mean": f"{baseline_sketch} | Schedule spread mean{metric_suffix} (↓)",
            }.get(metric, f"{baseline_sketch} | {metric}{metric_suffix}")
            _heatmap(
                perf_axes_base[j],
                _sketch_grid(baseline_sketch, metric),
                xlabels=xlabels,
                ylabels=ylabels,
                title=title,
                cmap="viridis_r",
            )

    if bias_axes is not None:
        diag_mats = {
            "abs_naive_bias": _grid("diag_abs_naive_bias"),
            "abs_ipw_bias": _grid("diag_abs_ipw_bias"),
            "abs_dsl_bias": _grid("diag_abs_dsl_bias"),
        }
        _heatmap(
            bias_axes[0],
            diag_mats["abs_naive_bias"],
            xlabels=xlabels,
            ylabels=ylabels,
            title="audit | abs naive bias (↓)",
            cmap="viridis_r",
        )
        _heatmap(
            bias_axes[1],
            diag_mats["abs_ipw_bias"],
            xlabels=xlabels,
            ylabels=ylabels,
            title="audit | abs IPW bias (↓)",
            cmap="viridis_r",
        )
        _heatmap(
            bias_axes[2],
            diag_mats["abs_dsl_bias"],
            xlabels=xlabels,
            ylabels=ylabels,
            title="audit | abs DSL bias (↓)",
            cmap="viridis_r",
        )
    fig.suptitle(
        "OPS changepoint-count grid"
        + (f" | layout={layout}" if layout != "default" else f" ({main_sketch})")
        + (f" | baseline={baseline_sketch}" if layout == "honesty" else "")
        + f" | y={y_axis} | agg={args.aggregate}"
        + (" | normalized" if args.normalize else "")
        + tau_label,
        fontsize=12,
    )

    out_fig = Path(args.output_figure)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=220)
    plt.close(fig)

    report = {
        "input_files": len(files),
        "runs": len(runs),
        "layout": str(layout),
        "sketch": str(main_sketch),
        "baseline_sketch": str(baseline_sketch) if layout == "honesty" else None,
        "include_bias_panels": bool(include_bias_panels),
        "aggregate": str(args.aggregate),
        "y_axis": str(y_axis),
        "normalize": bool(args.normalize),
        "filters": {
            "feature_mode": [str(x) for x in args.feature_mode],
            "leaf_query_rate": [float(x) for x in args.leaf_query_rate],
            "root_weight": [float(x) for x in args.root_weight],
            "schedule_consistency_weight": [float(x) for x in args.schedule_consistency_weight],
            "c3_audit_strategy": [str(x) for x in args.c3_audit_strategy],
            "c3_include_root": [int(x) for x in args.c3_include_root],
        },
        "train_docs_values": train_docs_values,
        "budgets": budgets,
        "violation_tau_values": taus,
        "metrics": {
            "root_mae": "Root MAE",
            "leaf_mae": "Leaf MAE",
            "merge_mae": "Merge MAE",
            "schedule_spread_mean": "Schedule spread mean",
            "merge_violation_rate": "Merge violation rate (thresholded; see violation_tau)",
            "diag_abs_naive_bias": "abs naive bias (audit)",
            "diag_abs_ipw_bias": "abs IPW bias (audit)",
            "diag_abs_dsl_bias": "abs DSL bias (audit)",
        },
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
