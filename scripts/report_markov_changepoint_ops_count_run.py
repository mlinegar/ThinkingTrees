#!/usr/bin/env python3
"""
Generate a small PDF + PNG report for a Markov changepoint OPS-count sweep.

This script is intentionally dependency-light (numpy + matplotlib) so it works in the repo venv.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


@dataclass(frozen=True)
class RunRow:
    path: str
    train_docs: int
    audit_fraction: float
    seed: int
    feature_mode: str
    leaf_query_rate: float
    local_law_weight: float
    schedule_consistency_weight: float
    c3_audit_strategy: str
    c3_include_root: bool
    count_scale: float
    learned_root_mae_n: float
    learned_merge_mae_n: float
    learned_spread_n: float
    exact_root_mae_n: float
    exact_merge_mae_n: float
    exact_spread_n: float
    unders_root_mae_n: float
    unders_merge_mae_n: float
    unders_spread_n: float
    flip_r1_root_mae_n: float
    flip_r1_merge_mae_n: float
    flip_r1_spread_n: float
    flip_r2_root_mae_n: float
    flip_r2_merge_mae_n: float
    flip_r2_spread_n: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Report Markov changepoint OPS-count sweep.")
    p.add_argument(
        "--input-root",
        type=str,
        required=True,
        help="Sweep output directory containing per-run seed_*.json files.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for report artifacts (default: <input_root>/report).",
    )
    p.add_argument(
        "--aggregate",
        choices=["median", "mean"],
        default="median",
        help="Aggregation across seeds for group-level summaries.",
    )
    p.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use normalized errors (divide by max_segments - 1).",
    )
    p.add_argument(
        "--score-merge-weight",
        type=float,
        default=0.50,
        help="Weight on merge MAE in the scalar ranking score.",
    )
    p.add_argument(
        "--score-spread-weight",
        type=float,
        default=0.25,
        help="Weight on schedule spread in the scalar ranking score.",
    )
    return p.parse_args()


def _is_close(a: float, b: float) -> bool:
    return bool(np.isclose(float(a), float(b), atol=1e-12, rtol=1e-9))


def _load_runs(files: Sequence[Path], *, normalize: bool) -> List[RunRow]:
    runs: List[RunRow] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {}) or {}
        objective = payload.get("objective", {}) or {}
        metrics = payload.get("metrics", {}) or {}
        learned = metrics.get("learned", {}) or {}
        unders = metrics.get("undersupported", {}) or {}
        exact = metrics.get("exact", {}) or {}
        flip_r1 = metrics.get("flip_R1", {}) or {}
        flip_r2 = metrics.get("flip_R2", {}) or {}

        train_docs = int(cfg.get("train_docs", -1))
        seed = int(cfg.get("seed", -1))
        audit_fraction = float(cfg.get("audit_fraction", float("nan")))
        feature_mode = str(cfg.get("feature_mode", ""))
        leaf_query_rate = float(cfg.get("leaf_query_rate", float("nan")))
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
        if not np.isfinite(count_scale) or count_scale <= 0:
            continue

        def _norm(x: float) -> float:
            if not normalize:
                return float(x)
            return float(x) / float(count_scale)

        try:
            learned_root = float(learned["root_mae"])
            learned_merge = float(learned["merge_mae"])
            learned_spread = float(learned["schedule_spread_mean"])
            unders_root = float(unders["root_mae"])
            unders_merge = float(unders["merge_mae"])
        except Exception:
            continue

        def _maybe(sketch: dict, key: str) -> float:
            try:
                return float(sketch[key])
            except Exception:
                return float("nan")

        exact_root = _maybe(exact, "root_mae")
        exact_merge = _maybe(exact, "merge_mae")
        exact_spread = _maybe(exact, "schedule_spread_mean")
        unders_spread = _maybe(unders, "schedule_spread_mean")
        flip1_root = _maybe(flip_r1, "root_mae")
        flip1_merge = _maybe(flip_r1, "merge_mae")
        flip1_spread = _maybe(flip_r1, "schedule_spread_mean")
        flip2_root = _maybe(flip_r2, "root_mae")
        flip2_merge = _maybe(flip_r2, "merge_mae")
        flip2_spread = _maybe(flip_r2, "schedule_spread_mean")

        if train_docs <= 0 or seed < 0 or not np.isfinite(audit_fraction):
            continue

        runs.append(
            RunRow(
                path=str(path),
                train_docs=int(train_docs),
                audit_fraction=float(audit_fraction),
                seed=int(seed),
                feature_mode=str(feature_mode),
                leaf_query_rate=float(leaf_query_rate),
                local_law_weight=float(local_law_weight),
                schedule_consistency_weight=float(schedule_consistency_weight),
                c3_audit_strategy=str(c3_audit_strategy),
                c3_include_root=bool(c3_include_root),
                count_scale=float(count_scale),
                learned_root_mae_n=_norm(learned_root),
                learned_merge_mae_n=_norm(learned_merge),
                learned_spread_n=_norm(learned_spread),
                exact_root_mae_n=_norm(exact_root),
                exact_merge_mae_n=_norm(exact_merge),
                exact_spread_n=_norm(exact_spread),
                unders_root_mae_n=_norm(unders_root),
                unders_merge_mae_n=_norm(unders_merge),
                unders_spread_n=_norm(unders_spread),
                flip_r1_root_mae_n=_norm(flip1_root),
                flip_r1_merge_mae_n=_norm(flip1_merge),
                flip_r1_spread_n=_norm(flip1_spread),
                flip_r2_root_mae_n=_norm(flip2_root),
                flip_r2_merge_mae_n=_norm(flip2_merge),
                flip_r2_spread_n=_norm(flip2_spread),
            )
        )
    return runs


def _reduce(xs: Sequence[float], *, agg: str) -> float:
    if len(xs) == 0:
        return float("nan")
    if agg == "median":
        return float(median(xs))
    if agg == "mean":
        return float(fmean(xs))
    raise ValueError(f"unsupported aggregate: {agg!r}")


def _groupby(rows: Sequence[RunRow], key_fn) -> Dict[Tuple[object, ...], List[RunRow]]:
    out: Dict[Tuple[object, ...], List[RunRow]] = {}
    for r in rows:
        k = tuple(key_fn(r))
        out.setdefault(k, []).append(r)
    return out


def _heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    *,
    xlabels: List[str],
    ylabels: List[str],
    title: str,
    cmap: str = "viridis_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _write_text_page(pdf: PdfPages, *, title: str, lines: Sequence[str]) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0.06, 0.06, 0.88, 0.88])
    ax.axis("off")
    ax.text(0.0, 1.0, title, fontsize=16, fontweight="bold", va="top")
    y = 0.95
    for line in lines:
        ax.text(0.0, y, str(line), fontsize=10.5, va="top", family="monospace")
        y -= 0.028
        if y < 0.05:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0.06, 0.06, 0.88, 0.88])
            ax.axis("off")
            y = 0.95
    pdf.savefig(fig)
    plt.close(fig)


def main() -> int:
    try:
        from scripts._markov_report_archive import archived_report_exit
    except ModuleNotFoundError:
        from _markov_report_archive import archived_report_exit

    return archived_report_exit(
        legacy_script="scripts/report_markov_changepoint_ops_count_run.py",
        replacements=(
            "scripts/report_markov_optimization_tradeoffs.py",
            "scripts/run_markov_optimization_tradeoff_pipeline.py",
        ),
        note=(
            "The dedicated OPS-count PDF report is a legacy non-v3 surface and has been archived."
        ),
    )

    args = parse_args()
    input_root = Path(args.input_root)
    if not input_root.exists():
        raise SystemExit(f"input_root not found: {input_root}")

    out_dir = Path(args.output_dir) if args.output_dir else (input_root / "report")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_root.rglob("seed_*.json"))
    if not files:
        raise SystemExit(f"no seed_*.json files found under {input_root}")

    runs = _load_runs(files, normalize=bool(args.normalize))
    if not runs:
        raise SystemExit("no valid runs loaded")

    agg = str(args.aggregate)
    score_merge_w = float(args.score_merge_weight)
    score_spread_w = float(args.score_spread_weight)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    train_docs_vals = sorted({r.train_docs for r in runs})
    budget_vals = sorted({r.audit_fraction for r in runs})
    c3_vals = sorted({r.c3_audit_strategy for r in runs})
    lqr_vals = sorted({r.leaf_query_rate for r in runs})
    llw_vals = sorted({r.local_law_weight for r in runs})
    scw_vals = sorted({r.schedule_consistency_weight for r in runs})
    reg_axis_name = "local_law_weight" if len(llw_vals) > 1 else "schedule_consistency_weight"
    reg_axis_vals = llw_vals if reg_axis_name == "local_law_weight" else scw_vals

    # Group into unique configurations per cell (train_docs x audit_fraction x knobs).
    group_keys = (
        "train_docs",
        "audit_fraction",
        "c3_audit_strategy",
        "leaf_query_rate",
        "local_law_weight",
        "schedule_consistency_weight",
        "c3_include_root",
        "feature_mode",
    )

    groups = _groupby(
        runs,
        lambda r: (
            r.train_docs,
            r.audit_fraction,
            r.c3_audit_strategy,
            r.leaf_query_rate,
            r.local_law_weight,
            r.schedule_consistency_weight,
            r.c3_include_root,
            r.feature_mode,
        ),
    )

    @dataclass(frozen=True)
    class GroupAgg:
        key: Tuple[object, ...]
        n: int
        root: float
        merge: float
        spread: float
        unders_root: float
        unders_merge: float
        score: float

    agg_rows: List[GroupAgg] = []
    for k, rs in groups.items():
        root = _reduce([x.learned_root_mae_n for x in rs], agg=agg)
        merge = _reduce([x.learned_merge_mae_n for x in rs], agg=agg)
        spread = _reduce([x.learned_spread_n for x in rs], agg=agg)
        u_root = _reduce([x.unders_root_mae_n for x in rs], agg=agg)
        u_merge = _reduce([x.unders_merge_mae_n for x in rs], agg=agg)
        score = float(root + score_merge_w * merge + score_spread_w * spread)
        agg_rows.append(GroupAgg(key=k, n=len(rs), root=root, merge=merge, spread=spread, unders_root=u_root, unders_merge=u_merge, score=score))

    # Global summaries.
    all_root = np.asarray([r.learned_root_mae_n for r in runs], dtype=np.float64)
    all_merge = np.asarray([r.learned_merge_mae_n for r in runs], dtype=np.float64)
    all_spread = np.asarray([r.learned_spread_n for r in runs], dtype=np.float64)
    all_exact_root = np.asarray([r.exact_root_mae_n for r in runs], dtype=np.float64)
    all_exact_merge = np.asarray([r.exact_merge_mae_n for r in runs], dtype=np.float64)
    all_exact_spread = np.asarray([r.exact_spread_n for r in runs], dtype=np.float64)
    all_u_root = np.asarray([r.unders_root_mae_n for r in runs], dtype=np.float64)
    all_u_merge = np.asarray([r.unders_merge_mae_n for r in runs], dtype=np.float64)
    all_u_spread = np.asarray([r.unders_spread_n for r in runs], dtype=np.float64)
    all_f1_root = np.asarray([r.flip_r1_root_mae_n for r in runs], dtype=np.float64)
    all_f1_merge = np.asarray([r.flip_r1_merge_mae_n for r in runs], dtype=np.float64)
    all_f1_spread = np.asarray([r.flip_r1_spread_n for r in runs], dtype=np.float64)
    all_f2_root = np.asarray([r.flip_r2_root_mae_n for r in runs], dtype=np.float64)
    all_f2_merge = np.asarray([r.flip_r2_merge_mae_n for r in runs], dtype=np.float64)
    all_f2_spread = np.asarray([r.flip_r2_spread_n for r in runs], dtype=np.float64)

    best_by_root = min(agg_rows, key=lambda x: x.root)
    best_by_score = min(agg_rows, key=lambda x: x.score)
    best_by_spread = min(agg_rows, key=lambda x: x.spread)
    # Avoid pathological "best root" picks where merge/spread are terrible.
    feasible = [g for g in agg_rows if float(g.merge) <= 0.10 and float(g.spread) <= 0.20]
    best_feasible_by_score: Optional[GroupAgg] = min(feasible, key=lambda x: x.score) if feasible else None

    # Helper to format a GroupAgg key.
    def _key_dict(g: GroupAgg) -> Dict[str, object]:
        return {name: val for name, val in zip(group_keys, g.key)}

    def _best_by_cell() -> Dict[Tuple[float, int], GroupAgg]:
        best: Dict[Tuple[float, int], GroupAgg] = {}
        for g in agg_rows:
            kd = _key_dict(g)
            cell = (float(kd["audit_fraction"]), int(kd["train_docs"]))
            cur = best.get(cell)
            if cur is None or float(g.score) < float(cur.score):
                best[cell] = g
        return best

    def _finite_median(arr: np.ndarray) -> float:
        xs = arr[np.isfinite(arr)]
        return float(np.median(xs)) if xs.size else float("nan")

    exact_root_med = _finite_median(all_exact_root)
    exact_merge_med = _finite_median(all_exact_merge)
    exact_spread_med = _finite_median(all_exact_spread)
    u_root_med = _finite_median(all_u_root)
    u_merge_med = _finite_median(all_u_merge)
    u_spread_med = _finite_median(all_u_spread)
    f1_root_med = _finite_median(all_f1_root)
    f1_merge_med = _finite_median(all_f1_merge)
    f1_spread_med = _finite_median(all_f1_spread)
    f2_root_med = _finite_median(all_f2_root)
    f2_merge_med = _finite_median(all_f2_merge)
    f2_spread_med = _finite_median(all_f2_spread)

    # Figure 1: metric distributions by the main regularization axis.
    def _dist_by_regularization(metric: str) -> Path:
        vals: Dict[float, List[float]] = {}
        for r in runs:
            reg_value = float(
                r.local_law_weight if reg_axis_name == "local_law_weight" else r.schedule_consistency_weight
            )
            vals.setdefault(reg_value, []).append(float(getattr(r, metric)))
        reg_sorted = sorted(vals.keys())
        data = [vals[x] for x in reg_sorted]
        fig, ax = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
        ax.boxplot(data, showfliers=False)
        ax.set_xticklabels([str(x) for x in reg_sorted])
        ax.set_xlabel(reg_axis_name)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} distribution by {reg_axis_name} ({'normalized' if args.normalize else 'raw'})")
        out = out_dir / f"box_{metric}_by_{reg_axis_name}.png"
        fig.savefig(out, dpi=220)
        plt.close(fig)
        return out

    fig_box_root = _dist_by_regularization("learned_root_mae_n")
    fig_box_spread = _dist_by_regularization("learned_spread_n")
    fig_box_merge = _dist_by_regularization("learned_merge_mae_n")

    # Figure 2: best-by-score heatmaps (root/merge/spread).
    def _best_cell_heatmaps() -> Path:
        best = _best_by_cell()

        x = train_docs_vals
        y = budget_vals
        mat_root = np.full((len(y), len(x)), np.nan, dtype=np.float64)
        mat_merge = np.full((len(y), len(x)), np.nan, dtype=np.float64)
        mat_spread = np.full((len(y), len(x)), np.nan, dtype=np.float64)
        for yi, b in enumerate(y):
            for xi, td in enumerate(x):
                g = best.get((float(b), int(td)))
                if g is None:
                    continue
                mat_root[yi, xi] = float(g.root)
                mat_merge[yi, xi] = float(g.merge)
                mat_spread[yi, xi] = float(g.spread)

        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
        _heatmap(axes[0], mat_root, xlabels=[str(v) for v in x], ylabels=[str(v) for v in y], title="Best score | root MAE")
        _heatmap(axes[1], mat_merge, xlabels=[str(v) for v in x], ylabels=[str(v) for v in y], title="Best score | merge MAE")
        _heatmap(axes[2], mat_spread, xlabels=[str(v) for v in x], ylabels=[str(v) for v in y], title="Best score | schedule spread")
        fig.suptitle(
            f"Best-by-score across knobs | agg={agg} | score = root + {score_merge_w}*merge + {score_spread_w}*spread",
            fontsize=12,
        )
        out = out_dir / "grid_best_by_score.png"
        fig.savefig(out, dpi=220)
        plt.close(fig)
        return out

    fig_grid_best = _best_cell_heatmaps()

    # Figure 2b: best-by-score "gap to undersupported" (negative = better than undersupported).
    def _gap_to_undersupported_heatmaps() -> Path:
        best = _best_by_cell()

        x = train_docs_vals
        y = budget_vals
        mat_root = np.full((len(y), len(x)), np.nan, dtype=np.float64)
        mat_merge = np.full((len(y), len(x)), np.nan, dtype=np.float64)
        mat_spread = np.full((len(y), len(x)), np.nan, dtype=np.float64)
        for yi, b in enumerate(y):
            for xi, td in enumerate(x):
                g = best.get((float(b), int(td)))
                if g is None:
                    continue
                mat_root[yi, xi] = float(g.root) - float(g.unders_root)
                mat_merge[yi, xi] = float(g.merge) - float(g.unders_merge)
                # Undersupported is associative, so spread baseline is ~0.
                mat_spread[yi, xi] = float(g.spread) - 0.0

        def _sym_lim(m: np.ndarray) -> float:
            xs = m[np.isfinite(m)]
            if xs.size == 0:
                return 1.0
            return float(np.max(np.abs(xs)))

        lim = max(_sym_lim(mat_root), _sym_lim(mat_merge), _sym_lim(mat_spread))
        vmin, vmax = -lim, lim

        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
        _heatmap(
            axes[0],
            mat_root,
            xlabels=[str(v) for v in x],
            ylabels=[str(v) for v in y],
            title="(best score) root - undersupported",
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
        )
        _heatmap(
            axes[1],
            mat_merge,
            xlabels=[str(v) for v in x],
            ylabels=[str(v) for v in y],
            title="(best score) merge - undersupported",
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
        )
        _heatmap(
            axes[2],
            mat_spread,
            xlabels=[str(v) for v in x],
            ylabels=[str(v) for v in y],
            title="(best score) spread - 0",
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
        )
        fig.suptitle(
            f"Best-by-score gap to baselines (negative is better) | agg={agg}",
            fontsize=12,
        )
        out = out_dir / "grid_gap_to_undersupported.png"
        fig.savefig(out, dpi=220)
        plt.close(fig)
        return out

    fig_grid_gap = _gap_to_undersupported_heatmaps()

    # Figure 3: tradeoff scatter (group-level).
    def _tradeoff_scatter() -> Path:
        fig, ax = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)
        cmap = plt.get_cmap("viridis")
        reg_sorted = sorted(reg_axis_vals)
        reg_to_color = {v: cmap(i / max(1, len(reg_sorted) - 1)) for i, v in enumerate(reg_sorted)}
        markers = {"uniform": "o", "span_weighted": "s", "hybrid_top_span": "^", "top_span": "x"}
        for g in agg_rows:
            kd = _key_dict(g)
            reg_value = float(kd[reg_axis_name])
            c3 = str(kd["c3_audit_strategy"])
            ax.scatter(
                float(g.root),
                float(g.spread),
                s=22,
                c=[reg_to_color.get(reg_value, (0.2, 0.2, 0.2, 1.0))],
                marker=markers.get(c3, "o"),
                alpha=0.70,
                linewidths=0.0,
            )
        ax.set_xlabel("root MAE" + (" (normalized)" if args.normalize else ""))
        ax.set_ylabel("schedule spread" + (" (normalized)" if args.normalize else ""))
        ax.set_title("Root vs associativity tradeoff (group medians)")
        # Legends: scw colors (few entries) + marker meanings.
        for reg_value in reg_sorted:
            ax.scatter([], [], c=[reg_to_color[reg_value]], s=30, label=f"{reg_axis_name}={reg_value:g}")
        leg1 = ax.legend(title=reg_axis_name, loc="upper right", frameon=True)
        ax.add_artist(leg1)
        for c3, mk in markers.items():
            ax.scatter([], [], c="k", s=28, marker=mk, label=c3)
        ax.legend(title="c3_audit_strategy", loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=True)
        out = out_dir / "scatter_root_vs_spread.png"
        fig.savefig(out, dpi=220)
        plt.close(fig)
        return out

    fig_scatter = _tradeoff_scatter()

    # Figure 3b: root vs merge tradeoff (group-level).
    def _tradeoff_scatter_root_vs_merge() -> Path:
        fig, ax = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)
        cmap = plt.get_cmap("viridis")
        reg_sorted = sorted(reg_axis_vals)
        reg_to_color = {v: cmap(i / max(1, len(reg_sorted) - 1)) for i, v in enumerate(reg_sorted)}
        markers = {"uniform": "o", "span_weighted": "s", "hybrid_top_span": "^", "top_span": "x"}
        for g in agg_rows:
            kd = _key_dict(g)
            reg_value = float(kd[reg_axis_name])
            c3 = str(kd["c3_audit_strategy"])
            ax.scatter(
                float(g.root),
                float(g.merge),
                s=22,
                c=[reg_to_color.get(reg_value, (0.2, 0.2, 0.2, 1.0))],
                marker=markers.get(c3, "o"),
                alpha=0.70,
                linewidths=0.0,
            )
        ax.set_xlabel("root MAE" + (" (normalized)" if args.normalize else ""))
        ax.set_ylabel("merge MAE" + (" (normalized)" if args.normalize else ""))
        ax.set_title("Root vs merge tradeoff (group medians)")
        for reg_value in reg_sorted:
            ax.scatter([], [], c=[reg_to_color[reg_value]], s=30, label=f"{reg_axis_name}={reg_value:g}")
        leg1 = ax.legend(title=reg_axis_name, loc="upper right", frameon=True)
        ax.add_artist(leg1)
        for c3, mk in markers.items():
            ax.scatter([], [], c="k", s=28, marker=mk, label=c3)
        ax.legend(title="c3_audit_strategy", loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=True)
        out = out_dir / "scatter_root_vs_merge.png"
        fig.savefig(out, dpi=220)
        plt.close(fig)
        return out

    fig_scatter_root_merge = _tradeoff_scatter_root_vs_merge()

    # Figure 4: best-by-score learning curves (one line per budget).
    def _best_by_score_lines() -> Path:
        best = _best_by_cell()
        x = train_docs_vals
        y = budget_vals

        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
        titles = ["Root MAE", "Merge MAE", "Schedule spread"]
        keys = ["root", "merge", "spread"]
        cmap = plt.get_cmap("viridis")
        colors = {b: cmap(i / max(1, len(y) - 1)) for i, b in enumerate(y)}
        for ax, title, key in zip(axes, titles, keys):
            for b in y:
                ys: List[float] = []
                for td in x:
                    g = best.get((float(b), int(td)))
                    ys.append(float(getattr(g, key)) if g is not None else float("nan"))
                ax.plot(x, ys, marker="o", linewidth=2, color=colors[b], label=f"budget={b:g}")
            if key == "root":
                if np.isfinite(exact_root_med):
                    ax.axhline(
                        exact_root_med,
                        color="black",
                        linestyle="--",
                        linewidth=1.7,
                        alpha=0.85,
                        label="exact (oracle)",
                    )
                if np.isfinite(u_root_med):
                    ax.axhline(
                        u_root_med,
                        color="gray",
                        linestyle=":",
                        linewidth=2.0,
                        alpha=0.85,
                        label="undersupported",
                    )
                if np.isfinite(f1_root_med):
                    ax.axhline(
                        f1_root_med,
                        color="#c44e52",
                        linestyle="-.",
                        linewidth=1.7,
                        alpha=0.80,
                        label="flip_R1",
                    )
                if np.isfinite(f2_root_med):
                    ax.axhline(
                        f2_root_med,
                        color="#dd8452",
                        linestyle="-.",
                        linewidth=1.7,
                        alpha=0.80,
                        label="flip_R2",
                    )
            if key == "merge":
                if np.isfinite(exact_merge_med):
                    ax.axhline(
                        exact_merge_med,
                        color="black",
                        linestyle="--",
                        linewidth=1.7,
                        alpha=0.85,
                        label="exact (oracle)",
                    )
                if np.isfinite(u_merge_med):
                    ax.axhline(
                        u_merge_med,
                        color="gray",
                        linestyle=":",
                        linewidth=2.0,
                        alpha=0.85,
                        label="undersupported",
                    )
                if np.isfinite(f1_merge_med):
                    ax.axhline(
                        f1_merge_med,
                        color="#c44e52",
                        linestyle="-.",
                        linewidth=1.7,
                        alpha=0.80,
                        label="flip_R1",
                    )
                if np.isfinite(f2_merge_med):
                    ax.axhline(
                        f2_merge_med,
                        color="#dd8452",
                        linestyle="-.",
                        linewidth=1.7,
                        alpha=0.80,
                        label="flip_R2",
                    )
            if key == "spread":
                ax.axhline(0.0, color="black", linestyle="--", linewidth=1.5, alpha=0.75, label="associative (=0)")
                # Only draw non-trivial spread baselines (exact/undersupported are typically 0).
                if np.isfinite(f1_spread_med) and abs(float(f1_spread_med)) > 1e-9:
                    ax.axhline(
                        f1_spread_med,
                        color="#c44e52",
                        linestyle="-.",
                        linewidth=1.7,
                        alpha=0.80,
                        label="flip_R1",
                    )
                if np.isfinite(f2_spread_med) and abs(float(f2_spread_med)) > 1e-9:
                    ax.axhline(
                        f2_spread_med,
                        color="#dd8452",
                        linestyle="-.",
                        linewidth=1.7,
                        alpha=0.80,
                        label="flip_R2",
                    )
            ax.set_xlabel("train_docs")
            ax.set_ylabel(key + (" (normalized)" if args.normalize else ""))
            ax.set_title(f"Best-by-score | {title}")
            ax.grid(True, alpha=0.25)
        axes[0].legend(loc="upper right", fontsize=9)
        fig.suptitle(
            f"Learning curves using best-by-score config in each (train_docs,budget) cell | score = root + {score_merge_w}*merge + {score_spread_w}*spread",
            fontsize=12,
        )
        out = out_dir / "lines_best_by_score.png"
        fig.savefig(out, dpi=220)
        plt.close(fig)
        return out

    fig_lines_best = _best_by_score_lines()

    # Figure 4: low-budget c3 strategy comparison (shows failure modes).
    def _lines_low_budget() -> Tuple[Path, Path]:
        budget = 0.05
        llw = 1.0 if any(_is_close(v, 1.0) for v in llw_vals) else (max(llw_vals) if llw_vals else 0.0)
        scw = 0.2 if any(_is_close(v, 0.2) for v in scw_vals) else (max(scw_vals) if scw_vals else 0.0)
        lqr = 1.0
        # Plot medians across seeds for each (c3, train_docs) at fixed budget/scw/lqr.
        sub = [
            g
            for g in agg_rows
            if _is_close(_key_dict(g)["audit_fraction"], budget)
            and _is_close(_key_dict(g)["local_law_weight"], llw)
            and _is_close(_key_dict(g)["schedule_consistency_weight"], scw)
            and _is_close(_key_dict(g)["leaf_query_rate"], lqr)
        ]
        if not sub:
            return (out_dir / "lines_low_budget_merge.png", out_dir / "lines_low_budget_spread.png")
        by_c3: Dict[str, Dict[int, GroupAgg]] = {}
        for g in sub:
            kd = _key_dict(g)
            by_c3.setdefault(str(kd["c3_audit_strategy"]), {})[int(kd["train_docs"])] = g
        x = train_docs_vals
        fig1, ax1 = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
        fig2, ax2 = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
        for c3 in sorted(by_c3.keys()):
            ys_merge = [float(by_c3[c3][td].merge) if td in by_c3[c3] else float("nan") for td in x]
            ys_spread = [float(by_c3[c3][td].spread) if td in by_c3[c3] else float("nan") for td in x]
            ax1.plot(x, ys_merge, marker="o", linewidth=2, label=c3)
            ax2.plot(x, ys_spread, marker="o", linewidth=2, label=c3)
        ax1.set_xlabel("train_docs")
        ax1.set_ylabel("merge MAE" + (" (normalized)" if args.normalize else ""))
        ax1.set_title(
            f"Low budget comparison | audit_fraction={budget:g}, local_law_weight={llw:g}, scw={scw:g}, leaf_query_rate={lqr:g}"
        )
        ax1.grid(True, alpha=0.25)
        ax1.legend()
        ax2.set_xlabel("train_docs")
        ax2.set_ylabel("schedule spread" + (" (normalized)" if args.normalize else ""))
        ax2.set_title(
            f"Low budget comparison | audit_fraction={budget:g}, local_law_weight={llw:g}, scw={scw:g}, leaf_query_rate={lqr:g}"
        )
        ax2.grid(True, alpha=0.25)
        ax2.legend()
        out1 = out_dir / "lines_low_budget_merge.png"
        out2 = out_dir / "lines_low_budget_spread.png"
        fig1.savefig(out1, dpi=220)
        fig2.savefig(out2, dpi=220)
        plt.close(fig1)
        plt.close(fig2)
        return out1, out2

    fig_low_merge, fig_low_spread = _lines_low_budget()

    # Emit stats JSON + markdown summary.
    summary = {
        "generated_at_utc": now,
        "input_root": str(input_root),
        "n_files": int(len(files)),
        "n_runs_loaded": int(len(runs)),
        "levels": {
            "train_docs": train_docs_vals,
            "audit_fraction": budget_vals,
            "c3_audit_strategy": c3_vals,
            "leaf_query_rate": lqr_vals,
            "local_law_weight": llw_vals,
            "schedule_consistency_weight": scw_vals,
        },
        "regularization_axis": {"name": reg_axis_name, "values": reg_axis_vals},
        "global": {
            "learned_root_mae_n_median": float(np.median(all_root)),
            "learned_root_mae_n_p10": float(np.percentile(all_root, 10)),
            "learned_root_mae_n_p90": float(np.percentile(all_root, 90)),
            "learned_merge_mae_n_median": float(np.median(all_merge)),
            "learned_spread_n_median": float(np.median(all_spread)),
            "exact_root_mae_n_median": float(exact_root_med),
            "exact_merge_mae_n_median": float(exact_merge_med),
            "exact_spread_n_median": float(exact_spread_med),
            "undersupported_root_mae_n_median": float(np.median(all_u_root)),
            "undersupported_merge_mae_n_median": float(np.median(all_u_merge)),
            "undersupported_spread_n_median": float(u_spread_med),
            "flip_R1_root_mae_n_median": float(f1_root_med),
            "flip_R1_merge_mae_n_median": float(f1_merge_med),
            "flip_R1_spread_n_median": float(f1_spread_med),
            "flip_R2_root_mae_n_median": float(f2_root_med),
            "flip_R2_merge_mae_n_median": float(f2_merge_med),
            "flip_R2_spread_n_median": float(f2_spread_med),
            "learned_beats_undersupported_root_rate": float(np.mean(all_root < all_u_root)),
            "learned_beats_undersupported_merge_rate": float(np.mean(all_merge < all_u_merge)),
        },
        "best": {
            "by_root": {**_key_dict(best_by_root), "root": best_by_root.root, "merge": best_by_root.merge, "spread": best_by_root.spread, "score": best_by_root.score},
            "by_spread": {**_key_dict(best_by_spread), "root": best_by_spread.root, "merge": best_by_spread.merge, "spread": best_by_spread.spread, "score": best_by_spread.score},
            "by_score": {**_key_dict(best_by_score), "root": best_by_score.root, "merge": best_by_score.merge, "spread": best_by_score.spread, "score": best_by_score.score},
            "by_score_feasible": (
                {**_key_dict(best_feasible_by_score), "root": best_feasible_by_score.root, "merge": best_feasible_by_score.merge, "spread": best_feasible_by_score.spread, "score": best_feasible_by_score.score}
                if best_feasible_by_score is not None
                else None
            ),
        },
        "score": {"merge_weight": score_merge_w, "spread_weight": score_spread_w, "aggregate": agg, "normalize": bool(args.normalize)},
        "figures": {
            f"box_root_by_{reg_axis_name}": str(fig_box_root),
            f"box_merge_by_{reg_axis_name}": str(fig_box_merge),
            f"box_spread_by_{reg_axis_name}": str(fig_box_spread),
            "grid_best_by_score": str(fig_grid_best),
            "grid_gap_to_undersupported": str(fig_grid_gap),
            "scatter_root_vs_spread": str(fig_scatter),
            "scatter_root_vs_merge": str(fig_scatter_root_merge),
            "lines_best_by_score": str(fig_lines_best),
            "lines_low_budget_merge": str(fig_low_merge),
            "lines_low_budget_spread": str(fig_low_spread),
        },
    }
    (out_dir / "summary_stats.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    md_lines: List[str] = []
    md_lines.append("# Markov OPS Count Sweep Report")
    md_lines.append("")
    md_lines.append(f"- generated_at_utc: `{now}`")
    md_lines.append(f"- input_root: `{input_root}`")
    md_lines.append(f"- runs_loaded: `{len(runs)}` (files: `{len(files)}`)")
    md_lines.append(f"- aggregate: `{agg}` | normalized: `{bool(args.normalize)}`")
    md_lines.append("")
    md_lines.append("## Key Findings")
    md_lines.append("")
    md_lines.append("- `spread` here is *schedule spread*: for each doc, we merge leaves in 3 different orders and compute `max(pred_root)-min(pred_root)`. Nonzero spread means the learned merge is not associative (tree-shape dependent).")
    if len(llw_vals) > 1:
        md_lines.append("- `local_law_weight` is now a first-class sweep axis: `0` is the no-local-law baseline, and larger values increase theorem-facing C1/C3 supervision.")
    md_lines.append("- Spread can increase with more training docs if the training objective does not strongly enforce associativity: the model can fit leaf/root losses better while becoming more schedule-dependent. `schedule_consistency_weight` remains a separate proxy-only lever.")
    if len(scw_vals) > 1:
        md_lines.append("- `schedule_consistency_weight` still controls the proxy associativity penalty, but it is no longer the only regularization axis in this sweep.")
    md_lines.append("- Learned sketch beats `undersupported` on root MAE in essentially all settings in this sweep, but does not beat it on median merge MAE (merge supervision is still the bottleneck).")
    md_lines.append("- `c3_audit_strategy=hybrid_top_span` (with `c3_include_root=true`) is unsafe at the smallest internal-node budgets (`audit_fraction=0.05`): it under-labels small merges and can explode *average* merge MAE.")
    md_lines.append("- Baselines: `exact` is the oracle (uses endpoints so it can add the join indicator); `undersupported` is the associative-but-biased count-only merge; `flip_R1/R2` are controlled non-mergeable summaries used as stress tests.")
    md_lines.append("")
    md_lines.append("## Global Summary (medians)")
    md_lines.append("")
    md_lines.append(f"- units: `{'normalized by (max_segments-1)' if bool(args.normalize) else 'raw counts'}`")
    md_lines.append(f"- learned: root `{summary['global']['learned_root_mae_n_median']:.4f}` (p10 `{summary['global']['learned_root_mae_n_p10']:.4f}`, p90 `{summary['global']['learned_root_mae_n_p90']:.4f}`) | merge `{summary['global']['learned_merge_mae_n_median']:.4f}` | spread `{summary['global']['learned_spread_n_median']:.4f}`")
    md_lines.append(f"- exact (oracle): root `{summary['global']['exact_root_mae_n_median']:.4f}` | merge `{summary['global']['exact_merge_mae_n_median']:.4f}` | spread `{summary['global']['exact_spread_n_median']:.4f}`")
    md_lines.append(f"- undersupported: root `{summary['global']['undersupported_root_mae_n_median']:.4f}` | merge `{summary['global']['undersupported_merge_mae_n_median']:.4f}` | spread `{summary['global']['undersupported_spread_n_median']:.4f}`")
    md_lines.append(f"- flip_R1: root `{summary['global']['flip_R1_root_mae_n_median']:.4f}` | merge `{summary['global']['flip_R1_merge_mae_n_median']:.4f}` | spread `{summary['global']['flip_R1_spread_n_median']:.4f}`")
    md_lines.append(f"- flip_R2: root `{summary['global']['flip_R2_root_mae_n_median']:.4f}` | merge `{summary['global']['flip_R2_merge_mae_n_median']:.4f}` | spread `{summary['global']['flip_R2_spread_n_median']:.4f}`")
    md_lines.append("")
    md_lines.append("## Best Configs (group medians)")
    md_lines.append("")
    for name in ("by_score", "by_score_feasible", "by_root", "by_spread"):
        b = summary["best"][name]
        if b is None:
            md_lines.append(f"- {name}: n/a")
            continue
        md_lines.append(f"- {name}: root `{b['root']:.4f}` | merge `{b['merge']:.4f}` | spread `{b['spread']:.4f}` | score `{b['score']:.4f}` | c3 `{b['c3_audit_strategy']}` | llw `{b['local_law_weight']}` | scw `{b['schedule_consistency_weight']}` | lqr `{b['leaf_query_rate']}` | budget `{b['audit_fraction']}` | train_docs `{b['train_docs']}`")
    md_lines.append("")
    md_lines.append("## Artifacts")
    md_lines.append("")
    md_lines.append(f"- PDF: `{out_dir / 'report.pdf'}`")
    for k, v in summary["figures"].items():
        md_lines.append(f"- {k}: `{v}`")
    (out_dir / "report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Build a compact PDF with the key figures.
    pdf_path = out_dir / "report.pdf"
    with PdfPages(pdf_path) as pdf:
        _write_text_page(
            pdf,
            title="Markov OPS Count Sweep Report",
            lines=[
                f"generated_at_utc: {now}",
                f"input_root: {input_root}",
                f"runs_loaded: {len(runs)} (files={len(files)})",
                f"aggregate: {agg} | normalized: {bool(args.normalize)}",
                "",
                "Key findings:",
                "- spread = max(pred_root across schedules) - min(pred_root across schedules)",
                f"- primary regularization axis in this report: {reg_axis_name}",
                "- local_law_weight=0 is the no-local-law baseline when present",
                "- learned root MAE dominates undersupported across this sweep",
                "- merge MAE remains worse than undersupported on average",
                "- hybrid_top_span is unsafe at audit_fraction=0.05 (c3root=1)",
                "",
                "Global medians:",
                f"- units: {'normalized by (max_segments-1)' if bool(args.normalize) else 'raw counts'}",
                f"- learned: root {summary['global']['learned_root_mae_n_median']:.4f} | merge {summary['global']['learned_merge_mae_n_median']:.4f} | spread {summary['global']['learned_spread_n_median']:.4f}",
                f"- exact:   root {summary['global']['exact_root_mae_n_median']:.4f} | merge {summary['global']['exact_merge_mae_n_median']:.4f} | spread {summary['global']['exact_spread_n_median']:.4f}",
                f"- unders:  root {summary['global']['undersupported_root_mae_n_median']:.4f} | merge {summary['global']['undersupported_merge_mae_n_median']:.4f} | spread {summary['global']['undersupported_spread_n_median']:.4f}",
                f"- flip_R1: root {summary['global']['flip_R1_root_mae_n_median']:.4f} | merge {summary['global']['flip_R1_merge_mae_n_median']:.4f} | spread {summary['global']['flip_R1_spread_n_median']:.4f}",
                f"- flip_R2: root {summary['global']['flip_R2_root_mae_n_median']:.4f} | merge {summary['global']['flip_R2_merge_mae_n_median']:.4f} | spread {summary['global']['flip_R2_spread_n_median']:.4f}",
                "",
                "Best-by-score config (group medians):",
                json.dumps(summary['best']['by_score'], indent=2, sort_keys=True),
                "",
                "Best-by-score subject to merge<=0.10 and spread<=0.20:",
                json.dumps(summary['best']['by_score_feasible'], indent=2, sort_keys=True),
            ],
        )
        for fig_path in [
            fig_box_root,
            fig_box_spread,
            fig_box_merge,
            fig_grid_best,
            fig_grid_gap,
            fig_scatter,
            fig_scatter_root_merge,
            fig_lines_best,
            fig_low_merge,
            fig_low_spread,
        ]:
            if not Path(fig_path).exists():
                continue
            img = plt.imread(str(fig_path))
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
            ax.axis("off")
            ax.imshow(img)
            pdf.savefig(fig)
            plt.close(fig)

    print(json.dumps({"output_dir": str(out_dir), "pdf": str(pdf_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
