#!/usr/bin/env python3
"""Build a curated simulation-intent report with plots and an optional PDF."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.util import safe_float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build curated simulation-intent report with plots.")
    p.add_argument("--output-root", type=Path, required=True, help="Curated simulation root.")
    p.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Markdown report path (default: <output-root>/figures/simulation_intent_report.md).",
    )
    p.add_argument(
        "--output-pdf",
        type=Path,
        default=None,
        help="PDF report path (default: same stem as markdown).",
    )
    p.add_argument("--emit-pdf", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _run_pandoc(md_path: Path, pdf_path: Path) -> bool:
    if shutil.which("pandoc") is None or shutil.which("pdflatex") is None:
        return False
    subprocess.run(
        [
            "pandoc",
            str(md_path.name),
            "-o",
            str(pdf_path.name),
            "--pdf-engine=pdflatex",
        ],
        cwd=str(md_path.parent),
        check=True,
    )
    return True


def _fmt_count(x: object) -> str:
    try:
        return str(int(x))  # type: ignore[arg-type]
    except Exception:
        return "0"


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _find_exact_utility_artifacts(output_root: Path) -> Optional[Dict[str, Path]]:
    candidates = [
        output_root,
        output_root / "exact_utility_transport_suite",
        output_root / "utility_transport",
    ]
    for root in candidates:
        summary = root / "utility_transport_summary.json"
        report_md = root / "utility_transport_report.md"
        figure = root / "figures" / "utility_transport_suite.png"
        if summary.exists():
            return {
                "root": root,
                "summary": summary,
                "report_md": report_md,
                "figure": figure,
            }
    return None


def _ensure_expectations(output_root: Path) -> Path:
    out_json = output_root / "simulation_expectations.json"
    out_md = output_root / "simulation_expectations.md"
    if out_json.exists() and out_md.exists():
        return out_json
    _run(
        [
            sys.executable,
            "scripts/check_simulation_expectations.py",
            "--output-root",
            str(output_root),
            "--output-json",
            str(out_json),
            "--output-markdown",
            str(out_md),
        ]
    )
    return out_json


_safe_float = safe_float


def _median(values: Sequence[float]) -> float:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if arr.size == 0:
        return math.nan
    return float(np.median(arr))


def _normalized_gap(observed: object, good: object, bad: object) -> float:
    obs = _safe_float(observed)
    good_v = _safe_float(good)
    bad_v = _safe_float(bad)
    if not all(math.isfinite(v) for v in [obs, good_v, bad_v]):
        return math.nan
    if bad_v <= good_v + 1e-12:
        return 0.0 if obs <= good_v + 1e-12 else 1.0
    return float(min(1.0, max(0.0, (obs - good_v) / (bad_v - good_v))))


def _utility_recovery_fraction(observed: object, good: object, bad: object) -> float:
    obs = _safe_float(observed)
    good_v = _safe_float(good)
    bad_v = _safe_float(bad)
    if not all(math.isfinite(v) for v in [obs, good_v, bad_v]):
        return math.nan
    if bad_v <= good_v + 1e-12:
        return 1.0 if obs <= good_v + 1e-12 else 0.0
    raw = (bad_v - obs) / (bad_v - good_v)
    return float(min(1.0, max(0.0, raw)))


def _build_grid(
    rows: Iterable[Tuple[float, float, float]],
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> np.ndarray:
    grid = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    grouped: Dict[Tuple[float, float], List[float]] = defaultdict(list)
    for x_val, y_val, score in rows:
        grouped[(float(x_val), float(y_val))].append(float(score))
    for y_idx, y_val in enumerate(y_values):
        for x_idx, x_val in enumerate(x_values):
            values = grouped.get((float(x_val), float(y_val)), [])
            if values:
                grid[y_idx, x_idx] = _median(values)
    return grid


def _format_axis_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _format_metric_value(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    if value == 0.0:
        return "0"
    if abs(value) >= 1e-2:
        return f"{value:.2f}"
    return f"{value:.1e}"


def _draw_heatmap(
    ax: plt.Axes,
    grid: np.ndarray,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    *,
    title: str,
    x_label: str,
    y_label: str,
    cmap,
) -> None:
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for y_idx in range(grid.shape[0]):
        for x_idx in range(grid.shape[1]):
            value = grid[y_idx, x_idx]
            if math.isnan(float(value)):
                text = "NA"
                color = "black"
            else:
                text = f"{float(value):.2f}"
                color = "white" if value <= 0.20 or value >= 0.80 else "black"
            ax.text(x_idx, y_idx, text, ha="center", va="center", fontsize=8, color=color)
    return im


def _draw_metric_heatmap(
    ax: plt.Axes,
    grid: np.ndarray,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    *,
    title: str,
    x_label: str,
    y_label: str,
    cmap,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    fmt: str = "{:.2f}",
):
    finite = np.asarray(grid[np.isfinite(grid)], dtype=float)
    lo = float(np.min(finite)) if finite.size and vmin is None else float(vmin or 0.0)
    hi = float(np.max(finite)) if finite.size and vmax is None else float(vmax or 1.0)
    if not math.isfinite(lo):
        lo = 0.0
    if not math.isfinite(hi) or hi <= lo + 1e-12:
        hi = lo + 1.0

    im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=lo, vmax=hi)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    span = max(hi - lo, 1e-12)
    for y_idx in range(grid.shape[0]):
        for x_idx in range(grid.shape[1]):
            value = grid[y_idx, x_idx]
            if math.isnan(float(value)):
                text = "NA"
                color = "black"
            else:
                rel = (float(value) - lo) / span
                text = fmt.format(float(value))
                color = "white" if rel <= 0.20 or rel >= 0.80 else "black"
            ax.text(x_idx, y_idx, text, ha="center", va="center", fontsize=8, color=color)
    return im


def _status_code(status: str) -> float:
    return {"pass": 0.0, "warn": 1.0, "fail": 2.0}.get(status, math.nan)


def _worst_status(statuses: Iterable[str]) -> str:
    rank = {"pass": 0, "warn": 1, "fail": 2}
    filtered = [s for s in statuses if s in rank]
    if not filtered:
        return "not_applicable"
    return max(filtered, key=lambda item: rank[item])


def _category_label(kind: str) -> str:
    return {
        "ceiling": "Ceiling",
        "budget_trend": "Support",
        "failure_mode": "Mismatch",
        "granularity": "Granularity",
    }[kind]


def _family_label(family: str) -> str:
    return {
        "markov_ops_count": "Markov",
        "segment_lda_ops_weight_recovery": "Segment-LDA",
        "segmented_lda_ctreepo": "Segmented-LDA C-TreePO",
        "mergeable_ablation": "Mergeable",
    }[family]


def _plot_scorecard(figures: Path, expectations: Sequence[Dict[str, object]]) -> str:
    family_order = [
        "markov_ops_count",
        "segment_lda_ops_weight_recovery",
        "segmented_lda_ctreepo",
        "mergeable_ablation",
    ]
    kind_order = ["ceiling", "budget_trend", "failure_mode", "granularity"]
    grouped: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for rec in expectations:
        grouped[(str(rec.get("family")), str(rec.get("kind")))].append(str(rec.get("status")))

    matrix = np.full((len(family_order), len(kind_order)), np.nan, dtype=float)
    labels: List[List[str]] = [["" for _ in kind_order] for _ in family_order]
    for f_idx, family in enumerate(family_order):
        for k_idx, kind in enumerate(kind_order):
            status = _worst_status(grouped.get((family, kind), []))
            if status == "not_applicable":
                continue
            matrix[f_idx, k_idx] = _status_code(status)
            labels[f_idx][k_idx] = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[status]

    cmap = ListedColormap(["#1a9850", "#fee08b", "#d73027"])
    cmap.set_bad("#d9d9d9")
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(figsize=(8.5, 3.8), constrained_layout=True)
    ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    ax.set_title("Expectation Scorecard", fontsize=13)
    ax.set_xticks(np.arange(len(kind_order)))
    ax.set_xticklabels([_category_label(kind) for kind in kind_order])
    ax.set_yticks(np.arange(len(family_order)))
    ax.set_yticklabels([_family_label(family) for family in family_order])
    ax.set_xticks(np.arange(-0.5, len(kind_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(family_order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for y_idx in range(matrix.shape[0]):
        for x_idx in range(matrix.shape[1]):
            if math.isnan(float(matrix[y_idx, x_idx])):
                ax.text(x_idx, y_idx, "N/A", ha="center", va="center", fontsize=9, color="black")
            else:
                ax.text(x_idx, y_idx, labels[y_idx][x_idx], ha="center", va="center", fontsize=9, color="black")
    path = figures / "simulation_intent_scorecard.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _markov_panel_rows(output_root: Path) -> Tuple[Dict[float, np.ndarray], List[float], List[float]]:
    grouped: Dict[float, List[Tuple[float, float, float]]] = defaultdict(list)
    x_values: set[float] = set()
    y_values: set[float] = set()
    for path in sorted((output_root / "markov").glob("*.json")):
        payload = _load_json(path)
        config = payload.get("config", {}) or {}
        metrics = payload.get("metrics", {}) or {}
        leaf = _safe_float(config.get("fixed_leaf_tokens"))
        train_docs = _safe_float(config.get("train_docs"))
        audit_fraction = _safe_float(config.get("audit_fraction"))
        score = _normalized_gap(
            (metrics.get("learned", {}) or {}).get("root_mae"),
            (metrics.get("exact", {}) or {}).get("root_mae"),
            (metrics.get("undersupported", {}) or {}).get("root_mae"),
        )
        if not all(math.isfinite(v) for v in [leaf, train_docs, audit_fraction, score]):
            continue
        grouped[leaf].append((train_docs, audit_fraction, score))
        x_values.add(train_docs)
        y_values.add(audit_fraction)
    x_sorted = sorted(x_values)
    y_sorted = sorted(y_values)
    return {leaf: _build_grid(rows, x_sorted, y_sorted) for leaf, rows in grouped.items()}, x_sorted, y_sorted


def _segment_panel_rows(output_root: Path) -> Tuple[Dict[float, np.ndarray], List[float], List[float]]:
    grouped: Dict[float, List[Tuple[float, float, float]]] = defaultdict(list)
    x_values: set[float] = set()
    y_values: set[float] = set()
    for path in sorted((output_root / "segment_lda").glob("*.json")):
        payload = _load_json(path)
        config = payload.get("config", {}) or {}
        metrics = payload.get("metrics", {}) or {}
        lam = _safe_float(config.get("lambda_multiplier"))
        train_docs = _safe_float(config.get("train_docs"))
        audit_fraction = _safe_float(config.get("audit_fraction"))
        score = _normalized_gap(
            (metrics.get("ridge", {}) or {}).get("root_mae"),
            (metrics.get("exact", {}) or {}).get("root_mae"),
            (metrics.get("undersupported", {}) or {}).get("root_mae"),
        )
        if not all(math.isfinite(v) for v in [lam, train_docs, audit_fraction, score]):
            continue
        grouped[lam].append((train_docs, audit_fraction, score))
        x_values.add(train_docs)
        y_values.add(audit_fraction)
    x_sorted = sorted(x_values)
    y_sorted = sorted(y_values)
    return {lam: _build_grid(rows, x_sorted, y_sorted) for lam, rows in grouped.items()}, x_sorted, y_sorted


def _ctree_panel_rows(output_root: Path) -> Tuple[Dict[float, np.ndarray], List[float], List[float]]:
    grouped: Dict[float, List[Tuple[float, float, float]]] = defaultdict(list)
    x_values: set[float] = set()
    y_values: set[float] = set()
    for path in sorted((output_root / "segmented_lda_ctreepo").glob("*.json")):
        payload = _load_json(path)
        config = payload.get("config", {}) or {}
        metrics = payload.get("metrics", {}) or {}
        leaf = _safe_float(config.get("fixed_leaf_tokens"))
        train_docs = _safe_float(config.get("n_books_train"))
        total_support = sum(
            _safe_float(config.get(key))
            for key in ["calibration_leaf_query_rate", "eval_leaf_query_rate", "eval_internal_query_rate"]
        )
        score = _normalized_gap(
            (metrics.get("estimated_calibrated_budgeted", {}) or {}).get("root_l1_mean"),
            (metrics.get("oracle_tree", {}) or {}).get("root_l1_mean"),
            (metrics.get("estimated_uncalibrated", {}) or {}).get("root_l1_mean"),
        )
        if not all(math.isfinite(v) for v in [leaf, train_docs, total_support, score]):
            continue
        grouped[leaf].append((train_docs, total_support, score))
        x_values.add(train_docs)
        y_values.add(total_support)
    x_sorted = sorted(x_values)
    y_sorted = sorted(y_values)
    return {leaf: _build_grid(rows, x_sorted, y_sorted) for leaf, rows in grouped.items()}, x_sorted, y_sorted


def _mergeable_panel(output_root: Path) -> Tuple[np.ndarray, List[float], List[float]]:
    payload = _load_json(output_root / "mergeable" / "chunk_quality_summary.json")
    rows = list(payload.get("rows", []) or [])
    one_pass = _safe_float(((payload.get("reference_rows", {}) or {}).get("one_pass_reference", {}) or {}).get("mean_abs_bias"))
    aligned_rows = [
        row
        for row in rows
        if str(row.get("method_name", "")).startswith("grid_fixed_")
        and bool(row.get("supports_target", False))
    ]
    worst = max((_safe_float(row.get("mean_abs_bias")) for row in aligned_rows), default=math.nan)
    grouped: List[Tuple[float, float, float]] = []
    x_values: set[float] = set()
    y_values: set[float] = set()
    for row in aligned_rows:
        chunk_budget = _safe_float(row.get("chunk_budget"))
        chunk_size = _safe_float(row.get("fixed_chunk_size"))
        score = _normalized_gap(row.get("mean_abs_bias"), one_pass, worst)
        if not all(math.isfinite(v) for v in [chunk_budget, chunk_size, score]):
            continue
        grouped.append((chunk_budget, chunk_size, score))
        x_values.add(chunk_budget)
        y_values.add(chunk_size)
    x_sorted = sorted(x_values)
    y_sorted = sorted(y_values)
    return _build_grid(grouped, x_sorted, y_sorted), x_sorted, y_sorted


def _bar_color(label: str) -> str:
    mapping = {
        "exact": "#1a9850",
        "oracle_tree": "#1a9850",
        "one_pass_m5": "#1a9850",
        "ridge_true_topics": "#4daf4a",
        "learned": "#66bd63",
        "ridge": "#66bd63",
        "estimated_calibrated_budgeted": "#66bd63",
        "full_model_m5": "#66bd63",
        "estimated_calibrated": "#a6d96a",
        "flip_R1": "#fdae61",
        "undersupported": "#d73027",
        "estimated_uncalibrated": "#d73027",
        "one_pass_m2": "#d73027",
        "full_model_m2": "#f46d43",
        "naive_majority": "#d73027",
        "naive_mean_of_means": "#a50026",
    }
    return mapping.get(label, "#7570b3")


def _plot_grouped_bars(
    ax: plt.Axes,
    categories: Sequence[str],
    series: Sequence[Tuple[str, Sequence[float]]],
    *,
    title: str,
    y_label: str,
) -> None:
    x = np.arange(len(categories), dtype=float)
    n = max(1, len(series))
    width = 0.78 / float(n)
    offsets = np.linspace(-0.39 + width / 2.0, 0.39 - width / 2.0, n)
    ymax = 0.0
    for offset, (name, values) in zip(offsets, series):
        vals = np.asarray([_safe_float(v) for v in values], dtype=float)
        ymax = max(ymax, float(np.nanmax(vals)) if vals.size else 0.0)
        ax.bar(x + offset, vals, width=width, label=name, color=_bar_color(name), edgecolor="black", linewidth=0.4)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(list(categories))
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)
    if ymax > 0.0 and math.isfinite(ymax):
        ax.set_ylim(0.0, ymax * 1.18)


def _median_at_max(
    rows: Sequence[Dict[str, object]],
    *,
    category_key: str,
    max_keys: Sequence[str],
    value_by_name: Dict[str, callable],
) -> Tuple[List[str], List[Tuple[str, List[float]]]]:
    categories = sorted({_safe_float(row.get(category_key)) for row in rows if math.isfinite(_safe_float(row.get(category_key)))})
    filtered = list(rows)
    for key in max_keys:
        vals = [_safe_float(row.get(key)) for row in filtered if math.isfinite(_safe_float(row.get(key)))]
        if vals:
            max_val = max(vals)
            filtered = [row for row in filtered if abs(_safe_float(row.get(key)) - max_val) <= 1e-12]
    series_out: List[Tuple[str, List[float]]] = []
    for name, extractor in value_by_name.items():
        per_category: List[float] = []
        for cat in categories:
            vals = [_safe_float(extractor(row)) for row in filtered if abs(_safe_float(row.get(category_key)) - cat) <= 1e-12]
            per_category.append(_median(vals))
        series_out.append((name, per_category))
    return [_format_axis_value(cat) for cat in categories], series_out


def _collect_markov_rows(output_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted((output_root / "markov").glob("*.json")):
        payload = _load_json(path)
        config = payload.get("config", {}) or {}
        metrics = payload.get("metrics", {}) or {}
        rows.append(
            {
                "leaf": _safe_float(config.get("fixed_leaf_tokens")),
                "train_docs": _safe_float(config.get("train_docs")),
                "audit_fraction": _safe_float(config.get("audit_fraction")),
                "exact": (metrics.get("exact", {}) or {}).get("root_mae"),
                "learned": (metrics.get("learned", {}) or {}).get("root_mae"),
                "undersupported": (metrics.get("undersupported", {}) or {}).get("root_mae"),
                "exact_merge": (metrics.get("exact", {}) or {}).get("merge_mae"),
                "learned_merge": (metrics.get("learned", {}) or {}).get("merge_mae"),
                "undersupported_merge": (metrics.get("undersupported", {}) or {}).get("merge_mae"),
            }
        )
    return rows


def _collect_markov_supervision_rows(output_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    root = output_root / "markov_supervision_narrative"
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*.json")):
        payload = _load_json(path)
        config = payload.get("config", {}) or {}
        metrics = payload.get("metrics", {}) or {}
        training_geometry = payload.get("training_geometry", {}) or {}
        mean_leaf_labels = _safe_float(training_geometry.get("mean_leaf_labels"))
        mean_internal_labels = _safe_float(training_geometry.get("mean_internal_labels"))
        mean_leaves = _safe_float(training_geometry.get("mean_leaves"))
        mean_internal_nodes = _safe_float(training_geometry.get("mean_internal_nodes"))
        total_local_nodes = mean_leaves + mean_internal_nodes
        if all(math.isfinite(v) for v in [mean_leaf_labels, mean_internal_labels, total_local_nodes]) and total_local_nodes > 0.0:
            local_oracle_coverage = (mean_leaf_labels + mean_internal_labels) / total_local_nodes
            local_undersupport = 1.0 - local_oracle_coverage
        else:
            local_oracle_coverage = math.nan
            local_undersupport = math.nan
        train_docs = _safe_float(config.get("train_docs"))
        root_queries_total = _safe_float(training_geometry.get("root_queries_total"))
        root_query_rate = math.nan
        if all(math.isfinite(v) for v in [root_queries_total, train_docs]) and train_docs > 0.0:
            root_query_rate = root_queries_total / train_docs
        rows.append(
            {
                "leaf": _safe_float(config.get("fixed_leaf_tokens")),
                "train_docs": train_docs,
                "model_family": str(config.get("model_family", "")),
                "regime": str(path.parent.name),
                "local_oracle_coverage": local_oracle_coverage,
                "local_undersupport": local_undersupport,
                "root_query_rate": root_query_rate,
                "exact_root": _safe_float((metrics.get("exact", {}) or {}).get("root_mae")),
                "exact_merge": _safe_float((metrics.get("exact", {}) or {}).get("merge_mae")),
                "exact_schedule": _safe_float((metrics.get("exact", {}) or {}).get("schedule_spread_mean")),
                "learned_root": _safe_float((metrics.get("learned", {}) or {}).get("root_mae")),
                "learned_merge": _safe_float((metrics.get("learned", {}) or {}).get("merge_mae")),
                "learned_schedule": _safe_float((metrics.get("learned", {}) or {}).get("schedule_spread_mean")),
                "undersupported_root": _safe_float((metrics.get("undersupported", {}) or {}).get("root_mae")),
                "undersupported_merge": _safe_float((metrics.get("undersupported", {}) or {}).get("merge_mae")),
                "undersupported_schedule": _safe_float((metrics.get("undersupported", {}) or {}).get("schedule_spread_mean")),
            }
        )
    return rows


def _collect_segment_rows_from_dir(
    input_dir: Path,
    *,
    topic_process: Optional[str] = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not input_dir.exists():
        return rows
    for path in sorted(input_dir.glob("*.json")):
        payload = _load_json(path)
        config = payload.get("config", {}) or {}
        if topic_process is not None and str(config.get("topic_process", "")) != str(topic_process):
            continue
        metrics = payload.get("metrics", {}) or {}
        topic_meta = payload.get("topic_meta", {}) or {}
        rows.append(
            {
                "lambda_multiplier": _safe_float(config.get("lambda_multiplier")),
                "train_docs": _safe_float(config.get("train_docs")),
                "audit_fraction": _safe_float(config.get("audit_fraction")),
                "leaf_tokens": _safe_float(config.get("leaf_tokens")),
                "topic_process": str(config.get("topic_process", "unknown")),
                "exact": (metrics.get("exact", {}) or {}).get("root_mae"),
                "ridge": (metrics.get("ridge", {}) or {}).get("root_mae"),
                "ridge_true_topics": (metrics.get("ridge_true_topics", {}) or {}).get("root_mae"),
                "undersupported": (metrics.get("undersupported", {}) or {}).get("root_mae"),
                "flip_R1": (metrics.get("flip_R1", {}) or {}).get("root_mae"),
                "ridge_merge_mae": (metrics.get("ridge", {}) or {}).get("merge_mae"),
                "ridge_leaf_accuracy_test": (metrics.get("ridge", {}) or {}).get("leaf_accuracy_test"),
                "test_mean_leaf_topic_purity": topic_meta.get("test_mean_leaf_topic_purity"),
                "mean_leaves": _safe_float((payload.get("training_geometry", {}) or {}).get("mean_leaves")),
                "mean_internal_nodes": _safe_float((payload.get("training_geometry", {}) or {}).get("mean_internal_nodes")),
            }
        )
    return rows


def _collect_segment_rows(output_root: Path) -> List[Dict[str, object]]:
    return _collect_segment_rows_from_dir(output_root / "segment_lda", topic_process="segments")


def _collect_lda_baseline_rows(output_root: Path) -> List[Dict[str, object]]:
    return _collect_segment_rows_from_dir(output_root / "segment_lda_lda_baseline", topic_process="bag_of_words")


def _collect_boundary_control_rows(output_root: Path) -> List[Dict[str, object]]:
    return _collect_segment_rows_from_dir(output_root / "segment_lda_boundary_controls")


def _median_for_control(
    rows: Sequence[Dict[str, object]],
    *,
    topic_process: str,
    leaf_tokens: int,
    lambda_multiplier: float,
    metric_key: str,
) -> float:
    subset = [
        row
        for row in rows
        if str(row.get("topic_process")) == str(topic_process)
        and abs(_safe_float(row.get("leaf_tokens")) - float(leaf_tokens)) <= 1e-12
        and abs(_safe_float(row.get("lambda_multiplier")) - float(lambda_multiplier)) <= 1e-12
    ]
    train_vals = [_safe_float(row.get("train_docs")) for row in subset if math.isfinite(_safe_float(row.get("train_docs")))]
    if not train_vals:
        return math.nan
    train_max = max(train_vals)
    vals = [
        _safe_float(row.get(metric_key))
        for row in subset
        if abs(_safe_float(row.get("train_docs")) - train_max) <= 1e-12
    ]
    return _median(vals)


def _segment_normalized_panels_from_rows(
    rows: Sequence[Dict[str, object]],
) -> Tuple[Dict[float, np.ndarray], List[float], List[float]]:
    fallback_bad_by_lam = _groupwise_max(
        rows,
        group_getter=lambda row: row.get("lambda_multiplier"),
        value_getter=lambda row: row.get("ridge"),
    )
    return _normalized_panels_from_rows(
        rows,
        observed_getter=lambda row: row.get("ridge"),
        good_getter=lambda row: row.get("exact"),
        bad_getter=lambda row: row.get("undersupported"),
        fallback_bad_by_group=fallback_bad_by_lam,
    )


def _segment_raw_panels_from_rows(
    rows: Sequence[Dict[str, object]],
    *,
    value_getter: Callable[[Dict[str, object]], object],
) -> Tuple[Dict[float, np.ndarray], List[float], List[float]]:
    grouped: Dict[float, List[Tuple[float, float, float]]] = defaultdict(list)
    x_values: set[float] = set()
    y_values: set[float] = set()
    for row in rows:
        lam = _safe_float(row.get("lambda_multiplier"))
        train_docs = _safe_float(row.get("train_docs"))
        audit_fraction = _safe_float(row.get("audit_fraction"))
        value = _safe_float(value_getter(row))
        if not all(math.isfinite(v) for v in [lam, train_docs, audit_fraction, value]):
            continue
        grouped[lam].append((train_docs, audit_fraction, value))
        x_values.add(train_docs)
        y_values.add(audit_fraction)
    x_sorted = sorted(x_values)
    y_sorted = sorted(y_values)
    return {lam: _build_grid(rows_lam, x_sorted, y_sorted) for lam, rows_lam in grouped.items()}, x_sorted, y_sorted


def _groupwise_max(
    rows: Sequence[Dict[str, object]],
    *,
    group_getter: Callable[[Dict[str, object]], object],
    value_getter: Callable[[Dict[str, object]], object],
) -> Dict[float, float]:
    maxima: Dict[float, float] = {}
    for row in rows:
        group_value = _safe_float(group_getter(row))
        value = _safe_float(value_getter(row))
        if not all(math.isfinite(v) for v in [group_value, value]):
            continue
        current = maxima.get(group_value, -math.inf)
        maxima[group_value] = max(current, value)
    return maxima


def _normalized_panels_from_rows(
    rows: Sequence[Dict[str, object]],
    *,
    observed_getter: Callable[[Dict[str, object]], object],
    good_getter: Callable[[Dict[str, object]], object],
    bad_getter: Optional[Callable[[Dict[str, object]], object]] = None,
    fallback_bad_by_group: Optional[Dict[float, float]] = None,
    force_group_bad: bool = False,
) -> Tuple[Dict[float, np.ndarray], List[float], List[float]]:
    grouped: Dict[float, List[Tuple[float, float, float]]] = defaultdict(list)
    x_values: set[float] = set()
    y_values: set[float] = set()
    for row in rows:
        lam = _safe_float(row.get("lambda_multiplier"))
        train_docs = _safe_float(row.get("train_docs"))
        audit_fraction = _safe_float(row.get("audit_fraction"))
        observed = _safe_float(observed_getter(row))
        good = _safe_float(good_getter(row))
        if not all(math.isfinite(v) for v in [lam, train_docs, audit_fraction, observed, good]):
            continue

        bad = math.nan
        if force_group_bad:
            if fallback_bad_by_group is not None:
                bad = _safe_float(fallback_bad_by_group.get(lam))
        else:
            if bad_getter is not None:
                bad = _safe_float(bad_getter(row))
            if (
                fallback_bad_by_group is not None
                and (not math.isfinite(bad) or bad <= good + 1e-12)
            ):
                bad = _safe_float(fallback_bad_by_group.get(lam))

        score = _normalized_gap(observed, good, bad)
        if not math.isfinite(score):
            continue
        grouped[lam].append((train_docs, audit_fraction, score))
        x_values.add(train_docs)
        y_values.add(audit_fraction)

    x_sorted = sorted(x_values)
    y_sorted = sorted(y_values)
    return {lam: _build_grid(rows_lam, x_sorted, y_sorted) for lam, rows_lam in grouped.items()}, x_sorted, y_sorted


def _collect_ctree_rows(output_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted((output_root / "segmented_lda_ctreepo").glob("*.json")):
        payload = _load_json(path)
        config = payload.get("config", {}) or {}
        metrics = payload.get("metrics", {}) or {}
        rows.append(
            {
                "leaf": _safe_float(config.get("fixed_leaf_tokens")),
                "n_books_train": _safe_float(config.get("n_books_train")),
                "total_query_rate": sum(
                    _safe_float(config.get(key))
                    for key in ["calibration_leaf_query_rate", "eval_leaf_query_rate", "eval_internal_query_rate"]
                ),
                "oracle_tree": (metrics.get("oracle_tree", {}) or {}).get("root_l1_mean"),
                "estimated_uncalibrated": (metrics.get("estimated_uncalibrated", {}) or {}).get("root_l1_mean"),
                "estimated_calibrated": (metrics.get("estimated_calibrated", {}) or {}).get("root_l1_mean"),
                "estimated_calibrated_budgeted": (metrics.get("estimated_calibrated_budgeted", {}) or {}).get("root_l1_mean"),
            }
        )
    return rows


def _plot_markov_family(output_root: Path, figures: Path) -> str:
    bar_rows = _collect_markov_rows(output_root)
    leaf_order = sorted({_safe_float(row.get("leaf")) for row in bar_rows if math.isfinite(_safe_float(row.get("leaf")))})
    if not leaf_order:
        raise ValueError("No Markov rows found for report")
    train_max = max(_safe_float(row.get("train_docs")) for row in bar_rows if math.isfinite(_safe_float(row.get("train_docs"))))
    audit_max = max(_safe_float(row.get("audit_fraction")) for row in bar_rows if math.isfinite(_safe_float(row.get("audit_fraction"))))
    eps = 1e-8
    colors = {"exact": "#1a9850", "learned": "#66bd63", "undersupported": "#d73027"}

    fig = plt.figure(figsize=(14.5, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(len(leaf_order), 3)
    for row_idx, leaf in enumerate(leaf_order):
        leaf_rows = [row for row in bar_rows if abs(_safe_float(row.get("leaf")) - leaf) <= 1e-12]
        audit_x = sorted(
            {
                _safe_float(row.get("audit_fraction"))
                for row in leaf_rows
                if abs(_safe_float(row.get("train_docs")) - train_max) <= 1e-12
                and math.isfinite(_safe_float(row.get("audit_fraction")))
            }
        )
        train_x = sorted(
            {
                _safe_float(row.get("train_docs"))
                for row in leaf_rows
                if abs(_safe_float(row.get("audit_fraction")) - audit_max) <= 1e-12
                and math.isfinite(_safe_float(row.get("train_docs")))
            }
        )

        panels = [
            (
                fig.add_subplot(gs[row_idx, 0]),
                "root_mae vs audit_fraction",
                audit_x,
                lambda row: _safe_float(row.get("audit_fraction")),
                {"exact": "exact", "learned": "learned", "undersupported": "undersupported"},
            ),
            (
                fig.add_subplot(gs[row_idx, 1]),
                "root_mae vs train_docs",
                train_x,
                lambda row: _safe_float(row.get("train_docs")),
                {"exact": "exact", "learned": "learned", "undersupported": "undersupported"},
            ),
            (
                fig.add_subplot(gs[row_idx, 2]),
                "merge_mae vs audit_fraction",
                audit_x,
                lambda row: _safe_float(row.get("audit_fraction")),
                {"exact": "exact_merge", "learned": "learned_merge", "undersupported": "undersupported_merge"},
            ),
        ]

        for ax, title, x_values, x_getter, key_map in panels:
            for name in ["exact", "learned", "undersupported"]:
                y_vals: List[float] = []
                for x in x_values:
                    subset = [
                        _safe_float(row.get(key_map[name]))
                        for row in leaf_rows
                        if abs(x_getter(row) - x) <= 1e-12
                        and (
                            abs(_safe_float(row.get("train_docs")) - train_max) <= 1e-12
                            if "audit_fraction" in title
                            else abs(_safe_float(row.get("audit_fraction")) - audit_max) <= 1e-12
                        )
                    ]
                    y_vals.append(max(eps, _median(subset)))
                ax.plot(
                    x_values,
                    y_vals,
                    marker="o",
                    linewidth=2.0,
                    color=colors[name],
                    label=name if row_idx == 0 else None,
                )
            ax.set_yscale("log")
            ax.grid(alpha=0.25, linewidth=0.8)
            ax.set_xlabel("audit_fraction" if "audit_fraction" in title else "train_docs")
            ax.set_ylabel(f"leaf={_format_axis_value(leaf)}\nMAE" if "root_mae" in title else f"leaf={_format_axis_value(leaf)}\nmerge MAE")
            if row_idx == 0:
                ax.set_title(title, fontsize=11)
        if row_idx == 0:
            panels[0][0].legend(frameon=False, fontsize=8)

    fig.suptitle("Markov support trends (log scale; lower is better)", fontsize=13)
    path = figures / "simulation_intent_markov_family.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _plot_markov_supervision_narrative(output_root: Path, figures: Path) -> Optional[str]:
    rows = _collect_markov_supervision_rows(output_root)
    if not rows:
        return None

    train_values = [_safe_float(row.get("train_docs")) for row in rows if math.isfinite(_safe_float(row.get("train_docs")))]
    if not train_values:
        return None
    train_target = max(train_values)
    rows = [row for row in rows if abs(_safe_float(row.get("train_docs")) - train_target) <= 1e-12]
    if not rows:
        return None

    family_leaf_order = [(family, leaf) for family in ["additive", "neural"] for leaf in [16.0, 32.0]]
    family_leaf_order = [
        (family, leaf)
        for family, leaf in family_leaf_order
        if any(str(row.get("model_family")) == family and abs(_safe_float(row.get("leaf")) - leaf) <= 1e-12 for row in rows)
    ]
    if not family_leaf_order:
        return None

    access_patterns = [
        ("Merge-access patterns", ["none", "sparse_merge", "full_merge", "full_local"]),
        ("Direct-access patterns", ["none", "root_only", "full_direct"]),
    ]
    metric_specs = [("Root utility recovery", "utility_recovery"), ("Merge MAE", "learned_merge")]
    colors = {"additive": "#1a9850", "neural": "#d73027"}
    linestyles = {16.0: "-", 32.0: "--"}
    eps = 1e-8

    fig = plt.figure(figsize=(15.0, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(len(metric_specs), len(access_patterns))
    for row_idx, (metric_title, metric_key) in enumerate(metric_specs):
        for col_idx, (panel_title, regime_order) in enumerate(access_patterns):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            for family, leaf in family_leaf_order:
                y_vals: List[float] = []
                for regime in regime_order:
                    subset_rows = [
                        row
                        for row in rows
                        if str(row.get("model_family")) == family
                        and abs(_safe_float(row.get("leaf")) - leaf) <= 1e-12
                        and str(row.get("regime")) == regime
                    ]
                    if metric_key == "utility_recovery":
                        subset = [
                            _utility_recovery_fraction(
                                row.get("learned_root"),
                                row.get("exact_root"),
                                row.get("undersupported_root"),
                            )
                            for row in subset_rows
                        ]
                        y_vals.append(_median(subset))
                    else:
                        subset = [_safe_float(row.get(metric_key)) for row in subset_rows]
                        y_vals.append(max(eps, _median(subset)))
                label = f"{family} leaf={_format_axis_value(leaf)}" if row_idx == 0 else None
                ax.plot(
                    np.arange(len(regime_order)),
                    y_vals,
                    marker="o",
                    linewidth=2.0,
                    color=colors[family],
                    linestyle=linestyles[leaf],
                    label=label,
                )
            if metric_key == "utility_recovery":
                ax.set_ylim(-0.02, 1.02)
                ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
                ax.axhline(1.0, color="black", linestyle=":", linewidth=1.0)
            else:
                ax.axhline(eps, color="black", linestyle=":", linewidth=1.0)
                ax.set_yscale("log")
            ax.grid(alpha=0.25, linewidth=0.8)
            ax.set_xticks(np.arange(len(regime_order)))
            ax.set_xticklabels([name.replace("_", "\n") for name in regime_order])
            ax.set_xlabel("oracle access pattern")
            ax.set_ylabel(metric_title)
            if row_idx == 0:
                ax.set_title(panel_title, fontsize=11)
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        (
            f"Markov oracle-access ablation at train_docs={_format_axis_value(train_target)} "
            "(top: utility recovery, bottom: merge fidelity)"
        ),
        fontsize=13,
    )
    path = figures / "simulation_intent_markov_supervision_family.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _plot_markov_support_recovery(output_root: Path, figures: Path) -> Optional[str]:
    rows = _collect_markov_supervision_rows(output_root)
    if not rows:
        return None
    train_values = [_safe_float(row.get("train_docs")) for row in rows if math.isfinite(_safe_float(row.get("train_docs")))]
    if not train_values:
        return None
    train_target = max(train_values)
    rows = [row for row in rows if abs(_safe_float(row.get("train_docs")) - train_target) <= 1e-12]
    merge_regimes = ["none", "sparse_merge", "full_merge", "full_local"]
    short_label = {"none": "N", "sparse_merge": "SM", "full_merge": "FM", "full_local": "FL"}
    combos = [(family, leaf) for family in ["additive", "neural"] for leaf in [16.0, 32.0]]
    combos = [
        (family, leaf)
        for family, leaf in combos
        if any(str(row.get("model_family")) == family and abs(_safe_float(row.get("leaf")) - leaf) <= 1e-12 for row in rows)
    ]
    if not combos:
        return None

    colors = {"additive": "#1a9850", "neural": "#d73027"}
    linestyles = {16.0: "-", 32.0: "--"}
    eps = 1e-8

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8), constrained_layout=True)
    for family, leaf in combos:
        subset = [
            row
            for row in rows
            if str(row.get("model_family")) == family and abs(_safe_float(row.get("leaf")) - leaf) <= 1e-12
        ]
        x_vals: List[float] = []
        y_utility: List[float] = []
        y_merge: List[float] = []
        labels: List[str] = []
        for regime in merge_regimes:
            regime_rows = [row for row in subset if str(row.get("regime")) == regime]
            if not regime_rows:
                continue
            coverage = _median([_safe_float(row.get("local_oracle_coverage")) for row in regime_rows])
            utility = _median(
                [
                    _utility_recovery_fraction(
                        row.get("learned_root"),
                        row.get("exact_root"),
                        row.get("undersupported_root"),
                    )
                    for row in regime_rows
                ]
            )
            merge_mae = _median([_safe_float(row.get("learned_merge")) for row in regime_rows])
            if not math.isfinite(coverage):
                continue
            x_vals.append(coverage)
            y_utility.append(utility)
            y_merge.append(max(eps, merge_mae))
            labels.append(short_label[regime])

        label = f"{family} leaf={_format_axis_value(leaf)}"
        axes[0].plot(
            x_vals,
            y_utility,
            marker="o",
            linewidth=2.0,
            color=colors[family],
            linestyle=linestyles[leaf],
            label=label,
        )
        axes[1].plot(
            x_vals,
            y_merge,
            marker="o",
            linewidth=2.0,
            color=colors[family],
            linestyle=linestyles[leaf],
            label=label,
        )
        for x, y, txt in zip(x_vals, y_utility, labels):
            axes[0].annotate(txt, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        for x, y, txt in zip(x_vals, y_merge, labels):
            axes[1].annotate(txt, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)

    axes[0].set_title("Root utility recovery vs local oracle coverage", fontsize=11)
    axes[0].set_xlabel("local oracle coverage")
    axes[0].set_ylabel("utility recovery")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].axhline(0.0, color="black", linestyle=":", linewidth=1.0)
    axes[0].axhline(1.0, color="black", linestyle=":", linewidth=1.0)
    axes[0].grid(alpha=0.25, linewidth=0.8)

    axes[1].set_title("Merge MAE vs local oracle coverage", fontsize=11)
    axes[1].set_xlabel("local oracle coverage")
    axes[1].set_ylabel("merge_mae")
    axes[1].set_yscale("log")
    axes[1].axhline(eps, color="black", linestyle=":", linewidth=1.0)
    axes[1].grid(alpha=0.25, linewidth=0.8)
    axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"Markov support adequacy at train_docs={_format_axis_value(train_target)} (N/SM/FM/FL = none/sparse/full/full-local)",
        fontsize=13,
    )
    path = figures / "simulation_intent_markov_support_recovery.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _plot_lda_baseline_family(output_root: Path, figures: Path) -> str:
    rows = _collect_lda_baseline_rows(output_root)
    worst_by_lam = _groupwise_max(
        rows,
        group_getter=lambda row: row.get("lambda_multiplier"),
        value_getter=lambda row: row.get("ridge"),
    )
    norm_panels, x_vals, y_vals = _normalized_panels_from_rows(
        rows,
        observed_getter=lambda row: row.get("ridge"),
        good_getter=lambda row: row.get("exact"),
        fallback_bad_by_group=worst_by_lam,
        force_group_bad=True,
    )
    categories, series = _median_at_max(
        rows,
        category_key="lambda_multiplier",
        max_keys=["train_docs", "audit_fraction"],
        value_by_name={
            "exact": lambda row: row["exact"],
            "ridge_true_topics": lambda row: row["ridge_true_topics"],
            "ridge": lambda row: row["ridge"],
            "undersupported": lambda row: row["undersupported"],
        },
    )

    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("#d9d9d9")

    fig = plt.figure(figsize=(13.0, 4.7), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.3])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]

    lam_order = sorted(norm_panels)
    im = None
    for idx, lam in enumerate(lam_order[:2]):
        im = _draw_heatmap(
            axes[idx],
            norm_panels[lam],
            [_format_axis_value(v) for v in x_vals],
            [_format_axis_value(v) for v in y_vals],
            title=f"Ordinary LDA | normalized ridge gap | lambda={_format_axis_value(lam)}",
            x_label="train_docs",
            y_label="audit_fraction",
            cmap=cmap,
        )
    _plot_grouped_bars(
        axes[2],
        categories,
        series,
        title="Ordinary LDA high-support root MAE",
        y_label="root_mae",
    )
    if im is not None:
        cbar = fig.colorbar(im, ax=axes[:2], fraction=0.05, pad=0.03)
        cbar.set_label("Within-ordinary-LDA normalized gap")
    path = figures / "simulation_intent_lda_baseline_family.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _plot_segment_family(output_root: Path, figures: Path) -> str:
    bar_rows = _collect_segment_rows(output_root)
    norm_panels, x_vals, y_vals = _segment_normalized_panels_from_rows(bar_rows)
    categories, series = _median_at_max(
        bar_rows,
        category_key="lambda_multiplier",
        max_keys=["train_docs", "audit_fraction"],
        value_by_name={
            "exact": lambda row: row["exact"],
            "ridge": lambda row: row["ridge"],
            "ridge_true_topics": lambda row: row["ridge_true_topics"],
            "undersupported": lambda row: row["undersupported"],
        },
    )

    fig = plt.figure(figsize=(13.0, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.25])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("#d9d9d9")

    control_lam = min(norm_panels) if norm_panels else 0.0
    sensitive_cands = [lam for lam in sorted(norm_panels) if lam > 0.0]
    sensitive_lam = sensitive_cands[-1] if sensitive_cands else (max(norm_panels) if norm_panels else 0.0)

    im0 = _draw_heatmap(
        axes[0],
        norm_panels.get(control_lam, np.full((len(y_vals), len(x_vals)), np.nan, dtype=float)),
        [_format_axis_value(v) for v in x_vals],
        [_format_axis_value(v) for v in y_vals],
        title=f"Control normalized gap | lambda={_format_axis_value(control_lam)}",
        x_label="train_docs",
        y_label="audit_fraction",
        cmap=cmap,
    )
    im1 = _draw_heatmap(
        axes[1],
        norm_panels.get(sensitive_lam, np.full((len(y_vals), len(x_vals)), np.nan, dtype=float)),
        [_format_axis_value(v) for v in x_vals],
        [_format_axis_value(v) for v in y_vals],
        title=f"Boundary-sensitive normalized gap | lambda={_format_axis_value(sensitive_lam)}",
        x_label="train_docs",
        y_label="audit_fraction",
        cmap=cmap,
    )
    _plot_grouped_bars(
        axes[2],
        categories,
        series,
        title="High-support root MAE by method",
        y_label="root_mae",
    )
    cbar = fig.colorbar(im1, ax=axes[:2], fraction=0.05, pad=0.03)
    cbar.set_label("Within-Segment-LDA normalized gap")
    path = figures / "simulation_intent_segment_lda_family.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _plot_boundary_control_family(output_root: Path, figures: Path) -> Optional[str]:
    rows = _collect_boundary_control_rows(output_root)
    if not rows:
        return None

    scenarios = [
        ("bag_of_words", 384.0, "Whole-Document LDA"),
        ("bag_of_words", 192.0, "Two-Leaf LDA"),
        ("segments", 192.0, "Two-Leaf Segment-LDA"),
    ]
    lambdas = [0.0, 2.0]
    methods = ["exact", "ridge_true_topics", "ridge", "undersupported"]
    labels = {
        "exact": "exact",
        "ridge_true_topics": "ridge_true_topics",
        "ridge": "ridge",
        "undersupported": "undersupported",
    }
    linestyles = {
        "exact": "-",
        "ridge_true_topics": "-",
        "ridge": "-",
        "undersupported": "--",
    }
    markers = {
        "exact": "o",
        "ridge_true_topics": "s",
        "ridge": "^",
        "undersupported": "D",
    }

    fig = plt.figure(figsize=(14.0, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(len(lambdas), len(scenarios))
    axes: List[plt.Axes] = []

    for row_idx, lam in enumerate(lambdas):
        for col_idx, (process, leaf_tokens, title) in enumerate(scenarios):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            axes.append(ax)
            subset = [
                row
                for row in rows
                if str(row.get("topic_process")) == str(process)
                and abs(_safe_float(row.get("leaf_tokens")) - float(leaf_tokens)) <= 1e-12
                and abs(_safe_float(row.get("lambda_multiplier")) - float(lam)) <= 1e-12
            ]
            train_docs_vals = sorted(
                {
                    _safe_float(row.get("train_docs"))
                    for row in subset
                    if math.isfinite(_safe_float(row.get("train_docs")))
                }
            )
            for method in methods:
                y_vals: List[float] = []
                for td in train_docs_vals:
                    vals = [
                        _safe_float(row.get(method))
                        for row in subset
                        if abs(_safe_float(row.get("train_docs")) - td) <= 1e-12
                    ]
                    y_vals.append(_median(vals))
                if not any(math.isfinite(v) for v in y_vals):
                    continue
                ax.plot(
                    train_docs_vals,
                    y_vals,
                    label=labels[method],
                    color=_bar_color(method),
                    linestyle=linestyles.get(method, "-"),
                    marker=markers.get(method, "o"),
                    linewidth=1.8,
                    markersize=5.0,
                )

            ax.set_xscale("log", base=2)
            panel_vals = [
                _safe_float(row.get(method))
                for row in subset
                for method in methods
                if math.isfinite(_safe_float(row.get(method)))
            ]
            ymax = max(panel_vals) if panel_vals else 0.0
            if math.isfinite(ymax) and ymax <= 1e-3:
                ax.set_yscale("linear")
                top = max(1e-8, ymax * 1.25)
                ax.set_ylim(-0.05 * top, top)
            else:
                ax.set_yscale("symlog", linthresh=1e-4)
            ax.set_xticks(train_docs_vals)
            ax.set_xticklabels([_format_axis_value(v) for v in train_docs_vals])
            ax.grid(axis="y", alpha=0.25, linewidth=0.8)
            ax.set_xlabel("train_docs")
            if col_idx == 0:
                ax.set_ylabel(f"root_mae | lambda={_format_axis_value(lam)}")
            ax.set_title(title, fontsize=11)

            if row_idx == 0:
                mean_leaves = _median([_safe_float(row.get("mean_leaves")) for row in subset])
                mean_internal = _median([_safe_float(row.get("mean_internal_nodes")) for row in subset])
                if math.isfinite(mean_leaves) and math.isfinite(mean_internal):
                    ax.text(
                        0.03,
                        0.97,
                        f"leaves~{_format_axis_value(mean_leaves)}, internal~{_format_axis_value(mean_internal)}",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
                    )

    handles, legend_labels = axes[0].get_legend_handles_labels() if axes else ([], [])
    if handles:
        fig.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=False, fontsize=9)
    path = figures / "simulation_intent_lda_boundary_controls.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _draw_topic_boxes(
    ax: plt.Axes,
    topics: Sequence[int],
    *,
    split_after: int,
    title: str,
    left_label: str,
    right_label: str,
    summary_lines: Sequence[str],
) -> None:
    topic_colors = {
        1: "#4c78a8",
        2: "#f58518",
        3: "#54a24b",
    }

    n = len(topics)
    ax.set_xlim(-0.2, float(n) + 0.2)
    ax.set_ylim(-1.45, 1.5)
    ax.axis("off")
    ax.set_title(title, fontsize=12)
    ax.text(float(n) / 2.0, 1.08, r"each box = realized latent token topic $z_{d,t}$", ha="center", va="bottom", fontsize=10)

    for idx, topic in enumerate(topics):
        rect = Rectangle(
            (float(idx), 0.25),
            1.0,
            0.45,
            facecolor=topic_colors.get(int(topic), "#cccccc"),
            edgecolor="black",
            linewidth=0.8,
        )
        ax.add_patch(rect)
        ax.text(idx + 0.5, 0.475, f"$z={int(topic)}$", ha="center", va="center", fontsize=9, color="white")

    ax.plot([float(split_after), float(split_after)], [0.18, 0.82], color="black", linewidth=1.2, linestyle="--")
    ax.text(split_after / 2.0, 0.95, left_label, ha="center", va="bottom", fontsize=10)
    ax.text((split_after + n) / 2.0, 0.95, right_label, ha="center", va="bottom", fontsize=10)

    ax.annotate(
        "",
        xy=(0.1, -0.2),
        xytext=(float(n) - 0.1, -0.2),
        arrowprops=dict(arrowstyle="<->", linewidth=0.9, color="black"),
    )
    ax.text(float(n) / 2.0, -0.02, r"parent span $A$", ha="center", va="bottom", fontsize=10)

    y0 = -0.45
    for idx, line in enumerate(summary_lines):
        ax.text(0.0, y0 - 0.22 * idx, line, ha="left", va="top", fontsize=10)


def _plot_lda_worked_examples(figures: Path) -> str:
    fig = plt.figure(figsize=(13.2, 5.3), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    _draw_topic_boxes(
        ax0,
        [1, 1, 2, 2, 2, 3, 3, 1],
        split_after=4,
        title="Ordinary LDA Worked Example",
        left_label=r"$L=[1,1,2,2]$",
        right_label=r"$R=[2,3,3,1]$",
        summary_lines=[
            r"$c(A)=(3,3,2)$",
            r"$b_{11}=1,\ b_{12}=1,\ b_{22}=2,\ b_{23}=1,\ b_{33}=1,\ b_{31}=1$",
            r"$f^\star(L)=4.5,\quad f^\star(R)=3.4,\quad f^\star(L)+f^\star(R)=7.9$",
            r"boundary term: $2 \to 2$ gives $\lambda W_{22}=2 \cdot 0.05 = 0.1$",
            r"$f^\star(A)=7.9+0.1=8.0$",
        ],
    )

    _draw_topic_boxes(
        ax1,
        [1, 1, 1, 1, 2, 2, 2, 2],
        split_after=4,
        title="Segment-LDA Worked Example",
        left_label=r"$L=[1,1,1,1]$",
        right_label=r"$R=[2,2,2,2]$",
        summary_lines=[
            r"$f^\star(L)=4.6,\quad f^\star(R)=2.3,\quad f^\star(L)+f^\star(R)=6.9$",
            r"boundary term: $1 \to 2$ gives $\lambda W_{12}=2 \cdot 0.60 = 1.2$",
            r"$f^\star(A)=6.9+1.2=8.1$",
            r"clean leaf-aligned setting: the only missing piece is the cross-leaf boundary weight",
        ],
    )

    path = figures / "simulation_intent_lda_worked_examples.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _plot_ctree_family(output_root: Path, figures: Path) -> str:
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("#d9d9d9")
    panels, x_vals, y_vals = _ctree_panel_rows(output_root)
    bar_rows = _collect_ctree_rows(output_root)
    categories, series = _median_at_max(
        bar_rows,
        category_key="leaf",
        max_keys=["n_books_train", "total_query_rate"],
        value_by_name={
            "oracle_tree": lambda row: row["oracle_tree"],
            "estimated_uncalibrated": lambda row: row["estimated_uncalibrated"],
            "estimated_calibrated_budgeted": lambda row: row["estimated_calibrated_budgeted"],
        },
    )

    fig = plt.figure(figsize=(13.0, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.25])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
    im = None
    for idx, leaf in enumerate(sorted(panels)[:2]):
        im = _draw_heatmap(
            axes[idx],
            panels[leaf],
            [_format_axis_value(v) for v in x_vals],
            [_format_axis_value(v) for v in y_vals],
            title=f"Within-family gap, leaf={_format_axis_value(leaf)}",
            x_label="n_books_train",
            y_label="total query rate",
            cmap=cmap,
        )
    _plot_grouped_bars(
        axes[2],
        categories,
        series,
        title="High-support root L1 by method",
        y_label="root_l1_mean",
    )
    cbar = fig.colorbar(im, ax=axes[:2], fraction=0.05, pad=0.03)
    cbar.set_label("Within-C-TreePO normalized gap")
    path = figures / "simulation_intent_ctreepo_family.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _plot_mergeable_family(output_root: Path, figures: Path) -> str:
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("#d9d9d9")
    grid, x_vals, y_vals = _mergeable_panel(output_root)
    k_phase = _load_json(output_root / "mergeable" / "k_m_phase_summary.json")
    rows = list(k_phase.get("rows", []) or [])
    selected = ["one_pass_m5", "full_model_m5", "one_pass_m2", "full_model_m2", "naive_majority", "naive_mean_of_means"]
    bar_labels = ["oracle m=5", "aligned m=5", "oracle m=2", "aligned m=2", "naive maj", "naive mean"]
    target_rows = {str(row.get("method_name")): row for row in rows if int(row.get("target_k", -1)) == 5}
    bar_values = [_safe_float((target_rows.get(name) or {}).get("mean_abs_bias")) for name in selected]

    fig = plt.figure(figsize=(12.8, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    im = _draw_heatmap(
        ax0,
        grid,
        [_format_axis_value(v) for v in x_vals],
        [_format_axis_value(v) for v in y_vals],
        title="Within-family aligned gap across chunk sizes",
        x_label="chunk_budget",
        y_label="chunk_size",
        cmap=cmap,
    )
    ax1.bar(
        np.arange(len(bar_labels)),
        np.asarray(bar_values, dtype=float),
        color=[_bar_color(name) for name in selected],
        edgecolor="black",
        linewidth=0.4,
    )
    ax1.set_xticks(np.arange(len(bar_labels)))
    ax1.set_xticklabels(bar_labels, rotation=25, ha="right")
    ax1.set_ylabel("mean_abs_bias")
    ax1.set_title("Target k=5 method comparison")
    ax1.grid(axis="y", alpha=0.25, linewidth=0.8)
    ymax = float(np.nanmax(np.asarray(bar_values, dtype=float)))
    if math.isfinite(ymax) and ymax > 0.0:
        ax1.set_ylim(0.0, ymax * 1.18)
    cbar = fig.colorbar(im, ax=[ax0], fraction=0.05, pad=0.03)
    cbar.set_label("Within-Mergeable normalized gap")
    path = figures / "simulation_intent_mergeable_family.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _build_report_figures(output_root: Path, expectations: Sequence[Dict[str, object]]) -> Dict[str, str]:
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    out = {
        "scorecard_png": _plot_scorecard(figures, expectations),
        "markov_family_png": _plot_markov_family(output_root, figures),
        "lda_worked_examples_png": _plot_lda_worked_examples(figures),
        "lda_baseline_png": _plot_lda_baseline_family(output_root, figures),
        "boundary_controls_png": "",
        "segment_family_png": _plot_segment_family(output_root, figures),
        "ctree_family_png": _plot_ctree_family(output_root, figures),
        "mergeable_family_png": _plot_mergeable_family(output_root, figures),
    }
    boundary_controls_png = _plot_boundary_control_family(output_root, figures)
    if boundary_controls_png is not None:
        out["boundary_controls_png"] = boundary_controls_png
    else:
        out.pop("boundary_controls_png", None)
    markov_supervision_png = _plot_markov_supervision_narrative(output_root, figures)
    if markov_supervision_png is not None:
        out["markov_supervision_png"] = markov_supervision_png
    markov_support_recovery_png = _plot_markov_support_recovery(output_root, figures)
    if markov_support_recovery_png is not None:
        out["markov_support_recovery_png"] = markov_support_recovery_png
    return out


def _family_status_rows(expectations: Iterable[Dict[str, object]]) -> List[str]:
    counts: Dict[str, Dict[str, int]] = {}
    for rec in expectations:
        family = str(rec.get("family", "unknown"))
        status = str(rec.get("status", "unknown"))
        fam = counts.setdefault(family, {"pass": 0, "warn": 0, "fail": 0, "not_applicable": 0})
        if status not in fam:
            fam[status] = 0
        fam[status] += 1

    lines = [
        "| Family | Pass | Warn | Fail | N/A |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for family in sorted(counts):
        fam = counts[family]
        lines.append(
            f"| `{family}` | {_fmt_count(fam.get('pass'))} | {_fmt_count(fam.get('warn'))} | {_fmt_count(fam.get('fail'))} | {_fmt_count(fam.get('not_applicable'))} |"
        )
    return lines


def _warning_lines(expectations: Iterable[Dict[str, object]], family: str) -> List[str]:
    out: List[str] = []
    for rec in expectations:
        if str(rec.get("family")) != family:
            continue
        if str(rec.get("status")) != "warn":
            continue
        title = str(rec.get("title", ""))
        summary = rec.get("observed_summary", {}) or {}
        note = str(summary.get("note", "")).strip()
        if note:
            out.append(f"- `{title}`: {note}")
        else:
            out.append(f"- `{title}`")
    return out[:6]


def _markov_supervision_recovery_table(output_root: Path) -> List[str]:
    rows = _collect_markov_supervision_rows(output_root)
    if not rows:
        return []
    train_values = [_safe_float(row.get("train_docs")) for row in rows if math.isfinite(_safe_float(row.get("train_docs")))]
    if not train_values:
        return []
    train_target = max(train_values)
    rows = [row for row in rows if abs(_safe_float(row.get("train_docs")) - train_target) <= 1e-12]
    regime_order = ["none", "sparse_merge", "full_merge", "root_only", "full_local", "full_direct"]
    combos = [(16, "additive"), (16, "neural"), (32, "additive"), (32, "neural")]
    lines = [
        "| Regime | leaf16 additive | leaf16 neural | leaf32 additive | leaf32 neural |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for regime in regime_order:
        vals = []
        for leaf, family in combos:
            subset = [
                _utility_recovery_fraction(
                    row.get("learned_root"),
                    row.get("exact_root"),
                    row.get("undersupported_root"),
                )
                for row in rows
                if abs(_safe_float(row.get("leaf")) - float(leaf)) <= 1e-12
                and str(row.get("model_family")) == family
                and str(row.get("regime")) == regime
            ]
            med = _median(subset)
            vals.append("NA" if not math.isfinite(med) else f"{100.0 * med:.0f}%")
        lines.append(f"| `{regime}` | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |")
    return lines


def _markov_support_table(output_root: Path) -> List[str]:
    rows = _collect_markov_supervision_rows(output_root)
    if not rows:
        return []
    train_values = [_safe_float(row.get("train_docs")) for row in rows if math.isfinite(_safe_float(row.get("train_docs")))]
    if not train_values:
        return []
    train_target = max(train_values)
    rows = [row for row in rows if abs(_safe_float(row.get("train_docs")) - train_target) <= 1e-12]
    regime_order = ["none", "sparse_merge", "full_merge", "root_only", "full_local", "full_direct"]
    lines = [
        "| Regime | leaf16 local coverage | leaf16 undersupport | leaf32 local coverage | leaf32 undersupport | root access |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for regime in regime_order:
        vals: List[str] = []
        for leaf in [16.0, 32.0]:
            subset = [
                row
                for row in rows
                if abs(_safe_float(row.get("leaf")) - leaf) <= 1e-12 and str(row.get("regime")) == regime
            ]
            coverage = _median([_safe_float(row.get("local_oracle_coverage")) for row in subset])
            undersupport = _median([_safe_float(row.get("local_undersupport")) for row in subset])
            vals.append("NA" if not math.isfinite(coverage) else f"{100.0 * coverage:.0f}%")
            vals.append("NA" if not math.isfinite(undersupport) else f"{100.0 * undersupport:.0f}%")
        root_access = _median(
            [_safe_float(row.get("root_query_rate")) for row in rows if str(row.get("regime")) == regime]
        )
        root_access_text = "NA" if not math.isfinite(root_access) else f"{100.0 * root_access:.0f}%"
        lines.append(f"| `{regime}` | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {root_access_text} |")
    return lines


def main() -> int:
    args = _parse_args()
    output_root = args.output_root.resolve()
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    expectations_path = _ensure_expectations(output_root)
    expectations = _load_json(expectations_path)
    summary = expectations.get("summary", {}) or {}
    records = list(expectations.get("expectations", []) or [])
    fig_paths = _build_report_figures(output_root, records)

    md_path = (
        args.output_markdown.resolve()
        if args.output_markdown is not None
        else (figures / "simulation_intent_report.md")
    )
    pdf_path = args.output_pdf.resolve() if args.output_pdf is not None else md_path.with_suffix(".pdf")

    lines: List[str] = []
    lines.append("# Curated Simulation Intent Report")
    lines.append("")
    lines.append(f"- Output root: `{output_root}`")
    lines.append(f"- Expectation report: `{expectations_path}`")
    lines.append(
        f"- Summary: `{_fmt_count(summary.get('n_pass'))} pass / {_fmt_count(summary.get('n_warn'))} warn / {_fmt_count(summary.get('n_fail'))} fail / {_fmt_count(summary.get('n_not_applicable'))} n/a`"
    )
    lines.append("")
    lines.append("## Color Convention")
    lines.append("")
    lines.append("- Most report-native heatmaps use the same color semantics: `green = closer to the intended good anchor`, `red = closer to the bad anchor`.")
    lines.append("- The main Markov support-trend page uses raw MAE curves on a log scale. `exact` is dark green, `learned` is light green, and `undersupported` is red.")
    lines.append("- The Markov oracle-access ablation mixes two scales on purpose: the top row is root utility recovery on `[0,1]` and the bottom row is raw `merge_mae` on a log scale.")
    lines.append("- Ordinary-LDA baseline panels use `(ridge.root_mae - exact) / (worst observed ridge.root_mae in the same lambda slice - exact)`, so the red anchor is the worst observed inferred-feature error in that slice.")
    lines.append("- Segment-LDA uses the canonical `(ridge.root_mae - exact) / (undersupported.root_mae - exact)` normalization when the boundary-sensitive undersupported anchor is non-degenerate, and falls back to the same worst-observed-slice normalization in the `lambda=0` control where `exact = undersupported = 0`.")
    lines.append("- Segmented-LDA C-TreePO panels use `(estimated_calibrated_budgeted.root_l1_mean - oracle_tree) / (estimated_uncalibrated.root_l1_mean - oracle_tree)`.")
    lines.append("- Mergeable panels use `(aligned.mean_abs_bias - one_pass_reference) / (worst_aligned_gap)`.")
    lines.append("- All normalized values are clipped into `[0, 1]`, but those values are only meaningful within a family. A green cell in C-TreePO is not numerically comparable to a green cell in Segment-LDA.")
    lines.append("- For panels normalized against the worst observed value in a slice, the colors are also slice-local: they tell you which settings are better or worse within that panel, not the absolute error scale across different panels.")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"![]({fig_paths['scorecard_png']}){{ width=92% }}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- The scorecard is discrete: `green = pass`, `yellow = warn`, `red = fail`.")
    lines.append("- The pages that follow are family-specific. They are intended to support within-family claims only.")
    lines.append("")
    lines.append("## Family Status")
    lines.append("")
    lines.extend(_family_status_rows(records))
    lines.append("")
    lines.append("## Selected Scope")
    lines.append("")
    lines.append("- Markov uses the leaner `leaf_tokens in {16,32}` grid rather than the very fine `leaf=8, train=8000` corners.")
    lines.append("- A plain-LDA bag-of-words control is now included as an expository baseline using the same OPS weight-recovery simulator and the same hard leaf-MAP feature map.")
    lines.append("- Segment-LDA uses the corrected control comparison: `topic_process=segments`, `topic_phi_estimator=true`, and `lambda_multiplier in {0,2}`.")
    lines.append("- Segmented-LDA C-TreePO and Mergeable use the passing curated batches from the readiness iteration.")
    lines.append("- The scorecard and expectation counts are unchanged: they still refer to the original curated family checks, while the plain-LDA page is included to clarify the modeling story rather than as a new pass/fail family.")
    lines.append("")
    lines.append(r"\newpage")
    lines.append("")
    lines.append("## Markov OPS Count")
    lines.append("")
    lines.append("**DGP**")
    lines.append("- Documents are piecewise-constant regime sequences from a finite-state Markov generator.")
    lines.append("- The oracle is the number of changepoints in a span: adjacent regime flips inside that span.")
    lines.append("- Fixed leaves partition the document; the comparison asks whether the learned sketch can preserve the changepoint-count oracle under tree reduction.")
    lines.append("")
    lines.append("**Plain-Language Version**")
    lines.append("- Each document is a sequence of latent regimes such as `A A A B B C C ...`.")
    lines.append("- The score of any span is just: `how many times did the regime change inside this span?`.")
    lines.append("- If we cut a document into leaves and later merge them back together, the key difficulty is the boundary between two adjacent leaves.")
    lines.append("- If the left leaf ends in regime `A` and the right leaf begins in regime `B`, the parent span has one extra changepoint at that join. If we do not know those boundary identities, we cannot correct for that join exactly.")
    lines.append("")
    lines.append("**How The Simulation Is Started**")
    lines.append("- The direct entrypoint is [run_markov_changepoint_ops_count_simulation.py](/home/mlinegar/ThinkingTrees/scripts/run_markov_changepoint_ops_count_simulation.py).")
    lines.append("- Large sweeps are usually built with [build_markov_changepoint_ops_count_cmds.py](/home/mlinegar/ThinkingTrees/scripts/build_markov_changepoint_ops_count_cmds.py), which emits one command per setting.")
    lines.append("- The new named-regime Markov supervision sweep for this report is built with [build_markov_supervision_narrative_cmds.py](/home/mlinegar/ThinkingTrees/scripts/build_markov_supervision_narrative_cmds.py).")
    lines.append("- A representative curated run is:")
    lines.append("")
    lines.append("```bash")
    lines.append("source venv/bin/activate")
    lines.append("python scripts/run_markov_changepoint_ops_count_simulation.py \\")
    lines.append("  --model-family neural \\")
    lines.append("  --feature-mode full \\")
    lines.append("  --fixed-leaf-tokens 16 \\")
    lines.append("  --train-docs 1000 \\")
    lines.append("  --test-docs 256 \\")
    lines.append("  --audit-policy fraction \\")
    lines.append("  --audit-fraction 0.05 \\")
    lines.append("  --c3-audit-strategy uniform \\")
    lines.append("  --leaf-query-rate 0.0 \\")
    lines.append("  --no-root-query \\")
    lines.append("  --device cuda \\")
    lines.append("  --json-summary outputs/.../markov_full_leaf16_lqr0p0_audit0p05_train1000_seed0.json")
    lines.append("```")
    lines.append("")
    lines.append("**What One Run Actually Does**")
    lines.append("1. Draw one train corpus and one test corpus from the same regime-transition family, but with different seeds so the test set stays fixed for a given run.")
    lines.append("2. Cut each document into fixed leaves of length `fixed_leaf_tokens`.")
    lines.append("3. For each leaf, build features from the latent regime sequence: first-regime one-hot, last-regime one-hot, normalized regime-transition counts, and leaf length.")
    lines.append("4. Build the balanced binary tree over those leaves and compute oracle changepoint counts for every realized internal node.")
    lines.append("5. Train the learned merger only on the oracle labels the budget pays for.")
    lines.append("6. Evaluate root error, merge error, and schedule spread on held-out test documents.")
    lines.append("")
    lines.append("**What The Figure Slice Is Actually Varying**")
    lines.append("- The paper-facing trend figure is intentionally small: `fixed_leaf_tokens in {16, 32}`, `train_docs in {1000, 8000}`, and `audit_fraction in {0.0, 0.05}`.")
    lines.append("- In that slice, `feature_mode=full`, `model_family=neural`, `leaf_query_rate=0.0`, and `include_root_query=false` are held fixed so the plot isolates sparse merge supervision.")
    lines.append("- `test_docs=256` is held fixed.")
    lines.append("- Separately, the named oracle-access ablation fixes `train_docs=8000` and varies six regimes: `none`, `sparse_merge`, `full_merge`, `root_only`, `full_local`, and `full_direct` across both learned families and both leaf sizes.")
    lines.append("")
    lines.append("**What The Full Markov Support Matrix Is**")
    lines.append("- The live rectangular matrix behind this report is larger than the plotted slice: `train_docs in {1000, 4000, 8000}`, `fixed_leaf_tokens in {16, 32}`, `audit_fraction in {0.0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0}`, `leaf_query_rate in {0.0, 0.25, 1.0}`, `include_root_query in {false, true}`, and `seed in {0, 1}`.")
    lines.append("- That is `576` total cells (`288` non-seed cells). The explicit matrix specification is [markov_local_support_grid.full_matrix.md](/home/mlinegar/ThinkingTrees/outputs/simulation_intent_curated_20260305_214500/markov_local_support_grid.full_matrix.md).")
    lines.append("- The matrix is generated from [build_markov_narrative_suite_cmds.py](/home/mlinegar/ThinkingTrees/scripts/build_markov_narrative_suite_cmds.py) and formalized by [write_markov_sweep_matrix.py](/home/mlinegar/ThinkingTrees/scripts/write_markov_sweep_matrix.py).")
    lines.append("- The live completed-cell fair-comparison summary is [markov_local_support_grid.comparison_classes.md](/home/mlinegar/ThinkingTrees/outputs/simulation_intent_curated_20260305_214500/markov_local_support_grid.comparison_classes.md).")
    lines.append("")
    lines.append("**Why The Matrix Needs A Fair-Comparison Policy**")
    lines.append("- `audit_fraction`, `leaf_query_rate`, and `include_root_query` are not interchangeable. They buy different kinds of labels.")
    lines.append("- A nominally similar budget can still mean very different information. For example, one internal-node label per document is not the same signal as six leaf labels per document, even if both are cheap.")
    lines.append("- Fair comparison therefore requires matching tree geometry first, then matching realized `leaf_label_coverage`, `internal_label_coverage`, and `root_query_rate`.")
    lines.append("- Runs with similar total labels per document but different label type are still only proxy comparisons, not equal-information comparisons.")
    lines.append("")
    lines.append("**Why That Matters**")
    lines.append("- In this curated slice the learned model gets no leaf labels and no root labels at all.")
    lines.append("- The only supervision comes from sampled balanced-tree internal nodes, so the learned root behavior is entirely a byproduct of learning a reusable merge rule.")
    lines.append("- That is why the root story can improve while merge and schedule diagnostics remain weaker or unstable: the run is intentionally stressing merge generalization rather than giving the model direct root supervision.")
    lines.append("")
    lines.append("**Concrete Supervision Geometry In The Curated Runs**")
    lines.append("- At `leaf_tokens=16`, a typical 384-token document has about `24` leaves and `23` internal nodes.")
    lines.append("- With `audit_fraction=0.05`, that means about `2` internal-node labels per training document on average, with zero leaf labels and zero root labels.")
    lines.append("- At `leaf_tokens=32`, the same document has about `12` leaves and `11` internal nodes, so the same `audit_fraction=0.05` gives only about `1` internal-node label per training document.")
    lines.append("- This explains why the `leaf=32` setting is the harder supervision regime even before looking at the error curves.")
    lines.append("")
    lines.append("**Methods Compared**")
    lines.append("- `exact`: the theorem-backed sketch/control ceiling.")
    lines.append("- `learned`: a neural leaf encoder + neural merger + scalar readout, with endpoints carried explicitly in state.")
    lines.append("- `undersupported`: count-only state without endpoint metadata, so it cannot add the cross-boundary correction term.")
    lines.append("")
    lines.append("**What Those States Mean Concretely**")
    lines.append("- `exact` carries exactly the information the merge rule needs: `(count, first_regime, last_regime)`.")
    lines.append("- `undersupported` carries only `(count)`. That means it knows how many changes happened inside each child span, but it has thrown away the one piece of information needed to know whether the join itself adds another change.")
    lines.append("- `learned` carries `(latent_vector, first_regime, last_regime)`. The neural part tries to learn the reusable count state; the explicit endpoints prevent the model from having to rediscover the join metadata from scratch.")
    lines.append("")
    lines.append("**What This Simulation Is Trying To Show**")
    lines.append("- First, the exact state should solve the problem exactly. If it does not, the implementation is wrong.")
    lines.append("- Second, the undersupported state should stay biased away from zero, because no amount of extra training data can recover information that the state representation has discarded.")
    lines.append("- Third, the learned state should move toward the exact ceiling as we add more training documents and more internal-node oracle labels.")
    lines.append("- The central claim is therefore a representation claim plus a learning claim: the task is mergeable in principle, and the intended learned operator can move toward that mergeable solution under supervision.")
    lines.append("")
    lines.append("**What Would Count As Success Or Failure**")
    lines.append("- Success: `exact` is at zero, `undersupported` stays materially above zero, and `learned.root_mae` drops below `undersupported.root_mae` as support increases.")
    lines.append("- Failure of the theorem-backed implementation: `exact` is not essentially zero.")
    lines.append("- Failure of the misspecification control: `undersupported` also goes to zero in a genuinely multi-leaf regime.")
    lines.append("- Failure of the intended method: the learned merger never beats the undersupported baseline at the root, or improves only when given direct root supervision instead of reusable merge supervision.")
    lines.append("")
    lines.append("**How To Read The Figure**")
    lines.append("- These are raw error curves, not normalized heatmaps.")
    lines.append("- Each row fixes a leaf size. The left panel shows `root_mae` as internal-node audit fraction increases, the middle panel shows `root_mae` as train-doc count increases, and the right panel shows `merge_mae` as internal-node audit fraction increases.")
    lines.append("- On every panel, lower is better. The intended visual pattern is: `exact` stays at the floor, `undersupported` stays high, and `learned` moves downward as support increases.")
    lines.append("- This is the cleanest direct answer to the question: do more C-TreePO-style internal labels and more training documents actually reduce error?")
    lines.append("")
    lines.append(f"![]({fig_paths['markov_family_png']}){{ width=98% }}")
    lines.append("")
    lines.append("**Paper-Style Reading Of The Figure**")
    lines.append("- The main point is now visible directly in the curves: for both leaf sizes, increasing internal-node audit and increasing train-doc count lower `learned.root_mae` relative to the undersupported baseline.")
    lines.append("- The leaf-size comparison is also local to this DGP. Here, larger leaves mean fewer internal nodes and therefore fewer paid internal labels per document at a fixed audit fraction, so the `leaf=32` learned curves stay higher.")
    lines.append("- The merge curve on the right is intentionally included because it shows the limit of the current learned operator: root error can improve faster than merge error.")
    lines.append("")
    if "markov_supervision_png" in fig_paths:
        lines.append("**Oracle-Access Pattern Ablation**")
        lines.append("- To make the training geometry concrete, we also ran a second high-support sweep at `train_docs=8000` with six named oracle-access patterns.")
        lines.append("- `none`: no leaf labels, no internal labels, no root labels.")
        lines.append("- `sparse_merge`: sparse internal-node labels only.")
        lines.append("- `full_merge`: all internal-node labels, but still no leaf or root labels.")
        lines.append("- `root_only`: one direct root label per document and nothing else.")
        lines.append("- `full_local`: all leaf labels plus all internal-node labels, but still no direct root label.")
        lines.append("- `full_direct`: all leaf labels, all internal-node labels, and the direct root label.")
        lines.append("- The key support metric is `local oracle coverage = (mean_leaf_labels + mean_internal_labels) / (mean_leaves + mean_internal_nodes)`.")
        lines.append("- Its complement is `local undersupport = 1 - local oracle coverage`.")
        lines.append("- This is the right “how undersupported are we?” number for the mergeable part of the problem, because it ignores direct root labels and measures how much local theorem-relevant supervision the learner actually sees.")
        lines.append("")
        if "markov_support_recovery_png" in fig_paths:
            lines.append(f"![]({fig_paths['markov_support_recovery_png']}){{ width=98% }}")
            lines.append("")
        lines.append("Realized support at the same `train_docs=max` slice:")
        lines.append("")
        lines.extend(_markov_support_table(output_root))
        lines.append("")
        lines.append(f"![]({fig_paths['markov_supervision_png']}){{ width=98% }}")
        lines.append("")
        lines.append("Median root-utility recovery at the same `train_docs=max` slice:")
        lines.append("")
        lines.extend(_markov_supervision_recovery_table(output_root))
        lines.append("")
        lines.append("**How To Read The Oracle-Access Ablation**")
        lines.append("- Start with the support figure above. If a curve is flat there, that means the model is not gaining much even as local oracle coverage increases.")
        lines.append("- The support table tells you how severe the starvation is. For example, `sparse_merge` is only about `4%` local oracle coverage in this DGP, while `full_merge` is about `48-49%`, and `full_local` is `100%`.")
        lines.append("- The top row is the direct utility question: how much of the root utility do we recover under each access pattern?")
        lines.append("- Higher is better in the top row: `0` means no improvement over the undersupported baseline and `1` means exact utility recovery.")
        lines.append("- The left column compares merge-access patterns: `none -> sparse_merge -> full_merge -> full_local`.")
        lines.append("- The right column compares direct-access patterns: `none -> root_only -> full_direct`.")
        lines.append("- The bottom row is the guardrail: lower `merge_mae` means the model is actually learning a reusable merge rule, not only fitting the root target.")
        lines.append("- Solid lines are `leaf=16`; dashed lines are `leaf=32`. Green lines are the structured additive family; red lines are the neural family.")
        lines.append("")
        lines.append("**What The Oracle-Access Ablation Actually Shows**")
        lines.append("- The structured additive family behaves exactly like the intended positive control. With `none` or `root_only`, it fails catastrophically because it never learns a reusable merge rule. As soon as it gets internal-node merge labels, even `sparse_merge`, root and merge error collapse to machine-zero scale.")
        lines.append("- The `root_only` regime is therefore doing useful work in the paper story: it can improve recovered utility without certifying that the model learned the mergeable solution.")
        lines.append("- The neural family is qualitatively better than `none`, but still far from the exact/additive ceiling. The new support figure shows that this is not only because the low-budget cells are starved: even by roughly `48-49%` local oracle coverage, the neural merger is still recovering only about `66-78%` of the root utility and its merge MAE remains large.")
        lines.append("- The neural family also shows why the schedule diagnostic matters. Even when root and merge MAE improve, `schedule_spread_mean` remains large, so the learned operator is still not behaving like a stable theorem-backed sketch.")
        lines.append("- `leaf=32` is consistently harder for the neural family. The same supervision policy buys fewer internal labels per document, and the learned errors stay materially above the `leaf=16` values.")
        lines.append("- In short: the Markov ablation now separates utility recovery from merge recovery. The exact/additive positive control recovers the utility once it gets merge access, `root_only` shows that direct utility supervision alone is not enough, and the neural merger improves utility but still falls short of law-like merge behavior.")
        lines.append("")
    lines.append("**What Is Going On In The Current Curated Result**")
    lines.append("- The exact ceiling is perfect, which confirms the implemented exact sketch really does carry the oracle-preserving state.")
    lines.append("- The undersupported baseline stays substantially above zero at the root, which is the intended misspecification control: without endpoint transport it misses cross-leaf changepoints.")
    lines.append("- The learned model does improve strongly at the root when `audit_fraction` and `train_docs` increase, so the main intended signal is present.")
    lines.append("- The confusing part is that merge error and schedule spread are still weak. That is not a contradiction once you notice the training setup: sparse balanced-merge supervision can be enough to lower root MAE without producing a schedule-stable merger.")
    lines.append("- In other words, the current curated Markov page is really showing `root recovery under sparse merge supervision`, not `full local-law recovery across all diagnostics`.")
    lines.append("- A paper should therefore describe this figure as evidence for learnable mergeability at the root under sparse internal supervision, and not as evidence that the learned operator already satisfies every local-law-style diagnostic.")
    lines.append("")
    lines.extend(_warning_lines(records, "markov_ops_count") or ["- No warnings recorded."])
    lines.append("")
    lines.append(r"\newpage")
    lines.append("")
    lines.append("## Plain LDA First")
    lines.append("")
    lines.append("**What Stays Fixed Across The LDA Pages**")
    lines.append(r"- The topic-word distributions $\phi_1, \ldots, \phi_K$ are drawn in the same way.")
    lines.append(r"- The span oracle $f^\star(A)$ is the same weighted combination of topic counts and topic bigrams.")
    lines.append("- The training problem is the same: query leaves and some internal spans, then fit a predictor for held-out root scores.")
    lines.append("- What changes between ordinary LDA and Segment-LDA is only the latent topic process inside the document.")
    lines.append("")
    lines.append("**Generative Model**")
    lines.append("$$")
    lines.append(r"\pi_d \sim \mathrm{Dirichlet}(\alpha \mathbf{1}_K), \qquad z_{d,t} \stackrel{\mathrm{iid}}{\sim} \mathrm{Categorical}(\pi_d), \qquad x_{d,t} \sim \mathrm{Categorical}(\phi_{z_{d,t}}).")
    lines.append("$$")
    lines.append(r"- Here each document has a topic mixture $\pi_d$, and each token independently chooses a topic from that mixture.")
    lines.append(r"- There are no persistent topic blocks here. Even inside one $16$-token leaf, the latent topic sequence can switch back and forth many times.")
    lines.append("")
    lines.append("**Oracle**")
    lines.append("$$")
    lines.append(r"f^\star(A) = \sum_{k=1}^K \theta_k\, c_k(A) + \lambda \sum_{i=1}^K \sum_{j=1}^K W_{ij}\, b_{ij}(A).")
    lines.append("$$")
    lines.append(r"- $c_k(A)$ is the number of tokens in span $A$ whose latent topic is $k$.")
    lines.append(r"- $b_{ij}(A)$ is the number of adjacent topic pairs $(i,j)$ inside $A$.")
    lines.append(r"- $\lambda = 0$ removes the bigram term and leaves only topic-count scoring. $\lambda > 0$ makes boundary-sensitive bigram information matter.")
    lines.append("")
    lines.append("**Document-Level Mixture Versus Realized Topic Sequence**")
    lines.append(r"- In ordinary LDA, a document first draws a mixture vector $\pi_d$. That is the document-level probability distribution over topics.")
    lines.append(r"- Conditioned on $\pi_d$, each token still draws one realized latent topic $z_{d,t} \in \{1,\ldots,K\}$. The oracle is defined on that realized latent topic sequence, not directly on the mixture vector.")
    lines.append(r"- So “adjacent topic pairs” means neighboring realized latent assignments $(z_{d,t}, z_{d,t+1})$, not adjacent words and not pairs of probability vectors.")
    lines.append(r"- If one wants the document-level conditional expectation for a span $A$ of length $n_A$, then under bag-of-words LDA")
    lines.append("$$")
    lines.append(r"\mathbb{E}[c_k(A)\mid \pi_d] = n_A\, \pi_{d,k}, \qquad \mathbb{E}[b_{ij}(A)\mid \pi_d] = (n_A-1)\, \pi_{d,i}\pi_{d,j}.")
    lines.append("$$")
    lines.append(r"- Therefore the conditional expected oracle score is")
    lines.append("$$")
    lines.append(r"\mathbb{E}[f^\star(A)\mid \pi_d] = n_A\, \theta^\top \pi_d + \lambda (n_A-1)\, \pi_d^\top W \pi_d.")
    lines.append("$$")
    lines.append(r"- This is the clean document-level answer. The simulator, however, scores the realized span after the token topics have been sampled, because leaf utility and cross-leaf boundary corrections live on that realized topic sequence.")
    lines.append("")
    lines.append("**Three-Topic Worked Example**")
    lines.append(r"- Take $K=3$, $\theta = (1.0, 0.5, -0.25)$, $\lambda = 2$, and nonzero bigram weights $W_{11}=0.10$, $W_{12}=0.60$, $W_{22}=0.05$, $W_{23}=0.70$, $W_{33}=0.10$, $W_{31}=0.40$, with all other $W_{ij}=0$.")
    lines.append(r"- Suppose the document-level topic mixture is $\pi_d = (0.5, 0.3, 0.2)$ and the span length is $n_A=8$. Then")
    lines.append("$$")
    lines.append(r"\mathbb{E}[f^\star(A)\mid \pi_d] = 8\,\theta^\top \pi_d + 2 \cdot 7\, \pi_d^\top W \pi_d \approx 7.68.")
    lines.append("$$")
    lines.append(r"- Now take one realized latent topic sequence sampled from that same document: $(1,1,2,2,2,3,3,1)$.")
    lines.append(r"- Its topic counts are $c(A) = (3,3,2)$.")
    lines.append(r"- Its adjacent topic pairs are $11,12,22,22,23,33,31$, so the nonzero bigram counts are $b_{11}=1$, $b_{12}=1$, $b_{22}=2$, $b_{23}=1$, $b_{33}=1$, $b_{31}=1$.")
    lines.append(r"- The count term is $1.0 \cdot 3 + 0.5 \cdot 3 - 0.25 \cdot 2 = 4.0$.")
    lines.append(r"- The weighted bigram term is $2 \cdot (0.10 + 0.60 + 0.05 + 0.05 + 0.70 + 0.10 + 0.40) = 4.0$.")
    lines.append(r"- So the oracle score of the whole span is $f^\star(A) = 8.0$.")
    lines.append(r"- If we split that span into two leaves, $L=(1,1,2,2)$ and $R=(2,3,3,1)$, then $f^\star(L)=4.5$ and $f^\star(R)=3.4$, which add up to $7.9$.")
    lines.append(r"- The missing $0.1$ is exactly the boundary pair between the leaves: the parent span contains one extra $2 \to 2$ transition at the join, and its contribution is $\lambda W_{22} = 2 \cdot 0.05 = 0.1$.")
    lines.append("- That is the key object in this family. Parent scores are not just sums of child scores; they also depend on the topic transition across the child boundary.")
    lines.append(r"- The exact summary is designed to preserve enough boundary information to add that missing $0.1$. An undersupported summary that only aggregates child-level totals cannot recover it.")
    lines.append("")
    lines.append(f"![]({fig_paths['lda_worked_examples_png']}){{ width=98% }}")
    lines.append("")
    lines.append(r"- Each colored box in this figure is one realized latent token topic $z_{d,t}$. The left panel visualizes the ordinary-LDA realized-span example above, and the right panel previews the leaf-aligned Segment-LDA example used on the next page.")
    lines.append("")
    lines.append("**What The Inferred-Feature Path Does In These Runs**")
    lines.append("$$")
    lines.append(r"\hat z_\ell = \arg\max_{k \in \{1,\dots,K\}} \sum_{t \in \ell} \log \phi_k(x_t).")
    lines.append("$$")
    lines.append(r"- For each leaf $\ell$, the current inferred-feature path compresses the leaf to one inferred topic label.")
    lines.append("- That is exactly the hard leaf-MAP step implemented in [segment_lda_ops_weight_recovery.py](/home/mlinegar/ThinkingTrees/src/ctreepo/sim/core/segment_lda_ops_weight_recovery.py#L2147).")
    lines.append(r"- Under ordinary LDA this means a leaf whose latent topic sequence might look like $(1,1,2,2)$ is summarized by one topic id before the downstream span features are built.")
    lines.append("")
    lines.append("**What We Expect To See Under Ordinary LDA**")
    lines.append(r"- The exact control should stay at $0$, because it is given the exact oracle-preserving summary.")
    lines.append(r"- The true-topic ridge fit should also be very close to $0$ when support is high, because the regression then sees the true latent count and bigram features.")
    lines.append(r"- The inferred-topic ridge fit should improve as $\mathrm{train\_docs}$ and $\mathrm{audit\_fraction}$ increase, but more gradually, because it has to work with leaf summaries built from inferred topics rather than the true latent topic sequence.")
    lines.append(r"- $\lambda=0$ should be easier than $\lambda>0$, because once the bigram term is active the method also has to track cross-token and cross-leaf topic transitions.")
    lines.append(r"- In the curated bag-of-words runs, held-out mean leaf-topic purity is about $0.44$ and held-out leaf-topic accuracy is about $0.59$, so the ordinary-LDA page should be read as a noisier recovery setting rather than a clean leaf-aligned setting.")
    lines.append("")
    lines.append("**What One Run Actually Does**")
    lines.append("1. Draw train and test documents from ordinary bag-of-words LDA with the same topic-word matrix but different random seeds.")
    lines.append(r"2. Cut each $384$-token document into $24$ leaves of $16$ tokens each.")
    lines.append("3. Query every leaf oracle label and a budgeted subset of internal-node oracle labels.")
    lines.append("4. Infer one topic per leaf by hard MAP, build span features from those inferred topics, and fit ridge to the queried span labels.")
    lines.append("5. Evaluate root, leaf, and merge errors on held-out documents against the true latent-topic oracle.")
    lines.append("")
    lines.append("**What The Figure Is Showing**")
    lines.append(r"- Left panels: within-slice normalized root MAE for ordinary LDA at $\lambda=0$ and $\lambda=2$.")
    lines.append(r"- For each fixed $\lambda$, green is the exact ceiling and red is the worst observed inferred-feature root MAE in that ordinary-LDA slice.")
    lines.append("- Right panel: high-support raw root MAE by method, with `ridge_true_topics` included to show what happens if the same linear oracle is fit from the true latent topic sequence rather than the hard leaf-MAP proxy.")
    lines.append("")
    lines.append(f"![]({fig_paths['lda_baseline_png']}){{ width=98% }}")
    lines.append("")
    lines.append("**Paper-Style Interpretation**")
    lines.append(r"- The main pattern matches the expectations above. The true-topic ridge fit is essentially exact: at high support it is about $2.9 \times 10^{-7}$ when $\lambda=0$ and about $2.3 \times 10^{-6}$ when $\lambda=2$.")
    lines.append(r"- The inferred-feature ridge estimator also improves with support, but it remains materially above the oracle ceiling: at high support it is about $6.29$ when $\lambda=0$ and about $17.50$ when $\lambda=2$.")
    lines.append(r"- The increase from $\lambda=0$ to $\lambda=2$ is also exactly what the worked example suggests. Once the score depends on topic transitions, the boundary-sensitive part of the problem becomes much harder to infer from compressed leaf summaries.")
    lines.append("- The heatmaps are therefore best read as support-trend plots, while the rightmost bar plot is the place to read the absolute magnitude of the remaining error.")
    lines.append("")
    if "boundary_controls_png" in fig_paths:
        control_rows = _collect_boundary_control_rows(output_root)
        bow_doc_true = _median_for_control(
            control_rows,
            topic_process="bag_of_words",
            leaf_tokens=384,
            lambda_multiplier=2.0,
            metric_key="ridge_true_topics",
        )
        bow_doc_ridge = _median_for_control(
            control_rows,
            topic_process="bag_of_words",
            leaf_tokens=384,
            lambda_multiplier=2.0,
            metric_key="ridge",
        )
        bow_two_under = _median_for_control(
            control_rows,
            topic_process="bag_of_words",
            leaf_tokens=192,
            lambda_multiplier=2.0,
            metric_key="undersupported",
        )
        bow_two_true = _median_for_control(
            control_rows,
            topic_process="bag_of_words",
            leaf_tokens=192,
            lambda_multiplier=2.0,
            metric_key="ridge_true_topics",
        )
        seg_two_under = _median_for_control(
            control_rows,
            topic_process="segments",
            leaf_tokens=192,
            lambda_multiplier=2.0,
            metric_key="undersupported",
        )
        seg_two_ridge = _median_for_control(
            control_rows,
            topic_process="segments",
            leaf_tokens=192,
            lambda_multiplier=2.0,
            metric_key="ridge",
        )
        lines.append("**Whole-Document And One-Boundary Controls**")
        lines.append(r"- A useful bridge control is to collapse the tree all the way down to one leaf by setting $\mathrm{leaf\_tokens} = 384$. Then there are no internal boundaries at all, so `exact` and `undersupported` coincide.")
        lines.append(r"- That one-leaf setting answers a narrow question: can the method learn the document-level realized oracle when no merge correction is needed?")
        lines.append(r"- The next nontrivial control is the $2$-leaf tree with $\mathrm{leaf\_tokens} = 192$. That introduces exactly one internal boundary, so it is the simplest setting in which a cross-leaf bigram correction can matter.")
        lines.append(r"- Running both bag-of-words LDA and Segment-LDA in that $2$-leaf geometry cleanly separates three issues: document-level recovery, mixed-leaf compression, and one-boundary merge correction.")
        lines.append("")
        lines.append(f"![]({fig_paths['boundary_controls_png']}){{ width=98% }}")
        lines.append("")
        lines.append("**How To Read The Control Figure**")
        lines.append(r"- Columns move from easiest to most structured: whole-document bag-of-words LDA, two-leaf bag-of-words LDA, and two-leaf Segment-LDA.")
        lines.append(r"- Rows compare $\lambda=0$ and $\lambda=2$. The upper row is count-only recovery; the lower row adds the boundary-sensitive bigram term.")
        lines.append("- Every panel shows raw `root_mae` versus `train_docs` on a symlog scale. Lower is better.")
        lines.append(r"- In the whole-document column, `undersupported` sits exactly on top of `exact`, because there is no internal boundary to miss.")
        lines.append(r"- In the two-leaf columns, any separation between `undersupported` and `exact` is the price of missing the single cross-leaf boundary term.")
        lines.append("")
        lines.append("**What The Expanded Controls Show**")
        lines.append(rf"- The one-leaf bag-of-words control confirms that the document-level problem itself is learnable: at high support and $\lambda=2$, the true-topic ridge fit is about ${bow_doc_true:.1e}$.")
        lines.append(rf"- But that same one-leaf control also shows why it is not enough as the main benchmark. With inferred hard-topic features, one leaf means one inferred topic for the entire document, and the high-support root MAE stays around ${bow_doc_ridge:.2f}$ under bag-of-words LDA.")
        lines.append(rf"- The two-leaf bag-of-words control is therefore the first genuinely tree-relevant ordinary-LDA test. It has exactly one boundary, but each leaf is still internally mixed; at high support and $\lambda=2$, the undersupported baseline is about ${bow_two_under:.2f}$, the true-topic ridge fit is about ${bow_two_true:.1e}$, and inferred ridge remains much larger.")
        lines.append(rf"- The two-leaf Segment-LDA control is the clean positive case: exactly one boundary and leaf-aligned topic blocks. There the undersupported baseline misses one cross-leaf term, staying around ${seg_two_under:.2f}$ at high support, while ridge falls to about ${seg_two_ridge:.1e}$.")
        lines.append("")
    lines.append(r"\newpage")
    lines.append("")
    lines.append("## Segment-LDA OPS Weight Recovery")
    lines.append("")
    lines.append("**What Changes Relative To Ordinary LDA**")
    lines.append(r"- The topic-word emissions $\phi_1, \ldots, \phi_K$ and the oracle $f^\star(A)$ stay the same.")
    lines.append("- The only change is that latent topics now persist for contiguous segments instead of being resampled independently token by token.")
    lines.append("- In the curated slice those segments are aligned to leaf boundaries, so a leaf is typically a single topic block.")
    lines.append("")
    lines.append("**Generative Model**")
    lines.append("$$")
    lines.append(r"\pi_d \sim \mathrm{Dirichlet}(\alpha \mathbf{1}_K), \qquad s_{d,m} \sim \mathrm{Categorical}(\pi_d)\ \text{with}\ s_{d,m} \neq s_{d,m-1}, \qquad z_{d,t} = s_{d,m}\ \text{for}\ t\ \text{in segment}\ m.")
    lines.append("$$")
    lines.append("- The curated runs also fix `align_segments_to_leaves=true`, so topic boundaries occur at leaf boundaries.")
    lines.append(r"- In this slice, a $16$-token leaf is intentionally a pure topic block. Empirically, held-out leaf-topic purity is $1.0$ in the curated segmented runs.")
    lines.append("")
    lines.append("**Oracle**")
    lines.append("$$")
    lines.append(r"f^\star(A) = \sum_{k=1}^K \theta_k\, c_k(A) + \lambda \sum_{i=1}^K \sum_{j=1}^K W_{ij}\, b_{ij}(A).")
    lines.append("$$")
    lines.append("- The oracle is exactly the same as in the plain-LDA page above. What changes is only the latent topic process.")
    lines.append("- So any difference between the ordinary-LDA page and this page should be interpreted as coming from the topic process and the resulting leaf geometry, not from a new scoring rule.")
    lines.append(r"- In both pages the sequence is: sample a realized latent topic path $(z_{d,1}, \ldots, z_{d,n})$, emit observed words from that path, and then evaluate $f^\star(A)$ as a deterministic function of the realized latent topics inside span $A$.")
    lines.append("")
    lines.append("**Three-Topic Worked Example With Leaf-Aligned Segments**")
    lines.append(r"- Keep the same numbers as above: $\theta = (1.0, 0.5, -0.25)$, $\lambda = 2$, and the same nonzero $W_{ij}$ values.")
    lines.append(r"- Now consider two pure-topic leaves: $L=(1,1,1,1)$ and $R=(2,2,2,2)$.")
    lines.append(r"- For $L$, the count term is $4.0$ and the bigram term is $2 \cdot (3 \cdot 0.10) = 0.6$, so $f^\star(L)=4.6$.")
    lines.append(r"- For $R$, the count term is $2.0$ and the bigram term is $2 \cdot (3 \cdot 0.05) = 0.3$, so $f^\star(R)=2.3$.")
    lines.append(r"- Their sum is $6.9$.")
    lines.append(r"- But the parent span $(1,1,1,1,2,2,2,2)$ contains one additional cross-leaf boundary pair $1 \to 2$, whose contribution is $\lambda W_{12} = 2 \cdot 0.60 = 1.2$.")
    lines.append(r"- So the true parent score is $6.9 + 1.2 = 8.1$.")
    lines.append("- This is the clean Segment-LDA setting: once each leaf is close to a single topic, the main remaining task is to learn the count weights and the boundary correction weights.")
    lines.append(r"- In this case an exact summary only needs the right child-boundary metadata to add the missing $1.2$, while an undersupported summary that only adds child totals will systematically miss it.")
    lines.append("")
    lines.append(f"![]({fig_paths['lda_worked_examples_png']}){{ width=98% }}")
    lines.append("")
    lines.append("- The right-hand panel of this toy figure is the Segment-LDA case: two pure-topic leaves plus one cross-leaf boundary correction.")
    lines.append("")
    lines.append("**Methods Compared**")
    lines.append("- `exact`: the exact sketch/control ceiling.")
    lines.append("- `ridge`: intended estimator using queried spans.")
    lines.append("- `ridge_true_topics`: favorable upper baseline with true topics available.")
    lines.append("- `undersupported`: summary family missing the boundary information needed when the bigram term matters.")
    lines.append("")
    lines.append("**What One Run Actually Does**")
    lines.append("1. Draw segmented documents from the same topic-word matrix family used above, but force topic changepoints to occur only at leaf boundaries.")
    lines.append("2. Query every leaf span and a budgeted subset of internal spans.")
    lines.append("3. Build span features from leaf-level inferred topics or true topics, depending on the method being evaluated.")
    lines.append(r"4. Fit ridge to recover $[\theta, \lambda W]$ from the queried spans.")
    lines.append("5. Evaluate root, merge, and weight-recovery errors on held-out documents.")
    lines.append("")
    lines.append("**What We Expect To See Under Segment-LDA**")
    lines.append(r"- The exact control should again stay at $0$.")
    lines.append(r"- $\lambda=0$ should be easy, because only topic counts matter and the leaf summaries are already aligned with single-topic blocks.")
    lines.append(r"- When $\lambda>0$, the undersupported estimator should remain separated from $0$, because it cannot reconstruct the cross-leaf boundary pair like the $1 \to 2$ term in the toy example above.")
    lines.append(r"- The ridge and true-topic ridge fits should move toward the exact ceiling as support increases, and in this curated slice they should be close because $\mathrm{topic\_phi\_estimator}=\mathrm{true}$ removes most upstream topic-estimation error.")
    lines.append("")
    lines.append("**How To Read The Figure**")
    lines.append(r"- Left panel: normalized control gap in the $\lambda=0$ slice. Because exact and undersupported both equal $0$ there, green is still the exact ceiling but red is defined as the worst observed ridge root MAE in that control slice.")
    lines.append(r"- Middle panel: within-Segment-LDA normalized gap for the boundary-sensitive $\lambda=2$ regime. Green means close to exact; red means close to undersupported.")
    lines.append("- Right panel: raw high-support `root_mae` by method. This is the reasonable direct comparison for this family.")
    lines.append("")
    lines.append(f"![]({fig_paths['segment_family_png']}){{ width=98% }}")
    lines.append("")
    lines.append("**Paper-Style Interpretation**")
    lines.append(r"- The observed pattern matches the expected clean case. In the $\lambda=0$ control, ridge drives root error essentially to zero; the bar plot shows that the absolute errors are tiny.")
    lines.append(r"- In the $\lambda=2$ regime, the gap between ridge and undersupported is the substantive signal. The latter misses the boundary correction term, while ridge moves toward the exact ceiling as support increases.")
    lines.append(r"- At the highest-support segmented setting in this curated slice, ridge root MAE is about $2.1 \times 10^{-5}$, while undersupported root MAE is about $2.83 \times 10^{-1}$ and the true-topic ridge fit is numerically indistinguishable from ridge.")
    lines.append("- So the main reading of this page is straightforward: once the latent topic process produces leaf-aligned topic blocks, the remaining difficulty is the intended one, namely learning the count weights and the cross-boundary bigram correction.")
    lines.append("")
    seg_warns = _warning_lines(records, "segment_lda_ops_weight_recovery")
    lines.append("- In this curated grid, the exact ceiling, boundary-sensitive separation, and support-trend checks pass.")
    lines.extend(seg_warns or ["- No warnings recorded."])
    lines.append("")
    lines.append(r"\newpage")
    lines.append("")
    lines.append("## Segmented-LDA C-TreePO")
    lines.append("")
    lines.append("**DGP**")
    lines.append("- Books are segmented topic documents with LDA-style word emissions.")
    lines.append("- This benchmark is end-to-end: topic-word estimation, leaf-level topic-mixture estimation/calibration, then tree aggregation with optional evaluation-time oracle guidance.")
    lines.append("- The report focuses on end-to-end root error, not direct comparability with the Segment-LDA weight-recovery family above.")
    lines.append("")
    lines.append("**Methods Compared**")
    lines.append("- `oracle_tree`: end-to-end ceiling with true leaf/topic information.")
    lines.append("- `estimated_uncalibrated`: learned pipeline without calibration/query-budget improvements.")
    lines.append("- `estimated_calibrated_budgeted`: intended query-budgeted path.")
    lines.append("")
    lines.append("**How To Read The Figure**")
    lines.append("- Left panels: within-C-TreePO normalized gap. Green means close to `oracle_tree`; red means close to `estimated_uncalibrated`.")
    lines.append("- Right panel: raw high-support `root_l1_mean` by method, which is the direct same-family comparison.")
    lines.append("")
    lines.append(f"![]({fig_paths['ctree_family_png']}){{ width=98% }}")
    lines.append("")
    ctree_warns = _warning_lines(records, "segmented_lda_ctreepo")
    lines.extend(ctree_warns or ["- No warnings recorded."])
    lines.append("")
    lines.append(r"\newpage")
    lines.append("")
    lines.append("## Mergeable Ablations")
    lines.append("")
    lines.append("**DGP**")
    lines.append("- Documents are toy token-score sequences with spike patterns such as boundary spikes, interior spikes, and multi-spike cases.")
    lines.append("- The default target is non-additive: document-level success depends on spike structure rather than a simple average.")
    lines.append("- The sweep varies chunk size and chunk budget to show when repeated aggregation preserves the target and when naive aggregation fails.")
    lines.append("")
    lines.append("**Methods Compared**")
    lines.append("- Chunk-quality heatmap: aligned repeated aggregation compared against the one-pass oracle reference.")
    lines.append("- Target-`k` bar chart: supported sketch order (`m=5`) versus unsupported (`m=2`) and naive baselines.")
    lines.append("")
    lines.append("**How To Read The Figure**")
    lines.append("- Left panel: within-Mergeable normalized gap across chunk size and chunk budget.")
    lines.append("- Right panel: raw `mean_abs_bias` for supported, unsupported, and naive methods at `target_k=5`.")
    lines.append("")
    lines.append(f"![]({fig_paths['mergeable_family_png']}){{ width=98% }}")
    lines.append("")
    merge_warns = _warning_lines(records, "mergeable_ablation")
    lines.append("- This family is the explicit exception to naive leaf monotonicity: the relevant question is whether supported configurations approach the one-pass reference, not whether smaller chunks monotonically help.")
    lines.extend(merge_warns or ["- No warnings recorded."])
    lines.append("")
    lines.append(r"\newpage")
    lines.append("")
    lines.append("## Overall Reading")
    lines.append("")
    if int(summary.get("n_fail", 0)) == 0:
        lines.append("- The curated suite has no hard failures.")
    else:
        lines.append(f"- The curated suite still has `{_fmt_count(summary.get('n_fail'))}` hard failures.")
    lines.append("- The remaining warnings are concentrated in Markov merge/granularity diagnostics, which remain visible without being mislabeled as end-to-end regressions.")
    lines.append("- The report now makes only within-family visual claims. Cross-family comparisons should be made from the text and the expectation statuses, not by comparing color intensity across different DGPs.")
    lines.append("- The intended claim is: each family works in its own intended setting, mismatch baselines stay separated where they should, and added support moves the intended method toward the family-specific ceiling/reference.")
    lines.append("")

    utility_suite = _find_exact_utility_artifacts(output_root)
    if utility_suite is not None:
        lines.append(r"\newpage")
        lines.append("")
        lines.append("## Exact Utility Transport / TreePO")
        lines.append("")
        lines.append("- This suite is the TreePO preference/utility side reframed in the theorem-aligned way: oracle-indexed utility transport first, with DPO/GRPO/PPO as objective-family instances.")
        lines.append("- The main reading is whether zero utility regret coincides with latent-state recovery, and whether tree structure only helps in the tree-relevant controls.")
        summary_payload = _load_json(utility_suite["summary"])
        util_summary = dict(summary_payload.get("summary", {}) or {})
        lines.append(
            f"- Utility-suite findings: `{_fmt_count(util_summary.get('n_pass'))} pass / {_fmt_count(util_summary.get('n_warn'))} warn / {_fmt_count(util_summary.get('n_fail'))} fail / {_fmt_count(util_summary.get('n_not_applicable'))} n/a`."
        )
        if utility_suite["figure"].exists():
            rel_fig = os.path.relpath(utility_suite["figure"], md_path.parent)
            lines.append("")
            lines.append(f"![]({rel_fig}){{ width=98% }}")
            lines.append("")
        if utility_suite["report_md"].exists():
            rel_md = os.path.relpath(utility_suite["report_md"], md_path.parent)
            lines.append(f"- Standalone detailed report: `{rel_md}`.")
        lines.append("")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    pdf_emitted = False
    if bool(args.emit_pdf):
        try:
            pdf_emitted = _run_pandoc(md_path, pdf_path)
        except Exception:
            pdf_emitted = False

    diag = {
        "output_root": str(output_root),
        "output_markdown": str(md_path),
        "output_pdf": str(pdf_path) if pdf_emitted else None,
        "pdf_emitted": bool(pdf_emitted),
        "summary": summary,
        "figures": fig_paths,
    }
    (md_path.parent / "simulation_intent_report_diagnostics.json").write_text(
        json.dumps(diag, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(diag, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
