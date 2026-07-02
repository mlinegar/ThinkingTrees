#!/usr/bin/env python3
"""Plot per-dimension Manifesto f/g ladder metrics.

The joint all-six ladder stores vector-task metrics inside each
``SplitMetrics.per_dimension`` payload. This post-processor expands those
metric dictionaries into flat rows and writes per-dimension plots without
rerunning any model inference.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import numpy as np

from plot_manifesto_fg_ladder_grid import (  # noqa: E402
    REPO_ROOT,
    _as_math_stage_label,
    _parse_datetime,
    _rel,
    _safe_float,
    _safe_int,
    _source_priority,
    _stage_label_math,
    _stage_sort_key,
    _utc_now,
)


DEFAULT_ROOT_GLOB = "combined_benoit_*"
DIMENSION_ORDER = (
    "economic",
    "social",
    "immigration",
    "eu",
    "environment",
    "decentralization",
)
METRIC_FIELDS = (
    "internal_f_pearson",
    "external_expert_pearson",
    "f_star_gap",
    "internal_f_mae",
    "external_expert_mae",
    "mean_prediction",
    "mean_teacher",
    "mean_expert",
    "internal_f_mae_1_7",
    "external_expert_mae_1_7",
    "mean_prediction_1_7",
    "mean_teacher_1_7",
    "mean_expert_1_7",
)
METRIC_FALLBACKS = {
    "internal_f_mae": "internal_f_mae_1_7",
    "external_expert_mae": "external_expert_mae_1_7",
    "mean_prediction": "mean_prediction_1_7",
    "mean_teacher": "mean_teacher_1_7",
    "mean_expert": "mean_expert_1_7",
    "internal_f_mae_1_7": "internal_f_mae",
    "external_expert_mae_1_7": "external_expert_mae",
    "mean_prediction_1_7": "mean_prediction",
    "mean_teacher_1_7": "mean_teacher",
    "mean_expert_1_7": "mean_expert",
}
CSV_FIELDS = (
    "dimension",
    "family",
    "axis_kind",
    "axis_value",
    "leaf_count",
    "leaf_size_tokens",
    "iteration",
    "stage_name",
    "stage_label",
    "trained",
    "n_internal",
    "n_external",
    *METRIC_FIELDS,
    "source_type",
    "source_root",
    "source_path",
    "source_created_at",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dimension_sort_key(dim: str) -> tuple[int, str]:
    try:
        return (DIMENSION_ORDER.index(str(dim)), str(dim))
    except ValueError:
        return (999, str(dim))


def _axis_label(row: Mapping[str, Any]) -> str:
    leaf_size = _safe_int(row.get("leaf_size_tokens"))
    if leaf_size is not None:
        return str(leaf_size)
    leaf_count = _safe_int(row.get("leaf_count") or row.get("axis_value"))
    return f"L={leaf_count}" if leaf_count is not None else "unknown"


def _stage_label(row: Mapping[str, Any]) -> str:
    return str(row.get("stage_label") or row.get("stage_name") or "")


def _fmt(value: Any) -> str:
    parsed = _safe_float(value)
    return "n/a" if parsed is None else f"{parsed:.3f}"


def _metric_value(row: Mapping[str, Any], metric: str) -> Optional[float]:
    value = _safe_float(row.get(metric))
    if value is not None:
        return value
    fallback = METRIC_FALLBACKS.get(str(metric))
    if fallback:
        return _safe_float(row.get(fallback))
    return None


def _metric_count(row: Mapping[str, Any]) -> int:
    return sum(_metric_value(row, field) is not None for field in METRIC_FIELDS)


def _normalize_dimension_row(
    *,
    dimension: str,
    base: Mapping[str, Any],
    metrics: Mapping[str, Any],
    source_type: str,
    source_root: Path,
    source_path: Path,
    source_created_at: Any = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dimension": str(dimension),
        "family": str(base.get("family") or "unknown"),
        "axis_kind": base.get("axis_kind")
        or ("leaf_size_tokens" if base.get("leaf_size_tokens") is not None else "leaf_count"),
        "axis_value": _safe_int(base.get("axis_value")),
        "leaf_count": _safe_int(base.get("leaf_count")),
        "leaf_size_tokens": _safe_int(base.get("leaf_size_tokens")),
        "iteration": _safe_int(base.get("iteration")),
        "stage_name": base.get("stage_name"),
        "stage_label": base.get("stage_label") or base.get("stage_name"),
        "trained": base.get("trained"),
        "n_internal": _safe_int(metrics.get("n_internal")),
        "n_external": _safe_int(metrics.get("n_external")),
        "source_type": source_type,
        "source_root": _rel(source_root),
        "source_path": _rel(source_path),
        "source_created_at": str(source_created_at or base.get("created_at") or ""),
    }
    if row["axis_value"] is None:
        row["axis_value"] = row["leaf_size_tokens"] or row["leaf_count"]
    for field in METRIC_FIELDS:
        row[field] = _safe_float(metrics.get(field))
    for field, fallback in METRIC_FALLBACKS.items():
        if row.get(field) is None and fallback in row:
            row[field] = row.get(fallback)
    return row


def _rows_from_payload(
    payload: Mapping[str, Any],
    *,
    eval_split: str,
    source_type: str,
    source_root: Path,
    source_path: Path,
    source_created_at: Any = None,
) -> list[dict[str, Any]]:
    split_metrics = payload.get("split_metrics")
    if not isinstance(split_metrics, Mapping):
        return []
    metrics = split_metrics.get(eval_split) or split_metrics.get("all") or {}
    if not isinstance(metrics, Mapping):
        return []
    per_dimension = metrics.get("per_dimension") or {}
    if not isinstance(per_dimension, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for dimension, dim_metrics in per_dimension.items():
        if isinstance(dim_metrics, Mapping):
            rows.append(
                _normalize_dimension_row(
                    dimension=str(dimension),
                    base=payload,
                    metrics=dim_metrics,
                    source_type=source_type,
                    source_root=source_root,
                    source_path=source_path,
                    source_created_at=source_created_at,
                )
            )
    return rows


def _rows_from_iteration_history(
    path: Path,
    *,
    source_root: Path,
    eval_split: str,
) -> list[dict[str, Any]]:
    try:
        payload = _read_json(path)
    except Exception as exc:
        print(f"warning: failed to read {path}: {exc}", file=sys.stderr)
        return []
    if not isinstance(payload, Mapping):
        return []
    base = {
        "family": payload.get("family"),
        "axis_kind": payload.get("axis_kind"),
        "axis_value": payload.get("axis_value"),
        "leaf_count": payload.get("leaf_count"),
        "leaf_size_tokens": payload.get("leaf_size_tokens"),
    }
    rows: list[dict[str, Any]] = []
    for iteration in payload.get("iterations") or []:
        if not isinstance(iteration, Mapping):
            continue
        merged = {**base, **dict(iteration)}
        rows.extend(
            _rows_from_payload(
                merged,
                eval_split=eval_split,
                source_type="iteration_history",
                source_root=source_root,
                source_path=path,
            )
        )
    return rows


def _rows_from_checkpoint(
    path: Path,
    *,
    source_root: Path,
    eval_split: str,
) -> list[dict[str, Any]]:
    try:
        payload = _read_json(path)
    except Exception as exc:
        print(f"warning: failed to read {path}: {exc}", file=sys.stderr)
        return []
    if not isinstance(payload, Mapping) or payload.get("phase") != "post_eval":
        return []
    return _rows_from_payload(
        payload,
        eval_split=eval_split,
        source_type="checkpoint",
        source_root=source_root,
        source_path=path,
        source_created_at=payload.get("created_at"),
    )


def _resolve_input_roots(inputs: Sequence[Path]) -> list[Path]:
    if inputs:
        return [path.resolve() for path in inputs]
    base = REPO_ROOT / "outputs" / "manifesto_fg_alternating"
    if not base.exists():
        return []
    roots = [path for path in base.glob(DEFAULT_ROOT_GLOB) if path.is_dir()]
    return sorted(roots, key=lambda path: path.stat().st_mtime)


def _collect_rows(
    roots: Sequence[Path],
    *,
    eval_split: str,
    include_partial: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in roots:
        ladder_dir = root / "ladder" if (root / "ladder").exists() else root
        for path in ladder_dir.glob("*/leaf*/iteration_history.json"):
            rows.extend(
                _rows_from_iteration_history(path, source_root=root, eval_split=eval_split)
            )
        if include_partial:
            for path in ladder_dir.glob("*/leaf*/step_checkpoints/iter_*_post_eval.json"):
                rows.extend(_rows_from_checkpoint(path, source_root=root, eval_split=eval_split))
    return _dedupe_rows(rows)


def _dedupe_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            row.get("dimension"),
            row.get("family"),
            row.get("axis_kind"),
            row.get("axis_value"),
            row.get("leaf_count"),
            row.get("leaf_size_tokens"),
            row.get("iteration"),
            row.get("stage_name"),
        )
        score = (
            _metric_count(row),
            _source_priority(str(row.get("source_type") or "")),
            _parse_datetime(row.get("source_created_at")),
        )
        current = best.get(key)
        if current is None:
            best[key] = row
            continue
        current_score = (
            _metric_count(current),
            _source_priority(str(current.get("source_type") or "")),
            _parse_datetime(current.get("source_created_at")),
        )
        if score >= current_score:
            best[key] = row
    return sorted(
        best.values(),
        key=lambda row: (
            _dimension_sort_key(str(row.get("dimension") or "")),
            str(row.get("family") or ""),
            _safe_int(row.get("leaf_size_tokens")) or _safe_int(row.get("leaf_count")) or 0,
            _stage_sort_key(row.get("stage_name"), row.get("iteration")),
        ),
    )


def _finite_metric_rows(rows: Sequence[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if _safe_int(row.get("leaf_size_tokens")) is not None
        and _metric_value(row, metric) is not None
    ]


def _metric_limits(
    rows: Sequence[dict[str, Any]],
    metric: str,
    *,
    force_unit: bool = False,
) -> tuple[Optional[float], Optional[float]]:
    vals = [_metric_value(row, metric) for row in rows]
    finite = [float(v) for v in vals if v is not None]
    if not finite:
        return None, None
    observed_min = min(finite)
    observed_max = max(finite)
    if force_unit:
        return min(0.0, observed_min), max(1.0, observed_max)
    pad = 0.04 * max(1e-9, observed_max - observed_min)
    return observed_min - pad, observed_max + pad


def _write_line_facets(
    rows: Sequence[dict[str, Any]],
    output: Path,
    *,
    metric: str,
    figure_title: str,
    ylabel: str,
    lower_is_better: bool = False,
) -> None:
    metric_rows = _finite_metric_rows(rows, metric)
    dimensions = sorted({str(row["dimension"]) for row in metric_rows}, key=_dimension_sort_key)
    if not dimensions:
        return
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=False, sharey=True)
    flat_axes = list(axes.flat)
    stages = sorted(
        {str(row.get("stage_name") or "") for row in metric_rows},
        key=lambda stage: _stage_sort_key(stage),
    )
    colors = plt.get_cmap("tab10")
    global_min, global_max = _metric_limits(
        metric_rows,
        metric,
        force_unit=metric.endswith("pearson"),
    )
    if lower_is_better:
        global_min = min(0.0, global_min or 0.0)
    for ax, dim in zip(flat_axes, dimensions):
        dim_rows = [row for row in metric_rows if str(row["dimension"]) == dim]
        for stage_idx, stage in enumerate(stages):
            stage_rows = [
                row
                for row in dim_rows
                if str(row.get("stage_name") or "") == stage
            ]
            if not stage_rows:
                continue
            stage_rows = sorted(stage_rows, key=lambda row: int(row["leaf_size_tokens"]))
            xs = np.asarray([int(row["leaf_size_tokens"]) for row in stage_rows], dtype=float)
            ys = np.asarray(
                [float(_metric_value(row, metric) or 0.0) for row in stage_rows],
                dtype=float,
            )
            label = _stage_label_math(stage_rows[0])
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                label=label,
                color=colors(stage_idx % 10),
            )
        ax.set_title(dim)
        ax.set_xscale("log", base=2)
        leaf_values = sorted({int(row["leaf_size_tokens"]) for row in dim_rows})
        ax.xaxis.set_major_locator(mticker.FixedLocator(leaf_values))
        ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(value) for value in leaf_values]))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        for label in ax.get_xticklabels():
            label.set_rotation(25)
            label.set_ha("right")
        ax.set_xlabel("leaf tokens")
        ax.grid(alpha=0.25)
        if metric == "f_star_gap":
            ax.axhline(0.0, color="#333333", linewidth=1.0, alpha=0.7)
    for ax in flat_axes[len(dimensions):]:
        ax.axis("off")
    if global_min is not None and global_max is not None:
        for ax in flat_axes[: len(dimensions)]:
            ax.set_ylim(global_min, global_max)
    flat_axes[0].set_ylabel(ylabel)
    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.93),
            ncol=min(6, len(handles)),
            frameon=False,
        )
    fig.suptitle(figure_title, fontsize=14, y=0.985)
    fig.subplots_adjust(top=0.84, hspace=0.44, wspace=0.22, bottom=0.10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _heatmap_matrix(
    rows: Sequence[dict[str, Any]],
    *,
    dimension: str,
    metric: str,
) -> tuple[np.ndarray, list[int], list[str]]:
    metric_rows = [
        row
        for row in _finite_metric_rows(rows, metric)
        if str(row.get("dimension") or "") == str(dimension)
    ]
    leaf_values = sorted({int(row["leaf_size_tokens"]) for row in metric_rows})
    stages = sorted(
        {str(row.get("stage_name") or "") for row in metric_rows},
        key=lambda stage: _stage_sort_key(stage),
    )
    matrix = np.full((len(stages), len(leaf_values)), np.nan, dtype=float)
    leaf_index = {value: idx for idx, value in enumerate(leaf_values)}
    stage_index = {stage: idx for idx, stage in enumerate(stages)}
    for row in metric_rows:
        matrix[
            stage_index[str(row.get("stage_name") or "")],
            leaf_index[int(row["leaf_size_tokens"])],
        ] = float(_metric_value(row, metric) or float("nan"))
    stage_labels: list[str] = []
    for stage in stages:
        matching = [row for row in metric_rows if str(row.get("stage_name") or "") == stage]
        stage_labels.append(_stage_label_math(matching[0]) if matching else _as_math_stage_label(stage))
    return matrix, leaf_values, stage_labels


def _annotate_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    cmap: Any,
    norm: Any,
) -> None:
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix[y, x]
            if not math.isfinite(float(value)):
                continue
            rgba = cmap(norm(float(value)))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "white" if luminance < 0.55 else "#111111"
            ax.text(
                x,
                y,
                f"{float(value):.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )


def _write_heatmap_facets(
    rows: Sequence[dict[str, Any]],
    output: Path,
    *,
    metric: str,
    figure_title: str,
    colorbar_label: str,
    cmap_name: str,
    diverging_zero: bool = False,
    force_unit: bool = False,
) -> None:
    metric_rows = _finite_metric_rows(rows, metric)
    dimensions = sorted({str(row["dimension"]) for row in metric_rows}, key=_dimension_sort_key)
    if not dimensions:
        return
    vmin, vmax = _metric_limits(metric_rows, metric, force_unit=force_unit)
    if vmin is None or vmax is None:
        return
    if diverging_zero:
        vmin = min(vmin, -0.05)
        vmax = max(vmax, 0.05)
        norm: Any = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2))
    flat_axes = list(axes.flat)
    image = None
    for ax, dim in zip(flat_axes, dimensions):
        matrix, leaf_values, stages = _heatmap_matrix(rows, dimension=dim, metric=metric)
        masked = np.ma.masked_invalid(matrix)
        image = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(dim)
        ax.set_xticks(np.arange(len(leaf_values)))
        ax.set_xticklabels([str(value) for value in leaf_values], rotation=25, ha="right")
        ax.set_yticks(np.arange(len(stages)))
        ax.set_yticklabels(stages)
        ax.set_xlabel("leaf tokens")
        _annotate_heatmap(ax, matrix, cmap=cmap, norm=norm)
    for ax in flat_axes[len(dimensions):]:
        ax.axis("off")
    if image is not None:
        cbar = fig.colorbar(image, ax=flat_axes[: len(dimensions)], fraction=0.026, pad=0.025)
        cbar.set_label(colorbar_label)
    fig.suptitle(figure_title, fontsize=14, y=0.985)
    fig.subplots_adjust(top=0.91, hspace=0.42, wspace=0.32, bottom=0.09, right=0.88)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def _best_rows_by_dimension(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        metric = _safe_float(row.get("external_expert_pearson"))
        if metric is None:
            continue
        dim = str(row.get("dimension") or "")
        current = best.get(dim)
        if current is None or metric > float(current["external_expert_pearson"]):
            best[dim] = row
    return [best[dim] for dim in sorted(best, key=_dimension_sort_key)]


def _write_markdown(rows: Sequence[dict[str, Any]], path: Path, *, roots: Sequence[Path]) -> None:
    best_rows = _best_rows_by_dimension(rows)
    lines = [
        "# Manifesto f/g Ladder Dimension Breakdown",
        "",
        f"Generated: `{_utc_now()}`",
        "",
        "## Inputs",
    ]
    for root in roots:
        lines.append(f"- `{_rel(root)}`")
    lines.extend(
        [
            "",
            "## Best External Pearson By Dimension",
            "",
            "| dimension | best ext_p | leaf | stage | ext_mae | int_p | gap | n |",
            "|---|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in best_rows:
        lines.append(
            "| {dim} | {ext_p} | {leaf} | {stage} | {ext_mae} | {int_p} | {gap} | {n} |".format(
                dim=row.get("dimension"),
                ext_p=_fmt(row.get("external_expert_pearson")),
                leaf=_axis_label(row),
                stage=_stage_label(row),
                ext_mae=_fmt(_metric_value(row, "external_expert_mae")),
                int_p=_fmt(row.get("internal_f_pearson")),
                gap=_fmt(row.get("f_star_gap")),
                n=row.get("n_external"),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `ext_p` is Pearson r against the gold expert dimension score.",
            "- `int_p` is Pearson r against the teacher f dimension score.",
            "- `gap` is `int_p - ext_p`; larger positive values are the reward-hacking warning direction.",
            "- The CSV contains every dimension, leaf, and f/g stage row.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and plot per-dimension Manifesto f/g ladder metrics."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        action="append",
        default=[],
        help="Run root or ladder dir. May be repeated.",
    )
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--family", default="dspy")
    parser.add_argument(
        "--dimensions",
        default="",
        help="Optional comma-separated dimension filter. Empty includes all dimensions.",
    )
    parser.add_argument(
        "--stages",
        default="",
        help="Optional comma-separated stage filter, e.g. 'f1g0,f1g1'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "manifesto_fg_alternating"
        / "benoit_dimension_breakdown_plots",
    )
    parser.add_argument(
        "--figure-title-prefix",
        default="Manifesto all-six joint f/g ladder",
    )
    parser.add_argument("--no-partial", action="store_true")
    return parser.parse_args(argv)


def _parse_csv_filter(value: str) -> set[str]:
    return {
        token.strip()
        for token in str(value or "").replace(";", ",").split(",")
        if token.strip()
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    roots = _resolve_input_roots(args.input_root)
    rows = _collect_rows(
        roots,
        eval_split=str(args.eval_split),
        include_partial=not bool(args.no_partial),
    )
    rows = [row for row in rows if str(row.get("family") or "") == str(args.family)]
    allowed_dimensions = _parse_csv_filter(str(args.dimensions or ""))
    if allowed_dimensions:
        rows = [row for row in rows if str(row.get("dimension") or "") in allowed_dimensions]
    allowed_stages = _parse_csv_filter(str(args.stages or ""))
    if allowed_stages:
        rows = [row for row in rows if str(row.get("stage_name") or "") in allowed_stages]
    rows = [row for row in rows if _metric_count(row) > 0]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_dir / "manifesto_fg_ladder_dimension_rows.csv")
    _write_line_facets(
        rows,
        output_dir / "manifesto_fg_ladder_dimension_ext_pearson.png",
        metric="external_expert_pearson",
        figure_title=f"{args.figure_title_prefix}: external Pearson by dimension",
        ylabel="external Pearson r",
    )
    _write_heatmap_facets(
        rows,
        output_dir / "manifesto_fg_ladder_dimension_ext_pearson_heatmap.png",
        metric="external_expert_pearson",
        figure_title=f"{args.figure_title_prefix}: external Pearson heatmaps",
        colorbar_label="Pearson r against expert (higher = better)",
        cmap_name="viridis",
        force_unit=True,
    )
    _write_heatmap_facets(
        rows,
        output_dir / "manifesto_fg_ladder_dimension_gap_heatmap.png",
        metric="f_star_gap",
        figure_title=f"{args.figure_title_prefix}: internal-external gap heatmaps",
        colorbar_label="internal Pearson minus external Pearson",
        cmap_name="coolwarm",
        diverging_zero=True,
    )
    _write_markdown(
        rows,
        output_dir / "manifesto_fg_ladder_dimension_summary.md",
        roots=roots,
    )

    best_rows = _best_rows_by_dimension(rows)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_roots": [_rel(root) for root in roots],
        "n_rows": len(rows),
        "dimensions": sorted({str(row.get("dimension") or "") for row in rows}, key=_dimension_sort_key),
        "stage_filter": sorted(allowed_stages) or None,
        "dimension_filter": sorted(allowed_dimensions) or None,
        "best_external_pearson_by_dimension": [
            {
                "dimension": row.get("dimension"),
                "external_expert_pearson": row.get("external_expert_pearson"),
                "leaf_size_tokens": row.get("leaf_size_tokens"),
                "stage_name": row.get("stage_name"),
                "stage_label": row.get("stage_label"),
                "n_external": row.get("n_external"),
            }
            for row in best_rows
        ],
        "outputs": {
            "csv": _rel(output_dir / "manifesto_fg_ladder_dimension_rows.csv"),
            "line_png": _rel(output_dir / "manifesto_fg_ladder_dimension_ext_pearson.png"),
            "external_heatmap_png": _rel(output_dir / "manifesto_fg_ladder_dimension_ext_pearson_heatmap.png"),
            "gap_heatmap_png": _rel(output_dir / "manifesto_fg_ladder_dimension_gap_heatmap.png"),
            "markdown": _rel(output_dir / "manifesto_fg_ladder_dimension_summary.md"),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
