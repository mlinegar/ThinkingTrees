#!/usr/bin/env python3
"""Plot Manifesto f/g alternating ladder grids.

The plotter accepts completed ``grid_summary.json`` files and live
``step_checkpoints/*_post_eval.json`` files. That lets us refresh paper-facing
grid figures while a long DSPy ladder is still running.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT_GLOB = "economic_benoit_*dspy*"
STAGE_ORDER = ("f1g_benoit", "f0", "fg", "fgf", "fgfg", "fgfgf", "fgfgfg")
POWER_STAGE_RE = re.compile(r"^f(\d+)g(\d+)$")
DEFAULT_EXTERNAL_PEARSON_MIN = 0.75
METRIC_FIELDS = (
    "internal_f_pearson",
    "external_expert_pearson",
    "f_star_gap",
    "internal_f_mae_1_7",
    "external_expert_mae_1_7",
    "mean_prediction_1_7",
    "mean_teacher_1_7",
    "mean_expert_1_7",
)
CSV_FIELDS = (
    "family",
    "axis_kind",
    "axis_value",
    "leaf_count",
    "leaf_size_tokens",
    "iteration",
    "stage_name",
    "stage_label",
    "trained",
    "n_eval",
    *METRIC_FIELDS,
    "source_type",
    "source_root",
    "source_path",
    "source_created_at",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _axis_label(row: dict[str, Any]) -> str:
    leaf_size = _safe_int(row.get("leaf_size_tokens"))
    if leaf_size is not None:
        return f"{leaf_size}"
    leaf_count = _safe_int(row.get("leaf_count") or row.get("axis_value"))
    return f"L={leaf_count}" if leaf_count is not None else "unknown"


POWER_STAGE_LABEL_RE = re.compile(r"^f\^?(\d+)\s*g\^?(\d+)$")


def _stage_label(row: dict[str, Any]) -> str:
    return str(row.get("stage_label") or row.get("stage_name") or "")


def _stage_label_math(row: dict[str, Any]) -> str:
    """Render stage labels like ``f^1g^0`` as TeX math mode for plotting."""
    raw = _stage_label(row).strip()
    return _as_math_stage_label(raw)


def _as_math_stage_label(label: str) -> str:
    text = label.strip()
    if not text:
        return text
    if text.startswith("$") and text.endswith("$"):
        return text
    match = POWER_STAGE_LABEL_RE.fullmatch(text)
    if match:
        return f"$f^{{{match.group(1)}}} g^{{{match.group(2)}}}$"
    if "^" in text:
        # Fall through: assume the caller already used power notation we can
        # wrap in a math environment.
        return f"${text}$"
    return text


def _stage_sort_key(stage: Any, iteration: Any = None) -> tuple[int, str]:
    stage_s = str(stage or "")
    if stage_s in STAGE_ORDER:
        return (STAGE_ORDER.index(stage_s), 0, 0, stage_s)
    match = POWER_STAGE_RE.fullmatch(stage_s)
    if match:
        f_degree = int(match.group(1))
        g_degree = int(match.group(2))
        return (100 + f_degree + g_degree, f_degree, g_degree, stage_s)
    iteration_i = _safe_int(iteration)
    if iteration_i is not None:
        return (500 + iteration_i, 0, 0, stage_s)
    return (999, 0, 0, stage_s)


def _metric_count(row: dict[str, Any]) -> int:
    return sum(_safe_float(row.get(field)) is not None for field in METRIC_FIELDS)


def _source_priority(source_type: str) -> int:
    return {
        "grid_summary": 3,
        "iteration_history": 2,
        "checkpoint": 1,
    }.get(source_type, 0)


def _normalize_flat_row(
    row: dict[str, Any],
    *,
    source_type: str,
    source_root: Path,
    source_path: Path,
    source_created_at: Any = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "family": str(row.get("family") or "unknown"),
        "axis_kind": row.get("axis_kind") or (
            "leaf_size_tokens" if row.get("leaf_size_tokens") is not None else "leaf_count"
        ),
        "axis_value": _safe_int(row.get("axis_value")),
        "leaf_count": _safe_int(row.get("leaf_count")),
        "leaf_size_tokens": _safe_int(row.get("leaf_size_tokens")),
        "iteration": _safe_int(row.get("iteration")),
        "stage_name": row.get("stage_name"),
        "stage_label": row.get("stage_label") or row.get("stage_name"),
        "trained": row.get("trained"),
        "n_eval": _safe_int(row.get("n_eval") or row.get("n")),
        "source_type": source_type,
        "source_root": _rel(source_root),
        "source_path": _rel(source_path),
        "source_created_at": str(source_created_at or row.get("created_at") or ""),
    }
    if out["axis_value"] is None:
        out["axis_value"] = out["leaf_size_tokens"] or out["leaf_count"]
    for field in METRIC_FIELDS:
        out[field] = _safe_float(row.get(field))
    return out


def _row_from_split_metrics(
    payload: dict[str, Any],
    *,
    eval_split: str,
    source_type: str,
    source_root: Path,
    source_path: Path,
    source_created_at: Any = None,
) -> dict[str, Any]:
    split_metrics = payload.get("split_metrics") or {}
    metrics = split_metrics.get(eval_split) or split_metrics.get("all") or {}
    row = dict(payload)
    row.update(
        {
            "n_eval": metrics.get("n"),
            **{field: metrics.get(field) for field in METRIC_FIELDS},
        }
    )
    return _normalize_flat_row(
        row,
        source_type=source_type,
        source_root=source_root,
        source_path=source_path,
        source_created_at=source_created_at,
    )


def _rows_from_grid_summary(path: Path, source_root: Path) -> list[dict[str, Any]]:
    try:
        payload = _read_json(path)
    except Exception as exc:
        print(f"warning: failed to read {path}: {exc}", file=sys.stderr)
        return []
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    created_at = payload.get("created_at")
    return [
        _normalize_flat_row(
            dict(row),
            source_type="grid_summary",
            source_root=source_root,
            source_path=path,
            source_created_at=created_at,
        )
        for row in rows
        if isinstance(row, dict)
    ]


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
    if not isinstance(payload, dict):
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
        if not isinstance(iteration, dict):
            continue
        merged = {**base, **iteration}
        rows.append(
            _row_from_split_metrics(
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
    if not isinstance(payload, dict) or payload.get("phase") != "post_eval":
        return []
    return [
        _row_from_split_metrics(
            payload,
            eval_split=eval_split,
            source_type="checkpoint",
            source_root=source_root,
            source_path=path,
            source_created_at=payload.get("created_at"),
        )
    ]


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
        if root.is_file() and root.name == "grid_summary.json":
            rows.extend(_rows_from_grid_summary(root, root.parent.parent))
            continue
        ladder_dir = root / "ladder" if (root / "ladder").exists() else root
        summary = ladder_dir / "grid_summary.json"
        if summary.exists():
            rows.extend(_rows_from_grid_summary(summary, root))
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
        and _safe_float(row.get(metric)) is not None
    ]


def _plot_metric(
    ax: plt.Axes,
    rows: Sequence[dict[str, Any]],
    *,
    metric: str,
    title: str,
    ylabel: str,
    lower_is_better: bool = False,
    external_pearson_max: Optional[float] = None,
) -> None:
    metric_rows = _finite_metric_rows(rows, metric)
    if not metric_rows:
        ax.axis("off")
        return
    metric_values = np.asarray([float(row[metric]) for row in metric_rows], dtype=float)
    stages = sorted(
        {str(row.get("stage_name") or "") for row in metric_rows},
        key=lambda stage: _stage_sort_key(stage),
    )
    colors = plt.get_cmap("tab10")
    for idx, stage in enumerate(stages):
        stage_rows = [
            row
            for row in metric_rows
            if str(row.get("stage_name") or "") == stage
        ]
        stage_rows = sorted(stage_rows, key=lambda row: int(row["leaf_size_tokens"]))
        xs = np.asarray([int(row["leaf_size_tokens"]) for row in stage_rows], dtype=float)
        ys = np.asarray([float(row[metric]) for row in stage_rows], dtype=float)
        color = colors(idx % 10)
        label = _stage_label_math(stage_rows[0]) if stage_rows else _as_math_stage_label(stage)
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=label, color=color)
        live_rows = [row for row in stage_rows if row.get("source_type") == "checkpoint"]
        if live_rows:
            ax.scatter(
                [int(row["leaf_size_tokens"]) for row in live_rows],
                [float(row[metric]) for row in live_rows],
                facecolors="none",
                edgecolors=[color],
                linewidths=1.8,
                s=70,
                zorder=5,
            )
    ax.set_xscale("log", base=2)
    leaf_values = sorted({int(row["leaf_size_tokens"]) for row in metric_rows})
    ax.xaxis.set_major_locator(mticker.FixedLocator(leaf_values))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(value) for value in leaf_values]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_ha("right")
    ax.set_title(title)
    ax.set_xlabel("leaf tokens")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if metric == "external_expert_pearson" and external_pearson_max is not None:
        ymin, ymax = ax.get_ylim()
        # Treat external_pearson_max as the expert-expert ceiling. Extend the
        # visible axis to cover any observed value above the ceiling rather
        # than clipping it, then draw the ceiling as an explicit line.
        observed_max = float(np.nanmax(metric_values))
        new_ymax = max(ymax, observed_max, external_pearson_max)
        ax.set_ylim(ymin, new_ymax)
        ax.axhline(
            external_pearson_max,
            color="#111111",
            linewidth=1.0,
            linestyle="--",
            alpha=0.8,
        )
        ax.text(
            0.985,
            external_pearson_max,
            f" expert r={external_pearson_max:.3f}",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="right",
            fontsize=7.5,
            color="#111111",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
                pad=1.2,
            ),
        )
    elif lower_is_better:
        _, ymax = ax.get_ylim()
        ymax = max(ymax, float(np.nanmax(metric_values)))
        ax.set_ylim(0.0, ymax)
    if metric == "f_star_gap":
        ax.axhline(0.0, color="#333333", linewidth=1.0, alpha=0.7)
    if lower_is_better:
        ax.text(
            0.02,
            0.04,
            "lower is better",
            transform=ax.transAxes,
            fontsize=8,
            color="#555555",
        )


def _write_grid_plot(
    rows: Sequence[dict[str, Any]],
    output: Path,
    *,
    figure_title: str,
    figure_subtitle: str,
    external_pearson_max: Optional[float],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    _plot_metric(
        axes[0, 0],
        rows,
        metric="external_expert_pearson",
        title="External expert Pearson",
        ylabel="Pearson r",
        external_pearson_max=external_pearson_max,
    )
    _plot_metric(
        axes[0, 1],
        rows,
        metric="external_expert_mae_1_7",
        title="External expert MAE",
        ylabel="MAE on 1-7 scale",
        lower_is_better=True,
    )
    _plot_metric(
        axes[1, 0],
        rows,
        metric="internal_f_pearson",
        title="Internal f-vs-teacher Pearson",
        ylabel="Pearson r",
    )
    _plot_metric(
        axes[1, 1],
        rows,
        metric="f_star_gap",
        title="Internal-external gap",
        ylabel="internal r - external r",
    )
    subtitle_lines = str(figure_subtitle).count("\n") + 1 if figure_subtitle else 0
    subtitle_block = 0.018 * max(subtitle_lines, 0)
    legend_y = (0.91 if subtitle_lines <= 1 else 0.88) - (0.012 * max(0, subtitle_lines - 1))
    top_margin = max(0.72, (0.81 if figure_subtitle else 0.84) - subtitle_block)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y if figure_subtitle else 0.93),
            ncol=min(5, len(handles)),
            frameon=False,
        )
    fig.suptitle(str(figure_title), y=0.985, fontsize=14)
    if figure_subtitle:
        fig.text(
            0.5,
            0.955,
            str(figure_subtitle),
            ha="center",
            va="top",
            fontsize=9.5,
            color="#444444",
            linespacing=1.35,
        )
    fig.subplots_adjust(
        top=top_margin,
        hspace=0.48,
        wspace=0.28,
        bottom=0.11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _heatmap_matrix(
    rows: Sequence[dict[str, Any]],
    *,
    metric: str,
) -> tuple[np.ndarray, list[int], list[str]]:
    metric_rows = _finite_metric_rows(rows, metric)
    leaf_values = sorted({int(row["leaf_size_tokens"]) for row in metric_rows})
    stages = sorted(
        {str(row.get("stage_name") or "") for row in metric_rows},
        key=lambda stage: _stage_sort_key(stage),
    )
    matrix = np.full((len(stages), len(leaf_values)), np.nan, dtype=float)
    leaf_index = {value: idx for idx, value in enumerate(leaf_values)}
    stage_index = {stage: idx for idx, stage in enumerate(stages)}
    for row in metric_rows:
        matrix[stage_index[str(row.get("stage_name") or "")], leaf_index[int(row["leaf_size_tokens"])]] = float(row[metric])
    stage_labels = []
    for stage in stages:
        matching = [row for row in metric_rows if str(row.get("stage_name") or "") == stage]
        stage_labels.append(_stage_label_math(matching[0]) if matching else _as_math_stage_label(stage))
    return matrix, leaf_values, stage_labels


def _draw_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    leaf_values: Sequence[int],
    stages: Sequence[str],
    *,
    title: str,
    cmap: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ceiling: Optional[float] = None,
    ceiling_label: str = "",
    floor: Optional[float] = None,
    floor_label: str = "",
    colorbar_label: str = "",
) -> None:
    if matrix.size == 0:
        ax.axis("off")
        return
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(leaf_values)))
    ax.set_xticklabels([str(value) for value in leaf_values], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(stages)))
    ax.set_yticklabels(stages)
    ax.set_xlabel("leaf tokens")
    # Choose cell text color per-cell against the cell background to
    # stay readable at both ends of the colormap.
    cmap_obj = matplotlib.colormaps.get_cmap(cmap) if isinstance(cmap, str) else cmap
    span = (vmax if vmax is not None else 1.0) - (vmin if vmin is not None else 0.0)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix[y, x]
            if not math.isfinite(float(value)):
                continue
            if span > 0:
                norm_value = (float(value) - (vmin or 0.0)) / span
                norm_value = max(0.0, min(1.0, norm_value))
            else:
                norm_value = 0.5
            rgba = cmap_obj(norm_value)
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "white" if luminance < 0.55 else "#111111"
            ax.text(x, y, f"{value:.3f}", ha="center", va="center",
                    fontsize=8, color=text_color)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if colorbar_label:
        cbar.set_label(colorbar_label, fontsize=8)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontsize(8)
    # Draw the expert-possible ceiling and the perfect-projection floor as
    # bold tick marks on the colorbar. Labels for these are carried by the
    # figure caption rather than inline text, which keeps the plot clean at
    # arbitrary rendering sizes.
    if ceiling is not None and vmin is not None and vmax is not None \
            and vmin <= ceiling <= vmax:
        cbar.ax.axhline(ceiling, color="#111111", linewidth=1.4)
        existing_ticks = list(cbar.get_ticks())
        if not any(abs(float(t) - ceiling) < 1e-6 for t in existing_ticks):
            new_ticks = sorted(existing_ticks + [ceiling])
            cbar.set_ticks(new_ticks)
            cbar.ax.get_yticklabels()[new_ticks.index(ceiling)].set_fontweight("bold")
    if floor is not None and vmin is not None and vmax is not None \
            and vmin <= floor <= vmax:
        cbar.ax.axhline(floor, color="#333333", linewidth=1.0, linestyle=":")


def _write_heatmap(
    rows: Sequence[dict[str, Any]],
    output: Path,
    *,
    figure_title: str,
    figure_subtitle: str,
    external_pearson_min: Optional[float],
    external_pearson_max: Optional[float],
    expert_ceiling: Optional[float] = None,
    f_star_gap_max: Optional[float] = None,
) -> None:
    # Keep the plot-body height fixed and let the header grow with the
    # subtitle line count. ``bbox_inches="tight"`` on savefig will capture
    # any title that overflows the figure; we just need the subtitle and
    # plot area not to collide below the suptitle.
    subtitle_lines = str(figure_subtitle).count("\n") + 1 if figure_subtitle else 0
    plot_body_in = 4.6
    suptitle_in = 0.55      # large enough to comfortably seat fontsize 14
    subtitle_line_in = 0.22
    gap_in = 0.28           # gap between last subtitle line and subplot titles
    subtitle_block_in = (
        subtitle_lines * subtitle_line_in + (gap_in if subtitle_lines else 0.0)
    )
    header_in = suptitle_in + subtitle_block_in
    fig_height = plot_body_in + header_in
    fig, axes = plt.subplots(1, 2, figsize=(13, fig_height))

    # External expert Pearson heatmap.
    matrix, leaf_values, stages = _heatmap_matrix(rows, metric="external_expert_pearson")
    ext_rows = _finite_metric_rows(rows, "external_expert_pearson")
    observed_min: Optional[float] = None
    observed_max: Optional[float] = None
    if ext_rows:
        observed_min = min(float(row["external_expert_pearson"]) for row in ext_rows)
        observed_max = max(float(row["external_expert_pearson"]) for row in ext_rows)

    # Lower bound: honor the caller's floor, but never crop out observed data.
    if external_pearson_min is None:
        vmin = observed_min
    elif observed_min is None:
        vmin = external_pearson_min
    else:
        vmin = min(float(external_pearson_min), observed_min)

    # Upper bound: caller's external_pearson_max is now treated as the expert
    # ceiling annotation; the colormap itself must include the observed max so
    # above-ceiling cells do not saturate.
    ceiling = expert_ceiling if expert_ceiling is not None else external_pearson_max
    vmax_candidates = [v for v in (external_pearson_max, observed_max, ceiling) if v is not None]
    vmax = max(vmax_candidates) if vmax_candidates else None
    if vmax is not None and observed_max is not None and vmax <= observed_max:
        vmax = observed_max + 0.005  # tiny margin so the top cell is not at the edge
    ceiling_label = f"expert r={ceiling:.3f}" if ceiling is not None else ""
    _draw_heatmap(
        axes[0],
        matrix,
        leaf_values,
        stages,
        title="External-expert Pearson r",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        ceiling=ceiling,
        ceiling_label=ceiling_label,
        colorbar_label="Pearson r (higher = better)",
    )

    # Internal-external Pearson gap heatmap: sequential, non-negative, lower = better.
    matrix, leaf_values, stages = _heatmap_matrix(rows, metric="f_star_gap")
    gap_rows = _finite_metric_rows(rows, "f_star_gap")
    gap_observed_max: Optional[float] = None
    if gap_rows:
        gap_observed_max = max(float(row["f_star_gap"]) for row in gap_rows)
    gap_vmax_candidates = [v for v in (f_star_gap_max, gap_observed_max, 0.2) if v is not None]
    gap_vmax = max(gap_vmax_candidates) if gap_vmax_candidates else 0.2
    _draw_heatmap(
        axes[1],
        matrix,
        leaf_values,
        stages,
        title="Internal-external Pearson gap",
        cmap="OrRd",
        vmin=0.0,
        vmax=gap_vmax,
        floor=0.0,
        floor_label="parity",
        colorbar_label="Pearson gap (lower = better)",
    )

    # Layout in figure coordinates (0 = bottom, 1 = top), top-down.
    suptitle_y = 1.0 - 0.30 / fig_height
    subtitle_top_y = 1.0 - suptitle_in / fig_height
    plot_top = 1.0 - header_in / fig_height
    fig.suptitle(str(figure_title), fontsize=14, y=suptitle_y)
    if figure_subtitle:
        fig.text(
            0.5,
            subtitle_top_y,
            str(figure_subtitle),
            ha="center",
            va="top",
            fontsize=9.5,
            color="#444444",
            linespacing=1.35,
        )
    fig.tight_layout(rect=(0.0, 0.02, 1.0, plot_top))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def _fmt(value: Any) -> str:
    value_f = _safe_float(value)
    return "n/a" if value_f is None else f"{value_f:.3f}"


def _write_markdown(rows: Sequence[dict[str, Any]], path: Path, *, roots: Sequence[Path]) -> None:
    finite = [row for row in rows if _safe_float(row.get("external_expert_pearson")) is not None]
    best = max(finite, key=lambda row: float(row["external_expert_pearson"])) if finite else None
    live_count = sum(1 for row in rows if row.get("source_type") == "checkpoint")
    lines = [
        "# Manifesto f/g Ladder Grid Plots",
        "",
        f"Generated: `{_utc_now()}`",
        "",
        "## Inputs",
    ]
    for root in roots:
        lines.append(f"- `{_rel(root)}`")
    lines.extend(["", "## Summary", ""])
    if best is not None:
        lines.append(
            "Best external Pearson: "
            f"`{_fmt(best.get('external_expert_pearson'))}` at "
            f"leaf `{_axis_label(best)}`, stage `{_stage_label(best)}`."
        )
    else:
        lines.append("No finite external Pearson rows were found.")
    if live_count:
        lines.append(
            f"{live_count} row(s) came from live checkpoints; open markers in the line plot denote those rows."
        )
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| leaf | k | stage | ext_p | ext_mae | int_p | f_star_gap | source |",
            "|---:|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {leaf} | {iteration} | {stage} | {ext_p} | {ext_mae} | {int_p} | {gap} | {source} |".format(
                leaf=_axis_label(row),
                iteration=row.get("iteration"),
                stage=_stage_label(row),
                ext_p=_fmt(row.get("external_expert_pearson")),
                ext_mae=_fmt(row.get("external_expert_mae_1_7")),
                int_p=_fmt(row.get("internal_f_pearson")),
                gap=_fmt(row.get("f_star_gap")),
                source=row.get("source_type"),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and plot Manifesto f/g alternating ladder grid results."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Run root or ladder dir. May be repeated. Default scans "
            f"outputs/manifesto_fg_alternating/{DEFAULT_ROOT_GLOB}."
        ),
    )
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--family", default="dspy")
    parser.add_argument(
        "--stages",
        default="",
        help=(
            "Optional comma-separated stage filter, e.g. 'fg,fgf'. "
            "Empty means include all stages."
        ),
    )
    parser.add_argument(
        "--stage-label",
        action="append",
        default=[],
        help=(
            "Optional display alias in raw=label form. May be repeated, "
            "e.g. --stage-label fg=initial-fg."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "manifesto_fg_alternating"
        / "benoit_grid_plots",
    )
    parser.add_argument(
        "--figure-title",
        default="Manifesto economic f/g ladder",
        help="Figure suptitle used for both the line grid and heatmap.",
    )
    parser.add_argument(
        "--figure-subtitle",
        default="",
        help="Optional subtitle shown below the main figure title.",
    )
    parser.add_argument(
        "--external-pearson-min",
        type=float,
        default=DEFAULT_EXTERNAL_PEARSON_MIN,
        help="Lower bound for the external Pearson heatmap color scale.",
    )
    parser.add_argument(
        "--external-pearson-max",
        type=float,
        default=None,
        help=(
            "Optional upper bound for external Pearson visuals. Pass the "
            "dimension-specific Table 3 expert-expert reference when you want a "
            "fixed ceiling. The colormap still extends to cover any observed "
            "value above this number; the ceiling is drawn as an annotated "
            "line on the colorbar."
        ),
    )
    parser.add_argument(
        "--expert-ceiling",
        type=float,
        default=None,
        help=(
            "Expert-expert Pearson r upper bound for this dimension (Benoit "
            "2025 Table 3). Drawn as an annotated line on the Pearson "
            "colorbar. Falls back to --external-pearson-max when not "
            "provided."
        ),
    )
    parser.add_argument(
        "--f-star-gap-max",
        type=float,
        default=None,
        help=(
            "Optional upper bound for the internal-external Pearson gap "
            "colormap. Default 0.2 with automatic extension when observed "
            "values exceed it."
        ),
    )
    parser.add_argument("--no-partial", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    roots = _resolve_input_roots(args.input_root)
    rows = _collect_rows(
        roots,
        eval_split=str(args.eval_split),
        include_partial=not bool(args.no_partial),
    )
    rows = [row for row in rows if str(row.get("family") or "") == str(args.family)]
    allowed_stages: set[str] = set()
    if str(args.stages or "").strip():
        allowed_stages = {
            token.strip()
            for token in str(args.stages).replace(";", ",").split(",")
            if token.strip()
        }
        rows = [row for row in rows if str(row.get("stage_name") or "") in allowed_stages]
    stage_labels: dict[str, str] = {}
    for item in args.stage_label or []:
        if "=" not in str(item):
            raise SystemExit(f"--stage-label must be raw=label, got {item!r}")
        raw, label = str(item).split("=", 1)
        raw = raw.strip()
        label = label.strip()
        if raw and label:
            stage_labels[raw] = label
    if stage_labels:
        rows = [
            {
                **row,
                "stage_label": stage_labels.get(str(row.get("stage_name") or ""), row.get("stage_name")),
            }
            for row in rows
        ]
    rows = [row for row in rows if _metric_count(row) > 0]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_dir / "manifesto_fg_ladder_grid_rows.csv")
    _write_grid_plot(
        rows,
        output_dir / "manifesto_fg_ladder_grid.png",
        figure_title=str(args.figure_title),
        figure_subtitle=str(args.figure_subtitle),
        external_pearson_max=_safe_float(args.external_pearson_max),
    )
    _write_heatmap(
        rows,
        output_dir / "manifesto_fg_ladder_heatmap.png",
        figure_title=str(args.figure_title),
        figure_subtitle=str(args.figure_subtitle),
        external_pearson_min=_safe_float(args.external_pearson_min),
        external_pearson_max=_safe_float(args.external_pearson_max),
        expert_ceiling=_safe_float(args.expert_ceiling),
        f_star_gap_max=_safe_float(args.f_star_gap_max),
    )
    _write_markdown(rows, output_dir / "manifesto_fg_ladder_grid.md", roots=roots)
    manifest = {
        "generated_at": _utc_now(),
        "input_roots": [_rel(root) for root in roots],
        "n_rows": len(rows),
        "figure_title": str(args.figure_title),
        "figure_subtitle": str(args.figure_subtitle) or None,
        "stage_filter": sorted(allowed_stages) or None,
        "stage_labels": stage_labels or None,
        "outputs": {
            "csv": _rel(output_dir / "manifesto_fg_ladder_grid_rows.csv"),
            "grid_png": _rel(output_dir / "manifesto_fg_ladder_grid.png"),
            "heatmap_png": _rel(output_dir / "manifesto_fg_ladder_heatmap.png"),
            "markdown": _rel(output_dir / "manifesto_fg_ladder_grid.md"),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
