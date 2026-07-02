#!/usr/bin/env python3
"""Render the paper figure for the joint all-six manifesto f/g ladder."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "paper"
    / "ctreepo"
    / "assets"
    / "benoit"
    / "figures"
    / "manifesto_fg_combined_ladder_f1g0_f1g1.pdf"
)
DIMENSION_ROWS = (
    REPO_ROOT
    / "outputs"
    / "manifesto_fg_alternating"
    / "combined_benoit_joint_teacher_all6_dspy_fixed_20260425_000122"
    / "plots_by_dimension"
    / "manifesto_fg_ladder_dimension_rows.csv"
)


DIMENSIONS = (
    ("economic", "Economic policy"),
    ("social", "Social policy"),
    ("immigration", "Immigration"),
    ("eu", "European integration"),
    ("environment", "Environment"),
    ("decentralization", "Decentralization"),
)
BENOIT_PROPRIETARY = {
    "economic": 0.870,
    "social": 0.920,
    "immigration": 0.890,
    "eu": 0.910,
    "environment": 0.820,
    "decentralization": 0.490,
}
BENOIT_BEST_OPEN = {
    "economic": 0.860,
    "social": 0.870,
    "immigration": 0.890,
    "eu": 0.860,
    "environment": 0.860,
    "decentralization": 0.450,
}
BENOIT_SPLIT_EXPERT = {
    "economic": 0.880,
    "social": 0.910,
    "immigration": 0.880,
    "eu": 0.950,
    "environment": 0.840,
    "decentralization": 0.780,
}
PEARSON_YLIM = (0.25, 1.0)


def _safe_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_dimension_series(
    path: Path, *, dimension: str, stage_name: str
) -> dict[int, float]:
    rows: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("dimension") != dimension or row.get("stage_name") != stage_name:
                continue
            value = _safe_float(row.get("external_expert_pearson"))
            leaf = row.get("leaf_size_tokens") or row.get("axis_value")
            if leaf and value is not None:
                rows[int(leaf)] = value
    return dict(sorted(rows.items()))


def _read_dimension_best_series(path: Path, *, dimension: str) -> dict[int, float]:
    rows: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("dimension") != dimension:
                continue
            value = _safe_float(row.get("external_expert_pearson"))
            leaf = row.get("leaf_size_tokens") or row.get("axis_value")
            if leaf and value is not None:
                leaf_int = int(leaf)
                rows[leaf_int] = max(rows.get(leaf_int, float("-inf")), value)
    return dict(sorted(rows.items()))


def _read_dimension_gap_series(
    path: Path, *, dimension: str, stage_name: str
) -> dict[int, float]:
    """Audit gap (f_star_gap) by leaf, at a specific stage, for a dimension."""
    rows: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("dimension") != dimension or row.get("stage_name") != stage_name:
                continue
            gap = _safe_float(row.get("f_star_gap"))
            leaf = row.get("leaf_size_tokens") or row.get("axis_value")
            if leaf and gap is not None:
                rows[int(leaf)] = gap
    return dict(sorted(rows.items()))


def _read_dimension_gap_at_best_series(
    path: Path, *, dimension: str
) -> dict[int, float]:
    """Audit gap at the cell where external Pearson is best, by leaf."""
    by_leaf_best: dict[int, tuple[float, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("dimension") != dimension:
                continue
            ep = _safe_float(row.get("external_expert_pearson"))
            gap = _safe_float(row.get("f_star_gap"))
            leaf = row.get("leaf_size_tokens") or row.get("axis_value")
            if leaf is None or ep is None or gap is None:
                continue
            leaf_int = int(leaf)
            if leaf_int not in by_leaf_best or ep > by_leaf_best[leaf_int][0]:
                by_leaf_best[leaf_int] = (ep, gap)
    return {leaf: gap for leaf, (_, gap) in sorted(by_leaf_best.items())}


def _macro_series(path: Path, stage_name: str) -> dict[int, float]:
    by_leaf: dict[int, list[float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("stage_name") != stage_name:
                continue
            value = _safe_float(row.get("external_expert_pearson"))
            leaf = row.get("leaf_size_tokens") or row.get("axis_value")
            if leaf and value is not None:
                by_leaf.setdefault(int(leaf), []).append(value)
    return {
        leaf: sum(values) / len(values)
        for leaf, values in sorted(by_leaf.items())
        if len(values) == len(DIMENSIONS)
    }


def _macro_best_series(path: Path) -> dict[int, float]:
    by_dimension_leaf: dict[tuple[str, int], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            dimension = row.get("dimension")
            value = _safe_float(row.get("external_expert_pearson"))
            leaf = row.get("leaf_size_tokens") or row.get("axis_value")
            if not dimension or not leaf or value is None:
                continue
            key = (dimension, int(leaf))
            by_dimension_leaf[key] = max(by_dimension_leaf.get(key, float("-inf")), value)

    by_leaf: dict[int, list[float]] = {}
    for (dimension, leaf), value in by_dimension_leaf.items():
        if dimension in {name for name, _ in DIMENSIONS}:
            by_leaf.setdefault(leaf, []).append(value)
    return {
        leaf: sum(values) / len(values)
        for leaf, values in sorted(by_leaf.items())
        if len(values) == len(DIMENSIONS)
    }


def _plot_series(ax: plt.Axes, series: dict[int, float], **kwargs: object) -> None:
    if series:
        ax.plot(list(series.keys()), list(series.values()), **kwargs)


def _read_leaf_ticks(path: Path) -> list[int]:
    leaves: set[int] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            leaf = row.get("leaf_size_tokens") or row.get("axis_value")
            if leaf:
                leaves.add(int(leaf))
    return sorted(leaves) or [256, 512, 1024, 2048, 4096, 8192]


def _leaf_label(leaf: int) -> str:
    if leaf < 1000:
        return str(leaf)
    if leaf % 1024 == 0:
        return f"{leaf // 1024}K"
    return f"{round(leaf / 1024):g}K"


def _add_refs(ax: plt.Axes, dimension: str) -> None:
    ax.axhline(
        BENOIT_PROPRIETARY[dimension],
        color="#3f3f46",
        linestyle=(0, (5, 3)),
        linewidth=1.05,
        label="Benoit 2025 proprietary ensemble",
    )
    ax.axhline(
        BENOIT_BEST_OPEN[dimension],
        color="#71717a",
        linestyle=(0, (1.2, 2.0)),
        linewidth=1.15,
        label="Benoit 2025 best open-weight",
    )
    ax.axhline(
        BENOIT_SPLIT_EXPERT[dimension],
        color="#a16207",
        linestyle=(0, (7, 2, 1.4, 2)),
        linewidth=1.1,
        label="Benoit split-expert reference",
    )


def _format_axis(ax: plt.Axes, *, leaves: list[int], show_ylabel: bool) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(leaves)
    ax.set_xticklabels([_leaf_label(leaf) for leaf in leaves])
    ax.set_xlabel("leaf tokens")
    if show_ylabel:
        ax.set_ylabel("held-out Pearson r")
    ax.grid(axis="y", color="#e4e4e7", linewidth=0.7)


def render(output: Path) -> None:
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 8.5,
            "legend.fontsize": 8,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.75))
    colors = {"round1": "#dc2626", "best": "#16a34a"}
    leaf_ticks = _read_leaf_ticks(DIMENSION_ROWS)

    for index, (dimension, title) in enumerate(DIMENSIONS):
        ax = axes.flat[index]
        _plot_series(
            ax,
            _read_dimension_series(DIMENSION_ROWS, dimension=dimension, stage_name="f1g1"),
            color=colors["round1"],
            marker="s",
            linewidth=1.55,
            markersize=3.8,
            label="after round 1",
        )
        _plot_series(
            ax,
            _read_dimension_best_series(DIMENSION_ROWS, dimension=dimension),
            color=colors["best"],
            marker="D",
            linewidth=1.6,
            markersize=3.7,
            label="best round",
        )
        _add_refs(ax, dimension)
        ax.set_title(title, pad=4)
        ax.set_ylim(*PEARSON_YLIM)
        _format_axis(ax, leaves=leaf_ticks, show_ylabel=index % 3 == 0)

    handles: list[object] = []
    labels: list[str] = []
    for ax in axes.flat:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.002),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.3,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=1.15, h_pad=1.0)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.08)

    round1_macro = _macro_series(DIMENSION_ROWS, "f1g1")
    best_macro = _macro_best_series(DIMENSION_ROWS)
    print("round 1 macro:", ", ".join(f"{k}={v:.3f}" for k, v in round1_macro.items()))
    print(
        "best round macro:",
        ", ".join(f"{k}={v:.3f}" for k, v in best_macro.items()),
    )


GAP_YLIMS = {
    "economic": (-0.02, 0.20),
    "social": (-0.02, 0.20),
    "immigration": (-0.02, 0.20),
    "eu": (-0.02, 0.20),
    "environment": (-0.02, 0.20),
    "decentralization": (0.0, 0.55),
}


DEFAULT_GAP_OUTPUT = (
    REPO_ROOT
    / "paper"
    / "ctreepo"
    / "assets"
    / "benoit"
    / "figures"
    / "manifesto_fg_combined_audit_gap.pdf"
)


def render_audit_gap(output: Path) -> None:
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 8.5,
            "legend.fontsize": 8,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.75))
    colors = {"round1": "#dc2626", "best": "#16a34a"}
    leaf_ticks = _read_leaf_ticks(DIMENSION_ROWS)

    for index, (dimension, title) in enumerate(DIMENSIONS):
        ax = axes.flat[index]
        _plot_series(
            ax,
            _read_dimension_gap_series(DIMENSION_ROWS, dimension=dimension, stage_name="f1g1"),
            color=colors["round1"],
            marker="s",
            linewidth=1.55,
            markersize=3.8,
            label="after round 1",
        )
        _plot_series(
            ax,
            _read_dimension_gap_at_best_series(DIMENSION_ROWS, dimension=dimension),
            color=colors["best"],
            marker="D",
            linewidth=1.6,
            markersize=3.7,
            label="at best round",
        )
        ax.axhline(
            0.09,
            color="#71717a",
            linestyle=(0, (1.2, 2.0)),
            linewidth=1.05,
            label="upper end of well-behaved range",
        )
        ax.axhline(0.0, color="#3f3f46", linestyle="-", linewidth=0.6, alpha=0.5)
        ax.set_title(title, pad=4)
        ax.set_ylim(*GAP_YLIMS[dimension])
        ax.set_xscale("log", base=2)
        ax.set_xticks(leaf_ticks)
        ax.set_xticklabels([_leaf_label(leaf) for leaf in leaf_ticks])
        ax.set_xlabel("leaf tokens")
        if index % 3 == 0:
            ax.set_ylabel("audit gap (internal $-$ external)")
        ax.grid(axis="y", color="#e4e4e7", linewidth=0.7)

    handles: list[object] = []
    labels: list[str] = []
    for ax in axes.flat:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.002),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.3,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=1.15, h_pad=1.0)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.08)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gap-output", type=Path, default=DEFAULT_GAP_OUTPUT)
    args = parser.parse_args()
    render(args.output)
    render_audit_gap(args.gap_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
