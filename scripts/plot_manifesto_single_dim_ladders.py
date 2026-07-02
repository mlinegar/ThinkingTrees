#!/usr/bin/env python3
"""Render the paper figure for single-dimension manifesto f/g ladders."""

from __future__ import annotations

import argparse
import csv
import json
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
    / "manifesto_fg_econ_decent_f1g0_f1g1.pdf"
)
BENOIT_INIT_ECON_CSV = (
    REPO_ROOT
    / "outputs"
    / "manifesto_fg_alternating"
    / "benoit_grid_plots_benoit_init"
    / "manifesto_fg_ladder_grid_rows.csv"
)
DECENT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "manifesto_fg_alternating"
    / "decentralization_benoit_g0init_fresh_dspy_20260426_1815"
    / "ladder"
    / "dspy"
)


def _safe_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_csv_series(path: Path, stage_names: Iterable[str]) -> dict[int, float]:
    stages = set(stage_names)
    rows: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("stage_name") not in stages:
                continue
            leaf = row.get("leaf_size_tokens") or row.get("axis_value")
            value = _safe_float(row.get("external_expert_pearson"))
            if leaf and value is not None:
                rows[int(leaf)] = value
    return dict(sorted(rows.items()))


def _read_csv_best_series(path: Path) -> dict[int, float]:
    rows: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            leaf = row.get("leaf_size_tokens") or row.get("axis_value")
            value = _safe_float(row.get("external_expert_pearson"))
            if leaf and value is not None:
                leaf_int = int(leaf)
                rows[leaf_int] = max(rows.get(leaf_int, float("-inf")), value)
    return dict(sorted(rows.items()))


def _metric_from_iteration(row: dict[str, object]) -> float | None:
    split_metrics = row.get("split_metrics")
    if not isinstance(split_metrics, dict):
        return None
    test_metrics = split_metrics.get("test")
    if not isinstance(test_metrics, dict):
        return None
    return _safe_float(test_metrics.get("external_expert_pearson"))


def _read_decentralization_series(root: Path, stage_name: str) -> dict[int, float]:
    rows: dict[int, float] = {}
    for leaf_dir in sorted(root.glob("leaf*tok")):
        history_path = leaf_dir / "iteration_history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            for row in history.get("iterations", []):
                if not isinstance(row, dict) or row.get("stage_name") != stage_name:
                    continue
                leaf = row.get("leaf_size_tokens") or row.get("axis_value")
                value = _metric_from_iteration(row)
                if leaf is not None and value is not None:
                    rows[int(leaf)] = value

        checkpoints = leaf_dir / "step_checkpoints"
        for checkpoint in sorted(checkpoints.glob("iter_*_post_eval.json")):
            row = json.loads(checkpoint.read_text(encoding="utf-8"))
            if row.get("stage_name") != stage_name:
                continue
            leaf = row.get("leaf_size_tokens") or row.get("axis_value")
            value = _metric_from_iteration(row)
            if leaf is not None and value is not None:
                rows[int(leaf)] = value
    return dict(sorted(rows.items()))


def _read_decentralization_best_series(root: Path) -> dict[int, float]:
    rows: dict[int, float] = {}
    for leaf_dir in sorted(root.glob("leaf*tok")):
        candidates: list[dict[str, object]] = []
        history_path = leaf_dir / "iteration_history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            candidates.extend(row for row in history.get("iterations", []) if isinstance(row, dict))

        checkpoints = leaf_dir / "step_checkpoints"
        for checkpoint in sorted(checkpoints.glob("iter_*_post_eval.json")):
            row = json.loads(checkpoint.read_text(encoding="utf-8"))
            if isinstance(row, dict):
                candidates.append(row)

        for row in candidates:
            leaf = row.get("leaf_size_tokens") or row.get("axis_value")
            value = _metric_from_iteration(row)
            if leaf is not None and value is not None:
                leaf_int = int(leaf)
                rows[leaf_int] = max(rows.get(leaf_int, float("-inf")), value)
    return dict(sorted(rows.items()))


def _plot_series(ax: plt.Axes, series: dict[int, float], **kwargs: object) -> None:
    if not series:
        return
    ax.plot(list(series.keys()), list(series.values()), **kwargs)


def _add_reference_lines(
    ax: plt.Axes, *, proprietary: float, best_open: float, split_expert: float
) -> None:
    ax.axhline(
        proprietary,
        color="#3f3f46",
        linestyle=(0, (5, 3)),
        linewidth=1.2,
        label="Benoit 2025 proprietary ensemble",
    )
    ax.axhline(
        best_open,
        color="#71717a",
        linestyle=(0, (1.2, 2.0)),
        linewidth=1.3,
        label="Benoit 2025 best open-weight",
    )
    ax.axhline(
        split_expert,
        color="#a16207",
        linestyle=(0, (7, 2, 1.4, 2)),
        linewidth=1.25,
        label="Benoit split-expert reference",
    )


def render(output: Path) -> None:
    econ_benoit_f1g0 = _read_csv_series(BENOIT_INIT_ECON_CSV, {"fg"})
    econ_best = _read_csv_best_series(BENOIT_INIT_ECON_CSV)
    decent_f1g0 = _read_decentralization_series(DECENT_ROOT, "f1g0")
    decent_f1g1 = _read_decentralization_series(DECENT_ROOT, "f1g1")
    decent_best = _read_decentralization_best_series(DECENT_ROOT)

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
            "legend.fontsize": 8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9), sharex=True)
    colors = {
        "f1g0": "#2563eb",
        "f1g1": "#dc2626",
        "best": "#16a34a",
    }
    markers = {"f1g0": "o", "f1g1": "s", "best": "D"}

    econ_ax, decent_ax = axes
    _plot_series(
        econ_ax,
        econ_benoit_f1g0,
        color=colors["f1g0"],
        marker=markers["f1g0"],
        linewidth=1.8,
        markersize=4.6,
        label=r"$f^1g^0$",
    )
    _plot_series(
        econ_ax,
        econ_best,
        color=colors["best"],
        marker=markers["best"],
        linewidth=1.85,
        markersize=4.6,
        label=r"best $f^xg^y$ at leaf",
    )
    _add_reference_lines(econ_ax, proprietary=0.870, best_open=0.860, split_expert=0.880)
    econ_ax.set_title("Economic policy")
    econ_ax.set_ylabel("held-out Pearson r")
    econ_ax.set_ylim(0.80, 0.895)

    _plot_series(
        decent_ax,
        decent_f1g0,
        color=colors["f1g0"],
        marker=markers["f1g0"],
        linewidth=1.8,
        markersize=4.6,
        label=r"$f^1g^0$",
    )
    _plot_series(
        decent_ax,
        decent_f1g1,
        color=colors["f1g1"],
        marker=markers["f1g1"],
        linewidth=1.8,
        markersize=4.6,
        label=r"$f^1g^1$",
    )
    _plot_series(
        decent_ax,
        decent_best,
        color=colors["best"],
        marker=markers["best"],
        linewidth=1.85,
        markersize=4.6,
        label=r"best $f^xg^y$ at leaf",
    )
    _add_reference_lines(decent_ax, proprietary=0.490, best_open=0.450, split_expert=0.780)
    decent_ax.set_title("Decentralization")
    decent_ax.set_ylim(0.40, 0.805)

    all_leaves = [256, 512, 1024, 2048, 4096, 8192]
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(all_leaves)
        ax.set_xticklabels(["256", "512", "1K", "2K", "4K", "8K"])
        ax.set_xlabel("leaf tokens")
        ax.grid(axis="y", color="#e4e4e7", linewidth=0.7)

    handles: list[object] = []
    labels: list[str] = []
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    legend_by_label = dict(zip(labels, handles))
    ordered_labels = [
        r"$f^1g^0$",
        r"$f^1g^1$",
        r"best $f^xg^y$ at leaf",
        "Benoit 2025 proprietary ensemble",
        "Benoit 2025 best open-weight",
        "Benoit split-expert reference",
    ]
    handles = [legend_by_label[label] for label in ordered_labels if label in legend_by_label]
    labels = [label for label in ordered_labels if label in legend_by_label]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.5,
    )
    fig.tight_layout(rect=(0, 0.22, 1, 1))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.08)
    png_output = output.with_suffix(".png")
    fig.savefig(png_output, dpi=200, bbox_inches="tight", pad_inches=0.08)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
