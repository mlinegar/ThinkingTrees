#!/usr/bin/env python3
"""Plot per-round audit-gap and Pearson trajectories for manifesto runs.

A diagnostic, not a paper figure. Outputs per-run multi-panel PDFs to
`outputs/manifesto_local_law_diagnostic/`.

For each run (single-dim economic, single-dim decentralization, joint
multi-dim) and each completed leaf size, this loads
`iteration_history.json` and plots:

  - External Pearson over rounds
  - Internal Pearson over rounds
  - Audit gap (internal - external) over rounds

For single-dimension runs the layout is one row of three panels (ext,
int, gap), with each completed leaf size as a separate colored line.
For the joint multi-dimension run we render six rows (one per
dimension), same three columns.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/mlinegar/ThinkingTrees")
OUT = REPO / "outputs" / "manifesto_local_law_diagnostic"
OUT.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Run inventory
# ----------------------------------------------------------------------

ECON_RUNS = [
    (
        REPO
        / "outputs/manifesto_fg_alternating/economic_benoit_g0init_f3g3_dspy_20260423_172036",
        [256, 512, 1024],
    ),
    (
        REPO
        / "outputs/manifesto_fg_alternating/economic_benoit_g0init_largeleaves_retry_20260424_085154",
        [2048, 4096, 8096],
    ),
]

DEC_RUNS = [
    (
        REPO
        / "outputs/manifesto_fg_alternating/decentralization_benoit_g0init_fresh_dspy_20260426_1815",
        [256, 512, 1024],
    ),
]

JOINT_RUN = (
    REPO
    / "outputs/manifesto_fg_alternating/combined_benoit_joint_teacher_all6_dspy_fixed_20260425_000122",
    [256, 512, 1024, 2048, 4096, 8096],
)


JOINT_DIMS = (
    "economic",
    "social",
    "immigration",
    "eu",
    "environment",
    "decentralization",
)


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------


def _load_history(run_dir: Path, leaf: int) -> dict | None:
    path = run_dir / "ladder" / "dspy" / f"leaf{leaf:04d}tok" / "iteration_history.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _per_round_metrics(
    history: dict, dimension: str | None = None
) -> list[tuple[int, float | None, float | None, float | None]]:
    """Returns [(round_idx, ext_p, int_p, gap), ...] for the given dim slice.

    `round_idx` is the iteration number; round 0 corresponds to the
    first cell (typically f^1 g^0), round 1 to f^1 g^1, etc.

    For single-dim runs `dimension` is None and the top-level
    split_metrics.test fields are used. For joint runs, slice into
    split_metrics.test.per_dimension[dimension].
    """
    rows = []
    for it in history.get("iterations", []):
        sm = it.get("split_metrics", {}).get("test", {}) or {}
        if dimension is not None:
            sm = sm.get("per_dimension", {}).get(dimension, {}) or {}
        ext = sm.get("external_expert_pearson")
        int_ = sm.get("internal_f_pearson")
        gap = sm.get("f_star_gap")
        rows.append((it.get("iteration"), _float(ext), _float(int_), _float(gap)))
    return rows


def _float(v) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

LEAF_COLORS = {
    256: "#2563eb",
    512: "#0891b2",
    1024: "#16a34a",
    2048: "#ca8a04",
    4096: "#ea580c",
    8096: "#dc2626",
}


def _plot_metric(
    ax: plt.Axes,
    *,
    leaf_to_rows: dict[int, list[tuple[int, float | None, float | None, float | None]]],
    metric_idx: int,
    title: str,
    ylim: tuple[float, float] | None = None,
):
    for leaf, rows in sorted(leaf_to_rows.items()):
        xs = [r[0] for r in rows if r[metric_idx] is not None]
        ys = [r[metric_idx] for r in rows if r[metric_idx] is not None]
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            color=LEAF_COLORS.get(leaf, "#444"),
            marker="o",
            markersize=3.5,
            linewidth=1.2,
            label=f"leaf {leaf}",
        )
    ax.set_xlabel("iteration")
    ax.set_title(title)
    ax.grid(axis="y", color="#e4e4e7", linewidth=0.6)
    if ylim:
        ax.set_ylim(*ylim)


PANEL_SPECS = [
    (1, "external Pearson (vs expert means)", "Pearson $r$"),
    (2, "internal Pearson (vs teacher trace)", "Pearson $r$"),
    (3, "audit gap (internal $-$ external)", "Pearson $r$ difference"),
]


def render_single_dim(
    run_specs: Iterable[tuple[Path, Iterable[int]]],
    *,
    title: str,
    out_path: Path,
):
    leaf_to_rows: dict[int, list] = {}
    for run_dir, leaves in run_specs:
        for leaf in leaves:
            hist = _load_history(run_dir, leaf)
            if hist is None:
                continue
            leaf_to_rows[leaf] = _per_round_metrics(hist, dimension=None)
    if not leaf_to_rows:
        print(f"  no data for {title}")
        return

    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (metric_idx, panel_title, ylabel) in zip(axes, PANEL_SPECS):
        _plot_metric(ax, leaf_to_rows=leaf_to_rows, metric_idx=metric_idx,
                     title=panel_title)
        ax.set_ylabel(ylabel)
    axes[0].legend(loc="best", fontsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def render_joint(out_path: Path):
    run_dir, leaves = JOINT_RUN
    plt.rcParams.update({"font.size": 8.5})
    fig, axes = plt.subplots(len(JOINT_DIMS), 3, figsize=(11, 2.5 * len(JOINT_DIMS)))

    for row_idx, dim in enumerate(JOINT_DIMS):
        leaf_to_rows: dict[int, list] = {}
        for leaf in leaves:
            hist = _load_history(run_dir, leaf)
            if hist is None:
                continue
            leaf_to_rows[leaf] = _per_round_metrics(hist, dimension=dim)
        for col_idx, (metric_idx, panel_title, ylabel) in enumerate(PANEL_SPECS):
            ax = axes[row_idx, col_idx]
            _plot_metric(ax, leaf_to_rows=leaf_to_rows, metric_idx=metric_idx,
                         title=f"{dim} — {panel_title}")
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc="best", fontsize=6, ncol=2)
    fig.suptitle("Joint multi-dim run, per-dimension trajectories", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    print("Single-dim economic")
    render_single_dim(
        ECON_RUNS,
        title="Single-dimension economic run",
        out_path=OUT / "singledim_economic_diagnostic.pdf",
    )
    print("Single-dim decentralization")
    render_single_dim(
        DEC_RUNS,
        title="Single-dimension decentralization run",
        out_path=OUT / "singledim_decentralization_diagnostic.pdf",
    )
    print("Joint multi-dim")
    render_joint(OUT / "joint_multidim_diagnostic.pdf")


if __name__ == "__main__":
    main()
