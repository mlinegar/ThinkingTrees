#!/usr/bin:wenv python3
"""Live single-dimension per-dim figure.

Mirrors the layout of the joint-run 6-panel figure (one panel per
dimension; x = leaf size; y = external Pearson; series for after
round 1 and best round) but pulls data from the in-progress
single-dimension runs at
`outputs/manifesto_fg_alternating/scalar_dims_benoit_all6_fresh8192_dspy_20260427_015845`,
using the environment reallocated-eval run only for leaves it actually
contains and falling back to older single-dimension sweeps only if the
new scalar run is missing a dimension entirely.

Renders both the held-out external Pearson view and an audit-gap
twin.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path("/home/mlinegar/ThinkingTrees")

OUT_PEARSON = (
    REPO
    / "paper/ctreepo/assets/benoit/figures/manifesto_singledim_per_dim_live.pdf"
)
OUT_AUDIT = (
    REPO
    / "paper/ctreepo/assets/benoit/figures/manifesto_singledim_per_dim_live_audit_gap.pdf"
)

NEW_RUN = (
    REPO
    / "outputs/manifesto_fg_alternating/scalar_dims_benoit_all6_fresh8192_dspy_20260427_015845"
)

ENV_REALLOC_EVAL_RUN = (
    REPO
    / "outputs/manifesto_fg_alternating/environment_benoit_eval_realloc_test48_existing_dspy_20260427_173044"
)

OLD_ECON_SMALL = REPO / "outputs/manifesto_fg_alternating/economic_benoit_g0init_f3g3_dspy_20260423_172036"
OLD_ECON_LARGE = REPO / "outputs/manifesto_fg_alternating/economic_benoit_g0init_largeleaves_retry_20260424_085154"
OLD_DEC = REPO / "outputs/manifesto_fg_alternating/decentralization_benoit_g0init_fresh_dspy_20260426_1815"


DIMENSIONS = (
    ("economic", "Economic policy"),
    ("social", "Social policy"),
    ("immigration", "Immigration"),
    ("eu", "European integration"),
    ("environment", "Environment"),
    ("decentralization", "Decentralization"),
)


BENOIT_PROPRIETARY = {
    "economic": 0.870, "social": 0.920, "immigration": 0.890,
    "eu": 0.910, "environment": 0.820, "decentralization": 0.490,
}
BENOIT_BEST_OPEN = {
    "economic": 0.860, "social": 0.870, "immigration": 0.890,
    "eu": 0.860, "environment": 0.860, "decentralization": 0.450,
}
BENOIT_SPLIT_EXPERT = {
    "economic": 0.880, "social": 0.910, "immigration": 0.880,
    "eu": 0.950, "environment": 0.840, "decentralization": 0.780,
}
PEARSON_YLIM = (0.25, 1.0)
GAP_YLIMS = {
    "economic": (-0.02, 0.20), "social": (-0.02, 0.20),
    "immigration": (-0.02, 0.20), "eu": (-0.02, 0.20),
    "environment": (-0.02, 0.40), "decentralization": (0.0, 0.55),
}

ALL_LEAVES = [256, 512, 1024, 2048, 4096, 8192]


def _safe_float(v):
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _load_iter_history(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _per_leaf_metrics(
    run_dir: Path, *, dim_subdir: str | None, leaves: Iterable[int]
) -> dict[int, list[tuple[str, float | None, float | None]]]:
    """Returns {leaf: [(stage_name, ext_p, gap), ...]}.

    `dim_subdir` is the per-dim subdir name in the new scalar_dims runs
    (e.g., "eu"); when None the run dir itself is treated as the ladder
    parent (the older single-dim runs are organized this way).
    """
    out: dict[int, list[tuple[str, float | None, float | None]]] = {}
    base = run_dir / dim_subdir if dim_subdir else run_dir
    for leaf in leaves:
        leaf_dir = base / "ladder" / "dspy" / f"leaf{leaf:04d}tok"
        ih = leaf_dir / "iteration_history.json"
        data = _load_iter_history(ih)
        rows_by_stage = {}
        if data is not None:
            for it in data.get("iterations", []):
                sm = it["split_metrics"].get("test", {}) or {}
                ep = _safe_float(sm.get("external_expert_pearson"))
                gap = _safe_float(sm.get("f_star_gap"))
                rows_by_stage[it["stage_name"]] = (it["stage_name"], ep, gap)
        for checkpoint in sorted((leaf_dir / "step_checkpoints").glob("iter_*_post_eval.json")):
            checkpoint_data = _load_iter_history(checkpoint)
            if checkpoint_data is None:
                continue
            sm = checkpoint_data.get("split_metrics", {}).get("test", {}) or {}
            ep = _safe_float(sm.get("external_expert_pearson"))
            gap = _safe_float(sm.get("f_star_gap"))
            stage_name = checkpoint_data.get("stage_name")
            if stage_name:
                rows_by_stage[str(stage_name)] = (str(stage_name), ep, gap)
        rows = list(rows_by_stage.values())
        if rows:
            out[leaf] = rows
    return out


def _series_round1(by_leaf: dict, *, metric_idx: int) -> dict[int, float]:
    return {
        L: row[metric_idx]
        for L, rows in sorted(by_leaf.items())
        for row in rows
        if row[0] == "f1g1" and row[metric_idx] is not None
    }


def _series_best_ext(by_leaf: dict) -> dict[int, float]:
    out = {}
    for L, rows in sorted(by_leaf.items()):
        valid = [r for r in rows if r[1] is not None]
        if valid:
            out[L] = max(r[1] for r in valid)
    return out


def _series_gap_at_best(by_leaf: dict) -> dict[int, float]:
    out = {}
    for L, rows in sorted(by_leaf.items()):
        valid = [r for r in rows if r[1] is not None and r[2] is not None]
        if valid:
            out[L] = max(valid, key=lambda r: r[1])[2]
    return out


def _load_dimension(dim: str) -> dict[int, list]:
    """Pull data for a single dimension.

    Primary source: the new scalar_dims run. For environment, the
    reallocated-eval run is preferred for leaves it contains, but it is
    not allowed to mask newer 4096/8192 scalar-run completions.
    """
    new = _per_leaf_metrics(NEW_RUN, dim_subdir=dim, leaves=ALL_LEAVES)
    if dim == "environment":
        reallocated = _per_leaf_metrics(ENV_REALLOC_EVAL_RUN, dim_subdir=None, leaves=ALL_LEAVES)
        if reallocated:
            return {**new, **reallocated}
    if new:
        return new
    if dim == "economic":
        small = _per_leaf_metrics(OLD_ECON_SMALL, dim_subdir=None, leaves=[256, 512, 1024])
        large = _per_leaf_metrics(OLD_ECON_LARGE, dim_subdir=None, leaves=[2048, 4096, 8096])
        return {**small, **large}
    if dim == "decentralization":
        return _per_leaf_metrics(OLD_DEC, dim_subdir=None, leaves=[256, 512, 1024])
    return {}


COLORS = {"round1": "#dc2626", "best": "#16a34a"}


def _plot_panel(ax, *, by_leaf, gap_mode: bool, dim: str):
    if gap_mode:
        round1 = _series_round1(by_leaf, metric_idx=2)
        best = _series_gap_at_best(by_leaf)
    else:
        round1 = _series_round1(by_leaf, metric_idx=1)
        best = _series_best_ext(by_leaf)

    if round1:
        ax.plot(list(round1.keys()), list(round1.values()),
                color=COLORS["round1"], marker="s", linewidth=1.55,
                markersize=3.8, label="after round 1")
    if best:
        label = "at best round" if gap_mode else "best round"
        ax.plot(list(best.keys()), list(best.values()),
                color=COLORS["best"], marker="D", linewidth=1.6,
                markersize=3.7, label=label)

    if gap_mode:
        ax.axhline(0.09, color="#71717a", linestyle=(0, (1.2, 2.0)),
                   linewidth=1.05, label="upper end of well-behaved range")
        ax.axhline(0.0, color="#3f3f46", linestyle="-", linewidth=0.6, alpha=0.5)
    else:
        ax.axhline(BENOIT_PROPRIETARY[dim], color="#3f3f46",
                   linestyle=(0, (5, 3)), linewidth=1.05,
                   label="Benoit 2025 proprietary ensemble")
        ax.axhline(BENOIT_BEST_OPEN[dim], color="#71717a",
                   linestyle=(0, (1.2, 2.0)), linewidth=1.15,
                   label="Benoit 2025 best open-weight")
        ax.axhline(BENOIT_SPLIT_EXPERT[dim], color="#a16207",
                   linestyle=(0, (7, 2, 1.4, 2)), linewidth=1.1,
                   label="Benoit split-expert reference")


def _format_axis(ax, *, gap_mode: bool, dim: str, show_ylabel: bool):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ALL_LEAVES)
    ax.set_xticklabels(["256", "512", "1K", "2K", "4K", "8K"])
    ax.set_xlabel("leaf tokens")
    if show_ylabel:
        ax.set_ylabel(
            "audit gap (internal $-$ external)" if gap_mode else "held-out Pearson r"
        )
    ax.grid(axis="y", color="#e4e4e7", linewidth=0.7)
    if gap_mode:
        ax.set_ylim(*GAP_YLIMS[dim])
    else:
        ax.set_ylim(*PEARSON_YLIM)


def render(out_path: Path, *, gap_mode: bool):
    plt.rcParams.update(
        {"axes.spines.top": False, "axes.spines.right": False,
         "font.size": 8.5, "legend.fontsize": 8}
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.75))

    for index, (dim, title) in enumerate(DIMENSIONS):
        ax = axes.flat[index]
        by_leaf = _load_dimension(dim)
        if by_leaf:
            _plot_panel(ax, by_leaf=by_leaf, gap_mode=gap_mode, dim=dim)
        else:
            # Empty panel: still draw axes + title.
            ax.text(0.5, 0.5, "data pending", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="#71717a")
        ax.set_title(title, pad=4)
        _format_axis(ax, gap_mode=gap_mode, dim=dim, show_ylabel=index % 3 == 0)

    handles, labels = [], []
    for ax in axes.flat:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for h, l in zip(ax_handles, ax_labels):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.002),
               ncol=3, frameon=False, columnspacing=1.2, handlelength=2.3)
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=1.15, h_pad=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pearson-output", type=Path, default=OUT_PEARSON)
    parser.add_argument("--audit-output", type=Path, default=OUT_AUDIT)
    args = parser.parse_args()
    render(args.pearson_output, gap_mode=False)
    render(args.audit_output, gap_mode=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
