#!/usr/bin/env python3
"""Plot learned classical-sketch supervision-rate comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


LEAF_COLORS = {
    16: "#0072B2",
    64: "#009E73",
    256: "#D55E00",
    512: "#CC79A7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=Path("outputs/classical_sketches_supervision_r_comparison_20260512"),
        help="Directory containing *_vs_root_only.csv comparison tables.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=None,
        help="Output stem. Defaults to <comparison-dir>/classical_supervision_r_frontier.",
    )
    parser.add_argument(
        "--include-uniform-pilot",
        action="store_true",
        help=(
            "Include available corrected uniform all-node cells. These are useful as "
            "a pilot overlay, but may be incomplete while the full grid is running."
        ),
    )
    parser.add_argument(
        "--x-axis",
        choices=("expected-nodes", "rate"),
        default="expected-nodes",
        help=(
            "Use expected supervised nodes per document or the supervision rate R "
            "as the horizontal axis."
        ),
    )
    return parser.parse_args()


def _read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _leaf_label(leaf: int) -> str:
    if int(leaf) == 512:
        return "full doc (1 leaf)"
    return f"{leaf} tok/leaf"


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#d7d7d7", linewidth=0.75, alpha=0.75)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.6, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8.5)


def _x_values(group: pd.DataFrame, *, x_axis: str) -> pd.Series:
    if x_axis == "rate":
        return group["R"].astype(float)
    return group["expected_labels_per_doc"]


def _root_x(*, x_axis: str) -> float:
    if x_axis == "rate":
        return 0.0
    return 1.0


def _configure_x_axis(ax: plt.Axes, *, x_axis: str) -> None:
    if x_axis == "rate":
        ax.set_xlabel("Extra internal-node rate R (%)\nall roots observed", fontsize=9.5)
        ax.set_xlim(-4.0, 104.0)
        ax.set_xticks([0, 10, 30, 50, 70, 100])
    else:
        ax.set_xscale("log")
        ax.set_xlabel("Expected supervised nodes per document", fontsize=9.5)


def _plot_frontier(
    ax: plt.Axes,
    *,
    root: pd.DataFrame,
    legacy: pd.DataFrame,
    uniform: pd.DataFrame,
    x_axis: str,
) -> None:
    for leaf, group in legacy.groupby("leaf_size"):
        leaf = int(leaf)
        color = LEAF_COLORS.get(leaf, "#555555")
        group = group.sort_values("R" if x_axis == "rate" else "expected_labels_per_doc")
        ax.plot(
            _x_values(group, x_axis=x_axis),
            group["rx_rel"],
            color=color,
            marker="o",
            markersize=4.8,
            linewidth=1.55,
            label=_leaf_label(leaf),
        )
    for _, row in root.sort_values("leaf_size").iterrows():
        leaf = int(row["leaf_size"])
        color = LEAF_COLORS.get(leaf, "#555555")
        ax.scatter(
            [_root_x(x_axis=x_axis)],
            [float(row["root_rel"])],
            marker="s",
            s=45,
            facecolor="white",
            edgecolor=color,
            linewidth=1.4,
            zorder=5,
        )
    if not uniform.empty:
        ax.plot(
            _x_values(uniform, x_axis=x_axis),
            uniform["rx_rel"],
            color="#222222",
            marker="D",
            markersize=4.8,
            linewidth=1.55,
            linestyle="--",
            label="uniform nodes, 16 tok/leaf",
        )
    _configure_x_axis(ax, x_axis=x_axis)
    ax.set_ylabel("Mean relative error", fontsize=9.5)
    ax.set_title("A. Supervision Frontier", loc="left", fontsize=10.5, fontweight="bold")
    _style_axes(ax)


def _plot_delta(
    ax: plt.Axes,
    *,
    legacy: pd.DataFrame,
    uniform: pd.DataFrame,
    x_axis: str,
) -> None:
    ax.axhline(0.0, color="#222222", linewidth=0.85, alpha=0.75)
    for leaf, group in legacy.groupby("leaf_size"):
        leaf = int(leaf)
        color = LEAF_COLORS.get(leaf, "#555555")
        group = group.sort_values("R" if x_axis == "rate" else "expected_labels_per_doc")
        ax.plot(
            _x_values(group, x_axis=x_axis),
            group["delta_rel"],
            color=color,
            marker="o",
            markersize=4.8,
            linewidth=1.55,
        )
    if not uniform.empty:
        ax.plot(
            _x_values(uniform, x_axis=x_axis),
            uniform["delta_rel"],
            color="#222222",
            marker="D",
            markersize=4.8,
            linewidth=1.55,
            linestyle="--",
        )
    _configure_x_axis(ax, x_axis=x_axis)
    ax.set_ylabel("Change vs root-only", fontsize=9.5)
    ax.set_title("B. Gain Relative to Root-Only", loc="left", fontsize=10.5, fontweight="bold")
    ax.text(
        0.02,
        0.07,
        "lower is better",
        transform=ax.transAxes,
        fontsize=8.2,
        color="#555555",
    )
    _style_axes(ax)


def _semantic_handles(*, include_uniform: bool) -> list[Line2D]:
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markersize=6,
            markerfacecolor="white",
            markeredgecolor="#444444",
            label="rootR100 leafR0 internalR0",
        ),
        Line2D(
            [0],
            [0],
            color="#444444",
            marker="o",
            linewidth=1.5,
            label="root+internal: all roots + R internal",
        ),
    ]
    if include_uniform:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#222222",
                marker="D",
                linestyle="--",
                linewidth=1.5,
                label="pilot: root = leaf = internal = R",
            )
        )
    for leaf, color in LEAF_COLORS.items():
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2,
                label=_leaf_label(leaf),
            )
        )
    return handles


def main() -> int:
    args = parse_args()
    comparison_dir = args.comparison_dir
    output_stem = args.output_stem or comparison_dir / "classical_supervision_r_frontier"

    root = _read_optional(comparison_dir / "root_only_baseline.csv")
    legacy = _read_optional(comparison_dir / "legacy_internal_vs_root_only.csv")
    uniform = (
        _read_optional(comparison_dir / "uniform_all_nodes_vs_root_only.csv")
        if bool(args.include_uniform_pilot)
        else pd.DataFrame()
    )
    if root.empty:
        raise SystemExit(f"missing root-only baseline in {comparison_dir}")
    if legacy.empty and uniform.empty:
        raise SystemExit(f"missing R-grid comparison tables in {comparison_dir}")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.35, 4.15), constrained_layout=False)
    _plot_frontier(axes[0], root=root, legacy=legacy, uniform=uniform, x_axis=args.x_axis)
    _plot_delta(axes[1], legacy=legacy, uniform=uniform, x_axis=args.x_axis)

    fig.subplots_adjust(left=0.085, right=0.99, top=0.84, bottom=0.29, wspace=0.24)
    fig.legend(
        handles=_semantic_handles(include_uniform=not uniform.empty),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3 if uniform.empty else 4,
        frameon=False,
        fontsize=7.7,
        handlelength=1.8,
        columnspacing=1.25,
    )
    fig.suptitle(
        (
            "Classical Sketch Root + Internal Supervision Sweep"
            if uniform.empty
            else "Classical Sketch Local-Supervision Rate Sweep"
        ),
        x=0.02,
        y=0.965,
        ha="left",
        fontsize=11.2,
        fontweight="bold",
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (
        (".pdf", {}),
        (".png", {"dpi": 360}),
        (".svg", {}),
    ):
        fig.savefig(output_stem.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)

    manifest = {
        "comparison_dir": str(comparison_dir),
        "include_uniform_pilot": bool(args.include_uniform_pilot),
        "x_axis": str(args.x_axis),
        "outputs": {
            "pdf": str(output_stem.with_suffix(".pdf")),
            "png": str(output_stem.with_suffix(".png")),
            "svg": str(output_stem.with_suffix(".svg")),
            "caption": str(output_stem.with_suffix(".caption.md")),
        },
        "notes": [
            "Root-only is rootR100_leafR0_internalR0.",
            "Root+internal R grid is all roots plus R-sampled internal nodes, no leaf supervision.",
        ],
    }
    if not uniform.empty:
        manifest["notes"].append(
            "Uniform R grid is root=leaf=internal=R over the full tree-node population."
        )
    caption = (
        "**Classical Sketch Root + Internal Supervision Frontier.** Mean relative "
        "error for learned classical sketches as a function of expected "
        "oracle-supervised tree nodes per document. Open squares are the root-only "
        "baseline (`rootR100_leafR0_internalR0`). Circles show the complete "
        "root+internal grid, which observes all document roots plus an `R` fraction "
        "of internal nodes and no leaf labels; in this figure `R100` therefore means "
        "all roots plus all internal nodes, not full-tree supervision. These cells "
        "are not constant-mass allocations: increasing `R` adds internal-node labels "
        "on top of the fixed all-root labels, rather than replacing root labels. "
        "Panel A shows absolute mean relative error; Panel B shows change relative "
        "to the matched root-only baseline, so values below zero improve on "
        "root-only. Since "
        "documents are capped at 512 tokens in this suite, the 512-token/leaf "
        "condition is the full-document / one-leaf endpoint; it collapses near "
        "the root-only point because there is no internal-node budget to sweep."
    )
    if args.x_axis == "rate":
        caption = caption.replace(
            "as a function of expected oracle-supervised tree nodes per document",
            "as a function of the internal-node supervision rate `R`",
        )
        caption += (
            " This rate-axis view spreads the R-grid evenly; unlike the budget-axis "
            "view, it does not encode the fact that larger token-per-leaf settings "
            "have many fewer internal nodes per document."
        )
    if not uniform.empty:
        caption += (
            " Black diamonds overlay currently available corrected uniform all-node "
            "pilot cells, where `R` means the same sampling rate for roots, leaves, "
            "and internal nodes."
        )
    output_stem.with_suffix(".caption.md").write_text(caption + "\n", encoding="utf-8")
    output_stem.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
