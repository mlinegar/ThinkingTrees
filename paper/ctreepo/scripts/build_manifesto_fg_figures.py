#!/usr/bin/env python3
"""Render publication-ready manifesto f/g ladder figures.

Produces three figure families for the minimal paper:

- ``manifesto_fg_ladder_benoit_init`` (App. H, Benoit-init lane, two-panel heatmap)
- ``manifesto_fg_ladder_raw_init``    (App. H, raw-init lane, two-panel heatmap)
- ``manifesto_fg_ladder_headline``    (§9.4, per-leaf best line plot)

All figures call :func:`paperplot.rcparams` up front and save PDF+PNG via
:func:`paperplot.save`, so typography and anchor colors stay consistent with
the rest of the paper figure suite.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import _ladder_io as ladder_io  # noqa: E402
import benoit_references as refs  # noqa: E402
import paperplot  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


LANE_CANONICAL_STAGES_AB: dict[str, tuple[tuple[int, int], ...]] = {
    refs.BENOIT_INIT_LANE: ((1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3)),
    refs.RAW_INIT_LANE: (
        (0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3),
    ),
}
LANE_TITLE = {
    refs.BENOIT_INIT_LANE: r"Benoit-init lane ($g^0 = g^{\mathrm{Benoit}}$)",
    refs.RAW_INIT_LANE: r"Raw-init lane ($g^0$ = own-Gemma baseline)",
}
LANE_MARKER = {refs.BENOIT_INIT_LANE: "o", refs.RAW_INIT_LANE: "s"}
LANE_LINESTYLE = {refs.BENOIT_INIT_LANE: "-", refs.RAW_INIT_LANE: "--"}


def _stage_display(ab: tuple[int, int]) -> str:
    a, b = ab
    return rf"$f^{{{a}}} g^{{{b}}}$"


def _prepare_lane_matrix(
    cells: list[ladder_io.LadderCell],
    *,
    lane: str,
    metric: str,
):
    """Build a (stages × leaves) matrix keyed by canonical (a, b) tuples.

    Inputs may use legacy (``fg``, ``fgf``) or power (``f1g1``) stage names
    interchangeably; both map to the same ``(a, b)`` identity via
    ``LadderCell.stage_ab``.
    """
    leaves = ladder_io.sorted_leaves(cells)
    observed: set[tuple[int, int]] = set()
    for cell in cells:
        ab = cell.stage_ab
        if ab is not None:
            observed.add(ab)
    stage_abs = [ab for ab in LANE_CANONICAL_STAGES_AB[lane] if ab in observed]
    # include any extra observed stages not in the canonical list
    extras = sorted(observed - set(stage_abs), key=lambda ab: (ab[0] + ab[1], ab[0]))
    stage_abs.extend(extras)

    matrix = np.full((len(stage_abs), len(leaves)), np.nan, dtype=float)
    mask = np.zeros_like(matrix, dtype=bool)

    groups: dict[tuple[int, int], list[ladder_io.LadderCell]] = {}
    for cell in cells:
        ab = cell.stage_ab
        if ab is None:
            continue
        groups.setdefault((cell.leaf_size_tokens, ab[0], ab[1]), []).append(cell)

    for (leaf, a, b), group in groups.items():
        group = sorted(group, key=lambda c: c.source_created_at, reverse=True)
        cell = group[0]
        try:
            r = stage_abs.index((a, b))
            c = leaves.index(leaf)
        except ValueError:
            continue
        value = getattr(cell, metric)
        if value is not None:
            matrix[r, c] = value

    # inherited mask: per leaf, across canonical-stage order, any stage whose
    # (ext_pearson, int_pearson, source_path) repeats the prior stage's tuple.
    for c_idx, leaf in enumerate(leaves):
        last_sig: tuple | None = None
        for r_idx, ab in enumerate(stage_abs):
            group = groups.get((leaf, ab[0], ab[1]))
            if not group:
                last_sig = None
                continue
            chosen = sorted(group, key=lambda c: c.source_created_at, reverse=True)[0]
            sig = (
                chosen.external_expert_pearson,
                chosen.internal_f_pearson,
                chosen.source_path,
            )
            if last_sig is not None and sig == last_sig:
                mask[r_idx, c_idx] = True
            last_sig = sig

    return matrix, stage_abs, leaves, mask


def render_lane_heatmap(
    cells: list[ladder_io.LadderCell],
    *,
    lane: str,
    output_stem: Path,
    anchor_color: str,
) -> Path:
    """Render the two-panel (external Pearson, gap) heatmap for one lane."""
    ext_matrix, stage_abs, leaves, ext_mask = _prepare_lane_matrix(
        cells, lane=lane, metric="external_expert_pearson"
    )
    gap_matrix, _, _, gap_mask = _prepare_lane_matrix(
        cells, lane=lane, metric="f_star_gap"
    )

    stage_display = [_stage_display(ab) for ab in stage_abs]

    fig_w, fig_h = paperplot.FIGSIZE_FULL_TALL
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))

    paperplot.heatmap_with_reference(
        axes[0],
        ext_matrix,
        row_labels=stage_display,
        col_labels=leaves,
        cmap="YlGn",
        vmin=refs.ECONOMIC_PEARSON_AXIS[0],
        vmax=refs.ECONOMIC_PEARSON_AXIS[1],
        reference_value=refs.BENOIT_SPLIT_EXPERT_R,
        reference_label=f"r={refs.BENOIT_SPLIT_EXPERT_R:.3f}",
        bold_reference=True,
        inherited_mask=ext_mask,
        colorbar_label="External Pearson r (higher = better)",
        title="External expert agreement",
        xlabel="leaf size (tokens)",
        ylabel="ladder stage",
    )
    paperplot.heatmap_with_reference(
        axes[1],
        gap_matrix,
        row_labels=stage_display,
        col_labels=leaves,
        cmap="RdBu_r",
        vmin=refs.ECONOMIC_GAP_AXIS[0],
        vmax=refs.ECONOMIC_GAP_AXIS[1],
        vcenter=refs.PARITY_GAP,
        reference_value=refs.PARITY_GAP,
        reference_label="parity",
        bold_reference=False,
        inherited_mask=gap_mask,
        colorbar_label="Internal$-$external gap (lower = better)",
        title="Internal-external Pearson gap",
        xlabel="leaf size (tokens)",
        ylabel="",
    )

    fig.suptitle(
        LANE_TITLE[lane],
        fontsize=10,
        color=anchor_color,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    written = paperplot.save(fig, output_stem)
    plt.close(fig)
    return Path(written[0])


def render_headline(
    cells_by_lane: dict[str, list[ladder_io.LadderCell]],
    *,
    output_stem: Path,
    anchor_color: str,
) -> Path:
    """Main-text line plot: per-leaf best external Pearson, both lanes."""
    fig, ax = plt.subplots(figsize=paperplot.FIGSIZE_ONE_COL_TALL)

    ax.axhline(
        refs.BENOIT_SPLIT_EXPERT_R,
        color=paperplot.ANCHOR_COLORS["theory"],
        linestyle="-.",
        linewidth=0.9,
        label=f"split-expert r={refs.BENOIT_SPLIT_EXPERT_R:.3f}",
        zorder=1,
    )

    for lane in (refs.BENOIT_INIT_LANE, refs.RAW_INIT_LANE):
        cells = cells_by_lane.get(lane)
        if not cells:
            continue
        bests = ladder_io.per_leaf_best(cells, metric="external_expert_pearson")
        xs = [b.leaf_size_tokens for b in bests]
        ys = [b.external_expert_pearson for b in bests]
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            marker=LANE_MARKER[lane],
            linestyle=LANE_LINESTYLE[lane],
            color=anchor_color,
            markersize=5,
            linewidth=1.3,
            label=LANE_TITLE[lane].replace(r"\mathrm", ""),
            zorder=3,
        )

    headline = None
    for cell in cells_by_lane.get(refs.RAW_INIT_LANE, []):
        if (
            cell.lane == refs.HEADLINE_CELL["lane"]
            and cell.leaf_size_tokens == refs.HEADLINE_CELL["leaf_size_tokens"]
            and cell.stage_name == refs.HEADLINE_CELL["stage_name"]
        ):
            headline = cell
            break
    if headline is not None and headline.external_expert_pearson is not None:
        ax.annotate(
            rf"$f^1 g^1$: $r={headline.external_expert_pearson:.3f}$,"
            rf" gap $={headline.f_star_gap:.3f}$",
            xy=(headline.leaf_size_tokens, headline.external_expert_pearson),
            xytext=(6, -14),
            textcoords="offset points",
            fontsize=7,
            color=anchor_color,
            ha="left",
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("leaf size (tokens)")
    ax.set_ylabel("External expert Pearson r")
    ax.set_ylim(refs.ECONOMIC_PEARSON_AXIS[0], refs.ECONOMIC_PEARSON_AXIS[1])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{int(v):d}"))
    ax.grid(True, which="both", axis="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title(
        "Leaf-size robustness of the alternating $f/g$ ladder",
        fontsize=9,
    )
    fig.tight_layout()
    written = paperplot.save(fig, output_stem, formats=("pdf", "png"))
    plt.close(fig)
    return Path(written[0])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benoit-csv", action="append", default=[])
    parser.add_argument("--raw-csv", action="append", default=[])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/ctreepo/assets/benoit/figures"),
    )
    parser.add_argument("--skip-headline-check", action="store_true")
    parser.add_argument(
        "--figures",
        nargs="+",
        default=["benoit_init", "raw_init", "headline"],
        choices=["benoit_init", "raw_init", "headline"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paperplot.rcparams()

    cells_by_lane: dict[str, list[ladder_io.LadderCell]] = {}
    if args.benoit_csv:
        cells_by_lane[refs.BENOIT_INIT_LANE] = ladder_io.load_cells(
            args.benoit_csv, lane=refs.BENOIT_INIT_LANE
        )
    if args.raw_csv:
        cells_by_lane[refs.RAW_INIT_LANE] = ladder_io.load_cells(
            args.raw_csv, lane=refs.RAW_INIT_LANE
        )

    if not args.skip_headline_check and refs.RAW_INIT_LANE in cells_by_lane:
        ladder_io.assert_headline(
            cells_by_lane[refs.RAW_INIT_LANE],
            expected=refs.HEADLINE_CELL,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    anchor = paperplot.ANCHOR_COLORS["manifesto"]
    written: list[Path] = []

    if "benoit_init" in args.figures and cells_by_lane.get(refs.BENOIT_INIT_LANE):
        out = render_lane_heatmap(
            cells_by_lane[refs.BENOIT_INIT_LANE],
            lane=refs.BENOIT_INIT_LANE,
            output_stem=args.output_dir / "manifesto_fg_ladder_benoit_init",
            anchor_color=anchor,
        )
        written.append(out)
    if "raw_init" in args.figures and cells_by_lane.get(refs.RAW_INIT_LANE):
        out = render_lane_heatmap(
            cells_by_lane[refs.RAW_INIT_LANE],
            lane=refs.RAW_INIT_LANE,
            output_stem=args.output_dir / "manifesto_fg_ladder_raw_init",
            anchor_color=anchor,
        )
        written.append(out)
    if "headline" in args.figures and cells_by_lane:
        out = render_headline(
            cells_by_lane,
            output_stem=args.output_dir / "manifesto_fg_ladder_headline",
            anchor_color=anchor,
        )
        written.append(out)

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
