#!/usr/bin/env python3
"""Render publication-ready Markov leaf-size figures.

Reads the v3 publication-scope summary.json (produced by the Markov
overnight bundle or one of the archived
``markov_v3_publication_leaf_size_fixed_docs_*`` runs) and renders:

- ``markov_leaf_size_fixed_recoverable`` — (root-share x leaf-tokens) grid of
  tree-oracle root MAE for the recoverable scope, Blues cmap.
- ``markov_leaf_size_fixed_structural`` — same grid for the structural scope.

The two figures share axis orientation and cell-text conventions with the
manifesto f/g ladder publication figures so the paper's figure suite reads
coherently.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import paperplot  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


SCOPE_TITLES = {
    "recoverable_v4": "Recoverable (oracle-attainable MAE floor)",
    "r12_seg10to12": "Structural (r12, seg 10-12)",
}
SCOPE_STEM = {
    "recoverable_v4": "markov_leaf_size_fixed_recoverable",
    "r12_seg10to12": "markov_leaf_size_fixed_structural",
}


def _panels_to_grid(panels: list[dict]) -> tuple[np.ndarray, list[int], list[int]]:
    """Reshape panel summaries into a (root_share x leaf_token) matrix."""
    leaves_set: set[int] = set()
    root_shares: list[int] = []
    for panel in panels:
        root_shares.append(int(panel["root_share"]))
        for leaf_str in panel.get("tree_root_mae_by_leaf_tokens", {}):
            leaves_set.add(int(leaf_str))
    leaves = sorted(leaves_set, reverse=True)  # 128 on the left, 8 on the right
    root_shares = sorted(set(root_shares), reverse=True)  # 100 on top, 10 on bottom
    matrix = np.full((len(root_shares), len(leaves)), np.nan, dtype=float)
    for panel in panels:
        r = root_shares.index(int(panel["root_share"]))
        for leaf_str, value in panel.get("tree_root_mae_by_leaf_tokens", {}).items():
            if value is None:
                continue
            c = leaves.index(int(leaf_str))
            matrix[r, c] = float(value)
    return matrix, root_shares, leaves


def _fno_references(panels: list[dict]) -> dict[int, float]:
    return {int(p["root_share"]): float(p.get("official_fno_root_mae") or 0.0)
            for p in panels}


def render_scope(
    scope_key: str,
    scope_data: dict,
    *,
    output_dir: Path,
    anchor_color: str,
) -> Path:
    panels = scope_data.get("panel_summaries", [])
    if not panels:
        raise ValueError(f"no panels for scope {scope_key}")
    matrix, root_shares, leaves = _panels_to_grid(panels)

    # Symmetric sequential scale across the full matrix, capped at the
    # observed max (rounded up to a clean tick).
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        raise ValueError(f"no finite values for scope {scope_key}")
    vmax = float(np.ceil(finite.max() * 20) / 20)  # nearest 0.05
    vmax = max(vmax, 0.05)
    vmin = 0.0

    fno_refs = _fno_references(panels)
    reference_value = (
        np.mean(list(fno_refs.values())) if fno_refs else None
    )
    if reference_value is not None and not (vmin < reference_value < vmax):
        reference_value = None  # fno-mae 0 would land on the axis edge

    fig, ax = plt.subplots(figsize=paperplot.FIGSIZE_FULL)
    paperplot.heatmap_with_reference(
        ax,
        matrix,
        row_labels=[f"R{rs}" for rs in root_shares],
        col_labels=leaves,
        cmap="Blues",
        vmin=vmin,
        vmax=vmax,
        reference_value=reference_value,
        reference_label=(
            f"FNO avg MAE = {reference_value:.3f}"
            if reference_value is not None else None
        ),
        bold_reference=True,
        colorbar_label="Tree-oracle root MAE (lower = better)",
        title=SCOPE_TITLES.get(scope_key, scope_key),
        xlabel="leaf size (tokens)",
        ylabel="root-share %",
    )
    fig.suptitle(
        "Markov leaf-size view (fixed train corpus)",
        fontsize=10,
        color=anchor_color,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    stem = output_dir / SCOPE_STEM.get(scope_key, f"markov_{scope_key}")
    written = paperplot.save(fig, stem)
    plt.close(fig)
    return Path(written[0])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to outputs/markov_v3_publication_leaf_size_fixed_docs_*/summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/ctreepo/assets/markov/figures"),
    )
    parser.add_argument(
        "--scopes",
        nargs="+",
        default=["recoverable_v4", "r12_seg10to12"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paperplot.rcparams()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.summary.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    scopes = summary.get("scopes", {})
    anchor = paperplot.ANCHOR_COLORS["markov"]
    written: list[Path] = []
    for scope_key in args.scopes:
        scope_data = scopes.get(scope_key)
        if scope_data is None:
            print(f"skip {scope_key} (not in summary)", file=sys.stderr)
            continue
        out = render_scope(scope_key, scope_data, output_dir=args.output_dir,
                           anchor_color=anchor)
        written.append(out)

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
