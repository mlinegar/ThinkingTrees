#!/usr/bin/env python3
"""Render simplified Markov leaf-size x supervision plots.

The existing allocation figures show every tree-supervision policy. This
renderer collapses those tree policies into one envelope: the best trained tree
available at each (root share, leaf size) cell. It keeps only simple comparison
anchors: the flat FNO line, the one-leaf parity marker, and the richer one-leaf
local-target diagnostic when available.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = (
    REPO_ROOT
    / "outputs"
    / "markov_v5_sticky_allocation_policy_paper_20260416"
    / "summary.json"
)
DEFAULT_DIAGNOSTICS = (
    REPO_ROOT
    / "outputs"
    / "markov_v5_simple_current_plots_20260415_233539"
    / "summary.json"
)
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "markov_v5_best_tree_leaf_supervision"

TREE_FAMILIES = ("root_only", "leaf_only", "depth_equal", "balanced_node")
TREE_FAMILY_LABELS = {
    "root_only": "root-only",
    "leaf_only": "leaf-local",
    "depth_equal": "depth-local",
    "balanced_node": "node-local",
}
SCOPE_ORDER = ("recoverable_v5_t128", "r12_p079")
SCOPE_SHORT = {
    "recoverable_v5_t128": "simple",
    "r12_p079": "hard",
}
SCOPE_SUBTITLES = {
    "recoverable_v5_t128": "4 colors, about 5 changes per 128-token document",
    "r12_p079": "12 colors, about 10 changes per 128-token document",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_diagnostics(path: Path) -> dict[str, dict[int, dict[str, float | None]]]:
    if not path.exists():
        return {}
    payload = _load_json(path)
    out: dict[str, dict[int, dict[str, float | None]]] = {}
    for scope_key, scope_data in dict(payload.get("scopes") or {}).items():
        panels: dict[int, dict[str, float | None]] = {}
        for panel in list(scope_data.get("panel_summaries") or []):
            root_share = int(panel.get("root_share", 0) or 0)
            canary = (panel.get("one_leaf_canary_root_mae_by_leaf_tokens") or {}).get("128")
            panels[root_share] = {
                "parity_tree": float(canary) if canary is not None else None,
                "fno_actual": _maybe_float(panel.get("official_fno_actual_root_mae")),
                "richer_local": _maybe_float(panel.get("one_leaf_duplicate_local_label_root_mae")),
                "empirical_bayes": _maybe_float(panel.get("empirical_bayes_root_mae")),
            }
        out[scope_key] = panels
    return out


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _points_by_leaf(points: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for point in points:
        leaf_tokens = int(point.get("leaf_tokens", 0) or 0)
        if leaf_tokens <= 0:
            continue
        root_mae = _maybe_float(point.get("root_mae"))
        if root_mae is None:
            continue
        out[leaf_tokens] = {
            "leaf_tokens": leaf_tokens,
            "root_mae": root_mae,
            "package_name": str(point.get("package_name", "") or ""),
        }
    return out


def _best_tree_points(panel: Mapping[str, Any], leaf_tokens: Sequence[int]) -> list[dict[str, Any]]:
    series = dict(panel.get("series") or {})
    by_family = {
        family: _points_by_leaf(list(series.get(family) or []))
        for family in TREE_FAMILIES
    }
    best: list[dict[str, Any]] = []
    for leaf in leaf_tokens:
        candidates: list[dict[str, Any]] = []
        for family, family_points in by_family.items():
            point = family_points.get(int(leaf))
            if not point:
                continue
            candidates.append(
                {
                    **point,
                    "tree_family": family,
                    "tree_family_label": TREE_FAMILY_LABELS.get(family, family),
                }
            )
        if not candidates:
            continue
        best.append(min(candidates, key=lambda row: float(row["root_mae"])))
    return best


def _scope_ylim(scope_view: Mapping[str, Any], diagnostics: Mapping[int, Mapping[str, float | None]]) -> tuple[float, float]:
    values: list[float] = []
    leaf_tokens = [int(v) for v in scope_view.get("leaf_tokens", [])]
    for panel in list(scope_view.get("panels") or []):
        values.extend(float(p["root_mae"]) for p in _best_tree_points(panel, leaf_tokens))
        fno = _maybe_float(panel.get("fno_root_mae"))
        if fno is not None:
            values.append(fno)
        root_share = int(panel.get("root_share", 0) or 0)
        diag = diagnostics.get(root_share, {})
        for key in ("parity_tree", "richer_local"):
            value = diag.get(key)
            if value is not None:
                values.append(float(value))
    if not values:
        return (0.0, 1.0)
    upper = max(values) * 1.12
    return (0.0, max(upper, 0.1))


def _panel_title(root_share: int, train_docs: int) -> str:
    labeled_docs = int(round(train_docs * root_share / 100.0))
    if root_share == 100:
        return f"All {labeled_docs:,} root docs"
    return f"R{root_share}: {labeled_docs:,} root docs"


def _render_scope(
    *,
    scope_key: str,
    scope_data: Mapping[str, Any],
    diagnostics: Mapping[int, Mapping[str, float | None]],
    output_dir: Path,
    panels_to_show: Sequence[int] | None,
    suffix: str,
    train_docs: int,
) -> list[dict[str, Any]]:
    view = dict(scope_data.get("replacement_view") or {})
    panels = list(view.get("panels") or [])
    if panels_to_show is not None:
        show = {int(v) for v in panels_to_show}
        panels = [p for p in panels if int(p.get("root_share", 0) or 0) in show]
    leaf_tokens = [int(v) for v in view.get("leaf_tokens", [])]
    if not panels or not leaf_tokens:
        return []

    if len(panels) <= 3:
        nrows, ncols = 1, len(panels)
        fig_size = (5.0 * len(panels), 4.7)
        legend_y = 0.01
        bottom = 0.22
    else:
        nrows, ncols = 2, 5
        fig_size = (15.8, 8.7)
        legend_y = 0.005
        bottom = 0.15

    fig, axes = plt.subplots(nrows, ncols, figsize=fig_size, sharey=True)
    axes_list = [axes] if len(panels) == 1 else list(axes.flatten())

    x_positions = list(range(len(leaf_tokens)))
    x_labels = [f"{leaf}\n({128 // leaf} leaves)" for leaf in leaf_tokens]
    leaf128_x = leaf_tokens.index(128) if 128 in leaf_tokens else None
    y_min, y_max = _scope_ylim(view, diagnostics)

    rows: list[dict[str, Any]] = []
    for idx, panel in enumerate(panels):
        ax = axes_list[idx]
        root_share = int(panel.get("root_share", 0) or 0)
        ax.set_title(_panel_title(root_share, train_docs), fontsize=10, fontweight="bold")

        best_points = _best_tree_points(panel, leaf_tokens)
        xs = [x_positions[leaf_tokens.index(int(p["leaf_tokens"]))] for p in best_points]
        ys = [float(p["root_mae"]) for p in best_points]
        if xs:
            ax.plot(
                xs,
                ys,
                color="#2166ac",
                marker="o",
                linewidth=2.4,
                markersize=5,
                label="Best trained tree",
            )

        fno = _maybe_float(panel.get("fno_root_mae"))
        if fno is not None:
            ax.plot(
                x_positions,
                [fno] * len(x_positions),
                color="#d18f00",
                linestyle=":",
                linewidth=2.0,
                label="Flat FNO",
            )

        diag = diagnostics.get(root_share, {})
        if leaf128_x is not None:
            parity = diag.get("parity_tree")
            if parity is not None:
                ax.scatter(
                    [leaf128_x],
                    [float(parity)],
                    marker="D",
                    s=48,
                    facecolors="white",
                    edgecolors="#222222",
                    linewidths=1.3,
                    zorder=8,
                    label="1-leaf parity",
                )
            richer = diag.get("richer_local")
            if richer is not None:
                ax.scatter(
                    [leaf128_x],
                    [float(richer)],
                    marker="^",
                    s=54,
                    color="#666666",
                    zorder=7,
                    label="Richer 1-leaf local target",
                )

        ax.set_xticks(x_positions, x_labels, fontsize=8)
        ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.25, linewidth=0.6)
        if idx % ncols == 0:
            ax.set_ylabel("Test MAE (lower is better)", fontsize=9)
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Leaf size", fontsize=9)

        for point in best_points:
            leaf = int(point["leaf_tokens"])
            rows.append(
                {
                    "scope_key": scope_key,
                    "scope_title": str(scope_data.get("title") or scope_key),
                    "root_share": root_share,
                    "leaf_tokens": leaf,
                    "best_tree_root_mae": float(point["root_mae"]),
                    "best_tree_family": str(point["tree_family"]),
                    "best_tree_family_label": str(point["tree_family_label"]),
                    "best_tree_package": str(point.get("package_name", "")),
                    "fno_root_mae": fno,
                    "one_leaf_parity_root_mae": diag.get("parity_tree"),
                    "richer_one_leaf_local_target_root_mae": diag.get("richer_local"),
                }
            )

    for ax in axes_list[len(panels) :]:
        ax.axis("off")

    legend_handles = [
        Line2D([0], [0], color="#2166ac", marker="o", linewidth=2.4, markersize=5, label="Best trained tree"),
        Line2D([0], [0], color="#d18f00", linestyle=":", linewidth=2.0, label="Flat FNO"),
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="D",
            markerfacecolor="white",
            markeredgecolor="#222222",
            markeredgewidth=1.3,
            markersize=6,
            label="1-leaf parity",
        ),
        Line2D(
            [0],
            [0],
            linestyle="None",
            marker="^",
            color="#666666",
            markersize=6,
            label="Richer 1-leaf local target",
        ),
    ]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=4,
        frameon=True,
        fancybox=True,
        fontsize=8,
        edgecolor="#cccccc",
    )

    title = str(scope_data.get("title") or scope_key)
    subtitle = SCOPE_SUBTITLES.get(scope_key, "")
    fig.suptitle(f"{title}: Best Trained Tree vs Flat FNO", y=0.99, fontsize=13, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.945, subtitle, ha="center", va="top", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0.0, bottom, 1.0, 0.925))

    short = SCOPE_SHORT.get(scope_key, scope_key.replace(":", "_"))
    png_path = output_dir / f"{short}_best_tree_leaf_supervision{suffix}.png"
    pdf_path = output_dir / f"{short}_best_tree_leaf_supervision{suffix}.pdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scope_key",
        "scope_title",
        "root_share",
        "leaf_tokens",
        "best_tree_root_mae",
        "best_tree_family",
        "best_tree_family_label",
        "best_tree_package",
        "fno_root_mae",
        "one_leaf_parity_root_mae",
        "richer_one_leaf_local_target_root_mae",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_report(path: Path, *, output_dir: Path, all_rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Markov Best-Tree Leaf/Supervision Plots",
        "",
        "These are simplified versions of the allocation-policy figures. The blue curve is the best trained tree available at each `(root share, leaf size)` cell, taking the minimum MAE over `root_only`, `leaf_only`, `depth_equal`, and `balanced_node` rows in the existing sticky allocation summary.",
        "",
        "Comparisons are intentionally sparse: flat FNO, one-leaf parity, and the richer one-leaf local-target diagnostic.",
        "",
        "## Figures",
        "",
    ]
    for name in [
        "simple_best_tree_leaf_supervision_all.png",
        "hard_best_tree_leaf_supervision_all.png",
        "simple_best_tree_leaf_supervision_compact.png",
        "hard_best_tree_leaf_supervision_compact.png",
    ]:
        lines.append(f"- [{name}]({name})")
    lines.extend(
        [
            "",
            "## Data",
            "",
            "- [best_tree_leaf_supervision_points.csv](best_tree_leaf_supervision_points.csv)",
            "",
            "## Notes",
            "",
            "- The Markov allocation artifact does not expose explicit learned-`f`/fixed-`f` ablation lanes. Those would need a separate run or a different summary source.",
            "- The CSV records which training family supplied each point on the best-tree envelope.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--train-docs", type=int, default=10240)
    args = parser.parse_args()

    summary = _load_json(args.summary)
    diagnostics = _load_diagnostics(args.diagnostics)
    scopes = dict(summary.get("scopes") or {})

    all_rows: list[dict[str, Any]] = []
    for scope_key in SCOPE_ORDER:
        scope_data = scopes.get(scope_key)
        if not scope_data:
            continue
        scope_diag = diagnostics.get(scope_key, {})
        all_rows.extend(
            _render_scope(
                scope_key=scope_key,
                scope_data=scope_data,
                diagnostics=scope_diag,
                output_dir=args.output_dir,
                panels_to_show=None,
                suffix="_all",
                train_docs=args.train_docs,
            )
        )
        all_rows.extend(
            _render_scope(
                scope_key=scope_key,
                scope_data=scope_data,
                diagnostics=scope_diag,
                output_dir=args.output_dir,
                panels_to_show=(100, 50, 10),
                suffix="_compact",
                train_docs=args.train_docs,
            )
        )

    _write_csv(args.output_dir / "best_tree_leaf_supervision_points.csv", all_rows)
    _write_report(args.output_dir / "report.md", output_dir=args.output_dir, all_rows=all_rows)
    print(f"wrote {args.output_dir / 'best_tree_leaf_supervision_points.csv'}")
    print(f"wrote {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
