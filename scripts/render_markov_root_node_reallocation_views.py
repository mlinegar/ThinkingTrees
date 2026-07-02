#!/usr/bin/env python3
"""Render root-data and root-to-node reallocation views for Markov v5."""
from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
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
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "markov_v5_best_tree_leaf_supervision"

TRAIN_DOCS = 10240
DOC_TOKENS = 128

SCOPE_ORDER = ("recoverable_v5_t128", "r12_p079")
SCOPE_SHORT = {
    "recoverable_v5_t128": "simple",
    "r12_p079": "hard",
}
SCOPE_SUBTITLES = {
    "recoverable_v5_t128": "4 colors, about 5 changes per 128-token document",
    "r12_p079": "12 colors, about 10 changes per 128-token document",
}

ROOT_ONLY = "root_only"
LOCAL_FAMILIES = ("leaf_only", "depth_equal", "balanced_node")
LOCAL_FAMILY_LABELS = {
    "leaf_only": "leaf-only",
    "depth_equal": "depth-equal",
    "balanced_node": "balanced-node",
}

LEAF_COLORS = {
    128: "#6f6f6f",
    64: "#2e7d32",
    32: "#2166ac",
    16: "#7b1fa2",
    8: "#c2185b",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _points_by_leaf(points: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for point in points:
        leaf_tokens = int(point.get("leaf_tokens", 0) or 0)
        root_mae = _maybe_float(point.get("root_mae"))
        if leaf_tokens <= 0 or root_mae is None:
            continue
        out[leaf_tokens] = {
            "leaf_tokens": int(leaf_tokens),
            "root_mae": float(root_mae),
            "package_name": str(point.get("package_name", "") or ""),
        }
    return out


def _panel_index(scope_data: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    view = dict(scope_data.get("replacement_view") or {})
    panels = list(view.get("panels") or [])
    return {
        int(panel.get("root_share", 0) or 0): dict(panel)
        for panel in panels
        if int(panel.get("root_share", 0) or 0) > 0
    }


def _root_only_point(panel: Mapping[str, Any], leaf_tokens: int) -> dict[str, Any] | None:
    series = dict(panel.get("series") or {})
    by_leaf = _points_by_leaf(list(series.get(ROOT_ONLY) or []))
    return by_leaf.get(int(leaf_tokens))


def _best_local_point(panel: Mapping[str, Any], leaf_tokens: int) -> dict[str, Any] | None:
    series = dict(panel.get("series") or {})
    candidates: list[dict[str, Any]] = []
    for family in LOCAL_FAMILIES:
        point = _points_by_leaf(list(series.get(family) or [])).get(int(leaf_tokens))
        if not point:
            continue
        candidates.append(
            {
                **point,
                "local_family": str(family),
                "local_family_label": LOCAL_FAMILY_LABELS[str(family)],
            }
        )
    if not candidates:
        return None
    return min(candidates, key=lambda item: float(item["root_mae"]))


def _root_docs(root_share: int, train_docs: int) -> int:
    return int(round(float(train_docs) * float(root_share) / 100.0))


def _leaves_per_doc(leaf_tokens: int) -> int:
    return int(round(float(DOC_TOKENS) / float(max(1, leaf_tokens))))


def _scope_title(scope_key: str, scope_data: Mapping[str, Any]) -> str:
    return str(scope_data.get("title") or scope_key)


def _set_root_share_axis(ax: plt.Axes, *, include_zero: bool = False) -> None:
    ticks = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
    if include_zero:
        ticks.append(0)
        ax.set_xlim(102, -2)
    else:
        ax.set_xlim(102, 8)
    ax.set_xticks(ticks)
    ax.grid(alpha=0.25, linewidth=0.6)


def _render_root_only_scaling(
    *,
    scope_key: str,
    scope_data: Mapping[str, Any],
    output_dir: Path,
    train_docs: int,
) -> list[dict[str, Any]]:
    panel_by_root = _panel_index(scope_data)
    root_shares = sorted(panel_by_root, reverse=True)
    view = dict(scope_data.get("replacement_view") or {})
    leaf_tokens = [int(value) for value in list(view.get("leaf_tokens") or [])]
    if not root_shares or not leaf_tokens:
        return []

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    rows: list[dict[str, Any]] = []
    y_values: list[float] = []

    for leaf in leaf_tokens:
        xs: list[int] = []
        ys: list[float] = []
        for root_share in root_shares:
            point = _root_only_point(panel_by_root[root_share], leaf)
            if not point:
                continue
            mae = float(point["root_mae"])
            xs.append(int(root_share))
            ys.append(mae)
            y_values.append(mae)
            rows.append(
                {
                    "scope_key": scope_key,
                    "scope_title": _scope_title(scope_key, scope_data),
                    "view": "root_only_scaling",
                    "root_share": int(root_share),
                    "root_docs": _root_docs(root_share, train_docs),
                    "leaf_tokens": int(leaf),
                    "leaves_per_doc": _leaves_per_doc(leaf),
                    "root_only_root_mae": mae,
                    "package_name": str(point.get("package_name", "")),
                    "fno_root_mae": _maybe_float(panel_by_root[root_share].get("fno_root_mae")),
                }
            )
        if xs:
            ax.plot(
                xs,
                ys,
                color=LEAF_COLORS.get(int(leaf), "#333333"),
                marker="o",
                linewidth=2.0,
                markersize=4.8,
                label=f"leaf {leaf} ({_leaves_per_doc(leaf)} leaves/doc)",
            )

    fno_xs: list[int] = []
    fno_ys: list[float] = []
    for root_share in root_shares:
        fno = _maybe_float(panel_by_root[root_share].get("fno_root_mae"))
        if fno is None:
            continue
        fno_xs.append(int(root_share))
        fno_ys.append(float(fno))
        y_values.append(float(fno))
    if fno_xs:
        ax.plot(
            fno_xs,
            fno_ys,
            color="#d18f00",
            linestyle=":",
            linewidth=2.4,
            label="flat FNO",
        )

    _set_root_share_axis(ax)
    upper = max(y_values) * 1.12 if y_values else 1.0
    ax.set_ylim(0.0, max(0.1, upper))
    ax.set_xlabel("Root-labeled training docs retained (%)")
    ax.set_ylabel("Test root MAE (lower is better)")
    title = _scope_title(scope_key, scope_data)
    subtitle = SCOPE_SUBTITLES.get(scope_key, "")
    ax.set_title(f"{title}: Root-Only Data Scaling by Leaf Geometry", fontsize=12, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.92, subtitle, ha="center", va="top", fontsize=9, color="#555555")
    note = (
        f"No local-node replacement here: R50 means {_root_docs(50, train_docs):,} root labels "
        f"from {int(train_docs):,} training docs, and total supervision mass falls with RXX."
    )
    fig.text(0.5, 0.035, textwrap.fill(note, width=115), ha="center", fontsize=8.5, color="#444444")
    ax.legend(loc="upper left", ncol=2, fontsize=8, frameon=True, edgecolor="#cccccc")
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.89))

    short = SCOPE_SHORT.get(scope_key, scope_key)
    png = output_dir / f"{short}_root_only_no_by_root_data.png"
    pdf = output_dir / f"{short}_root_only_no_by_root_data.pdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {pdf}")
    return rows


def _render_root_to_node_reallocation(
    *,
    scope_key: str,
    scope_data: Mapping[str, Any],
    output_dir: Path,
    train_docs: int,
) -> list[dict[str, Any]]:
    panel_by_root = _panel_index(scope_data)
    root_shares = sorted(panel_by_root, reverse=True)
    leaf_tokens = [64, 32, 16, 8]
    if not root_shares:
        return []

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.3), sharey=True)
    axes_list = list(axes.flatten())
    rows: list[dict[str, Any]] = []
    y_values: list[float] = []

    for ax, leaf in zip(axes_list, leaf_tokens):
        root_xs: list[int] = []
        root_ys: list[float] = []
        realloc_xs: list[int] = []
        realloc_ys: list[float] = []
        fno_xs: list[int] = []
        fno_ys: list[float] = []

        for root_share in root_shares:
            panel = panel_by_root[root_share]
            root_point = _root_only_point(panel, leaf)
            if root_point:
                mae = float(root_point["root_mae"])
                root_xs.append(int(root_share))
                root_ys.append(mae)
                y_values.append(mae)

            if int(root_share) == 100 and root_point:
                local_point = {
                    **root_point,
                    "local_family": "root_only",
                    "local_family_label": "all-root reference",
                }
            else:
                local_point = _best_local_point(panel, leaf)
            if local_point:
                mae = float(local_point["root_mae"])
                realloc_xs.append(int(root_share))
                realloc_ys.append(mae)
                y_values.append(mae)
                rows.append(
                    {
                        "scope_key": scope_key,
                        "scope_title": _scope_title(scope_key, scope_data),
                        "view": "root_to_node_reallocation",
                        "root_share": int(root_share),
                        "root_docs": _root_docs(root_share, train_docs),
                        "local_mass_percent": int(100 - root_share),
                        "leaf_tokens": int(leaf),
                        "leaves_per_doc": _leaves_per_doc(leaf),
                        "root_only_root_mae": (
                            float(root_point["root_mae"]) if root_point else ""
                        ),
                        "best_reallocated_root_mae": mae,
                        "best_reallocated_family": str(local_point["local_family"]),
                        "best_reallocated_family_label": str(local_point["local_family_label"]),
                        "best_reallocated_package": str(local_point.get("package_name", "")),
                        "fno_root_mae": _maybe_float(panel.get("fno_root_mae")),
                    }
                )

            fno = _maybe_float(panel.get("fno_root_mae"))
            if fno is not None:
                fno_xs.append(int(root_share))
                fno_ys.append(float(fno))
                y_values.append(float(fno))

        ax.plot(
            root_xs,
            root_ys,
            color="#2e7d32",
            marker="o",
            linewidth=1.9,
            markersize=4.5,
        )
        ax.plot(
            realloc_xs,
            realloc_ys,
            color="#2166ac",
            marker="s",
            linewidth=2.2,
            markersize=4.8,
        )
        ax.plot(
            fno_xs,
            fno_ys,
            color="#d18f00",
            linestyle=":",
            linewidth=2.0,
        )
        ax.set_title(f"leaf {leaf}: {_leaves_per_doc(leaf)} leaves/doc", fontsize=10, fontweight="bold")
        _set_root_share_axis(ax)
        ax.set_xlabel("Root mass retained (%)")
        ax.set_ylabel("Test root MAE")

    upper = max(y_values) * 1.12 if y_values else 1.0
    for ax in axes_list:
        ax.set_ylim(0.0, max(0.1, upper))

    legend_handles = [
        Line2D([0], [0], color="#2e7d32", marker="o", linewidth=1.9, markersize=5, label="root-only tree (shrinking mass)"),
        Line2D([0], [0], color="#2166ac", marker="s", linewidth=2.2, markersize=5, label="best node-reallocated tree (fixed mass)"),
        Line2D([0], [0], color="#d18f00", linestyle=":", linewidth=2.0, label="flat FNO (root-only)"),
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=3,
        frameon=True,
        edgecolor="#cccccc",
        fontsize=8.5,
    )

    title = _scope_title(scope_key, scope_data)
    subtitle = SCOPE_SUBTITLES.get(scope_key, "")
    fig.suptitle(
        f"{title}: Moving Root Supervision to Node Supervision",
        y=0.98,
        fontsize=12,
        fontweight="bold",
    )
    if subtitle:
        fig.text(0.5, 0.935, subtitle, ha="center", va="top", fontsize=9, color="#555555")
    note = (
        "Blue keeps total full-document-equivalent supervision mass at 100% by replacing "
        "the missing root mass with sampled leaf/internal labels; green and amber keep only "
        "the reduced root-label budget."
    )
    fig.text(0.5, 0.015, textwrap.fill(note, width=120), ha="center", fontsize=8.5, color="#444444")
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.91))

    short = SCOPE_SHORT.get(scope_key, scope_key)
    png = output_dir / f"{short}_root_to_node_reallocation_by_leaf.png"
    pdf = output_dir / f"{short}_root_to_node_reallocation_by_leaf.pdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {pdf}")
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_mechanics_note(path: Path, *, train_docs: int) -> None:
    lines = [
        "# Markov Root-to-Node Reallocation Mechanics",
        "",
        "This note describes the supervision accounting behind the root-share labels in the simplified Markov allocation plots.",
        "",
        "## Units",
        "",
        f"- Training set size in these figures: `{int(train_docs):,}` documents.",
        f"- Document length in the plotted Markov bundles: `{DOC_TOKENS}` tokens.",
        "- `leaf_tokens` determines the tree geometry: `leaf64` has 2 leaves/doc, `leaf32` has 4, `leaf16` has 8, and `leaf8` has 16.",
        "- One root label has full-document-equivalent mass `1.0`.",
        "- A node label on a span of `s` tokens has mass `s / 128`. For example, a `leaf16` label has mass `16/128 = 0.125`, so eight such leaf labels equal one root label in mass even though they are eight raw oracle calls.",
        "",
        "## Root-Only Ladder",
        "",
        "- Package `fullXX` means retain only `XX%` of the root labels and add no local-node labels.",
        f"- Example: `full50` uses `{_root_docs(50, train_docs):,}` root-labeled documents out of `{int(train_docs):,}`. Its total supervision mass is about `0.5` per training document.",
        "- These points answer: how well does the neural operator/tree do if we only have X root-level labels?",
        "",
        "## Reallocation Ladder",
        "",
        "- Reallocation packages keep the same retained root mass `XX/100`, then replace the missing mass `1 - XX/100` with sampled node labels so total mass stays at `1.0`.",
        "- `rXX_leaf_mass_eq_YYp0`: retain `XX%` root mass and put the missing `YY% = 100 - XX` mass on leaves only.",
        "- `rXX_depth_equal_mass_eq_YYp0`: retain `XX%` root mass and spread the missing mass evenly over leaf depth plus the available non-root merge depths. The runner resolves this into leaf/internal label rates for the chosen leaf geometry.",
        "- `r100_node_mass_eq_YYp0`: historical package name for balanced local-node mass. The `YY` suffix is the local mass; the effective retained root mass is `100 - YY` percent. The same label rate is applied to covered leaves and internals, then the root budget is set to the residual mass.",
        "",
        "The resolver lives in `scripts/run_markov_optimization_tradeoff_pipeline.py` (`_resolve_supervision_recovery_package_for_scope`). It first computes geometry for the requested `leaf_tokens`, then sets label rates:",
        "",
        "- Leaf-only: `leaf_rate = local_mass_target / leaf_mass_full`; internal supervision is disabled. Since the leaves partition the document, `leaf_mass_full = 1.0`, so `r50_leaf_mass_eq_50p0` uses `leaf_rate = 0.5`.",
        "- Depth-equal: let `D` be the number of eligible non-root internal merge depths. The local mass is split across `D + 1` levels: one leaf level and `D` internal levels. The leaf rate is `local_mass_target / (D + 1) / leaf_mass_full`; the internal rate is `(D * local_mass_target / (D + 1)) / internal_mass_full`.",
        "- Balanced-node: the same rate is applied to all covered leaves and internal merge nodes: `shared_rate = local_mass_target / (leaf_mass_full + internal_mass_full)`.",
        "",
        "Concrete example at `leaf16`, R50:",
        "",
        "- Leaf-only: root mass `0.5`, leaf mass `0.5`, internal mass `0.0`, `leaf_rate = 0.5`, total mass `1.0`.",
        "- Depth-equal: root mass `0.5`, leaf mass `0.1667`, internal mass `0.3333`, `leaf_rate = internal_rate = 0.1667`, total mass `1.0`.",
        "- Balanced-node: root mass `0.5`, leaf mass `0.125`, internal mass `0.375`, `leaf_rate = internal_rate = 0.125`, total mass `1.0`.",
        "",
        "## Random Sampling",
        "",
        "- Root labels are a random subset of training documents at the requested retained root mass.",
        "- Local labels are random samples from the eligible leaf/internal nodes implied by the resolved label rates.",
        "- The mass accounting is span-weighted. Raw call counts can be larger than one call/doc because small nodes are cheap in full-document-equivalent mass.",
        "- The optimization rows use the authoritative manifest path with `local_estimand_mode = span_mass_ipw_sum`, so sampled node labels represent the intended span-mass population rather than just the observed subset.",
        "",
        "## Plot Interpretation",
        "",
        "- The root-only plot compares `fullXX` across leaf geometries. Moving from R100 to R10 removes root-level labels and also reduces total supervision mass.",
        "- The reallocation plot compares two different questions at the same root share: green is the lower-budget root-only model; blue is the best available fixed-mass node-reallocated tree at that leaf geometry.",
        "- The blue line is an ex post envelope over `leaf_only`, `depth_equal`, and `balanced_node`; the companion CSV records which policy supplied each point.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_report(path: Path) -> None:
    lines = [
        "# Markov Root Data and Node Reallocation Views",
        "",
        "Additional simplified views built from the sticky allocation summary.",
        "",
        "## Figures",
        "",
    ]
    for name in [
        "simple_root_only_no_by_root_data.png",
        "hard_root_only_no_by_root_data.png",
        "simple_root_to_node_reallocation_by_leaf.png",
        "hard_root_to_node_reallocation_by_leaf.png",
    ]:
        lines.append(f"- [{name}]({name})")
    lines.extend(
        [
            "",
            "## Data",
            "",
            "- [root_only_no_by_root_data.csv](root_only_no_by_root_data.csv)",
            "- [root_to_node_reallocation_points.csv](root_to_node_reallocation_points.csv)",
            "- [reallocation_mechanics.md](reallocation_mechanics.md)",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--train-docs", type=int, default=TRAIN_DOCS)
    args = parser.parse_args()

    summary = _load_json(args.summary)
    scopes = dict(summary.get("scopes") or {})

    root_rows: list[dict[str, Any]] = []
    realloc_rows: list[dict[str, Any]] = []
    for scope_key in SCOPE_ORDER:
        scope_data = scopes.get(scope_key)
        if not scope_data:
            continue
        root_rows.extend(
            _render_root_only_scaling(
                scope_key=scope_key,
                scope_data=dict(scope_data),
                output_dir=args.output_dir,
                train_docs=int(args.train_docs),
            )
        )
        realloc_rows.extend(
            _render_root_to_node_reallocation(
                scope_key=scope_key,
                scope_data=dict(scope_data),
                output_dir=args.output_dir,
                train_docs=int(args.train_docs),
            )
        )

    _write_csv(args.output_dir / "root_only_no_by_root_data.csv", root_rows)
    _write_csv(args.output_dir / "root_to_node_reallocation_points.csv", realloc_rows)
    _write_mechanics_note(args.output_dir / "reallocation_mechanics.md", train_docs=int(args.train_docs))
    _write_report(args.output_dir / "root_node_reallocation_report.md")
    print(f"wrote {args.output_dir / 'root_only_no_by_root_data.csv'}")
    print(f"wrote {args.output_dir / 'root_to_node_reallocation_points.csv'}")
    print(f"wrote {args.output_dir / 'reallocation_mechanics.md'}")
    print(f"wrote {args.output_dir / 'root_node_reallocation_report.md'}")


if __name__ == "__main__":
    main()
