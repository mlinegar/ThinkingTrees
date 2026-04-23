#!/usr/bin/env python3
"""Generate alternate figure variants for the allocation-policy replacement view.

These are test versions with simpler, more direct language. They do NOT
overwrite the canonical figures — they write to a separate output directory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.render_markov_sticky_allocation_policy_grid import (
    ROOT_ONLY_FAMILY, LEAF_ONLY_FAMILY, DEPTH_EQUAL_FAMILY, BALANCED_NODE_FAMILY,
    REPLACEMENT_LEAF_TOKENS,
    RECOVERABLE_SCOPE_KEY, STRUCTURAL_SCOPE_KEY,
    _build_replacement_view, _build_allocation_coverage_summary,
)
from scripts.render_markov_sticky_simple_fixed10240_current import (
    _build_current_supervision_recovery_summary,
)

SCOPE_TITLES = {
    "recoverable_v5_t128": "Counting Topic Changes (Simple Case)",
    "structural_core_v2_t128::r12_p079": "Counting Topic Changes (Harder Case)",
    "r12_p079": "Counting Topic Changes (Harder Case)",
}

# Map scope keys used in allocation grid to scope keys used in sticky bundle summary
STICKY_SCOPE_MAP = {
    "recoverable_v5_t128": "recoverable_v5_t128",
    "r12_p079": "r12_p079",
}

STYLE_B = {
    ROOT_ONLY_FAMILY: {
        "label": "Root labels only",
        "color": "#2e7d32",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2.2,
    },
    LEAF_ONLY_FAMILY: {
        "label": "Replace missing root labels\nwith leaf labels",
        "color": "#1f77b4",
        "linestyle": "--",
        "marker": "s",
        "linewidth": 1.8,
    },
    DEPTH_EQUAL_FAMILY: {
        "label": "Replace missing root labels\nwith labels at all depths",
        "color": "#7b1fa2",
        "linestyle": "--",
        "marker": "D",
        "linewidth": 1.8,
    },
    BALANCED_NODE_FAMILY: {
        "label": "Replace missing root labels\nwith labels at all nodes",
        "color": "#c2185b",
        "linestyle": "--",
        "marker": "^",
        "linewidth": 1.8,
    },
    "fno": {
        "label": "Flat FNO (sees full document)",
        "color": "#d18f00",
        "linestyle": ":",
        "marker": None,
        "linewidth": 1.8,
    },
}

# Diagnostic marker styles
DIAG_STYLES = {
    "parity_canary": {
        "label": "1-leaf tree, no local laws\n(should match FNO)",
        "color": "white",
        "edgecolor": "#333333",
        "marker": "D",
        "size": 8,
        "zorder": 10,
    },
    "fno_point": {
        "label": "FNO actual point",
        "color": "#d32f2f",
        "edgecolor": "#d32f2f",
        "marker": "x",
        "size": 9,
        "zorder": 10,
    },
    "richer_local": {
        "label": "1-leaf tree, richer local targets\n(ceiling diagnostic)",
        "color": "#666666",
        "edgecolor": "#666666",
        "marker": "^",
        "size": 7,
        "zorder": 9,
    },
    "empirical_bayes": {
        "label": "Empirical Bayes limit\n(DGP known)",
        "color": "#006064",
        "edgecolor": "#006064",
        "marker": None,
        "linestyle": "--",
        "linewidth": 1.2,
    },
}


def _load_sticky_diagnostics(sticky_summary_path: Path) -> dict:
    """Load diagnostic marker data from the sticky bundle summary."""
    if not sticky_summary_path.exists():
        return {}
    with open(sticky_summary_path) as f:
        d = json.load(f)
    out = {}
    for scope_key, scope_data in d.get("scopes", {}).items():
        panels = {}
        for panel in scope_data.get("panel_summaries", []):
            rs = int(panel.get("root_share", 0))
            panels[rs] = {
                "fno_actual": panel.get("official_fno_actual_root_mae"),
                "canary_128": (panel.get("one_leaf_canary_root_mae_by_leaf_tokens") or {}).get("128"),
                "richer_local": panel.get("one_leaf_duplicate_local_label_root_mae"),
                "empirical_bayes": panel.get("empirical_bayes_root_mae"),
            }
        out[scope_key] = panels
    return out


def _render_variant(
    *,
    replacement_view: Mapping[str, Any],
    title: str,
    output_path: Path,
    style: dict,
    diagnostics: dict | None = None,
    panels_to_show: Sequence[int] | None = None,
    panel_title_fn=None,
    train_doc_count: int = 10240,
    subtitle: str = "",
):
    panels = list(replacement_view.get("panels") or [])
    if panels_to_show is not None:
        panels = [p for p in panels if int(p.get("root_share", 0)) in panels_to_show]

    n_panels = len(panels)
    if n_panels <= 3:
        nrows, ncols = 1, n_panels
        fig_w, fig_h = 5.5 * ncols, 5.5
    elif n_panels <= 5:
        nrows, ncols = 1, n_panels
        fig_w, fig_h = 3.5 * ncols, 5.5
    else:
        nrows, ncols = 2, 5
        fig_w, fig_h = 16.0, 10.0

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharey=False)
    if n_panels == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes.flatten()) if hasattr(axes, 'flatten') else list(axes)

    x_tokens = [int(v) for v in REPLACEMENT_LEAF_TOKENS]
    x_positions = list(range(len(x_tokens)))
    x_labels = [f"{t}\n({128 // t})" for t in x_tokens]
    leaf128_x = x_positions[x_tokens.index(128)] if 128 in x_tokens else None

    families_with_data = set()
    has_diagnostics = False
    for panel in panels:
        for fk in (ROOT_ONLY_FAMILY, LEAF_ONLY_FAMILY, DEPTH_EQUAL_FAMILY, BALANCED_NODE_FAMILY):
            if list(panel.get("series", {}).get(fk) or []):
                families_with_data.add(fk)
        if panel.get("fno_root_mae") is not None:
            families_with_data.add("fno")

    for idx, panel in enumerate(panels):
        ax = axes_list[idx]
        rs = int(panel.get("root_share", 0))
        if panel_title_fn:
            ax.set_title(panel_title_fn(rs, train_doc_count), fontsize=10, fontweight='bold')
        else:
            ax.set_title(f"R{rs}", fontsize=11, fontweight='bold')

        series_map = dict(panel.get("series") or {})
        for fk in (ROOT_ONLY_FAMILY, LEAF_ONLY_FAMILY, DEPTH_EQUAL_FAMILY, BALANCED_NODE_FAMILY):
            points = list(series_map.get(fk) or [])
            if not points:
                continue
            s = style[fk]
            xs, ys = [], []
            for p in points:
                lt = int(p.get("leaf_tokens", 0))
                if lt not in x_tokens:
                    continue
                xs.append(x_positions[x_tokens.index(lt)])
                ys.append(float(p["root_mae"]))
            if xs:
                ax.plot(xs, ys,
                        color=s["color"], linestyle=s["linestyle"],
                        marker=s["marker"], linewidth=s.get("linewidth", 1.8),
                        markersize=5)

        # FNO horizontal line
        if panel.get("fno_root_mae") is not None:
            s = style["fno"]
            y = float(panel["fno_root_mae"])
            ax.plot(x_positions, [y] * len(x_positions),
                    color=s["color"], linestyle=s["linestyle"],
                    linewidth=s.get("linewidth", 1.6))

        # Diagnostic markers
        if diagnostics and rs in diagnostics and leaf128_x is not None:
            diag = diagnostics[rs]

            # FNO actual point (red X at leaf128)
            fno_val = diag.get("fno_actual")
            if fno_val is not None:
                ds = DIAG_STYLES["fno_point"]
                ax.scatter([leaf128_x], [float(fno_val)],
                           color=ds["color"], marker=ds["marker"],
                           s=ds["size"]**2, zorder=ds["zorder"], linewidths=2)
                has_diagnostics = True

            # Parity canary (hollow diamond at leaf128)
            canary_val = diag.get("canary_128")
            if canary_val is not None:
                ds = DIAG_STYLES["parity_canary"]
                ax.scatter([leaf128_x], [float(canary_val)],
                           facecolors=ds["color"], edgecolors=ds["edgecolor"],
                           marker=ds["marker"], s=ds["size"]**2,
                           zorder=ds["zorder"], linewidths=1.5)
                has_diagnostics = True

            # Richer local labels (gray triangle at leaf128)
            richer_val = diag.get("richer_local")
            if richer_val is not None:
                ds = DIAG_STYLES["richer_local"]
                ax.scatter([leaf128_x], [float(richer_val)],
                           color=ds["color"], marker=ds["marker"],
                           s=ds["size"]**2, zorder=ds["zorder"])
                has_diagnostics = True

            # Empirical Bayes (teal dashed line)
            eb_val = diag.get("empirical_bayes")
            if eb_val is not None:
                ds = DIAG_STYLES["empirical_bayes"]
                ax.axhline(float(eb_val), color=ds["color"],
                           linestyle=ds["linestyle"], linewidth=ds["linewidth"],
                           alpha=0.6)
                has_diagnostics = True

        ax.set_xticks(x_positions, x_labels, fontsize=8)
        ax.grid(alpha=0.25, linewidth=0.6)
        if idx % ncols == 0:
            ax.set_ylabel("Test MAE\n(lower is better)", fontsize=9)
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Leaf tokens\n(leaves per doc)", fontsize=9)

    # Hide unused axes
    for ax in axes_list[len(panels):]:
        ax.axis("off")

    # Legend — allocation families
    family_order = [fk for fk in (ROOT_ONLY_FAMILY, LEAF_ONLY_FAMILY, DEPTH_EQUAL_FAMILY, BALANCED_NODE_FAMILY, "fno")
                    if fk in families_with_data]
    legend_handles = [
        Line2D([0], [0],
               color=style[fk]["color"], linestyle=style[fk]["linestyle"],
               marker=style[fk].get("marker"), linewidth=style[fk].get("linewidth", 1.8),
               markersize=6, label=style[fk]["label"])
        for fk in family_order
    ]
    # Add diagnostic legend entries
    if has_diagnostics:
        for dk in ("parity_canary", "fno_point", "richer_local", "empirical_bayes"):
            ds = DIAG_STYLES[dk]
            if ds.get("marker"):
                h = Line2D([0], [0], linestyle="None",
                           marker=ds["marker"], markersize=6,
                           color=ds.get("color", ds.get("edgecolor", "gray")),
                           markerfacecolor=ds.get("color", "white"),
                           markeredgecolor=ds.get("edgecolor", "gray"),
                           markeredgewidth=1.5,
                           label=ds["label"])
            else:
                h = Line2D([0], [0],
                           color=ds["color"], linestyle=ds["linestyle"],
                           linewidth=ds["linewidth"], alpha=0.6,
                           label=ds["label"])
            legend_handles.append(h)

    ncol_legend = 3 if n_panels > 3 else min(len(legend_handles), 3)
    fig.legend(legend_handles, [h.get_label() for h in legend_handles],
               loc="lower center",
               bbox_to_anchor=(0.5, 0.005 if n_panels <= 3 else 0.005),
               ncol=ncol_legend, frameon=True, fancybox=True,
               fontsize=8, edgecolor='#cccccc',
               handletextpad=0.5, columnspacing=1.2)

    fig.suptitle(title, y=0.99, fontsize=13, fontweight='bold')
    if subtitle:
        fig.text(0.5, 0.945, subtitle, ha="center", va="top", fontsize=9, color="#555555")

    if n_panels <= 3:
        fig.tight_layout(rect=(0.0, 0.18, 1.0, 0.93))
    else:
        fig.tight_layout(rect=(0.0, 0.14, 1.0, 0.93))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {output_path}")


def _render_budget_split(
    *,
    pure_allocation_view: Mapping[str, Any],
    replacement_view: Mapping[str, Any],
    title: str,
    subtitle: str,
    output_path: Path,
    train_doc_count: int = 10240,
):
    """Render a panel figure showing MAE vs budget split for each leaf geometry."""
    panels = list(pure_allocation_view.get("panels") or [])
    # Also add leaf64 from replacement view (only leaf-only policy available there)
    leaf64_leaf_only_points = []
    for rp in list(replacement_view.get("panels") or []):
        rs = int(rp.get("root_share", 0))
        for pt in list(rp.get("series", {}).get(LEAF_ONLY_FAMILY) or []):
            if int(pt.get("leaf_tokens", 0)) == 64:
                leaf64_leaf_only_points.append({"root_share": rs, "root_mae": float(pt["root_mae"])})
    # Build a leaf64 panel and prepend
    if leaf64_leaf_only_points:
        # Get root-only ref for leaf64 from replacement R100
        ref64 = None
        for rp in list(replacement_view.get("panels") or []):
            if int(rp.get("root_share", 0)) == 100:
                for pt in list(rp.get("series", {}).get(ROOT_ONLY_FAMILY) or []):
                    if int(pt.get("leaf_tokens", 0)) == 64:
                        ref64 = {"root_share": 100, "root_mae": float(pt["root_mae"])}
        leaf64_panel = {
            "leaf_tokens": 64,
            "series": {LEAF_ONLY_FAMILY: leaf64_leaf_only_points},
            "root_only_reference": ref64,
        }
        panels = [leaf64_panel] + panels
    if not panels:
        print(f"  skipping {output_path.name} (no pure allocation data)")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 5.0), sharey=False)
    if len(panels) == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes)

    # Get FNO MAE at each root share from replacement view
    fno_by_rs = {}
    for rp in list(replacement_view.get("panels") or []):
        rs = int(rp.get("root_share", 0))
        fno_val = rp.get("fno_root_mae")
        if fno_val is not None:
            fno_by_rs[rs] = float(fno_val)

    # Get root-only MAE at each (root_share, leaf_tokens) from replacement view
    root_only_by_rs_lt = {}
    for rp in list(replacement_view.get("panels") or []):
        rs = int(rp.get("root_share", 0))
        for pt in list(rp.get("series", {}).get(ROOT_ONLY_FAMILY) or []):
            lt = int(pt.get("leaf_tokens", 0))
            root_only_by_rs_lt[(rs, lt)] = float(pt["root_mae"])

    policy_styles = {
        LEAF_ONLY_FAMILY: {"label": "Local labels on leaves only", "color": "#1f77b4", "ls": "--", "marker": "s"},
        DEPTH_EQUAL_FAMILY: {"label": "Local labels at all depths", "color": "#7b1fa2", "ls": "--", "marker": "D"},
        BALANCED_NODE_FAMILY: {"label": "Local labels at all nodes", "color": "#c2185b", "ls": "--", "marker": "^"},
    }

    has_policies = set()
    for ax, panel in zip(axes_list, panels):
        lt = int(panel.get("leaf_tokens", 0))
        n_leaves = 128 // lt
        ax.set_title(f"{lt}-token leaves ({n_leaves} per doc)", fontsize=11, fontweight="bold")

        root_shares = sorted(set(
            [100] + [int(pt["root_share"]) for fk in policy_styles for pt in panel["series"].get(fk, [])]
        ), reverse=True)

        # FNO line across root shares
        fno_xs = [rs for rs in root_shares if rs in fno_by_rs]
        fno_ys = [fno_by_rs[rs] for rs in fno_xs]
        if fno_xs:
            ax.plot(fno_xs, fno_ys, color="#d18f00", linestyle=":", linewidth=1.8,
                    label="Flat FNO" if ax == axes_list[0] else None)

        # Root-only curve (green) across root shares
        ro_xs = [rs for rs in root_shares if (rs, lt) in root_only_by_rs_lt]
        ro_ys = [root_only_by_rs_lt[(rs, lt)] for rs in ro_xs]
        if ro_xs:
            ax.plot(ro_xs, ro_ys, color="#2e7d32", linestyle="-", marker="o",
                    linewidth=2.0, markersize=5,
                    label="Root labels only\n(lower total budget)" if ax == axes_list[0] else None)

        # All-root reference star
        ref = panel.get("root_only_reference") or {}
        if ref:
            ax.scatter([int(ref["root_share"])], [float(ref["root_mae"])],
                       color="#2e7d32", marker="*", s=120, zorder=10,
                       label="All-root reference (100%)" if ax == axes_list[0] else None)

        # Allocation policy curves
        for fk, ps in policy_styles.items():
            points = list(panel["series"].get(fk) or [])
            if not points:
                continue
            has_policies.add(fk)
            xs = [int(pt["root_share"]) for pt in points]
            ys = [float(pt["root_mae"]) for pt in points]
            ax.plot(xs, ys, color=ps["color"], linestyle=ps["ls"], marker=ps["marker"],
                    linewidth=1.8, markersize=5,
                    label=ps["label"] if ax == axes_list[0] else None)

        ax.set_xlim(105, -5)
        ax.set_xticks([100, 80, 60, 40, 20, 0])
        ax.set_xticklabels(["100%", "80%", "60%", "40%", "20%", "0%"])
        ax.set_xlabel("% of token budget as root labels\n(remainder as local labels)", fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.6)
        if ax == axes_list[0]:
            ax.set_ylabel("Test MAE (lower is better)", fontsize=9)

    fig.suptitle(title, y=0.99, fontsize=13, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.945, subtitle, ha="center", va="top", fontsize=9, color="#555555")

    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=3, frameon=True, fancybox=True, fontsize=8,
               edgecolor="#cccccc", handletextpad=0.5, columnspacing=1.2)

    fig.tight_layout(rect=(0.0, 0.14, 1.0, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {output_path}")


def main():
    base = REPO_ROOT / "outputs" / "markov_v5_simple_fixed10240_quick_20260414_utc"
    out_dir = REPO_ROOT / "outputs" / "allocation_figure_variants"
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = _build_current_supervision_recovery_summary(base)

    # Load diagnostic data from the sticky bundle
    sticky_summary = REPO_ROOT / "outputs" / "markov_v5_simple_current_plots_20260415_233539" / "summary.json"
    all_diagnostics = _load_sticky_diagnostics(sticky_summary)

    # Also need pure allocation views
    from scripts.render_markov_sticky_allocation_policy_grid import _build_pure_allocation_view

    for scope_key, scope_label in [
        (RECOVERABLE_SCOPE_KEY, "simple"),
        (STRUCTURAL_SCOPE_KEY, "hard"),
    ]:
        rv = _build_replacement_view(merged, scope_key=scope_key, train_doc_count=10240)
        base_title = SCOPE_TITLES.get(scope_key, scope_key)
        sticky_key = STICKY_SCOPE_MAP.get(scope_key, scope_key)
        diag = all_diagnostics.get(sticky_key, {})

        dgp_subtitle = {
            "simple": "4 colors, 16 tokens, ~5 topic changes per 128-token document, 10,240 training documents",
            "hard": "12 colors, 48 tokens, ~10 topic changes per 128-token document, 10,240 training documents",
        }.get(scope_label, "")

        def panel_title_b(rs, tdc):
            n = int(tdc * rs / 100)
            pct = int(rs)
            if pct == 100:
                return f"All {n:,} docs labeled"
            return f"{pct}% labeled ({n:,} docs)"

        # ── Variant B: plain-English panels + diagnostics ──
        _render_variant(
            replacement_view=rv,
            title=f"{base_title}\nWhat happens when we replace root labels with local labels?",
            subtitle=dgp_subtitle,
            output_path=out_dir / f"{scope_label}_variant_b_plain_panels.png",
            style=STYLE_B,
            diagnostics=diag,
            panel_title_fn=panel_title_b,
            train_doc_count=10240,
        )

        # ── Variant C: 3-panel compact + diagnostics ──
        _render_variant(
            replacement_view=rv,
            title=f"{base_title}: full budget vs. 50% vs. 10% root labels",
            subtitle=dgp_subtitle,
            output_path=out_dir / f"{scope_label}_variant_c_three_panels.png",
            style=STYLE_B,
            diagnostics=diag,
            panels_to_show=[100, 50, 10],
            panel_title_fn=panel_title_b,
            train_doc_count=10240,
        )

        # ── Budget-split view: MAE vs % budget as root labels ──
        pv = _build_pure_allocation_view(merged, scope_key=scope_key, train_doc_count=10240)
        _render_budget_split(
            pure_allocation_view=pv,
            replacement_view=rv,
            title=f"{base_title}\nHow does splitting the token budget between root and local labels affect accuracy?",
            subtitle=dgp_subtitle + "  |  Total token budget fixed at 100% level",
            output_path=out_dir / f"{scope_label}_budget_split.png",
            train_doc_count=10240,
        )

    print(f"\nAll variants written to {out_dir}/")


if __name__ == "__main__":
    main()
