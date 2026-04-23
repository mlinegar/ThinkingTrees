#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.launch_markov_sticky_simple_fixed10240_quick import (  # noqa: E402
    ALLOCATION_MASS_PRESERVING_ROOT_SHARES,
    ALLOCATION_POLICY_RUN_KEY,
    ALLOCATION_REPLACEMENT_ROOT_SHARES,
    RECOVERABLE_SCOPE_KEY,
    STRUCTURAL_SCOPE_KEY,
    _balanced_node_mass_package,
    _depth_equal_mass_package,
    _leaf_mass_package,
    _root_only_package,
)
from scripts.plot_markov_v3_fixed_train_leaf_size_publication import (  # noqa: E402
    _scope_presentation,
)
from scripts.render_markov_sticky_simple_fixed10240_current import (  # noqa: E402
    _build_current_supervision_recovery_summary,
    _normalize_rows_by_train_docs,
)


TREE_FAMILY = "tree_neural"
FNO_FAMILY_PRIORITY = ("official_fno", "fno", "official_fno_sumlen")
REPLACEMENT_LEAF_TOKENS = (128, 64, 32, 16, 8)
PURE_ALLOCATION_LEAF_TOKENS = (32, 16, 8)
PURE_ALLOCATION_ROOT_SHARES = (100, *ALLOCATION_MASS_PRESERVING_ROOT_SHARES)

ROOT_ONLY_FAMILY = "root_only"
LEAF_ONLY_FAMILY = "leaf_only"
DEPTH_EQUAL_FAMILY = "depth_equal"
BALANCED_NODE_FAMILY = "balanced_node"

FAMILY_STYLE = {
    ROOT_ONLY_FAMILY: {
        "label": "Root-only tree",
        "color": "#2e7d32",
        "linestyle": "-",
        "marker": "o",
    },
    LEAF_ONLY_FAMILY: {
        "label": "Leaf-only same-mass",
        "color": "#1f77b4",
        "linestyle": "--",
        "marker": "s",
    },
    DEPTH_EQUAL_FAMILY: {
        "label": "Depth-equal same-mass",
        "color": "#7b1fa2",
        "linestyle": "--",
        "marker": "D",
    },
    BALANCED_NODE_FAMILY: {
        "label": "Balanced-node same-mass",
        "color": "#c2185b",
        "linestyle": "--",
        "marker": "^",
    },
    "fno": {
        "label": "Official FNO",
        "color": "#d18f00",
        "linestyle": ":",
        "marker": None,
    },
    "all_root_reference": {
        "label": "All-root reference (full100)",
        "color": "#2e7d32",
        "linestyle": "None",
        "marker": "*",
    },
}

_FULL_RE = re.compile(r"^full(\d+)$")
_LEAF_ONLY_RE = re.compile(r"^r(\d+)_leaf_mass_eq_(\d+(?:p\d+)?)$")
_DEPTH_EQUAL_RE = re.compile(r"^r(\d+)_depth_equal_mass_eq_(\d+(?:p\d+)?)$")
_NODE_RE = re.compile(r"^r100_node_mass_eq_(\d+(?:p\d+)?)$")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")


def _parse_mass_suffix(raw: str) -> float:
    return float(str(raw or "0").replace("p", "."))


def _classify_allocation_package(package_name: str) -> dict[str, Any]:
    name = str(package_name or "").strip()
    if not name:
        return {}
    match = _FULL_RE.match(name)
    if match:
        root_share = int(match.group(1))
        return {
            "family": ROOT_ONLY_FAMILY,
            "root_share": int(root_share),
            "local_mass_percent": max(0.0, 100.0 - float(root_share)),
            "package_name": name,
        }
    match = _LEAF_ONLY_RE.match(name)
    if match:
        return {
            "family": LEAF_ONLY_FAMILY,
            "root_share": int(match.group(1)),
            "local_mass_percent": _parse_mass_suffix(match.group(2)),
            "package_name": name,
        }
    match = _DEPTH_EQUAL_RE.match(name)
    if match:
        return {
            "family": DEPTH_EQUAL_FAMILY,
            "root_share": int(match.group(1)),
            "local_mass_percent": _parse_mass_suffix(match.group(2)),
            "package_name": name,
        }
    match = _NODE_RE.match(name)
    if match:
        local_mass_percent = _parse_mass_suffix(match.group(1))
        root_share = int(round(100.0 - float(local_mass_percent)))
        return {
            "family": BALANCED_NODE_FAMILY,
            "root_share": int(root_share),
            "local_mass_percent": float(local_mass_percent),
            "package_name": name,
        }
    return {}


def _fixed_leaf_tokens(row: Mapping[str, Any]) -> int:
    for key in ("fixed_leaf_tokens", "executed_fixed_leaf_tokens", "requested_fixed_leaf_tokens"):
        value = int(row.get(key, 0) or 0)
        if value > 0:
            return int(value)
    return 0


def _test_root_mae(row: Mapping[str, Any]) -> float | None:
    for key in ("test_root_mae_mean", "tree_test_root_mae", "test_root_mae"):
        value = row.get(key)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _scope_rows(
    merged_summary: Mapping[str, Any],
    *,
    scope_key: str,
    train_doc_count: int,
) -> list[dict[str, Any]]:
    recovery = dict(merged_summary.get("supervision_recovery") or {})
    scopes = dict(recovery.get("scopes") or {})
    scope_payload = dict(scopes.get(str(scope_key)) or {})
    rows_by_train_docs = _normalize_rows_by_train_docs(scope_payload)
    row_group = dict(rows_by_train_docs.get(str(int(train_doc_count))) or {})
    return [dict(row or {}) for row in list(row_group.get("rows") or [])]


def _build_tree_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for raw_row in rows:
        row = dict(raw_row or {})
        if str(row.get("baseline_family", "") or "") != TREE_FAMILY:
            continue
        package_name = str(row.get("package_name", "") or "").strip()
        leaf_tokens = _fixed_leaf_tokens(row)
        if not package_name or leaf_tokens <= 0:
            continue
        index[(package_name, leaf_tokens)] = row
    return index


def _build_fno_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    preferred: dict[tuple[str, int], tuple[int, dict[str, Any]]] = {}
    for raw_row in rows:
        row = dict(raw_row or {})
        family = str(row.get("baseline_family", "") or "").strip()
        if family not in FNO_FAMILY_PRIORITY:
            continue
        package_name = str(row.get("package_name", "") or "").strip()
        leaf_tokens = _fixed_leaf_tokens(row)
        if not package_name or leaf_tokens <= 0:
            continue
        priority = FNO_FAMILY_PRIORITY.index(family)
        key = (package_name, leaf_tokens)
        current = preferred.get(key)
        if current is None or priority < current[0]:
            preferred[key] = (priority, row)
    return {key: dict(payload) for key, (_, payload) in preferred.items()}


def _series_from_index(
    index: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    package_name: str,
    leaf_tokens: Sequence[int],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for leaf_token in leaf_tokens:
        row = index.get((str(package_name), int(leaf_token)))
        if not row:
            continue
        root_mae = _test_root_mae(row)
        if root_mae is None:
            continue
        points.append(
            {
                "leaf_tokens": int(leaf_token),
                "root_mae": float(root_mae),
                "package_name": str(package_name),
            }
        )
    return points


def _replacement_caption_text(*, train_doc_count: int) -> str:
    return (
        f"Each panel fixes the retained root-label budget at the panel's RXX share of {int(train_doc_count):,} training documents. "
        "The green curve is the root-only tree baseline, so its total supervision mass falls with the root share: `fullXX` keeps only the root-label budget and has total full-document-equivalent mass `XX/100`. "
        "The blue, purple, and rose curves keep the same root-label budget as the panel but reallocate the missing supervision mass locally so the total full-document-equivalent mass stays fixed at `1.0`, matching `full100`. "
        "Blue sends that local mass to leaves only; purple spreads it evenly across leaves and the available non-root merge depths; rose distributes it across covered leaf/internal nodes with balanced local allocation. "
        "At `leaf64` and especially `leaf128`, the internal-allocation families are not distinct because there are too few non-root internal layers, so the meaningful policy comparison is concentrated at `leaf32`, `leaf16`, and `leaf8`. "
        "The amber dotted line is the official FNO baseline at the same root-label budget as the panel."
    )


def _pure_allocation_caption_text() -> str:
    return (
        "These panels compare only fixed-total-mass policies. "
        "The colored curves all keep total full-document-equivalent supervision mass fixed at `1.0` while varying how much of that mass stays on root labels versus being reallocated locally. "
        "The green star is the `full100` all-root reference point only; the lower-root `fullXX` root-only ladder is intentionally excluded here because it lowers total supervision mass and would not be comparable. "
        "Blue is leaf-only, purple is depth-equal over leaves plus available non-root merge depths, and rose is balanced local-node mass. "
        "Only `leaf32`, `leaf16`, and `leaf8` are shown because those are the geometries where the local-allocation policies are genuinely distinct."
    )


def _build_replacement_view(
    merged_summary: Mapping[str, Any],
    *,
    scope_key: str,
    train_doc_count: int,
) -> dict[str, Any]:
    rows = _scope_rows(
        merged_summary,
        scope_key=scope_key,
        train_doc_count=train_doc_count,
    )
    tree_index = _build_tree_index(rows)
    fno_index = _build_fno_index(rows)
    panels: list[dict[str, Any]] = []
    for root_share in ALLOCATION_REPLACEMENT_ROOT_SHARES:
        root_package = _root_only_package(int(root_share))
        panel = {
            "root_share": int(root_share),
            "series": {
                ROOT_ONLY_FAMILY: _series_from_index(
                    tree_index,
                    package_name=root_package,
                    leaf_tokens=REPLACEMENT_LEAF_TOKENS,
                ),
                LEAF_ONLY_FAMILY: [],
                DEPTH_EQUAL_FAMILY: [],
                BALANCED_NODE_FAMILY: [],
            },
            "fno_root_mae": None,
        }
        fno_row = fno_index.get((root_package, 128))
        if fno_row is not None:
            root_mae = _test_root_mae(fno_row)
            if root_mae is not None:
                panel["fno_root_mae"] = float(root_mae)
        if int(root_share) < 100:
            panel["series"][LEAF_ONLY_FAMILY] = _series_from_index(
                tree_index,
                package_name=_leaf_mass_package(int(root_share)),
                leaf_tokens=REPLACEMENT_LEAF_TOKENS,
            )
            panel["series"][DEPTH_EQUAL_FAMILY] = _series_from_index(
                tree_index,
                package_name=_depth_equal_mass_package(int(root_share)),
                leaf_tokens=PURE_ALLOCATION_LEAF_TOKENS,
            )
            panel["series"][BALANCED_NODE_FAMILY] = _series_from_index(
                tree_index,
                package_name=_balanced_node_mass_package(int(root_share)),
                leaf_tokens=PURE_ALLOCATION_LEAF_TOKENS,
            )
        panels.append(panel)
    return {
        "root_shares": [int(value) for value in ALLOCATION_REPLACEMENT_ROOT_SHARES],
        "leaf_tokens": [int(value) for value in REPLACEMENT_LEAF_TOKENS],
        "panels": panels,
        "caption_text": _replacement_caption_text(train_doc_count=int(train_doc_count)),
    }


def _build_pure_allocation_view(
    merged_summary: Mapping[str, Any],
    *,
    scope_key: str,
    train_doc_count: int,
) -> dict[str, Any]:
    rows = _scope_rows(
        merged_summary,
        scope_key=scope_key,
        train_doc_count=train_doc_count,
    )
    tree_index = _build_tree_index(rows)
    leaf_payloads: list[dict[str, Any]] = []
    for leaf_token in PURE_ALLOCATION_LEAF_TOKENS:
        series_payload: dict[str, list[dict[str, Any]]] = {
            LEAF_ONLY_FAMILY: [],
            DEPTH_EQUAL_FAMILY: [],
            BALANCED_NODE_FAMILY: [],
        }
        for root_share in ALLOCATION_MASS_PRESERVING_ROOT_SHARES:
            for family_key, package_name in (
                (LEAF_ONLY_FAMILY, _leaf_mass_package(int(root_share))),
                (DEPTH_EQUAL_FAMILY, _depth_equal_mass_package(int(root_share))),
                (BALANCED_NODE_FAMILY, _balanced_node_mass_package(int(root_share))),
            ):
                row = tree_index.get((package_name, int(leaf_token)))
                if not row:
                    continue
                root_mae = _test_root_mae(row)
                if root_mae is None:
                    continue
                series_payload[family_key].append(
                    {
                        "root_share": int(root_share),
                        "root_mae": float(root_mae),
                        "package_name": str(package_name),
                    }
                )
        root_only_reference_row = tree_index.get((_root_only_package(100), int(leaf_token)))
        root_only_reference = None
        if root_only_reference_row is not None:
            root_mae = _test_root_mae(root_only_reference_row)
            if root_mae is not None:
                root_only_reference = {
                    "root_share": 100,
                    "root_mae": float(root_mae),
                    "package_name": _root_only_package(100),
                }
        leaf_payloads.append(
            {
                "leaf_tokens": int(leaf_token),
                "root_only_reference": root_only_reference,
                "series": series_payload,
            }
        )
    return {
        "root_shares": [int(value) for value in PURE_ALLOCATION_ROOT_SHARES],
        "leaf_tokens": [int(value) for value in PURE_ALLOCATION_LEAF_TOKENS],
        "panels": leaf_payloads,
        "caption_text": _pure_allocation_caption_text(),
    }


def _build_allocation_coverage_summary(
    merged_summary: Mapping[str, Any],
    *,
    train_doc_count: int,
) -> dict[str, Any]:
    recovery = dict(merged_summary.get("supervision_recovery") or {})
    scopes = dict(recovery.get("scopes") or {})
    coverage: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train_doc_count": int(train_doc_count),
        "replacement_root_shares": [int(value) for value in ALLOCATION_REPLACEMENT_ROOT_SHARES],
        "pure_allocation_root_shares": [int(value) for value in PURE_ALLOCATION_ROOT_SHARES],
        "scopes": {},
    }
    for scope_key, scope_payload in scopes.items():
        rows_by_train_docs = _normalize_rows_by_train_docs(scope_payload)
        row_group = dict(rows_by_train_docs.get(str(int(train_doc_count))) or {})
        rows = [dict(row or {}) for row in list(row_group.get("rows") or [])]
        tree_index = _build_tree_index(rows)
        fno_index = _build_fno_index(rows)
        replacement_root_shares: dict[str, Any] = {}
        for root_share in ALLOCATION_REPLACEMENT_ROOT_SHARES:
            replacement_root_shares[str(int(root_share))] = {
                "root_only_leaf_tokens": [
                    int(value)
                    for value in REPLACEMENT_LEAF_TOKENS
                    if (_root_only_package(int(root_share)), int(value)) in tree_index
                ],
                "leaf_only_leaf_tokens": [
                    int(value)
                    for value in REPLACEMENT_LEAF_TOKENS
                    if (_leaf_mass_package(int(root_share)), int(value)) in tree_index
                ]
                if int(root_share) < 100
                else [],
                "depth_equal_leaf_tokens": [
                    int(value)
                    for value in PURE_ALLOCATION_LEAF_TOKENS
                    if (_depth_equal_mass_package(int(root_share)), int(value)) in tree_index
                ]
                if int(root_share) < 100
                else [],
                "balanced_node_leaf_tokens": [
                    int(value)
                    for value in PURE_ALLOCATION_LEAF_TOKENS
                    if (_balanced_node_mass_package(int(root_share)), int(value)) in tree_index
                ]
                if int(root_share) < 100
                else [],
                "fno_leaf128_present": (_root_only_package(int(root_share)), 128) in fno_index,
            }
        pure_leaf_tokens: dict[str, Any] = {}
        for leaf_token in PURE_ALLOCATION_LEAF_TOKENS:
            pure_leaf_tokens[str(int(leaf_token))] = {
                "root_only_reference_present": (_root_only_package(100), int(leaf_token)) in tree_index,
                "leaf_only_root_shares": [
                    int(root_share)
                    for root_share in ALLOCATION_MASS_PRESERVING_ROOT_SHARES
                    if (_leaf_mass_package(int(root_share)), int(leaf_token)) in tree_index
                ],
                "depth_equal_root_shares": [
                    int(root_share)
                    for root_share in ALLOCATION_MASS_PRESERVING_ROOT_SHARES
                    if (_depth_equal_mass_package(int(root_share)), int(leaf_token)) in tree_index
                ],
                "balanced_node_root_shares": [
                    int(root_share)
                    for root_share in ALLOCATION_MASS_PRESERVING_ROOT_SHARES
                    if (_balanced_node_mass_package(int(root_share)), int(leaf_token)) in tree_index
                ],
            }
        coverage["scopes"][str(scope_key)] = {
            "scope_label": str(scope_payload.get("scope_label", "") or ""),
            "replacement_root_shares": replacement_root_shares,
            "pure_allocation_leaf_tokens": pure_leaf_tokens,
        }
    return coverage


def _plot_replacement_scope(
    *,
    scope_key: str,
    replacement_view: Mapping[str, Any],
    title: str,
    subtitle: str,
    train_doc_count: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(16.0, 9.5), sharey=False)
    axes_list = list(axes.flatten())
    x_tokens = [int(value) for value in REPLACEMENT_LEAF_TOKENS]
    x_positions = list(range(len(x_tokens)))
    x_labels = [f"{int(token)}\n({int(round(128 / int(token)))})" for token in x_tokens]
    for idx, panel in enumerate(list(replacement_view.get("panels") or [])):
        ax = axes_list[idx]
        root_share = int(panel.get("root_share", 0) or 0)
        ax.set_title(f"R{root_share}", fontsize=10)
        series_map = dict(panel.get("series") or {})
        for family_key in (
            ROOT_ONLY_FAMILY,
            LEAF_ONLY_FAMILY,
            DEPTH_EQUAL_FAMILY,
            BALANCED_NODE_FAMILY,
        ):
            points = [dict(item or {}) for item in list(series_map.get(family_key) or [])]
            if not points:
                continue
            style = dict(FAMILY_STYLE[family_key])
            xs = []
            ys = []
            for point in points:
                leaf_token = int(point.get("leaf_tokens", 0) or 0)
                if leaf_token not in x_tokens:
                    continue
                xs.append(x_positions[x_tokens.index(leaf_token)])
                ys.append(float(point["root_mae"]))
            if xs:
                ax.plot(
                    xs,
                    ys,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    linewidth=1.8,
                    markersize=5,
                )
        if panel.get("fno_root_mae") is not None:
            style = dict(FAMILY_STYLE["fno"])
            y = float(panel["fno_root_mae"])
            ax.plot(
                x_positions,
                [y] * len(x_positions),
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.6,
            )
        ax.set_xticks(x_positions, x_labels, fontsize=8)
        ax.grid(alpha=0.25, linewidth=0.6)
        if idx % 5 == 0:
            ax.set_ylabel("Root MAE")
        if idx >= 5:
            ax.set_xlabel("Leaf tokens\n(leaves/doc)")
    for ax in axes_list[len(list(replacement_view.get("panels") or [])) :]:
        ax.axis("off")

    # Determine which families have at least one data point across all panels.
    _families_with_data: set[str] = set()
    for _panel in list(replacement_view.get("panels") or []):
        for _fk in (ROOT_ONLY_FAMILY, LEAF_ONLY_FAMILY, DEPTH_EQUAL_FAMILY, BALANCED_NODE_FAMILY):
            if list(_panel.get("series", {}).get(_fk) or []):
                _families_with_data.add(_fk)
    if any(_panel.get("fno_root_mae") is not None for _panel in list(replacement_view.get("panels") or [])):
        _families_with_data.add("fno")

    _legend_family_order = [
        fk for fk in (ROOT_ONLY_FAMILY, LEAF_ONLY_FAMILY, DEPTH_EQUAL_FAMILY, BALANCED_NODE_FAMILY, "fno")
        if fk in _families_with_data
    ]
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=FAMILY_STYLE[fk]["color"],
            linestyle=FAMILY_STYLE[fk]["linestyle"],
            marker=FAMILY_STYLE[fk]["marker"],
            linewidth=1.8 if FAMILY_STYLE[fk]["linestyle"] != "None" else 0.0,
            markersize=6,
            label=FAMILY_STYLE[fk]["label"],
        )
        for fk in _legend_family_order
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.13),
        ncol=min(len(legend_handles), 3),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        f"{title}\nAllocation policy replacement view at fixed training set size ({int(train_doc_count):,} docs)",
        y=0.99,
    )
    fig.text(
        0.5,
        0.94,
        textwrap.fill(str(subtitle or ""), width=110),
        ha="center",
        va="top",
        fontsize=9,
    )
    fig.text(
        0.015,
        0.012,
        str(replacement_view.get("caption_text", "") or ""),
        ha="left",
        va="bottom",
        fontsize=8,
        wrap=True,
    )
    fig.tight_layout(rect=(0.0, 0.22, 1.0, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_pure_allocation_scope(
    *,
    scope_key: str,
    pure_allocation_view: Mapping[str, Any],
    title: str,
    subtitle: str,
    train_doc_count: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.7), sharey=False)
    axes_list = list(axes.flatten())
    for ax, panel in zip(axes_list, list(pure_allocation_view.get("panels") or [])):
        leaf_tokens = int(panel.get("leaf_tokens", 0) or 0)
        ax.set_title(f"leaf{leaf_tokens} ({int(round(128 / leaf_tokens))} leaves/doc)", fontsize=10)
        series_map = dict(panel.get("series") or {})
        for family_key in (LEAF_ONLY_FAMILY, DEPTH_EQUAL_FAMILY, BALANCED_NODE_FAMILY):
            points = [dict(item or {}) for item in list(series_map.get(family_key) or [])]
            if not points:
                continue
            style = dict(FAMILY_STYLE[family_key])
            xs = [int(point["root_share"]) for point in points]
            ys = [float(point["root_mae"]) for point in points]
            ax.plot(
                xs,
                ys,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=5,
            )
        root_ref = dict(panel.get("root_only_reference") or {})
        if root_ref:
            style = dict(FAMILY_STYLE["all_root_reference"])
            ax.plot(
                [int(root_ref["root_share"])],
                [float(root_ref["root_mae"])],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=10,
            )
        ax.set_xlim(100, 0)
        ax.set_xticks([100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0])
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_xlabel("Retained root share")
        ax.set_ylabel("Root MAE")

    # Determine which families have at least one data point across all panels.
    _pa_families_with_data: set[str] = set()
    for _pa_panel in list(pure_allocation_view.get("panels") or []):
        for _fk in (LEAF_ONLY_FAMILY, DEPTH_EQUAL_FAMILY, BALANCED_NODE_FAMILY):
            if list(_pa_panel.get("series", {}).get(_fk) or []):
                _pa_families_with_data.add(_fk)
        if _pa_panel.get("all_root_reference_mae") is not None:
            _pa_families_with_data.add("all_root_reference")

    _pa_legend_order = [
        fk for fk in ("all_root_reference", LEAF_ONLY_FAMILY, DEPTH_EQUAL_FAMILY, BALANCED_NODE_FAMILY)
        if fk in _pa_families_with_data
    ]
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=FAMILY_STYLE[fk]["color"],
            linestyle=FAMILY_STYLE[fk]["linestyle"],
            marker=FAMILY_STYLE[fk]["marker"],
            linewidth=1.8 if FAMILY_STYLE[fk]["linestyle"] != "None" else 0.0,
            markersize=6 if FAMILY_STYLE[fk]["marker"] != "*" else 10,
            label=FAMILY_STYLE[fk]["label"],
        )
        for fk in _pa_legend_order
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.12),
        ncol=min(len(legend_handles), 4),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        f"{title}\nPure allocation view at fixed total supervision mass ({int(train_doc_count):,} docs)",
        y=0.99,
    )
    fig.text(
        0.5,
        0.93,
        textwrap.fill(str(subtitle or ""), width=110),
        ha="center",
        va="top",
        fontsize=9,
    )
    fig.text(
        0.015,
        0.012,
        str(pure_allocation_view.get("caption_text", "") or ""),
        ha="left",
        va="bottom",
        fontsize=8,
        wrap=True,
    )
    fig.tight_layout(rect=(0.0, 0.22, 1.0, 0.92))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _scope_artifact_prefix(scope_key: str) -> str:
    return "recoverable" if str(scope_key) == RECOVERABLE_SCOPE_KEY else "structural"


def _build_report_markdown(
    *,
    output_root: Path,
    base_output_root: Path,
    overlay_output_roots: Sequence[Path],
    train_doc_count: int,
    figures: Mapping[str, str],
    coverage: Mapping[str, Any],
) -> str:
    lines = [
        "# Sticky Allocation-Policy Grid",
        "",
        f"- Generated: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Base output root: `{base_output_root}`",
        f"- Overlay output roots: `{', '.join(str(path) for path in overlay_output_roots) if overlay_output_roots else '(none)'}`",
        f"- Train docs: `{int(train_doc_count)}`",
        "",
        "## Figures",
    ]
    for key, path in figures.items():
        lines.append(f"- `{key}`: `{path}`")
    lines.extend(
        [
            "",
            "## Coverage",
            f"- Coverage JSON: `{output_root / 'coverage.json'}`",
            f"- Summary JSON: `{output_root / 'summary.json'}`",
            "",
            "## Existing Reference Surfaces",
            "- Current sticky root-only + leaf-only bundle: "
            "`outputs/markov_v5_simple_current_plots_20260415_233539/report.md`",
            "- Older v3 leaf-only same-mass report at 10240: "
            "`outputs/markov_v3_depth_redistribution_large_train_stable_20260411_084653/mass_preserving_leaf_only_large_train/tradeoff_report/report.md`",
            "- Older v3 depth-equal same-mass report at 10240: "
            "`outputs/markov_v3_depth_redistribution_large_train_stable_20260411_084653/mass_preserving_depth_equal_large_train/tradeoff_report/report.md`",
        ]
    )
    scope_coverage = dict(coverage.get("scopes") or {})
    for scope_key, payload in scope_coverage.items():
        lines.extend(
            [
                "",
                f"### {scope_key}",
                f"- Scope label: `{payload.get('scope_label', '')}`",
            ]
        )
        replacement_root_shares = dict(payload.get("replacement_root_shares") or {})
        for root_share in ALLOCATION_REPLACEMENT_ROOT_SHARES:
            root_payload = dict(replacement_root_shares.get(str(int(root_share))) or {})
            lines.append(
                f"- R{int(root_share)} replacement coverage: "
                f"root-only {root_payload.get('root_only_leaf_tokens', [])}, "
                f"leaf-only {root_payload.get('leaf_only_leaf_tokens', [])}, "
                f"depth-equal {root_payload.get('depth_equal_leaf_tokens', [])}, "
                f"balanced-node {root_payload.get('balanced_node_leaf_tokens', [])}, "
                f"FNO128={bool(root_payload.get('fno_leaf128_present', False))}"
            )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render sticky allocation-policy plots from the merged sticky 10240 "
            "supervision-recovery rows."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "markov_v5_simple_fixed10240_quick_20260414_utc",
    )
    parser.add_argument(
        "--overlay-output-root",
        type=Path,
        nargs="*",
        default=[],
        help="Optional overlay roots whose landed rows override duplicate task names.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / f"markov_v5_sticky_allocation_policy_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--train-doc-count", type=int, default=10240)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else (REPO_ROOT / args.output_root)
    )
    overlay_output_roots = [
        path if path.is_absolute() else (REPO_ROOT / path)
        for path in list(args.overlay_output_root or [])
    ]
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else (REPO_ROOT / args.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_summary = _build_current_supervision_recovery_summary(
        base_output_root,
        overlay_output_roots=overlay_output_roots,
    )
    coverage = _build_allocation_coverage_summary(
        merged_summary,
        train_doc_count=int(args.train_doc_count),
    )

    figures: dict[str, str] = {}
    summary_payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_output_root": str(base_output_root),
        "overlay_output_roots": [str(path) for path in overlay_output_roots],
        "train_doc_count": int(args.train_doc_count),
        "scopes": {},
    }

    for scope_key in (RECOVERABLE_SCOPE_KEY, STRUCTURAL_SCOPE_KEY):
        recovery = dict(merged_summary.get("supervision_recovery") or {})
        scope_payloads = dict(recovery.get("scopes") or {})
        if str(scope_key) not in scope_payloads:
            continue
        presentation = _scope_presentation(recovery, primary_scope_key=str(scope_key))
        title = str(presentation.get("title", scope_key) or scope_key)
        subtitle = str(presentation.get("subtitle", "") or "")
        replacement_view = _build_replacement_view(
            merged_summary,
            scope_key=str(scope_key),
            train_doc_count=int(args.train_doc_count),
        )
        pure_allocation_view = _build_pure_allocation_view(
            merged_summary,
            scope_key=str(scope_key),
            train_doc_count=int(args.train_doc_count),
        )
        prefix = _scope_artifact_prefix(str(scope_key))
        replacement_path = (
            output_dir
            / "figures"
            / f"{prefix}_allocation_replacement_train{int(args.train_doc_count)}.png"
        )
        pure_allocation_path = (
            output_dir
            / "figures"
            / f"{prefix}_allocation_pure_train{int(args.train_doc_count)}.png"
        )
        _plot_replacement_scope(
            scope_key=str(scope_key),
            replacement_view=replacement_view,
            title=title,
            subtitle=subtitle,
            train_doc_count=int(args.train_doc_count),
            output_path=replacement_path,
        )
        _plot_pure_allocation_scope(
            scope_key=str(scope_key),
            pure_allocation_view=pure_allocation_view,
            title=title,
            subtitle=subtitle,
            train_doc_count=int(args.train_doc_count),
            output_path=pure_allocation_path,
        )
        figures[f"{prefix}_replacement"] = str(replacement_path)
        figures[f"{prefix}_pure_allocation"] = str(pure_allocation_path)
        summary_payload["scopes"][str(scope_key)] = {
            "title": title,
            "subtitle": subtitle,
            "replacement_view": replacement_view,
            "pure_allocation_view": pure_allocation_view,
        }

    _write_json(output_dir / "summary.json", summary_payload)
    _write_json(output_dir / "coverage.json", coverage)
    (output_dir / "report.md").write_text(
        _build_report_markdown(
            output_root=output_dir,
            base_output_root=base_output_root,
            overlay_output_roots=overlay_output_roots,
            train_doc_count=int(args.train_doc_count),
            figures=figures,
            coverage=coverage,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary_json": str(output_dir / "summary.json"),
                "coverage_json": str(output_dir / "coverage.json"),
                "report_md": str(output_dir / "report.md"),
                "figures": figures,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
