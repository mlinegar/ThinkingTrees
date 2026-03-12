#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.utility_transport_expectations import UtilityTransportRow  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot exact utility transport suite.")
    p.add_argument("--summary-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def _rows(path: Path) -> List[UtilityTransportRow]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [UtilityTransportRow(**row) for row in payload.get("rows", [])]


def _finite_rows(rows: Iterable[UtilityTransportRow], attr: str) -> List[UtilityTransportRow]:
    out: List[UtilityTransportRow] = []
    for row in rows:
        value = float(getattr(row, attr))
        if math.isfinite(value):
            out.append(row)
    return out


def _median(values: Sequence[float]) -> float:
    arr = [float(v) for v in values if math.isfinite(float(v))]
    if not arr:
        return float("nan")
    return float(np.median(np.asarray(arr, dtype=np.float64)))


def _curve(rows: Iterable[UtilityTransportRow], *, x_attr: str, y_attr: str) -> tuple[np.ndarray, np.ndarray]:
    buckets: dict[float, list[float]] = {}
    for row in rows:
        x = float(getattr(row, x_attr))
        y = float(getattr(row, y_attr))
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        buckets.setdefault(x, []).append(y)
    xs = np.asarray(sorted(buckets.keys()), dtype=np.float64)
    ys = np.asarray([_median(buckets[x]) for x in xs], dtype=np.float64)
    return xs, ys


def _full_local_rows(rows: Iterable[UtilityTransportRow]) -> List[UtilityTransportRow]:
    out: List[UtilityTransportRow] = []
    for row in rows:
        if row.leaf_label_coverage > 0.0 and abs(row.leaf_label_coverage - row.internal_label_coverage) <= 1e-9:
            out.append(row)
    return out


def _internal_only_rows(rows: Iterable[UtilityTransportRow]) -> List[UtilityTransportRow]:
    out: List[UtilityTransportRow] = []
    for row in rows:
        if abs(row.leaf_label_coverage) <= 1e-9 and row.internal_label_coverage > 0.0:
            out.append(row)
    return out


def _select_max_train_docs(rows: Iterable[UtilityTransportRow]) -> List[UtilityTransportRow]:
    rows = list(rows)
    if not rows:
        return []
    best = max(int(r.train_docs) for r in rows)
    return [r for r in rows if int(r.train_docs) == best]


def _select_max_docs_and_canonical_leaf(rows: Iterable[UtilityTransportRow], *, canonical_leaf: float) -> List[UtilityTransportRow]:
    rows = _select_max_train_docs(list(rows))
    if not rows:
        return []
    exact = [r for r in rows if abs(float(r.leaves_per_doc) - float(canonical_leaf)) <= 1e-9]
    if exact:
        return exact
    distinct = sorted({float(r.leaves_per_doc) for r in rows if math.isfinite(float(r.leaves_per_doc))})
    if not distinct:
        return rows
    nearest = min(distinct, key=lambda x: abs(x - float(canonical_leaf)))
    return [r for r in rows if abs(float(r.leaves_per_doc) - nearest) <= 1e-9]


def _plot_markov_local(ax: plt.Axes, rows: Sequence[UtilityTransportRow], *, metric: str) -> None:
    lane_rows = [
        r
        for r in rows
        if r.lane == "markov"
        and r.oracle_profile == "markov_count_endpoints"
        and r.slice_name == "support_curves_local"
        and r.objective_family == "supervised_state"
        and r.structural_arm == "tree_neural_supported"
    ]
    lane_rows = _select_max_train_docs(_full_local_rows(lane_rows))
    if not lane_rows:
        ax.axis("off")
        return
    for leaf in sorted({int(r.fixed_leaf_tokens) for r in lane_rows}):
        subset = [r for r in lane_rows if int(r.fixed_leaf_tokens) == leaf]
        xs, ys = _curve(subset, x_attr="local_oracle_coverage", y_attr=metric)
        if len(xs) == 0:
            continue
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=f"leaf={leaf}")
    ax.set_title(f"Markov endpoints: {metric.replace('_', ' ')}")
    ax.set_xlabel("local oracle coverage")
    ax.set_ylabel(metric.replace("_", " "))
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=8)


def _plot_boundary_local(ax: plt.Axes, rows: Sequence[UtilityTransportRow]) -> None:
    lane_rows = [
        r
        for r in rows
        if r.lane == "boundary_topic"
        and r.oracle_profile == "topic_plus_boundary"
        and r.slice_name == "support_curves_local"
        and r.objective_family == "supervised_state"
        and r.structural_arm == "tree_neural_supported"
    ]
    lane_rows = _select_max_train_docs(_full_local_rows(lane_rows))
    if not lane_rows:
        ax.axis("off")
        return
    for leaves in sorted({int(round(r.leaves_per_doc)) for r in lane_rows}):
        subset = [r for r in lane_rows if int(round(r.leaves_per_doc)) == leaves]
        xs, ys = _curve(subset, x_attr="local_oracle_coverage", y_attr="utility_regret")
        if len(xs) == 0:
            continue
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=f"leaves={leaves}")
    ax.set_title("Boundary-topic: utility regret")
    ax.set_xlabel("local oracle coverage")
    ax.set_ylabel("utility regret")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=8)


def _plot_nonseparable_structural(ax: plt.Axes, rows: Sequence[UtilityTransportRow]) -> None:
    lane_rows = [
        r
        for r in rows
        if r.lane == "nonseparable"
        and r.oracle_profile in {"dgp1_complementarity_and", "dgp2_boundary_interaction"}
        and r.slice_name == "structural_matrix"
        and r.objective_family == "supervised_state"
    ]
    lane_rows = _select_max_docs_and_canonical_leaf(lane_rows, canonical_leaf=4.0)
    arms = [
        "oracle_exact",
        "tree_exact_supported",
        "tree_neural_supported",
        "tree_undersupported",
        "flat_equal_info",
        "flat_span_equal_info",
        "one_leaf_control",
        "right_rule_wrong_chunker",
    ]
    vals = [
        _median([r.utility_regret for r in lane_rows if r.structural_arm == arm])
        for arm in arms
    ]
    if not any(math.isfinite(v) for v in vals):
        ax.axis("off")
        return
    x = np.arange(len(arms), dtype=float)
    ax.bar(x, vals, color=["#2e7d32", "#66bb6a", "#1f77b4", "#ff7043", "#8d99ae", "#457b9d", "#9467bd", "#c62828"])
    ax.set_title("Nonseparable structural anchors")
    ax.set_ylabel("utility regret")
    ax.set_xticks(x)
    ax.set_xticklabels([arm.replace("_", "\n") for arm in arms], fontsize=8)
    ax.grid(axis="y", alpha=0.25)


def _plot_objective_comparison(axes: Sequence[plt.Axes], rows: Sequence[UtilityTransportRow]) -> None:
    specs = [
        ("markov", "markov_count_endpoints", "tree_neural_supported", "objective_family_high_support", 8.0, "Markov endpoints"),
        ("boundary_topic", "topic_plus_boundary", "tree_neural_supported", "objective_family_high_support", 6.0, "Boundary-topic"),
        ("nonseparable", "dgp2_boundary_interaction", "tree_neural_supported", "structural_matrix", 4.0, "Nonseparable"),
    ]
    objective_order = [
        "supervised_state",
        "supervised_root",
        "dpo",
        "grpo",
        "ppo",
        "hybrid_supervised_plus_dpo",
        "hybrid_supervised_plus_grpo",
        "hybrid_supervised_plus_ppo",
    ]
    for ax, (lane, profile, arm, slice_name, canonical_leaf, title) in zip(axes, specs):
        lane_rows = [r for r in rows if r.lane == lane and r.oracle_profile == profile and r.structural_arm == arm and r.slice_name == slice_name]
        lane_rows = _select_max_docs_and_canonical_leaf(lane_rows, canonical_leaf=canonical_leaf)
        vals = [_median([r.utility_regret for r in lane_rows if r.objective_family == obj]) for obj in objective_order]
        if not any(math.isfinite(v) for v in vals):
            ax.axis("off")
            continue
        x = np.arange(len(objective_order), dtype=float)
        ax.bar(x, vals, color="#1f77b4")
        ax.set_title(title)
        ax.set_ylabel("utility regret")
        ax.set_xticks(x)
        ax.set_xticklabels([obj.replace("hybrid_supervised_plus_", "hybrid+\n").replace("_", "\n") for obj in objective_order], fontsize=7)
        ax.grid(axis="y", alpha=0.25)


def _plot_preference_curves(axes: Sequence[plt.Axes], rows: Sequence[UtilityTransportRow]) -> None:
    specs = [
        ("dpo", "pairwise_prefs_per_doc", "pairwise prefs / doc"),
        ("grpo", "group_pref_groups_per_doc", "GRPO groups / doc"),
        ("ppo", "ppo_rollouts_per_doc", "PPO rollouts / doc"),
    ]
    lane_rows = [
        r
        for r in rows
        if r.lane == "markov"
        and r.oracle_profile == "markov_count_endpoints"
        and r.structural_arm == "tree_neural_supported"
        and r.slice_name in {"support_curves_preferences", "objective_family_high_support"}
    ]
    lane_rows = _select_max_docs_and_canonical_leaf(lane_rows, canonical_leaf=8.0)
    for ax, (objective_family, x_attr, xlabel) in zip(axes, specs):
        subset = [r for r in lane_rows if r.objective_family == objective_family]
        xs, ys = _curve(subset, x_attr=x_attr, y_attr="utility_regret")
        if len(xs) == 0:
            ax.axis("off")
            continue
        ax.plot(xs, ys, marker="o", linewidth=2.0, color="#1f77b4")
        ax.set_title(f"Markov endpoints: {objective_family.upper()}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("utility regret")
        ax.grid(alpha=0.25)


def _plot_structural_controls(axes: Sequence[plt.Axes], rows: Sequence[UtilityTransportRow]) -> None:
    specs = [
        ("markov", "markov_count_endpoints", "structural_controls_anchor", 8.0, "Markov endpoints"),
        ("boundary_topic", "topic_plus_boundary", "structural_controls_anchor", 6.0, "Boundary-topic"),
        ("nonseparable", "dgp2_boundary_interaction", "structural_matrix", 4.0, "Nonseparable"),
    ]
    arms = [
        "oracle_exact",
        "tree_exact_supported",
        "tree_neural_supported",
        "tree_undersupported",
        "flat_equal_info",
        "flat_span_equal_info",
        "one_leaf_control",
        "right_rule_wrong_chunker",
    ]
    for ax, (lane, profile, slice_name, canonical_leaf, title) in zip(axes, specs):
        lane_rows = [
            r
            for r in rows
            if r.lane == lane
            and r.oracle_profile == profile
            and r.slice_name == slice_name
            and r.objective_family == "supervised_state"
        ]
        lane_rows = _select_max_docs_and_canonical_leaf(lane_rows, canonical_leaf=canonical_leaf)
        vals = [_median([r.utility_regret for r in lane_rows if r.structural_arm == arm]) for arm in arms]
        if not any(math.isfinite(v) for v in vals):
            ax.axis("off")
            continue
        x = np.arange(len(arms), dtype=float)
        ax.bar(x, vals, color="#4c78a8")
        ax.set_title(title)
        ax.set_ylabel("utility regret")
        ax.set_xticks(x)
        ax.set_xticklabels([arm.replace("_", "\n") for arm in arms], fontsize=7)
        ax.grid(axis="y", alpha=0.25)


def _plot_targeted_nonseparable_ppo(fig: plt.Figure, rows: Sequence[UtilityTransportRow]) -> bool:
    lane_rows = [
        r
        for r in rows
        if r.lane == "nonseparable"
        and r.oracle_profile == "dgp2_boundary_interaction"
        and r.slice_name == "structural_matrix"
        and r.objective_family in {"ppo", "hybrid_supervised_plus_ppo"}
        and r.structural_arm in {
            "tree_neural_supported",
            "flat_equal_info",
            "flat_span_equal_info",
            "tree_undersupported",
            "one_leaf_control",
        }
    ]
    train_docs_vals = sorted({int(r.train_docs) for r in lane_rows})
    leaves_vals = sorted({int(round(r.leaves_per_doc)) for r in lane_rows})
    if not lane_rows or not train_docs_vals or not leaves_vals:
        return False
    axes = fig.subplots(len(train_docs_vals), len(leaves_vals), squeeze=False)
    style_map = {
        "tree_neural_supported": ("#1f77b4", "-"),
        "flat_equal_info": ("#8d99ae", "--"),
        "flat_span_equal_info": ("#457b9d", "-."),
        "tree_undersupported": ("#ff7043", ":"),
        "one_leaf_control": ("#9467bd", "-"),
    }
    for row_idx, train_docs in enumerate(train_docs_vals):
        for col_idx, leaves in enumerate(leaves_vals):
            ax = axes[row_idx][col_idx]
            panel_rows = [
                r
                for r in lane_rows
                if int(r.train_docs) == train_docs and int(round(r.leaves_per_doc)) == leaves
            ]
            if not panel_rows:
                ax.axis("off")
                continue
            for objective_family in ("ppo", "hybrid_supervised_plus_ppo"):
                for arm, (color, line_style) in style_map.items():
                    subset = [r for r in panel_rows if r.objective_family == objective_family and r.structural_arm == arm]
                    xs, ys = _curve(subset, x_attr="ppo_rollouts_per_doc", y_attr="utility_regret")
                    if len(xs) == 0:
                        continue
                    label = f"{objective_family}:{arm}" if row_idx == 0 and col_idx == 0 else None
                    ax.plot(xs, ys, marker="o", linewidth=1.8, linestyle=line_style, color=color, alpha=0.9, label=label)
            ax.set_title(f"train={train_docs}, leaves={leaves}")
            ax.set_xlabel("ppo rollouts / doc")
            ax.set_ylabel("utility regret")
            ax.grid(alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncols=2, frameon=False, fontsize=8)
    fig.suptitle("Nonseparable PPO repair grid", fontsize=14)
    return True


def _plot_targeted_nonseparable_objectives(fig: plt.Figure, rows: Sequence[UtilityTransportRow]) -> bool:
    lane_rows = [
        r
        for r in rows
        if r.lane == "nonseparable"
        and r.oracle_profile == "dgp2_boundary_interaction"
        and r.slice_name == "objective_family_high_support"
        and r.structural_arm in {"tree_neural_supported", "flat_equal_info", "flat_span_equal_info", "tree_undersupported"}
        and r.objective_family in {"dpo", "grpo", "ppo", "hybrid_supervised_plus_ppo"}
    ]
    if not lane_rows:
        return False
    ax = fig.subplots(1, 1)
    objective_order = ["dpo", "grpo", "ppo", "hybrid_supervised_plus_ppo"]
    arm_order = ["tree_neural_supported", "flat_equal_info", "flat_span_equal_info", "tree_undersupported"]
    width = 0.18
    x = np.arange(len(objective_order), dtype=float)
    colors = {
        "tree_neural_supported": "#1f77b4",
        "flat_equal_info": "#8d99ae",
        "flat_span_equal_info": "#457b9d",
        "tree_undersupported": "#ff7043",
    }
    for arm_idx, arm in enumerate(arm_order):
        vals = [_median([r.utility_regret for r in lane_rows if r.objective_family == obj and r.structural_arm == arm]) for obj in objective_order]
        ax.bar(x + (arm_idx - 1.5) * width, vals, width=width, label=arm, color=colors[arm])
    ax.set_title("Nonseparable high-support objective anchors")
    ax.set_ylabel("utility regret")
    ax.set_xticks(x)
    ax.set_xticklabels([obj.replace("hybrid_supervised_plus_", "hybrid+\n") for obj in objective_order])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    return True


def _plot_targeted_nonseparable_fairness(fig: plt.Figure, rows: Sequence[UtilityTransportRow]) -> bool:
    lane_rows = [
        r
        for r in rows
        if r.lane == "nonseparable"
        and r.oracle_profile == "dgp2_boundary_interaction"
        and r.slice_name in {"structural_matrix", "objective_family_high_support"}
        and r.objective_family == "hybrid_supervised_plus_ppo"
        and r.structural_arm in {"flat_equal_info", "flat_span_equal_info"}
    ]
    if not lane_rows:
        return False
    ax = fig.subplots(1, 1)
    labels = []
    vals = []
    for arm in ("flat_equal_info", "flat_span_equal_info"):
        subset = [r for r in lane_rows if r.structural_arm == arm]
        labels.append(arm)
        vals.append(_median([r.utility_regret for r in subset]))
    ax.bar(np.arange(len(labels), dtype=float), vals, color=["#8d99ae", "#457b9d"])
    ax.set_title("Local-support fairness: flat root-only vs flat span-matched")
    ax.set_ylabel("utility regret")
    ax.set_xticks(np.arange(len(labels), dtype=float))
    ax.set_xticklabels([x.replace("_", "\n") for x in labels])
    ax.grid(axis="y", alpha=0.25)
    return True


def main() -> int:
    args = parse_args()
    rows = _rows(args.summary_json)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    _plot_markov_local(axes[0, 0], rows, metric="root_mae")
    _plot_markov_local(axes[0, 1], rows, metric="merge_mae")
    _plot_boundary_local(axes[1, 0], rows)
    _plot_nonseparable_structural(axes[1, 1], rows)
    fig.suptitle("Exact utility transport overview", fontsize=14)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"wrote_figure | {args.output}")

    objective_path = args.output.with_name(f"{args.output.stem}_objective_comparison.png")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    _plot_objective_comparison(list(axes), rows)
    fig.savefig(objective_path, dpi=180)
    plt.close(fig)
    print(f"wrote_figure | {objective_path}")

    preference_path = args.output.with_name(f"{args.output.stem}_preference_curves.png")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    _plot_preference_curves(list(axes), rows)
    fig.savefig(preference_path, dpi=180)
    plt.close(fig)
    print(f"wrote_figure | {preference_path}")

    structural_path = args.output.with_name(f"{args.output.stem}_structural_controls.png")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    _plot_structural_controls(list(axes), rows)
    fig.savefig(structural_path, dpi=180)
    plt.close(fig)
    print(f"wrote_figure | {structural_path}")

    targeted_path = args.output.with_name(f"{args.output.stem}_targeted_nonseparable_ppo.png")
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    if _plot_targeted_nonseparable_ppo(fig, rows):
        fig.savefig(targeted_path, dpi=180)
        print(f"wrote_figure | {targeted_path}")
    plt.close(fig)

    targeted_obj_path = args.output.with_name(f"{args.output.stem}_targeted_nonseparable_objectives.png")
    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    if _plot_targeted_nonseparable_objectives(fig, rows):
        fig.savefig(targeted_obj_path, dpi=180)
        print(f"wrote_figure | {targeted_obj_path}")
    plt.close(fig)

    fairness_path = args.output.with_name(f"{args.output.stem}_targeted_nonseparable_fairness.png")
    fig = plt.figure(figsize=(8, 5), constrained_layout=True)
    if _plot_targeted_nonseparable_fairness(fig, rows):
        fig.savefig(fairness_path, dpi=180)
        print(f"wrote_figure | {fairness_path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
