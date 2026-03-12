#!/usr/bin/env python3
"""Plot generic-k phase behavior versus sketch order gap (m-k)."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Dict, List

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tree.mergeable_ablation import (
    ChunkerPolicy,
    KSketchEstimator,
    KSketchMethodSpec,
    SelectorPolicy,
    SpikeCountMixtureDistributionSpec,
    run_k_target_recovery_study,
)
from src.ctreepo.sim.objective_semantics import mergeable_probability_target_objective_semantics


def _parse_int_csv(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def _parse_float_csv(s: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in s.split(",") if x.strip())


def _parse_str_csv(s: str) -> tuple[str, ...]:
    out = tuple(x.strip() for x in s.split(",") if x.strip())
    if len(out) == 0:
        raise ValueError("expected a non-empty string CSV")
    return out


def _bias_ci_from_summary_row(row: Dict[str, object], z: float = 1.96) -> tuple[float, float, float]:
    """
    Approximate SE/CI for the mean signed bias from summary stats.

    Uses: rmse^2 = var + bias^2 over replicate-level estimates.
    """
    n_rep = max(1, int(row["n_replicates"]))
    bias = float(row["bias"])
    rmse = float(row["rmse"])
    var = max(0.0, (rmse * rmse) - (bias * bias))
    se = math.sqrt(var / float(n_rep))
    return se, bias - z * se, bias + z * se


def _relation(m: int, k: int) -> str:
    if m < k:
        return "unsupported"
    if m == k:
        return "exact"
    return "oversupported"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot bias-vs-(m-k) with naive baselines and regime penalties."
    )
    parser.add_argument("--p-spike-doc", type=float, default=0.62)
    parser.add_argument("--p-boundary-given-spike", type=float, default=0.35)
    parser.add_argument("--spike-count-support", type=str, default="1,2,3,4,5")
    parser.add_argument("--spike-count-probs", type=str, default="0.10,0.20,0.25,0.25,0.20")
    parser.add_argument("--target-ks", type=str, default="2,3,4,5")
    parser.add_argument("--sketch-orders", type=str, default="2,3,4,5,6")
    parser.add_argument("--n-tokens", type=int, default=32)
    parser.add_argument("--proxy-noise", type=float, default=0.12)
    parser.add_argument("--boundary-span-tokens", type=int, default=4)
    parser.add_argument("--n-replicates", type=int, default=120)
    parser.add_argument("--docs-per-replicate", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--weighting-modes",
        type=str,
        default="doc,leaf,token",
        help="Comma-separated weighting modes for side-by-side reporting.",
    )
    parser.add_argument(
        "--legacy-weighting-mode",
        type=str,
        default="doc",
        choices=("doc", "leaf", "token"),
        help="Explicit label for legacy scalar fields.",
    )
    parser.add_argument(
        "--budget-values",
        type=str,
        default="1,2,3,4,5,6,8,10",
        help="Chunk-budget values to evaluate in the budget sweep panel.",
    )
    parser.add_argument(
        "--budget-target-k",
        type=int,
        default=None,
        help="Target k used for budget sweep panel (default: max target-ks).",
    )
    parser.add_argument(
        "--budget-sketch-order",
        type=int,
        default=None,
        help="Sketch order m for budget sweep panel (default: budget-target-k).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/mergeable_k_m_phase.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--json-summary",
        type=str,
        default="outputs/mergeable_k_m_phase_summary.json",
        help="Optional JSON summary path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    support = _parse_int_csv(args.spike_count_support)
    probs = _parse_float_csv(args.spike_count_probs)
    target_ks = _parse_int_csv(args.target_ks)
    sketch_orders = _parse_int_csv(args.sketch_orders)
    budget_values = _parse_int_csv(args.budget_values)
    weighting_modes = _parse_str_csv(args.weighting_modes)

    spec = SpikeCountMixtureDistributionSpec(
        p_spike_doc=args.p_spike_doc,
        p_boundary_given_spike=args.p_boundary_given_spike,
        spike_count_support=support,
        spike_count_probs_given_spike=probs,
        n_tokens=args.n_tokens,
        proxy_noise=args.proxy_noise,
        boundary_span_tokens=args.boundary_span_tokens,
    )

    methods: List[KSketchMethodSpec] = []
    for m in sketch_orders:
        methods.append(
            KSketchMethodSpec(
                name=f"one_pass_m{m}",
                description=f"one-pass oracle top-{m}",
                estimator=KSketchEstimator.MERGE_SAFE_TOPK,
                chunker=ChunkerPolicy.FIXED,
                selector=SelectorPolicy.ALL,
                sketch_order=m,
                chunk_budget=None,
                fixed_chunk_size=10**9,
            )
        )
        methods.append(
            KSketchMethodSpec(
                name=f"full_model_m{m}",
                description=f"full-model aligned top-{m}",
                estimator=KSketchEstimator.MERGE_SAFE_TOPK,
                chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
                selector=SelectorPolicy.TOP_PROXY,
                sketch_order=m,
                chunk_budget=6,
            )
        )
    methods.append(
        KSketchMethodSpec(
            name="naive_majority",
            description="naive majority baseline",
            estimator=KSketchEstimator.NAIVE_MAJORITY,
            chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
            selector=SelectorPolicy.TOP_PROXY,
            sketch_order=1,
            chunk_budget=6,
        )
    )
    methods.append(
        KSketchMethodSpec(
            name="naive_mean_of_means",
            description="naive mean-of-means baseline",
            estimator=KSketchEstimator.NAIVE_MEAN_OF_MEANS,
            chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
            selector=SelectorPolicy.TOP_PROXY,
            sketch_order=1,
            chunk_budget=6,
        )
    )

    summaries = run_k_target_recovery_study(
        distribution=spec,
        target_ks=target_ks,
        methods=methods,
        n_replicates=args.n_replicates,
        docs_per_replicate=args.docs_per_replicate,
        seed=args.seed,
        weighting_modes=weighting_modes,
        legacy_weighting_mode=args.legacy_weighting_mode,
    )
    rows = [asdict(s) for s in summaries]
    for row in rows:
        se, lo, hi = _bias_ci_from_summary_row(row)
        row["se_bias"] = se
        row["ci95_bias_low"] = lo
        row["ci95_bias_high"] = hi

    # Main phase curves: mean abs bias vs delta=(m-k) for one-pass/full-model.
    family_rows = {
        "one_pass": [r for r in rows if r["method_name"].startswith("one_pass_m")],
        "full_model": [r for r in rows if r["method_name"].startswith("full_model_m")],
    }
    delta_curve: Dict[str, Dict[int, float]] = {"one_pass": {}, "full_model": {}}
    for family, rset in family_rows.items():
        buckets: Dict[int, List[float]] = {}
        for r in rset:
            d = int(r["sketch_order"]) - int(r["target_k"])
            buckets.setdefault(d, []).append(float(r["mean_abs_bias"]))
        for d, vals in buckets.items():
            delta_curve[family][d] = sum(vals) / float(len(vals))

    naive_majority_mean = sum(
        float(r["mean_abs_bias"]) for r in rows if r["method_name"] == "naive_majority"
    ) / max(1, sum(1 for r in rows if r["method_name"] == "naive_majority"))
    naive_mom_mean = sum(
        float(r["mean_abs_bias"]) for r in rows if r["method_name"] == "naive_mean_of_means"
    ) / max(1, sum(1 for r in rows if r["method_name"] == "naive_mean_of_means"))

    # Regime penalties by k for one-pass/full-model.
    penalty = {}
    for family, rset in family_rows.items():
        penalty[family] = {}
        for k in sorted(set(target_ks)):
            exact = [float(r["mean_abs_bias"]) for r in rset if int(r["target_k"]) == k and int(r["sketch_order"]) == k]
            unsupported = [float(r["mean_abs_bias"]) for r in rset if int(r["target_k"]) == k and int(r["sketch_order"]) < k]
            oversup = [float(r["mean_abs_bias"]) for r in rset if int(r["target_k"]) == k and int(r["sketch_order"]) > k]
            exact_mean = sum(exact) / float(len(exact)) if exact else float("nan")
            unsup_mean = sum(unsupported) / float(len(unsupported)) if unsupported else float("nan")
            oversup_mean = sum(oversup) / float(len(oversup)) if oversup else float("nan")
            penalty[family][k] = {
                "unsupported_minus_exact": unsup_mean - exact_mean if unsupported else float("nan"),
                "oversupported_minus_exact": oversup_mean - exact_mean if oversup else float("nan"),
            }

    # Budget sweep: hold target k fixed, vary chunk budget.
    if len(budget_values) == 0:
        raise ValueError("budget-values must be non-empty")
    if any(int(b) < 1 for b in budget_values):
        raise ValueError("budget-values must be >= 1")
    budget_target_k = int(args.budget_target_k) if args.budget_target_k is not None else max(target_ks)
    budget_sketch_order = (
        int(args.budget_sketch_order)
        if args.budget_sketch_order is not None
        else budget_target_k
    )
    if budget_target_k < 2:
        raise ValueError("budget-target-k must be >= 2")
    if budget_sketch_order < 1:
        raise ValueError("budget-sketch-order must be >= 1")

    budget_methods: List[KSketchMethodSpec] = [
        KSketchMethodSpec(
            name="budget_one_pass_reference",
            description=f"one-pass reference top-{budget_sketch_order}",
            estimator=KSketchEstimator.MERGE_SAFE_TOPK,
            chunker=ChunkerPolicy.FIXED,
            selector=SelectorPolicy.ALL,
            sketch_order=budget_sketch_order,
            chunk_budget=None,
            fixed_chunk_size=10**9,
        )
    ]
    for b in budget_values:
        budget_methods.append(
            KSketchMethodSpec(
                name=f"budget_full_model_b{b}",
                description=f"full-model top-{budget_sketch_order} with budget={b}",
                estimator=KSketchEstimator.MERGE_SAFE_TOPK,
                chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
                selector=SelectorPolicy.TOP_PROXY,
                sketch_order=budget_sketch_order,
                chunk_budget=int(b),
            )
        )
        budget_methods.append(
            KSketchMethodSpec(
                name=f"budget_wrong_chunker_b{b}",
                description=f"wrong chunker top-{budget_sketch_order} with budget={b}",
                estimator=KSketchEstimator.MERGE_SAFE_TOPK,
                chunker=ChunkerPolicy.ADAPTIVE_MISSPECIFIED,
                selector=SelectorPolicy.BOTTOM_PROXY,
                sketch_order=budget_sketch_order,
                chunk_budget=int(b),
            )
        )

    budget_summaries = run_k_target_recovery_study(
        distribution=spec,
        target_ks=(budget_target_k,),
        methods=budget_methods,
        n_replicates=args.n_replicates,
        docs_per_replicate=args.docs_per_replicate,
        seed=args.seed + 913_337,
        weighting_modes=weighting_modes,
        legacy_weighting_mode=args.legacy_weighting_mode,
    )
    budget_rows = [asdict(s) for s in budget_summaries]
    for row in budget_rows:
        se, lo, hi = _bias_ci_from_summary_row(row)
        row["se_bias"] = se
        row["ci95_bias_low"] = lo
        row["ci95_bias_high"] = hi

    budget_index = {(r["method_name"], int(r["target_k"])): r for r in budget_rows}

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5), constrained_layout=True)

    ax = axes[0]
    for family, color, marker in (
        ("one_pass", "#1f77b4", "o"),
        ("full_model", "#2ca02c", "s"),
    ):
        xs = sorted(delta_curve[family].keys())
        ys = [delta_curve[family][x] for x in xs]
        ax.plot(xs, ys, marker=marker, color=color, label=f"{family} top-m")
    if len(delta_curve["one_pass"]) > 0:
        xmin = min(delta_curve["one_pass"].keys())
        xmax = max(delta_curve["one_pass"].keys())
    else:
        xmin, xmax = -3, 3
    ax.hlines(naive_majority_mean, xmin=xmin, xmax=xmax, colors="#d62728", linestyles="--", label="naive-majority")
    ax.hlines(naive_mom_mean, xmin=xmin, xmax=xmax, colors="#9467bd", linestyles="--", label="naive-mean-of-means")
    ax.axvline(0, color="#444444", linewidth=1, alpha=0.7)
    ax.set_xlabel("Delta = sketch order m - target k")
    ax.set_ylabel("Mean absolute bias")
    ax.set_title("Bias vs Sketch Gap (with naive baselines)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    ax2 = axes[1]
    ks = sorted(set(target_ks))
    x = list(range(len(ks)))
    width = 0.18
    one_unsup = [penalty["one_pass"][k]["unsupported_minus_exact"] for k in ks]
    one_over = [penalty["one_pass"][k]["oversupported_minus_exact"] for k in ks]
    full_unsup = [penalty["full_model"][k]["unsupported_minus_exact"] for k in ks]
    full_over = [penalty["full_model"][k]["oversupported_minus_exact"] for k in ks]
    ax2.bar([v - 1.5 * width for v in x], one_unsup, width=width, color="#1f77b4", alpha=0.9, label="one-pass: unsupported-exact")
    ax2.bar([v - 0.5 * width for v in x], one_over, width=width, color="#1f77b4", alpha=0.45, label="one-pass: oversup-exact")
    ax2.bar([v + 0.5 * width for v in x], full_unsup, width=width, color="#2ca02c", alpha=0.9, label="full-model: unsupported-exact")
    ax2.bar([v + 1.5 * width for v in x], full_over, width=width, color="#2ca02c", alpha=0.45, label="full-model: oversup-exact")
    ax2.axhline(0.0, color="#444444", linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(k) for k in ks])
    ax2.set_xlabel("Target k")
    ax2.set_ylabel("Delta abs-bias vs exact(m=k)")
    ax2.set_title("Penalty Comparison by Regime")
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(axis="y", alpha=0.2)

    ax3 = axes[2]
    full_bias = []
    full_ci = []
    wrong_bias = []
    wrong_ci = []
    full_abs = []
    wrong_abs = []
    for b in budget_values:
        full_row = budget_index[(f"budget_full_model_b{b}", budget_target_k)]
        wrong_row = budget_index[(f"budget_wrong_chunker_b{b}", budget_target_k)]
        full_bias.append(float(full_row["bias"]))
        full_ci.append(1.96 * float(full_row["se_bias"]))
        full_abs.append(float(full_row["mean_abs_bias"]))
        wrong_bias.append(float(wrong_row["bias"]))
        wrong_ci.append(1.96 * float(wrong_row["se_bias"]))
        wrong_abs.append(float(wrong_row["mean_abs_bias"]))

    one_pass_row = budget_index[("budget_one_pass_reference", budget_target_k)]
    one_pass_bias = float(one_pass_row["bias"])
    one_pass_ci = 1.96 * float(one_pass_row["se_bias"])

    x_budget = [int(b) for b in budget_values]
    ax3.errorbar(
        x_budget,
        full_bias,
        yerr=full_ci,
        color="#2ca02c",
        marker="o",
        capsize=3,
        label="full-model bias (95% CI)",
    )
    ax3.errorbar(
        x_budget,
        wrong_bias,
        yerr=wrong_ci,
        color="#ff7f0e",
        marker="s",
        capsize=3,
        label="wrong-chunker bias (95% CI)",
    )
    ax3.plot(
        x_budget,
        full_abs,
        color="#2ca02c",
        linestyle="--",
        alpha=0.6,
        label="full-model abs-bias",
    )
    ax3.plot(
        x_budget,
        wrong_abs,
        color="#ff7f0e",
        linestyle="--",
        alpha=0.6,
        label="wrong-chunker abs-bias",
    )
    ax3.hlines(
        one_pass_bias,
        xmin=min(x_budget),
        xmax=max(x_budget),
        colors="#1f77b4",
        linestyles="-",
        label="one-pass bias",
    )
    ax3.fill_between(
        x_budget,
        [one_pass_bias - one_pass_ci for _ in x_budget],
        [one_pass_bias + one_pass_ci for _ in x_budget],
        color="#1f77b4",
        alpha=0.15,
        linewidth=0,
        label="one-pass 95% CI",
    )
    ax3.axhline(0.0, color="#444444", linewidth=1, alpha=0.8)
    ax3.set_xlabel("Chunk budget")
    ax3.set_ylabel("Bias / absolute bias")
    ax3.set_title(f"Budget Sweep (k={budget_target_k}, m={budget_sketch_order})")
    ax3.legend(frameon=False, fontsize=8)
    ax3.grid(alpha=0.2)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)

    summary_path = Path(args.json_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "distribution": asdict(spec),
        "target_ks": list(target_ks),
        "sketch_orders": list(sketch_orders),
        "objective": mergeable_probability_target_objective_semantics(
            name="generic_k_recovery_target_family",
            target_ks=target_ks,
            metadata={"family": "mergeable_k_m_phase"},
        ),
        "weighting_modes": list(weighting_modes),
        "legacy_weighting_mode": str(args.legacy_weighting_mode),
        "naive_majority_mean_abs_bias": naive_majority_mean,
        "naive_mean_of_means_mean_abs_bias": naive_mom_mean,
        "delta_curve": {k: {str(d): v for d, v in dmap.items()} for k, dmap in delta_curve.items()},
        "penalty": penalty,
        "rows": rows,
        "budget_values": list(budget_values),
        "budget_target_k": budget_target_k,
        "budget_sketch_order": budget_sketch_order,
        "budget_rows": budget_rows,
        "output_figure": str(out_path),
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote_figure | {out_path}")
    print(f"wrote_summary | {summary_path}")
    print(
        f"naive_baselines | majority={naive_majority_mean:.4f} "
        f"mean_of_means={naive_mom_mean:.4f}"
    )
    print(
        "budget_trend | "
        f"full_abs_start={full_abs[0]:.4f} full_abs_end={full_abs[-1]:.4f} "
        f"full_bias_end={full_bias[-1]:+.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
