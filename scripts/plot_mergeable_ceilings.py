#!/usr/bin/env python3
"""Plot sketch-sufficiency and budget-retention ceilings for mergeable ablations.

This figure is designed to make two "theoretical maxima" visually explicit:

1) Sketch sufficiency ceiling: if sketch order m < target k, the summary cannot
   represent the needed order statistic, and bias jumps sharply.
2) Evidence-retention ceiling: with a finite chunk budget, the best possible
   selector (oracle top-true) upper-bounds any proxy-based selector.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tree.mergeable_ablation import (  # noqa: E402
    ChunkerPolicy,
    KSketchEstimator,
    KSketchMethodSpec,
    SelectorPolicy,
    SpikeCountMixtureDistributionSpec,
    run_k_target_recovery_study,
)
from src.ctreepo.sim.objective_semantics import mergeable_probability_target_objective_semantics


def _parse_int_csv(s: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def _parse_float_csv(s: str) -> Tuple[float, ...]:
    return tuple(float(x.strip()) for x in s.split(",") if x.strip())


def _parse_str_csv(s: str) -> Tuple[str, ...]:
    out = tuple(x.strip() for x in s.split(",") if x.strip())
    if len(out) == 0:
        raise ValueError("expected a non-empty string CSV")
    return out


def _mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / float(len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot theoretical ceilings (sketch sufficiency + budget retention) for mergeable simulations."
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
        "--phase-chunk-budget",
        type=int,
        default=6,
        help="Chunk budget used for the adaptive full-model curve in the sketch-gap panel.",
    )
    parser.add_argument(
        "--budget-values",
        type=str,
        default="1,2,3,4,5,6,8,10",
        help="Chunk-budget values to evaluate in the budget-ceiling panel.",
    )
    parser.add_argument(
        "--budget-target-k",
        type=int,
        default=None,
        help="Target k used for budget panel (default: max target-ks).",
    )
    parser.add_argument(
        "--budget-sketch-order",
        type=int,
        default=None,
        help="Sketch order m used for budget panel (default: budget-target-k).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/mergeable_ceilings.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--json-summary",
        type=str,
        default="outputs/mergeable_ceilings_summary.json",
        help="JSON summary output path.",
    )
    return parser.parse_args()


def _delta_curve(rows: List[dict], method_prefix: str) -> Dict[int, float]:
    buckets: Dict[int, List[float]] = {}
    for row in rows:
        name = str(row["method_name"])
        if not name.startswith(method_prefix):
            continue
        d = int(row["sketch_order"]) - int(row["target_k"])
        buckets.setdefault(d, []).append(float(row["mean_abs_bias"]))
    return {d: _mean(vals) for d, vals in buckets.items()}


def main() -> int:
    args = parse_args()
    support = _parse_int_csv(args.spike_count_support)
    probs = _parse_float_csv(args.spike_count_probs)
    target_ks = _parse_int_csv(args.target_ks)
    sketch_orders = _parse_int_csv(args.sketch_orders)
    budget_values = sorted(_parse_int_csv(args.budget_values))
    weighting_modes = _parse_str_csv(args.weighting_modes)
    if len(budget_values) == 0:
        raise ValueError("--budget-values must be non-empty")
    if any(b < 1 for b in budget_values):
        raise ValueError("--budget-values must be >= 1")

    spec = SpikeCountMixtureDistributionSpec(
        p_spike_doc=args.p_spike_doc,
        p_boundary_given_spike=args.p_boundary_given_spike,
        spike_count_support=support,
        spike_count_probs_given_spike=probs,
        n_tokens=args.n_tokens,
        proxy_noise=args.proxy_noise,
        boundary_span_tokens=args.boundary_span_tokens,
    )

    # Panel A: sketch sufficiency ceiling (m-k phase).
    phase_methods: List[KSketchMethodSpec] = []
    for m in sketch_orders:
        phase_methods.append(
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
        phase_methods.append(
            KSketchMethodSpec(
                name=f"full_model_proxy_m{m}",
                description=f"adaptive aligned + top-proxy, top-{m}",
                estimator=KSketchEstimator.MERGE_SAFE_TOPK,
                chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
                selector=SelectorPolicy.TOP_PROXY,
                sketch_order=m,
                chunk_budget=int(args.phase_chunk_budget) if args.phase_chunk_budget > 0 else None,
            )
        )
    phase_methods.extend(
        [
            KSketchMethodSpec(
                name="naive_majority",
                description="naive majority baseline",
                estimator=KSketchEstimator.NAIVE_MAJORITY,
                chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
                selector=SelectorPolicy.TOP_PROXY,
                sketch_order=1,
                chunk_budget=int(args.phase_chunk_budget) if args.phase_chunk_budget > 0 else None,
            ),
            KSketchMethodSpec(
                name="naive_mean_of_means",
                description="naive mean-of-means baseline",
                estimator=KSketchEstimator.NAIVE_MEAN_OF_MEANS,
                chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
                selector=SelectorPolicy.TOP_PROXY,
                sketch_order=1,
                chunk_budget=int(args.phase_chunk_budget) if args.phase_chunk_budget > 0 else None,
            ),
        ]
    )

    phase_summaries = run_k_target_recovery_study(
        distribution=spec,
        target_ks=target_ks,
        methods=phase_methods,
        n_replicates=args.n_replicates,
        docs_per_replicate=args.docs_per_replicate,
        seed=args.seed,
        weighting_modes=weighting_modes,
        legacy_weighting_mode=args.legacy_weighting_mode,
    )
    phase_rows = [asdict(s) for s in phase_summaries]

    one_pass_delta = _delta_curve(phase_rows, "one_pass_m")
    full_proxy_delta = _delta_curve(phase_rows, "full_model_proxy_m")
    naive_majority_mean = _mean(
        [float(r["mean_abs_bias"]) for r in phase_rows if r["method_name"] == "naive_majority"]
    )
    naive_mom_mean = _mean(
        [float(r["mean_abs_bias"]) for r in phase_rows if r["method_name"] == "naive_mean_of_means"]
    )

    # Panel B: evidence-retention ceiling (budget sweep) with oracle selector bound.
    budget_target_k = int(args.budget_target_k) if args.budget_target_k is not None else max(target_ks)
    budget_sketch_order = (
        int(args.budget_sketch_order)
        if args.budget_sketch_order is not None
        else budget_target_k
    )
    if budget_target_k < 2:
        raise ValueError("--budget-target-k must be >= 2")
    if budget_sketch_order < 1:
        raise ValueError("--budget-sketch-order must be >= 1")

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
                name=f"budget_aligned_top_proxy_b{b}",
                description=f"aligned + top-proxy, budget={b}",
                estimator=KSketchEstimator.MERGE_SAFE_TOPK,
                chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
                selector=SelectorPolicy.TOP_PROXY,
                sketch_order=budget_sketch_order,
                chunk_budget=int(b),
            )
        )
        budget_methods.append(
            KSketchMethodSpec(
                name=f"budget_aligned_oracle_b{b}",
                description=f"aligned + oracle(top-true), budget={b}",
                estimator=KSketchEstimator.MERGE_SAFE_TOPK,
                chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
                selector=SelectorPolicy.TOP_TRUE,
                sketch_order=budget_sketch_order,
                chunk_budget=int(b),
            )
        )
        budget_methods.append(
            KSketchMethodSpec(
                name=f"budget_wrong_chunker_b{b}",
                description=f"misspecified + bottom-proxy, budget={b}",
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
    budget_index = {(str(r["method_name"]), int(r["target_k"])): r for r in budget_rows}

    def _abs_bias_series(prefix: str) -> List[float]:
        series: List[float] = []
        for b in budget_values:
            row = budget_index[(f"{prefix}{b}", budget_target_k)]
            series.append(float(row["mean_abs_bias"]))
        return series

    aligned_proxy_abs = _abs_bias_series("budget_aligned_top_proxy_b")
    aligned_oracle_abs = _abs_bias_series("budget_aligned_oracle_b")
    wrong_abs = _abs_bias_series("budget_wrong_chunker_b")
    one_pass_abs = float(budget_index[("budget_one_pass_reference", budget_target_k)]["mean_abs_bias"])

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.0), constrained_layout=True)

    # Sketch-gap panel.
    ax = axes[0]
    for label, curve, color, marker in (
        ("one-pass oracle", one_pass_delta, "#1f77b4", "o"),
        (f"full model (budget={args.phase_chunk_budget})", full_proxy_delta, "#2ca02c", "s"),
    ):
        xs = sorted(curve.keys())
        ys = [curve[x] for x in xs]
        ax.plot(xs, ys, marker=marker, color=color, label=label)
    if one_pass_delta:
        xmin = min(one_pass_delta.keys())
        xmax = max(one_pass_delta.keys())
    else:
        xmin, xmax = -3, 3
    ax.axvspan(xmin, -1e-9, color="#444444", alpha=0.07, linewidth=0)
    ax.hlines(naive_majority_mean, xmin=xmin, xmax=xmax, colors="#d62728", linestyles="--", label="naive majority")
    ax.hlines(naive_mom_mean, xmin=xmin, xmax=xmax, colors="#9467bd", linestyles="--", label="naive mean-of-means")
    ax.axvline(0, color="#444444", linewidth=1, alpha=0.75)
    ax.set_xlabel("Sketch gap Δ = m − k")
    ax.set_ylabel("Mean absolute bias")
    ax.set_title("Sketch-Sufficiency Ceiling (information-limited when m<k)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)

    # Budget-ceiling panel.
    ax2 = axes[1]
    x_budget = [int(b) for b in budget_values]
    ax2.plot(
        x_budget,
        aligned_oracle_abs,
        marker="o",
        color="#111111",
        linestyle="--",
        label="oracle selector (upper bound)",
    )
    ax2.plot(
        x_budget,
        aligned_proxy_abs,
        marker="o",
        color="#2ca02c",
        label="full model (top-proxy)",
    )
    ax2.plot(
        x_budget,
        wrong_abs,
        marker="s",
        color="#ff7f0e",
        label="ablation: wrong chunker + bottom-proxy",
    )
    ax2.hlines(
        one_pass_abs,
        xmin=min(x_budget),
        xmax=max(x_budget),
        colors="#1f77b4",
        linestyles="-",
        label="one-pass oracle (ceiling)",
    )
    ax2.set_xlabel("Chunk budget b")
    ax2.set_ylabel("Mean absolute bias")
    ax2.set_title(f"Budget-Retention Ceiling (k={budget_target_k}, m={budget_sketch_order})")
    ax2.grid(alpha=0.2)
    ax2.legend(frameon=False, fontsize=9)

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
            metadata={"family": "mergeable_ceilings"},
        ),
        "weighting_modes": list(weighting_modes),
        "legacy_weighting_mode": str(args.legacy_weighting_mode),
        "phase_chunk_budget": int(args.phase_chunk_budget),
        "phase_rows": phase_rows,
        "phase_delta_curve": {
            "one_pass": {str(k): v for k, v in one_pass_delta.items()},
            "full_model_top_proxy": {str(k): v for k, v in full_proxy_delta.items()},
        },
        "phase_naive_mean_abs_bias": {
            "naive_majority": naive_majority_mean,
            "naive_mean_of_means": naive_mom_mean,
        },
        "budget_values": list(budget_values),
        "budget_target_k": budget_target_k,
        "budget_sketch_order": budget_sketch_order,
        "budget_rows": budget_rows,
        "budget_series_mean_abs_bias": {
            "one_pass": one_pass_abs,
            "aligned_top_proxy": aligned_proxy_abs,
            "aligned_oracle_top_true": aligned_oracle_abs,
            "wrong_chunker_bottom_proxy": wrong_abs,
        },
        "output_figure": str(out_path),
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote_figure | {out_path}")
    print(f"wrote_summary | {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
