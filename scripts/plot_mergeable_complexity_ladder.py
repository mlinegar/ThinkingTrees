#!/usr/bin/env python3
"""Plot staged mergeable ablation complexity ladder and failure signatures."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tree.mergeable_ablation import (  # noqa: E402
    SpikeCountMixtureDistributionSpec,
    SpikeMixtureDistributionSpec,
    default_four_parameter_method_specs,
    default_k_sketch_method_specs,
    default_three_parameter_method_specs,
    default_two_parameter_method_specs,
    run_four_parameter_recovery_study,
    run_k_target_recovery_study,
    run_spike_prevalence_recovery_study,
    run_three_parameter_recovery_study,
    run_two_parameter_recovery_study,
)
from src.ctreepo.sim.objective_semantics import mergeable_parameter_vector_objective_semantics


def _parse_int_csv(s: str) -> Tuple[int, ...]:
    out = tuple(int(x.strip()) for x in s.split(",") if x.strip())
    if len(out) == 0:
        raise ValueError("Expected non-empty comma-separated int list")
    return out


def _parse_str_csv(s: str) -> Tuple[str, ...]:
    out = tuple(x.strip() for x in s.split(",") if x.strip())
    if len(out) == 0:
        raise ValueError("Expected non-empty comma-separated string list")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run staged 1->5 complexity ladder and plot aggregate/parameter-specific failure patterns."
        )
    )
    parser.add_argument("--p-spike-doc", type=float, default=0.62)
    parser.add_argument("--p-two-spikes-given-spike", type=float, default=0.45)
    parser.add_argument("--p-multi-given-two-spikes", type=float, default=0.35)
    parser.add_argument("--p-boundary-given-spike", type=float, default=0.35)
    parser.add_argument("--n-tokens", type=int, default=32)
    parser.add_argument("--proxy-noise", type=float, default=0.12)
    parser.add_argument("--boundary-span-tokens", type=int, default=4)
    parser.add_argument(
        "--generic-target-ks",
        type=str,
        default="2,3,4,5",
        help="Comma-separated k values for stage-5 generic-k recovery.",
    )
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
        "--output",
        type=str,
        default="outputs/mergeable_complexity_ladder.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--json-summary",
        type=str,
        default="outputs/mergeable_complexity_ladder_summary.json",
        help="JSON summary output path.",
    )
    return parser.parse_args()


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) == 0:
        return float("nan")
    return float(sum(vals)) / float(len(vals))


def _add_metric(
    stage_metrics: Dict[str, Dict[str, float]],
    method: str,
    stage_key: str,
    value: float,
) -> None:
    stage_metrics.setdefault(method, {})
    stage_metrics[method][stage_key] = float(value)


def _fmt_method(name: str) -> str:
    mapping = {
        "one_pass_oracle": "one-pass oracle",
        "full_model_aligned": "full model",
        "full_model_limited_sketch": "full model, m<k",
        "full_model_missing_three_stat": "missing 3rd-order stat",
        "full_model_missing_boundary_stat": "missing boundary stat",
        "naive_majority_same_chunker": "naive majority",
        "naive_mean_of_means_same_chunker": "naive mean-of-means",
        "right_rule_wrong_chunker": "right rule, wrong chunker",
    }
    return mapping.get(name, name.replace("_", " "))


def main() -> int:
    args = parse_args()
    weighting_modes = _parse_str_csv(args.weighting_modes)

    base_spec = SpikeMixtureDistributionSpec(
        p_spike_doc=args.p_spike_doc,
        p_two_spikes_given_spike=args.p_two_spikes_given_spike,
        p_multi_given_two_spikes=args.p_multi_given_two_spikes,
        p_boundary_given_spike=args.p_boundary_given_spike,
        n_tokens=args.n_tokens,
        proxy_noise=args.proxy_noise,
        boundary_span_tokens=args.boundary_span_tokens,
    )
    target_ks = _parse_int_csv(args.generic_target_ks)

    stage_metrics: Dict[str, Dict[str, float]] = {}
    stage_rows: Dict[str, Dict[str, dict]] = {
        "stage1": {},
        "stage2": {},
        "stage3": {},
        "stage4": {},
        "stage5": {},
    }

    # Stage 1: recover P(spike)
    methods_stage12 = list(default_two_parameter_method_specs())
    stage1 = run_spike_prevalence_recovery_study(
        distribution=base_spec,
        methods=methods_stage12,
        n_replicates=args.n_replicates,
        docs_per_replicate=args.docs_per_replicate,
        seed=args.seed,
        weighting_modes=weighting_modes,
        legacy_weighting_mode=args.legacy_weighting_mode,
    )
    for row in stage1:
        r = asdict(row)
        method = str(r["method_name"])
        stage_rows["stage1"][method] = r
        _add_metric(stage_metrics, method, "stage1", float(r["mean_abs_bias"]))

    # Stage 2: add P(>=2 | spike)
    stage2 = run_two_parameter_recovery_study(
        distribution=base_spec,
        methods=methods_stage12,
        n_replicates=args.n_replicates,
        docs_per_replicate=args.docs_per_replicate,
        seed=args.seed + 10_000,
        weighting_modes=weighting_modes,
        legacy_weighting_mode=args.legacy_weighting_mode,
    )
    for row in stage2:
        r = asdict(row)
        method = str(r["method_name"])
        stage_rows["stage2"][method] = r
        agg = _mean([
            float(r["mean_abs_bias_p_spike"]),
            float(r["mean_abs_bias_p_two_given_spike"]),
        ])
        _add_metric(stage_metrics, method, "stage2", agg)

    # Stage 3: add P(boundary | spike)
    stage3 = run_three_parameter_recovery_study(
        distribution=base_spec,
        methods=default_three_parameter_method_specs(),
        n_replicates=args.n_replicates,
        docs_per_replicate=args.docs_per_replicate,
        seed=args.seed + 20_000,
        weighting_modes=weighting_modes,
        legacy_weighting_mode=args.legacy_weighting_mode,
    )
    for row in stage3:
        r = asdict(row)
        method = str(r["method_name"])
        stage_rows["stage3"][method] = r
        agg = _mean([
            float(r["mean_abs_bias_p_spike"]),
            float(r["mean_abs_bias_p_two_given_spike"]),
            float(r["mean_abs_bias_p_boundary_given_spike"]),
        ])
        _add_metric(stage_metrics, method, "stage3", agg)

    # Stage 4: add P(>=3 | spike)
    stage4 = run_four_parameter_recovery_study(
        distribution=base_spec,
        methods=default_four_parameter_method_specs(),
        n_replicates=args.n_replicates,
        docs_per_replicate=args.docs_per_replicate,
        seed=args.seed + 30_000,
        weighting_modes=weighting_modes,
        legacy_weighting_mode=args.legacy_weighting_mode,
    )
    for row in stage4:
        r = asdict(row)
        method = str(r["method_name"])
        stage_rows["stage4"][method] = r
        agg = _mean([
            float(r["mean_abs_bias_p_spike"]),
            float(r["mean_abs_bias_p_two_given_spike"]),
            float(r["mean_abs_bias_p_three_given_spike"]),
            float(r["mean_abs_bias_p_boundary_given_spike"]),
        ])
        _add_metric(stage_metrics, method, "stage4", agg)

    # Stage 5: generic-k family P(count>=k | spike)
    count_spec = SpikeCountMixtureDistributionSpec(
        p_spike_doc=args.p_spike_doc,
        p_boundary_given_spike=args.p_boundary_given_spike,
        spike_count_support=(1, 2, 3, 4, 5),
        spike_count_probs_given_spike=(
            1.0 - args.p_two_spikes_given_spike,
            args.p_two_spikes_given_spike * (1.0 - args.p_multi_given_two_spikes),
            args.p_two_spikes_given_spike * args.p_multi_given_two_spikes * 0.50,
            args.p_two_spikes_given_spike * args.p_multi_given_two_spikes * 0.30,
            args.p_two_spikes_given_spike * args.p_multi_given_two_spikes * 0.20,
        ),
        n_tokens=args.n_tokens,
        proxy_noise=args.proxy_noise,
        boundary_span_tokens=args.boundary_span_tokens,
    )
    stage5 = run_k_target_recovery_study(
        distribution=count_spec,
        target_ks=target_ks,
        methods=default_k_sketch_method_specs(target_max_k=max(target_ks)),
        n_replicates=args.n_replicates,
        docs_per_replicate=args.docs_per_replicate,
        seed=args.seed + 40_000,
        weighting_modes=weighting_modes,
        legacy_weighting_mode=args.legacy_weighting_mode,
    )
    stage5_by_method: Dict[str, List[dict]] = {}
    for row in stage5:
        r = asdict(row)
        method = str(r["method_name"])
        stage5_by_method.setdefault(method, []).append(r)
    for method, rows in stage5_by_method.items():
        rows_sorted = sorted(rows, key=lambda x: int(x["target_k"]))
        stage_rows["stage5"][method] = {
            "target_ks": [int(r["target_k"]) for r in rows_sorted],
            "rows": rows_sorted,
        }
        agg = _mean(float(r["mean_abs_bias"]) for r in rows_sorted)
        _add_metric(stage_metrics, method, "stage5", agg)

    method_order = [
        "one_pass_oracle",
        "full_model_aligned",
        "full_model_limited_sketch",
        "full_model_missing_three_stat",
        "full_model_missing_boundary_stat",
        "right_rule_wrong_chunker",
        "naive_majority_same_chunker",
        "naive_mean_of_means_same_chunker",
    ]
    stage_order = ["stage1", "stage2", "stage3", "stage4", "stage5"]
    stage_labels = {
        "stage1": "S1: P(spike)",
        "stage2": "S2: +P(>=2|spike)",
        "stage3": "S3: +P(boundary|spike)",
        "stage4": "S4: +P(>=3|spike)",
        "stage5": "S5: generic-k family",
    }
    method_colors = {
        "one_pass_oracle": "#1f77b4",
        "full_model_aligned": "#2ca02c",
        "full_model_limited_sketch": "#17becf",
        "full_model_missing_three_stat": "#ff7f0e",
        "full_model_missing_boundary_stat": "#bcbd22",
        "right_rule_wrong_chunker": "#d62728",
        "naive_majority_same_chunker": "#9467bd",
        "naive_mean_of_means_same_chunker": "#8c564b",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2), constrained_layout=True)

    ax0 = axes[0]
    x_vals = list(range(len(stage_order)))
    for method in method_order:
        ys = [stage_metrics.get(method, {}).get(s, float("nan")) for s in stage_order]
        if all(math.isnan(y) for y in ys):
            continue
        ax0.plot(
            x_vals,
            ys,
            marker="o",
            linewidth=2,
            color=method_colors.get(method, "#333333"),
            label=_fmt_method(method),
        )
    ax0.set_xticks(x_vals)
    ax0.set_xticklabels([stage_labels[s] for s in stage_order], rotation=20, ha="right")
    ax0.set_ylabel("Aggregate mean absolute bias")
    ax0.set_title("Complexity Ladder: What Stays Accurate as Targets Get Harder")
    ax0.grid(alpha=0.25)
    ax0.legend(frameon=False, fontsize=8)

    ax1 = axes[1]
    stage4_methods = [
        m
        for m in method_order
        if m in stage_rows["stage4"]
    ]
    param_keys = [
        "mean_abs_bias_p_spike",
        "mean_abs_bias_p_two_given_spike",
        "mean_abs_bias_p_three_given_spike",
        "mean_abs_bias_p_boundary_given_spike",
    ]
    param_labels = ["P(spike)", "P(>=2|spike)", "P(>=3|spike)", "P(boundary|spike)"]

    heat = [
        [float(stage_rows["stage4"][m][k]) for k in param_keys]
        for m in stage4_methods
    ]
    vmax = max(max(row) for row in heat) if len(heat) > 0 else 1.0
    im = ax1.imshow(heat, aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)
    ax1.set_xticks(list(range(len(param_labels))))
    ax1.set_xticklabels(param_labels, rotation=20, ha="right")
    ax1.set_yticks(list(range(len(stage4_methods))))
    ax1.set_yticklabels([_fmt_method(m) for m in stage4_methods])
    ax1.set_title("Stage 4 Failure Signature by Parameter")

    for i, row in enumerate(heat):
        for j, val in enumerate(row):
            txt_color = "white" if val > (0.55 * vmax) else "black"
            ax1.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=txt_color)

    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Mean absolute bias")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)

    payload = {
        "config": {
            "p_spike_doc": float(args.p_spike_doc),
            "p_two_spikes_given_spike": float(args.p_two_spikes_given_spike),
            "p_multi_given_two_spikes": float(args.p_multi_given_two_spikes),
            "p_boundary_given_spike": float(args.p_boundary_given_spike),
            "n_tokens": int(args.n_tokens),
            "proxy_noise": float(args.proxy_noise),
            "boundary_span_tokens": int(args.boundary_span_tokens),
            "generic_target_ks": list(target_ks),
            "n_replicates": int(args.n_replicates),
            "docs_per_replicate": int(args.docs_per_replicate),
            "seed": int(args.seed),
            "weighting_modes": list(weighting_modes),
            "legacy_weighting_mode": str(args.legacy_weighting_mode),
        },
        "objective": mergeable_parameter_vector_objective_semantics(
            name="mergeable_complexity_ladder_target_family",
            parameter_names=(
                "p_spike_doc",
                "p_two_given_spike",
                "p_three_given_spike",
                "p_boundary_given_spike",
            ),
            optimized_against="stagewise_parameter_vector_recovery",
            metadata={
                "family": "mergeable_complexity_ladder",
                "generic_target_ks": list(target_ks),
                "includes_generic_k_stage": True,
            },
        ),
        "stage_order": stage_order,
        "stage_labels": stage_labels,
        "method_order": method_order,
        "method_display_names": {m: _fmt_method(m) for m in method_order},
        "stage_metrics": stage_metrics,
        "stage_rows": stage_rows,
        "output_figure": str(out_path),
    }

    summary_path = Path(args.json_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote_figure | {out_path}")
    print(f"wrote_summary | {summary_path}")

    for method in method_order:
        vals = stage_metrics.get(method)
        if not vals:
            continue
        s2 = vals.get("stage2", float("nan"))
        s4 = vals.get("stage4", float("nan"))
        s5 = vals.get("stage5", float("nan"))
        print(
            f"method={method} | agg_abs_bias stage2={s2:.4f} "
            f"stage4={s4:.4f} stage5={s5:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
