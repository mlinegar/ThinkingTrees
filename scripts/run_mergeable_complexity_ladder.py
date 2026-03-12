#!/usr/bin/env python3
"""Run a staged 1->4 parameter complexity ladder for mergeable ablation studies."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tree.mergeable_ablation import (
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


def _parse_str_csv(s: str) -> tuple[str, ...]:
    out = tuple(x.strip() for x in s.split(",") if x.strip())
    if len(out) == 0:
        raise ValueError("expected a non-empty string CSV")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run staged 1->4 parameter recovery to build intuition from simple to complex."
    )
    parser.add_argument("--p-spike-doc", type=float, default=0.62)
    parser.add_argument("--p-two-spikes-given-spike", type=float, default=0.45)
    parser.add_argument("--p-multi-given-two-spikes", type=float, default=0.35)
    parser.add_argument("--p-boundary-given-spike", type=float, default=0.35)
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
        "--generic-target-ks",
        type=str,
        default="2,3,4,5",
        help="Comma-separated k values for stage5 generic-k recovery.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weighting_modes = _parse_str_csv(args.weighting_modes)
    spec = SpikeMixtureDistributionSpec(
        p_spike_doc=args.p_spike_doc,
        p_two_spikes_given_spike=args.p_two_spikes_given_spike,
        p_multi_given_two_spikes=args.p_multi_given_two_spikes,
        p_boundary_given_spike=args.p_boundary_given_spike,
        n_tokens=args.n_tokens,
        proxy_noise=args.proxy_noise,
        boundary_span_tokens=args.boundary_span_tokens,
    )

    print(
        "ladder_config | "
        f"p_spike={spec.p_spike_doc:.3f} p_two|spike={spec.p_two_spikes_given_spike:.3f} "
        f"p_multi|two={spec.p_multi_given_two_spikes:.3f} p_boundary|spike={spec.p_boundary_given_spike:.3f} "
        f"reps={args.n_replicates} docs_per_rep={args.docs_per_replicate} tokens={spec.n_tokens}"
    )
    target_ks = tuple(int(x.strip()) for x in args.generic_target_ks.split(",") if x.strip())

    stage1 = run_spike_prevalence_recovery_study(
        distribution=spec,
        n_replicates=args.n_replicates,
        docs_per_replicate=args.docs_per_replicate,
        seed=args.seed,
        weighting_modes=weighting_modes,
        legacy_weighting_mode=args.legacy_weighting_mode,
    )
    print("\n[stage1] recover p_spike")
    print("method | mean_abs_bias")
    for s in stage1:
        print(f"{s.method_name} | {s.mean_abs_bias:.4f}")

    stage2 = run_two_parameter_recovery_study(
        distribution=spec,
        methods=default_two_parameter_method_specs(),
        n_replicates=args.n_replicates,
        docs_per_replicate=args.docs_per_replicate,
        seed=args.seed + 10_000,
        weighting_modes=weighting_modes,
        legacy_weighting_mode=args.legacy_weighting_mode,
    )
    print("\n[stage2] recover p_spike, p_two|spike")
    print("method | abs_bias_spike | abs_bias_two|spike")
    for s in stage2:
        print(
            f"{s.method_name} | {s.mean_abs_bias_p_spike:.4f} | "
            f"{s.mean_abs_bias_p_two_given_spike:.4f}"
        )

    stage3 = run_three_parameter_recovery_study(
        distribution=spec,
        methods=default_three_parameter_method_specs(),
        n_replicates=args.n_replicates,
        docs_per_replicate=args.docs_per_replicate,
        seed=args.seed + 20_000,
        weighting_modes=weighting_modes,
        legacy_weighting_mode=args.legacy_weighting_mode,
    )
    print("\n[stage3] add p_boundary|spike")
    print("method | abs_bias_spike | abs_bias_two|spike | abs_bias_boundary|spike")
    for s in stage3:
        print(
            f"{s.method_name} | {s.mean_abs_bias_p_spike:.4f} | "
            f"{s.mean_abs_bias_p_two_given_spike:.4f} | {s.mean_abs_bias_p_boundary_given_spike:.4f}"
        )

    stage4 = run_four_parameter_recovery_study(
        distribution=spec,
        methods=default_four_parameter_method_specs(),
        n_replicates=args.n_replicates,
        docs_per_replicate=args.docs_per_replicate,
        seed=args.seed + 30_000,
        weighting_modes=weighting_modes,
        legacy_weighting_mode=args.legacy_weighting_mode,
    )
    print("\n[stage4] add p_three+|spike")
    print("method | abs_bias_spike | abs_bias_two|spike | abs_bias_three|spike | abs_bias_boundary|spike")
    for s in stage4:
        print(
            f"{s.method_name} | {s.mean_abs_bias_p_spike:.4f} | "
            f"{s.mean_abs_bias_p_two_given_spike:.4f} | "
            f"{s.mean_abs_bias_p_three_given_spike:.4f} | {s.mean_abs_bias_p_boundary_given_spike:.4f}"
        )

    generic_spec = SpikeCountMixtureDistributionSpec(
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
        distribution=generic_spec,
        target_ks=target_ks,
        methods=default_k_sketch_method_specs(target_max_k=max(target_ks)),
        n_replicates=args.n_replicates,
        docs_per_replicate=args.docs_per_replicate,
        seed=args.seed + 40_000,
        weighting_modes=weighting_modes,
        legacy_weighting_mode=args.legacy_weighting_mode,
    )
    print("\n[stage5] generic-k targets P(count>=k | spike)")
    print("method | k | supports_k | abs_bias")
    for s in sorted(stage5, key=lambda x: (x.target_k, x.mean_abs_bias)):
        print(
            f"{s.method_name} | {s.target_k} | {int(bool(s.supports_target))} | "
            f"{s.mean_abs_bias:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
