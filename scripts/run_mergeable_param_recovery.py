#!/usr/bin/env python3
"""Run spike-prevalence parameter recovery study for mergeable ablations."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tree.mergeable_ablation import (
    SpikeMixtureDistributionSpec,
    run_four_parameter_recovery_study,
    run_spike_prevalence_recovery_study,
    run_three_parameter_recovery_study,
    run_two_parameter_recovery_study,
)
from src.ctreepo.sim.objective_semantics import mergeable_parameter_vector_objective_semantics


def _parse_str_csv(s: str) -> tuple[str, ...]:
    out = tuple(x.strip() for x in s.split(",") if x.strip())
    if len(out) == 0:
        raise ValueError("expected a non-empty string CSV")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover known DGP spike prevalence under repeated aggregation ablations."
    )
    parser.add_argument("--p-spike-doc", type=float, default=0.50, help="True P(doc has spike).")
    parser.add_argument("--p-boundary-given-spike", type=float, default=0.50)
    parser.add_argument("--p-two-spikes-given-spike", type=float, default=0.25)
    parser.add_argument(
        "--p-multi-given-two-spikes",
        type=float,
        default=0.0,
        help="Within the >=2-spike mass, probability of generating explicit 3+ spike documents.",
    )
    parser.add_argument("--n-tokens", type=int, default=32)
    parser.add_argument("--proxy-noise", type=float, default=0.08)
    parser.add_argument(
        "--boundary-span-tokens",
        type=int,
        default=4,
        help="Boundary window size for boundary-spike parameter and sufficient statistic.",
    )
    parser.add_argument("--n-replicates", type=int, default=200)
    parser.add_argument("--docs-per-replicate", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--two-param",
        action="store_true",
        help="Recover both p_spike and p_two_spikes|spike (plus p_two_doc).",
    )
    mode_group.add_argument(
        "--three-param",
        action="store_true",
        help="Recover p_spike, p_two_spikes|spike, and p_boundary|spike jointly.",
    )
    mode_group.add_argument(
        "--four-param",
        action="store_true",
        help="Recover p_spike, p_two|spike, p_three_plus|spike, and p_boundary|spike jointly.",
    )
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
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weighting_modes = _parse_str_csv(args.weighting_modes)
    spec = SpikeMixtureDistributionSpec(
        p_spike_doc=args.p_spike_doc,
        p_boundary_given_spike=args.p_boundary_given_spike,
        p_two_spikes_given_spike=args.p_two_spikes_given_spike,
        p_multi_given_two_spikes=args.p_multi_given_two_spikes,
        n_tokens=args.n_tokens,
        proxy_noise=args.proxy_noise,
        boundary_span_tokens=args.boundary_span_tokens,
    )
    if args.four_param:
        summaries = run_four_parameter_recovery_study(
            distribution=spec,
            n_replicates=args.n_replicates,
            docs_per_replicate=args.docs_per_replicate,
            seed=args.seed,
            weighting_modes=weighting_modes,
            legacy_weighting_mode=args.legacy_weighting_mode,
        )
        objective = mergeable_parameter_vector_objective_semantics(
            name="mergeable_four_parameter_target",
            parameter_names=(
                "p_spike_doc",
                "p_two_given_spike",
                "p_three_given_spike",
                "p_boundary_given_spike",
            ),
            optimized_against="four_parameter_recovery",
            metadata={"family": "mergeable_param_recovery"},
        )
    elif args.three_param:
        summaries = run_three_parameter_recovery_study(
            distribution=spec,
            n_replicates=args.n_replicates,
            docs_per_replicate=args.docs_per_replicate,
            seed=args.seed,
            weighting_modes=weighting_modes,
            legacy_weighting_mode=args.legacy_weighting_mode,
        )
        objective = mergeable_parameter_vector_objective_semantics(
            name="mergeable_three_parameter_target",
            parameter_names=("p_spike_doc", "p_two_given_spike", "p_boundary_given_spike"),
            optimized_against="three_parameter_recovery",
            metadata={"family": "mergeable_param_recovery"},
        )
    elif args.two_param:
        summaries = run_two_parameter_recovery_study(
            distribution=spec,
            n_replicates=args.n_replicates,
            docs_per_replicate=args.docs_per_replicate,
            seed=args.seed,
            weighting_modes=weighting_modes,
            legacy_weighting_mode=args.legacy_weighting_mode,
        )
        objective = mergeable_parameter_vector_objective_semantics(
            name="mergeable_two_parameter_target",
            parameter_names=("p_spike_doc", "p_two_given_spike"),
            optimized_against="two_parameter_recovery",
            metadata={"family": "mergeable_param_recovery"},
        )
    else:
        summaries = run_spike_prevalence_recovery_study(
            distribution=spec,
            n_replicates=args.n_replicates,
            docs_per_replicate=args.docs_per_replicate,
            seed=args.seed,
            weighting_modes=weighting_modes,
            legacy_weighting_mode=args.legacy_weighting_mode,
        )
        objective = mergeable_parameter_vector_objective_semantics(
            name="mergeable_spike_prevalence_target",
            parameter_names=("p_spike_doc",),
            optimized_against="spike_prevalence_recovery",
            metadata={"family": "mergeable_param_recovery"},
        )

    if args.json:
        payload = {
            "distribution": asdict(spec),
            "weighting_modes": list(weighting_modes),
            "legacy_weighting_mode": str(args.legacy_weighting_mode),
            "objective": objective,
            "summaries": [asdict(s) for s in summaries],
        }
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    if args.four_param:
        p_three_true = spec.p_two_spikes_given_spike * spec.p_multi_given_two_spikes
        print(
            f"p_spike_true={spec.p_spike_doc:.4f} p_two_given_spike_true={spec.p_two_spikes_given_spike:.4f} "
            f"p_three_given_spike_true={p_three_true:.4f} p_boundary_given_spike_true={spec.p_boundary_given_spike:.4f} "
            f"reps={args.n_replicates} docs_per_rep={args.docs_per_replicate} tokens={spec.n_tokens} "
            f"boundary_span={spec.boundary_span_tokens}"
        )
        print(
            "method | supports_two | supports_three | supports_boundary | p_spike_hat | bias_spike | "
            "p_two|spike_hat | bias_two|spike | p_three|spike_hat | bias_three|spike | "
            "p_boundary|spike_hat | bias_boundary|spike"
        )
        for s in summaries:
            print(
                f"{s.method_name} | {int(bool(s.supports_two_spike))} | {int(bool(s.supports_three_spike))} | "
                f"{int(bool(s.supports_boundary_spike))} | {s.mean_hat_p_spike:.4f} | {s.bias_p_spike:+.4f} | "
                f"{s.mean_hat_p_two_given_spike:.4f} | {s.bias_p_two_given_spike:+.4f} | "
                f"{s.mean_hat_p_three_given_spike:.4f} | {s.bias_p_three_given_spike:+.4f} | "
                f"{s.mean_hat_p_boundary_given_spike:.4f} | {s.bias_p_boundary_given_spike:+.4f}"
            )
            print(f"  {s.description}")
    elif args.three_param:
        print(
            f"p_spike_true={spec.p_spike_doc:.4f} p_two_given_spike_true={spec.p_two_spikes_given_spike:.4f} "
            f"p_boundary_given_spike_true={spec.p_boundary_given_spike:.4f} reps={args.n_replicates} "
            f"docs_per_rep={args.docs_per_replicate} tokens={spec.n_tokens} boundary_span={spec.boundary_span_tokens}"
        )
        print(
            "method | supports_two | supports_boundary | p_spike_hat | bias_spike | "
            "p_two|spike_hat | bias_two|spike | p_boundary|spike_hat | bias_boundary|spike"
        )
        for s in summaries:
            print(
                f"{s.method_name} | {int(bool(s.supports_two_spike))} | {int(bool(s.supports_boundary_spike))} | "
                f"{s.mean_hat_p_spike:.4f} | {s.bias_p_spike:+.4f} | "
                f"{s.mean_hat_p_two_given_spike:.4f} | {s.bias_p_two_given_spike:+.4f} | "
                f"{s.mean_hat_p_boundary_given_spike:.4f} | {s.bias_p_boundary_given_spike:+.4f}"
            )
            print(f"  {s.description}")
    elif args.two_param:
        print(
            f"p_spike_true={spec.p_spike_doc:.4f} p_two_given_spike_true={spec.p_two_spikes_given_spike:.4f} "
            f"reps={args.n_replicates} docs_per_rep={args.docs_per_replicate} tokens={spec.n_tokens} "
            f"boundary_span={spec.boundary_span_tokens}"
        )
        print(
            "method | supports_two | p_spike_hat | bias_spike | p_two|spike_hat | bias_two|spike | p_two_doc_hat | bias_two_doc"
        )
        for s in summaries:
            print(
                f"{s.method_name} | {int(bool(s.supports_two_spike))} | "
                f"{s.mean_hat_p_spike:.4f} | {s.bias_p_spike:+.4f} | "
                f"{s.mean_hat_p_two_given_spike:.4f} | {s.bias_p_two_given_spike:+.4f} | "
                f"{s.mean_hat_p_two_doc:.4f} | {s.bias_p_two_doc:+.4f}"
            )
            print(f"  {s.description}")
    else:
        print(
            f"theta_true={spec.p_spike_doc:.4f} reps={args.n_replicates} "
            f"docs_per_rep={args.docs_per_replicate} tokens={spec.n_tokens} "
            f"boundary_span={spec.boundary_span_tokens}"
        )
        print(
            "method | mean_est | mean_bias | mean_abs_bias | sample_target_bias | std_est | rmse"
        )
        for s in summaries:
            print(
                f"{s.method_name} | {s.mean_estimate:.4f} | {s.mean_bias:+.4f} | "
                f"{s.mean_abs_bias:.4f} | {s.sample_target_bias:+.4f} | {s.std_estimate:.4f} | {s.rmse:.4f}"
            )
            print(f"  {s.description}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
