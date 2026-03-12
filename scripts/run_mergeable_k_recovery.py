#!/usr/bin/env python3
"""Run generic-k recovery for P(count>=k | spike) under mergeable ablations."""

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
    SpikeCountMixtureDistributionSpec,
    run_k_target_recovery_study,
    sketch_insufficiency_counterexample,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover generic k-targets P(count>=k | spike) and expose sketch insufficiency."
    )
    parser.add_argument("--p-spike-doc", type=float, default=0.62)
    parser.add_argument("--p-boundary-given-spike", type=float, default=0.35)
    parser.add_argument("--spike-count-support", type=str, default="1,2,3,4,5")
    parser.add_argument("--spike-count-probs", type=str, default="0.28,0.27,0.20,0.15,0.10")
    parser.add_argument("--target-ks", type=str, default="2,3,4,5")
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
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--show-counterexample",
        action="store_true",
        help="Print explicit same-sketch/different-truth counterexample for m vs m+1.",
    )
    parser.add_argument("--counterexample-m", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    support = _parse_int_csv(args.spike_count_support)
    probs = _parse_float_csv(args.spike_count_probs)
    target_ks = _parse_int_csv(args.target_ks)
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
    summaries = run_k_target_recovery_study(
        distribution=spec,
        target_ks=target_ks,
        n_replicates=args.n_replicates,
        docs_per_replicate=args.docs_per_replicate,
        seed=args.seed,
        weighting_modes=weighting_modes,
        legacy_weighting_mode=args.legacy_weighting_mode,
    )

    if args.json:
        payload = {
            "distribution": asdict(spec),
            "target_ks": list(target_ks),
            "weighting_modes": list(weighting_modes),
            "legacy_weighting_mode": str(args.legacy_weighting_mode),
            "objective": mergeable_probability_target_objective_semantics(
                name="generic_k_recovery_target_family",
                target_ks=target_ks,
                metadata={"family": "mergeable_k_recovery"},
            ),
            "summaries": [asdict(s) for s in summaries],
        }
        if args.show_counterexample:
            m = max(1, int(args.counterexample_m))
            k = m + 1
            a, b, sig = sketch_insufficiency_counterexample(sketch_order=m, target_k=k)
            payload["counterexample"] = {
                "sketch_order": m,
                "target_k": k,
                "doc_a_scores": a,
                "doc_b_scores": b,
                "shared_topm_signature": sig,
            }
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    print(
        f"generic_k | p_spike={spec.p_spike_doc:.3f} support={list(support)} probs={list(probs)} "
        f"targets={list(target_ks)} reps={args.n_replicates} docs_per_rep={args.docs_per_replicate}"
    )
    print("method | k | supports_k | true_p>=k|spike | hat_p>=k|spike | bias | abs_bias")
    rows = sorted(summaries, key=lambda s: (s.target_k, s.mean_abs_bias))
    for s in rows:
        print(
            f"{s.method_name} | {s.target_k} | {int(bool(s.supports_target))} | "
            f"{s.true_p_at_least_k_given_spike:.4f} | {s.mean_hat_p_at_least_k_given_spike:.4f} | "
            f"{s.bias:+.4f} | {s.mean_abs_bias:.4f}"
        )

    if args.show_counterexample:
        m = max(1, int(args.counterexample_m))
        k = m + 1
        a, b, sig = sketch_insufficiency_counterexample(sketch_order=m, target_k=k)
        print("")
        print(f"counterexample | sketch_order={m} target_k={k}")
        print(f"shared_topm_signature={list(sig)}")
        print(f"doc_a_top{m}={sorted(a, reverse=True)[:m]}")
        print(f"doc_b_top{m}={sorted(b, reverse=True)[:m]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
