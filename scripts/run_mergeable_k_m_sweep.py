#!/usr/bin/env python3
"""Sweep sketch order m versus target k for generic-k recovery."""

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


def _relation(sketch_order: int, target_k: int) -> str:
    if sketch_order < target_k:
        return "unsupported(m<k)"
    if sketch_order == target_k:
        return "exact(m=k)"
    return "oversupported(m>k)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore unsupported vs over-supported regimes by sweeping sketch order m against target k."
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
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    support = _parse_int_csv(args.spike_count_support)
    probs = _parse_float_csv(args.spike_count_probs)
    target_ks = _parse_int_csv(args.target_ks)
    sketch_orders = _parse_int_csv(args.sketch_orders)
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

    methods = []
    for m in sketch_orders:
        methods.append(
            KSketchMethodSpec(
                name=f"one_pass_m{m}",
                description=f"one-pass oracle with top-{m} sketch",
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
                description=f"aligned adaptive chunking with top-{m} sketch",
                estimator=KSketchEstimator.MERGE_SAFE_TOPK,
                chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
                selector=SelectorPolicy.TOP_PROXY,
                sketch_order=m,
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

    rows = []
    for s in summaries:
        row = asdict(s)
        row["relation"] = _relation(s.sketch_order, s.target_k)
        row["method_family"] = "one_pass" if s.method_name.startswith("one_pass_") else "full_model"
        rows.append(row)

    if args.json:
        payload = {
            "distribution": asdict(spec),
            "target_ks": list(target_ks),
            "sketch_orders": list(sketch_orders),
            "weighting_modes": list(weighting_modes),
            "legacy_weighting_mode": str(args.legacy_weighting_mode),
            "objective": mergeable_probability_target_objective_semantics(
                name="generic_k_recovery_target_family",
                target_ks=target_ks,
                metadata={"family": "mergeable_k_m_sweep"},
            ),
            "rows": rows,
        }
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    print(
        f"mk_sweep | targets={list(target_ks)} sketch_orders={list(sketch_orders)} "
        f"reps={args.n_replicates} docs_per_rep={args.docs_per_replicate}"
    )
    print("method_family | method | k | m | relation | supports_k | abs_bias | bias")
    for r in sorted(rows, key=lambda x: (x["method_family"], x["target_k"], x["sketch_order"])):
        print(
            f"{r['method_family']} | {r['method_name']} | {r['target_k']} | {r['sketch_order']} | "
            f"{r['relation']} | {int(bool(r['supports_target']))} | {r['mean_abs_bias']:.4f} | {r['bias']:+.4f}"
        )

    # Compact phase summary.
    print("")
    print("phase_summary | method_family | k | unsupported_mean_abs_bias | exact_mean_abs_bias | oversupported_mean_abs_bias")
    for family in ("one_pass", "full_model"):
        for k in sorted(set(target_ks)):
            sub = [r for r in rows if r["method_family"] == family and r["target_k"] == k]
            unsupported = [r["mean_abs_bias"] for r in sub if r["sketch_order"] < k]
            exact = [r["mean_abs_bias"] for r in sub if r["sketch_order"] == k]
            oversup = [r["mean_abs_bias"] for r in sub if r["sketch_order"] > k]
            def _avg(xs):
                return sum(xs) / float(len(xs)) if xs else float("nan")
            print(
                f"phase_summary | {family} | {k} | {_avg(unsupported):.4f} | "
                f"{_avg(exact):.4f} | {_avg(oversup):.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
