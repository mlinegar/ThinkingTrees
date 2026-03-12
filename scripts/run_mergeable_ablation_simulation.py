#!/usr/bin/env python3
"""Run simple repeated-aggregation ablations for mergeable vs naive methods."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tree.mergeable_ablation import run_default_ablation_suite, worked_failure_examples
from src.ctreepo.sim.objective_semantics import mergeable_document_objective_semantics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run numeric ablations for repeated tree aggregation failure modes."
    )
    parser.add_argument("--n-docs", type=int, default=240, help="Number of toy documents.")
    parser.add_argument("--n-tokens", type=int, default=32, help="Tokens per toy document.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--show-worked-examples",
        action="store_true",
        help="Include two tiny deterministic worked examples in output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = run_default_ablation_suite(
        n_docs=args.n_docs,
        n_tokens=args.n_tokens,
        seed=args.seed,
    )

    if args.json:
        payload = {
            "n_docs": args.n_docs,
            "n_tokens": args.n_tokens,
            "seed": args.seed,
            "objective": mergeable_document_objective_semantics(
                name="mergeable_ablation_document_objective",
                objective_profile="spike_exists",
                metadata={"family": "mergeable_ablation"},
            ),
            "summaries": [asdict(s) for s in summaries],
        }
        if args.show_worked_examples:
            payload["worked_examples"] = worked_failure_examples()
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    print(f"docs={args.n_docs} tokens={args.n_tokens} seed={args.seed}")
    print(
        "method | mean_abs_error | label_error | order_spread | order_flip | mean_chunks_kept"
    )
    for s in summaries:
        print(
            f"{s.name} | {s.mean_abs_error:.4f} | {s.label_error_rate:.4f} | "
            f"{s.order_spread_mean:.4f} | {s.order_flip_rate:.4f} | {s.mean_chunks_kept:.2f}"
        )
        print(f"  {s.description}")

    if args.show_worked_examples:
        print("\nworked_examples:")
        for name, tokens, outputs in worked_failure_examples():
            print(f"- {name}")
            print(f"  tokens={tokens}")
            for label, value in outputs:
                print(f"  {label}={value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
