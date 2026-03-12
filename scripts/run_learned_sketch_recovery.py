#!/usr/bin/env python3
"""Run learned sketch recovery experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import directly from module file to avoid triggering src/tree/__init__.py
# which eagerly imports the full LLM stack (dspy, etc.)
import importlib.util as _ilu

_ls_path = str(REPO_ROOT / "src" / "tree" / "learned_sketch.py")
_spec = _ilu.spec_from_file_location("learned_sketch", _ls_path)
_ls = _ilu.module_from_spec(_spec)
sys.modules["learned_sketch"] = _ls
_spec.loader.exec_module(_ls)

DEFAULT_DISTRIBUTION = _ls.DEFAULT_DISTRIBUTION
SpikeCountMixtureDistributionSpec = _ls.SpikeCountMixtureDistributionSpec
run_learning_curve_experiment = _ls.run_learning_curve_experiment
run_convergence_comparison = _ls.run_convergence_comparison
run_phase_diagram_experiment = _ls.run_phase_diagram_experiment
run_audit_budget_experiment = _ls.run_audit_budget_experiment


def _parse_int_csv(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learned sketch recovery experiments."
    )
    parser.add_argument(
        "--experiment",
        choices=["curves", "comparison", "phase", "budget", "all"],
        default="curves",
        help="Which experiment to run.",
    )
    parser.add_argument("--target-k", type=int, default=5)
    parser.add_argument("--state-dims", type=str, default="3,4,5,6,7")
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: stdout).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state_dims = _parse_int_csv(args.state_dims)

    experiments = []
    if args.experiment in ("curves", "all"):
        print(f"Running learning curve experiment (k={args.target_k}, m={list(state_dims)})")
        result = run_learning_curve_experiment(
            target_k=args.target_k,
            state_dims=state_dims,
            n_steps=args.n_steps,
            seed=args.seed,
        )
        experiments.append(result)

    if args.experiment in ("comparison", "all"):
        print(f"Running convergence comparison (k={args.target_k}, m={args.target_k})")
        result = run_convergence_comparison(
            target_k=args.target_k,
            state_dim=args.target_k,
            n_steps=args.n_steps,
            seed=args.seed,
        )
        experiments.append(result)

    if args.experiment in ("phase", "all"):
        target_ks = _parse_int_csv("2,3,4,5")
        print(f"Running phase diagram (ks={list(target_ks)}, ms={list(state_dims)})")
        result = run_phase_diagram_experiment(
            target_ks=target_ks,
            state_dims=state_dims,
            n_steps=args.n_steps,
            seed=args.seed,
        )
        experiments.append(result)

    if args.experiment in ("budget", "all"):
        print(f"Running audit budget experiment (k={args.target_k})")
        result = run_audit_budget_experiment(
            target_k=args.target_k,
            state_dim=args.target_k,
            n_steps=args.n_steps,
            seed=args.seed,
        )
        experiments.append(result)

    payload = experiments[0] if len(experiments) == 1 else {"experiments": experiments}

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"Results written to {output_path}")
    else:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
