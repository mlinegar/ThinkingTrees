#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


LANE_TO_RUNNER = {
    "markov": "scripts/run_markov_treepo_preference.py",
    "nonseparable": "scripts/run_nonseparable_treepo_preference.py",
    "boundary_topic": "scripts/run_boundary_topic_treepo_preference.py",
}

LANE_PROFILES = {
    "markov": ("markov_count_only", "markov_count_endpoints"),
    "nonseparable": (
        "dgp1_complementarity_and",
        "dgp1_complementarity_control",
        "dgp2_boundary_interaction",
        "dgp2_boundary_zero",
    ),
    "boundary_topic": ("topic_mass_only", "topic_plus_boundary"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build command list for exact utility transport suite.")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--cmd-file", type=Path, required=True)
    p.add_argument("--train-docs", type=int, nargs="+", default=[128, 512])
    p.add_argument("--objective-family", type=str, nargs="+", default=["supervised_state", "dpo", "grpo", "ppo"])
    p.add_argument("--structural-arm", type=str, nargs="+", default=["oracle_exact", "tree_neural_supported", "tree_undersupported", "flat_equal_info", "one_leaf_control"])
    p.add_argument("--seed", type=int, nargs="+", default=[0, 1])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    lines: list[str] = []
    for lane, runner in LANE_TO_RUNNER.items():
        for oracle_profile in LANE_PROFILES[lane]:
            for objective_family in args.objective_family:
                for structural_arm in args.structural_arm:
                    for train_docs in args.train_docs:
                        for seed in args.seed:
                            out = (
                                args.output_root
                                / lane
                                / oracle_profile
                                / objective_family
                                / structural_arm
                                / f"train_{train_docs}"
                                / f"seed_{seed}.json"
                            )
                            cmd = (
                                f"source venv/bin/activate && python {runner} "
                                f"--oracle-profile {oracle_profile} "
                                f"--objective-family {objective_family} "
                                f"--structural-arm {structural_arm} "
                                f"--train-docs {train_docs} --test-docs 128 "
                                f"--seed {seed} --json-summary {out}"
                            )
                            lines.append(cmd)
    args.cmd_file.parent.mkdir(parents=True, exist_ok=True)
    args.cmd_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote_cmds | {args.cmd_file} | n={len(lines)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
