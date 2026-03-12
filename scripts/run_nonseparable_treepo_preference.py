#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.core.nonseparable_treepo_preference import (  # noqa: E402
    NonseparableExactUtilityConfig,
    run_nonseparable_exact_utility_experiment,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run exact nonseparable TreePO utility simulation.")
    p.add_argument("--oracle-profile", type=str, default="dgp1_complementarity_and")
    p.add_argument("--objective-family", type=str, default="supervised_state")
    p.add_argument("--structural-arm", type=str, default="tree_neural_supported")
    p.add_argument("--train-docs", type=int, default=256)
    p.add_argument("--test-docs", type=int, default=128)
    p.add_argument("--fixed-leaf-tokens", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--pairwise-prefs-per-doc", type=int, default=4)
    p.add_argument("--group-pref-groups-per-doc", type=int, default=2)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--ppo-rollouts-per-doc", type=int, default=4)
    p.add_argument("--ppo-kl-weight", type=float, default=0.02)
    p.add_argument("--entropy-weight", type=float, default=0.01)
    p.add_argument("--ppo-advantage-center", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ppo-advantage-normalize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ppo-reward-baseline", type=str, default="mean_reward", choices=["mean_reward", "none"])
    p.add_argument("--ppo-clip-epsilon", type=float, default=0.2)
    p.add_argument("--leaf-label-rate", type=float, default=0.0)
    p.add_argument("--internal-label-rate", type=float, default=0.0)
    p.add_argument("--root-query-rate", type=float, default=1.0)
    p.add_argument("--count-max", type=int, default=5)
    p.add_argument("--n-binary-leaves", type=int, default=4)
    p.add_argument("--use-cuda", action="store_true")
    p.add_argument("--cuda-device", type=int, default=None)
    p.add_argument("--json-summary", type=Path, required=True)
    p.add_argument("--json", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = NonseparableExactUtilityConfig(
        oracle_profile=str(args.oracle_profile),
        objective_family=str(args.objective_family),
        structural_arm=str(args.structural_arm),
        train_docs=int(args.train_docs),
        test_docs=int(args.test_docs),
        fixed_leaf_tokens=int(args.fixed_leaf_tokens),
        seed=int(args.seed),
        n_epochs=int(args.n_epochs),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        pairwise_prefs_per_doc=int(args.pairwise_prefs_per_doc),
        group_pref_groups_per_doc=int(args.group_pref_groups_per_doc),
        group_size=int(args.group_size),
        ppo_rollouts_per_doc=int(args.ppo_rollouts_per_doc),
        ppo_kl_weight=float(args.ppo_kl_weight),
        entropy_weight=float(args.entropy_weight),
        ppo_advantage_center=bool(args.ppo_advantage_center),
        ppo_advantage_normalize=bool(args.ppo_advantage_normalize),
        ppo_reward_baseline=str(args.ppo_reward_baseline),
        ppo_clip_epsilon=float(args.ppo_clip_epsilon),
        leaf_label_rate=float(args.leaf_label_rate),
        internal_label_rate=float(args.internal_label_rate),
        root_query_rate=float(args.root_query_rate),
        count_max=int(args.count_max),
        n_binary_leaves=int(args.n_binary_leaves),
        use_cuda=bool(args.use_cuda),
        cuda_device=int(args.cuda_device) if args.cuda_device is not None else None,
    )
    summary = run_nonseparable_exact_utility_experiment(cfg)
    args.json_summary.parent.mkdir(parents=True, exist_ok=True)
    args.json_summary.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote_json | {args.json_summary}")
    print(
        f"lane={summary.lane} | oracle_profile={summary.oracle_profile} | "
        f"objective_family={summary.objective_family} | structural_arm={summary.structural_arm} | "
        f"utility_regret={summary.metrics.get('utility_regret')}"
    )
    if bool(args.json):
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
