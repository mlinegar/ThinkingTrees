#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TargetedCommand:
    lane: str
    slice_name: str
    objective_family: str
    structural_arm: str
    cmd: str
    json_summary: str


def _cmd(
    *,
    runner: str,
    json_summary: Path,
    oracle_profile: str,
    objective_family: str,
    structural_arm: str,
    train_docs: int,
    test_docs: int,
    seed: int,
    n_epochs: int,
    batch_size: int,
    hidden_dim: int,
    fixed_leaf_tokens: int,
    n_binary_leaves: int,
    leaf_label_rate: float,
    internal_label_rate: float,
    root_query_rate: float,
    pairwise_prefs_per_doc: int,
    group_pref_groups_per_doc: int,
    group_size: int,
    ppo_rollouts_per_doc: int,
    lr: float,
    ppo_kl_weight: float,
    entropy_weight: float,
    ppo_advantage_center: bool,
    ppo_advantage_normalize: bool,
    ppo_reward_baseline: str,
    ppo_clip_epsilon: float,
    use_cuda: bool,
) -> str:
    parts = [
        "source venv/bin/activate &&",
        "python",
        runner,
        "--oracle-profile",
        str(oracle_profile),
        "--objective-family",
        str(objective_family),
        "--structural-arm",
        str(structural_arm),
        "--train-docs",
        str(int(train_docs)),
        "--test-docs",
        str(int(test_docs)),
        "--seed",
        str(int(seed)),
        "--n-epochs",
        str(int(n_epochs)),
        "--batch-size",
        str(int(batch_size)),
        "--hidden-dim",
        str(int(hidden_dim)),
        "--fixed-leaf-tokens",
        str(int(fixed_leaf_tokens)),
        "--n-binary-leaves",
        str(int(n_binary_leaves)),
        "--leaf-label-rate",
        str(float(leaf_label_rate)),
        "--internal-label-rate",
        str(float(internal_label_rate)),
        "--root-query-rate",
        str(float(root_query_rate)),
        "--pairwise-prefs-per-doc",
        str(int(pairwise_prefs_per_doc)),
        "--group-pref-groups-per-doc",
        str(int(group_pref_groups_per_doc)),
        "--group-size",
        str(int(group_size)),
        "--ppo-rollouts-per-doc",
        str(int(ppo_rollouts_per_doc)),
        "--lr",
        str(float(lr)),
        "--ppo-kl-weight",
        str(float(ppo_kl_weight)),
        "--entropy-weight",
        str(float(entropy_weight)),
        "--ppo-reward-baseline",
        str(ppo_reward_baseline),
        "--ppo-clip-epsilon",
        str(float(ppo_clip_epsilon)),
        f"--{'ppo-advantage-center' if ppo_advantage_center else 'no-ppo-advantage-center'}",
        f"--{'ppo-advantage-normalize' if ppo_advantage_normalize else 'no-ppo-advantage-normalize'}",
    ]
    if use_cuda:
        parts.append("--use-cuda")
    parts.extend(["--json-summary", str(json_summary)])
    return " ".join(parts)


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ppo_and_hybrid_specs(root: Path, seeds: Iterable[int]) -> List[TargetedCommand]:
    runner = "scripts/run_nonseparable_treepo_preference.py"
    specs: List[TargetedCommand] = []
    objective_bundles = [
        ("ppo", dict(leaf=0.0, internal=0.0, root=0.0)),
        ("hybrid_supervised_plus_ppo", dict(leaf=1.0, internal=1.0, root=0.0)),
    ]
    arms = [
        "oracle_exact",
        "tree_exact_supported",
        "tree_neural_supported",
        "flat_equal_info",
        "flat_span_equal_info",
        "tree_undersupported",
        "one_leaf_control",
    ]
    for objective_family, support in objective_bundles:
        for train_docs in (128, 512, 2048, 8192):
            for n_binary_leaves in (2, 4, 8):
                for ppo_rollouts in (16, 64, 128, 256, 512):
                    for hidden_dim in (64, 128):
                        for lr in (1e-3, 3e-3):
                            for ppo_kl_weight in (0.0, 0.02, 0.05):
                                for entropy_weight in (0.0, 0.01):
                                    for n_epochs in (25, 50):
                                        for seed in seeds:
                                            for structural_arm in arms:
                                                out = (
                                                    root
                                                    / "nonseparable"
                                                    / "dgp2_boundary_interaction"
                                                    / "structural_matrix"
                                                    / objective_family
                                                    / structural_arm
                                                    / (
                                                        f"train_{train_docs}__nleaves_{n_binary_leaves}"
                                                        f"__ppo_{ppo_rollouts}__hd_{hidden_dim}"
                                                        f"__lr_{lr:g}__kl_{ppo_kl_weight:g}__ent_{entropy_weight:g}"
                                                        f"__ep_{n_epochs}"
                                                    )
                                                    / f"seed_{seed}.json"
                                                )
                                                specs.append(
                                                    TargetedCommand(
                                                        lane="nonseparable",
                                                        slice_name="structural_matrix",
                                                        objective_family=objective_family,
                                                        structural_arm=structural_arm,
                                                        cmd=_cmd(
                                                            runner=runner,
                                                            json_summary=out,
                                                            oracle_profile="dgp2_boundary_interaction",
                                                            objective_family=objective_family,
                                                            structural_arm=structural_arm,
                                                            train_docs=train_docs,
                                                            test_docs=512,
                                                            seed=seed,
                                                            n_epochs=n_epochs,
                                                            batch_size=64,
                                                            hidden_dim=hidden_dim,
                                                            fixed_leaf_tokens=1,
                                                            n_binary_leaves=n_binary_leaves,
                                                            leaf_label_rate=float(support["leaf"]),
                                                            internal_label_rate=float(support["internal"]),
                                                            root_query_rate=float(support["root"]),
                                                            pairwise_prefs_per_doc=0,
                                                            group_pref_groups_per_doc=0,
                                                            group_size=4,
                                                            ppo_rollouts_per_doc=ppo_rollouts,
                                                            lr=lr,
                                                            ppo_kl_weight=ppo_kl_weight,
                                                            entropy_weight=entropy_weight,
                                                            ppo_advantage_center=True,
                                                            ppo_advantage_normalize=True,
                                                            ppo_reward_baseline="mean_reward",
                                                            ppo_clip_epsilon=0.2,
                                                            use_cuda=True,
                                                        ),
                                                        json_summary=str(out),
                                                    )
                                                )
    return specs


def _anchor_specs(root: Path, seeds: Iterable[int]) -> List[TargetedCommand]:
    runner = "scripts/run_nonseparable_treepo_preference.py"
    specs: List[TargetedCommand] = []
    objective_bundles = [
        ("dpo", dict(pairwise=32, groups=0, ppo=0, leaf=0.0, internal=0.0, root=0.0)),
        ("grpo", dict(pairwise=0, groups=16, ppo=0, leaf=0.0, internal=0.0, root=0.0)),
        ("ppo", dict(pairwise=0, groups=0, ppo=512, leaf=0.0, internal=0.0, root=0.0)),
        ("hybrid_supervised_plus_ppo", dict(pairwise=0, groups=0, ppo=512, leaf=1.0, internal=1.0, root=0.0)),
    ]
    arms = [
        "oracle_exact",
        "tree_exact_supported",
        "tree_neural_supported",
        "flat_equal_info",
        "flat_span_equal_info",
        "tree_undersupported",
        "one_leaf_control",
    ]
    for objective_family, bundle in objective_bundles:
        for train_docs in (2048, 8192):
            for n_binary_leaves in (2, 4, 8):
                for seed in seeds:
                    for structural_arm in arms:
                        out = (
                            root
                            / "nonseparable"
                            / "dgp2_boundary_interaction"
                            / "objective_family_high_support"
                            / objective_family
                            / structural_arm
                            / f"train_{train_docs}__nleaves_{n_binary_leaves}"
                            / f"seed_{seed}.json"
                        )
                        specs.append(
                            TargetedCommand(
                                lane="nonseparable",
                                slice_name="objective_family_high_support",
                                objective_family=objective_family,
                                structural_arm=structural_arm,
                                cmd=_cmd(
                                    runner=runner,
                                    json_summary=out,
                                    oracle_profile="dgp2_boundary_interaction",
                                    objective_family=objective_family,
                                    structural_arm=structural_arm,
                                    train_docs=train_docs,
                                    test_docs=512,
                                    seed=seed,
                                    n_epochs=50 if objective_family in {"ppo", "hybrid_supervised_plus_ppo"} else 25,
                                    batch_size=64,
                                    hidden_dim=128 if objective_family in {"ppo", "hybrid_supervised_plus_ppo"} else 64,
                                    fixed_leaf_tokens=1,
                                    n_binary_leaves=n_binary_leaves,
                                    leaf_label_rate=float(bundle["leaf"]),
                                    internal_label_rate=float(bundle["internal"]),
                                    root_query_rate=float(bundle["root"]),
                                    pairwise_prefs_per_doc=int(bundle["pairwise"]),
                                    group_pref_groups_per_doc=int(bundle["groups"]),
                                    group_size=4,
                                    ppo_rollouts_per_doc=int(bundle["ppo"]),
                                    lr=1e-3 if objective_family in {"ppo", "hybrid_supervised_plus_ppo"} else 3e-3,
                                    ppo_kl_weight=0.02,
                                    entropy_weight=0.01,
                                    ppo_advantage_center=True,
                                    ppo_advantage_normalize=True,
                                    ppo_reward_baseline="mean_reward",
                                    ppo_clip_epsilon=0.2,
                                    use_cuda=True,
                                ),
                                json_summary=str(out),
                            )
                        )
    return specs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build targeted round-2 exact utility transport simulations.")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--gpu-cmd-file", type=Path, default=None)
    p.add_argument("--gpu-manifest", type=Path, default=None)
    p.add_argument("--matrix-md", type=Path, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    gpu_cmd_file = args.gpu_cmd_file or (args.output_root / "cmds" / "gpu_cmds.txt")
    gpu_manifest = args.gpu_manifest or (args.output_root / "cmds" / "gpu_manifest.jsonl")
    matrix_md = args.matrix_md or (args.output_root / "matrix.md")
    gpu_specs = _ppo_and_hybrid_specs(args.output_root, args.seeds) + _anchor_specs(args.output_root, args.seeds)
    _write_lines(gpu_cmd_file, (spec.cmd for spec in gpu_specs))
    gpu_manifest.parent.mkdir(parents=True, exist_ok=True)
    with gpu_manifest.open("w", encoding="utf-8") as handle:
        for spec in gpu_specs:
            handle.write(json.dumps(asdict(spec), sort_keys=True) + "\n")
    matrix_md.parent.mkdir(parents=True, exist_ok=True)
    matrix_md.write_text(
        "\n".join(
            [
                "# Exact Utility Transport Targeted Round 2",
                "",
                f"- GPU jobs: `{len(gpu_specs)}`",
                "- Lane: `nonseparable / dgp2_boundary_interaction`",
                "- Goals:",
                "  - remove the remaining PPO tree-relevance hard fail",
                "  - add a matched `flat_span_equal_info` control for local-supervision fairness",
                "",
                "## Blocks",
                "",
                "- `structural_matrix`: PPO and hybrid-PPO optimizer/budget sweep",
                "- `objective_family_high_support`: DPO, GRPO, PPO, and hybrid-PPO anchors",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote_gpu_cmds | {gpu_cmd_file} | n={len(gpu_specs)}")
    print(f"wrote_gpu_manifest | {gpu_manifest}")
    print(f"wrote_matrix_md | {matrix_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
