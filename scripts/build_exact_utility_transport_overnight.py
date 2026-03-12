#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]

MARKOV_RUNNER = "scripts/run_markov_treepo_preference.py"
NONSEPARABLE_RUNNER = "scripts/run_nonseparable_treepo_preference.py"
BOUNDARY_TOPIC_RUNNER = "scripts/run_boundary_topic_treepo_preference.py"


@dataclass(frozen=True)
class CommandSpec:
    device_class: str
    lane: str
    oracle_profile: str
    slice_name: str
    objective_family: str
    structural_arm: str
    train_docs: int
    test_docs: int
    seed: int
    support_tag: str
    json_summary: str
    cmd: str
    fixed_leaf_tokens: Optional[int] = None
    n_binary_leaves: Optional[int] = None
    n_leaves: Optional[int] = None
    leaf_label_rate: Optional[float] = None
    internal_label_rate: Optional[float] = None
    root_query_rate: Optional[float] = None
    pairwise_prefs_per_doc: Optional[int] = None
    group_pref_groups_per_doc: Optional[int] = None
    group_size: Optional[int] = None
    ppo_rollouts_per_doc: Optional[int] = None
    n_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    hidden_dim: Optional[int] = None


def _safe_float_tag(value: float) -> str:
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p").replace("-", "m")


def _support_tag(
    *,
    leaf: float = 0.0,
    internal: float = 0.0,
    root: float = 0.0,
    pairwise: int = 0,
    groups: int = 0,
    group_size: int = 0,
    ppo: int = 0,
    label: Optional[str] = None,
) -> str:
    if label:
        return str(label)
    parts: List[str] = []
    if leaf > 0.0:
        parts.append(f"leaf{_safe_float_tag(leaf)}")
    if internal > 0.0:
        parts.append(f"internal{_safe_float_tag(internal)}")
    if root > 0.0:
        parts.append(f"root{_safe_float_tag(root)}")
    if pairwise > 0:
        parts.append(f"dpo{int(pairwise)}")
    if groups > 0:
        parts.append(f"grpo{int(groups)}x{int(group_size)}")
    if ppo > 0:
        parts.append(f"ppo{int(ppo)}")
    if not parts:
        return "zero_support"
    return "__".join(parts)


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
    fixed_leaf_tokens: Optional[int] = None,
    n_binary_leaves: Optional[int] = None,
    n_leaves: Optional[int] = None,
    leaf_label_rate: float = 0.0,
    internal_label_rate: float = 0.0,
    root_query_rate: float = 0.0,
    pairwise_prefs_per_doc: int = 0,
    group_pref_groups_per_doc: int = 0,
    group_size: int = 4,
    ppo_rollouts_per_doc: int = 0,
    use_cuda: bool = False,
) -> str:
    pieces = [
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
    ]
    if fixed_leaf_tokens is not None:
        pieces.extend(["--fixed-leaf-tokens", str(int(fixed_leaf_tokens))])
    if n_binary_leaves is not None:
        pieces.extend(["--n-binary-leaves", str(int(n_binary_leaves))])
    if n_leaves is not None:
        pieces.extend(["--n-leaves", str(int(n_leaves))])
    if use_cuda:
        pieces.append("--use-cuda")
    pieces.extend(["--json-summary", str(json_summary)])
    return " ".join(pieces)


def _markov_main_specs(root: Path, seeds: Iterable[int]) -> List[CommandSpec]:
    specs: List[CommandSpec] = []
    configs = [
        {
            "oracle_profile": "markov_count_endpoints",
            "train_docs": [128, 512, 2048],
            "fixed_leaf_tokens": [8, 16, 32],
        },
        {
            "oracle_profile": "markov_count_only",
            "train_docs": [512, 2048],
            "fixed_leaf_tokens": [16, 32],
        },
    ]

    for cfg in configs:
        profile = str(cfg["oracle_profile"])
        for train_docs in list(cfg["train_docs"]):
            for fixed_leaf_tokens in list(cfg["fixed_leaf_tokens"]):
                for seed in seeds:
                    for internal_rate in (0.05, 0.25, 0.5, 1.0):
                        tag = _support_tag(internal=internal_rate, label=f"internal_only_{_safe_float_tag(internal_rate)}")
                        out = (
                            root
                            / "markov"
                            / profile
                            / "support_curves_local"
                            / "supervised_state"
                            / "tree_neural_supported"
                            / f"train_{train_docs}__leaf_{fixed_leaf_tokens}"
                            / tag
                            / f"seed_{seed}.json"
                        )
                        specs.append(
                            CommandSpec(
                                device_class="gpu",
                                lane="markov",
                                oracle_profile=profile,
                                slice_name="support_curves_local",
                                objective_family="supervised_state",
                                structural_arm="tree_neural_supported",
                                train_docs=train_docs,
                                test_docs=256,
                                seed=int(seed),
                                fixed_leaf_tokens=int(fixed_leaf_tokens),
                                leaf_label_rate=0.0,
                                internal_label_rate=float(internal_rate),
                                root_query_rate=0.0,
                                pairwise_prefs_per_doc=0,
                                group_pref_groups_per_doc=0,
                                group_size=4,
                                ppo_rollouts_per_doc=0,
                                support_tag=tag,
                                n_epochs=25,
                                batch_size=32,
                                hidden_dim=96,
                                json_summary=str(out),
                                cmd=_cmd(
                                    runner=MARKOV_RUNNER,
                                    json_summary=out,
                                    oracle_profile=profile,
                                    objective_family="supervised_state",
                                    structural_arm="tree_neural_supported",
                                    train_docs=train_docs,
                                    test_docs=256,
                                    seed=seed,
                                    n_epochs=25,
                                    batch_size=32,
                                    hidden_dim=96,
                                    fixed_leaf_tokens=fixed_leaf_tokens,
                                    leaf_label_rate=0.0,
                                    internal_label_rate=internal_rate,
                                    root_query_rate=0.0,
                                    pairwise_prefs_per_doc=0,
                                    group_pref_groups_per_doc=0,
                                    group_size=4,
                                    ppo_rollouts_per_doc=0,
                                    use_cuda=True,
                                ),
                            )
                        )
                    for full_rate in (0.05, 0.25, 0.5, 1.0):
                        tag = _support_tag(leaf=full_rate, internal=full_rate, label=f"full_local_{_safe_float_tag(full_rate)}")
                        out = (
                            root
                            / "markov"
                            / profile
                            / "support_curves_local"
                            / "supervised_state"
                            / "tree_neural_supported"
                            / f"train_{train_docs}__leaf_{fixed_leaf_tokens}"
                            / tag
                            / f"seed_{seed}.json"
                        )
                        specs.append(
                            CommandSpec(
                                device_class="gpu",
                                lane="markov",
                                oracle_profile=profile,
                                slice_name="support_curves_local",
                                objective_family="supervised_state",
                                structural_arm="tree_neural_supported",
                                train_docs=train_docs,
                                test_docs=256,
                                seed=int(seed),
                                fixed_leaf_tokens=int(fixed_leaf_tokens),
                                leaf_label_rate=float(full_rate),
                                internal_label_rate=float(full_rate),
                                root_query_rate=0.0,
                                pairwise_prefs_per_doc=0,
                                group_pref_groups_per_doc=0,
                                group_size=4,
                                ppo_rollouts_per_doc=0,
                                support_tag=tag,
                                n_epochs=25,
                                batch_size=32,
                                hidden_dim=96,
                                json_summary=str(out),
                                cmd=_cmd(
                                    runner=MARKOV_RUNNER,
                                    json_summary=out,
                                    oracle_profile=profile,
                                    objective_family="supervised_state",
                                    structural_arm="tree_neural_supported",
                                    train_docs=train_docs,
                                    test_docs=256,
                                    seed=seed,
                                    n_epochs=25,
                                    batch_size=32,
                                    hidden_dim=96,
                                    fixed_leaf_tokens=fixed_leaf_tokens,
                                    leaf_label_rate=full_rate,
                                    internal_label_rate=full_rate,
                                    root_query_rate=0.0,
                                    pairwise_prefs_per_doc=0,
                                    group_pref_groups_per_doc=0,
                                    group_size=4,
                                    ppo_rollouts_per_doc=0,
                                    use_cuda=True,
                                ),
                            )
                        )

    pref_configs = [
        ("dpo", {"pairwise": [1, 4, 16, 32]}),
        ("grpo", {"groups": [1, 4, 16], "group_size": 4}),
        ("ppo", {"ppo": [1, 4, 16]}),
        ("hybrid_supervised_plus_dpo", {"bundles": [(0.05, 1), (0.25, 4), (0.5, 16), (1.0, 32)]}),
    ]
    for objective_family, pref_cfg in pref_configs:
        profile_sets = [
            ("markov_count_endpoints", [128, 512, 2048], [8, 16, 32]),
            ("markov_count_only", [512], [16, 32]),
        ]
        for profile, train_grid, leaf_grid in profile_sets:
            for train_docs in train_grid:
                for fixed_leaf_tokens in leaf_grid:
                    for seed in seeds:
                        if objective_family == "dpo":
                            for pairwise in pref_cfg["pairwise"]:
                                tag = _support_tag(pairwise=pairwise)
                                out = (
                                    root
                                    / "markov"
                                    / profile
                                    / "support_curves_preferences"
                                    / objective_family
                                    / "tree_neural_supported"
                                    / f"train_{train_docs}__leaf_{fixed_leaf_tokens}"
                                    / tag
                                    / f"seed_{seed}.json"
                                )
                                specs.append(
                                    CommandSpec(
                                        device_class="gpu",
                                        lane="markov",
                                        oracle_profile=profile,
                                        slice_name="support_curves_preferences",
                                        objective_family=objective_family,
                                        structural_arm="tree_neural_supported",
                                        train_docs=train_docs,
                                        test_docs=256,
                                        seed=int(seed),
                                        fixed_leaf_tokens=int(fixed_leaf_tokens),
                                        leaf_label_rate=0.0,
                                        internal_label_rate=0.0,
                                        root_query_rate=0.0,
                                        pairwise_prefs_per_doc=int(pairwise),
                                        group_pref_groups_per_doc=0,
                                        group_size=4,
                                        ppo_rollouts_per_doc=0,
                                        support_tag=tag,
                                        n_epochs=30,
                                        batch_size=32,
                                        hidden_dim=96,
                                        json_summary=str(out),
                                        cmd=_cmd(
                                            runner=MARKOV_RUNNER,
                                            json_summary=out,
                                            oracle_profile=profile,
                                            objective_family=objective_family,
                                            structural_arm="tree_neural_supported",
                                            train_docs=train_docs,
                                            test_docs=256,
                                            seed=seed,
                                            n_epochs=30,
                                            batch_size=32,
                                            hidden_dim=96,
                                            fixed_leaf_tokens=fixed_leaf_tokens,
                                            leaf_label_rate=0.0,
                                            internal_label_rate=0.0,
                                            root_query_rate=0.0,
                                            pairwise_prefs_per_doc=pairwise,
                                            group_pref_groups_per_doc=0,
                                            group_size=4,
                                            ppo_rollouts_per_doc=0,
                                            use_cuda=True,
                                        ),
                                    )
                                )
                        elif objective_family == "grpo":
                            for groups in pref_cfg["groups"]:
                                tag = _support_tag(groups=groups, group_size=int(pref_cfg["group_size"]))
                                out = (
                                    root
                                    / "markov"
                                    / profile
                                    / "support_curves_preferences"
                                    / objective_family
                                    / "tree_neural_supported"
                                    / f"train_{train_docs}__leaf_{fixed_leaf_tokens}"
                                    / tag
                                    / f"seed_{seed}.json"
                                )
                                specs.append(
                                    CommandSpec(
                                        device_class="gpu",
                                        lane="markov",
                                        oracle_profile=profile,
                                        slice_name="support_curves_preferences",
                                        objective_family=objective_family,
                                        structural_arm="tree_neural_supported",
                                        train_docs=train_docs,
                                        test_docs=256,
                                        seed=int(seed),
                                        fixed_leaf_tokens=int(fixed_leaf_tokens),
                                        leaf_label_rate=0.0,
                                        internal_label_rate=0.0,
                                        root_query_rate=0.0,
                                        pairwise_prefs_per_doc=0,
                                        group_pref_groups_per_doc=int(groups),
                                        group_size=int(pref_cfg["group_size"]),
                                        ppo_rollouts_per_doc=0,
                                        support_tag=tag,
                                        n_epochs=30,
                                        batch_size=32,
                                        hidden_dim=96,
                                        json_summary=str(out),
                                        cmd=_cmd(
                                            runner=MARKOV_RUNNER,
                                            json_summary=out,
                                            oracle_profile=profile,
                                            objective_family=objective_family,
                                            structural_arm="tree_neural_supported",
                                            train_docs=train_docs,
                                            test_docs=256,
                                            seed=seed,
                                            n_epochs=30,
                                            batch_size=32,
                                            hidden_dim=96,
                                            fixed_leaf_tokens=fixed_leaf_tokens,
                                            leaf_label_rate=0.0,
                                            internal_label_rate=0.0,
                                            root_query_rate=0.0,
                                            pairwise_prefs_per_doc=0,
                                            group_pref_groups_per_doc=groups,
                                            group_size=int(pref_cfg["group_size"]),
                                            ppo_rollouts_per_doc=0,
                                            use_cuda=True,
                                        ),
                                    )
                                )
                        elif objective_family == "ppo":
                            for rollouts in pref_cfg["ppo"]:
                                tag = _support_tag(ppo=rollouts)
                                out = (
                                    root
                                    / "markov"
                                    / profile
                                    / "support_curves_preferences"
                                    / objective_family
                                    / "tree_neural_supported"
                                    / f"train_{train_docs}__leaf_{fixed_leaf_tokens}"
                                    / tag
                                    / f"seed_{seed}.json"
                                )
                                specs.append(
                                    CommandSpec(
                                        device_class="gpu",
                                        lane="markov",
                                        oracle_profile=profile,
                                        slice_name="support_curves_preferences",
                                        objective_family=objective_family,
                                        structural_arm="tree_neural_supported",
                                        train_docs=train_docs,
                                        test_docs=256,
                                        seed=int(seed),
                                        fixed_leaf_tokens=int(fixed_leaf_tokens),
                                        leaf_label_rate=0.0,
                                        internal_label_rate=0.0,
                                        root_query_rate=0.0,
                                        pairwise_prefs_per_doc=0,
                                        group_pref_groups_per_doc=0,
                                        group_size=4,
                                        ppo_rollouts_per_doc=int(rollouts),
                                        support_tag=tag,
                                        n_epochs=30,
                                        batch_size=32,
                                        hidden_dim=96,
                                        json_summary=str(out),
                                        cmd=_cmd(
                                            runner=MARKOV_RUNNER,
                                            json_summary=out,
                                            oracle_profile=profile,
                                            objective_family=objective_family,
                                            structural_arm="tree_neural_supported",
                                            train_docs=train_docs,
                                            test_docs=256,
                                            seed=seed,
                                            n_epochs=30,
                                            batch_size=32,
                                            hidden_dim=96,
                                            fixed_leaf_tokens=fixed_leaf_tokens,
                                            leaf_label_rate=0.0,
                                            internal_label_rate=0.0,
                                            root_query_rate=0.0,
                                            pairwise_prefs_per_doc=0,
                                            group_pref_groups_per_doc=0,
                                            group_size=4,
                                            ppo_rollouts_per_doc=rollouts,
                                            use_cuda=True,
                                        ),
                                    )
                                )
                        else:
                            for local_rate, pairwise in pref_cfg["bundles"]:
                                tag = _support_tag(leaf=local_rate, internal=local_rate, pairwise=pairwise, label=f"hybrid_local_{_safe_float_tag(local_rate)}__dpo{pairwise}")
                                out = (
                                    root
                                    / "markov"
                                    / profile
                                    / "support_curves_preferences"
                                    / objective_family
                                    / "tree_neural_supported"
                                    / f"train_{train_docs}__leaf_{fixed_leaf_tokens}"
                                    / tag
                                    / f"seed_{seed}.json"
                                )
                                specs.append(
                                    CommandSpec(
                                        device_class="gpu",
                                        lane="markov",
                                        oracle_profile=profile,
                                        slice_name="support_curves_preferences",
                                        objective_family=objective_family,
                                        structural_arm="tree_neural_supported",
                                        train_docs=train_docs,
                                        test_docs=256,
                                        seed=int(seed),
                                        fixed_leaf_tokens=int(fixed_leaf_tokens),
                                        leaf_label_rate=float(local_rate),
                                        internal_label_rate=float(local_rate),
                                        root_query_rate=0.0,
                                        pairwise_prefs_per_doc=int(pairwise),
                                        group_pref_groups_per_doc=0,
                                        group_size=4,
                                        ppo_rollouts_per_doc=0,
                                        support_tag=tag,
                                        n_epochs=30,
                                        batch_size=32,
                                        hidden_dim=96,
                                        json_summary=str(out),
                                        cmd=_cmd(
                                            runner=MARKOV_RUNNER,
                                            json_summary=out,
                                            oracle_profile=profile,
                                            objective_family=objective_family,
                                            structural_arm="tree_neural_supported",
                                            train_docs=train_docs,
                                            test_docs=256,
                                            seed=seed,
                                            n_epochs=30,
                                            batch_size=32,
                                            hidden_dim=96,
                                            fixed_leaf_tokens=fixed_leaf_tokens,
                                            leaf_label_rate=local_rate,
                                            internal_label_rate=local_rate,
                                            root_query_rate=0.0,
                                            pairwise_prefs_per_doc=pairwise,
                                            group_pref_groups_per_doc=0,
                                            group_size=4,
                                            ppo_rollouts_per_doc=0,
                                            use_cuda=True,
                                        ),
                                    )
                                )

    for profile in ("markov_count_endpoints", "markov_count_only"):
        for train_docs in (512, 2048):
            for fixed_leaf_tokens in (8, 16, 32):
                for seed in seeds:
                    objective_bundles = [
                        ("supervised_state", dict(leaf_label_rate=1.0, internal_label_rate=1.0, root_query_rate=0.0, pairwise_prefs_per_doc=0, group_pref_groups_per_doc=0, group_size=4, ppo_rollouts_per_doc=0)),
                        ("supervised_root", dict(leaf_label_rate=0.0, internal_label_rate=0.0, root_query_rate=1.0, pairwise_prefs_per_doc=0, group_pref_groups_per_doc=0, group_size=4, ppo_rollouts_per_doc=0)),
                        ("dpo", dict(leaf_label_rate=0.0, internal_label_rate=0.0, root_query_rate=0.0, pairwise_prefs_per_doc=32, group_pref_groups_per_doc=0, group_size=4, ppo_rollouts_per_doc=0)),
                        ("grpo", dict(leaf_label_rate=0.0, internal_label_rate=0.0, root_query_rate=0.0, pairwise_prefs_per_doc=0, group_pref_groups_per_doc=16, group_size=4, ppo_rollouts_per_doc=0)),
                        ("ppo", dict(leaf_label_rate=0.0, internal_label_rate=0.0, root_query_rate=0.0, pairwise_prefs_per_doc=0, group_pref_groups_per_doc=0, group_size=4, ppo_rollouts_per_doc=16)),
                        ("hybrid_supervised_plus_dpo", dict(leaf_label_rate=1.0, internal_label_rate=1.0, root_query_rate=0.0, pairwise_prefs_per_doc=32, group_pref_groups_per_doc=0, group_size=4, ppo_rollouts_per_doc=0)),
                        ("hybrid_supervised_plus_grpo", dict(leaf_label_rate=1.0, internal_label_rate=1.0, root_query_rate=0.0, pairwise_prefs_per_doc=0, group_pref_groups_per_doc=16, group_size=4, ppo_rollouts_per_doc=0)),
                        ("hybrid_supervised_plus_ppo", dict(leaf_label_rate=1.0, internal_label_rate=1.0, root_query_rate=0.0, pairwise_prefs_per_doc=0, group_pref_groups_per_doc=0, group_size=4, ppo_rollouts_per_doc=16)),
                    ]
                    for objective_family, bundle in objective_bundles:
                        tag = _support_tag(
                            leaf=float(bundle["leaf_label_rate"]),
                            internal=float(bundle["internal_label_rate"]),
                            root=float(bundle["root_query_rate"]),
                            pairwise=int(bundle["pairwise_prefs_per_doc"]),
                            groups=int(bundle["group_pref_groups_per_doc"]),
                            group_size=int(bundle["group_size"]),
                            ppo=int(bundle["ppo_rollouts_per_doc"]),
                            label="high_support_anchor",
                        )
                        out = (
                            root
                            / "markov"
                            / profile
                            / "objective_family_high_support"
                            / objective_family
                            / "tree_neural_supported"
                            / f"train_{train_docs}__leaf_{fixed_leaf_tokens}"
                            / tag
                            / f"seed_{seed}.json"
                        )
                        specs.append(
                            CommandSpec(
                                device_class="gpu",
                                lane="markov",
                                oracle_profile=profile,
                                slice_name="objective_family_high_support",
                                objective_family=objective_family,
                                structural_arm="tree_neural_supported",
                                train_docs=train_docs,
                                test_docs=256,
                                seed=int(seed),
                                fixed_leaf_tokens=int(fixed_leaf_tokens),
                                leaf_label_rate=float(bundle["leaf_label_rate"]),
                                internal_label_rate=float(bundle["internal_label_rate"]),
                                root_query_rate=float(bundle["root_query_rate"]),
                                pairwise_prefs_per_doc=int(bundle["pairwise_prefs_per_doc"]),
                                group_pref_groups_per_doc=int(bundle["group_pref_groups_per_doc"]),
                                group_size=int(bundle["group_size"]),
                                ppo_rollouts_per_doc=int(bundle["ppo_rollouts_per_doc"]),
                                support_tag=tag,
                                n_epochs=30,
                                batch_size=32,
                                hidden_dim=96,
                                json_summary=str(out),
                                cmd=_cmd(
                                    runner=MARKOV_RUNNER,
                                    json_summary=out,
                                    oracle_profile=profile,
                                    objective_family=objective_family,
                                    structural_arm="tree_neural_supported",
                                    train_docs=train_docs,
                                    test_docs=256,
                                    seed=seed,
                                    n_epochs=30,
                                    batch_size=32,
                                    hidden_dim=96,
                                    fixed_leaf_tokens=fixed_leaf_tokens,
                                    leaf_label_rate=float(bundle["leaf_label_rate"]),
                                    internal_label_rate=float(bundle["internal_label_rate"]),
                                    root_query_rate=float(bundle["root_query_rate"]),
                                    pairwise_prefs_per_doc=int(bundle["pairwise_prefs_per_doc"]),
                                    group_pref_groups_per_doc=int(bundle["group_pref_groups_per_doc"]),
                                    group_size=int(bundle["group_size"]),
                                    ppo_rollouts_per_doc=int(bundle["ppo_rollouts_per_doc"]),
                                    use_cuda=True,
                                ),
                            )
                        )
    return specs


def _boundary_topic_main_specs(root: Path, seeds: Iterable[int]) -> List[CommandSpec]:
    specs: List[CommandSpec] = []
    profile_sets = [
        ("topic_plus_boundary", [128, 512, 2048], [2, 6, 10], list(seeds)),
        ("topic_mass_only", [512, 2048], [2, 6, 10], [0]),
    ]
    for profile, train_grid, n_leaves_grid, seed_grid in profile_sets:
        for train_docs in train_grid:
            for n_leaves in n_leaves_grid:
                for seed in seed_grid:
                    for full_rate in (0.05, 0.25, 0.5, 1.0):
                        tag = _support_tag(leaf=full_rate, internal=full_rate, label=f"full_local_{_safe_float_tag(full_rate)}")
                        out = (
                            root
                            / "boundary_topic"
                            / profile
                            / "support_curves_local"
                            / "supervised_state"
                            / "tree_neural_supported"
                            / f"train_{train_docs}__leaves_{n_leaves}"
                            / tag
                            / f"seed_{seed}.json"
                        )
                        specs.append(
                            CommandSpec(
                                device_class="gpu",
                                lane="boundary_topic",
                                oracle_profile=profile,
                                slice_name="support_curves_local",
                                objective_family="supervised_state",
                                structural_arm="tree_neural_supported",
                                train_docs=train_docs,
                                test_docs=256,
                                seed=int(seed),
                                n_leaves=int(n_leaves),
                                fixed_leaf_tokens=1,
                                leaf_label_rate=float(full_rate),
                                internal_label_rate=float(full_rate),
                                root_query_rate=0.0,
                                pairwise_prefs_per_doc=0,
                                group_pref_groups_per_doc=0,
                                group_size=4,
                                ppo_rollouts_per_doc=0,
                                support_tag=tag,
                                n_epochs=25,
                                batch_size=32,
                                hidden_dim=96,
                                json_summary=str(out),
                                cmd=_cmd(
                                    runner=BOUNDARY_TOPIC_RUNNER,
                                    json_summary=out,
                                    oracle_profile=profile,
                                    objective_family="supervised_state",
                                    structural_arm="tree_neural_supported",
                                    train_docs=train_docs,
                                    test_docs=256,
                                    seed=seed,
                                    n_epochs=25,
                                    batch_size=32,
                                    hidden_dim=96,
                                    fixed_leaf_tokens=1,
                                    n_leaves=n_leaves,
                                    leaf_label_rate=full_rate,
                                    internal_label_rate=full_rate,
                                    root_query_rate=0.0,
                                    pairwise_prefs_per_doc=0,
                                    group_pref_groups_per_doc=0,
                                    group_size=4,
                                    ppo_rollouts_per_doc=0,
                                    use_cuda=True,
                                ),
                            )
                        )

    for profile in ("topic_plus_boundary", "topic_mass_only"):
        for train_docs in (512, 2048):
            for n_leaves in (2, 6, 10):
                for seed in ([0, 1] if profile == "topic_plus_boundary" else [0]):
                    objective_bundles = [
                        ("supervised_state", dict(leaf_label_rate=1.0, internal_label_rate=1.0, root_query_rate=0.0, pairwise_prefs_per_doc=0, group_pref_groups_per_doc=0, group_size=4, ppo_rollouts_per_doc=0)),
                        ("supervised_root", dict(leaf_label_rate=0.0, internal_label_rate=0.0, root_query_rate=1.0, pairwise_prefs_per_doc=0, group_pref_groups_per_doc=0, group_size=4, ppo_rollouts_per_doc=0)),
                        ("dpo", dict(leaf_label_rate=0.0, internal_label_rate=0.0, root_query_rate=0.0, pairwise_prefs_per_doc=32, group_pref_groups_per_doc=0, group_size=4, ppo_rollouts_per_doc=0)),
                        ("grpo", dict(leaf_label_rate=0.0, internal_label_rate=0.0, root_query_rate=0.0, pairwise_prefs_per_doc=0, group_pref_groups_per_doc=16, group_size=4, ppo_rollouts_per_doc=0)),
                        ("ppo", dict(leaf_label_rate=0.0, internal_label_rate=0.0, root_query_rate=0.0, pairwise_prefs_per_doc=0, group_pref_groups_per_doc=0, group_size=4, ppo_rollouts_per_doc=16)),
                        ("hybrid_supervised_plus_dpo", dict(leaf_label_rate=1.0, internal_label_rate=1.0, root_query_rate=0.0, pairwise_prefs_per_doc=32, group_pref_groups_per_doc=0, group_size=4, ppo_rollouts_per_doc=0)),
                        ("hybrid_supervised_plus_grpo", dict(leaf_label_rate=1.0, internal_label_rate=1.0, root_query_rate=0.0, pairwise_prefs_per_doc=0, group_pref_groups_per_doc=16, group_size=4, ppo_rollouts_per_doc=0)),
                        ("hybrid_supervised_plus_ppo", dict(leaf_label_rate=1.0, internal_label_rate=1.0, root_query_rate=0.0, pairwise_prefs_per_doc=0, group_pref_groups_per_doc=0, group_size=4, ppo_rollouts_per_doc=16)),
                    ]
                    for objective_family, bundle in objective_bundles:
                        tag = _support_tag(
                            leaf=float(bundle["leaf_label_rate"]),
                            internal=float(bundle["internal_label_rate"]),
                            root=float(bundle["root_query_rate"]),
                            pairwise=int(bundle["pairwise_prefs_per_doc"]),
                            groups=int(bundle["group_pref_groups_per_doc"]),
                            group_size=int(bundle["group_size"]),
                            ppo=int(bundle["ppo_rollouts_per_doc"]),
                            label="high_support_anchor",
                        )
                        out = (
                            root
                            / "boundary_topic"
                            / profile
                            / "objective_family_high_support"
                            / objective_family
                            / "tree_neural_supported"
                            / f"train_{train_docs}__leaves_{n_leaves}"
                            / tag
                            / f"seed_{seed}.json"
                        )
                        specs.append(
                            CommandSpec(
                                device_class="gpu",
                                lane="boundary_topic",
                                oracle_profile=profile,
                                slice_name="objective_family_high_support",
                                objective_family=objective_family,
                                structural_arm="tree_neural_supported",
                                train_docs=train_docs,
                                test_docs=256,
                                seed=int(seed),
                                n_leaves=int(n_leaves),
                                fixed_leaf_tokens=1,
                                leaf_label_rate=float(bundle["leaf_label_rate"]),
                                internal_label_rate=float(bundle["internal_label_rate"]),
                                root_query_rate=float(bundle["root_query_rate"]),
                                pairwise_prefs_per_doc=int(bundle["pairwise_prefs_per_doc"]),
                                group_pref_groups_per_doc=int(bundle["group_pref_groups_per_doc"]),
                                group_size=int(bundle["group_size"]),
                                ppo_rollouts_per_doc=int(bundle["ppo_rollouts_per_doc"]),
                                support_tag=tag,
                                n_epochs=25,
                                batch_size=32,
                                hidden_dim=96,
                                json_summary=str(out),
                                cmd=_cmd(
                                    runner=BOUNDARY_TOPIC_RUNNER,
                                    json_summary=out,
                                    oracle_profile=profile,
                                    objective_family=objective_family,
                                    structural_arm="tree_neural_supported",
                                    train_docs=train_docs,
                                    test_docs=256,
                                    seed=seed,
                                    n_epochs=25,
                                    batch_size=32,
                                    hidden_dim=96,
                                    fixed_leaf_tokens=1,
                                    n_leaves=n_leaves,
                                    leaf_label_rate=float(bundle["leaf_label_rate"]),
                                    internal_label_rate=float(bundle["internal_label_rate"]),
                                    root_query_rate=float(bundle["root_query_rate"]),
                                    pairwise_prefs_per_doc=int(bundle["pairwise_prefs_per_doc"]),
                                    group_pref_groups_per_doc=int(bundle["group_pref_groups_per_doc"]),
                                    group_size=int(bundle["group_size"]),
                                    ppo_rollouts_per_doc=int(bundle["ppo_rollouts_per_doc"]),
                                    use_cuda=True,
                                ),
                            )
                        )
                    for pairwise in (1, 4, 16):
                        tag = _support_tag(pairwise=pairwise)
                        out = (
                            root
                            / "boundary_topic"
                            / profile
                            / "support_curves_preferences"
                            / "dpo"
                            / "tree_neural_supported"
                            / f"train_{train_docs}__leaves_{n_leaves}"
                            / tag
                            / f"seed_{seed}.json"
                        )
                        specs.append(
                            CommandSpec(
                                device_class="gpu",
                                lane="boundary_topic",
                                oracle_profile=profile,
                                slice_name="support_curves_preferences",
                                objective_family="dpo",
                                structural_arm="tree_neural_supported",
                                train_docs=train_docs,
                                test_docs=256,
                                seed=int(seed),
                                n_leaves=int(n_leaves),
                                fixed_leaf_tokens=1,
                                leaf_label_rate=0.0,
                                internal_label_rate=0.0,
                                root_query_rate=0.0,
                                pairwise_prefs_per_doc=int(pairwise),
                                group_pref_groups_per_doc=0,
                                group_size=4,
                                ppo_rollouts_per_doc=0,
                                support_tag=tag,
                                n_epochs=25,
                                batch_size=32,
                                hidden_dim=96,
                                json_summary=str(out),
                                cmd=_cmd(
                                    runner=BOUNDARY_TOPIC_RUNNER,
                                    json_summary=out,
                                    oracle_profile=profile,
                                    objective_family="dpo",
                                    structural_arm="tree_neural_supported",
                                    train_docs=train_docs,
                                    test_docs=256,
                                    seed=seed,
                                    n_epochs=25,
                                    batch_size=32,
                                    hidden_dim=96,
                                    fixed_leaf_tokens=1,
                                    n_leaves=n_leaves,
                                    leaf_label_rate=0.0,
                                    internal_label_rate=0.0,
                                    root_query_rate=0.0,
                                    pairwise_prefs_per_doc=pairwise,
                                    group_pref_groups_per_doc=0,
                                    group_size=4,
                                    ppo_rollouts_per_doc=0,
                                    use_cuda=True,
                                ),
                            )
                        )
                    for rollouts in (1, 4, 16):
                        tag = _support_tag(ppo=rollouts)
                        out = (
                            root
                            / "boundary_topic"
                            / profile
                            / "support_curves_preferences"
                            / "ppo"
                            / "tree_neural_supported"
                            / f"train_{train_docs}__leaves_{n_leaves}"
                            / tag
                            / f"seed_{seed}.json"
                        )
                        specs.append(
                            CommandSpec(
                                device_class="gpu",
                                lane="boundary_topic",
                                oracle_profile=profile,
                                slice_name="support_curves_preferences",
                                objective_family="ppo",
                                structural_arm="tree_neural_supported",
                                train_docs=train_docs,
                                test_docs=256,
                                seed=int(seed),
                                n_leaves=int(n_leaves),
                                fixed_leaf_tokens=1,
                                leaf_label_rate=0.0,
                                internal_label_rate=0.0,
                                root_query_rate=0.0,
                                pairwise_prefs_per_doc=0,
                                group_pref_groups_per_doc=0,
                                group_size=4,
                                ppo_rollouts_per_doc=int(rollouts),
                                support_tag=tag,
                                n_epochs=25,
                                batch_size=32,
                                hidden_dim=96,
                                json_summary=str(out),
                                cmd=_cmd(
                                    runner=BOUNDARY_TOPIC_RUNNER,
                                    json_summary=out,
                                    oracle_profile=profile,
                                    objective_family="ppo",
                                    structural_arm="tree_neural_supported",
                                    train_docs=train_docs,
                                    test_docs=256,
                                    seed=seed,
                                    n_epochs=25,
                                    batch_size=32,
                                    hidden_dim=96,
                                    fixed_leaf_tokens=1,
                                    n_leaves=n_leaves,
                                    leaf_label_rate=0.0,
                                    internal_label_rate=0.0,
                                    root_query_rate=0.0,
                                    pairwise_prefs_per_doc=0,
                                    group_pref_groups_per_doc=0,
                                    group_size=4,
                                    ppo_rollouts_per_doc=rollouts,
                                    use_cuda=True,
                                ),
                            )
                        )
    return specs


def _cpu_control_specs(root: Path, seeds: Iterable[int]) -> List[CommandSpec]:
    specs: List[CommandSpec] = []
    # Nonseparable full structural/control matrix on CPU.
    nonsep_profiles = [
        "dgp1_complementarity_and",
        "dgp1_complementarity_control",
        "dgp2_boundary_interaction",
        "dgp2_boundary_zero",
    ]
    nonsep_arms = [
        "oracle_exact",
        "tree_exact_supported",
        "tree_neural_supported",
        "tree_undersupported",
        "flat_equal_info",
        "one_leaf_control",
        "right_rule_wrong_chunker",
    ]
    nonsep_objectives = [
        ("supervised_state", dict(leaf_label_rate=1.0, internal_label_rate=1.0, root_query_rate=0.0, pairwise_prefs_per_doc=0, group_pref_groups_per_doc=0, group_size=4, ppo_rollouts_per_doc=0)),
        ("supervised_root", dict(leaf_label_rate=0.0, internal_label_rate=0.0, root_query_rate=1.0, pairwise_prefs_per_doc=0, group_pref_groups_per_doc=0, group_size=4, ppo_rollouts_per_doc=0)),
        ("dpo", dict(leaf_label_rate=0.0, internal_label_rate=0.0, root_query_rate=0.0, pairwise_prefs_per_doc=32, group_pref_groups_per_doc=0, group_size=4, ppo_rollouts_per_doc=0)),
        ("grpo", dict(leaf_label_rate=0.0, internal_label_rate=0.0, root_query_rate=0.0, pairwise_prefs_per_doc=0, group_pref_groups_per_doc=16, group_size=4, ppo_rollouts_per_doc=0)),
        ("ppo", dict(leaf_label_rate=0.0, internal_label_rate=0.0, root_query_rate=0.0, pairwise_prefs_per_doc=0, group_pref_groups_per_doc=0, group_size=4, ppo_rollouts_per_doc=16)),
        ("hybrid_supervised_plus_dpo", dict(leaf_label_rate=1.0, internal_label_rate=1.0, root_query_rate=0.0, pairwise_prefs_per_doc=32, group_pref_groups_per_doc=0, group_size=4, ppo_rollouts_per_doc=0)),
        ("hybrid_supervised_plus_grpo", dict(leaf_label_rate=1.0, internal_label_rate=1.0, root_query_rate=0.0, pairwise_prefs_per_doc=0, group_pref_groups_per_doc=16, group_size=4, ppo_rollouts_per_doc=0)),
        ("hybrid_supervised_plus_ppo", dict(leaf_label_rate=1.0, internal_label_rate=1.0, root_query_rate=0.0, pairwise_prefs_per_doc=0, group_pref_groups_per_doc=0, group_size=4, ppo_rollouts_per_doc=16)),
    ]
    for profile in nonsep_profiles:
        for objective_family, bundle in nonsep_objectives:
            for arm in nonsep_arms:
                for train_docs in (128, 512, 2048):
                    for n_binary_leaves in (2, 4, 8):
                        for seed in seeds:
                            tag = _support_tag(
                                leaf=float(bundle["leaf_label_rate"]),
                                internal=float(bundle["internal_label_rate"]),
                                root=float(bundle["root_query_rate"]),
                                pairwise=int(bundle["pairwise_prefs_per_doc"]),
                                groups=int(bundle["group_pref_groups_per_doc"]),
                                group_size=int(bundle["group_size"]),
                                ppo=int(bundle["ppo_rollouts_per_doc"]),
                                label="high_support_anchor",
                            )
                            out = (
                                root
                                / "nonseparable"
                                / profile
                                / "structural_matrix"
                                / objective_family
                                / arm
                                / f"train_{train_docs}__nleaves_{n_binary_leaves}"
                                / tag
                                / f"seed_{seed}.json"
                            )
                            epochs = 1 if arm in {"oracle_exact", "tree_exact_supported"} else 20
                            specs.append(
                                CommandSpec(
                                    device_class="cpu",
                                    lane="nonseparable",
                                    oracle_profile=profile,
                                    slice_name="structural_matrix",
                                    objective_family=objective_family,
                                    structural_arm=arm,
                                    train_docs=train_docs,
                                    test_docs=256,
                                    seed=int(seed),
                                    n_binary_leaves=int(n_binary_leaves),
                                    fixed_leaf_tokens=1,
                                    leaf_label_rate=float(bundle["leaf_label_rate"]),
                                    internal_label_rate=float(bundle["internal_label_rate"]),
                                    root_query_rate=float(bundle["root_query_rate"]),
                                    pairwise_prefs_per_doc=int(bundle["pairwise_prefs_per_doc"]),
                                    group_pref_groups_per_doc=int(bundle["group_pref_groups_per_doc"]),
                                    group_size=int(bundle["group_size"]),
                                    ppo_rollouts_per_doc=int(bundle["ppo_rollouts_per_doc"]),
                                    support_tag=tag,
                                    n_epochs=epochs,
                                    batch_size=64,
                                    hidden_dim=64,
                                    json_summary=str(out),
                                    cmd=_cmd(
                                        runner=NONSEPARABLE_RUNNER,
                                        json_summary=out,
                                        oracle_profile=profile,
                                        objective_family=objective_family,
                                        structural_arm=arm,
                                        train_docs=train_docs,
                                        test_docs=256,
                                        seed=seed,
                                        n_epochs=epochs,
                                        batch_size=64,
                                        hidden_dim=64,
                                        fixed_leaf_tokens=1,
                                        n_binary_leaves=n_binary_leaves,
                                        leaf_label_rate=float(bundle["leaf_label_rate"]),
                                        internal_label_rate=float(bundle["internal_label_rate"]),
                                        root_query_rate=float(bundle["root_query_rate"]),
                                        pairwise_prefs_per_doc=int(bundle["pairwise_prefs_per_doc"]),
                                        group_pref_groups_per_doc=int(bundle["group_pref_groups_per_doc"]),
                                        group_size=int(bundle["group_size"]),
                                        ppo_rollouts_per_doc=int(bundle["ppo_rollouts_per_doc"]),
                                        use_cuda=False,
                                    ),
                                )
                            )

    # CPU anchor controls for Markov and boundary-topic structural comparisons.
    control_matrix = [
        ("markov", MARKOV_RUNNER, ["markov_count_endpoints", "markov_count_only"], [8, 16, 32], None),
        ("boundary_topic", BOUNDARY_TOPIC_RUNNER, ["topic_plus_boundary", "topic_mass_only"], [2, 6, 10], "n_leaves"),
    ]
    control_arms = ["oracle_exact", "tree_exact_supported", "tree_undersupported", "flat_equal_info", "one_leaf_control", "right_rule_wrong_chunker"]
    control_objectives = [
        ("supervised_state", dict(leaf_label_rate=1.0, internal_label_rate=1.0, root_query_rate=0.0, pairwise_prefs_per_doc=0, group_pref_groups_per_doc=0, group_size=4, ppo_rollouts_per_doc=0)),
        ("dpo", dict(leaf_label_rate=0.0, internal_label_rate=0.0, root_query_rate=0.0, pairwise_prefs_per_doc=32, group_pref_groups_per_doc=0, group_size=4, ppo_rollouts_per_doc=0)),
        ("hybrid_supervised_plus_dpo", dict(leaf_label_rate=1.0, internal_label_rate=1.0, root_query_rate=0.0, pairwise_prefs_per_doc=32, group_pref_groups_per_doc=0, group_size=4, ppo_rollouts_per_doc=0)),
    ]
    for lane, runner, profiles, geometry_values, geometry_kind in control_matrix:
        for profile in profiles:
            for objective_family, bundle in control_objectives:
                for arm in control_arms:
                    if arm == "right_rule_wrong_chunker" and lane == "markov":
                        continue
                    for train_docs in (512, 2048):
                        for geom in geometry_values:
                            for seed in seeds:
                                tag = _support_tag(
                                    leaf=float(bundle["leaf_label_rate"]),
                                    internal=float(bundle["internal_label_rate"]),
                                    root=float(bundle["root_query_rate"]),
                                    pairwise=int(bundle["pairwise_prefs_per_doc"]),
                                    groups=int(bundle["group_pref_groups_per_doc"]),
                                    group_size=int(bundle["group_size"]),
                                    ppo=int(bundle["ppo_rollouts_per_doc"]),
                                    label="control_anchor",
                                )
                                geom_part = f"train_{train_docs}__leaf_{geom}" if lane == "markov" else f"train_{train_docs}__leaves_{geom}"
                                out = (
                                    root
                                    / lane
                                    / profile
                                    / "structural_controls_anchor"
                                    / objective_family
                                    / arm
                                    / geom_part
                                    / tag
                                    / f"seed_{seed}.json"
                                )
                                epochs = 1 if arm in {"oracle_exact", "tree_exact_supported"} else 20
                                kwargs = dict(
                                    runner=runner,
                                    json_summary=out,
                                    oracle_profile=profile,
                                    objective_family=objective_family,
                                    structural_arm=arm,
                                    train_docs=train_docs,
                                    test_docs=256,
                                    seed=seed,
                                    n_epochs=epochs,
                                    batch_size=64,
                                    hidden_dim=64,
                                    fixed_leaf_tokens=(geom if lane == "markov" else 1),
                                    n_leaves=(geom if lane == "boundary_topic" else None),
                                    leaf_label_rate=float(bundle["leaf_label_rate"]),
                                    internal_label_rate=float(bundle["internal_label_rate"]),
                                    root_query_rate=float(bundle["root_query_rate"]),
                                    pairwise_prefs_per_doc=int(bundle["pairwise_prefs_per_doc"]),
                                    group_pref_groups_per_doc=int(bundle["group_pref_groups_per_doc"]),
                                    group_size=int(bundle["group_size"]),
                                    ppo_rollouts_per_doc=int(bundle["ppo_rollouts_per_doc"]),
                                    use_cuda=False,
                                )
                                specs.append(
                                    CommandSpec(
                                        device_class="cpu",
                                        lane=lane,
                                        oracle_profile=profile,
                                        slice_name="structural_controls_anchor",
                                        objective_family=objective_family,
                                        structural_arm=arm,
                                        train_docs=train_docs,
                                        test_docs=256,
                                        seed=int(seed),
                                        fixed_leaf_tokens=(int(geom) if lane == "markov" else 1),
                                        n_leaves=(int(geom) if lane == "boundary_topic" else None),
                                        leaf_label_rate=float(bundle["leaf_label_rate"]),
                                        internal_label_rate=float(bundle["internal_label_rate"]),
                                        root_query_rate=float(bundle["root_query_rate"]),
                                        pairwise_prefs_per_doc=int(bundle["pairwise_prefs_per_doc"]),
                                        group_pref_groups_per_doc=int(bundle["group_pref_groups_per_doc"]),
                                        group_size=int(bundle["group_size"]),
                                        ppo_rollouts_per_doc=int(bundle["ppo_rollouts_per_doc"]),
                                        support_tag=tag,
                                        n_epochs=epochs,
                                        batch_size=64,
                                        hidden_dim=64,
                                        json_summary=str(out),
                                        cmd=_cmd(**kwargs),
                                    )
                                )
    return specs


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(path: Path, specs: Iterable[CommandSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for spec in specs:
            handle.write(json.dumps(asdict(spec), sort_keys=True) + "\n")


def _write_matrix_md(path: Path, gpu_specs: List[CommandSpec], cpu_specs: List[CommandSpec]) -> None:
    lines = [
        "# Exact Utility Transport Overnight Matrix",
        "",
        "## Summary",
        "",
        f"- GPU jobs: `{len(gpu_specs)}`",
        f"- CPU jobs: `{len(cpu_specs)}`",
        f"- Total jobs: `{len(gpu_specs) + len(cpu_specs)}`",
        "",
        "## GPU slices",
        "",
        "- `markov/support_curves_local`: tree-neural local-coverage sweeps across tree-relevant and tree-irrelevant profiles.",
        "- `markov/support_curves_preferences`: DPO/GRPO/PPO/hybrid sweeps for the main tree arm.",
        "- `markov/objective_family_high_support`: matched high-support objective-family comparison.",
        "- `boundary_topic/support_curves_local`: exact topic-boundary local-coverage sweeps.",
        "- `boundary_topic/objective_family_high_support`: matched high-support objective-family comparison.",
        "- `boundary_topic/support_curves_preferences`: DPO and PPO support sweeps in the bridge lane.",
        "",
        "## CPU slices",
        "",
        "- `nonseparable/structural_matrix`: dense structural and objective-family matrix for the exact analytic lane.",
        "- `markov/structural_controls_anchor`: matched structural controls at anchor budgets.",
        "- `boundary_topic/structural_controls_anchor`: matched structural controls at anchor budgets.",
        "",
    ]
    for device, specs in (("GPU", gpu_specs), ("CPU", cpu_specs)):
        slice_counts: Dict[str, int] = {}
        lane_counts: Dict[str, int] = {}
        for spec in specs:
            slice_counts[f"{spec.lane}/{spec.slice_name}"] = slice_counts.get(f"{spec.lane}/{spec.slice_name}", 0) + 1
            lane_counts[spec.lane] = lane_counts.get(spec.lane, 0) + 1
        lines.append(f"## {device} counts by lane")
        lines.append("")
        for key in sorted(lane_counts):
            lines.append(f"- `{key}`: `{lane_counts[key]}`")
        lines.append("")
        lines.append(f"## {device} counts by slice")
        lines.append("")
        for key in sorted(slice_counts):
            lines.append(f"- `{key}`: `{slice_counts[key]}`")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a large overnight exact utility transport matrix.")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--gpu-cmd-file", type=Path, required=True)
    p.add_argument("--cpu-cmd-file", type=Path, required=True)
    p.add_argument("--gpu-manifest", type=Path, required=True)
    p.add_argument("--cpu-manifest", type=Path, required=True)
    p.add_argument("--matrix-json", type=Path, required=True)
    p.add_argument("--matrix-md", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    gpu_specs = _markov_main_specs(args.output_root, args.seeds) + _boundary_topic_main_specs(args.output_root, args.seeds)
    cpu_specs = _cpu_control_specs(args.output_root, args.seeds)
    _write_lines(args.gpu_cmd_file, (spec.cmd for spec in gpu_specs))
    _write_lines(args.cpu_cmd_file, (spec.cmd for spec in cpu_specs))
    _write_manifest(args.gpu_manifest, gpu_specs)
    _write_manifest(args.cpu_manifest, cpu_specs)
    args.matrix_json.parent.mkdir(parents=True, exist_ok=True)
    args.matrix_json.write_text(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "gpu_jobs": len(gpu_specs),
                "cpu_jobs": len(cpu_specs),
                "gpu_manifest": str(args.gpu_manifest),
                "cpu_manifest": str(args.cpu_manifest),
                "seeds": list(map(int, args.seeds)),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_matrix_md(args.matrix_md, gpu_specs, cpu_specs)
    print(f"wrote_gpu_cmds | {args.gpu_cmd_file} | n={len(gpu_specs)}")
    print(f"wrote_cpu_cmds | {args.cpu_cmd_file} | n={len(cpu_specs)}")
    print(f"wrote_gpu_manifest | {args.gpu_manifest}")
    print(f"wrote_cpu_manifest | {args.cpu_manifest}")
    print(f"wrote_matrix_json | {args.matrix_json}")
    print(f"wrote_matrix_md | {args.matrix_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
