from __future__ import annotations

import random
from pathlib import Path
import subprocess
import sys

import torch

from src.ctreepo.sim.core.boundary_topic_treepo_preference import (
    BoundaryTopicExactUtilityConfig,
    BoundaryTopicExactUtilityDGP,
    run_boundary_topic_exact_utility_experiment,
)
from src.ctreepo.sim.core.exact_utility_common import (
    FlatSpanPolicy,
    _balanced_span_leaf_indices,
    _ppo_style_loss,
    _sample_fractional_indices,
)
from src.ctreepo.sim.core.markov_treepo_preference import (
    MarkovExactUtilityConfig,
    MarkovExactUtilityDGP,
    run_markov_exact_utility_experiment,
)
from src.ctreepo.sim.core.nonseparable_treepo_preference import (
    NonseparableExactUtilityConfig,
    NonseparableExactUtilityDGP,
    run_nonseparable_exact_utility_experiment,
)
from src.ctreepo.sim.utility_transport_expectations import build_utility_transport_report


def test_markov_exact_utility_alignment() -> None:
    dgp = MarkovExactUtilityDGP(oracle_profile="markov_count_endpoints", n_regimes=3, max_count=6)
    truth = dgp.action_from_state_tuple((3, 1, 2))
    wrong = dgp.action_from_state_tuple((4, 1, 0))
    assert dgp.state_distance(truth, truth) == 0.0
    assert dgp.utility(truth, truth) == 1.0
    assert dgp.state_distance(wrong, truth) > 0.0
    assert dgp.utility(wrong, truth) < 1.0


def test_sample_fractional_indices_are_nested_and_keep_rng_aligned() -> None:
    low_rng = random.Random(102)
    high_rng = random.Random(102)

    low = _sample_fractional_indices(22, 2.0 / 22.0, low_rng)
    high = _sample_fractional_indices(22, 6.0 / 22.0, high_rng)

    assert set(low).issubset(high)

    followup_low = _sample_fractional_indices(17, 3.0 / 17.0, low_rng)
    followup_high = _sample_fractional_indices(17, 3.0 / 17.0, high_rng)

    assert followup_low == followup_high


def test_nonseparable_exact_utility_alignment() -> None:
    dgp = NonseparableExactUtilityDGP(oracle_profile="dgp1_complementarity_and", count_max=5, n_binary_leaves=4)
    truth = dgp.action_idx(3, 4)
    wrong = dgp.action_idx(4, 1)
    assert dgp.state_distance(truth, truth) == 0.0
    assert dgp.utility(truth, truth) == 1.0
    assert dgp.state_distance(wrong, truth) > 0.0
    assert dgp.utility(wrong, truth) < 1.0


def test_boundary_topic_exact_utility_alignment() -> None:
    dgp = BoundaryTopicExactUtilityDGP(oracle_profile="topic_plus_boundary", n_topics=3, n_leaves=4)
    truth = dgp.action_idx((2, 1, 1, 0, 2))
    wrong = dgp.action_idx((1, 2, 1, 1, 2))
    assert dgp.state_distance(truth, truth) == 0.0
    assert dgp.utility(truth, truth) == 1.0
    assert dgp.state_distance(wrong, truth) > 0.0
    assert dgp.utility(wrong, truth) < 1.0


def test_oracle_exact_zero_regret_smoke() -> None:
    markov = run_markov_exact_utility_experiment(
        MarkovExactUtilityConfig(
            objective_family="supervised_state",
            structural_arm="oracle_exact",
            train_docs=8,
            test_docs=4,
            n_epochs=1,
            batch_size=4,
            hidden_dim=16,
            fixed_leaf_tokens=8,
            max_tokens=32,
            min_tokens=32,
            min_segments=4,
            max_segments=6,
            min_seg_len=2,
            max_seg_len=8,
            use_cuda=False,
        )
    )
    nonsep = run_nonseparable_exact_utility_experiment(
        NonseparableExactUtilityConfig(
            objective_family="supervised_state",
            structural_arm="oracle_exact",
            train_docs=16,
            test_docs=8,
            n_epochs=1,
            batch_size=8,
            hidden_dim=16,
            use_cuda=False,
        )
    )
    topic = run_boundary_topic_exact_utility_experiment(
        BoundaryTopicExactUtilityConfig(
            objective_family="supervised_state",
            structural_arm="oracle_exact",
            train_docs=16,
            test_docs=8,
            n_epochs=1,
            batch_size=8,
            hidden_dim=16,
            use_cuda=False,
        )
    )
    assert float(markov.metrics["utility_regret"]) == 0.0
    assert float(nonsep.metrics["utility_regret"]) == 0.0
    assert float(topic.metrics["utility_regret"]) == 0.0
    assert markov.objective["kind"] == "preference_training_objective"
    assert nonsep.objective["kind"] == "preference_training_objective"
    assert topic.objective["kind"] == "preference_training_objective"
    assert markov.objective["metadata"]["objective_family"] == "supervised_state"
    assert nonsep.objective["metadata"]["objective_family"] == "supervised_state"
    assert topic.objective["metadata"]["objective_family"] == "supervised_state"


def test_markov_tree_neural_supported_smoke() -> None:
    summary = run_markov_exact_utility_experiment(
        MarkovExactUtilityConfig(
            objective_family="supervised_state",
            structural_arm="tree_neural_supported",
            train_docs=8,
            test_docs=4,
            n_epochs=1,
            batch_size=4,
            hidden_dim=16,
            fixed_leaf_tokens=8,
            max_tokens=32,
            min_tokens=32,
            min_segments=4,
            max_segments=6,
            min_seg_len=2,
            max_seg_len=8,
            leaf_label_rate=0.25,
            internal_label_rate=0.25,
            root_query_rate=0.0,
            pairwise_prefs_per_doc=0,
            group_pref_groups_per_doc=0,
            ppo_rollouts_per_doc=0,
            use_cuda=False,
        )
    )
    assert "utility_regret" in summary.metrics
    assert float(summary.metrics["utility_regret"]) >= 0.0


def test_markov_tree_undersupported_smoke() -> None:
    summary = run_markov_exact_utility_experiment(
        MarkovExactUtilityConfig(
            objective_family="supervised_state",
            structural_arm="tree_undersupported",
            train_docs=8,
            test_docs=4,
            n_epochs=1,
            batch_size=4,
            hidden_dim=16,
            fixed_leaf_tokens=8,
            max_tokens=32,
            min_tokens=32,
            min_segments=4,
            max_segments=6,
            min_seg_len=2,
            max_seg_len=8,
            leaf_label_rate=0.25,
            internal_label_rate=0.25,
            root_query_rate=0.0,
            use_cuda=False,
        )
    )
    assert "utility_regret" in summary.metrics
    assert float(summary.metrics["utility_regret"]) >= 0.0


def test_boundary_topic_tree_undersupported_smoke() -> None:
    summary = run_boundary_topic_exact_utility_experiment(
        BoundaryTopicExactUtilityConfig(
            objective_family="supervised_state",
            structural_arm="tree_undersupported",
            train_docs=8,
            test_docs=4,
            n_epochs=1,
            batch_size=4,
            hidden_dim=16,
            leaf_label_rate=0.25,
            internal_label_rate=0.25,
            root_query_rate=0.0,
            use_cuda=False,
        )
    )
    assert "utility_regret" in summary.metrics
    assert float(summary.metrics["utility_regret"]) >= 0.0


def test_nonseparable_flat_span_equal_info_smoke() -> None:
    summary = run_nonseparable_exact_utility_experiment(
        NonseparableExactUtilityConfig(
            oracle_profile="dgp2_boundary_interaction",
            objective_family="hybrid_supervised_plus_ppo",
            structural_arm="flat_span_equal_info",
            train_docs=8,
            test_docs=4,
            n_epochs=1,
            batch_size=4,
            hidden_dim=8,
            leaf_label_rate=1.0,
            internal_label_rate=1.0,
            root_query_rate=0.0,
            ppo_rollouts_per_doc=1,
            n_binary_leaves=2,
            use_cuda=False,
        )
    )
    assert "utility_regret" in summary.metrics
    assert float(summary.metrics["utility_regret"]) >= 0.0


def test_flat_span_policy_shape_and_balanced_spans() -> None:
    model = FlatSpanPolicy(obs_dim=2, hidden_dim=8, n_actions=5)
    leaf_stack = torch.randn(4, 2)
    encoded = model.encode_leaf_batch(leaf_stack)
    logits_single = model.logits_from_encoded_span(encoded, (0,))
    logits_many = model.logits_from_encoded_span(encoded, (0, 1, 2, 3))
    assert tuple(logits_single.shape) == (5,)
    assert tuple(logits_many.shape) == (5,)
    spans = _balanced_span_leaf_indices(4)
    assert spans == [(0, 1), (2, 3), (0, 1, 2, 3)]


def test_ppo_style_loss_stays_finite_with_centering_and_normalization() -> None:
    dgp = NonseparableExactUtilityDGP(oracle_profile="dgp2_boundary_zero", count_max=5, n_binary_leaves=4)
    logits = torch.tensor([0.1, 0.2, -0.3, 0.0, 0.5], dtype=torch.float32)
    loss = _ppo_style_loss(
        logits,
        true_idx=0,
        dgp=dgp,
        n_rollouts=1,
        kl_weight=0.02,
        entropy_weight=0.01,
        advantage_center=True,
        advantage_normalize=True,
        reward_baseline="mean_reward",
        clip_epsilon=0.2,
    )
    assert torch.isfinite(loss)
    zero_var_loss = _ppo_style_loss(
        torch.zeros(4, dtype=torch.float32),
        true_idx=0,
        dgp=dgp,
        n_rollouts=8,
        kl_weight=0.0,
        entropy_weight=0.0,
        advantage_center=True,
        advantage_normalize=True,
        reward_baseline="mean_reward",
        clip_epsilon=0.2,
    )
    assert torch.isfinite(zero_var_loss)


def test_expectation_harness_prefers_flat_span_for_local_supervision(tmp_path: Path) -> None:
    root = tmp_path / "fairness"
    common = {
        "lane": "nonseparable",
        "oracle_profile": "dgp2_boundary_interaction",
        "objective_family": "hybrid_supervised_plus_ppo",
        "config": {
            "train_docs": 128,
            "seed": 0,
            "fixed_leaf_tokens": 1,
            "objective_family": "hybrid_supervised_plus_ppo",
            "structural_arm": "",
        },
        "budget": {
            "train_docs": 128,
            "test_docs": 64,
            "doc_scale_tokens": 4.0,
            "fixed_leaf_tokens": 1,
            "leaves_per_doc": 4.0,
            "leaf_label_coverage": 1.0,
            "internal_label_coverage": 1.0,
            "root_query_rate": 0.0,
            "local_oracle_coverage": 1.0,
            "pairwise_prefs_per_doc": 0.0,
            "group_pref_groups_per_doc": 0.0,
            "group_size": 4,
            "ppo_rollouts_per_doc": 64.0,
            "total_oracle_calls_estimate": 1024.0,
        },
        "metadata": {
            "tree_relevance": "tree_relevant",
            "lean_theorems": ["MainTheorems.oracle_indexed_objective_transport"],
        },
    }
    entries = [
        ("tree_neural_supported", 0.05),
        ("flat_equal_info", 0.20),
        ("flat_span_equal_info", 0.06),
        ("tree_undersupported", 0.12),
        ("one_leaf_control", 0.09),
    ]
    for arm, regret in entries:
        payload = dict(common)
        payload["structural_arm"] = arm
        payload["config"] = dict(common["config"])
        payload["config"]["structural_arm"] = arm
        payload["metrics"] = {
            "root": {"exact_state_accuracy": 1.0 - regret, "utility_regret": regret, "state_l1": regret},
            "utility_regret": regret,
            "root_mae": regret,
            "merge_mae": regret,
        }
        path = root / "nonseparable" / "dgp2_boundary_interaction" / "structural_matrix" / "hybrid_supervised_plus_ppo" / arm / "seed_0.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    report = build_utility_transport_report(root)
    findings = [f for f in report.findings if f.kind == "tree_relevance"]
    assert findings
    assert findings[0].observed["flat_arm"] == "flat_span_equal_info"


def test_exact_utility_scripts_and_report_smoke(tmp_path: Path) -> None:
    root = tmp_path / "utility_suite"
    cmds = [
        [
            sys.executable,
            "scripts/run_markov_treepo_preference.py",
            "--objective-family",
            "supervised_state",
            "--structural-arm",
            "oracle_exact",
            "--train-docs",
            "8",
            "--test-docs",
            "4",
            "--n-epochs",
            "1",
            "--batch-size",
            "4",
            "--hidden-dim",
            "16",
            "--fixed-leaf-tokens",
            "8",
            "--json-summary",
            str(root / "markov.json"),
        ],
        [
            sys.executable,
            "scripts/run_nonseparable_treepo_preference.py",
            "--objective-family",
            "supervised_state",
            "--structural-arm",
            "oracle_exact",
            "--train-docs",
            "8",
            "--test-docs",
            "4",
            "--n-epochs",
            "1",
            "--batch-size",
            "4",
            "--hidden-dim",
            "16",
            "--json-summary",
            str(root / "nonsep.json"),
        ],
        [
            sys.executable,
            "scripts/run_boundary_topic_treepo_preference.py",
            "--objective-family",
            "supervised_state",
            "--structural-arm",
            "oracle_exact",
            "--train-docs",
            "8",
            "--test-docs",
            "4",
            "--n-epochs",
            "1",
            "--batch-size",
            "4",
            "--hidden-dim",
            "16",
            "--json-summary",
            str(root / "topic.json"),
        ],
    ]
    for cmd in cmds:
        subprocess.run(cmd, cwd="/home/mlinegar/ThinkingTrees", check=True)
    subprocess.run(
        [
            sys.executable,
            "scripts/report_treepo_preference_suite.py",
            "--output-root",
            str(root),
            "--output-markdown",
            str(root / "utility_transport_report.md"),
        ],
        cwd="/home/mlinegar/ThinkingTrees",
        check=True,
    )
    assert (root / "utility_transport_summary.json").exists()
    assert (root / "utility_transport_summary.csv").exists()
    assert (root / "utility_transport_report.md").exists()
    assert (root / "figures" / "utility_transport_suite.png").exists()
    report = build_utility_transport_report(root)
    assert report.rows
    assert report.findings
