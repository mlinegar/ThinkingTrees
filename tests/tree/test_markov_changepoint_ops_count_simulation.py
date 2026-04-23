import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.tree.markov_changepoint_honesty_simulation import ChangepointMarkovDoc  # noqa: E402
from src.ctreepo.sim.cli.run_markov_changepoint_ops_count import parse_args as parse_markov_ops_args  # noqa: E402
from src.tree.markov_changepoint_ops_count_simulation import (  # noqa: E402
    AdditiveCountSketch,
    MarkovOPSDataBundle,
    OPSCountConfig,
    _build_objective_summary,
    _sample_internal_audit_indices,
    _audit_estimator_diagnostics,
    build_markov_changepoint_ops_count_data_bundle,
    _eval_count_only_family,
    _eval_exact_family,
    _eval_flip_family,
    _eval_leaf_bucket_family,
    _leaf_spans,
    run_markov_changepoint_ops_count_experiment,
)
from src.ctreepo.sim.core.markov_capability import classify_capability  # noqa: E402


def _toy_doc_two_constant_leaves() -> ChangepointMarkovDoc:
    # Regimes: [0,0,0 | 1,1,1] with a single join changepoint.
    token_regimes = (0, 0, 0, 1, 1, 1)
    tokens = tuple(range(len(token_regimes)))
    transition_regimes = token_regimes[1:]
    true_boundaries = (2,)
    return ChangepointMarkovDoc(
        tokens=tokens,
        token_regimes=token_regimes,
        transition_regimes=transition_regimes,
        true_boundaries=true_boundaries,
    )


def test_exact_sketch_hits_zero_root_distortion_and_zero_schedule_spread():
    doc = _toy_doc_two_constant_leaves()
    m = _eval_exact_family([doc], leaf_tokens=3, tau=0.0)
    assert m.root_mae == pytest.approx(0.0, abs=1e-12)
    assert m.schedule_spread_mean == pytest.approx(0.0, abs=1e-12)
    assert m.merge_violation_rate == pytest.approx(0.0, abs=1e-12)


def test_under_supported_sketch_misses_join_changepoint():
    doc = _toy_doc_two_constant_leaves()
    m = _eval_count_only_family([doc], leaf_tokens=3, tau=0.0)
    assert m.root_mae == pytest.approx(1.0, abs=1e-12)
    assert m.merge_violation_rate > 0.0


def test_leaf_bucket_family_breaks_c1_while_preserving_root():
    doc = _toy_doc_two_constant_leaves()
    m = _eval_leaf_bucket_family([doc], leaf_tokens=3, tau=0.0)
    assert m.leaf_mae > 0.0
    assert m.root_mae == pytest.approx(0.0, abs=1e-12)


def test_flip_on_range_has_zero_distortion_at_R1_but_drifts_at_R2():
    doc = _toy_doc_two_constant_leaves()
    m1 = _eval_flip_family([doc], leaf_tokens=3, tau=0.0, rounds=1)
    m2 = _eval_flip_family([doc], leaf_tokens=3, tau=0.0, rounds=2)
    assert m1.root_mae == pytest.approx(0.0, abs=1e-12)
    assert m2.root_mae == pytest.approx(1.0, abs=1e-12)
    assert m2.resummary_root_drift_r1 == pytest.approx(1.0, abs=1e-12)


def test_markov_models_resummarize_over_full_summary_state():
    from src.ctreepo.sim.core.markov_neural_operator_baselines import FNOCountSketch

    n_regimes = 3
    first = torch.nn.functional.one_hot(torch.tensor(1), num_classes=n_regimes).to(torch.float32)
    last = torch.nn.functional.one_hot(torch.tensor(2), num_classes=n_regimes).to(torch.float32)
    core = torch.tensor([0.2, -0.1, 0.5, 0.3], dtype=torch.float32)
    features = torch.cat([first, last, core], dim=0)

    fno = FNOCountSketch(
        vocab_size=8,
        leaf_tokens=8,
        state_dim=5,
        hidden_dim=8,
        target_scale=4.0,
        n_regimes=n_regimes,
        fno_width=8,
        fno_n_modes=4,
        fno_n_layers=1,
    )
    additive = AdditiveCountSketch(
        feature_dim=int(features.numel()),
        hidden_dim=8,
        target_scale=4.0,
        n_regimes=n_regimes,
        use_endpoints=True,
    )

    # Test additive with feature-based encoding.
    state = additive.encode_leaf(features)
    replay = additive.encode_summary(additive.decode_summary(state))
    assert replay.shape == state.shape
    assert torch.allclose(replay, state, atol=1e-12, rtol=0.0)
    assert torch.allclose(
        additive.predict_count_from_state(replay),
        additive.predict_count_from_state(state),
        atol=1e-12,
        rtol=0.0,
    )

    # Test FNO with token-based encoding.
    fno_state = fno.encode_leaf_tokens([0, 1, 2, 3, 4, 5, 6, 7], device=torch.device("cpu"))
    fno_replay = fno.encode_summary(fno.decode_summary(fno_state))
    assert fno_replay.shape == fno_state.shape
    assert torch.allclose(
        fno.predict_count_from_state(fno_replay),
        fno.predict_count_from_state(fno_state),
        atol=1e-4,
        rtol=0.0,
    )


def test_ipw_and_dsl_are_unbiased_under_nonuniform_sampling_naive_is_biased():
    rng = np.random.default_rng(0)
    n = 250
    values = np.linspace(0.0, 1.0, n, dtype=np.float64)
    scores = values + 0.05
    base = 0.20
    pi = base * (scores / float(np.mean(scores)))
    pi = np.clip(pi, 0.05, 1.0)

    # Oracle-quality predictions => DSL variance should collapse.
    diag = _audit_estimator_diagnostics(
        values.tolist(),
        values.tolist(),
        pi.tolist(),
        trials=1200,
        seed=int(rng.integers(0, 2**31 - 1)),
    )

    assert abs(diag.naive_bias) > 0.02
    assert abs(diag.ipw_bias) < 0.02
    assert abs(diag.dsl_bias) < 0.02
    assert math.isfinite(diag.ipw_var) and diag.ipw_var > 0.0
    assert math.isfinite(diag.dsl_var) and diag.dsl_var < diag.ipw_var


def test_experiment_emits_selection_demo_diagnostics_on_learned_merge_population():
    cfg = OPSCountConfig(
        n_regimes=3,
        vocab_size=48,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=8,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        train_docs=4,
        test_docs=4,
        feature_mode="full",
        state_dim=8,
        hidden_dim=32,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.5,
        include_root_query=True,
        violation_tau=0.0,
        seed=0,
        use_cuda=False,
        torch_threads=1,
    )
    out = run_markov_changepoint_ops_count_experiment(cfg)
    diag = out.estimator_diagnostics
    assert "selection_demo_base_rate" in diag
    assert "selection_demo_pi_min" in diag
    assert "selection_demo_n_units" in diag
    assert float(diag["selection_demo_n_units"]) > 0.0
    assert math.isfinite(float(diag["naive_bias"]))
    assert math.isfinite(float(diag["ipw_bias"]))
    assert math.isfinite(float(diag["dsl_bias"]))


def test_leaf_query_rate_reduces_leaf_label_budget_in_training_geometry():
    common = dict(
        n_regimes=3,
        vocab_size=48,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=8,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        train_docs=6,
        test_docs=2,
        feature_mode="full",
        state_dim=8,
        hidden_dim=16,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.5,
        include_root_query=True,
        violation_tau=0.0,
        seed=123,
        use_cuda=False,
        torch_threads=1,
    )
    full = run_markov_changepoint_ops_count_experiment(
        OPSCountConfig(**{**common, "leaf_query_rate": 1.0})
    )
    sparse = run_markov_changepoint_ops_count_experiment(
        OPSCountConfig(**{**common, "leaf_query_rate": 0.25})
    )
    g_full = full.training_geometry
    g_sparse = sparse.training_geometry
    assert float(g_sparse["mean_leaf_labels"]) < float(g_full["mean_leaf_labels"])
    assert int(g_sparse["leaf_labels_total"]) < int(g_full["leaf_labels_total"])


def test_markov_ops_cli_parses_theorem_feature_fields():
    args = parse_markov_ops_args(
        [
            "--tree-c2-mode",
            "fiber",
            "--theorem-feature-adapter",
            "markov_count_sketch",
            "--theorem-pair-same-threshold",
            "0.25",
            "--theorem-pair-diff-threshold",
            "0.75",
        ]
    )

    assert args.tree_c2_mode == "fiber"
    assert args.theorem_feature_adapter == "markov_count_sketch"
    assert float(args.theorem_pair_same_threshold) == pytest.approx(0.25)
    assert float(args.theorem_pair_diff_threshold) == pytest.approx(0.75)


def test_formal_local_law_parameterization_resolves_c1_and_c3_weights():
    cfg = OPSCountConfig(
        local_law_weight=0.75,
        c1_relative_weight=1.0,
        c2_relative_weight=0.0,
        c3_relative_weight=2.0,
        leaf_weight=9.0,
        c3_weight=9.0,
        model_family="neural",
        include_root_query=False,
        schedule_consistency_weight=0.1,
        use_cuda=False,
    )
    objective = _build_objective_summary(cfg)

    assert objective["parameterization"] == "formal_local_law_weight"
    assert objective["local_law_weight"] == pytest.approx(0.75)
    assert objective["local_law_c1_weight"] == pytest.approx(0.25)
    assert objective["local_law_c3_weight"] == pytest.approx(0.50)
    assert objective["local_law_c2_weight"] == pytest.approx(0.0)
    assert objective["local_law_c1_share"] == pytest.approx(1.0 / 3.0)
    assert objective["local_law_c3_share"] == pytest.approx(2.0 / 3.0)
    assert objective["optimization_root_weight"] == pytest.approx(0.25)
    assert objective["proxy_schedule_consistency_weight"] == pytest.approx(0.1)
    assert objective["theorem_terms"][0]["paper_condition"] == "C1"
    assert objective["theorem_terms"][1]["paper_condition"] == "C3"
    assert objective["proxy_terms"][0]["evidence_status"] == "proxy_only"


def test_experiment_reports_objective_decomposition_in_summary():
    cfg = OPSCountConfig(
        n_regimes=3,
        vocab_size=48,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=8,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        train_docs=4,
        val_docs=2,
        test_docs=4,
        feature_mode="full",
        state_dim=8,
        hidden_dim=32,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.5,
        include_root_query=True,
        violation_tau=0.0,
        seed=11,
        use_cuda=False,
        torch_threads=1,
    )
    out = run_markov_changepoint_ops_count_experiment(cfg)

    objective = out.objective
    assert objective["parameterization"] == "legacy_term_weights"
    assert objective["training_scheme"] == "weighted_neural_objective"
    assert objective["local_law_weight"] == pytest.approx(float(cfg.leaf_weight + cfg.c3_weight))
    assert objective["local_law_c1_weight"] == pytest.approx(float(cfg.leaf_weight))
    assert objective["local_law_c3_weight"] == pytest.approx(float(cfg.c3_weight))
    assert objective["root_supervision_active"] is True
    learned = out.metrics["learned"]
    assert "train_objective_full_labels" in learned
    assert "val_objective_full_labels" in learned
    assert "test_objective_full_labels" in learned
    assert "val_root_mae" in learned
    assert "test_root_mae" in learned
    assert "test_c2_idempotence_mae_n" in learned
    assert "test_c2_r4_mae_n" in learned
    assert "test_resummary_root_drift_r2_n" in learned
    assert "train_objective_c2_term" in learned
    assert "train_unweighted_objective_full_labels" in learned
    assert "test_unweighted_objective_full_labels" in learned
    assert "generalization_gap_unweighted_objective_full_labels" in learned
    assert "learned_val" in out.metrics
    assert "learned_test" in out.metrics
    assert int(out.config["effective_val_seed"]) == int(
        cfg.data_seed if cfg.data_seed is not None else cfg.seed
    ) + int(cfg.val_seed_offset)
    assert "generalization_gap_objective_full_labels" in learned
    assert "objective" in out.to_json()


def test_experiment_emits_local_law_learnability_and_artifacts(tmp_path: Path):
    cfg = OPSCountConfig(
        n_regimes=3,
        vocab_size=48,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=8,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        train_docs=4,
        val_docs=2,
        test_docs=4,
        feature_mode="full",
        state_dim=8,
        hidden_dim=32,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        law_package="root_only",
        audit_policy="fraction",
        audit_fraction=0.5,
        include_root_query=True,
        violation_tau=0.0,
        seed=17,
        use_cuda=False,
        torch_threads=1,
        suite_role="support_scaling",
        artifact_dir=str(tmp_path / "artifacts"),
    )
    out = run_markov_changepoint_ops_count_experiment(cfg)
    payload = json.loads(out.to_json())
    learnability = dict(payload["local_law_learnability"])
    assert learnability["family"] == "markov_ops_count"
    assert learnability["selection"]["selection_split"] == "val"
    assert learnability["selection"]["selection_metric"] == "configured_objective"
    policies = dict(learnability["policies"])
    assert policies["oracle_g"]["role"] == "oracle_g"
    assert policies["root_only"]["role"] == "baseline_g"
    counterexample_names = {item["name"] for item in learnability["counterexamples"]}
    assert counterexample_names == {"leaf_bucket", "count_only", "flip_R2"}

    g_artifacts = dict(payload["g_artifacts"])
    assert "oracle_g" in g_artifacts
    assert "baseline_g" in g_artifacts
    for artifact in g_artifacts.values():
        manifest_path = Path(artifact["manifest_path"])
        assert manifest_path.exists()


def test_markov_split_objective_exposes_hajek_selection_when_local_laws_are_estimable() -> None:
    cfg = OPSCountConfig(
        n_regimes=3,
        vocab_size=48,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=8,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        train_docs=6,
        val_docs=3,
        test_docs=4,
        feature_mode="full",
        state_dim=8,
        hidden_dim=32,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        local_law_weight=0.5,
        c1_relative_weight=1.0,
        c2_relative_weight=0.0,
        c3_relative_weight=1.0,
        audit_policy="fraction",
        audit_fraction=0.5,
        c3_audit_strategy="uniform",
        include_root_query=True,
        violation_tau=0.0,
        seed=23,
        use_cuda=False,
        torch_threads=1,
    )

    out = run_markov_changepoint_ops_count_experiment(cfg)
    learned = dict(out.metrics["learned"])
    learnability = dict(out.local_law_learnability)

    assert learned["val_objective_selection_metric_name"] == "configured_objective_hajek"
    assert math.isfinite(float(learned["val_objective_selection_metric_value"]))
    assert math.isfinite(float(learned["val_configured_objective_hajek"]))
    assert learnability["selection"]["selection_metric"] == "configured_objective_hajek"


def test_markov_split_objective_falls_back_to_exact_when_c3_support_is_not_identified() -> None:
    cfg = OPSCountConfig(
        n_regimes=3,
        vocab_size=48,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=8,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        train_docs=6,
        val_docs=3,
        test_docs=4,
        feature_mode="full",
        state_dim=8,
        hidden_dim=32,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        local_law_weight=0.5,
        c1_relative_weight=0.0,
        c2_relative_weight=0.0,
        c3_relative_weight=1.0,
        audit_policy="fraction",
        audit_fraction=0.5,
        c3_audit_strategy="top_span",
        include_root_query=True,
        violation_tau=0.0,
        seed=29,
        use_cuda=False,
        torch_threads=1,
    )

    out = run_markov_changepoint_ops_count_experiment(cfg)
    learned = dict(out.metrics["learned"])
    learnability = dict(out.local_law_learnability)

    assert learned["val_objective_selection_metric_name"] == "configured_objective"
    assert math.isfinite(float(learned["val_objective_selection_metric_value"]))
    assert learnability["selection"]["selection_metric"] == "configured_objective"


def test_default_markov_ops_count_config_is_root_only_no_local_law_baseline():
    cfg = OPSCountConfig(use_cuda=False)
    objective = _build_objective_summary(cfg)

    assert cfg.leaf_weight == pytest.approx(0.0)
    assert cfg.c3_weight == pytest.approx(0.0)
    assert objective["parameterization"] == "legacy_term_weights"
    assert objective["local_law_weight"] == pytest.approx(0.0)
    assert objective["local_law_c1_weight"] == pytest.approx(0.0)
    assert objective["local_law_c3_weight"] == pytest.approx(0.0)
    assert objective["local_law_active"] is False
    assert objective["root_supervision_active"] is True


def test_formal_local_law_defaults_to_equal_split_and_normalized_root_weight():
    cfg = OPSCountConfig(
        local_law_weight=0.6,
        model_family="neural",
        use_cuda=False,
    )
    objective = _build_objective_summary(cfg)

    assert objective["parameterization"] == "formal_local_law_weight"
    assert objective["weighting_scheme"] == "normalized_lambda_tradeoff"
    assert objective["local_law_c1_weight"] == pytest.approx(0.2)
    assert objective["local_law_c2_weight"] == pytest.approx(0.2)
    assert objective["local_law_c3_weight"] == pytest.approx(0.2)
    assert objective["task_objective_weight"] == pytest.approx(0.4)
    assert objective["task_objective_weight_source"] == "derived_from_local_law_weight"
    assert objective["optimization_root_weight"] == pytest.approx(0.4)
    assert objective["optimization_weight_mass_no_proxy"] == pytest.approx(1.0)


def test_formal_local_law_can_use_explicit_task_weight_override():
    cfg = OPSCountConfig(
        local_law_weight=0.6,
        task_objective_weight=1.5,
        model_family="neural",
        use_cuda=False,
    )
    objective = _build_objective_summary(cfg)

    assert objective["parameterization"] == "formal_local_law_weight"
    assert objective["weighting_scheme"] == "explicit_task_plus_local_law"
    assert objective["task_objective_weight"] == pytest.approx(1.5)
    assert objective["task_objective_weight_source"] == "explicit_task_objective_weight"
    assert objective["local_law_c1_weight"] == pytest.approx(0.2)
    assert objective["local_law_c2_weight"] == pytest.approx(0.2)
    assert objective["local_law_c3_weight"] == pytest.approx(0.2)
    assert objective["optimization_root_weight"] == pytest.approx(1.5)
    assert objective["optimization_weight_mass_no_proxy"] == pytest.approx(2.1)


def test_formal_objective_reports_weighted_and_unweighted_metrics_separately():
    common = dict(
        n_regimes=3,
        vocab_size=48,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=8,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        train_docs=8,
        test_docs=8,
        model_family="neural",
        feature_mode="full",
        state_dim=8,
        hidden_dim=24,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.5,
        c3_audit_strategy="uniform",
        c3_include_root=True,
        leaf_query_rate=1.0,
        include_root_query=True,
        violation_tau=0.0,
        seed=19,
        use_cuda=False,
        torch_threads=1,
    )
    out = run_markov_changepoint_ops_count_experiment(
        OPSCountConfig(
            **common,
            local_law_weight=0.25,
            c1_relative_weight=1.0,
            c2_relative_weight=0.0,
            c3_relative_weight=4.0,
        )
    )

    objective = out.objective
    learned = out.metrics["learned"]
    assert objective["optimization_root_weight"] == pytest.approx(0.75)
    assert objective["task_objective_weight"] == pytest.approx(0.75)
    assert objective["task_objective_weight_source"] == "derived_from_local_law_weight"
    assert objective["local_law_c1_weight"] == pytest.approx(0.05)
    assert objective["local_law_c2_weight"] == pytest.approx(0.0)
    assert objective["local_law_c3_weight"] == pytest.approx(0.20)
    assert objective["optimization_weight_mass_no_proxy"] == pytest.approx(1.0)
    assert float(learned["train_objective_full_labels"]) == pytest.approx(
        float(learned["train_optimization_objective_full_labels"])
    )
    assert float(learned["test_objective_full_labels"]) == pytest.approx(
        float(learned["test_optimization_objective_full_labels"])
    )
    assert float(learned["train_objective_task_objective_term"]) == pytest.approx(
        float(learned["train_objective_root_term"])
    )
    assert float(learned["test_optimization_objective_task_objective_term"]) == pytest.approx(
        float(learned["test_optimization_objective_root_term"])
    )
    assert float(learned["train_unweighted_objective_task_objective_term"]) == pytest.approx(
        float(learned["train_unweighted_objective_root_term"])
    )
    assert "train_unweighted_objective_root_term" in learned
    assert "test_unweighted_objective_root_term" in learned


def test_formal_objective_explicit_task_weight_override_flows_into_metrics():
    common = dict(
        n_regimes=3,
        vocab_size=48,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=8,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        train_docs=8,
        test_docs=8,
        model_family="neural",
        feature_mode="full",
        state_dim=8,
        hidden_dim=24,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.5,
        c3_audit_strategy="uniform",
        c3_include_root=True,
        leaf_query_rate=1.0,
        include_root_query=True,
        violation_tau=0.0,
        seed=29,
        use_cuda=False,
        torch_threads=1,
    )
    out = run_markov_changepoint_ops_count_experiment(
        OPSCountConfig(
            **common,
            local_law_weight=0.25,
            task_objective_weight=1.25,
            c1_relative_weight=1.0,
            c2_relative_weight=0.0,
            c3_relative_weight=4.0,
        )
    )

    objective = out.objective
    learned = out.metrics["learned"]
    assert objective["weighting_scheme"] == "explicit_task_plus_local_law"
    assert objective["task_objective_weight_source"] == "explicit_task_objective_weight"
    assert objective["task_objective_weight"] == pytest.approx(1.25)
    assert objective["optimization_root_weight"] == pytest.approx(1.25)
    assert objective["local_law_c1_weight"] == pytest.approx(0.05)
    assert objective["local_law_c3_weight"] == pytest.approx(0.20)
    assert objective["optimization_weight_mass_no_proxy"] == pytest.approx(1.50)
    assert float(learned["train_objective_task_objective_term"]) == pytest.approx(
        float(learned["train_objective_root_term"])
    )
    assert float(learned["test_unweighted_objective_task_objective_term"]) == pytest.approx(
        float(learned["test_unweighted_objective_root_term"])
    )


def test_learned_markov_c2_is_exact_in_full_summary_space():
    cfg = OPSCountConfig(
        n_regimes=3,
        vocab_size=48,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=8,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        train_docs=8,
        test_docs=8,
        model_family="neural",
        feature_mode="full",
        state_dim=8,
        hidden_dim=24,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.5,
        c3_audit_strategy="uniform",
        c3_include_root=True,
        leaf_query_rate=1.0,
        include_root_query=True,
        local_law_weight=0.5,
        violation_tau=0.0,
        seed=23,
        use_cuda=False,
        torch_threads=1,
    )
    out = run_markov_changepoint_ops_count_experiment(cfg)
    learned = out.metrics["learned"]

    assert float(learned["train_c2_idempotence_mae"]) == pytest.approx(0.0, abs=1e-12)
    assert float(learned["c2_idempotence_mae"]) == pytest.approx(0.0, abs=1e-12)
    assert float(learned["train_objective_c2_term"]) == pytest.approx(0.0, abs=1e-12)
    assert float(learned["test_objective_c2_term"]) == pytest.approx(0.0, abs=1e-12)


def test_internal_audit_sampler_top_span_prioritizes_large_merges_and_includes_root():
    rng = random.Random(0)
    # Root merge is index 6 (last); top-span should pick large spans next.
    merge_sizes = (2, 2, 2, 2, 4, 4, 8)
    idx = _sample_internal_audit_indices(
        n_internal=7,
        k=2,
        strategy="top_span",
        merge_sizes=merge_sizes,
        include_root=True,
        rng=rng,
    )
    assert idx is not None
    assert 6 in idx
    assert len(idx) == 2


def test_guided_eval_curve_q0_matches_learned_and_q1_hits_exact_for_additive():
    cfg = OPSCountConfig(
        n_regimes=3,
        vocab_size=48,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=8,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        train_docs=8,
        test_docs=6,
        model_family="additive",
        feature_mode="full",
        hidden_dim=32,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.5,
        c3_audit_strategy="uniform",
        c3_include_root=True,
        leaf_query_rate=1.0,
        include_root_query=True,
        eval_guidance_qs=(0.0, 0.5, 1.0),
        eval_guidance_trials=2,
        eval_guidance_seed_offset=12345,
        eval_guidance_include_root=True,
        violation_tau=0.0,
        seed=7,
        use_cuda=False,
        torch_threads=1,
    )
    out = run_markov_changepoint_ops_count_experiment(cfg)
    learned_root = float(((out.metrics.get("learned") or {}).get("root_mae", float("nan"))))
    guided = out.metrics.get("guided_eval_curve") or {}
    points = guided.get("points") or []
    by_q = {float(p["q"]): float(p["root_mae"]) for p in points if isinstance(p, dict) and "q" in p}

    assert 0.0 in by_q
    assert 1.0 in by_q
    assert abs(float(by_q[0.0]) - float(learned_root)) <= 1e-12
    assert float(by_q[1.0]) <= 1e-12


def test_guided_eval_curve_is_deterministic_for_fixed_seed_neural():
    cfg = OPSCountConfig(
        n_regimes=3,
        vocab_size=48,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=8,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        train_docs=6,
        test_docs=4,
        model_family="neural",
        feature_mode="full",
        state_dim=8,
        hidden_dim=24,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.5,
        c3_audit_strategy="uniform",
        c3_include_root=True,
        leaf_query_rate=1.0,
        include_root_query=True,
        eval_guidance_qs=(0.0, 0.5, 1.0),
        eval_guidance_trials=2,
        eval_guidance_seed_offset=2222,
        eval_guidance_include_root=True,
        violation_tau=0.0,
        seed=3,
        use_cuda=False,
        torch_threads=1,
    )
    out_a = run_markov_changepoint_ops_count_experiment(cfg)
    out_b = run_markov_changepoint_ops_count_experiment(cfg)
    ga = (out_a.metrics.get("guided_eval_curve") or {}).get("points")
    gb = (out_b.metrics.get("guided_eval_curve") or {}).get("points")
    assert ga == gb


def test_data_seed_holds_exact_corpus_fixed_across_model_seeds():
    common = dict(
        n_regimes=3,
        vocab_size=48,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=8,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        train_docs=8,
        test_docs=8,
        model_family="neural",
        feature_mode="full",
        state_dim=8,
        hidden_dim=24,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.5,
        c3_audit_strategy="uniform",
        c3_include_root=True,
        leaf_query_rate=1.0,
        include_root_query=True,
        violation_tau=0.0,
        data_seed=17,
        use_cuda=False,
        torch_threads=1,
    )
    out_a = run_markov_changepoint_ops_count_experiment(
        OPSCountConfig(**common, seed=0, model_seed=0)
    )
    out_b = run_markov_changepoint_ops_count_experiment(
        OPSCountConfig(**common, seed=1, model_seed=1)
    )

    assert int(out_a.config["effective_data_seed"]) == 17
    assert int(out_b.config["effective_data_seed"]) == 17
    assert int(out_a.config["effective_model_seed"]) == 0
    assert int(out_b.config["effective_model_seed"]) == 1

    exact_a = out_a.metrics["exact"]
    exact_b = out_b.metrics["exact"]
    for key in ("root_mae", "leaf_mae", "merge_mae", "schedule_spread_mean"):
        assert float(exact_a[key]) == pytest.approx(float(exact_b[key]))


def test_data_seed_holds_exact_test_corpus_fixed_across_hyperparameter_settings():
    common = dict(
        n_regimes=3,
        vocab_size=48,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=8,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        train_docs=8,
        val_docs=4,
        test_docs=8,
        model_family="neural",
        feature_mode="full",
        state_dim=8,
        hidden_dim=24,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.5,
        c3_audit_strategy="uniform",
        c3_include_root=True,
        leaf_query_rate=1.0,
        include_root_query=True,
        violation_tau=0.0,
        data_seed=21,
        model_seed=0,
        use_cuda=False,
        torch_threads=1,
    )
    out_a = run_markov_changepoint_ops_count_experiment(
        OPSCountConfig(**common, seed=0, local_law_weight=0.0, schedule_consistency_weight=0.0)
    )
    out_b = run_markov_changepoint_ops_count_experiment(
        OPSCountConfig(**common, seed=0, local_law_weight=0.9, schedule_consistency_weight=0.2)
    )

    exact_a = out_a.metrics["exact"]
    exact_b = out_b.metrics["exact"]
    for key in ("root_mae", "leaf_mae", "merge_mae", "schedule_spread_mean"):
        assert float(exact_a[key]) == pytest.approx(float(exact_b[key]))


def test_capability_classifier_threshold_boundaries():
    full = classify_capability(
        baseline_theorem_score=1.0,
        baseline_spread=1.0,
        baseline_root_mae=1.0,
        selected_theorem_score=0.89,
        selected_spread=0.89,
        selected_root_mae=1.05,
    )
    theorem_only = classify_capability(
        baseline_theorem_score=1.0,
        baseline_spread=1.0,
        baseline_root_mae=1.0,
        selected_theorem_score=0.89,
        selected_spread=0.89,
        selected_root_mae=1.06,
    )
    root_only = classify_capability(
        baseline_theorem_score=1.0,
        baseline_spread=1.0,
        baseline_root_mae=1.0,
        selected_theorem_score=0.95,
        selected_spread=0.95,
        selected_root_mae=1.02,
    )
    failure = classify_capability(
        baseline_theorem_score=1.0,
        baseline_spread=1.0,
        baseline_root_mae=1.0,
        selected_theorem_score=0.95,
        selected_spread=0.95,
        selected_root_mae=1.08,
    )

    assert full.capability_status == "full_success"
    assert theorem_only.capability_status == "theorem_only"
    assert root_only.capability_status == "root_only"
    assert failure.capability_status == "failure"


def test_learned_summary_exposes_train_side_local_law_diagnostics():
    cfg = OPSCountConfig(
        n_regimes=3,
        vocab_size=48,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=8,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        train_docs=8,
        test_docs=8,
        model_family="neural",
        feature_mode="full",
        state_dim=8,
        hidden_dim=24,
        n_epochs=2,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.5,
        c3_audit_strategy="uniform",
        c3_include_root=True,
        leaf_query_rate=1.0,
        include_root_query=True,
        violation_tau=0.0,
        seed=5,
        data_seed=11,
        model_seed=5,
        use_cuda=False,
        torch_threads=1,
    )
    out = run_markov_changepoint_ops_count_experiment(cfg)
    learned = out.metrics["learned"]
    learned_train = out.metrics["learned_train"]

    for key in (
        "train_root_mae",
        "train_leaf_mae",
        "train_c2_idempotence_mae",
        "train_merge_mae",
        "train_schedule_spread_mean",
        "generalization_gap_root_mae",
        "generalization_gap_leaf_mae",
        "generalization_gap_c2_idempotence_mae",
        "generalization_gap_merge_mae",
        "generalization_gap_schedule_spread_mean",
        "gap_to_exact_root_mae",
        "gap_to_exact_leaf_mae",
        "gap_to_exact_c2_idempotence_mae",
        "gap_to_exact_merge_mae",
        "train_loss_curve",
        "epochs_completed",
        "training_selection_metric_curve",
        "training_selection_mode",
        "training_selection_split",
        "training_selection_metric_name",
        "training_selection_metric_value",
        "training_selection_best_epoch",
    ):
        assert key in learned
    assert isinstance(learned_train, dict)
    assert len(list(learned["train_loss_curve"])) == 2
    assert int(learned["epochs_completed"]) == 2
    assert learned["training_selection_mode"] == "final_epoch_no_validation"
    assert learned["training_selection_split"] == "config"
    assert learned["training_selection_metric_name"] == "train_loss_final"
    assert len(list(learned["training_selection_metric_curve"])) == 2
    assert 0 <= int(learned["training_selection_best_epoch"]) < 2


def test_no_validation_run_marks_final_epoch_selection_mode() -> None:
    cfg = OPSCountConfig(
        n_regimes=3,
        vocab_size=48,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=8,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        train_docs=6,
        val_docs=0,
        test_docs=4,
        model_family="neural",
        feature_mode="full",
        state_dim=8,
        hidden_dim=24,
        n_epochs=2,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.5,
        c3_audit_strategy="uniform",
        c3_include_root=True,
        leaf_query_rate=1.0,
        include_root_query=True,
        violation_tau=0.0,
        seed=31,
        use_cuda=False,
        torch_threads=1,
    )
    out = run_markov_changepoint_ops_count_experiment(cfg)
    learned = out.metrics["learned"]

    assert learned["training_selection_mode"] == "final_epoch_no_validation"
    assert learned["training_selection_split"] == "config"
    assert learned["training_selection_metric_name"] == "train_loss_final"
    assert int(learned["training_selection_best_epoch"]) == 1
    assert len(list(learned["training_selection_metric_curve"])) == 2


def test_doc_level_baseline_emits_matched_no_tree_metrics(tmp_path: Path) -> None:
    cfg = OPSCountConfig(
        n_regimes=3,
        vocab_size=48,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=8,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        train_docs=8,
        val_docs=4,
        test_docs=6,
        model_family="neural",
        feature_mode="full",
        state_dim=8,
        hidden_dim=24,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.5,
        c3_audit_strategy="uniform",
        c3_include_root=True,
        leaf_query_rate=1.0,
        include_root_query=True,
        include_doc_level_baseline=True,
        include_doc_level_ridge_baseline=True,
        violation_tau=0.0,
        seed=37,
        use_cuda=False,
        torch_threads=1,
        artifact_dir=str(tmp_path / "artifacts"),
    )
    out = run_markov_changepoint_ops_count_experiment(cfg)
    doc_level = out.metrics.get("doc_level")
    doc_level_training = out.metrics.get("doc_level_training")
    doc_level_ridge = out.metrics.get("doc_level_ridge")
    doc_level_ridge_training = out.metrics.get("doc_level_ridge_training")
    assert isinstance(doc_level, dict)
    assert math.isfinite(float(doc_level.get("root_mae", float("nan"))))
    assert isinstance(doc_level_training, dict)
    assert doc_level_training["baseline_family"] == "neural"
    assert doc_level_training["input_view"] == "single_full_document_leaf"
    assert doc_level_training["uses_tree_merges"] is False
    assert doc_level_training["training_surface"] == "supervision_dataset"
    assert doc_level_training["supervision_mode"] == "dense_scalar_regression"
    assert doc_level_training["representation_kind"] == "dense_feature_vector"
    assert doc_level_training["target_kind"] == "scalar"
    assert doc_level_training["optimizer_family"] == "gradient_based_dense_regression"
    assert doc_level_training["optimizer_backend"] == "torch_mlp"
    assert int(doc_level_training["supervision_rows"]) == int(cfg.train_docs)
    supervision_path = Path(str(doc_level_training["supervision_artifact_path"]))
    assert supervision_path.exists()
    assert isinstance(doc_level_ridge, dict)
    assert math.isfinite(float(doc_level_ridge.get("root_mae", float("nan"))))
    assert isinstance(doc_level_ridge_training, dict)
    assert doc_level_ridge_training["baseline_family"] == "ridge"
    assert doc_level_ridge_training["input_view"] == "single_full_document_leaf"
    assert doc_level_ridge_training["uses_tree_merges"] is False
    assert doc_level_ridge_training["training_surface"] == "supervision_dataset"
    assert doc_level_ridge_training["supervision_mode"] == "dense_scalar_regression"
    assert doc_level_ridge_training["representation_kind"] == "dense_feature_vector"
    assert doc_level_ridge_training["target_kind"] == "scalar"
    assert (
        doc_level_ridge_training["optimizer_family"] == "closed_form_linear_regression"
    )
    assert doc_level_ridge_training["optimizer_backend"] == "closed_form_ridge"
    assert float(doc_level_ridge_training["ridge_alpha"]) == 1.0
    assert int(doc_level_ridge_training["supervision_rows"]) == int(cfg.train_docs)
    assert Path(str(doc_level_ridge_training["supervision_artifact_path"])).exists()


def test_leaf_ridge_tree_baseline_emits_finite_metrics_and_training_metadata(
    tmp_path: Path,
) -> None:
    cfg = OPSCountConfig(
        n_regimes=3,
        vocab_size=48,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=8,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        train_docs=8,
        val_docs=4,
        test_docs=6,
        model_family="neural",
        feature_mode="full",
        state_dim=8,
        hidden_dim=24,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.5,
        c3_audit_strategy="uniform",
        c3_include_root=True,
        leaf_query_rate=0.5,
        include_root_query=True,
        include_leaf_ridge_tree_baseline=True,
        violation_tau=0.0,
        seed=41,
        use_cuda=False,
        torch_threads=1,
        artifact_dir=str(tmp_path / "artifacts"),
    )
    out = run_markov_changepoint_ops_count_experiment(cfg)
    baseline = out.metrics.get("leaf_ridge_tree")
    training = out.metrics.get("leaf_ridge_tree_training")
    assert isinstance(baseline, dict)
    assert math.isfinite(float(baseline.get("root_mae", float("nan"))))
    assert math.isfinite(float(baseline.get("leaf_mae", float("nan"))))
    assert math.isfinite(float(baseline.get("merge_mae", float("nan"))))
    assert isinstance(training, dict)
    assert training["baseline_family"] == "leaf_ridge_tree"
    assert training["input_view"] == "sampled_leaf_core_features"
    assert training["uses_tree_merges"] is True
    assert training["training_surface"] == "supervision_dataset"
    assert training["supervision_mode"] == "dense_scalar_regression"
    assert training["representation_kind"] == "dense_feature_vector"
    assert training["target_kind"] == "scalar"
    assert training["optimizer_family"] == "closed_form_linear_regression"
    assert training["optimizer_backend"] == "closed_form_ridge"
    assert math.isclose(float(training["leaf_query_rate"]), 0.5)
    assert float(training["ridge_alpha"]) == 1.0
    assert int(training["supervision_rows"]) > 0
    assert Path(str(training["supervision_artifact_path"])).exists()


def test_leaf_knn_tree_baseline_emits_finite_metrics_and_training_metadata(
    tmp_path: Path,
) -> None:
    cfg = OPSCountConfig(
        n_regimes=3,
        vocab_size=48,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=8,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        train_docs=8,
        val_docs=4,
        test_docs=6,
        model_family="neural",
        feature_mode="full",
        state_dim=8,
        hidden_dim=24,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.5,
        c3_audit_strategy="uniform",
        c3_include_root=True,
        leaf_query_rate=0.5,
        include_root_query=True,
        include_leaf_knn_tree_baseline=True,
        leaf_knn_neighbors=4,
        violation_tau=0.0,
        seed=43,
        use_cuda=False,
        torch_threads=1,
        artifact_dir=str(tmp_path / "artifacts"),
    )
    out = run_markov_changepoint_ops_count_experiment(cfg)
    baseline = out.metrics.get("leaf_knn_tree")
    training = out.metrics.get("leaf_knn_tree_training")
    assert isinstance(baseline, dict)
    assert math.isfinite(float(baseline.get("root_mae", float("nan"))))
    assert math.isfinite(float(baseline.get("leaf_mae", float("nan"))))
    assert math.isfinite(float(baseline.get("merge_mae", float("nan"))))
    assert float(baseline.get("schedule_spread_mean", float("nan"))) == 0.0
    assert isinstance(training, dict)
    assert training["baseline_family"] == "leaf_knn_tree"
    assert training["input_view"] == "sampled_leaf_core_features"
    assert training["uses_tree_merges"] is True
    assert training["training_surface"] == "supervision_dataset"
    assert (
        training["supervision_mode"]
        == "dense_feature_vector__scalar__instance_based_local_regression"
    )
    assert training["representation_kind"] == "dense_feature_vector"
    assert training["target_kind"] == "scalar"
    assert training["optimizer_family"] == "instance_based_local_regression"
    assert training["optimizer_backend"] == "distance_weighted_knn"
    assert math.isclose(float(training["leaf_query_rate"]), 0.5)
    assert int(training["knn_neighbors"]) == 4
    assert int(training["supervision_rows"]) > 0
    assert Path(str(training["supervision_artifact_path"])).exists()


def test_leaf_endpoint_table_tree_baseline_emits_finite_metrics_and_training_metadata(
    tmp_path: Path,
) -> None:
    cfg = OPSCountConfig(
        n_regimes=3,
        vocab_size=48,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=8,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        train_docs=8,
        val_docs=4,
        test_docs=6,
        model_family="neural",
        feature_mode="full",
        state_dim=8,
        hidden_dim=24,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.5,
        c3_audit_strategy="uniform",
        c3_include_root=True,
        leaf_query_rate=0.5,
        include_root_query=True,
        include_leaf_endpoint_table_tree_baseline=True,
        violation_tau=0.0,
        seed=47,
        use_cuda=False,
        torch_threads=1,
        artifact_dir=str(tmp_path / "artifacts"),
    )
    out = run_markov_changepoint_ops_count_experiment(cfg)
    baseline = out.metrics.get("leaf_endpoint_table_tree")
    training = out.metrics.get("leaf_endpoint_table_tree_training")
    assert isinstance(baseline, dict)
    assert math.isfinite(float(baseline.get("root_mae", float("nan"))))
    assert math.isfinite(float(baseline.get("leaf_mae", float("nan"))))
    assert math.isfinite(float(baseline.get("merge_mae", float("nan"))))
    assert float(baseline.get("schedule_spread_mean", float("nan"))) == 0.0
    assert isinstance(training, dict)
    assert training["baseline_family"] == "leaf_endpoint_table_tree"
    assert training["input_view"] == "sampled_leaf_endpoints_length"
    assert training["uses_tree_merges"] is True
    assert training["training_surface"] == "supervision_dataset"
    assert (
        training["supervision_mode"]
        == "dense_feature_vector__scalar__piecewise_local_regression"
    )
    assert training["representation_kind"] == "dense_feature_vector"
    assert training["target_kind"] == "scalar"
    assert training["optimizer_family"] == "piecewise_local_regression"
    assert training["optimizer_backend"] == "endpoint_length_group_mean"
    assert math.isclose(float(training["leaf_query_rate"]), 0.5)
    assert int(training["supervision_rows"]) > 0
    assert Path(str(training["supervision_artifact_path"])).exists()


def test_leaf_rf_tree_baseline_emits_finite_metrics_and_training_metadata(
    tmp_path: Path,
) -> None:
    cfg = OPSCountConfig(
        n_regimes=3,
        vocab_size=48,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=8,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        train_docs=8,
        val_docs=4,
        test_docs=6,
        model_family="neural",
        feature_mode="full",
        state_dim=8,
        hidden_dim=24,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.5,
        c3_audit_strategy="uniform",
        c3_include_root=True,
        leaf_query_rate=0.5,
        include_root_query=True,
        include_leaf_rf_tree_baseline=True,
        rf_n_estimators=32,
        rf_max_depth=8,
        rf_min_samples_leaf=2,
        violation_tau=0.0,
        seed=53,
        use_cuda=False,
        torch_threads=1,
        artifact_dir=str(tmp_path / "artifacts"),
    )
    out = run_markov_changepoint_ops_count_experiment(cfg)
    baseline = out.metrics.get("leaf_rf_tree")
    training = out.metrics.get("leaf_rf_tree_training")
    assert isinstance(baseline, dict)
    assert math.isfinite(float(baseline.get("root_mae", float("nan"))))
    assert math.isfinite(float(baseline.get("leaf_mae", float("nan"))))
    assert math.isfinite(float(baseline.get("merge_mae", float("nan"))))
    assert float(baseline.get("schedule_spread_mean", float("nan"))) == 0.0
    assert isinstance(training, dict)
    assert training["baseline_family"] == "leaf_rf_tree"
    assert training["input_view"] == "sampled_leaf_core_features"
    assert training["uses_tree_merges"] is True
    assert training["training_surface"] == "supervision_dataset"
    assert (
        training["supervision_mode"]
        == "dense_feature_vector__scalar__tree_ensemble_regression"
    )
    assert training["representation_kind"] == "dense_feature_vector"
    assert training["target_kind"] == "scalar"
    assert training["optimizer_family"] == "tree_ensemble_regression"
    assert training["optimizer_backend"] == "random_forest_regressor"
    assert int(training["rf_n_estimators"]) == 32
    assert int(training["rf_max_depth"]) == 8
    assert int(training["rf_min_samples_leaf"]) == 2
    assert int(training["supervision_rows"]) > 0
    assert Path(str(training["supervision_artifact_path"])).exists()


def test_leaf_dt_tree_baseline_emits_finite_metrics_and_training_metadata(
    tmp_path: Path,
) -> None:
    cfg = OPSCountConfig(
        n_regimes=3,
        vocab_size=48,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=8,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        train_docs=8,
        val_docs=4,
        test_docs=6,
        model_family="neural",
        feature_mode="full",
        state_dim=8,
        hidden_dim=24,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.5,
        c3_audit_strategy="uniform",
        c3_include_root=True,
        leaf_query_rate=0.5,
        include_root_query=True,
        include_leaf_dt_tree_baseline=True,
        rf_max_depth=8,
        rf_min_samples_leaf=2,
        violation_tau=0.0,
        seed=59,
        use_cuda=False,
        torch_threads=1,
        artifact_dir=str(tmp_path / "artifacts"),
    )
    out = run_markov_changepoint_ops_count_experiment(cfg)
    baseline = out.metrics.get("leaf_dt_tree")
    training = out.metrics.get("leaf_dt_tree_training")
    assert isinstance(baseline, dict)
    assert math.isfinite(float(baseline.get("root_mae", float("nan"))))
    assert math.isfinite(float(baseline.get("leaf_mae", float("nan"))))
    assert math.isfinite(float(baseline.get("merge_mae", float("nan"))))
    assert float(baseline.get("schedule_spread_mean", float("nan"))) == 0.0
    assert isinstance(training, dict)
    assert training["baseline_family"] == "leaf_dt_tree"
    assert training["input_view"] == "sampled_leaf_core_features"
    assert training["uses_tree_merges"] is True
    assert training["training_surface"] == "supervision_dataset"
    assert (
        training["supervision_mode"] == "dense_feature_vector__scalar__tree_regression"
    )
    assert training["representation_kind"] == "dense_feature_vector"
    assert training["target_kind"] == "scalar"
    assert training["optimizer_family"] == "tree_regression"
    assert training["optimizer_backend"] == "decision_tree_regressor"
    assert int(training["tree_max_depth"]) == 8
    assert int(training["tree_min_samples_leaf"]) == 2
    assert int(training["supervision_rows"]) > 0
    assert Path(str(training["supervision_artifact_path"])).exists()


def test_sampled_leaf_pool_sweep_emits_finite_metrics_and_efficiency_artifacts(
    tmp_path: Path,
) -> None:
    cfg = OPSCountConfig(
        n_regimes=3,
        vocab_size=48,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=8,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        train_docs=8,
        val_docs=4,
        test_docs=6,
        model_family="neural",
        feature_mode="full",
        state_dim=8,
        hidden_dim=24,
        n_epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.5,
        c3_audit_strategy="uniform",
        c3_include_root=True,
        leaf_query_rate=1.0,
        include_root_query=True,
        include_sampled_leaf_pool_ridge_baseline=True,
        include_sampled_leaf_pool_rf_baseline=True,
        sampled_leaf_pool_leaf_counts=(1, 2, 4),
        violation_tau=0.0,
        seed=41,
        use_cuda=False,
        torch_threads=1,
        artifact_dir=str(tmp_path / "artifacts"),
    )
    out = run_markov_changepoint_ops_count_experiment(cfg)
    sweep = out.metrics.get("sampled_leaf_pool_budget_sweep")
    assert isinstance(sweep, dict)
    points = list(sweep.get("points") or [])
    assert [int(point["leaf_budget"]) for point in points] == [1, 2, 4]
    for point in points:
        ridge = dict(point.get("ridge", {}) or {})
        rf = dict(point.get("rf", {}) or {})
        ridge_training = dict(point.get("ridge_training", {}) or {})
        rf_training = dict(point.get("rf_training", {}) or {})
        test_obs = dict(point.get("test_observation", {}) or {})
        assert math.isfinite(float(ridge["root_mae"]))
        assert math.isfinite(float(rf["root_mae"]))
        assert ridge_training["input_view"] == "sampled_leaf_pool_uniform"
        assert rf_training["input_view"] == "sampled_leaf_pool_uniform"
        assert int(ridge_training["sample_leaf_budget"]) == int(point["leaf_budget"])
        assert int(rf_training["sample_leaf_budget"]) == int(point["leaf_budget"])
        assert math.isfinite(float(test_obs["sampled_leaves_mean"]))
        assert math.isfinite(float(test_obs["sampled_tokens_mean"]))
        assert math.isfinite(float(test_obs["sampled_token_fraction_mean"]))
        assert Path(str(ridge_training["supervision_artifact_path"])).exists()


def test_exact_family_mode_skips_learned_training_and_emits_stress_family():
    cfg = OPSCountConfig(
        n_regimes=3,
        vocab_size=48,
        min_tokens=64,
        max_tokens=64,
        min_segments=4,
        max_segments=4,
        min_seg_len=8,
        max_seg_len=16,
        fixed_leaf_tokens=16,
        train_docs=4,
        test_docs=4,
        exact_family="flip_R2",
        use_cuda=False,
        torch_threads=1,
    )
    out = run_markov_changepoint_ops_count_experiment(cfg)
    assert "stress_family" in out.metrics
    assert "learned" not in out.metrics
    stress = out.metrics["stress_family"]
    assert stress["stress_family_name"] == "flip_R2"
    assert float(stress["test_c2_r2_mae_n"]) >= 0.0


def test_token_full_feature_mode_supports_observed_token_doc_learning(tmp_path: Path) -> None:
    cfg = OPSCountConfig(
        n_regimes=4,
        vocab_size=16,
        generator_profile="piecewise_palette",
        min_tokens=96,
        max_tokens=96,
        min_segments=2,
        max_segments=6,
        min_seg_len=8,
        max_seg_len=24,
        fixed_leaf_tokens=16,
        train_docs=64,
        val_docs=16,
        test_docs=32,
        model_family="neural",
        feature_mode="token_full",
        state_dim=16,
        hidden_dim=64,
        n_epochs=4,
        batch_size=8,
        lr=5e-4,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.25,
        c3_audit_strategy="uniform",
        c3_include_root=True,
        leaf_query_rate=0.5,
        include_root_query=True,
        include_doc_level_baseline=True,
        include_doc_level_ridge_baseline=True,
        include_rf_root_baseline=True,
        violation_tau=0.0,
        seed=7,
        use_cuda=False,
        torch_threads=1,
        artifact_dir=str(tmp_path / "artifacts"),
    )
    out = run_markov_changepoint_ops_count_experiment(cfg)
    learned = dict(out.metrics.get("learned", {}) or {})
    doc_level = dict(out.metrics.get("doc_level", {}) or {})
    doc_level_ridge = dict(out.metrics.get("doc_level_ridge", {}) or {})
    rf_root = dict(out.metrics.get("rf_root", {}) or {})
    doc_level_training = dict(out.metrics.get("doc_level_training", {}) or {})
    doc_level_ridge_training = dict(out.metrics.get("doc_level_ridge_training", {}) or {})
    undersupported = dict(out.metrics.get("undersupported", {}) or {})

    assert out.config["feature_mode"] == "token_full"
    assert out.config["train_target_diagnostics"]["is_constant"] is False
    assert math.isfinite(float(learned["root_mae"]))
    assert math.isfinite(float(doc_level["root_mae"]))
    assert math.isfinite(float(doc_level_ridge["root_mae"]))
    assert math.isfinite(float(rf_root["root_mae"]))
    assert math.isfinite(float(undersupported["root_mae"]))
    assert doc_level_training["input_view"] == "single_full_document_leaf"
    assert doc_level_training["baseline_family"] == "neural"
    assert doc_level_ridge_training["input_view"] == "single_full_document_leaf"
    assert doc_level_ridge_training["optimizer_backend"] == "closed_form_ridge"


def test_markov_data_bundle_reuse_preserves_split_signatures_across_comparisons(
    tmp_path: Path,
) -> None:
    base_cfg = OPSCountConfig(
        n_regimes=4,
        vocab_size=16,
        generator_profile="piecewise_palette",
        min_tokens=96,
        max_tokens=96,
        min_segments=2,
        max_segments=6,
        min_seg_len=8,
        max_seg_len=24,
        fixed_leaf_tokens=16,
        train_docs=64,
        val_docs=16,
        test_docs=32,
        model_family="neural",
        feature_mode="token_full",
        state_dim=16,
        hidden_dim=64,
        n_epochs=4,
        batch_size=8,
        lr=5e-4,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.25,
        c3_audit_strategy="uniform",
        c3_include_root=True,
        leaf_query_rate=0.5,
        include_root_query=True,
        include_doc_level_baseline=True,
        include_doc_level_ridge_baseline=True,
        include_rf_root_baseline=True,
        violation_tau=0.0,
        seed=11,
        use_cuda=False,
        torch_threads=1,
        artifact_dir=str(tmp_path / "root_only_artifacts"),
    )
    bundle_path = tmp_path / "markov_bundle.json"
    build_markov_changepoint_ops_count_data_bundle(base_cfg).save(bundle_path)
    bundle = MarkovOPSDataBundle.load(bundle_path)

    root_only = run_markov_changepoint_ops_count_experiment(base_cfg, data_bundle=bundle)
    local_label_cfg = OPSCountConfig(
        **{
            **base_cfg.__dict__,
            "local_law_weight": 0.5,
            "c1_relative_weight": 1.0,
            "c2_relative_weight": 0.0,
            "c3_relative_weight": 1.0,
            "audit_fraction": 1.0,
            "leaf_query_rate": 1.0,
            "artifact_dir": str(tmp_path / "local_label_artifacts"),
        }
    )
    local_label = run_markov_changepoint_ops_count_experiment(
        local_label_cfg, data_bundle=bundle
    )

    for summary in (root_only, local_label):
        assert summary.config["data_bundle_source"] == "provided"
        assert summary.config["train_corpus_signature"] == bundle.train_corpus_signature
        assert summary.config["val_corpus_signature"] == bundle.val_corpus_signature
        assert summary.config["test_corpus_signature"] == bundle.test_corpus_signature
        assert summary.config["degenerate_root_target_detected"] is False
        assert math.isfinite(float(summary.metrics["learned"]["root_mae"]))

    assert (
        root_only.config["train_corpus_signature"]
        == local_label.config["train_corpus_signature"]
    )
    assert root_only.config["val_corpus_signature"] == local_label.config["val_corpus_signature"]
    assert (
        root_only.config["test_corpus_signature"]
        == local_label.config["test_corpus_signature"]
    )


def test_disjoint_palette_doc_level_ridge_breakdown_shows_bigram_signal(tmp_path: Path) -> None:
    cfg = OPSCountConfig(
        n_regimes=4,
        vocab_size=16,
        generator_profile="piecewise_disjoint_palette",
        min_tokens=96,
        max_tokens=96,
        min_segments=2,
        max_segments=6,
        min_seg_len=8,
        max_seg_len=24,
        fixed_leaf_tokens=16,
        train_docs=1024,
        val_docs=128,
        test_docs=256,
        model_family="neural",
        feature_mode="token_full",
        state_dim=16,
        hidden_dim=64,
        n_epochs=2,
        batch_size=32,
        lr=5e-4,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.0,
        c3_audit_strategy="uniform",
        c3_include_root=True,
        leaf_query_rate=0.0,
        include_root_query=True,
        include_doc_level_ridge_baseline=True,
        doc_level_ridge_alpha=0.0,
        doc_level_ridge_breakdown_orders=(1, 2, 3),
        violation_tau=0.0,
        seed=37,
        use_cuda=False,
        torch_threads=1,
        artifact_dir=str(tmp_path / "artifacts"),
    )
    out = run_markov_changepoint_ops_count_experiment(cfg)
    unigram = dict(out.metrics.get("doc_level_ridge_unigram", {}) or {})
    bigram = dict(out.metrics.get("doc_level_ridge_bigram", {}) or {})
    trigram = dict(out.metrics.get("doc_level_ridge_trigram", {}) or {})
    unigram_training = dict(out.metrics.get("doc_level_ridge_unigram_training", {}) or {})
    bigram_training = dict(out.metrics.get("doc_level_ridge_bigram_training", {}) or {})
    trigram_training = dict(out.metrics.get("doc_level_ridge_trigram_training", {}) or {})

    assert math.isfinite(float(unigram["root_mae"]))
    assert math.isfinite(float(bigram["root_mae"]))
    assert math.isfinite(float(trigram["root_mae"]))
    assert float(bigram["root_mae"]) < float(unigram["root_mae"])
    assert float(bigram["root_mae"]) < 1e-4
    assert unigram_training["input_view"] == "full_document_token_unigram_counts"
    assert bigram_training["input_view"] == "full_document_token_bigram_counts"
    assert trigram_training["input_view"] == "full_document_token_trigram_counts"
    assert bigram_training["ngram_orders"] == [2]
    assert trigram_training["ngram_orders"] == [3]


def test_root_target_diagnostics_flag_constant_fixed_segment_bundle() -> None:
    cfg = OPSCountConfig(
        n_regimes=4,
        vocab_size=16,
        generator_profile="piecewise_palette",
        min_tokens=96,
        max_tokens=96,
        min_segments=6,
        max_segments=6,
        min_seg_len=8,
        max_seg_len=24,
        train_docs=8,
        val_docs=4,
        test_docs=4,
        seed=19,
        use_cuda=False,
        torch_threads=1,
    )
    out = run_markov_changepoint_ops_count_experiment(cfg)
    assert out.config["degenerate_root_target_detected"] is True
    assert out.config["train_target_diagnostics"]["is_constant"] is True
    assert out.config["test_target_diagnostics"]["n_unique"] == 1


def test_hazard_aliased_generator_preserves_changepoint_truth_and_all_tokens() -> None:
    cfg = OPSCountConfig(
        n_regimes=5,
        vocab_size=24,
        generator_profile="hazard_aliased",
        min_tokens=96,
        max_tokens=96,
        min_segments=6,
        max_segments=6,
        min_seg_len=8,
        max_seg_len=24,
        train_docs=4,
        val_docs=2,
        test_docs=3,
        seed=23,
        use_cuda=False,
        torch_threads=1,
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(cfg)
    docs = tuple(bundle.train_docs) + tuple(bundle.val_docs) + tuple(bundle.test_docs)
    assert len(docs) == 9
    for doc in docs:
        n = len(doc.tokens)
        assert n == 96
        assert len(doc.token_regimes) == n
        assert len(doc.transition_regimes) == n - 1
        recovered = tuple(
            idx
            for idx, (a, b) in enumerate(zip(doc.token_regimes[:-1], doc.token_regimes[1:]))
            if int(a) != int(b)
        )
        assert tuple(int(x) for x in doc.true_boundaries) == recovered
        assert len(doc.true_boundaries) == sum(
            int(a) != int(b) for a, b in zip(doc.token_regimes[:-1], doc.token_regimes[1:])
        )
        spans = _leaf_spans(n, leaf_tokens=16)
        assert spans[0][0] == 0
        assert spans[-1][1] == n


def test_disjoint_palette_generator_makes_regime_color_recoverable_from_tokens() -> None:
    cfg = OPSCountConfig(
        n_regimes=4,
        vocab_size=16,
        generator_profile="piecewise_disjoint_palette",
        min_tokens=96,
        max_tokens=96,
        min_segments=2,
        max_segments=6,
        min_seg_len=8,
        max_seg_len=24,
        train_docs=4,
        val_docs=2,
        test_docs=3,
        seed=29,
        use_cuda=False,
        torch_threads=1,
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(cfg)
    docs = tuple(bundle.train_docs) + tuple(bundle.val_docs) + tuple(bundle.test_docs)
    token_blocks = np.array_split(np.arange(int(cfg.vocab_size), dtype=np.int64), int(cfg.n_regimes))
    token_to_regime = {
        int(token): int(regime)
        for regime, block in enumerate(token_blocks)
        for token in block.tolist()
    }
    assert len(token_to_regime) == int(cfg.vocab_size)
    for doc in docs:
        n = len(doc.tokens)
        spans = _leaf_spans(n, leaf_tokens=16)
        recovered_regimes = tuple(token_to_regime[int(token)] for token in doc.tokens)
        assert recovered_regimes == tuple(int(x) for x in doc.token_regimes)
        recovered_boundaries = tuple(
            idx
            for idx, (a, b) in enumerate(zip(recovered_regimes[:-1], recovered_regimes[1:]))
            if int(a) != int(b)
        )
        assert recovered_boundaries == tuple(int(x) for x in doc.true_boundaries)
        assert sum(end - start for start, end in spans) == n
        assert all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))


def test_piecewise_palette_generator_preserves_changepoint_truth_and_all_tokens() -> None:
    cfg = OPSCountConfig(
        n_regimes=4,
        vocab_size=16,
        generator_profile="piecewise_palette",
        min_tokens=96,
        max_tokens=96,
        min_segments=2,
        max_segments=6,
        min_seg_len=8,
        max_seg_len=32,
        train_docs=4,
        val_docs=2,
        test_docs=3,
        seed=37,
        use_cuda=False,
        torch_threads=1,
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(cfg)
    docs = tuple(bundle.train_docs) + tuple(bundle.val_docs) + tuple(bundle.test_docs)
    assert len(docs) == 9
    for doc in docs:
        n = len(doc.tokens)
        assert n == 96
        assert len(doc.token_regimes) == n
        assert len(doc.transition_regimes) == n - 1
        recovered = tuple(
            idx
            for idx, (a, b) in enumerate(zip(doc.token_regimes[:-1], doc.token_regimes[1:]))
            if int(a) != int(b)
        )
        assert tuple(int(x) for x in doc.true_boundaries) == recovered
        spans = _leaf_spans(n, leaf_tokens=16)
        assert spans[0][0] == 0
        assert spans[-1][1] == n
        assert sum(end - start for start, end in spans) == n


def test_token_bow_feature_mode_emits_finite_observed_token_metrics(tmp_path: Path) -> None:
    cfg = OPSCountConfig(
        n_regimes=4,
        vocab_size=16,
        min_tokens=96,
        max_tokens=96,
        min_segments=6,
        max_segments=6,
        min_seg_len=8,
        max_seg_len=24,
        fixed_leaf_tokens=16,
        train_docs=64,
        val_docs=16,
        test_docs=32,
        model_family="neural",
        feature_mode="token_bow",
        state_dim=16,
        hidden_dim=64,
        n_epochs=4,
        batch_size=8,
        lr=5e-4,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.0,
        c3_audit_strategy="uniform",
        c3_include_root=True,
        leaf_query_rate=0.0,
        include_root_query=True,
        include_doc_level_baseline=True,
        include_doc_level_ridge_baseline=True,
        include_rf_root_baseline=True,
        violation_tau=0.0,
        seed=13,
        use_cuda=False,
        torch_threads=1,
        artifact_dir=str(tmp_path / "artifacts"),
    )
    out = run_markov_changepoint_ops_count_experiment(cfg)
    learned = dict(out.metrics.get("learned", {}) or {})
    doc_level = dict(out.metrics.get("doc_level", {}) or {})
    doc_level_ridge = dict(out.metrics.get("doc_level_ridge", {}) or {})
    rf_root = dict(out.metrics.get("rf_root", {}) or {})

    assert out.config["feature_mode"] == "token_bow"
    assert math.isfinite(float(learned["root_mae"]))
    assert math.isfinite(float(doc_level["root_mae"]))
    assert math.isfinite(float(doc_level_ridge["root_mae"]))
    assert math.isfinite(float(rf_root["root_mae"]))


def test_hazard_aliased_profile_emits_finite_observed_token_metrics(tmp_path: Path) -> None:
    cfg = OPSCountConfig(
        n_regimes=6,
        vocab_size=32,
        generator_profile="hazard_aliased",
        min_tokens=128,
        max_tokens=128,
        min_segments=8,
        max_segments=8,
        min_seg_len=8,
        max_seg_len=24,
        fixed_leaf_tokens=16,
        train_docs=96,
        val_docs=24,
        test_docs=48,
        model_family="neural",
        feature_mode="token_bow",
        state_dim=24,
        hidden_dim=96,
        n_epochs=4,
        batch_size=8,
        lr=5e-4,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.0,
        c3_audit_strategy="uniform",
        c3_include_root=True,
        leaf_query_rate=0.0,
        include_root_query=True,
        include_doc_level_baseline=True,
        include_doc_level_ridge_baseline=True,
        include_rf_root_baseline=True,
        violation_tau=0.0,
        seed=29,
        use_cuda=False,
        torch_threads=1,
        artifact_dir=str(tmp_path / "artifacts"),
    )
    out = run_markov_changepoint_ops_count_experiment(cfg)
    learned = dict(out.metrics.get("learned", {}) or {})
    doc_level = dict(out.metrics.get("doc_level", {}) or {})
    doc_level_ridge = dict(out.metrics.get("doc_level_ridge", {}) or {})
    rf_root = dict(out.metrics.get("rf_root", {}) or {})
    exact = dict(out.metrics.get("exact", {}) or {})

    assert out.config["generator_profile"] == "hazard_aliased"
    assert out.config["feature_mode"] == "token_bow"
    assert math.isfinite(float(learned["root_mae"]))
    assert math.isfinite(float(doc_level["root_mae"]))
    assert math.isfinite(float(doc_level_ridge["root_mae"]))
    assert math.isfinite(float(rf_root["root_mae"]))
    assert float(exact["root_mae"]) == 0.0


def test_doc_sequence_baseline_emits_finite_metrics_and_training_metadata(
    tmp_path: Path,
) -> None:
    cfg = OPSCountConfig(
        n_regimes=4,
        vocab_size=16,
        generator_profile="piecewise_markov",
        min_tokens=96,
        max_tokens=96,
        min_segments=2,
        max_segments=6,
        min_seg_len=8,
        max_seg_len=32,
        fixed_leaf_tokens=16,
        train_docs=96,
        val_docs=24,
        test_docs=48,
        model_family="neural",
        feature_mode="token_full",
        state_dim=16,
        hidden_dim=64,
        n_epochs=3,
        batch_size=8,
        lr=5e-4,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.0,
        c3_audit_strategy="uniform",
        c3_include_root=True,
        leaf_query_rate=0.0,
        include_root_query=True,
        include_doc_sequence_baseline=True,
        violation_tau=0.0,
        seed=31,
        use_cuda=False,
        torch_threads=1,
        artifact_dir=str(tmp_path / "artifacts"),
    )
    out = run_markov_changepoint_ops_count_experiment(cfg)
    baseline = dict(out.metrics.get("doc_sequence", {}) or {})
    training = dict(out.metrics.get("doc_sequence_training", {}) or {})
    assert math.isfinite(float(baseline.get("root_mae", float("nan"))))
    assert math.isfinite(float(baseline.get("leaf_mae", float("nan"))))
    assert float(baseline.get("schedule_spread_mean", float("nan"))) == 0.0
    assert training["input_view"] == "full_document_token_sequence"
    assert training["baseline_family"] == "official_neuraloperator_fno"
    assert training["uses_tree_merges"] is False
    assert training["optimizer_backend"] == "official_neuraloperator_fno_count_classifier"
    assert training["operator_backend"] == "official_neuraloperator_package"
    assert training["token_embedding_backend"] == "learned_token_embedding"
    assert training["readout_mode"] == "count_support_classification"
    assert training["root_summary_auxiliary_heads"] == []
    assert training["root_label_only_supervision"] is True
    assert training["doc_sequence_objective_requested"] == "count_ce_only"
    assert training["doc_sequence_objective_effective"] == "count_ce_only"
    assert training["fno_n_layers"] == 4
    assert training["sequence_input_backend"] == "shared_token_sequence_arrays"
    assert set(training["sequence_input_signatures"]) == {"train", "val", "test"}
    assert math.isfinite(float(training["test_exact_match_rate"]))


def test_doc_transformer_baseline_emits_finite_metrics_and_training_metadata(
    tmp_path: Path,
) -> None:
    cfg = OPSCountConfig(
        n_regimes=4,
        vocab_size=16,
        generator_profile="piecewise_markov",
        min_tokens=96,
        max_tokens=96,
        min_segments=2,
        max_segments=6,
        min_seg_len=8,
        max_seg_len=32,
        fixed_leaf_tokens=16,
        train_docs=96,
        val_docs=24,
        test_docs=48,
        model_family="neural",
        feature_mode="token_full",
        state_dim=16,
        hidden_dim=64,
        n_epochs=3,
        batch_size=8,
        lr=5e-4,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.0,
        c3_audit_strategy="uniform",
        c3_include_root=True,
        leaf_query_rate=0.0,
        include_root_query=True,
        include_doc_transformer_baseline=True,
        violation_tau=0.0,
        seed=31,
        use_cuda=False,
        torch_threads=1,
        artifact_dir=str(tmp_path / "artifacts"),
    )
    out = run_markov_changepoint_ops_count_experiment(cfg)
    baseline = dict(out.metrics.get("doc_transformer", {}) or {})
    training = dict(out.metrics.get("doc_transformer_training", {}) or {})
    assert math.isfinite(float(baseline.get("root_mae", float("nan"))))
    assert math.isfinite(float(baseline.get("leaf_mae", float("nan"))))
    assert float(baseline.get("schedule_spread_mean", float("nan"))) == 0.0
    assert training["input_view"] == "full_document_token_sequence"
    assert training["baseline_family"] == "full_sequence_boundary_transformer"
    assert training["uses_tree_merges"] is False
    assert training["optimizer_backend"] == "full_sequence_boundary_transformer_regression_count_classifier"
    assert training["token_embedding_backend"] == "learned_token_embedding"
    assert training["position_embedding_backend"] == "learned_position_embedding"
    assert training["readout_mode"] == "summed_boundary_probabilities_with_count_classification"
    assert training["root_summary_auxiliary_heads"] == ["count_class"]
    assert training["root_label_only_supervision"] is True
    assert training["doc_transformer_head_family"] == "boundary_sum_count_hybrid"
    assert training["sequence_input_backend"] == "shared_token_sequence_arrays"
    assert set(training["sequence_input_signatures"]) == {"train", "val", "test"}
    assert math.isfinite(float(training["test_exact_match_rate"]))


def test_full_document_neural_baselines_share_exact_sequence_inputs(
    tmp_path: Path,
) -> None:
    cfg = OPSCountConfig(
        n_regimes=4,
        vocab_size=16,
        generator_profile="piecewise_markov",
        min_tokens=96,
        max_tokens=96,
        min_segments=2,
        max_segments=6,
        min_seg_len=8,
        max_seg_len=32,
        fixed_leaf_tokens=16,
        train_docs=96,
        val_docs=24,
        test_docs=48,
        model_family="neural",
        feature_mode="token_full",
        state_dim=16,
        hidden_dim=64,
        n_epochs=3,
        batch_size=8,
        lr=5e-4,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.0,
        c3_audit_strategy="uniform",
        c3_include_root=True,
        leaf_query_rate=0.0,
        include_root_query=True,
        include_doc_sequence_baseline=True,
        include_doc_transformer_baseline=True,
        violation_tau=0.0,
        seed=31,
        use_cuda=False,
        torch_threads=1,
        artifact_dir=str(tmp_path / "artifacts"),
    )
    out = run_markov_changepoint_ops_count_experiment(cfg)
    seq_training = dict(out.metrics.get("doc_sequence_training", {}) or {})
    tr_training = dict(out.metrics.get("doc_transformer_training", {}) or {})
    assert seq_training["sequence_input_backend"] == "shared_token_sequence_arrays"
    assert tr_training["sequence_input_backend"] == "shared_token_sequence_arrays"
    assert seq_training["sequence_input_signatures"] == tr_training["sequence_input_signatures"]
    assert out.config["full_sequence_input_signatures"] == seq_training["sequence_input_signatures"]


def test_full_document_neural_baseline_objective_and_head_family_overrides(
    tmp_path: Path,
) -> None:
    cfg = OPSCountConfig(
        n_regimes=4,
        vocab_size=16,
        generator_profile="piecewise_disjoint_palette",
        min_tokens=96,
        max_tokens=96,
        min_segments=2,
        max_segments=6,
        min_seg_len=8,
        max_seg_len=24,
        fixed_leaf_tokens=16,
        train_docs=128,
        val_docs=32,
        test_docs=64,
        model_family="neural",
        feature_mode="token_full",
        state_dim=32,
        hidden_dim=128,
        n_epochs=3,
        batch_size=16,
        lr=5e-4,
        weight_decay=0.0,
        audit_policy="fraction",
        audit_fraction=0.0,
        c3_audit_strategy="uniform",
        c3_include_root=True,
        leaf_query_rate=0.0,
        include_root_query=True,
        include_doc_sequence_baseline=True,
        include_doc_transformer_baseline=True,
        doc_sequence_objective="count_ce_only",
        doc_transformer_head_family="pooled_count_classifier",
        doc_transformer_layers=3,
        violation_tau=0.0,
        seed=41,
        use_cuda=False,
        torch_threads=1,
        artifact_dir=str(tmp_path / "artifacts"),
    )
    out = run_markov_changepoint_ops_count_experiment(cfg)
    seq_training = dict(out.metrics.get("doc_sequence_training", {}) or {})
    tr_training = dict(out.metrics.get("doc_transformer_training", {}) or {})

    assert seq_training["optimizer_backend"] == "official_neuraloperator_fno_count_classifier"
    assert seq_training["doc_sequence_objective_requested"] == "count_ce_only"
    assert seq_training["doc_sequence_objective_effective"] == "count_ce_only"
    assert seq_training["root_label_only_supervision"] is True
    assert tr_training["optimizer_backend"] == "full_sequence_boundary_transformer_count_classifier"
    assert tr_training["doc_transformer_head_family"] == "pooled_count_classifier"
    assert tr_training["doc_transformer_layers"] == 3
    assert tr_training["root_label_only_supervision"] is True


def test_neural_model_family_routes_to_fno():
    """model_family='neural' routes to FNOCountSketch."""
    cfg = OPSCountConfig(
        n_regimes=2,
        vocab_size=8,
        min_tokens=32,
        max_tokens=32,
        min_segments=2,
        max_segments=4,
        fixed_leaf_tokens=8,
        train_docs=4,
        val_docs=2,
        test_docs=4,
        model_family="neural",
        state_dim=8,
        hidden_dim=16,
        n_epochs=1,
        batch_size=4,
        lr=1e-3,
        fno_width=16,
        fno_n_modes=4,
        fno_n_layers=1,
        use_cuda=False,
        seed=42,
    )
    out = run_markov_changepoint_ops_count_experiment(cfg)
    assert out.metrics["learned"]["root_mae"] >= 0.0
    assert out.metrics["learned"]["n_docs"] == 4
    # Verify FNO-specific artifact metadata
    g_arts = out.g_artifacts
    if g_arts:
        for art in g_arts.values():
            if hasattr(art, "manifest") and art.manifest:
                manifest = art.manifest
                if "state_layout" in manifest:
                    assert "fno" in str(manifest["state_layout"]).lower()


def test_fno_tree_training_with_all_local_laws():
    """FNO tree model trains with C1+C2+C3 local laws and reports all metrics."""
    import math

    cfg = OPSCountConfig(
        n_regimes=2,
        vocab_size=8,
        min_tokens=32,
        max_tokens=32,
        min_segments=2,
        max_segments=4,
        fixed_leaf_tokens=8,
        train_docs=6,
        val_docs=2,
        test_docs=4,
        model_family="neural",
        state_dim=8,
        hidden_dim=16,
        n_epochs=3,
        batch_size=4,
        lr=1e-3,
        fno_width=16,
        fno_n_modes=4,
        fno_n_layers=1,
        law_package="all_laws",
        local_law_weight=0.5,
        use_cuda=False,
        seed=42,
        torch_threads=1,
    )
    out = run_markov_changepoint_ops_count_experiment(cfg)
    learned = out.metrics["learned"]

    # Root MAE is finite and reported.
    assert math.isfinite(float(learned["root_mae"]))
    assert int(learned["n_docs"]) == 4

    # All local law metrics are present and finite.
    for key in (
        "leaf_mae", "merge_mae", "c2_idempotence_mae",
        "test_root_mae", "test_leaf_mae", "test_merge_mae",
        "test_c2_idempotence_mae",
        "train_root_mae", "train_leaf_mae", "train_merge_mae",
        "train_c2_idempotence_mae",
    ):
        assert key in learned, f"missing key: {key}"
        assert math.isfinite(float(learned[key])), f"non-finite {key}={learned[key]}"

    # Objective weights reflect all_laws (equal split of C1/C2/C3).
    obj = out.objective
    assert float(obj["local_law_c1_weight"]) > 0.0
    assert float(obj["local_law_c2_weight"]) > 0.0
    assert float(obj["local_law_c3_weight"]) > 0.0
    # Equal split: each should be local_law_weight / 3.
    expected_each = 0.5 / 3.0
    assert abs(float(obj["local_law_c1_weight"]) - expected_each) < 1e-6
    assert abs(float(obj["local_law_c2_weight"]) - expected_each) < 1e-6
    assert abs(float(obj["local_law_c3_weight"]) - expected_each) < 1e-6

    # Training completed all epochs.
    assert int(learned["epochs_completed"]) == 3
    assert len(list(learned["train_loss_curve"])) == 3

    # Learnability payload is present.
    ll = out.local_law_learnability
    assert ll is not None
    assert len(ll) > 0
