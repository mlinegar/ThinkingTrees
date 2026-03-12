import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.tree.markov_changepoint_honesty_simulation import ChangepointMarkovDoc  # noqa: E402
from src.tree.markov_changepoint_ops_count_simulation import (  # noqa: E402
    AdditiveCountSketch,
    LearnedCountSketch,
    OPSCountConfig,
    _build_objective_summary,
    _sample_internal_audit_indices,
    _audit_estimator_diagnostics,
    _eval_count_only_family,
    _eval_exact_family,
    _eval_flip_family,
    _eval_leaf_bucket_family,
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
    n_regimes = 3
    first = torch.nn.functional.one_hot(torch.tensor(1), num_classes=n_regimes).to(torch.float32)
    last = torch.nn.functional.one_hot(torch.tensor(2), num_classes=n_regimes).to(torch.float32)
    core = torch.tensor([0.2, -0.1, 0.5, 0.3], dtype=torch.float32)
    features = torch.cat([first, last, core], dim=0)

    learned = LearnedCountSketch(
        feature_dim=int(features.numel()),
        state_dim=5,
        hidden_dim=8,
        target_scale=4.0,
        n_regimes=n_regimes,
        use_endpoints=True,
    )
    additive = AdditiveCountSketch(
        feature_dim=int(features.numel()),
        hidden_dim=8,
        target_scale=4.0,
        n_regimes=n_regimes,
        use_endpoints=True,
    )

    for model in (learned, additive):
        state = model.encode_leaf(features)
        replay = model.encode_summary(model.decode_summary(state))
        assert replay.shape == state.shape
        assert torch.allclose(replay, state, atol=1e-12, rtol=0.0)
        assert torch.allclose(
            model.predict_count_from_state(replay),
            model.predict_count_from_state(state),
            atol=1e-12,
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
