import pytest

from src.tree.mergeable_ablation import (
    AblationSpec,
    AggregatorPolicy,
    Chunk,
    ChunkerPolicy,
    MergeOrder,
    SelectorPolicy,
    SpikeMixtureDistributionSpec,
    default_generalization_stress_scenarios,
    default_four_parameter_method_specs,
    default_k_sketch_method_specs,
    default_nonlanguage_chunk_quality_scenarios,
    default_three_parameter_method_specs,
    default_two_parameter_method_specs,
    KSketchEstimator,
    KSketchMethodSpec,
    SpikeCountMixtureDistributionSpec,
    ToyTokenDocument,
    TokenPattern,
    aggregate_chunks,
    evaluate_document,
    generate_toy_token_document,
    run_default_ablation_suite,
    run_four_parameter_recovery_study,
    run_chunk_quality_sweep,
    run_chunk_quality_coverage_sweep,
    run_k_target_recovery_study,
    run_three_parameter_generalization_sweep,
    run_spike_prevalence_recovery_study,
    run_three_parameter_recovery_study,
    run_two_parameter_recovery_study,
    sample_spike_mixture_documents,
    sketch_insufficiency_counterexample,
)


def test_merge_safe_max_is_order_invariant():
    chunks = [
        Chunk(start=0, end=2, values=(0.1, 0.2), proxy_values=(0.1, 0.2)),
        Chunk(start=2, end=4, values=(0.9, 0.1), proxy_values=(0.9, 0.1)),
        Chunk(start=4, end=6, values=(0.1, 0.2), proxy_values=(0.1, 0.2)),
    ]
    left = aggregate_chunks(
        chunks,
        aggregator=AggregatorPolicy.MERGE_SAFE_MAX,
        merge_order=MergeOrder.LEFT_TO_RIGHT,
    )
    right = aggregate_chunks(
        chunks,
        aggregator=AggregatorPolicy.MERGE_SAFE_MAX,
        merge_order=MergeOrder.RIGHT_TO_LEFT,
    )
    rnd = aggregate_chunks(
        chunks,
        aggregator=AggregatorPolicy.MERGE_SAFE_MAX,
        merge_order=MergeOrder.RANDOM,
        seed=7,
    )
    assert left == pytest.approx(right)
    assert left == pytest.approx(rnd)


def test_naive_mean_of_means_can_be_order_sensitive():
    chunks = [
        Chunk(start=0, end=1, values=(1.0,), proxy_values=(1.0,)),
        Chunk(start=1, end=2, values=(0.0,), proxy_values=(0.0,)),
        Chunk(start=2, end=3, values=(0.0,), proxy_values=(0.0,)),
    ]
    left = aggregate_chunks(
        chunks,
        aggregator=AggregatorPolicy.NAIVE_MEAN_OF_MEANS,
        merge_order=MergeOrder.LEFT_TO_RIGHT,
    )
    right = aggregate_chunks(
        chunks,
        aggregator=AggregatorPolicy.NAIVE_MEAN_OF_MEANS,
        merge_order=MergeOrder.RIGHT_TO_LEFT,
    )
    assert left == pytest.approx(0.25)
    assert right == pytest.approx(0.5)
    assert abs(left - right) > 1e-6


def test_right_rule_wrong_chunker_can_fail_due_to_selection():
    # Spike is real and proxy-visible; aligned chunking+selection keeps it,
    # while misspecified chunking+selection drops it under tight budget.
    doc = ToyTokenDocument(
        token_scores=(0.10, 0.10, 0.11, 0.98, 0.09, 0.10, 0.11, 0.10, 0.09, 0.10, 0.11, 0.10),
        proxy_scores=(0.10, 0.10, 0.12, 0.97, 0.09, 0.10, 0.11, 0.10, 0.09, 0.10, 0.11, 0.10),
    )
    aligned = AblationSpec(
        name="aligned",
        description="",
        chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
        selector=SelectorPolicy.TOP_PROXY,
        aggregator=AggregatorPolicy.MERGE_SAFE_MAX,
        chunk_budget=2,
    )
    wrong = AblationSpec(
        name="wrong",
        description="",
        chunker=ChunkerPolicy.ADAPTIVE_MISSPECIFIED,
        selector=SelectorPolicy.BOTTOM_PROXY,
        aggregator=AggregatorPolicy.MERGE_SAFE_MAX,
        chunk_budget=2,
    )
    good = evaluate_document(doc, spec=aligned, seed=11)
    bad = evaluate_document(doc, spec=wrong, seed=11)
    assert good.true_label == 1
    assert good.estimated_label == 1
    assert bad.estimated_label == 0


def test_default_ablation_suite_reports_expected_method_names():
    summaries = run_default_ablation_suite(n_docs=80, n_tokens=32, seed=13)
    names = {s.name for s in summaries}
    assert names == {
        "merge_safe_oracle_aligned",
        "naive_majority_same_chunker",
        "naive_mean_of_means_same_chunker",
        "right_rule_wrong_chunker",
    }


def test_spike_prevalence_recovery_baseline_has_lower_bias_than_bad_ablations():
    summaries = run_spike_prevalence_recovery_study(
        distribution=SpikeMixtureDistributionSpec(
            p_spike_doc=0.55,
            p_boundary_given_spike=0.5,
            p_two_spikes_given_spike=0.2,
            n_tokens=32,
            proxy_noise=0.08,
        ),
        n_replicates=80,
        docs_per_replicate=120,
        seed=19,
    )
    by_name = {s.method_name: s for s in summaries}
    baseline = by_name["merge_safe_oracle_aligned"]
    wrong_chunker = by_name["right_rule_wrong_chunker"]
    naive_majority = by_name["naive_majority_same_chunker"]

    assert baseline.mean_abs_bias < 0.05
    assert wrong_chunker.mean_abs_bias > baseline.mean_abs_bias + 0.20
    assert naive_majority.mean_abs_bias > baseline.mean_abs_bias + 0.20


def test_two_parameter_recovery_full_model_and_one_pass_dominate_ablations():
    summaries = run_two_parameter_recovery_study(
        distribution=SpikeMixtureDistributionSpec(
            p_spike_doc=0.55,
            p_boundary_given_spike=0.5,
            p_two_spikes_given_spike=0.2,
            n_tokens=32,
            proxy_noise=0.08,
        ),
        methods=default_two_parameter_method_specs(),
        n_replicates=80,
        docs_per_replicate=120,
        seed=29,
    )
    by_name = {s.method_name: s for s in summaries}
    one_pass = by_name["one_pass_oracle"]
    full = by_name["full_model_aligned"]
    naive = by_name["naive_majority_same_chunker"]
    wrong = by_name["right_rule_wrong_chunker"]

    assert one_pass.mean_abs_bias_p_spike < 0.05
    assert full.mean_abs_bias_p_spike < 0.07
    assert one_pass.mean_abs_bias_p_two_given_spike < 0.10
    assert full.mean_abs_bias_p_two_given_spike < 0.12

    assert naive.mean_abs_bias_p_spike > full.mean_abs_bias_p_spike + 0.15
    assert wrong.mean_abs_bias_p_spike > full.mean_abs_bias_p_spike + 0.15
    assert naive.mean_abs_bias_p_two_given_spike > full.mean_abs_bias_p_two_given_spike + 0.15
    assert wrong.mean_abs_bias_p_two_given_spike > full.mean_abs_bias_p_two_given_spike + 0.15


def test_weighting_views_match_under_equal_length_documents():
    method = AblationSpec(
        name="one_pass_fixed",
        description="",
        chunker=ChunkerPolicy.FIXED,
        selector=SelectorPolicy.ALL,
        aggregator=AggregatorPolicy.MERGE_SAFE_MAX,
        chunk_budget=None,
        fixed_chunk_size=10**9,
    )
    summaries = run_spike_prevalence_recovery_study(
        distribution=SpikeMixtureDistributionSpec(
            p_spike_doc=0.60,
            p_boundary_given_spike=0.40,
            p_two_spikes_given_spike=0.20,
            n_tokens=32,
            proxy_noise=0.08,
        ),
        methods=[method],
        n_replicates=12,
        docs_per_replicate=60,
        seed=3,
    )
    s = summaries[0]
    assert s.legacy_weighting_mode == "doc"
    assert s.weighting_views is not None
    doc_hat = float(s.weighting_views["doc"]["mean_hat"])
    leaf_hat = float(s.weighting_views["leaf"]["mean_hat"])
    token_hat = float(s.weighting_views["token"]["mean_hat"])
    assert doc_hat == pytest.approx(leaf_hat, abs=1e-12)
    assert doc_hat == pytest.approx(token_hat, abs=1e-12)


def test_weighting_views_diverge_under_variable_length_with_wrong_chunker():
    method = AblationSpec(
        name="right_rule_wrong_chunker",
        description="",
        chunker=ChunkerPolicy.ADAPTIVE_MISSPECIFIED,
        selector=SelectorPolicy.BOTTOM_PROXY,
        aggregator=AggregatorPolicy.MERGE_SAFE_MAX,
        chunk_budget=6,
    )
    summaries = run_spike_prevalence_recovery_study(
        distribution=SpikeMixtureDistributionSpec(
            p_spike_doc=0.70,
            p_boundary_given_spike=0.50,
            p_two_spikes_given_spike=0.25,
            n_tokens=32,
            proxy_noise=0.12,
            token_length_support=(8, 16, 32, 64, 128),
            token_length_probs=(0.20, 0.20, 0.20, 0.20, 0.20),
        ),
        methods=[method],
        n_replicates=12,
        docs_per_replicate=80,
        seed=7,
    )
    s = summaries[0]
    assert s.weighting_views is not None
    doc_hat = float(s.weighting_views["doc"]["mean_hat"])
    leaf_hat = float(s.weighting_views["leaf"]["mean_hat"])
    token_hat = float(s.weighting_views["token"]["mean_hat"])
    assert abs(token_hat - doc_hat) > 0.05
    assert abs(leaf_hat - doc_hat) > 0.05


def test_token_weighted_boundary_bias_still_penalizes_missing_boundary_stat():
    summaries = run_three_parameter_recovery_study(
        distribution=SpikeMixtureDistributionSpec(
            p_spike_doc=0.55,
            p_boundary_given_spike=0.50,
            p_two_spikes_given_spike=0.20,
            n_tokens=32,
            proxy_noise=0.08,
            boundary_span_tokens=4,
        ),
        methods=default_three_parameter_method_specs(),
        n_replicates=20,
        docs_per_replicate=80,
        seed=5,
    )
    by_name = {s.method_name: s for s in summaries}
    full = by_name["full_model_aligned"]
    missing = by_name["full_model_missing_boundary_stat"]
    assert full.weighting_views is not None
    assert missing.weighting_views is not None
    full_tok = float(
        full.weighting_views["token"]["p_boundary_given_spike"]["mean_abs_bias"]
    )
    missing_tok = float(
        missing.weighting_views["token"]["p_boundary_given_spike"]["mean_abs_bias"]
    )
    assert missing_tok > full_tok + 0.10


def test_three_parameter_recovery_requires_boundary_sufficient_statistic():
    summaries = run_three_parameter_recovery_study(
        distribution=SpikeMixtureDistributionSpec(
            p_spike_doc=0.55,
            p_boundary_given_spike=0.5,
            p_two_spikes_given_spike=0.2,
            n_tokens=32,
            proxy_noise=0.08,
            boundary_span_tokens=4,
        ),
        methods=default_three_parameter_method_specs(),
        n_replicates=80,
        docs_per_replicate=120,
        seed=37,
    )
    by_name = {s.method_name: s for s in summaries}
    one_pass = by_name["one_pass_oracle"]
    full = by_name["full_model_aligned"]
    missing_boundary = by_name["full_model_missing_boundary_stat"]
    naive = by_name["naive_majority_same_chunker"]
    wrong = by_name["right_rule_wrong_chunker"]

    assert one_pass.supports_boundary_spike
    assert full.supports_boundary_spike
    assert not missing_boundary.supports_boundary_spike

    assert one_pass.mean_abs_bias_p_boundary_given_spike < 0.08
    assert full.mean_abs_bias_p_boundary_given_spike < 0.12

    assert missing_boundary.mean_abs_bias_p_boundary_given_spike > full.mean_abs_bias_p_boundary_given_spike + 0.12
    assert naive.mean_abs_bias_p_boundary_given_spike > full.mean_abs_bias_p_boundary_given_spike + 0.12
    assert wrong.mean_abs_bias_p_boundary_given_spike > full.mean_abs_bias_p_boundary_given_spike + 0.12


def test_variable_length_spike_mixture_sampler_respects_length_support():
    docs = sample_spike_mixture_documents(
        spec=SpikeMixtureDistributionSpec(
            p_spike_doc=0.5,
            p_boundary_given_spike=0.3,
            p_two_spikes_given_spike=0.2,
            token_length_support=(3, 7, 11),
            token_length_probs=(0.2, 0.5, 0.3),
            boundary_span_tokens=1,
        ),
        n_docs=300,
        seed=17,
    )
    lengths = {doc.n_tokens for doc in docs}
    assert lengths.issubset({3, 7, 11})
    assert len(lengths) >= 2


def test_generate_multi_spikes_pattern_has_at_least_three_spikes():
    doc = generate_toy_token_document(
        pattern=TokenPattern.MULTI_SPIKES,
        n_tokens=24,
        seed=5,
    )
    n_spikes = sum(1 for v in doc.token_scores if v >= 0.90)
    assert n_spikes >= 3


def test_spike_mixture_sampler_can_force_multi_spike_documents():
    docs = sample_spike_mixture_documents(
        spec=SpikeMixtureDistributionSpec(
            p_spike_doc=1.0,
            p_boundary_given_spike=0.0,
            p_two_spikes_given_spike=1.0,
            p_multi_given_two_spikes=1.0,
            n_tokens=24,
            proxy_noise=0.08,
        ),
        n_docs=80,
        seed=9,
    )
    min_spikes = min(sum(1 for v in doc.token_scores if v >= 0.90) for doc in docs)
    assert min_spikes >= 3


def test_generalization_stress_sweep_full_model_beats_naive_and_wrong_chunker():
    summaries = run_three_parameter_generalization_sweep(
        scenarios=default_generalization_stress_scenarios(),
        n_replicates=40,
        docs_per_replicate=80,
        seed=23,
    )
    by_scenario_method = {(s.scenario_name, s.method_name): s for s in summaries}
    scenarios = {s.scenario_name for s in summaries}

    for scenario_name in scenarios:
        full = by_scenario_method[(scenario_name, "full_model_aligned")]
        naive = by_scenario_method[(scenario_name, "naive_majority_same_chunker")]
        wrong = by_scenario_method[(scenario_name, "right_rule_wrong_chunker")]

        assert full.aggregate_mean_abs_bias < naive.aggregate_mean_abs_bias
        assert full.aggregate_mean_abs_bias < wrong.aggregate_mean_abs_bias

    for s in summaries:
        if s.scenario_name == "baseline_balanced_fixed":
            assert s.generalization_gap_vs_baseline == pytest.approx(0.0)


def test_generalization_sweep_boundary_span_retuning_reduces_boundary_bias():
    scenarios = default_generalization_stress_scenarios()
    frozen = run_three_parameter_generalization_sweep(
        scenarios=scenarios,
        n_replicates=30,
        docs_per_replicate=80,
        seed=31,
        align_boundary_span_to_distribution=False,
    )
    retuned = run_three_parameter_generalization_sweep(
        scenarios=scenarios,
        n_replicates=30,
        docs_per_replicate=80,
        seed=31,
        align_boundary_span_to_distribution=True,
    )

    frozen_map = {(s.scenario_name, s.method_name): s for s in frozen}
    retuned_map = {(s.scenario_name, s.method_name): s for s in retuned}

    key = ("variable_length_balanced", "full_model_aligned")
    assert retuned_map[key].mean_abs_bias_p_boundary_given_spike < (
        frozen_map[key].mean_abs_bias_p_boundary_given_spike - 0.10
    )


def test_hard_noncorner_adversarial_still_separates_full_model_from_ablations():
    scenarios = [
        s for s in default_generalization_stress_scenarios()
        if s.name == "hard_noncorner_adversarial"
    ]
    assert len(scenarios) == 1
    scenario = scenarios[0]

    # Explicitly guard against corner-probability artifacts.
    assert 0.15 < scenario.distribution.p_two_spikes_given_spike < 0.85
    assert 0.15 < scenario.distribution.p_boundary_given_spike < 0.85
    assert 0.15 < scenario.distribution.p_spike_doc < 0.85

    summaries = run_three_parameter_generalization_sweep(
        scenarios=scenarios,
        n_replicates=35,
        docs_per_replicate=90,
        seed=41,
    )
    by_method = {s.method_name: s for s in summaries}
    full = by_method["full_model_aligned"]
    naive = by_method["naive_majority_same_chunker"]
    wrong = by_method["right_rule_wrong_chunker"]
    missing_boundary = by_method["full_model_missing_boundary_stat"]

    assert full.aggregate_mean_abs_bias < 0.25
    assert naive.aggregate_mean_abs_bias > full.aggregate_mean_abs_bias + 0.20
    assert wrong.aggregate_mean_abs_bias > full.aggregate_mean_abs_bias + 0.10
    assert missing_boundary.mean_abs_bias_p_boundary_given_spike > 0.25


def test_multi_spike_noncorner_scenario_is_recoverable_for_full_model():
    scenarios = [
        s for s in default_generalization_stress_scenarios()
        if s.name == "multi_spike_noncorner"
    ]
    assert len(scenarios) == 1
    summaries = run_three_parameter_generalization_sweep(
        scenarios=scenarios,
        n_replicates=40,
        docs_per_replicate=90,
        seed=43,
    )
    by_method = {s.method_name: s for s in summaries}
    full = by_method["full_model_aligned"]
    one_pass = by_method["one_pass_oracle"]
    naive = by_method["naive_majority_same_chunker"]

    assert full.mean_abs_bias_p_two_given_spike < 0.15
    assert one_pass.mean_abs_bias_p_two_given_spike < 0.15
    assert naive.mean_abs_bias_p_two_given_spike > full.mean_abs_bias_p_two_given_spike + 0.25


def test_four_parameter_recovery_needs_third_order_statistic():
    summaries = run_four_parameter_recovery_study(
        distribution=SpikeMixtureDistributionSpec(
            p_spike_doc=0.62,
            p_boundary_given_spike=0.35,
            p_two_spikes_given_spike=0.45,
            p_multi_given_two_spikes=0.35,
            n_tokens=32,
            proxy_noise=0.12,
            boundary_span_tokens=4,
        ),
        methods=default_four_parameter_method_specs(),
        n_replicates=60,
        docs_per_replicate=100,
        seed=47,
    )
    by_name = {s.method_name: s for s in summaries}
    full = by_name["full_model_aligned"]
    one_pass = by_name["one_pass_oracle"]
    missing_three = by_name["full_model_missing_three_stat"]
    naive = by_name["naive_majority_same_chunker"]

    assert full.supports_three_spike
    assert one_pass.supports_three_spike
    assert not missing_three.supports_three_spike

    assert full.mean_abs_bias_p_three_given_spike < 0.12
    assert one_pass.mean_abs_bias_p_three_given_spike < 0.12
    assert missing_three.mean_abs_bias_p_three_given_spike > full.mean_abs_bias_p_three_given_spike + 0.18
    assert naive.mean_abs_bias_p_three_given_spike > full.mean_abs_bias_p_three_given_spike + 0.20


def test_sketch_insufficiency_counterexample_same_signature_different_truth():
    m = 3
    k = 4
    a, b, sig = sketch_insufficiency_counterexample(sketch_order=m, target_k=k, n_tokens=16)
    assert tuple(sorted(a, reverse=True)[:m]) == pytest.approx(sig)
    assert tuple(sorted(b, reverse=True)[:m]) == pytest.approx(sig)
    count_a = sum(1 for v in a if v >= 0.90)
    count_b = sum(1 for v in b if v >= 0.90)
    assert count_a == m
    assert count_b >= k


def test_generic_k_recovery_limited_sketch_fails_for_large_k():
    summaries = run_k_target_recovery_study(
        distribution=SpikeCountMixtureDistributionSpec(
            p_spike_doc=0.62,
            p_boundary_given_spike=0.3,
            spike_count_support=(1, 2, 3, 4, 5),
            spike_count_probs_given_spike=(0.10, 0.20, 0.25, 0.25, 0.20),
            n_tokens=40,
            proxy_noise=0.12,
            boundary_span_tokens=4,
        ),
        target_ks=(2, 3, 4, 5),
        methods=default_k_sketch_method_specs(target_max_k=5),
        n_replicates=40,
        docs_per_replicate=100,
        seed=53,
    )
    by_method_k = {(s.method_name, s.target_k): s for s in summaries}
    full_k5 = by_method_k[("full_model_aligned", 5)]
    limited_k5 = by_method_k[("full_model_limited_sketch", 5)]
    naive_k5 = by_method_k[("naive_majority_same_chunker", 5)]

    assert full_k5.supports_target
    assert not limited_k5.supports_target

    assert full_k5.mean_abs_bias < 0.12
    assert limited_k5.mean_abs_bias > full_k5.mean_abs_bias + 0.18
    assert naive_k5.mean_abs_bias > full_k5.mean_abs_bias + 0.20


def test_generic_k_oversupported_behaves_like_exact_supported_for_one_pass():
    methods = [
        KSketchMethodSpec(
            name="one_pass_m3",
            description="",
            estimator=KSketchEstimator.MERGE_SAFE_TOPK,
            chunker=ChunkerPolicy.FIXED,
            selector=SelectorPolicy.ALL,
            sketch_order=3,
            chunk_budget=None,
            fixed_chunk_size=10**9,
        ),
        KSketchMethodSpec(
            name="one_pass_m5",
            description="",
            estimator=KSketchEstimator.MERGE_SAFE_TOPK,
            chunker=ChunkerPolicy.FIXED,
            selector=SelectorPolicy.ALL,
            sketch_order=5,
            chunk_budget=None,
            fixed_chunk_size=10**9,
        ),
        KSketchMethodSpec(
            name="one_pass_m7",
            description="",
            estimator=KSketchEstimator.MERGE_SAFE_TOPK,
            chunker=ChunkerPolicy.FIXED,
            selector=SelectorPolicy.ALL,
            sketch_order=7,
            chunk_budget=None,
            fixed_chunk_size=10**9,
        ),
    ]
    summaries = run_k_target_recovery_study(
        distribution=SpikeCountMixtureDistributionSpec(
            p_spike_doc=0.62,
            p_boundary_given_spike=0.35,
            spike_count_support=(1, 2, 3, 4, 5),
            spike_count_probs_given_spike=(0.10, 0.20, 0.25, 0.25, 0.20),
            n_tokens=40,
            proxy_noise=0.10,
            boundary_span_tokens=4,
        ),
        target_ks=(5,),
        methods=methods,
        n_replicates=50,
        docs_per_replicate=120,
        seed=59,
    )
    by_name = {s.method_name: s for s in summaries}
    m3 = by_name["one_pass_m3"]
    m5 = by_name["one_pass_m5"]
    m7 = by_name["one_pass_m7"]

    assert not m3.supports_target
    assert m5.supports_target
    assert m7.supports_target

    assert m3.mean_abs_bias > m5.mean_abs_bias + 0.20
    # over-supported should be close to exact-supported (sampling noise only).
    assert abs(m7.mean_abs_bias - m5.mean_abs_bias) < 0.03


def test_chunk_quality_sweep_perfect_token_leaves_and_budget_effect():
    summaries = run_chunk_quality_sweep(
        distribution=SpikeCountMixtureDistributionSpec(
            p_spike_doc=1.0,
            p_boundary_given_spike=0.0,
            spike_count_support=(3,),
            spike_count_probs_given_spike=(1.0,),
            n_tokens=24,
            proxy_noise=0.0,
            boundary_span_tokens=4,
        ),
        target_k=3,
        sketch_order=3,
        chunk_sizes=(1,),
        chunk_budgets=(1, 4),
        chunker=ChunkerPolicy.FIXED,
        selector=SelectorPolicy.TOP_PROXY,
        n_replicates=20,
        docs_per_replicate=80,
        seed=61,
        include_references=True,
    )
    by_name = {s.method_name: s for s in summaries}
    perfect = by_name["perfect_token_leaves_all"]
    low_budget = by_name["grid_fixed_s1_b1"]
    high_budget = by_name["grid_fixed_s1_b4"]

    assert perfect.mean_spike_token_recall > 0.99
    assert perfect.mean_spike_token_isolation > 0.99
    assert perfect.mean_target_capture_rate > 0.99
    assert perfect.mean_abs_bias < 1e-6

    # With only one kept token-leaf and k=3 target, capture necessarily fails.
    assert low_budget.mean_target_capture_rate < 0.01
    assert low_budget.mean_abs_bias > 0.95

    # Increasing budget restores capture/recovery in this deterministic setting.
    assert high_budget.mean_target_capture_rate > 0.99
    assert high_budget.mean_abs_bias < 1e-6


def test_nonlanguage_scenarios_defined_and_recoverable_with_aligned_chunking():
    scenarios = default_nonlanguage_chunk_quality_scenarios()
    assert len(scenarios) >= 4

    scenario = scenarios[0]
    summaries = run_chunk_quality_sweep(
        distribution=scenario.distribution,
        target_k=5,
        sketch_order=5,
        chunk_sizes=(1, 2, 4, 8),
        chunk_budgets=(2, 8),
        chunker=ChunkerPolicy.ADAPTIVE_ALIGNED,
        selector=SelectorPolicy.TOP_PROXY,
        n_replicates=20,
        docs_per_replicate=80,
        seed=73,
        include_references=True,
    )
    by_name = {s.method_name: s for s in summaries}
    low = by_name["grid_aligned_s8_b2"]
    high = by_name["grid_aligned_s8_b8"]

    # Increasing budget should not hurt and usually improves aligned chunking.
    assert high.mean_abs_bias <= low.mean_abs_bias + 0.05
    assert high.mean_target_capture_rate >= low.mean_target_capture_rate - 0.02

    # Reference methods remain sensible in non-language scenarios.
    assert by_name["one_pass_reference"].mean_abs_bias < 0.12
    assert by_name["perfect_token_leaves_all"].mean_spike_token_recall > 0.99


def test_chunk_quality_coverage_sweep_detects_undercoverage_when_budget_is_too_small():
    summaries = run_chunk_quality_coverage_sweep(
        distribution=SpikeCountMixtureDistributionSpec(
            p_spike_doc=1.0,
            p_boundary_given_spike=0.0,
            spike_count_support=(3,),
            spike_count_probs_given_spike=(1.0,),
            n_tokens=24,
            proxy_noise=0.0,
            boundary_span_tokens=4,
        ),
        target_k=3,
        sketch_order=3,
        chunk_sizes=(1,),
        chunk_budgets=(1, 4),
        chunker=ChunkerPolicy.FIXED,
        selector=SelectorPolicy.TOP_PROXY,
        ci_level=0.95,
        n_replicates=20,
        docs_per_replicate=80,
        seed=79,
        include_references=True,
    )
    by_name = {s.method_name: s for s in summaries}
    low_budget = by_name["grid_fixed_s1_b1"]
    high_budget = by_name["grid_fixed_s1_b4"]
    perfect = by_name["perfect_token_leaves_all"]

    assert perfect.empirical_coverage > 0.95
    assert high_budget.empirical_coverage > 0.95
    assert low_budget.mean_abs_bias > 0.95
    assert low_budget.empirical_coverage < 0.10
