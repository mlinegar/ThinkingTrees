import math

import pytest

from src.core.logged_supervision import SamplingMetadata
from src.tree.ipw import (
    KFoldSplit,
    NodeType,
    TreeSample,
    effective_sample_size,
    empirical_bernstein_radius,
    eval_samples_fold,
    ipw_preference_loss,
    ipw_violation_rate,
    kfold_ipw_preference_empirical_bernstein_ci,
    kfold_ipw_violation_empirical_bernstein_ci,
    weighted_variance,
)
from src.tree.ipw_simulation import (
    ChunkScenario,
    SamplingDesign,
    compute_chunk_targets,
    draw_logged_tree_samples,
    evaluate_empirical_bernstein_coverage,
    generate_chunk_population,
)
from src.tree.ipw_toy_problems import (
    ChunkGranularity,
    ChunkPattern,
    ExampleExpectation,
    ImbalanceProfile,
    LengthProfile,
    OraclePreferenceProfile,
    compute_doc_policy_outcome_from_mergeable_sketch,
    generate_toy_chunk_population,
    mergeable_sketch_example_specs,
    run_mergeable_sketch_examples,
    toy_population_diagnostics,
)


def _sample(
    *,
    doc_id: str,
    node_id: str,
    violation: int,
    preference_loss: float,
    node_propensity: float,
) -> TreeSample:
    return TreeSample(
        doc_id=doc_id,
        node_id=node_id,
        node_type=NodeType.LEAF,
        violation=violation,
        preference_loss=preference_loss,
        sampling=SamplingMetadata(
            document_propensity=1.0,
            unit_propensity=node_propensity,
            label_propensity=1.0,
        ),
    )


def test_empirical_bernstein_radius_matches_lean_formula():
    samples = [
        _sample(doc_id="d1", node_id="n1", violation=1, preference_loss=0.9, node_propensity=0.5),
        _sample(doc_id="d1", node_id="n2", violation=0, preference_loss=0.3, node_propensity=1.0),
        _sample(doc_id="d2", node_id="n3", violation=1, preference_loss=0.7, node_propensity=0.25),
    ]
    delta = 0.1

    var = weighted_variance(samples, lambda s: float(s.violation))
    n_eff = effective_sample_size(samples)
    expected = math.sqrt((2.0 * var * math.log(2.0 / delta)) / n_eff) + (
        (7.0 / 3.0) * math.log(2.0 / delta) / (n_eff - 1.0)
    )

    actual = empirical_bernstein_radius(
        samples,
        lambda s: float(s.violation),
        delta,
        value_min=0.0,
        value_max=1.0,
    )
    assert actual == pytest.approx(expected)


def test_empirical_bernstein_radius_is_zero_when_neff_at_most_one():
    samples = [
        _sample(doc_id="d1", node_id="n1", violation=1, preference_loss=0.2, node_propensity=0.5),
    ]
    assert effective_sample_size(samples) == pytest.approx(1.0)
    assert (
        empirical_bernstein_radius(
            samples,
            lambda s: float(s.violation),
            0.05,
            value_min=0.0,
            value_max=1.0,
        )
        == 0.0
    )


def test_kfold_violation_ci_uses_fold_eb_averaging():
    samples = [
        _sample(doc_id="doc-a", node_id="a1", violation=1, preference_loss=0.8, node_propensity=0.5),
        _sample(doc_id="doc-a", node_id="a2", violation=0, preference_loss=0.4, node_propensity=1.0),
        _sample(doc_id="doc-b", node_id="b1", violation=1, preference_loss=0.7, node_propensity=0.25),
        _sample(doc_id="doc-b", node_id="b2", violation=1, preference_loss=0.6, node_propensity=1.0),
    ]
    split = KFoldSplit.from_doc_ids(["doc-a", "doc-b"], k=2, shuffle=False)
    delta = 0.2

    actual = kfold_ipw_violation_empirical_bernstein_ci(split, samples, delta=delta)

    folds = [eval_samples_fold(split, idx, samples) for idx in range(split.K)]
    folds = [fold for fold in folds if fold]
    per_fold_delta = delta / len(folds)
    mean_est = sum(ipw_violation_rate(fold) for fold in folds) / len(folds)
    mean_radius = sum(
        empirical_bernstein_radius(
            fold,
            lambda s: float(s.violation),
            per_fold_delta,
            value_min=0.0,
            value_max=1.0,
        )
        for fold in folds
    ) / len(folds)
    expected = (max(0.0, mean_est - mean_radius), min(1.0, mean_est + mean_radius))

    assert actual[0] == pytest.approx(expected[0])
    assert actual[1] == pytest.approx(expected[1])


def test_kfold_preference_ci_uses_fold_eb_averaging():
    samples = [
        _sample(doc_id="doc-a", node_id="a1", violation=1, preference_loss=0.9, node_propensity=0.5),
        _sample(doc_id="doc-a", node_id="a2", violation=0, preference_loss=0.2, node_propensity=1.0),
        _sample(doc_id="doc-b", node_id="b1", violation=1, preference_loss=0.6, node_propensity=0.25),
        _sample(doc_id="doc-b", node_id="b2", violation=1, preference_loss=0.5, node_propensity=1.0),
    ]
    split = KFoldSplit.from_doc_ids(["doc-a", "doc-b"], k=2, shuffle=False)
    delta = 0.2

    actual = kfold_ipw_preference_empirical_bernstein_ci(split, samples, delta=delta)

    folds = [eval_samples_fold(split, idx, samples) for idx in range(split.K)]
    folds = [fold for fold in folds if fold]
    per_fold_delta = delta / len(folds)
    mean_est = sum(ipw_preference_loss(fold) for fold in folds) / len(folds)
    mean_radius = sum(
        empirical_bernstein_radius(
            fold,
            lambda s: float(s.preference_loss),
            per_fold_delta,
            value_min=0.0,
            value_max=1.0,
        )
        for fold in folds
    ) / len(folds)
    expected = (max(0.0, mean_est - mean_radius), min(1.0, mean_est + mean_radius))

    assert actual[0] == pytest.approx(expected[0])
    assert actual[1] == pytest.approx(expected[1])


def test_compute_chunk_targets_separable_vs_nonseparable_context_dependence():
    local_signal = 0.35

    separable_a = compute_chunk_targets(
        local_signal,
        scenario=ChunkScenario.SEPARABLE,
        doc_mean_signal=-0.8,
        doc_signal_dispersion=0.1,
    )
    separable_b = compute_chunk_targets(
        local_signal,
        scenario=ChunkScenario.SEPARABLE,
        doc_mean_signal=0.8,
        doc_signal_dispersion=0.9,
    )
    assert separable_a == pytest.approx(separable_b)

    nonseparable_a = compute_chunk_targets(
        local_signal,
        scenario=ChunkScenario.NONSEPARABLE,
        doc_mean_signal=-0.8,
        doc_signal_dispersion=0.1,
    )
    nonseparable_b = compute_chunk_targets(
        local_signal,
        scenario=ChunkScenario.NONSEPARABLE,
        doc_mean_signal=0.8,
        doc_signal_dispersion=0.9,
    )
    assert abs(nonseparable_a[0] - nonseparable_b[0]) > 1e-3
    assert abs(nonseparable_a[1] - nonseparable_b[1]) > 1e-3


def test_empirical_bernstein_ci_coverage_separable_population():
    population = generate_chunk_population(
        n_docs=64,
        chunks_per_doc=8,
        scenario=ChunkScenario.SEPARABLE,
        seed=17,
    )
    result = evaluate_empirical_bernstein_coverage(
        population,
        n_trials=250,
        delta=0.10,
        seed=23,
    )

    assert result.violation_coverage >= 0.84
    assert result.preference_coverage >= 0.84
    assert result.mean_sample_count > 0.0
    assert result.mean_effective_sample_size > 0.0
    assert 0.0 <= result.violation_mean_width <= 1.0
    assert 0.0 <= result.preference_mean_width <= 1.0


def test_empirical_bernstein_ci_coverage_nonseparable_population():
    population = generate_chunk_population(
        n_docs=64,
        chunks_per_doc=8,
        scenario=ChunkScenario.NONSEPARABLE,
        seed=31,
    )
    result = evaluate_empirical_bernstein_coverage(
        population,
        n_trials=250,
        delta=0.10,
        seed=41,
    )

    assert result.violation_coverage >= 0.84
    assert result.preference_coverage >= 0.84
    assert result.mean_sample_count > 0.0
    assert result.mean_effective_sample_size > 0.0
    assert 0.0 <= result.violation_mean_width <= 1.0
    assert 0.0 <= result.preference_mean_width <= 1.0


def test_empirical_bernstein_ci_coverage_doc_nonseparable_population():
    population = generate_chunk_population(
        n_docs=64,
        chunks_per_doc=8,
        scenario=ChunkScenario.DOC_NONSEPARABLE,
        seed=73,
    )
    result = evaluate_empirical_bernstein_coverage(
        population,
        n_trials=250,
        delta=0.10,
        seed=83,
    )

    assert result.violation_coverage >= 0.84
    assert result.preference_coverage >= 0.84
    assert result.mean_sample_count > 0.0
    assert result.mean_effective_sample_size > 0.0
    assert 0.0 <= result.violation_mean_width <= 1.0
    assert 0.0 <= result.preference_mean_width <= 1.0


def test_draw_logged_tree_samples_wor_respects_fixed_sizes_and_propensities():
    population = generate_chunk_population(
        n_docs=12,
        chunks_per_doc=5,
        scenario=ChunkScenario.SEPARABLE,
        seed=101,
    )

    sampled = draw_logged_tree_samples(
        population,
        seed=7,
        sampling_design=SamplingDesign.WOR,
        wor_docs_sample=4,
        wor_chunks_per_doc_sample=2,
    )

    assert len(sampled) == 8
    sampled_doc_ids = {sample.doc_id for sample in sampled}
    assert len(sampled_doc_ids) == 4

    for doc_id in sampled_doc_ids:
        assert sum(1 for sample in sampled if sample.doc_id == doc_id) == 2

    expected_doc_prop = 4.0 / 12.0
    expected_node_prop = 2.0 / 5.0
    for sample in sampled:
        assert sample.sampling.document_propensity == pytest.approx(expected_doc_prop)
        assert sample.sampling.unit_propensity == pytest.approx(expected_node_prop)


def test_empirical_bernstein_compare_bernoulli_vs_wor_runs():
    population = generate_chunk_population(
        n_docs=48,
        chunks_per_doc=6,
        scenario=ChunkScenario.NONSEPARABLE,
        seed=131,
    )

    bern = evaluate_empirical_bernstein_coverage(
        population,
        n_trials=150,
        delta=0.10,
        seed=149,
        sampling_design=SamplingDesign.BERNOULLI,
    )
    wor = evaluate_empirical_bernstein_coverage(
        population,
        n_trials=150,
        delta=0.10,
        seed=151,
        sampling_design=SamplingDesign.WOR,
        wor_docs_sample=24,
        wor_chunks_per_doc_sample=3,
    )

    assert bern.sampling_design == SamplingDesign.BERNOULLI.value
    assert wor.sampling_design == SamplingDesign.WOR.value
    assert bern.mean_sample_count > 0.0
    assert wor.mean_sample_count > 0.0
    assert bern.violation_coverage >= 0.80
    assert wor.violation_coverage >= 0.80
    assert bern.preference_coverage >= 0.80
    assert wor.preference_coverage >= 0.80


def test_toy_population_front_loaded_pattern_has_early_signal():
    pop = generate_toy_chunk_population(
        n_docs=8,
        chunks_per_doc=12,
        scenario=ChunkScenario.NONSEPARABLE,
        granularity=ChunkGranularity.WORD,
        pattern=ChunkPattern.FRONT_LOADED,
        imbalance=ImbalanceProfile.MODERATE,
        seed=7,
    )
    first_doc = pop.chunks[:12]
    first_signal = first_doc[0].local_signal
    last_signal = first_doc[-1].local_signal
    assert first_signal > last_signal


def test_toy_population_adversarial_profile_pushes_high_signal_to_low_propensity():
    pop = generate_toy_chunk_population(
        n_docs=24,
        chunks_per_doc=16,
        scenario=ChunkScenario.NONSEPARABLE,
        granularity=ChunkGranularity.CHAR,
        pattern=ChunkPattern.SPIKE,
        imbalance=ImbalanceProfile.ADVERSARIAL,
        seed=17,
    )
    diag = toy_population_diagnostics(pop)
    assert diag.high_signal_low_propensity_overlap >= 0.50
    assert diag.min_joint_propensity < diag.median_joint_propensity
    assert diag.max_joint_weight > 5.0


def test_toy_population_word_and_char_coverages_run():
    for granularity in (ChunkGranularity.WORD, ChunkGranularity.CHAR):
        pop = generate_toy_chunk_population(
            n_docs=40,
            chunks_per_doc=14,
            scenario=ChunkScenario.DOC_NONSEPARABLE,
            granularity=granularity,
            pattern=ChunkPattern.BOUNDARY,
            imbalance=ImbalanceProfile.SEVERE,
            seed=41 if granularity == ChunkGranularity.WORD else 43,
        )
        result = evaluate_empirical_bernstein_coverage(
            pop,
            n_trials=100,
            delta=0.10,
            seed=53 if granularity == ChunkGranularity.WORD else 59,
            sampling_design=SamplingDesign.BERNOULLI,
        )
        assert result.violation_coverage >= 0.75
        assert result.preference_coverage >= 0.75


def test_toy_population_variable_lengths_for_adaptive_chunker_cases():
    pop = generate_toy_chunk_population(
        n_docs=36,
        chunks_per_doc=10,
        min_chunks_per_doc=3,
        max_chunks_per_doc=24,
        length_profile=LengthProfile.BIMODAL,
        scenario=ChunkScenario.NONSEPARABLE,
        granularity=ChunkGranularity.WORD,
        pattern=ChunkPattern.BOUNDARY,
        imbalance=ImbalanceProfile.SEVERE,
        seed=211,
    )
    diag = toy_population_diagnostics(pop)
    assert diag.min_doc_length >= 3
    assert diag.max_doc_length <= 24
    assert diag.max_doc_length - diag.min_doc_length >= 8
    assert diag.p90_doc_length > diag.p50_doc_length


def test_toy_population_long_tail_profile_has_spread_doc_lengths():
    pop = generate_toy_chunk_population(
        n_docs=48,
        chunks_per_doc=12,
        min_chunks_per_doc=4,
        max_chunks_per_doc=30,
        length_profile=LengthProfile.LONG_TAIL,
        scenario=ChunkScenario.DOC_NONSEPARABLE,
        granularity=ChunkGranularity.CHAR,
        pattern=ChunkPattern.SPIKE,
        imbalance=ImbalanceProfile.ADVERSARIAL,
        seed=223,
    )
    diag = toy_population_diagnostics(pop)
    assert diag.min_doc_length >= 4
    assert diag.max_doc_length <= 30
    assert diag.max_doc_length > diag.p50_doc_length
    assert diag.p90_doc_length >= diag.p50_doc_length


def test_mergeable_oracle_policy_order_invariant_for_nonadditive_profile():
    signals = [0.95, -0.20, 0.87, 0.11, -0.33, 0.62, 0.19]
    lhs = compute_doc_policy_outcome_from_mergeable_sketch(
        signals,
        oracle_preference=OraclePreferenceProfile.TOPK_SPIKE,
    )
    rhs = compute_doc_policy_outcome_from_mergeable_sketch(
        list(reversed(signals)),
        oracle_preference=OraclePreferenceProfile.TOPK_SPIKE,
    )
    assert lhs == pytest.approx(rhs)


def test_nonadditive_mergeable_oracle_distinguishes_spiky_vs_flat_same_mean():
    flat = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    spiky = [1.0, 0.2, 0.2, 0.2, 0.2, -0.6]  # same mean (0.2), very different concentration

    additive_flat = compute_doc_policy_outcome_from_mergeable_sketch(
        flat,
        oracle_preference=OraclePreferenceProfile.ADDITIVE_MEAN,
    )
    additive_spiky = compute_doc_policy_outcome_from_mergeable_sketch(
        spiky,
        oracle_preference=OraclePreferenceProfile.ADDITIVE_MEAN,
    )
    nonadd_flat = compute_doc_policy_outcome_from_mergeable_sketch(
        flat,
        oracle_preference=OraclePreferenceProfile.TOPK_SPIKE,
    )
    nonadd_spiky = compute_doc_policy_outcome_from_mergeable_sketch(
        spiky,
        oracle_preference=OraclePreferenceProfile.TOPK_SPIKE,
    )

    assert additive_flat == pytest.approx(additive_spiky)
    assert nonadd_spiky > nonadd_flat + 0.10


def test_toy_population_oracle_preference_profiles_change_true_preference():
    kwargs = dict(
        n_docs=28,
        chunks_per_doc=12,
        scenario=ChunkScenario.DOC_NONSEPARABLE,
        granularity=ChunkGranularity.WORD,
        pattern=ChunkPattern.SPIKE,
        imbalance=ImbalanceProfile.MODERATE,
        length_profile=LengthProfile.UNIFORM,
        seed=331,
    )
    pop_add = generate_toy_chunk_population(
        oracle_preference=OraclePreferenceProfile.ADDITIVE_MEAN,
        **kwargs,
    )
    pop_nonadd = generate_toy_chunk_population(
        oracle_preference=OraclePreferenceProfile.TOPK_SPIKE,
        **kwargs,
    )
    assert abs(pop_add.true_preference_loss - pop_nonadd.true_preference_loss) > 1e-3


def test_mergeable_sketch_examples_include_positive_and_negative_cases():
    specs = mergeable_sketch_example_specs()
    labels = {spec.expectation for spec in specs}
    assert ExampleExpectation.POSITIVE in labels
    assert ExampleExpectation.NEGATIVE in labels


def test_run_mergeable_sketch_examples_runs_with_labeled_cases():
    runs = run_mergeable_sketch_examples(
        designs=[SamplingDesign.BERNOULLI],
        n_docs=32,
        chunks_per_doc=12,
        min_chunks_per_doc=4,
        max_chunks_per_doc=24,
        n_trials=60,
        delta=0.10,
        population_seed=401,
        trial_seed=503,
    )
    assert len(runs) >= 4
    labels = {run.expectation for run in runs}
    assert ExampleExpectation.POSITIVE in labels
    assert ExampleExpectation.NEGATIVE in labels

    positive_weights = [run.diagnostics.max_joint_weight for run in runs if run.expectation == ExampleExpectation.POSITIVE]
    negative_weights = [run.diagnostics.max_joint_weight for run in runs if run.expectation == ExampleExpectation.NEGATIVE]
    assert positive_weights
    assert negative_weights
    assert max(negative_weights) >= max(positive_weights)

    for run in runs:
        assert 0.0 <= run.coverage["violation_coverage"] <= 1.0
        assert 0.0 <= run.coverage["preference_coverage"] <= 1.0
