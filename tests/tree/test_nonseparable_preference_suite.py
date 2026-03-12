from src.tree.nonseparable_preference_suite import (
    NonseparableSuiteConfig,
    run_nonseparable_preference_suite,
)


def test_nonseparable_suite_passes_separation_gates_in_default_regime():
    result = run_nonseparable_preference_suite(
        NonseparableSuiteConfig(
            n_replicates=12,
            n_pairs_per_replicate=80,
            seed=0,
        )
    ).to_dict()
    for dgp in result["dgps"]:
        assert bool(dgp["strong_separation_pass"])
        for check in dgp["separation_checks"]:
            assert bool(check["passes_gate"])


def test_nonseparable_suite_bound_consistency_holds_cellwise():
    result = run_nonseparable_preference_suite(
        NonseparableSuiteConfig(
            n_replicates=10,
            n_pairs_per_replicate=60,
            seed=11,
        )
    ).to_dict()
    for dgp in result["dgps"]:
        for arm in dgp["arms"]:
            assert 0.0 <= float(arm["mean_utility_regret"]) <= 1.0
            assert 0.0 <= float(arm["mean_bound_envelope"]) <= 1.0
            assert bool(arm["bound_consistent"])
