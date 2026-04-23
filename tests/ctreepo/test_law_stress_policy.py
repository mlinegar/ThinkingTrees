from __future__ import annotations

from src.ctreepo.sim.suite.law_stress_policy import (
    resolve_lda_law_stress_policy,
    resolve_markov_law_stress_policy,
)


def test_markov_law_stress_policy_smoke_defaults_match_expected_grid() -> None:
    policy = resolve_markov_law_stress_policy(smoke=True)
    assert policy.smoke is True
    assert policy.sanity.train_docs == (32,)
    assert policy.sanity.n_regimes == (2,)
    assert policy.transition_map.audit_fractions == (0.1, 1.0)
    assert policy.mechanism.selection_limit == 1
    assert policy.weight_ablation.caps == ((64, 256),)


def test_lda_law_stress_policy_publication_defaults_match_expected_grid() -> None:
    policy = resolve_lda_law_stress_policy(smoke=False)
    assert policy.smoke is False
    assert policy.sanity.taus == (1.0, 4.0, 16.0)
    assert policy.transition_map.lambda_multipliers == (0.0, 0.1, 0.5, 1.0, 1.5, 3.0)
    assert policy.mechanism.analysis_partition_modes == (
        "aligned",
        "coarsen_2x",
        "shift_half",
        "random_same_count",
    )
