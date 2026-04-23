from __future__ import annotations

from src.ctreepo.sim.suite.identifiable_zero_publication_policy import (
    resolve_identifiable_zero_longrun_policy,
    resolve_identifiable_zero_publication_clean_policy,
)


def test_identifiable_zero_publication_clean_policy_defaults_match_expected_grid() -> None:
    policy = resolve_identifiable_zero_publication_clean_policy()
    assert policy.profile == "publication_clean"
    assert policy.segment.train_docs == (12000,)
    assert policy.ctree.calibration_rates == (0.01, 0.02, 0.05, 0.1)
    assert policy.markov.task_objective_weights == (1.0,)


def test_identifiable_zero_longrun_policy_defaults_match_expected_grid() -> None:
    policy = resolve_identifiable_zero_longrun_policy()
    assert policy.profile == "longrun_equiv_v1"
    assert policy.segment_scale.train_docs == (100, 200, 500, 1000, 2000, 4000, 8000, 12000)
    assert policy.markov_equiv.eval_guidance_trials == 8
    assert policy.pilot_cmd_count == 240
