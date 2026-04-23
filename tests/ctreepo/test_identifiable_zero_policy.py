from __future__ import annotations

from src.ctreepo.sim.suite.identifiable_zero_policy import (
    IdentifiableZeroPolicy,
    resolve_identifiable_zero_policy,
)


def test_identifiable_zero_policy_smoke_defaults_match_expected_grid() -> None:
    policy = resolve_identifiable_zero_policy("smoke")
    assert isinstance(policy, IdentifiableZeroPolicy)
    assert policy.profile == "smoke"
    assert policy.segment_train_docs == (200, 500)
    assert policy.ctree_train_docs == (128, 256)
    assert policy.markov_audit_fractions == (0.1, 0.2, 0.5, 1.0)


def test_identifiable_zero_policy_paper_defaults_match_expected_grid() -> None:
    policy = resolve_identifiable_zero_policy("paper")
    assert policy.profile == "paper"
    assert policy.segment_lambda_multipliers == (0.0, 0.25, 1.0)
    assert policy.ctree_eval_internal_rates == (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
    assert policy.markov_train_docs == (100, 200, 500, 1000, 2000)


def test_identifiable_zero_policy_walk_long_defaults_match_expected_grid() -> None:
    policy = resolve_identifiable_zero_policy("walk_long")
    assert policy.profile == "walk_long"
    assert policy.segment_seeds == tuple(range(16))
    assert policy.ctree_eval_leaf_rates == (0.0, 0.25, 0.5, 1.0)
    assert policy.markov_train_docs == (100, 200, 500, 1000, 2000, 4000)
