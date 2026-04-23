from __future__ import annotations

from src.ctreepo.sim.suite.learnability_policy import (
    IdentifiableZeroLearnabilityPolicy,
    resolve_identifiable_zero_learnability_policy,
)


def test_identifiable_zero_learnability_policy_defaults_match_expected_grid() -> None:
    policy = IdentifiableZeroLearnabilityPolicy()
    assert policy.profile == "paper"
    assert policy.train_docs_grid == (500, 1000, 2000, 4000, 8000)
    assert policy.label_rate_grid == (0.02, 0.05, 0.1, 0.2, 0.4)
    assert policy.heldout_docs == 2000
    assert policy.base_seeds == (0, 1, 2, 3, 4, 5)
    assert policy.hero_seeds == (6, 7, 8, 9, 10, 11)
    assert policy.markov_sampled_leaf_pool_leaf_counts == tuple()


def test_identifiable_zero_learnability_policy_smoke_profile_defaults_match_expected_grid() -> None:
    policy = resolve_identifiable_zero_learnability_policy(profile_name="smoke")
    assert policy.profile == "smoke"
    assert policy.train_docs_grid == (16,)
    assert policy.label_rate_grid == (0.1,)
    assert policy.heldout_docs == 16
    assert policy.base_seeds == (0,)
    assert policy.markov_sampled_leaf_pool_leaf_counts == (1, 2, 4, 8)


def test_identifiable_zero_learnability_policy_resolves_overrides() -> None:
    policy = resolve_identifiable_zero_learnability_policy(
        profile_name="smoke",
        train_docs_grid="64 128",
        label_rate_grid="0.1 0.25",
        heldout_docs=256,
        base_seeds="1 3 5",
        hero_seeds="8 9",
        ctree_eval_guidance_rates="0 0.5",
        markov_sampled_leaf_pool_leaf_counts="2 6",
    )
    assert policy.profile == "smoke"
    assert policy.train_docs_grid == (64, 128)
    assert policy.label_rate_grid == (0.1, 0.25)
    assert policy.heldout_docs == 256
    assert policy.base_seeds == (1, 3, 5)
    assert policy.hero_seeds == (8, 9)
    assert policy.ctree_eval_guidance_rates == (0.0, 0.5)
    assert policy.markov_sampled_leaf_pool_leaf_counts == (2, 6)
    assert policy.to_shell_exports()["TRAIN_DOCS_GRID"] == "64 128"
