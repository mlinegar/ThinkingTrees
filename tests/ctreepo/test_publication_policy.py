from __future__ import annotations

from src.ctreepo.sim.suite.publication_policy import (
    PublicationCtreepoPolicy,
    resolve_publication_ctreepo_policy,
)


def test_publication_policy_smoke_defaults_match_expected_grid() -> None:
    policy = resolve_publication_ctreepo_policy("smoke")
    assert isinstance(policy, PublicationCtreepoPolicy)
    assert policy.profile == "smoke"
    assert policy.seeds == (0,)
    assert policy.train_docs_lda == (128,)
    assert policy.leaf_tokens_hard == (16,)
    assert policy.n_books_test_hard == 32


def test_publication_policy_publication_defaults_match_expected_grid() -> None:
    policy = resolve_publication_ctreepo_policy("publication")
    assert policy.profile == "publication"
    assert policy.seeds == (0, 1, 2, 3, 4, 5, 6, 7)
    assert policy.q_rates == (0.0, 0.25, 0.5)
    assert policy.train_docs_hard_upper == (1024, 2048, 4096)
    assert policy.cal_rates_lda == (0.0, 0.05, 0.1)
