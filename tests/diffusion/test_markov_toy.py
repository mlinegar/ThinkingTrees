from __future__ import annotations

from src.diffusion.markov_toy import run_markov_toy_experiment


def test_markov_toy_exact_latent_checkpoint_matches_full_path() -> None:
    payload = run_markov_toy_experiment(["A", "A", "B", "B"], chunk_size=2, rounds=3)

    assert payload["exact_state_matches_full_path"] is True
    assert payload["exact_root_state"]["changepoints"] == 1
    assert payload["budget_formula"]["value"] == 0.0


def test_markov_toy_count_only_counterexample_is_detected() -> None:
    payload = run_markov_toy_experiment(
        ["A", "B"],
        chunk_size=1,
        rounds=2,
        eps_leaf=0.1,
        eps_merge=0.2,
        eps_idemp=0.05,
    )

    assert payload["count_only_matches_full_path"] is False
    assert payload["count_only_root_state"] == 0
    assert payload["count_only_full_path_value"] == 1
    assert payload["budget_formula"]["value"] == 0.35
