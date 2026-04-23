from __future__ import annotations

from src.training.search_trace import (
    SearchSpec,
    expand_search_trials,
    select_best_trial,
)


def test_expand_search_trials_grid_and_false_flag() -> None:
    spec = SearchSpec.from_dict(
        {
            "mode": "grid",
            "dimensions": [
                {"flag": "--lr", "values": [0.001, 0.0005]},
                {"flag": "--delta-head", "values": [True, False], "false_flag": "--no-delta-head"},
            ],
        }
    )

    trials = expand_search_trials(spec, base_seed=7)

    assert len(trials) == 4
    assert trials[0]["trial_id"] == "trial_000"
    assert trials[0]["seed"] == 7
    assert trials[1]["seed"] == 8
    assert trials[0]["arg_tokens"] == ["--lr", "0.001", "--delta-head"]
    assert trials[1]["arg_tokens"] == ["--lr", "0.001", "--no-delta-head"]


def test_select_best_trial_uses_runtime_as_tie_breaker() -> None:
    selected = select_best_trial(
        [
            {
                "trial_id": "trial_000",
                "trial_index": 0,
                "success": True,
                "selection_metrics": {
                    "validation_mae": 0.12,
                    "training_time_seconds": 6.0,
                },
            },
            {
                "trial_id": "trial_001",
                "trial_index": 1,
                "success": True,
                "selection_metrics": {
                    "validation_mae": 0.12,
                    "training_time_seconds": 5.0,
                },
            },
        ],
        selection_metric="validation_mae",
        tie_breaker_metric="training_time_seconds",
    )

    assert selected is not None
    assert selected["trial_id"] == "trial_001"
