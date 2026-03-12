from __future__ import annotations

from src.benchmark.memory_defaults import (
    parse_temporal_main_scenario_id,
    recommend_manifesto_memory_defaults,
)


def test_parse_temporal_main_scenario_id() -> None:
    parsed = parse_temporal_main_scenario_id("temporal_main_sem_on_learned_chunker")
    assert parsed is not None
    assert parsed["semantic_memory_features"] is True
    assert parsed["learn_loss_weights"] is True
    assert parsed["windowing_mode"] == "chunker"


def test_recommend_defaults_prefers_eligible_candidate() -> None:
    artifact = {
        "results": [
            {
                "id": "temporal_main_sem_on_learned_chunker",
                "status": "passed",
                "actual_outcome": "pass",
                "expectation_met": True,
                "metrics": {
                    "test_rile_mae": 0.12,
                    "test_delta_count": 30,
                    "test_delta_improvement": 0.09,
                    "val_delta_improvement": 0.07,
                },
            },
            {
                "id": "temporal_main_sem_off_fixed_uniform",
                "status": "passed",
                "actual_outcome": "pass",
                "expectation_met": True,
                "metrics": {
                    "test_rile_mae": 0.10,
                    "test_delta_count": 30,
                    "test_delta_improvement": 0.08,
                    "val_delta_improvement": 0.06,
                },
            },
            {
                "id": "temporal_main_sem_on_fixed_uniform",
                "status": "passed",
                "actual_outcome": "pass",
                "expectation_met": True,
                "metrics": {
                    "test_rile_mae": 0.40,
                    "test_delta_count": 30,
                    "test_delta_improvement": 0.25,
                    "val_delta_improvement": 0.20,
                },
            },
        ]
    }

    rec = recommend_manifesto_memory_defaults(artifact, min_delta_count=20, max_rile_mae=0.20)
    assert rec["selected_scenario_id"] == "temporal_main_sem_on_learned_chunker"
    defaults = rec["recommended_defaults"]
    assert defaults["semantic_memory_features"] is True
    assert defaults["learn_loss_weights"] is True
    assert defaults["windowing_mode"] == "chunker"


def test_recommend_defaults_falls_back_when_no_valid_candidates() -> None:
    artifact = {
        "results": [
            {
                "id": "temporal_main_sem_off_fixed_uniform",
                "status": "failed",
                "actual_outcome": "fail",
                "expectation_met": False,
                "metrics": {},
            }
        ]
    }
    rec = recommend_manifesto_memory_defaults(artifact)
    assert rec["selected_scenario_id"] == "temporal_main_sem_on_learned_chunker"
