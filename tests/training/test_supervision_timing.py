from __future__ import annotations

from src.training.supervision.timing import (
    ACQUISITION_ASYNC_PREFERENCE_QUEUE,
    ACQUISITION_SYNCHRONOUS_OPTIMIZER_METRIC,
    ACTIVATION_EPOCH_BOUNDARY,
    ACTIVATION_IMMEDIATE,
    CONSUMER_CTREEPO_GRADIENT,
    CONSUMER_GEPA_OPTIMIZER,
    default_label_lag_policy,
    supervision_timing_contract,
)


def test_supervision_timing_contract_describes_epoch_lag() -> None:
    timing = supervision_timing_contract(
        acquisition_policy=ACQUISITION_ASYNC_PREFERENCE_QUEUE,
        activation_barrier=ACTIVATION_EPOCH_BOUNDARY,
        consumer=CONSUMER_CTREEPO_GRADIENT,
        producer="teacher_worker",
        delivery_mode="preference_store",
    )

    assert timing["acquisition_policy"] == "async_preference_queue"
    assert timing["activation_barrier"] == "epoch_boundary"
    assert timing["blocking"] is False
    assert timing["label_lag_policy"] == (
        "completed_during_epoch_k_active_no_earlier_than_epoch_k_plus_1"
    )


def test_supervision_timing_contract_describes_gepa_immediate_metric() -> None:
    timing = supervision_timing_contract(
        acquisition_policy=ACQUISITION_SYNCHRONOUS_OPTIMIZER_METRIC,
        activation_barrier=ACTIVATION_IMMEDIATE,
        consumer=CONSUMER_GEPA_OPTIMIZER,
        producer="metric_callback",
        delivery_mode="dspy_gepa_metric",
        blocking=True,
    )

    assert timing["acquisition_policy"] == "synchronous_optimizer_metric"
    assert timing["activation_barrier"] == "immediate"
    assert timing["blocking"] is True
    assert timing["label_lag_policy"] == "completed_labels_active_immediately"
    assert default_label_lag_policy(ACTIVATION_IMMEDIATE) == timing["label_lag_policy"]
