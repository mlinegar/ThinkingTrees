import pytest

from src.tasks.manifesto.expert_scale import (
    EXPERT_SCALE_NORMALIZED_1_7,
    EXPERT_SCALE_RAW,
    expert_scale_bounds,
    normalize_benoit_expert_mean,
    resolve_benoit_expert_target,
    scorer_1_7_to_expert_target,
)


def _pearson(xs, ys):
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / (x_var * y_var) ** 0.5


def test_non_eu_expert_mean_normalizes_from_zero_ten_to_one_seven():
    assert normalize_benoit_expert_mean(0.0, "environment") == 1.0
    assert normalize_benoit_expert_mean(5.0, "environment") == 4.0
    assert normalize_benoit_expert_mean(10.0, "environment") == 7.0


def test_eu_expert_mean_is_native_one_seven():
    assert normalize_benoit_expert_mean(1.11, "eu") == 1.11
    assert normalize_benoit_expert_mean(6.93, "eu") == 6.93


def test_row_resolver_keeps_raw_and_normalized_targets_separate():
    row = {"benoit_expert_mean": 8.0}
    assert resolve_benoit_expert_target(row, dimension="environment", scale=EXPERT_SCALE_RAW) == 8.0
    assert resolve_benoit_expert_target(
        row,
        dimension="environment",
        scale=EXPERT_SCALE_NORMALIZED_1_7,
    ) == pytest.approx(5.8)


def test_environment_raw_target_and_scorer_bounds_are_separate():
    assert expert_scale_bounds(dimension="environment", scale=EXPERT_SCALE_RAW) == (0.0, 10.0)
    assert resolve_benoit_expert_target(
        {"benoit_expert_mean": 5.0},
        dimension="environment",
        scale=EXPERT_SCALE_RAW,
    ) == 5.0
    assert scorer_1_7_to_expert_target(
        4.0,
        dimension="environment",
        scale=EXPERT_SCALE_RAW,
    ) == pytest.approx(5.0)


def test_non_eu_linear_rescale_keeps_pearson_but_changes_mae():
    raw = [0.0, 4.0, 10.0, 8.0]
    normalized = [normalize_benoit_expert_mean(value, "environment") for value in raw]
    predictions = [1.0, 3.0, 6.0, 7.0]

    assert _pearson(predictions, raw) == pytest.approx(_pearson(predictions, normalized))

    raw_mae = sum(abs(pred - truth) for pred, truth in zip(predictions, raw)) / len(raw)
    normalized_mae = sum(abs(pred - truth) for pred, truth in zip(predictions, normalized)) / len(raw)
    assert raw_mae != pytest.approx(normalized_mae)
