from __future__ import annotations

import pytest

from src.core.local_law_adjustment import (
    LocalLawObservation,
    aggregate_local_law_observations,
    corrected_local_law_loss,
    local_law_objective_mean,
)


def test_corrected_local_law_loss_keeps_proxy_when_unobserved() -> None:
    assert corrected_local_law_loss(
        proxy_loss=0.4,
        oracle_loss=None,
        observed=False,
        propensity=0.0,
    ) == pytest.approx(0.4)


def test_corrected_local_law_loss_returns_oracle_when_fully_observed() -> None:
    assert corrected_local_law_loss(
        proxy_loss=0.4,
        oracle_loss=0.1,
        observed=True,
        propensity=1.0,
    ) == pytest.approx(0.1)


def test_corrected_local_law_loss_applies_ipw_residual() -> None:
    assert corrected_local_law_loss(
        proxy_loss=0.4,
        oracle_loss=0.1,
        observed=True,
        propensity=0.5,
    ) == pytest.approx(-0.2)


def test_aggregate_local_law_observations_reports_diagnostics() -> None:
    aggregate = aggregate_local_law_observations(
        [
            LocalLawObservation(proxy_loss=0.4, observed=False, propensity=0.0, depth=0),
            LocalLawObservation(proxy_loss=0.3, oracle_loss=0.1, observed=True, propensity=0.5, depth=1),
        ],
        gamma_depth=0.5,
        local_law_weight=0.4,
    )

    assert aggregate.population_count == 2
    assert aggregate.sampled_count == 1
    assert aggregate.proxy_total == pytest.approx(0.4 + 0.5 * 0.3)
    assert aggregate.corrected_total == pytest.approx(0.4 + 0.5 * -0.1)
    assert aggregate.local_law_weight == pytest.approx(0.4)


def test_local_law_objective_mean_corrected_uses_loss_rows() -> None:
    objective = local_law_objective_mean(
        [
            LocalLawObservation(proxy_loss=0.25, observed=False, propensity=0.0, depth=0),
            LocalLawObservation(
                proxy_loss=0.5,
                oracle_loss=0.25,
                observed=True,
                propensity=0.5,
                depth=1,
            ),
        ],
        gamma_depth=0.5,
        objective_mode="corrected_local_law",
    )

    assert objective == pytest.approx((0.25 + 0.5 * 0.0) / 1.5)


def test_local_law_objective_mean_sampled_ipw_uses_oracle_rows_only() -> None:
    objective = local_law_objective_mean(
        [
            LocalLawObservation(proxy_loss=99.0, observed=False, propensity=0.0, depth=0),
            LocalLawObservation(
                proxy_loss=99.0,
                oracle_loss=0.25,
                observed=True,
                propensity=0.25,
                depth=0,
            ),
            LocalLawObservation(
                proxy_loss=99.0,
                oracle_loss=1.0,
                observed=True,
                propensity=0.5,
                depth=0,
            ),
        ],
        objective_mode="sampled_ipw",
    )

    assert objective == pytest.approx((4.0 * 0.25 + 2.0 * 1.0) / 6.0)


def test_local_law_objective_mean_sampled_ipw_no_observations_zero() -> None:
    assert local_law_objective_mean(
        [LocalLawObservation(proxy_loss=0.25, observed=False, propensity=0.0)],
        objective_mode="sampled_ipw",
    ) == pytest.approx(0.0)
