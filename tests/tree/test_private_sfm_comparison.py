import pytest

from src.tree.private_sfm_comparison import (
    SFMComparisonConfig,
    run_sfm_style_comparison,
)


def _tiny_config(**overrides) -> SFMComparisonConfig:
    base = dict(
        n_values=(200,),
        n_trials=24,
        merge_counts=(2,),
        universe_size=2_000_000,
        epsilons=(1.0,),
        buckets=128,
        levels=8,
        n_min_est=1,
        n_max_est=100_000,
        include_hll_non_private=False,
        include_ours_ridge_sym=False,
        seed=7,
    )
    base.update(overrides)
    return SFMComparisonConfig(**base)


def test_theory_floor_gap_is_reported():
    summary = run_sfm_style_comparison(_tiny_config(enable_ipw=False))
    assert len(summary.rows) > 0

    for row in summary.rows:
        assert row.theory_rrmse_floor is not None
        assert row.theory_rrmse_floor >= 0.0
        assert row.rrmse_gap_to_theory_floor == pytest.approx(
            max(0.0, row.rrmse - row.theory_rrmse_floor)
        )
        assert row.ipw_audit_rate is None
        assert row.ipw_preference_loss is None
        assert row.true_preference_loss is None


def test_ipw_full_audit_matches_true_preference_loss():
    summary = run_sfm_style_comparison(
        _tiny_config(
            enable_ipw=True,
            ipw_audit_rates=(1.0,),
            ipw_sampling_scheme="prediction_stratified",
            ipw_delta=0.05,
        )
    )
    assert len(summary.rows) > 0

    for row in summary.rows:
        assert row.ipw_audit_rate == pytest.approx(1.0)
        assert row.ipw_sample_count == row.n_trials
        assert row.true_preference_loss is not None
        assert row.ipw_preference_loss is not None
        assert row.ipw_preference_loss == pytest.approx(row.true_preference_loss, abs=1e-10)
        assert row.ipw_preference_ci_low is not None
        assert row.ipw_preference_ci_high is not None
        assert row.ipw_preference_ci_low <= row.true_preference_loss <= row.ipw_preference_ci_high

