import math

import pytest

from scripts.report_markov_optimization_tradeoffs import (
    _build_hazard_panel_mean_guess_check,
    _hazard_panel_mean_guess_lines,
)
from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (
    _bundle_with_fixed_eval_splits,
    resolve_full_doc_diagnostic_grid,
)
from src.ctreepo.sim.core.markov_changepoint_ops_count import (
    MarkovOPSDataBundle,
    _root_count_diagnostics,
)
from src.ctreepo.sim.core.markov_hazard_panels import (
    build_markov_hazard_panel_data_bundle,
    condition_to_ops_overrides,
    resolve_markov_hazard_condition,
    resolve_markov_hazard_panel,
    sticky_markov_switch_probability,
)


def test_hazard_panel_registry_resolves_aliases() -> None:
    panel = resolve_markov_hazard_panel("paper_hazard_panel_v1_t128")
    assert panel.panel_id == "paper_hazard_panel_v1_t128"
    assert [condition.condition_id for condition in panel.conditions] == [
        "paper_v1_t128_r4_p031",
        "paper_v1_t128_r12_p031",
        "paper_v1_t128_r4_p079",
        "paper_v1_t128_r12_p079",
    ]
    long_panel = resolve_markov_hazard_panel("paper_hazard_panel_v1_t2048")
    assert len(long_panel.conditions) == 4
    assert {condition.doc_tokens for condition in long_panel.conditions} == {2048}

    condition = resolve_markov_hazard_condition("structural_core_v2_t128::r12_p079")
    assert condition.condition_id == "paper_v1_t128_r12_p079"
    assert condition.n_regimes == 12

    single = resolve_markov_hazard_panel("recoverable_v5_t2048")
    assert single.conditions[0].condition_id == "recoverable_v5_t2048"


def test_hazard_switch_probability_matches_structural_cell_calibration() -> None:
    condition = resolve_markov_hazard_condition("r4_p031")
    assert condition.hazard_switch_prob == pytest.approx(4.0 / 127.0)
    assert sticky_markov_switch_probability(
        doc_tokens=128,
        expected_boundaries=4.0,
    ) == pytest.approx(4.0 / 127.0)

    overrides = condition_to_ops_overrides(
        condition,
        train_docs=10,
        val_docs=2,
        test_docs=3,
    )
    assert overrides["generator_profile"] == "hazard_topic"
    assert overrides["n_regimes"] == 4
    assert overrides["vocab_size"] == 16
    assert overrides["hazard_switch_prob"] == pytest.approx(4.0 / 127.0)


def test_hazard_panel_bundle_records_stratified_metadata() -> None:
    bundle = build_markov_hazard_panel_data_bundle(
        "paper_hazard_panel_v1_t128",
        train_docs=80,
        val_docs=16,
        test_docs=32,
        seed=7,
    )
    metadata = dict(bundle.metadata)
    assert metadata["hazard_panel_id"] == "paper_hazard_panel_v1_t128"
    assert len(metadata["condition_ids"]["train"]) == 80
    assert len(metadata["condition_ids"]["val"]) == 16
    assert len(metadata["condition_ids"]["test"]) == 32
    assert set(metadata["condition_counts"]["train"]) == {
        "paper_v1_t128_r4_p031",
        "paper_v1_t128_r12_p031",
        "paper_v1_t128_r4_p079",
        "paper_v1_t128_r12_p079",
    }

    diagnostics = _root_count_diagnostics(
        bundle.train_docs,
        condition_ids=metadata["condition_ids"]["train"],
    )
    assert diagnostics["n_unique"] > 1
    assert math.isfinite(float(diagnostics["global_mean_baseline_mae"]))
    assert math.isfinite(float(diagnostics["condition_mean_baseline_mae"]))
    assert diagnostics["condition_diagnostics"]


def test_hazard_panel_paper_bundle_and_prefixes_are_condition_balanced(
    tmp_path,
) -> None:
    bundle = build_markov_hazard_panel_data_bundle(
        "paper_hazard_panel_v1_t128",
        train_docs=10240,
        val_docs=1024,
        test_docs=1024,
        seed=0,
    )
    metadata = dict(bundle.metadata)
    expected_ids = {
        "paper_v1_t128_r4_p031",
        "paper_v1_t128_r12_p031",
        "paper_v1_t128_r4_p079",
        "paper_v1_t128_r12_p079",
    }
    assert metadata["condition_counts"]["train"] == {
        condition_id: 2560 for condition_id in expected_ids
    }
    assert metadata["condition_counts"]["val"] == {
        condition_id: 256 for condition_id in expected_ids
    }
    assert metadata["condition_counts"]["test"] == {
        condition_id: 256 for condition_id in expected_ids
    }

    train_ids = list(metadata["condition_ids"]["train"])
    for prefix, expected_per_condition in ((1024, 256), (4096, 1024), (10240, 2560)):
        counts = {
            condition_id: train_ids[:prefix].count(condition_id)
            for condition_id in expected_ids
        }
        assert counts == {
            condition_id: expected_per_condition for condition_id in expected_ids
        }

    path = tmp_path / "base_bundle.json"
    bundle.save(path)
    loaded = MarkovOPSDataBundle.load(path)
    assert dict(loaded.metadata)["condition_counts"] == metadata["condition_counts"]

    sliced, _source = _bundle_with_fixed_eval_splits(
        base_bundle=loaded,
        base_source=str(path),
        train_doc_count=4096,
    )
    sliced_metadata = dict(sliced.metadata)
    assert len(sliced_metadata["condition_ids"]["train"]) == 4096
    assert sliced_metadata["condition_counts"]["train"] == {
        condition_id: 1024 for condition_id in expected_ids
    }

    diagnostics = _root_count_diagnostics(
        sliced.train_docs,
        condition_ids=sliced_metadata["condition_ids"]["train"],
    )
    assert diagnostics["histogram"]
    assert diagnostics["quantiles"]
    assert math.isfinite(float(diagnostics["global_mean_baseline_mae"]))
    assert math.isfinite(float(diagnostics["condition_mean_baseline_mae"]))
    assert float(diagnostics["mean_guess_gap"]) > 0.0


def test_structural_grid_still_uses_same_v2_overrides() -> None:
    cells = {
        cell.cell_id: cell
        for cell in resolve_full_doc_diagnostic_grid("structural_core_v2_t128")
    }
    r4 = cells["r4_p031"]
    assert r4.hazard_switch_prob == pytest.approx(4.0 / 127.0)
    assert r4.config_overrides["generator_profile"] == "hazard_topic"
    assert r4.config_overrides["n_regimes"] == 4
    assert r4.config_overrides["vocab_size"] == 16
    assert r4.config_overrides["hazard_switch_prob"] == pytest.approx(4.0 / 127.0)

    r12 = cells["r12_p079"]
    assert r12.config_overrides["n_regimes"] == 12
    assert r12.config_overrides["vocab_size"] == 48
    assert r12.config_overrides["hazard_switch_prob"] == pytest.approx(10.0 / 127.0)


def test_report_mean_guess_check_renders_from_recovery_row() -> None:
    summary = {
        "supervision_recovery": {
            "family_rows": [
                {
                    "source_summary_json": "/tmp/run/summary.json",
                    "config": {
                        "hazard_panel_id": "paper_hazard_panel_v1_t128",
                        "test_target_diagnostics": {
                            "n_docs": 32,
                            "global_mean_baseline_mae": 3.0,
                            "condition_mean_baseline_mae": 1.0,
                            "mean_guess_gap": 2.0,
                            "condition_diagnostics": {
                                "paper_v1_t128_r4_p031": {"n_docs": 8},
                                "paper_v1_t128_r12_p079": {"n_docs": 8},
                            },
                        },
                    },
                }
            ],
        }
    }
    check = _build_hazard_panel_mean_guess_check(summary)
    assert check["status"] == "ready"
    assert check["rows"][0]["mean_guess_gap"] == pytest.approx(2.0)

    lines = _hazard_panel_mean_guess_lines(summary)
    assert any("paper_hazard_panel_v1_t128" in line for line in lines)
    assert any("global mean MAE" in line for line in lines)
