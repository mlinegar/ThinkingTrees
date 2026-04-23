from __future__ import annotations

from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (
    _attach_markov_witness_gap_fields,
    _payload_from_saved_runs,
)


def _base_row(
    *,
    family: str,
    train_docs: int,
    test_root: float,
    train_root: float = 0.02,
    val_root: float = 0.03,
    objective_variant: str = "count_ce_only",
    device_requested: str = "cpu",
    device_resolved: str = "cpu",
) -> dict[str, object]:
    return {
        "benchmark": "recoverable_v4",
        "cell_id": "recoverable_core",
        "baseline_family": family,
        "train_doc_count": train_docs,
        "fixed_leaf_tokens": 64,
        "n_regimes": 4,
        "test_root_mae_mean": test_root,
        "train_root_mae_mean": train_root,
        "val_root_mae_mean": val_root,
        "comparison_semantics_label": "recoverable",
        "selection_metric": "val_root_mae_mean",
        "objective_variant": objective_variant,
        "device_requested": device_requested,
        "device_resolved": device_resolved,
        "seed": 0,
    }


def test_witness_gap_rules_block_false_information_barrier() -> None:
    rows = _attach_markov_witness_gap_fields(
        [
            _base_row(family="palette_block_exact", train_docs=100, test_root=0.0),
            _base_row(family="ridge_control", train_docs=100, test_root=0.005),
            _base_row(
                family="official_fno",
                train_docs=100,
                test_root=0.18,
                train_root=0.15,
                val_root=0.17,
            ),
        ]
    )
    by_family = {str(row["baseline_family"]): row for row in rows}
    assert by_family["official_fno"]["cause_code"] != "information_barrier"
    assert float(by_family["official_fno"]["gap_to_ridge_control"]) > 0.0


def test_witness_gap_rules_flag_objective_mismatch_and_scaling_limit() -> None:
    rows = _attach_markov_witness_gap_fields(
        [
            _base_row(family="palette_block_exact", train_docs=100, test_root=0.0),
            _base_row(family="ridge_control", train_docs=100, test_root=0.002),
            _base_row(
                family="official_fno_sumlen",
                train_docs=100,
                test_root=0.14,
                train_root=0.08,
                val_root=0.10,
                objective_variant="count_ce_only",
            ),
            _base_row(
                family="official_fno_sumlen",
                train_docs=100,
                test_root=0.21,
                train_root=0.09,
                val_root=0.12,
                objective_variant="count_ce_plus_scalar_mse",
            ),
            _base_row(
                family="cnn1d",
                train_docs=100,
                test_root=0.20,
                train_root=0.09,
                val_root=0.13,
            ),
            _base_row(
                family="cnn1d",
                train_docs=1000,
                test_root=0.12,
                train_root=0.05,
                val_root=0.07,
            ),
        ]
    )
    by_key = {
        (str(row["baseline_family"]), int(row["train_doc_count"]), str(row["objective_variant"])): row
        for row in rows
    }
    assert (
        by_key[("official_fno_sumlen", 100, "count_ce_plus_scalar_mse")]["cause_code"]
        == "objective_mismatch"
    )
    assert by_key[("cnn1d", 100, "count_ce_only")]["cause_code"] == "optimization_limit"


def test_payload_from_saved_runs_exposes_witness_gap_table_and_cause_codes() -> None:
    payload = _payload_from_saved_runs(
        runs=[
            _base_row(family="palette_block_exact", train_docs=100, test_root=0.0),
            _base_row(family="ridge_control", train_docs=100, test_root=0.001),
            _base_row(
                family="official_fno",
                train_docs=100,
                test_root=0.16,
                train_root=0.10,
                val_root=0.12,
                device_requested="cuda",
                device_resolved="cpu",
            ),
        ]
    )

    assert payload["selection_metric_curve_summary"]
    assert payload["backend_device_summary"]
    assert payload["witness_gap_table"]
    row = next(
        item for item in payload["witness_gap_table"] if item["baseline_family"] == "official_fno"
    )
    assert "gap_to_ridge_control" in row
    assert "gap_to_exact_witness" in row
    assert row["cause_code"] == "implementation_path_issue"
