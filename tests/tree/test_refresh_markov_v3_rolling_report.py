from __future__ import annotations

import json
from pathlib import Path

import scripts.refresh_markov_v3_rolling_report as rolling_refresh


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_discover_v3_bundle_records_uses_dynamic_discovery_and_excludes_shadow_roots(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    outputs_root = tmp_path / "outputs"
    bundle_root = (
        outputs_root
        / "markov_v3_publication_fullval_20260412_0142"
        / "root_budget_publication_multileaf_fullval"
    )
    _write_json(
        bundle_root / "experiment_status.json",
        {
            "state": "completed",
            "completed_items": 10,
            "failed_items": 0,
            "active_items": 0,
            "pending_items": 0,
        },
    )
    shadow_root = (
        outputs_root
        / "markov_v3_depth_equal_optimized_shadow_20260413_172523"
        / "depth_equal_publication_focus_xlarge"
    )
    _write_json(
        shadow_root / "experiment_status.json",
        {
            "state": "running",
            "completed_items": 1,
            "failed_items": 0,
            "active_items": 2,
            "pending_items": 3,
        },
    )

    monkeypatch.setattr(rolling_refresh, "OUTPUTS_ROOT", outputs_root)
    records = rolling_refresh._discover_v3_bundle_records(outputs_root)

    assert len(records) == 1
    record = records[0]
    assert record["root_name"] == "markov_v3_publication_fullval_20260412_0142"
    assert record["root_prefix"] == "markov_v3_publication_fullval_"
    assert record["bundle_name"] == "root_budget_publication_multileaf_fullval"
    assert record["bundle_group"] == "multileaf_root_budget"
    assert record["source_tier"] == "publication_fullval"
    assert record["attempt_lineage"] == (
        "markov_v3_publication_fullval_20260412_0142/"
        "root_budget_publication_multileaf_fullval"
    )
    assert "recoverable_ordered_families_leaf064" in record["affected_panels"]


def test_materialize_panel_slots_renders_placeholder_for_missing_bundle_summary(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "report"
    summary = {"figures": {}}
    bundle_records = [
        {
            "bundle_label": "markov_v3_publication_fullval_20260412_0142/oneleaf_root_budget_publication_fullval",
            "bundle_group": "oneleaf_root_budget",
            "summary_ready": False,
        }
    ]

    coverage = rolling_refresh._materialize_panel_slots(
        output_dir,
        summary=summary,
        bundle_records=bundle_records,
    )

    panel = coverage["recoverable_ordered_families_leaf128"]
    assert panel["status"] == "placeholder"
    assert panel["missing_bundles"] == [
        "markov_v3_publication_fullval_20260412_0142/oneleaf_root_budget_publication_fullval"
    ]
    assert Path(panel["panel_path"]).exists()
