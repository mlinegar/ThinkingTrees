from __future__ import annotations

import json
import math
from pathlib import Path

from src.ctreepo.sim.suite.identifiable_zero_learnability import main as learnability_suite_main
from src.ctreepo.sim.suite.markov_observed_token import main as markov_observed_token_suite_main
from src.ctreepo.sim.suite.learned_sketch_smoke import main as learned_sketch_suite_main


REQUIRED_SUITE_META_KEYS = {
    "schema_version",
    "suite_name",
    "suite_role",
    "profile",
    "policy",
    "manifest_file",
    "group_manifest_files",
    "selected_groups",
}


def test_learned_sketch_smoke_suite_e2e(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    out_root = tmp_path / "learned_sketch_smoke"

    assert learned_sketch_suite_main(["build", "--output-root", str(out_root)]) == 0
    assert learned_sketch_suite_main(
        ["run", "--output-root", str(out_root), "--jobs", "1", "--gpu-tokens", "none"]
    ) == 0
    assert learned_sketch_suite_main(
        ["report", "--output-root", str(out_root), "--no-emit-pdf"]
    ) == 0

    meta = json.loads((out_root / "suite_meta.json").read_text(encoding="utf-8"))
    assert REQUIRED_SUITE_META_KEYS <= set(meta)
    assert meta["suite_name"] == "learned-sketch-smoke"

    summary_files = sorted((out_root / "learned_sketch_simulation").rglob("*.json"))
    assert len(summary_files) == 1
    payload = json.loads(summary_files[0].read_text(encoding="utf-8"))
    rows = list(payload.get("rows", []) or [])
    assert len(rows) == 1
    row = rows[0]
    assert math.isfinite(float(row["learned_relative_rmse"]))
    assert math.isfinite(float(row["hll_relative_rmse"]))
    assert math.isfinite(float(row["distance_to_hll_floor_rel_rmse"]))
    assert math.isfinite(float(row["learned_schedule_spread_mean"]))
    assert math.isfinite(float(row["train_total_queries_estimate"]))
    assert float(row["hll_schedule_spread_mean"]) == 0.0

    diagnostics = json.loads(
        (out_root / "figures" / "learned_sketch_smoke" / "learned_sketch_smoke_latest_diagnostics.json").read_text(
            encoding="utf-8"
        )
    )
    assert diagnostics["row_count"] == 1
    assert diagnostics["metric_checks"]["has_rows"]["pass"] is True
    assert diagnostics["metric_checks"]["finite_learned_relative_rmse"]["pass"] is True
    assert diagnostics["metric_checks"]["finite_hll_relative_rmse"]["pass"] is True
    assert diagnostics["metric_checks"]["finite_distance_to_hll_floor_rel_rmse"]["pass"] is True
    assert diagnostics["metric_checks"]["finite_learned_schedule_spread_mean"]["pass"] is True
    assert diagnostics["metric_checks"]["finite_train_total_queries_estimate"]["pass"] is True
    assert diagnostics["metric_checks"]["hll_schedule_spread_zero"]["pass"] is True


def test_markov_learnability_smoke_suite_e2e(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    out_root = tmp_path / "markov_learnability_smoke"

    assert learnability_suite_main(
        [
            "build",
            "--output-root",
            str(out_root),
            "--profile",
            "smoke",
            "--groups",
            "markov_baseline",
            "--markov-device",
            "cpu",
            "--torch-threads",
            "1",
        ]
    ) == 0
    assert learnability_suite_main(
        [
            "run",
            "--output-root",
            str(out_root),
            "--profile",
            "smoke",
            "--groups",
            "markov_baseline",
            "--jobs",
            "1",
            "--gpu-tokens",
            "none",
        ]
    ) == 0
    assert learnability_suite_main(
        ["report", "--output-root", str(out_root), "--no-emit-pdf"]
    ) == 0

    diagnostics = json.loads(
        (
            out_root / "figures" / "learnability" / "identifiable_zero_learnability_latest_diagnostics.json"
        ).read_text(encoding="utf-8")
    )
    assert diagnostics["subset_mode"] == "markov_only"
    assert diagnostics["markov"]["n_rows"] == 2
    assert diagnostics["markov"]["sampled_leaf_pool_rows"] == 16
    assert diagnostics["ctree"]["n_rows"] == 0
    assert diagnostics["setup_alignment"]["applicable"] is False
    sampled_eff = dict(diagnostics["markov"].get("sampled_leaf_pool_efficiency", {}) or {})
    assert "baseline" in sampled_eff
    assert sampled_eff["baseline"]["selected_slice"]["train_docs"] == 16
    assert math.isclose(float(sampled_eff["baseline"]["selected_slice"]["audit_fraction"]), 0.1)
    assert "sampled_leaf_pool_ridge" in sampled_eff["baseline"]["points"]
    assert "sampled_leaf_pool_rf" in sampled_eff["baseline"]["points"]
    ladder = dict(diagnostics["markov"].get("capacity_ladder", {}) or {})
    assert "baseline" in ladder
    assert ladder["baseline"]["selected_slice"]["train_docs"] == 16
    ladder_labels = [str(point["label"]) for point in ladder["baseline"]["points"]]
    assert "leaf bucket" in ladder_labels
    assert "leaf endpoint table" in ladder_labels
    assert "leaf DT tree" in ladder_labels
    assert "leaf RF tree" in ladder_labels

    markov_jsons = sorted((out_root / "markov_changepoint_ops_count").rglob("seed_0.json"))
    assert len(markov_jsons) == 2

    seen_families = set()
    for path in markov_jsons:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = dict(payload.get("config", {}) or {})
        metrics = dict(payload.get("metrics", {}) or {})
        seen_families.add(str(config.get("model_family", "")))
        learned = dict(metrics.get("learned", {}) or {})
        leaf_bucket = dict(metrics.get("leaf_bucket", {}) or {})
        leaf_ridge_tree = dict(metrics.get("leaf_ridge_tree", {}) or {})
        leaf_endpoint_table_tree = dict(metrics.get("leaf_endpoint_table_tree", {}) or {})
        leaf_dt_tree = dict(metrics.get("leaf_dt_tree", {}) or {})
        leaf_knn_tree = dict(metrics.get("leaf_knn_tree", {}) or {})
        leaf_rf_tree = dict(metrics.get("leaf_rf_tree", {}) or {})
        doc_level = dict(metrics.get("doc_level", {}) or {})
        doc_level_ridge = dict(metrics.get("doc_level_ridge", {}) or {})
        rf_root = dict(metrics.get("rf_root", {}) or {})
        sample_sweep = dict(metrics.get("sampled_leaf_pool_budget_sweep", {}) or {})
        exact = dict(metrics.get("exact", {}) or {})

        assert math.isfinite(float(learned["root_mae"]))
        assert math.isfinite(float(learned["schedule_spread_mean"]))
        assert math.isfinite(float(leaf_bucket["root_mae"]))
        assert float(leaf_bucket["schedule_spread_mean"]) == 0.0
        assert math.isfinite(float(leaf_ridge_tree["root_mae"]))
        assert math.isfinite(float(leaf_ridge_tree["schedule_spread_mean"]))
        assert math.isfinite(float(leaf_endpoint_table_tree["root_mae"]))
        assert float(leaf_endpoint_table_tree["schedule_spread_mean"]) == 0.0
        assert math.isfinite(float(leaf_dt_tree["root_mae"]))
        assert float(leaf_dt_tree["schedule_spread_mean"]) == 0.0
        assert math.isfinite(float(leaf_knn_tree["root_mae"]))
        assert float(leaf_knn_tree["schedule_spread_mean"]) == 0.0
        assert math.isfinite(float(leaf_rf_tree["root_mae"]))
        assert float(leaf_rf_tree["schedule_spread_mean"]) == 0.0
        assert math.isfinite(float(doc_level["root_mae"]))
        assert math.isfinite(float(doc_level_ridge["root_mae"]))
        assert math.isfinite(float(rf_root["root_mae"]))
        assert float(exact["root_mae"]) == 0.0
        points = list(sample_sweep.get("points") or [])
        assert [int(point["leaf_budget"]) for point in points] == [1, 2, 4, 8]

    assert seen_families == {"neural", "additive"}


def test_markov_observed_token_suite_e2e(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    out_root = tmp_path / "markov_observed_token"

    assert markov_observed_token_suite_main(
        [
            "build",
            "--output-root",
            str(out_root),
            "--profile",
            "smoke",
            "--device",
            "cpu",
            "--torch-threads",
            "1",
        ]
    ) == 0
    assert markov_observed_token_suite_main(
        [
            "run",
            "--output-root",
            str(out_root),
            "--jobs",
            "1",
            "--gpu-tokens",
            "none",
        ]
    ) == 0
    assert markov_observed_token_suite_main(
        ["report", "--output-root", str(out_root), "--no-emit-pdf"]
    ) == 0

    meta = json.loads((out_root / "suite_meta.json").read_text(encoding="utf-8"))
    assert REQUIRED_SUITE_META_KEYS <= set(meta)
    assert meta["suite_name"] == "markov-observed-token"
    assert meta["selected_groups"] == ["root_only", "local_labels"]
    bundle_path = Path(str(meta["data_bundle_file"]))
    assert bundle_path.exists()

    root_only_path = (
        out_root / "markov_changepoint_ops_count" / "root_only" / "seed_0.json"
    )
    local_labels_path = (
        out_root / "markov_changepoint_ops_count" / "local_labels" / "seed_0.json"
    )
    root_only = json.loads(root_only_path.read_text(encoding="utf-8"))
    local_labels = json.loads(local_labels_path.read_text(encoding="utf-8"))

    root_cfg = dict(root_only.get("config", {}) or {})
    local_cfg = dict(local_labels.get("config", {}) or {})
    root_metrics = dict(root_only.get("metrics", {}) or {})
    local_metrics = dict(local_labels.get("metrics", {}) or {})

    assert root_cfg["feature_mode"] == "token_full"
    assert local_cfg["feature_mode"] == "token_full"
    assert root_cfg["generator_profile"] == "piecewise_palette"
    assert local_cfg["generator_profile"] == "piecewise_palette"
    assert root_cfg["data_bundle_source"] == "provided"
    assert local_cfg["data_bundle_source"] == "provided"
    assert root_cfg["degenerate_root_target_detected"] is False
    assert local_cfg["degenerate_root_target_detected"] is False
    assert root_cfg["train_corpus_signature"] == local_cfg["train_corpus_signature"]
    assert root_cfg["val_corpus_signature"] == local_cfg["val_corpus_signature"]
    assert root_cfg["test_corpus_signature"] == local_cfg["test_corpus_signature"]
    assert root_cfg["full_sequence_input_backend"] == "shared_token_sequence_arrays"
    assert root_cfg["full_sequence_input_signatures"] == local_cfg["full_sequence_input_signatures"]
    assert (
        root_metrics["doc_sequence_training"]["sequence_input_signatures"]
        == root_metrics["doc_transformer_training"]["sequence_input_signatures"]
    )
    assert root_metrics["doc_sequence_training"]["root_label_only_supervision"] is True
    assert root_metrics["doc_transformer_training"]["root_label_only_supervision"] is True
    assert math.isfinite(float(root_metrics["doc_sequence_training"]["test_exact_match_rate"]))
    assert math.isfinite(float(root_metrics["doc_transformer_training"]["test_exact_match_rate"]))
    assert math.isfinite(float(root_metrics["learned"]["root_mae"]))
    assert math.isfinite(float(root_metrics["doc_sequence"]["root_mae"]))
    assert math.isfinite(float(root_metrics["doc_transformer"]["root_mae"]))
    assert math.isfinite(float(root_metrics["doc_level"]["root_mae"]))
    assert math.isfinite(float(root_metrics["doc_level_ridge"]["root_mae"]))
    assert math.isfinite(float(root_metrics["doc_level_ridge_unigram"]["root_mae"]))
    assert math.isfinite(float(root_metrics["doc_level_ridge_bigram"]["root_mae"]))
    assert math.isfinite(float(root_metrics["doc_level_ridge_trigram"]["root_mae"]))
    assert math.isfinite(float(local_metrics["learned"]["root_mae"]))
    assert math.isfinite(float(local_metrics["doc_sequence"]["root_mae"]))
    assert math.isfinite(float(local_metrics["doc_transformer"]["root_mae"]))
    assert math.isfinite(float(local_metrics["leaf_endpoint_table_tree"]["root_mae"]))
    assert math.isfinite(float(local_metrics["leaf_dt_tree"]["root_mae"]))
    assert math.isfinite(float(local_metrics["leaf_knn_tree"]["root_mae"]))
    assert math.isfinite(float(local_metrics["leaf_rf_tree"]["root_mae"]))
    assert float(local_metrics["leaf_endpoint_table_tree"]["schedule_spread_mean"]) == 0.0
    assert float(local_metrics["leaf_dt_tree"]["schedule_spread_mean"]) == 0.0
    assert float(local_metrics["leaf_knn_tree"]["schedule_spread_mean"]) == 0.0
    assert float(local_metrics["leaf_rf_tree"]["schedule_spread_mean"]) == 0.0

    diagnostics = json.loads(
        (
            out_root
            / "figures"
            / "markov_observed_token"
            / "markov_observed_token_latest_diagnostics.json"
        ).read_text(encoding="utf-8")
    )
    assert diagnostics["checks"]["matching_split_signatures"]["pass"] is True
    assert diagnostics["checks"]["token_only_features"]["pass"] is True
    assert diagnostics["checks"]["root_only_has_no_local_labels"]["pass"] is True
    assert diagnostics["checks"]["local_labels_enabled"]["pass"] is True
    assert diagnostics["checks"]["full_doc_sequence_learning_is_finite"]["pass"] is True
    assert diagnostics["checks"]["full_doc_transformer_learning_is_finite"]["pass"] is True
    assert diagnostics["checks"]["nondegenerate_root_target"]["pass"] is True
    assert diagnostics["checks"]["simple_local_controls_preserve_global_structure"]["pass"] is True
