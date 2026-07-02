from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src.ctreepo.contracts import (
    LEAF_UNIT_STREAM_ITEM,
    LEAF_UNIT_TEXT_TOKEN,
    RUN_MANIFEST_SCHEMA_VERSION,
    SOURCE_KIND_EXTERNAL_STATE,
    SOURCE_KIND_RAW_INPUT,
    run_manifest_metadata,
    tree_bundle_metadata,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TREEPO_SRC = REPO_ROOT / "treepo" / "src"
if str(TREEPO_SRC) not in sys.path:
    sys.path.insert(0, str(TREEPO_SRC))


def _load_script(path: str):
    module_path = REPO_ROOT / path
    spec = importlib.util.spec_from_file_location(module_path.stem, str(module_path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_path.stem] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_publication_entrypoint_audit_registry_is_complete() -> None:
    mod = _load_script("scripts/audit_publication_entrypoints.py")

    assert mod.audit_records() == []


def test_general_run_target_registry_is_complete() -> None:
    mod = _load_script("scripts/audit_run_targets.py")

    assert mod.main(["--check"]) == 0


def test_tree_bundle_audit_requires_runner_manifest_contract(tmp_path: Path) -> None:
    mod = _load_script("scripts/audit_tree_bundle_contracts.py")
    valid = tree_bundle_metadata(
        domain="manifesto_rile",
        leaf_unit=LEAF_UNIT_TEXT_TOKEN,
        source_kind=SOURCE_KIND_RAW_INPUT,
        dimension="economic",
    )
    _write_json(tmp_path / "valid" / "grid_summary.json", {"tree_bundle_manifest": valid["tree_bundle_manifest"]})
    _write_json(tmp_path / "missing" / "grid_summary.json", {"rows": []})

    assert mod.main([str(tmp_path / "valid"), "--require-tree-bundle"]) == 0
    assert mod.main([str(tmp_path / "missing"), "--require-tree-bundle"]) == 2


def test_tree_bundle_audit_requires_external_state_opt_in(tmp_path: Path) -> None:
    mod = _load_script("scripts/audit_tree_bundle_contracts.py")
    external = tree_bundle_metadata(
        domain="manifesto_rile",
        leaf_unit=LEAF_UNIT_TEXT_TOKEN,
        source_kind=SOURCE_KIND_EXTERNAL_STATE,
        dimension="economic",
        external_state_producer="g_benoit",
    )
    _write_json(tmp_path / "summary.json", {"tree_bundle_manifest": external["tree_bundle_manifest"]})

    assert mod.main([str(tmp_path), "--require-tree-bundle"]) == 2
    assert mod.main([str(tmp_path), "--require-tree-bundle", "--allow-external-state"]) == 0


def test_artifact_quarantine_classifies_contract_states(tmp_path: Path) -> None:
    mod = _load_script("scripts/quarantine_ctreepo_artifacts.py")
    valid = tree_bundle_metadata(
        domain="classical_sketch",
        leaf_unit=LEAF_UNIT_STREAM_ITEM,
        source_kind=SOURCE_KIND_RAW_INPUT,
        state_dim=64,
        summary_dim=32,
        include_legacy_manifesto_aliases=False,
    )
    _write_json(tmp_path / "valid" / "summary.json", {"tree_bundle_manifest": valid["tree_bundle_manifest"]})
    run = run_manifest_metadata(
        run_id="runtime.smoke",
        domain="runtime_eval",
        role="longbench_runtime",
        backend="runtime",
        status="planned",
    )
    _write_json(tmp_path / "run" / "run_manifest.json", run)
    _write_json(tmp_path / "missing" / "summary.json", {"rows": []})
    _write_json(
        tmp_path / "bad_dim" / "summary.json",
        {
            "tree_bundle_manifest": {
                **valid["tree_bundle_manifest"],
                "state_dim": 16,
                "summary_dim": 32,
            }
        },
    )
    report_dir = tmp_path / "report"

    assert mod.main([str(tmp_path), "--output-dir", str(report_dir), "--csv"]) == 0
    report = json.loads((report_dir / "artifact_quarantine_report.json").read_text())
    counts = report["counts"]
    assert counts["valid_treebundle_v1"] == 1
    assert counts["valid_run_manifest_v1"] == 1
    assert counts["missing_contract"] == 1
    assert counts["invalid_state_dim"] == 1
    assert (report_dir / "artifact_quarantine_report.csv").exists()


def test_run_manifest_audit_cli_checks_publication_readiness(tmp_path: Path) -> None:
    mod = _load_script("scripts/audit_run_manifests.py")
    tree = tree_bundle_metadata(
        domain="manifesto_rile",
        leaf_unit=LEAF_UNIT_TEXT_TOKEN,
        source_kind=SOURCE_KIND_RAW_INPUT,
        dimension="economic",
    )
    ready = run_manifest_metadata(
        run_id="manifesto.ready",
        domain="manifesto_rile",
        role="fg_ladder_runner",
        backend="dspy",
        tree_bundle=tree,
        f_init="official_oracle",
        g_init="raw_concat",
        f_lineage={"init": "official_oracle"},
        g_lineage={"init": "raw_concat"},
        audit_results={"ok": True},
        publication_ready=True,
    )
    planned = {
        **ready,
        "run_id": "manifesto.planned",
        "status": "planned",
        "publication_ready": False,
    }
    _write_json(tmp_path / "ready" / "run_manifest.json", ready)
    _write_json(tmp_path / "planned" / "run_manifest.json", planned)

    assert mod.main([str(tmp_path / "ready"), "--require-publication-ready", "--require-lineage"]) == 0
    assert mod.main([str(tmp_path / "ready"), "--require-objective"]) == 0
    assert mod.main([str(tmp_path / "planned"), "--require-publication-ready"]) == 2


def test_run_manifest_audit_finds_nested_runner_output_manifest(tmp_path: Path) -> None:
    mod = _load_script("scripts/audit_run_manifests.py")
    run = run_manifest_metadata(
        run_id="nested",
        domain="manifesto_rile",
        role="fg_ladder_runner",
        backend="dspy",
        status="completed",
        f_init="f0",
        g_init="g0",
        f_lineage={"init": "f0"},
        g_lineage={"init": "g0"},
        audit_results={"ok": True},
        quarantine={"classification": "valid_treebundle_v1"},
        publication_ready=True,
    )
    _write_json(tmp_path / "grid_summary.json", {"run_manifest": run, "rows": []})

    assert mod.main([str(tmp_path), "--require-run-manifest"]) == 0


def test_general_runner_plan_only_writes_run_manifest(tmp_path: Path) -> None:
    mod = _load_script("scripts/run_ctreepo.py")

    assert mod.main(
        [
            "--target",
            "runtime.longbench",
            "--plan-only",
            "--output-root",
            str(tmp_path),
        ]
    ) == 0
    payload = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == RUN_MANIFEST_SCHEMA_VERSION
    assert payload["domain"] == "runtime_eval"
    assert payload["status"] == "planned"
    assert payload["metadata"]["expected_input_contract"] == "runtime_eval_manifest"


def test_objective_vocabulary_audit_rejects_public_legacy_terms(tmp_path: Path) -> None:
    mod = _load_script("scripts/audit_objective_vocabulary.py")
    ok = tmp_path / "ok.tex"
    bad = tmp_path / "bad.tex"
    compat = tmp_path / "compat.lean"
    ok.write_text("The public objective uses local_law_weight and root_share.\n", encoding="utf-8")
    bad.write_text(
        "The public JSON emitted selected_lambda_local and oracle_observation_mode.\n",
        encoding="utf-8",
    )
    compat.write_text("-- Backward-compatible alias for oracle recovery.\n", encoding="utf-8")

    assert mod.main([str(ok), str(compat)]) == 0
    assert mod.main([str(bad)]) == 2


def test_treepo_backend_rejects_silent_learned_sketch_ignore(tmp_path: Path) -> None:
    from treepo.bench.runner import run_single

    with pytest.raises(ValueError, match="include_learned=True requires"):
        run_single(
            experiment="classical-sketches",
            config={
                "execution_backend": "treepo",
                "include_learned": True,
                "n_docs": 1,
                "min_tokens": 8,
                "max_tokens": 8,
                "leaf_size": 4,
                "include_families": ("distinct",),
            },
            json_out=tmp_path / "summary.json",
            csv_out=tmp_path / "summary.csv",
        )


def test_classical_sketch_rows_emit_tree_bundle_metadata() -> None:
    from treepo.bench.classical_sketches import (
        ClassicalSketchComparisonConfig,
        run_classical_sketch_comparison,
    )

    summary = run_classical_sketch_comparison(
        ClassicalSketchComparisonConfig(
            n_docs=2,
            min_tokens=8,
            max_tokens=8,
            leaf_size=4,
            include_families=("distinct",),
        )
    )
    row = summary.rows[0]
    manifest = row["tree_bundle_manifest"]
    assert manifest["domain"] == "classical_sketch"
    assert manifest["leaf_unit"] == "stream_item"
    assert manifest["state_contract"] == "oracle_state"
    assert row["f_init"] == "official_oracle"
    assert row["g_init"] == "official_merge"
