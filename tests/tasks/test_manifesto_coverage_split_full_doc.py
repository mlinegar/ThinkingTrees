from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts import build_manifesto_coverage_split as split_cli
from scripts import run_manifesto_full_doc_dspy_global_f as global_f_cli
from scripts import run_manifesto_full_doc_gemma4_benchmark as full_doc_cli
from src.ctreepo.run_registry import get_run_target, run_targets_by_name
from src.experiments.adapters import _inferred_role_refs, _script_family


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _source_root(tmp_path: Path, *, dimensions: tuple[str, ...], n_docs: int = 12) -> Path:
    source_root = tmp_path / "source"
    for dim_index, dim in enumerate(dimensions):
        rows = [
            {
                "manifesto_id": f"doc_{idx:03d}",
                "benoit_expert_mean_1_7": 1.0 + ((idx + dim_index) % 7),
            }
            for idx in range(n_docs)
        ]
        _write_jsonl(source_root / dim / "per_manifesto.jsonl", rows)
    return source_root


class _FakeManifestoDataset:
    def __init__(self, *args, **kwargs) -> None:
        self.samples = {
            f"doc_{idx:03d}": SimpleNamespace(
                text=("short manifesto text " if idx % 2 else "long manifesto text " * (idx + 2)),
                party_id=1000 + idx,
                year=2000 + idx,
            )
            for idx in range(20)
        }

    def get_sample(self, doc_id: str):
        return self.samples.get(str(doc_id))


def test_manifesto_coverage_split_writes_run_manifest_and_stable_digest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dimensions = ("economic", "social", "environment")
    source_root = _source_root(tmp_path, dimensions=dimensions)
    monkeypatch.setattr(split_cli, "ManifestoDataset", _FakeManifestoDataset)

    first = split_cli.build_coverage_split(
        source_root=source_root,
        output_dir=tmp_path / "split_a",
        dimensions=dimensions,
        train_n=5,
        val_n=2,
        test_n=2,
        seed=0,
        mp_data_dir=None,
        length_floor_chars=8,
        max_weight_ratio=4.0,
    )
    second = split_cli.build_coverage_split(
        source_root=source_root,
        output_dir=tmp_path / "split_b",
        dimensions=dimensions,
        train_n=5,
        val_n=2,
        test_n=2,
        seed=0,
        mp_data_dir=None,
        length_floor_chars=8,
        max_weight_ratio=4.0,
    )

    assert first["split_manifest_digest"] == second["split_manifest_digest"]
    split_ids = json.loads((tmp_path / "split_a" / "split_ids.json").read_text(encoding="utf-8"))
    assert set(split_ids) == {"train", "val", "test"}
    assert not (set(split_ids["train"]) & set(split_ids["val"]))
    assert not (set(split_ids["train"]) & set(split_ids["test"]))
    assert all("sampling_weight" in row for row in first["selected_docs"])

    run_manifest = json.loads((tmp_path / "split_a" / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["schema_version"] == "ctreepo.run_manifest.v1"
    assert run_manifest["role"] == "coverage_split_builder"
    assert run_manifest["metadata"]["sampling_plan"]["strategy"] == "all6_soft_inverse_sqrt_length"


def test_full_doc_gemma4_mock_benchmark_writes_common_sidecars(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dimensions = ("economic", "social")
    source_root = _source_root(tmp_path, dimensions=dimensions)
    monkeypatch.setattr(split_cli, "ManifestoDataset", _FakeManifestoDataset)
    monkeypatch.setattr(full_doc_cli, "ManifestoDataset", _FakeManifestoDataset)

    split_cli.build_coverage_split(
        source_root=source_root,
        output_dir=tmp_path / "split",
        dimensions=dimensions,
        train_n=4,
        val_n=2,
        test_n=2,
        seed=0,
        mp_data_dir=None,
        length_floor_chars=8,
        max_weight_ratio=4.0,
    )

    assert (
        full_doc_cli.main(
            [
                "--split-dir",
                str(tmp_path / "split"),
                "--output-dir",
                str(tmp_path / "benchmark"),
                "--source-root",
                str(source_root),
                "--dimensions",
                ",".join(dimensions),
                "--splits",
                "test",
                "--mock-predictions",
                "--max-n",
                "1",
            ]
        )
        == 0
    )

    output_dir = tmp_path / "benchmark"
    for name in (
        "experiment_manifest.json",
        "experiment_status.json",
        "artifacts.json",
        "results.jsonl",
        "predictions.jsonl",
        "calls.jsonl",
        "summary.json",
        "run_manifest.json",
    ):
        assert (output_dir / name).exists()
    rows = [
        json.loads(line)
        for line in (output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert {row["dimension"] for row in rows} == set(dimensions)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["metrics"]["prediction_rows"] == 2
    assert "macro_external_expert_pearson" in summary["metrics"]
    run_manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["input_contracts"][0]["kind"] == "manifesto_coverage_split"


def test_full_doc_dspy_global_f_mock_writes_common_sidecars(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dimensions = ("economic", "social")
    source_root = _source_root(tmp_path, dimensions=dimensions)
    monkeypatch.setattr(split_cli, "ManifestoDataset", _FakeManifestoDataset)
    monkeypatch.setattr(global_f_cli, "ManifestoDataset", _FakeManifestoDataset)

    split_cli.build_coverage_split(
        source_root=source_root,
        output_dir=tmp_path / "split",
        dimensions=dimensions,
        train_n=4,
        val_n=2,
        test_n=2,
        seed=0,
        mp_data_dir=None,
        length_floor_chars=8,
        max_weight_ratio=4.0,
    )

    assert (
        global_f_cli.main(
            [
                "--split-dir",
                str(tmp_path / "split"),
                "--output-dir",
                str(tmp_path / "global_f"),
                "--source-root",
                str(source_root),
                "--dimensions",
                ",".join(dimensions),
                "--train-docs",
                "2",
                "--val-docs",
                "1",
                "--test-docs",
                "1",
                "--optimizer",
                "none",
                "--mock-predictions",
                "--eval-num-threads",
                "1",
            ]
        )
        == 0
    )

    output_dir = tmp_path / "global_f"
    for name in (
        "experiment_manifest.json",
        "experiment_status.json",
        "artifacts.json",
        "results.jsonl",
        "predictions.jsonl",
        "calls.jsonl",
        "summary.json",
        "run_manifest.json",
        "examples_manifest.json",
    ):
        assert (output_dir / name).exists()
    rows = [
        json.loads(line)
        for line in (output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert {row["dimension"] for row in rows} == set(dimensions)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["tree_bundle_kind"] == "raw_manifesto_single_leaf_document"
    assert summary["g_init"] == "identity_single_leaf"
    assert summary["metrics"]["prediction_rows"] == 2
    assert summary["program_save"]["ok"] is True
    assert Path(summary["artifacts"]["program_dir"], "program.pkl").exists()
    assert Path(summary["artifacts"]["program_state_json"]).exists()
    examples_manifest = json.loads((output_dir / "examples_manifest.json").read_text(encoding="utf-8"))
    assert examples_manifest["single_leaf_per_document"] is True
    run_manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["role"] == "full_doc_dspy_global_f"
    assert run_manifest["input_contracts"][1]["tree_bundle_kind"] == "raw_manifesto_single_leaf_document"
    assert Path(run_manifest["f_lineage"]["program_dir"], "program.pkl").exists()

    assert (
        global_f_cli.main(
            [
                "--split-dir",
                str(tmp_path / "split"),
                "--output-dir",
                str(tmp_path / "global_f_loaded"),
                "--source-root",
                str(source_root),
                "--dimensions",
                ",".join(dimensions),
                "--train-docs",
                "2",
                "--val-docs",
                "1",
                "--test-docs",
                "1",
                "--optimizer",
                "none",
                "--mock-predictions",
                "--eval-num-threads",
                "1",
                "--initial-program-dir",
                str(output_dir / "program" / "dspy_program"),
            ]
        )
        == 0
    )
    loaded_manifest = json.loads((tmp_path / "global_f_loaded" / "run_manifest.json").read_text(encoding="utf-8"))
    assert loaded_manifest["f_init"] == "loaded_dspy_global_f_raw_document"
    assert loaded_manifest["f_lineage"]["initial_program_dir"] == str(output_dir / "program" / "dspy_program")


def test_manifesto_run_targets_and_runtime_adapter_register_new_entrypoints() -> None:
    targets = run_targets_by_name()
    assert "manifesto.coverage_split" in targets
    assert "manifesto.full_doc_gemma4_benchmark" in targets
    assert "manifesto.full_doc_dspy_global_f" in targets
    assert get_run_target("manifesto.coverage_split").role == "coverage_split_builder"
    assert get_run_target("manifesto.full_doc_gemma4_benchmark").expected_input_contract == "manifesto_coverage_split"
    assert get_run_target("manifesto.full_doc_dspy_global_f").role == "full_doc_dspy_global_f"

    assert _script_family("build_manifesto_coverage_split.py") == (
        "manifesto_rile",
        "coverage_split",
        "coverage_split_builder",
    )
    assert _script_family("run_manifesto_full_doc_gemma4_benchmark.py") == (
        "manifesto_rile",
        "full_doc_direct_scorer",
        "manifesto_full_doc_direct",
    )
    assert _script_family("run_manifesto_full_doc_dspy_global_f.py") == (
        "manifesto_rile",
        "full_doc_dspy_global_f",
        "manifesto_full_doc_dspy_global_f",
    )
    roles = _inferred_role_refs(
        "run_manifesto_full_doc_gemma4_benchmark.py",
        ["--model", "gemma4", "--base-url", "http://localhost:8010/v1"],
    )
    assert roles["scorer"]["model"] == "gemma4"
    roles = _inferred_role_refs(
        "run_manifesto_full_doc_dspy_global_f.py",
        ["--model", "gemma4", "--base-url", "http://localhost:8010/v1"],
    )
    assert roles["scorer"]["model"] == "gemma4"


def test_supervised_launcher_forwards_split_ids_dir_and_checks_digest() -> None:
    script = Path("scripts/run_benoit_supervised_dspy_ladder.sh").read_text(encoding="utf-8")
    assert 'SPLIT_IDS_DIR="${SPLIT_IDS_DIR:-}"' in script
    assert '--alignment-run-dir "${SPLIT_IDS_DIR}"' in script
    assert "split_manifest_digest mismatch" in script
    assert '"${SPLIT_IDS_DIR}"' in script
