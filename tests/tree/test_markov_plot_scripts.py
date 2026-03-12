from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_markov_payload(path: Path, *, train_docs: int, seed: int, local_law_weight: float) -> None:
    payload = {
        "config": {
            "train_docs": int(train_docs),
            "seed": int(seed),
            "violation_tau": 0.0,
            "audit_policy": "random",
            "audit_fraction": 0.1,
            "feature_mode": "full",
            "leaf_query_rate": 1.0,
            "root_weight": 1.0,
            "schedule_consistency_weight": 0.0,
            "c3_audit_strategy": "uniform",
            "c3_include_root": True,
            "max_segments": 4,
        },
        "objective": {
            "local_law_weight": float(local_law_weight),
        },
        "training_geometry": {
            "mean_leaves": 4.0,
            "mean_internal_labels": 2.0,
            "root_queries_total": 100,
            "leaf_labels_total": 400,
            "internal_labels_total": 200,
            "total_queries_estimate": 700,
        },
        "metrics": {
            name: {
                "root_mae": base,
                "leaf_mae": base / 2.0,
                "merge_mae": base / 3.0,
                "merge_violation_rate": 0.05,
                "schedule_spread_mean": base / 4.0,
            }
            for name, base in {
                "learned": 0.4,
                "exact": 0.1,
                "undersupported": 0.5,
                "flip_R1": 0.6,
                "flip_R2": 0.7,
            }.items()
        },
        "estimator_diagnostics": {
            "naive_bias": 0.1,
            "ipw_bias": 0.05,
            "dsl_bias": 0.02,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_inputs(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    for llw in (0.0, 0.5):
        for seed, train_docs in enumerate((100, 200)):
            out = root / f"llw_{llw:g}" / f"seed_{seed}.json"
            _write_markov_payload(out, train_docs=train_docs, seed=seed, local_law_weight=llw)
    return root


def test_markov_grid_plot_writes_multi_llw_manifest(tmp_path: Path) -> None:
    input_root = _build_inputs(tmp_path)
    output_figure = tmp_path / "grid.png"
    output_json = tmp_path / "grid.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/plot_markov_changepoint_ops_count_grid.py",
            "--input-glob",
            str(input_root / "**/*seed_*.json"),
            "--layout",
            "honesty",
            "--aggregate",
            "median",
            "--output-figure",
            str(output_figure),
            "--output-json",
            str(output_json),
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["mode"] == "multi_local_law_weight"
    assert report["local_law_weights"] == [0.0, 0.5]
    assert len(report["children"]) == 2
    for child in report["children"]:
        assert Path(child["figure"]).is_file()
        assert Path(child["report"]).is_file()


def test_markov_lines_plot_writes_multi_llw_manifest(tmp_path: Path) -> None:
    input_root = _build_inputs(tmp_path)
    output_figure = tmp_path / "lines.png"
    output_json = tmp_path / "lines.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/plot_markov_changepoint_ops_count_lines.py",
            "--input-glob",
            str(input_root / "**/*seed_*.json"),
            "--aggregate",
            "median",
            "--band",
            "none",
            "--output-figure",
            str(output_figure),
            "--output-json",
            str(output_json),
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["mode"] == "multi_local_law_weight"
    assert report["local_law_weights"] == [0.0, 0.5]
    assert len(report["children"]) == 2
    for child in report["children"]:
        assert Path(child["figure"]).is_file()
        assert Path(child["report"]).is_file()
