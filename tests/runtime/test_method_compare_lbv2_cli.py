from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_method_compare_lbv2_dry_run_writes_manifest(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    method_root = tmp_path / "method_compare"
    profile_dir = method_root / "baseline_llm"
    profile_dir.mkdir(parents=True)
    (profile_dir / "final_stats.json").write_text(
        json.dumps({"profile": "baseline_llm"}), encoding="utf-8"
    )
    (method_root / "method_compare_manifest.json").write_text(
        json.dumps({"entries": [{"profile": "baseline_llm", "run_dir": str(profile_dir)}]}),
        encoding="utf-8",
    )
    fixture = tmp_path / "lb.jsonl"
    fixture.write_text(
        json.dumps(
            {
                "_id": "x",
                "domain": "law",
                "sub_domain": "contracts",
                "difficulty": "easy",
                "length": "short",
                "question": "Which option is named?",
                "choice_A": "Alpha",
                "choice_B": "Beta",
                "choice_C": "Gamma",
                "choice_D": "Delta",
                "answer": "A",
                "context": "Alpha is named.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "lbv2.yaml"
    config.write_text(
        "\n".join(
            [
                "benchmark:",
                "  name: longbench_v2",
                f"  dataset_path: {fixture}",
                "scorer:",
                "  endpoint: http://localhost:8000/v1",
                "  model: mock-model",
                "runtime_defaults:",
                "  verifier_enabled: false",
                "  repair_enabled: false",
                "embedder:",
                "  mock: true",
                "phases:",
                "  - phase_id: S0",
                "    tasks: [all]",
                "    lengths: [4096]",
                "    seeds: [0]",
                "    num_samples: 1",
                "    split: test",
                "    methods: [llm_direct_official]",
            ]
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "lbv2_out"

    subprocess.check_call(
        [
            sys.executable,
            "scripts/run_method_compare_lbv2.py",
            "--method-compare-dir",
            str(method_root),
            "--lbv2-config",
            str(config),
            "--output-root",
            str(output_root),
            "--profiles",
            "baseline_llm",
            "--include-raw-variants",
            "--dry-run",
        ],
        cwd=repo_root,
    )

    manifest = json.loads(
        (output_root / "method_compare_lbv2_manifest.json").read_text(encoding="utf-8")
    )
    methods = [entry["method"] for entry in manifest["entries"]]
    assert methods == ["baseline_llm_raw", "baseline_llm_trained"]
    assert (output_root / "configs" / "baseline_llm_raw.yaml").exists()
    assert (output_root / "experiment_manifest.json").exists()
    assert (output_root / "artifacts.json").exists()
    assert (output_root / "results.jsonl").exists()
