from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_runtime_eval_cli_longbench_fixture_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fixture = tmp_path / "longbench_fixture.jsonl"
    fixture.write_text(
        json.dumps(
            {
                "_id": "cli-1",
                "domain": "law",
                "sub_domain": "contracts",
                "difficulty": "easy",
                "length": "short",
                "question": "Which option names the delivery party?",
                "choice_A": "Alpha",
                "choice_B": "Beta",
                "choice_C": "Gamma",
                "choice_D": "Delta",
                "answer": "C",
                "context": "Gamma is named as the delivery party in the final clause.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "runtime_longbench.yaml"
    output_root = tmp_path / "outputs"
    experiment_id = "lb_smoke"
    config_path.write_text(
        "\n".join(
            [
                "benchmark:",
                "  name: longbench_v2",
                f"  dataset_path: {fixture}",
                "methods: [full_context, retrieval, summary_tree, state_tree, neural_operator]",
                "scorer:",
                "  model: mock-model",
                "  endpoint: http://localhost:8000/v1",
                "embedder:",
                "  mock: true",
                "  mock_dim: 16",
                "state_model:",
                "  kind: neural_operator",
                "  checkpoint: ''",
                "oracle:",
                "  kind: benchmark_labels",
                "runtime_defaults:",
                "  cap_tokens: 512",
                "  safety_tokens: 16",
                "  max_output_tokens: 16",
                "  chunk_tokens: 12",
                "  overlap_tokens: 0",
                "  leaf_memory_tokens: 16",
                "  merge_memory_tokens: 16",
                "  retrieval_top_k: 1",
                "  retrieval_chunk_tokens: 12",
                "  retrieval_overlap_tokens: 0",
                "  verifier_enabled: false",
                "  repair_enabled: false",
                "phases:",
                "  - phase_id: S0",
                "    tasks: [all]",
                "    lengths: [4096]",
                "    seeds: [0]",
                "    num_samples: 1",
                "    split: test",
            ]
        ),
        encoding="utf-8",
    )

    subprocess.check_call(
        [
            sys.executable,
            "scripts/run_runtime_eval.py",
            "init",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_root),
            "--experiment-id",
            experiment_id,
        ],
        cwd=repo_root,
    )
    experiment_dir = output_root / experiment_id
    subprocess.check_call(
        [
            sys.executable,
            "scripts/run_runtime_eval.py",
            "run",
            "--experiment-dir",
            str(experiment_dir),
            "--mock-llm",
            "--max-problems",
            "1",
        ],
        cwd=repo_root,
    )
    subprocess.check_call(
        [sys.executable, "scripts/run_runtime_eval.py", "aggregate", "--experiment-dir", str(experiment_dir)],
        cwd=repo_root,
    )

    assert (experiment_dir / "metrics.json").exists()
    assert (experiment_dir / "predictions.jsonl").exists()
    assert (experiment_dir / "calls.jsonl").exists()
    assert (experiment_dir / "steps.jsonl").exists()
    metrics = json.loads((experiment_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_predictions"] == 5
    assert metrics["n_surface_calls"] > 0
    assert metrics["by_domain"]["law"] >= 0.0
    calls_text = (experiment_dir / "calls.jsonl").read_text(encoding="utf-8")
    assert '"role": "scorer"' in calls_text
    assert "state_operator" not in calls_text
