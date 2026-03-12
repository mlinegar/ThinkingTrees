from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_prompt_batch_tuner_dry_run(tmp_path: Path) -> None:
    train_pairs = tmp_path / "summary_pairs_train.jsonl"
    eval_pairs = tmp_path / "summary_pairs_val.jsonl"
    output_dir = tmp_path / "prompt_batch_run"

    _write_jsonl(
        train_pairs,
        [
            {
                "id": "doc1_h1",
                "example_id": "doc1",
                "split": "train",
                "hop": 1,
                "input_text": "Tax policy should support working families.",
                "target_summary": "Supports working-family tax policy.",
                "source_rile_raw": 12.0,
            },
            {
                "id": "doc1_h2",
                "example_id": "doc1",
                "split": "train",
                "hop": 2,
                "input_text": "Supports working-family tax policy.",
                "target_summary": "Pro-working-family tax stance.",
                "source_rile_raw": 12.0,
            },
        ],
    )
    _write_jsonl(
        eval_pairs,
        [
            {
                "id": "doc2_h1",
                "example_id": "doc2",
                "split": "val",
                "hop": 1,
                "input_text": "Public investment and social welfare expansion are priorities.",
                "target_summary": "Prioritizes public investment and welfare expansion.",
                "source_rile_raw": -18.0,
            }
        ],
    )

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "tune_manifesto_summary_prompts_batch.py"),
        "--train-pairs",
        str(train_pairs),
        "--eval-pairs",
        str(eval_pairs),
        "--output-dir",
        str(output_dir),
        "--dry-run",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    manifest_path = output_dir / "prompt_batch_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "dry_run"
    assert manifest["counts"]["train_examples"] == 2
    assert manifest["counts"]["eval_examples"] == 1
    assert manifest["artifacts"]["train_dataset_snapshot"].endswith("train_dataset_snapshot.jsonl")
