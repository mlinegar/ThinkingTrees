from __future__ import annotations

import importlib
import json
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def test_build_sft_example_formats_prompt_and_target() -> None:
    cli = importlib.import_module("scripts.train_manifesto_summary_sft")

    row = {
        "id": "pair_1",
        "example_id": "ex_1",
        "split": "train",
        "hop": 2,
        "input_text": "Input manifesto text.",
        "target_summary": "Target summary text.",
        "source_rile_raw": -12.5,
    }

    built = cli.build_sft_example(row)
    assert built["hop"] == 2
    assert built["target_summary"] == "Target summary text."
    assert "Target directional score to preserve" not in built["prompt"]
    assert "Do NOT mention any numeric score" in built["prompt"]
    assert "Resummary hop: 2" in built["prompt"]
    assert built["text"].endswith("Target summary text.")

    built_conditional = cli.build_sft_example(row, include_score_conditioning=True)
    assert "Target directional score to preserve: -12.50" in built_conditional["prompt"]


def test_train_manifesto_summary_sft_cli_dry_run(tmp_path: Path) -> None:
    cli = importlib.import_module("scripts.train_manifesto_summary_sft")

    train_pairs = tmp_path / "summary_pairs_train.jsonl"
    eval_pairs = tmp_path / "summary_pairs_val.jsonl"

    _write_jsonl(
        train_pairs,
        [
            {
                "id": "train_1",
                "example_id": "doc_1",
                "split": "train",
                "hop": 1,
                "input_text": "Policy text A",
                "target_summary": "Summary A",
                "source_rile_raw": 10.0,
            },
            {
                "id": "train_2",
                "example_id": "doc_2",
                "split": "train",
                "hop": 2,
                "input_text": "Policy text B",
                "target_summary": "Summary B",
                "source_rile_raw": -5.0,
            },
        ],
    )
    _write_jsonl(
        eval_pairs,
        [
            {
                "id": "val_1",
                "example_id": "doc_3",
                "split": "val",
                "hop": 1,
                "input_text": "Policy text C",
                "target_summary": "Summary C",
                "source_rile_raw": 0.0,
            }
        ],
    )

    output_dir = tmp_path / "sft_out"
    rc = cli.main(
        [
            "--train-pairs",
            str(train_pairs),
            "--eval-pairs",
            str(eval_pairs),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )
    assert rc == 0

    manifest = json.loads((output_dir / "sft_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "dry_run"
    assert manifest["counts"]["train_examples"] == 2
    assert manifest["counts"]["eval_examples"] == 1

    assert _count_jsonl(output_dir / "train_dataset_snapshot.jsonl") == 2
    assert _count_jsonl(output_dir / "eval_dataset_snapshot.jsonl") == 1
