from __future__ import annotations

import json
from pathlib import Path

from src.tasks.manifesto.result_rows import (
    get_text_for_row,
    load_rows_by_dimension,
    load_rows_by_id,
    order_split_rows,
    row_manifesto_id,
    row_summary,
    row_target_score,
    row_teacher_score,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_manifesto_result_row_accessors_and_targets() -> None:
    row = {
        "doc_id": "doc_1",
        "root_summary": "summary text",
        "predictions": {"economic": 5.0},
        "benoit_expert_mean": 4.0,
    }
    assert row_manifesto_id(row) == "doc_1"
    assert row_summary(row) == "summary text"
    assert row_teacher_score(row, dimension="economic") == 5.0
    assert row_target_score(row, dimension="economic", target_source="teacher") == 5.0
    assert row_target_score(row, dimension="economic", target_source="expert") is not None


def test_order_split_rows_is_deterministic_and_text_lookup_prefers_row_text() -> None:
    rows = {
        f"doc_{idx}": {"manifesto_id": f"doc_{idx}", "text": f"text {idx}"}
        for idx in range(6)
    }
    first = order_split_rows(rows, train_n=2, val_n=1, test_n=1, seed=7)
    second = order_split_rows(rows, train_n=2, val_n=1, test_n=1, seed=7)
    assert first == second
    assert set(first) == {"train", "val", "test"}
    picked = next(iter(first["train"]))
    assert get_text_for_row(row=rows[picked], split_texts={}, dataset=None).startswith("text")


def test_manifesto_result_row_loaders(tmp_path: Path) -> None:
    rows = [
        {"manifesto_id": "a", "value": 1},
        {"manifesto_id": "b", "value": 2},
    ]
    path = tmp_path / "economic" / "per_manifesto.jsonl"
    _write_jsonl(path, rows)
    assert set(load_rows_by_id(path)) == {"a", "b"}
    loaded = load_rows_by_dimension(tmp_path, ["economic"])
    assert loaded["economic"]["a"]["value"] == 1
