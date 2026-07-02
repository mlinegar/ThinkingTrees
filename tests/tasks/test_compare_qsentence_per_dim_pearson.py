from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.compare_qsentence_per_dim_pearson as cmp
from src.experiments.metrics import pearson


def _write_eval(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _rows(dim, pairs):
    return [
        {"dimension": dim, "doc_id": f"d{i}", "prediction": p, "teacher_score": t}
        for i, (p, t) in enumerate(pairs)
    ]


def test_pearson_basic_and_degenerate():
    assert pearson([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert pearson([1, 1, 1], [1, 2, 3]) is None  # zero variance
    assert pearson([1.0], [1.0]) is None  # < 2 points


def test_resolve_run_dir(tmp_path: Path):
    eval_path = (
        tmp_path / "run" / "dspy" / "leafq008" / "prediction_records" / "iter_02_post_eval.jsonl"
    )
    _write_eval(eval_path, _rows("rile", [(0.1, 0.2)]))
    resolved = cmp._resolve_eval_path(str(tmp_path / "run"), leaf=8, it=2)
    assert resolved == eval_path
    # direct file path also accepted
    assert cmp._resolve_eval_path(str(eval_path), leaf=8, it=2) == eval_path


def test_resolve_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        cmp._resolve_eval_path(str(tmp_path / "nope"), leaf=8, it=2)


def test_per_dim_pearson_table_matches_manual(tmp_path: Path):
    rows = _rows("rile", [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)])
    pd = cmp._per_dim_pred_truth(rows)
    tbl = cmp._pearson_table(pd)
    assert tbl["rile"] == pytest.approx(1.0)  # perfectly correlated
    # dims with no rows are None
    assert tbl["domain_1"] is None


def test_control_vs_self_is_zero_delta(tmp_path: Path, capsys):
    eval_path = (
        tmp_path / "run" / "dspy" / "leafq008" / "prediction_records" / "iter_02_post_eval.jsonl"
    )
    rows = _rows("rile", [(0.1, 0.5), (0.3, 0.2), (0.9, 0.7)]) + _rows(
        "domain_1", [(0.2, 0.1), (0.4, 0.6), (0.1, 0.3)]
    )
    _write_eval(eval_path, rows)
    rc = cmp.main(
        [
            "--control", str(tmp_path / "run"),
            "--test", str(tmp_path / "run"),
            "--leaf", "8", "--iter", "2",
            "--labels", "control,selfcheck",
            "--json-out", str(tmp_path / "out.json"),
        ]
    )
    assert rc == 0
    payload = json.loads((tmp_path / "out.json").read_text())
    assert payload["tests"][0]["mean_delta_vs_control"] == pytest.approx(0.0)
    assert payload["tests"][0]["dims_improved"] == 0
