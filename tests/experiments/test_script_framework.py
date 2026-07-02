from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.experiments.ladder_reporting import (
    summarize_ladder_grid,
    write_alternating_markdown_summary,
)
from src.experiments.metrics import pearson, rankdata, regression_metrics, spearman
from src.experiments.script_io import (
    JsonlCallCache,
    append_jsonl,
    read_json,
    read_jsonl,
    require_within_chars,
    stable_digest,
    stable_hash,
    write_json,
)
from src.experiments.script_parse import (
    coerce_scalar,
    mean,
    parse_csv,
    parse_float_grid,
    parse_float_list,
    parse_int_grid,
    parse_int_list,
    parse_str_list,
    parse_token_list,
    safe_float,
)
from src.experiments.tree_helpers import root_node, split_trees_for_eval, summary_coverage
from src.tree.labeled import LabeledNode, LabeledTree


def test_script_io_json_jsonl_hash_and_cache(tmp_path: Path) -> None:
    payload = {"b": 2, "a": [1]}
    json_path = write_json(tmp_path / "nested" / "payload.json", payload)
    assert read_json(json_path) == payload
    rows_path = append_jsonl(tmp_path / "rows.jsonl", {"x": 1})
    append_jsonl(rows_path, [{"x": 2}])
    assert read_jsonl(rows_path) == [{"x": 1}, {"x": 2}]
    assert stable_digest({"a": 1, "b": 2}) == stable_digest({"b": 2, "a": 1})
    assert stable_hash("abc") == stable_hash("abc")
    assert require_within_chars("abc", max_chars=3, label="ok") == "abc"
    with pytest.raises(RuntimeError):
        require_within_chars("abcd", max_chars=3, label="too_long")

    cache = JsonlCallCache(tmp_path / "calls.jsonl")
    first = cache.put("k", {"kind": "score", "value": 1})
    second = cache.put("k", {"kind": "score", "value": 2})
    assert first["value"] == second["value"] == 1
    assert cache.get("k")["kind"] == "score"
    assert cache.stats()["entries"] == 1


def test_script_parse_and_metrics_helpers() -> None:
    assert parse_csv("a;b,c", allowed=("a", "b", "c")) == ("a", "b", "c")
    assert parse_int_grid("1,2;3") == (1, 2, 3)
    assert parse_float_grid("1.5,2") == (1.5, 2.0)
    assert parse_token_list("a b,c;d") == ["a", "b", "c", "d"]
    assert parse_int_list("1 2,3", default=(9,)) == [1, 2, 3]
    assert parse_int_list("", default=(9,)) == [9]
    assert parse_float_list(None, default=(0.5,)) == [0.5]
    assert parse_str_list("a;b", default=(), separators=",") == ["a;b"]
    assert safe_float("nan") is None
    assert coerce_scalar("true") is True
    assert coerce_scalar("4") == 4
    assert mean([1, "2", "bad"]) == pytest.approx(1.5)

    assert rankdata([10, 10, 30]) == [1.5, 1.5, 3.0]
    assert pearson([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    metrics = regression_metrics(
        [{"pred": 1, "truth": 2}, {"pred": 3, "truth": 3}],
        pred_key="pred",
        truth_key="truth",
    )
    assert metrics["n"] == 2
    assert metrics["mae"] == pytest.approx(0.5)


def test_ladder_reporting_preserves_requested_fields(tmp_path: Path) -> None:
    rows = summarize_ladder_grid(
        [
            {
                "family": "dspy",
                "axis_kind": "leaf_size_tokens",
                "axis_value": 16,
                "leaf_size_tokens": 16,
                "iterations": [
                    {
                        "iteration": 0,
                        "stage_name": "fg",
                        "trained": "none",
                        "split_metrics": {"test": {"n": 2, "internal_f_pearson": 0.5}},
                    }
                ],
            }
        ],
        eval_split="test",
        row_fields=("family", "axis_kind", "axis_value", "leaf_size_tokens"),
        metric_fields=("internal_f_pearson",),
    )
    assert rows == [
        {
            "family": "dspy",
            "axis_kind": "leaf_size_tokens",
            "axis_value": 16,
            "leaf_size_tokens": 16,
            "iteration": 0,
            "stage_name": "fg",
            "stage_label": "fg",
            "f_degree": None,
            "g_degree": None,
            "trained": "none",
            "n_eval": 2,
            "internal_f_pearson": 0.5,
        }
    ]
    md = tmp_path / "summary.md"
    write_alternating_markdown_summary(rows, md, eval_split="test")
    assert "Alternating ladder grid summary" in md.read_text(encoding="utf-8")


def test_tree_helpers_root_split_and_summary_coverage() -> None:
    tree = LabeledTree(doc_id="doc", document_text="a b", document_score=1.0, metadata={"split": "train"})
    tree.add_node(LabeledNode(node_id="l", doc_id="doc", level=0, text="a", score=1.0))
    tree.add_node(
        LabeledNode(
            node_id="r",
            doc_id="doc",
            level=1,
            text="a b",
            score=1.0,
            left_child_id="l",
            right_child_id="l",
            metadata={"teacher_summary": "root summary"},
        )
    )
    assert root_node(tree).node_id == "r"
    train, eval_trees = split_trees_for_eval([tree], eval_split="test", train_split="train")
    assert train == [tree]
    assert eval_trees == []
    coverage = summary_coverage([tree])
    assert coverage["roots_with_summary"] == 1
    assert coverage["partial_artifact"] is True
