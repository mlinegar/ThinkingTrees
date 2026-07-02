from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ctreepo.contracts import (
    SOURCE_KIND_EXTERNAL_STATE,
    SOURCE_KIND_RAW_INPUT,
    legacy_tree_bundle_kind_for_source_kind,
    legacy_tree_text_source_for_source_kind,
    source_kind_for_legacy_tree_text_source,
    source_kind_for_tree_bundle_kind,
)
from src.ctreepo.manifesto_qsentence_runner import (
    format_leaf_artifact_template,
    leafq_dir,
    leafq_label,
    load_leafq_trees,
    resolve_leaf_artifact,
)
from src.tasks.manifesto.script_utils import (
    append_jsonl,
    mean,
    parse_compact_dimensions,
    parse_csv,
    parse_int_grid,
    read_json,
    read_jsonl,
    safe_float,
    safe_int,
    write_json,
)
from src.tasks.manifesto.span_targets import COMPACT_TARGET_DIMENSIONS


def test_manifesto_script_json_helpers_round_trip(tmp_path: Path) -> None:
    payload = {"b": 2, "a": [1, 2]}
    path = write_json(tmp_path / "nested" / "payload.json", payload)
    assert read_json(path) == payload
    assert path.read_text(encoding="utf-8").endswith("\n")

    rows_path = append_jsonl(tmp_path / "rows.jsonl", [{"a": 1}], append=False)
    append_jsonl(rows_path, [{"b": 2}])
    assert read_jsonl(rows_path) == [{"a": 1}, {"b": 2}]
    assert [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()] == [
        {"a": 1},
        {"b": 2},
    ]


def test_manifesto_script_numeric_helpers_ignore_invalid_values() -> None:
    assert safe_float("3.5") == 3.5
    assert safe_float("nan") is None
    assert safe_int("7") == 7
    assert safe_int("bad", default=4) == 4
    assert mean([1, "2", None, "nan", "bad"]) == pytest.approx(1.5)


def test_parse_int_grid_accepts_common_forms_and_rejects_empty_or_nonpositive() -> None:
    assert parse_int_grid("1,2;3") == (1, 2, 3)
    assert parse_int_grid(["4", 5], name="leaf grid") == (4, 5)

    for raw in ("", "0", "1,-2", "a"):
        with pytest.raises(ValueError):
            parse_int_grid(raw, name="leaf grid")


def test_parse_csv_allowed_values() -> None:
    assert parse_csv("economic;social,eu", allowed=("economic", "social", "eu")) == (
        "economic",
        "social",
        "eu",
    )
    with pytest.raises(ValueError):
        parse_csv("economic,unknown", allowed=("economic",))


def test_parse_compact_dimensions_preserves_canonical_order() -> None:
    assert parse_compact_dimensions("") == tuple(COMPACT_TARGET_DIMENSIONS)
    assert parse_compact_dimensions("all") == tuple(COMPACT_TARGET_DIMENSIONS)
    parsed = parse_compact_dimensions("domain_3,rile,domain_1")
    assert parsed == tuple(
        dim for dim in COMPACT_TARGET_DIMENSIONS if dim in {"rile", "domain_1", "domain_3"}
    )
    with pytest.raises(ValueError):
        parse_compact_dimensions("domain_1,not_a_dimension")


def test_qsentence_leaf_paths_and_artifact_resolution(tmp_path: Path) -> None:
    assert leafq_label(8) == "leafq008"
    assert leafq_dir(tmp_path, 16) == tmp_path / "leafq016"
    assert load_leafq_trees(tmp_path, 16) is None
    assert (
        format_leaf_artifact_template(
            "runs/{leafq}/{leaf}/{leaf_qsentences}/{row_label}.json",
            8,
        )
        == "runs/leafq008/8/8/leafq008.json"
    )
    assert resolve_leaf_artifact(None, "tmpl/{leafq}", "default.json", "f", 8) == "tmpl/leafq008"
    assert resolve_leaf_artifact(" static.json ", None, "default.json", "g", 8) == "static.json"
    assert resolve_leaf_artifact("", "", "default.json", "g", 8) == "default.json"
    with pytest.raises(ValueError):
        resolve_leaf_artifact("static.json", "tmpl/{leafq}", "default.json", "f", 8)


def test_source_kind_legacy_mapping_round_trips() -> None:
    for source_kind in (SOURCE_KIND_RAW_INPUT, SOURCE_KIND_EXTERNAL_STATE):
        legacy_kind = legacy_tree_bundle_kind_for_source_kind(source_kind)
        legacy_text_source = legacy_tree_text_source_for_source_kind(source_kind)
        assert source_kind_for_tree_bundle_kind(legacy_kind) == source_kind
        assert source_kind_for_legacy_tree_text_source(legacy_text_source) == source_kind

    assert source_kind_for_legacy_tree_text_source("") == SOURCE_KIND_RAW_INPUT
    with pytest.raises(ValueError):
        source_kind_for_tree_bundle_kind("unknown_bundle")
    with pytest.raises(ValueError):
        source_kind_for_legacy_tree_text_source("unknown_text_source")
