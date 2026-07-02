"""Tests for the supervision axis on the q-sentence ladder runner.

Covers (a) parsing/validation of the ``--supervision`` comma-list, (b) the
named-level -> low-level-knob override mapping (including strict FNO-vs-DSPy
scoping and back-compat pass-through of the ``default`` level), and (c) the
full leaf x supervision grid enumeration + per-cell labeling in the summary,
with the family build + alternating driver stubbed so no GPU/LLM job runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.run_manifesto_qsentence_dspy_ladder as R


# --------------------------------------------------------------------------- #
# Parsing / validation
# --------------------------------------------------------------------------- #
def test_parse_supervision_grid_defaults_when_absent():
    assert R.parse_supervision_grid(None) == ("default",)
    assert R.parse_supervision_grid("") == ("default",)


def test_parse_supervision_grid_named_levels_order_preserved_dedup():
    assert R.parse_supervision_grid("root,leaf,node,mix") == ("root", "leaf", "node", "mix")
    assert R.parse_supervision_grid("node,root,node") == ("node", "root")


def test_parse_supervision_grid_rejects_unknown():
    with pytest.raises(ValueError) as exc:
        R.parse_supervision_grid("root,bogus")
    assert "bogus" in str(exc.value)


def test_every_level_maps_only_known_arg_attributes():
    args = R.parse_args(["--family", "fno"])
    for name in R.SUPERVISION_LEVELS:
        for attr in R.SUPERVISION_LEVELS[name].overrides:
            assert hasattr(args, attr), f"{name} overrides unknown arg {attr!r}"


# --------------------------------------------------------------------------- #
# Level -> knob override mapping
# --------------------------------------------------------------------------- #
def test_default_level_is_exact_passthrough():
    args = R.parse_args(
        [
            "--family",
            "fno",
            "--fno-root-weight",
            "7.0",
            "--fno-leaf-weight",
            "0.25",
            "--fno-merge-weight",
            "0.75",
        ]
    )
    scoped = R._apply_supervision_level(args, "default")
    assert scoped.fno_root_weight == 7.0
    assert scoped.fno_leaf_weight == 0.25
    assert scoped.fno_merge_weight == 0.75
    # original untouched
    assert args.fno_root_weight == 7.0


@pytest.mark.parametrize(
    "level,expected",
    [
        ("root", (1.0, 0.0, 0.0)),
        ("leaf", (0.0, 1.0, 0.0)),
        ("node", (0.0, 1.0, 1.0)),
        ("mix", (3.0, 1.0, 1.0)),
    ],
)
def test_named_levels_override_weights(level, expected):
    args = R.parse_args(["--family", "fno"])
    scoped = R._apply_supervision_level(args, level)
    got = (scoped.fno_root_weight, scoped.fno_leaf_weight, scoped.fno_merge_weight)
    assert got == expected
    # original args are never mutated
    assert (args.fno_root_weight, args.fno_leaf_weight, args.fno_merge_weight) != expected or level == "default"


def test_dspy_rejects_non_default_supervision():
    args = R.parse_args(["--family", "dspy"])
    with pytest.raises(SystemExit):
        R._apply_supervision_level(args, "node")
    # default is allowed for dspy (identity)
    R._apply_supervision_level(args, "default")


# --------------------------------------------------------------------------- #
# Full grid enumeration (family + driver stubbed)
# --------------------------------------------------------------------------- #
class _FakeTree:
    pass


def _stub_grid(monkeypatch):
    """Stub out tree loading, family build, driver, and finetune export."""

    monkeypatch.setattr(R, "load_leafq_trees", lambda _dir, _leaf: [_FakeTree()])
    monkeypatch.setattr(R, "_build_fno_family", lambda args: object())
    monkeypatch.setattr(R, "_build_family", lambda args: object())
    monkeypatch.setattr(
        R, "split_trees_for_eval", lambda trees, **kw: (list(trees), list(trees))
    )
    monkeypatch.setattr(
        R, "export_manifesto_finetune_bundle_from_args", lambda **kw: None
    )
    monkeypatch.setattr(R, "run_alternating_family", lambda **kwargs: [])


def test_full_leaf_x_supervision_grid_enumerates(monkeypatch, tmp_path):
    _stub_grid(monkeypatch)

    rc = R.main(
        [
            "--family",
            "fno",
            "--leaf-qsentences",
            "1,2",
            "--supervision",
            "root,node",
            "--max-iterations",
            "0",
            "--output-dir",
            str(tmp_path),
            "--fg-grid-dir",
            str(tmp_path),  # unused (load stubbed)
        ]
    )
    assert rc == 0

    summary = json.loads((tmp_path / "grid_summary.json").read_text())
    assert summary["supervision_axis"] == ["root", "node"]
    assert summary["leaf_qsentences"] == [1, 2]

    # 2 leaf x 2 supervision = 4 cells, one per_row_path each, disambiguated by
    # the nested sup_<level> segment.
    assert sorted(summary["per_row_paths"]) == sorted(
        [
            "fno/leafq001/sup_root/iteration_history.json",
            "fno/leafq001/sup_node/iteration_history.json",
            "fno/leafq002/sup_root/iteration_history.json",
            "fno/leafq002/sup_node/iteration_history.json",
        ]
    )

    # Each cell's per-row payload records both axis fields and the concrete
    # weight overrides at a disambiguated on-disk path.
    seen_cells = set()
    for leaf in (1, 2):
        for sup, weights in (("root", (1.0, 0.0, 0.0)), ("node", (0.0, 1.0, 1.0))):
            row_path = (
                tmp_path
                / "fno"
                / R.leafq_label(leaf)
                / R.supervision_label(sup)
                / "iteration_history.json"
            )
            assert row_path.exists(), row_path
            payload = json.loads(row_path.read_text())
            assert payload["supervision"] == sup
            assert payload["leaf_qsentences"] == leaf
            sw = payload["supervision_weights"]
            assert (sw["fno_root_weight"], sw["fno_leaf_weight"], sw["fno_merge_weight"]) == weights
            seen_cells.add((leaf, sup))
    assert seen_cells == {(1, "root"), (1, "node"), (2, "root"), (2, "node")}


def test_default_supervision_preserves_legacy_row_path(monkeypatch, tmp_path):
    _stub_grid(monkeypatch)

    rc = R.main(
        [
            "--family",
            "fno",
            "--leaf-qsentences",
            "4",
            "--max-iterations",
            "0",
            "--output-dir",
            str(tmp_path),
            "--fg-grid-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    # No --supervision => single implicit 'default' cell at the LEGACY path
    # (no sup_* segment), proving strict back-compat.
    legacy = tmp_path / "fno" / R.leafq_label(4) / "iteration_history.json"
    assert legacy.exists()
    payload = json.loads(legacy.read_text())
    assert payload["supervision"] == "default"

    summary = json.loads((tmp_path / "grid_summary.json").read_text())
    assert summary["supervision_axis"] == ["default"]
    assert summary["per_row_paths"] == ["fno/leafq004/iteration_history.json"]
