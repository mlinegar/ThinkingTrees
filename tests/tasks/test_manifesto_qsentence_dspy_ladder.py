from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

import pytest

from src.ctreepo.distillation import load_labeled_trees
from src.ctreepo.dspy_family import DSPyFamily, _strict_gepa_evaluate_errors
from src.ctreepo.manifesto_qsentence_dspy_family import (
    ManifestoQSentenceDSPyFamily,
    ManifestoQSentenceDSPyFamilyConfig,
)
from src.tasks.manifesto.span_annotations import (
    ManifestoQSentence,
    reconstruct_manifesto,
)
from src.tasks.manifesto.span_targets import (
    COMPACT_TARGET_DIMENSIONS,
    aggregate_cmp_codes,
    normalize_cmp_code,
    parse_compact_scores_json,
)


def test_cmp_code_normalization_and_aggregation() -> None:
    assert normalize_cmp_code("605.1") == "605"
    assert normalize_cmp_code(6051) == "605"
    assert normalize_cmp_code("2021") == "202"
    assert normalize_cmp_code("H") == "H"
    assert normalize_cmp_code("000") == "000"

    target = aggregate_cmp_codes(["104", "202", "H", "000"])
    assert target["total_items"] == 4
    assert target["total_non_header"] == 3
    assert target["right_count"] == 1
    assert target["left_count"] == 1
    assert target["rile_raw"] == 0.0
    assert target["compact"]["rile"] == 0.5
    assert target["domain_counts"]["domain_1"] == 1
    assert target["domain_counts"]["domain_2"] == 1


def test_reconstruct_manifesto_preserves_qsentence_spans() -> None:
    rows = [
        ManifestoQSentence("11110_202001", 1, "Jobs now.", "104", "104"),
        ManifestoQSentence("11110_202001", 2, "Public services.", "202", "202"),
    ]
    reconstructed = reconstruct_manifesto("11110_202001", rows)
    assert reconstructed.text == "Jobs now.\nPublic services."
    first, second = reconstructed.qsentences
    assert reconstructed.text[first.char_start:first.char_end] == "Jobs now."
    assert reconstructed.text[second.char_start:second.char_end] == "Public services."


def test_parse_compact_scores_json_accepts_escaped_json_payload() -> None:
    escaped = r'{\"cmp_state\": {\"compact_targets\": {\"rile\": 0.5, \"domain_1\": 0.25}}}'

    parsed = parse_compact_scores_json(escaped)

    assert parsed["rile"] == 0.5
    assert parsed["domain_1"] == 0.25


def test_parse_compact_scores_json_accepts_flat_dotted_container_key() -> None:
    # DiffusionGemma frequently flattens the nested path into one dotted key.
    flat = (
        '{"cmp_state.compact_targets": {"rile": 0.5, "domain_1": 0, "domain_3": 1},'
        ' "total_non_header": 8}'
    )
    parsed = parse_compact_scores_json(flat)
    assert parsed["rile"] == 0.5
    assert parsed["domain_1"] == 0.0
    assert parsed["domain_3"] == 1.0


def test_parse_compact_scores_json_does_not_interpret_offschema_states() -> None:
    # The parser does LOSSLESS STRUCTURAL normalization only. Non-canonical
    # model states (RILE as a word, salience map, list-of-policy-objects, an
    # abstain flag) require INTERPRETATION, which is the f LLM readout's job,
    # not the parser's. The parser returns {} and the caller defers to f.
    # (No hard-coded word->number / domain-guess / abstain-keyword heuristics.)
    case_a = (
        '{"RILE": "pos", "salience": {"1": 0.8, "2": 0.0}, "policy": "integrity"}'
    )
    assert parse_compact_scores_json(case_a) == {}
    case_b = '{"policy": [{"domain": 1, "salience": 0.4}]}'
    assert parse_compact_scores_json(case_b) == {}
    assert parse_compact_scores_json('{"not_relevant": true}') == {}


def test_parse_compact_scores_json_canonical_and_dotted_still_lossless() -> None:
    # Canonical, dotted-container, bare compact_targets, and plain dim maps are
    # the same data restructured -> lossless, still parsed.
    assert parse_compact_scores_json(
        '{"cmp_state": {"compact_targets": {"rile": 0.5, "domain_3": 1}}}'
    ) == {"rile": 0.5, "domain_3": 1.0}
    assert parse_compact_scores_json(
        '{"cmp_state.compact_targets": {"rile": 0.5, "domain_1": 0}}'
    ) == {"rile": 0.5, "domain_1": 0.0}
    assert parse_compact_scores_json('{"rile": 0.3, "domain_2": 0.4}') == {
        "rile": 0.3, "domain_2": 0.4,
    }


def test_parse_compact_scores_json_garbage_returns_empty() -> None:
    assert parse_compact_scores_json("totally not json") == {}
    assert parse_compact_scores_json("") == {}


def test_parse_compact_scores_json_accepts_fenced_json_with_trailing_note() -> None:
    raw = """```json
    {
      "cmp_state": {
        "compact_targets": {
          "rile": 0.8,
          "domain_1": 0.0,
          "domain_2": 0.0,
          "domain_3": 0.0,
          "domain_4": 0.0,
          "domain_5": 0.0,
          "domain_6": 0.0,
          "domain_7": 0.0
        }
      }
    }
    ```
    Note: extra prose with {non-json braces} should not change parsing.
    """

    parsed = parse_compact_scores_json(raw)

    assert parsed["rile"] == 0.8
    assert set(parsed) == set(COMPACT_TARGET_DIMENSIONS)


def test_qsentence_config_defaults_to_strict_optimizer_errors() -> None:
    config = ManifestoQSentenceDSPyFamilyConfig(
        leaf_size_tokens=128,
        lm_context_window_tokens=4096,
        max_completion_tokens=256,
        prompt_template_overhead_tokens=512,
    )

    assert config.strict_optimizer_errors is True
    assert config.strict_optimizer_max_errors == 1


def test_strict_gepa_evaluate_errors_caps_internal_dspy_tolerance() -> None:
    from dspy.teleprompt.gepa import gepa_utils

    original = gepa_utils.Evaluate
    with _strict_gepa_evaluate_errors(max_errors=1):
        evaluator = gepa_utils.Evaluate(
            devset=[],
            metric=lambda gold, pred: 0.0,
            max_errors=500,
            provide_traceback=None,
        )
        assert evaluator.max_errors == 1
        assert evaluator.provide_traceback is True

    assert gepa_utils.Evaluate is original



def _write_synthetic_qsentence_csv(path: Path) -> None:
    rows = [
        {
            "text": "Cut taxes for workers.",
            "cmp_code": "104",
            "eu_code": "",
            "pos": 1,
            "manifesto_id": "11110_202001",
            "party": 11110,
            "date": 202001,
            "language": "english",
            "annotations": True,
            "translation_en": False,
        },
        {
            "text": "Protect public healthcare.",
            "cmp_code": "202",
            "eu_code": "",
            "pos": 2,
            "manifesto_id": "11110_202001",
            "party": 11110,
            "date": 202001,
            "language": "english",
            "annotations": True,
            "translation_en": False,
        },
        {
            "text": "Support business innovation.",
            "cmp_code": "305",
            "eu_code": "",
            "pos": 3,
            "manifesto_id": "11110_202001",
            "party": 11110,
            "date": 202001,
            "language": "english",
            "annotations": True,
            "translation_en": False,
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_synthetic_mpds_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["party", "date", "rile"])
        writer.writeheader()
        writer.writerow({"party": 11110, "date": 202001, "rile": 33.333333})


def _qs_family_for_reward_tests() -> ManifestoQSentenceDSPyFamily:
    return ManifestoQSentenceDSPyFamily(
        config=ManifestoQSentenceDSPyFamilyConfig(
            leaf_size_tokens=128,
            lm_context_window_tokens=4096,
            max_completion_tokens=256,
            prompt_template_overhead_tokens=512,
        )
    )



def test_qsentence_g_inference_retries_invalid_compact_state(monkeypatch) -> None:
    family = ManifestoQSentenceDSPyFamily(
        config=ManifestoQSentenceDSPyFamilyConfig(
            leaf_size_tokens=128,
            lm_context_window_tokens=4096,
            max_completion_tokens=256,
            prompt_template_overhead_tokens=512,
            target_dimensions=("rile", "domain_1"),
            g_inference_retries=1,
            g_require_canonical_state=True,
        )
    )
    valid_state = json.dumps(
        {"cmp_state": {"compact_targets": {"rile": 0.5, "domain_1": 0.25}}},
        sort_keys=True,
    )
    calls = []

    def fake_apply_g(self, g_program, *, prompt: str) -> str:
        calls.append(prompt)
        return "not a compact state" if len(calls) == 1 else valid_state

    monkeypatch.setattr(DSPyFamily, "_apply_g", fake_apply_g)

    assert family._apply_g(object(), prompt="merge prompt") == valid_state
    assert calls == ["merge prompt", "merge prompt"]


def test_qsentence_g_inference_fail_fast_rejects_first_invalid_state(monkeypatch) -> None:
    family = ManifestoQSentenceDSPyFamily(
        config=ManifestoQSentenceDSPyFamilyConfig(
            leaf_size_tokens=128,
            lm_context_window_tokens=4096,
            max_completion_tokens=256,
            prompt_template_overhead_tokens=512,
            target_dimensions=("rile", "domain_1"),
            g_inference_retries=1,
            fail_fast_on_invalid_g_state=True,
            g_require_canonical_state=True,
        )
    )
    calls = []

    def fake_apply_g(self, g_program, *, prompt: str) -> str:
        calls.append(prompt)
        return "not a compact state"

    monkeypatch.setattr(DSPyFamily, "_apply_g", fake_apply_g)

    with pytest.raises(RuntimeError, match="fail-fast enabled"):
        family._apply_g(object(), prompt="merge prompt")
    assert calls == ["merge prompt"]


def test_qsentence_g_inference_accepts_noncanonical_state_by_default(monkeypatch) -> None:
    # Default (g_require_canonical_state=False): f is an LLM readout, so a
    # NON-EMPTY g output is accepted as-is and NOT retried for failing the
    # strict compact schema. f interprets it downstream.
    family = ManifestoQSentenceDSPyFamily(
        config=ManifestoQSentenceDSPyFamilyConfig(
            leaf_size_tokens=128,
            lm_context_window_tokens=4096,
            max_completion_tokens=256,
            prompt_template_overhead_tokens=512,
            target_dimensions=("rile", "domain_1"),
            g_inference_retries=2,
        )
    )
    offschema = '{"RILE": "pos", "salience": {"1": 0.8}}'
    calls = []

    def fake_apply_g(self, g_program, *, prompt: str) -> str:
        calls.append(prompt)
        return offschema

    monkeypatch.setattr(DSPyFamily, "_apply_g", fake_apply_g)

    # Accepted on the first attempt, no retries, returned verbatim.
    assert family._apply_g(object(), prompt="merge prompt") == offschema
    assert calls == ["merge prompt"]


def test_qsentence_g_inference_still_retries_empty_output(monkeypatch) -> None:
    # Even in flexible mode, genuinely EMPTY output is a real failure -> retried.
    family = ManifestoQSentenceDSPyFamily(
        config=ManifestoQSentenceDSPyFamilyConfig(
            leaf_size_tokens=128,
            lm_context_window_tokens=4096,
            max_completion_tokens=256,
            prompt_template_overhead_tokens=512,
            target_dimensions=("rile", "domain_1"),
            g_inference_retries=1,
        )
    )
    good = '{"anything non-empty": true}'
    calls = []

    def fake_apply_g(self, g_program, *, prompt: str) -> str:
        calls.append(prompt)
        return "" if len(calls) == 1 else good

    monkeypatch.setattr(DSPyFamily, "_apply_g", fake_apply_g)
    assert family._apply_g(object(), prompt="merge prompt") == good
    assert calls == ["merge prompt", "merge prompt"]


def test_qsentence_g_inference_requires_all_active_dimensions(monkeypatch) -> None:
    family = ManifestoQSentenceDSPyFamily(
        config=ManifestoQSentenceDSPyFamilyConfig(
            leaf_size_tokens=128,
            lm_context_window_tokens=4096,
            max_completion_tokens=256,
            prompt_template_overhead_tokens=512,
            target_dimensions=("rile", "domain_1"),
            g_inference_retries=0,
            g_require_canonical_state=True,
        )
    )
    partial_state = json.dumps({"cmp_state": {"compact_targets": {"rile": 0.5}}})

    def fake_apply_g(self, g_program, *, prompt: str) -> str:
        return partial_state

    monkeypatch.setattr(DSPyFamily, "_apply_g", fake_apply_g)

    with pytest.raises(RuntimeError, match="domain_1"):
        family._apply_g(object(), prompt="merge prompt")


def test_qsentence_g_candidate_reward_requires_parseable_compact_targets() -> None:
    family = _qs_family_for_reward_tests()
    target = {
        "rile": 0.5,
        "domain_1": 0.125,
        "domain_2": 0.125,
        "domain_3": 0.125,
        "domain_4": 0.125,
        "domain_5": 0.125,
        "domain_6": 0.125,
        "domain_7": 0.125,
    }

    family._apply_f_scores = lambda f_program, *, response: dict(target)  # type: ignore[method-assign]

    exact_state = json.dumps({"cmp_state": {"compact_targets": target}})
    assert family._score_g_candidate_state(
        summary=exact_state,
        target=target,
        f_program=object(),
    ) == 1.0

    prose_only_reward = family._score_g_candidate_state(
        summary="This looks like a balanced manifesto.",
        target=target,
        f_program=object(),
    )
    assert prose_only_reward == family._g_reward_weights()[1]


def test_qsentence_g_candidate_reward_penalizes_partial_compact_targets() -> None:
    family = _qs_family_for_reward_tests()
    target = {
        "rile": 0.5,
        "domain_1": 0.125,
        "domain_2": 0.125,
        "domain_3": 0.125,
        "domain_4": 0.125,
        "domain_5": 0.125,
        "domain_6": 0.125,
        "domain_7": 0.125,
    }
    family._apply_f_scores = lambda f_program, *, response: dict(target)  # type: ignore[method-assign]

    partial_state = json.dumps({"cmp_state": {"compact_targets": {"rile": 0.5}}})
    partial_reward = family._score_g_candidate_state(
        summary=partial_state,
        target=target,
        f_program=object(),
    )

    assert family._g_reward_weights() == (0.75, 0.25)
    assert partial_reward == 0.75 * (1.0 / 8.0) + 0.25
    assert partial_reward < 0.5


def test_qsentence_g_all_laws_reward_blend() -> None:
    """C1/C3a law reward terms blend with the C2 base, stay in [0,1], and reward
    f-readout agreement between g's state and the raw span (C1) / child concat (C3a)."""
    target = {f"domain_{i}": 0.125 for i in range(1, 8)}
    target["rile"] = 0.5
    family = ManifestoQSentenceDSPyFamily(
        config=ManifestoQSentenceDSPyFamilyConfig(
            leaf_size_tokens=128,
            lm_context_window_tokens=4096,
            max_completion_tokens=256,
            prompt_template_overhead_tokens=512,
            g_law_c1_reward_weight=1.0,
            g_law_c3a_reward_weight=1.0,
        )
    )
    exact_state = json.dumps({"cmp_state": {"compact_targets": target}})

    # f reads EVERY input as the target -> perfect C2 + perfect C1/C3a agreement.
    family._apply_f_scores = lambda f_program, *, response: dict(target)  # type: ignore[method-assign]
    perfect = family._score_g_candidate_state(
        summary=exact_state,
        target=target,
        f_program=object(),
        law_context={"c1_raw_text": "raw span", "c3a_concat": ""},
    )
    assert perfect == pytest.approx(1.0)  # C2=1, C1=1 -> blended (1+1)/(1+1)=1

    # f disagrees on the raw span (C1) -> blended reward drops below pure C2.
    def split_f(f_program, *, response):
        if response == "raw span":
            return {k: (0.0 if k == "rile" else v) for k, v in target.items()}
        return dict(target)

    family._apply_f_scores = split_f  # type: ignore[method-assign]
    degraded = family._score_g_candidate_state(
        summary=exact_state,
        target=target,
        f_program=object(),
        law_context={"c1_raw_text": "raw span", "c3a_concat": ""},
    )
    assert 0.0 <= degraded < 1.0  # C1 disagreement pulls the blend down
    # weight redistributes when no law context is present -> pure C2 (=1.0)
    no_ctx = family._score_g_candidate_state(
        summary=exact_state, target=target, f_program=object(), law_context={}
    )
    assert no_ctx == pytest.approx(1.0)


def test_qsentence_grid_builder_exact_leaf_and_parent_targets(tmp_path: Path) -> None:
    from scripts import build_manifesto_qsentence_dspy_labeled_grid as cli

    corpus_csv = tmp_path / "manifesto_corpus_df.csv"
    mpds_csv = tmp_path / "manifesto_maindataset.csv"
    out = tmp_path / "grid"
    _write_synthetic_qsentence_csv(corpus_csv)
    _write_synthetic_mpds_csv(mpds_csv)

    rc = cli.main(
        [
            "--corpus-csv",
            str(corpus_csv),
            "--mpds-csv",
            str(mpds_csv),
            "--leaf-qsentences",
            "1",
            "--train-n",
            "1",
            "--val-n",
            "0",
            "--test-n",
            "0",
            "--output-dir",
            str(out),
        ]
    )
    assert rc == 0

    trees = load_labeled_trees(out / "leafq001" / "labeled_trees.jsonl")
    assert len(trees) == 1
    tree = trees[0]
    leaves = tree.get_leaves()
    assert len(leaves) == 3
    assert all((leaf.metadata or {})["total_qsentences"] == 1 for leaf in leaves)

    root = tree.get_node(tree.levels[-1][-1])
    assert root is not None
    leaf_counts: Counter[str] = Counter()
    for leaf in leaves:
        leaf_counts.update(dict((leaf.metadata or {}).get("cmp_counts") or {}))
    assert dict(sorted(leaf_counts.items())) == dict((root.metadata or {}).get("cmp_counts") or {})
    assert (root.metadata or {})["total_qsentences"] == 3
    assert abs(float((root.metadata or {})["rile_norm"]) - (2.0 / 3.0)) < 1e-6
    assert abs(float((tree.metadata or {})["mpds_rile_norm"]) - (2.0 / 3.0)) < 1e-5

    rows = [
        json.loads(line)
        for line in (out / "leafq001" / "teacher_node_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == tree.num_chunks


def test_qsentence_family_records_handle_odd_orphan_as_pass_through(tmp_path: Path) -> None:
    from scripts import build_manifesto_qsentence_dspy_labeled_grid as cli

    corpus_csv = tmp_path / "manifesto_corpus_df.csv"
    mpds_csv = tmp_path / "manifesto_maindataset.csv"
    out = tmp_path / "grid"
    _write_synthetic_qsentence_csv(corpus_csv)
    _write_synthetic_mpds_csv(mpds_csv)
    assert cli.main(
        [
            "--corpus-csv",
            str(corpus_csv),
            "--mpds-csv",
            str(mpds_csv),
            "--leaf-qsentences",
            "1",
            "--train-n",
            "1",
            "--val-n",
            "0",
            "--test-n",
            "0",
            "--output-dir",
            str(out),
        ]
    ) == 0

    tree = load_labeled_trees(out / "leafq001" / "labeled_trees.jsonl")[0]
    family = ManifestoQSentenceDSPyFamily(
        config=ManifestoQSentenceDSPyFamilyConfig(
            leaf_size_tokens=128,
            lm_context_window_tokens=4096,
            max_completion_tokens=256,
            prompt_template_overhead_tokens=512,
        )
    )
    g_records = family._qsentence_g_records([tree])
    assert len(g_records) == tree.num_chunks
    orphan_records = [
        row
        for row in g_records
        if "Promote this only child" in str(row.get("prompt") or "")
    ]
    assert orphan_records
    assert "RIGHT_STATE" not in orphan_records[0]["prompt"]



def _build_single_tree_grid(tmp_path: Path):
    from scripts import build_manifesto_qsentence_dspy_labeled_grid as cli

    corpus_csv = tmp_path / "manifesto_corpus_df.csv"
    mpds_csv = tmp_path / "manifesto_maindataset.csv"
    out = tmp_path / "grid"
    _write_synthetic_qsentence_csv(corpus_csv)
    _write_synthetic_mpds_csv(mpds_csv)
    assert cli.main(
        [
            "--corpus-csv", str(corpus_csv),
            "--mpds-csv", str(mpds_csv),
            "--leaf-qsentences", "1",
            "--train-n", "1", "--val-n", "0", "--test-n", "0",
            "--output-dir", str(out),
        ]
    ) == 0
    return load_labeled_trees(out / "leafq001" / "labeled_trees.jsonl")[0]


def test_g_records_carry_lopsidedness(tmp_path: Path) -> None:
    """Every g node record exposes sibling-mass lopsidedness in [0,1].

    The lopsidedness drives C2-calibration weighting: deep lopsided merges (where
    mass-weighting strictly beats equal-averaging) get a larger reward weight so g
    cannot collapse to a balanced-leaf averager. Leaves carry lopsidedness 0.
    """
    from src.ctreepo.manifesto_qsentence_dspy_family import _lopsidedness_weight

    tree = _build_single_tree_grid(tmp_path)
    family = ManifestoQSentenceDSPyFamily(
        config=ManifestoQSentenceDSPyFamilyConfig(
            leaf_size_tokens=128,
            lm_context_window_tokens=4096,
            max_completion_tokens=256,
            prompt_template_overhead_tokens=512,
        )
    )
    records = family._qsentence_g_records([tree])
    assert records, "expected g node records"
    for r in records:
        lop = r["metadata"]["lopsidedness"]
        assert 0.0 <= lop <= 1.0
        if r["metadata"]["is_leaf"]:
            assert lop == 0.0  # leaves have no children -> neutral weight
    # strength 0 => all weights 1.0; strength>0 => lopsided merges weigh more
    lops = [r["metadata"]["lopsidedness"] for r in records]
    assert all(_lopsidedness_weight(l, strength=0.0) == 1.0 for l in lops)
    assert all(_lopsidedness_weight(l, strength=4.0) >= 1.0 for l in lops)


def test_treepo_preference_bridge_consumes_qsentence_grid(tmp_path: Path) -> None:
    from src.ctreepo.treepo_bridge.manifesto_preferences import (
        build_manifesto_qsentence_preferences,
    )
    from src.training.preference.optimizer_adapters import build_dpo_training_records
    from treepo.state import make_unit_id

    tree = _build_single_tree_grid(tmp_path)
    preferences = build_manifesto_qsentence_preferences([tree], mode="ranked")
    dpo_rows = build_dpo_training_records(preferences)

    assert len(preferences) == tree.num_chunks
    assert len(dpo_rows) == tree.num_chunks * 2
    records = preferences.to_records("general")
    assert any(record["unit_type"] == "qsentence" for record in records)
    assert any(record["unit_type"] == "root" for record in records)
    by_node = {record["node_id"]: record for record in records}
    first_leaf_id = str(tree.levels[0][0])
    root_id = str(tree.levels[-1][0])
    first_leaf = by_node[first_leaf_id]
    root = by_node[root_id]
    assert first_leaf["unit_id"] == make_unit_id(tree.doc_id, first_leaf_id)
    assert first_leaf["tree_id"] == tree.doc_id
    assert first_leaf["doc_id"] == tree.doc_id
    assert first_leaf["node_id"] == first_leaf_id
    assert first_leaf["level"] == 0
    assert first_leaf["position"] == 0
    assert first_leaf["parent_id"] is not None
    assert first_leaf["left_child_id"] is None
    assert first_leaf["right_child_id"] is None
    assert root["unit_id"] == make_unit_id(tree.doc_id, root_id)
    assert root["parent_id"] is None
    assert root["left_child_id"] is not None
    assert root["right_child_id"] is not None
    assert (
        "Convert this Manifesto Project quasi-sentence span"
        in records[0]["context"]
    )
    assert records[0]["candidates"][0]["id"] == "gold_cmp_state"
    assert (
        records[0]["metadata"]["label_source"]
        == "manifesto_qsentence_cmp_annotations_v1"
    )


def test_qsentence_bridge_exports_finetune_and_dspy_adapter_bundles(tmp_path: Path) -> None:
    from src.ctreepo.treepo_bridge.manifesto_preferences import (
        build_manifesto_qsentence_preferences,
        export_manifesto_qsentence_finetune_adapters,
    )

    tree = _build_single_tree_grid(tmp_path)
    preferences = build_manifesto_qsentence_preferences([tree], mode="ranked")

    result = export_manifesto_qsentence_finetune_adapters(
        preferences,
        tmp_path / "adapter_exports",
        save_hf=False,
    )

    assert result["summary"]["adapter_names"] == [
        "dspy_examples",
        "embedding",
        "trl_dpo",
        "trl_grpo",
        "trl_reward",
        "trl_scalar_reward",
        "trl_sft",
    ]
    assert result["summary"]["learning_adapter_names"] == ["thinkingtrees_dspy"]

    dspy_export = result["adapters"]["dspy_examples"]
    assert dspy_export["counts"]["sft"] == tree.num_chunks
    assert dspy_export["counts"]["dpo"] == tree.num_chunks * 2
    dspy_sft = Path(dspy_export["files"]["sft"])
    dspy_dpo = Path(dspy_export["files"]["dpo"])
    assert dspy_sft.exists()
    assert dspy_dpo.exists()
    first_dspy_row = json.loads(dspy_dpo.read_text(encoding="utf-8").splitlines()[0])
    assert first_dspy_row["preferred"] == "A"
    assert first_dspy_row["metadata"]["unit_type"] in {"qsentence", "merge", "root"}

    embedding = result["adapters"]["embedding"]
    assert Path(embedding["files"]["embedding_ranked"]).exists()
    assert embedding["counts"]["embedding_triplets"] == tree.num_chunks * 2

    dspy_learning = result["learning_adapters"]["thinkingtrees_dspy"]
    assert dspy_learning["dry_run"] is True
    assert dspy_learning["trainer"] == "src.ctreepo.dspy_family:DSPyFamily"
    assert dspy_learning["prepared"]["core_adapter"] == "dspy_examples"
    assert Path(dspy_learning["prepared"]["files"]["dpo"]).exists()




def test_qsentence_bridge_exports_complete_finetune_bundle(tmp_path: Path) -> None:
    from src.ctreepo.treepo_bridge.manifesto_preferences import (
        export_manifesto_qsentence_finetune_bundle,
    )

    tree = _build_single_tree_grid(tmp_path)
    result = export_manifesto_qsentence_finetune_bundle(
        [tree],
        tmp_path / "bundle",
        mode="ranked",
        save_hf=False,
    )

    assert result["bundle_kind"] == "manifesto_qsentence"
    assert result["summary"]["n_trees"] == 1
    assert result["counts"]["dataset"] == tree.num_chunks
    assert Path(result["files"]["tree_records"]).exists()
    assert Path(result["files"]["dataset"]).exists()
    assert result["finetune_adapters"]["summary"]["learning_adapter_names"] == ["thinkingtrees_dspy"]
    assert Path(result["finetune_adapters"]["learning_adapters"]["thinkingtrees_dspy"]["prepared"]["files"]["dpo"]).exists()


def test_export_manifesto_qsentence_preferences_script_writes_treepo_artifacts(tmp_path: Path) -> None:
    from scripts import export_manifesto_qsentence_preferences as cli
    from treepo.state import make_unit_id
    from treepo.tree import load_tree_records

    tree = _build_single_tree_grid(tmp_path)
    labeled_trees = tmp_path / "grid" / "leafq001" / "labeled_trees.jsonl"
    output_dir = tmp_path / "preferences"

    assert cli.main([
        "--labeled-trees", str(labeled_trees),
        "--output-dir", str(output_dir),
        "--mode", "ranked",
    ]) == 0

    result = json.loads((output_dir / "manifesto_qsentence_preferences_result.json").read_text(encoding="utf-8"))
    assert result["mode"] == "ranked"
    assert result["n_trees"] == 1
    assert result["counts"]["units"] == tree.num_chunks
    assert result["counts"]["dpo"] == tree.num_chunks * 2
    assert result["counts"]["reward"] == tree.num_chunks * 2
    assert result["counts"]["grpo"] == tree.num_chunks
    finetune = result["finetune_adapters"]
    assert finetune["summary"]["n_adapters"] == 7
    assert finetune["summary"]["learning_adapter_names"] == ["thinkingtrees_dspy"]
    assert Path(finetune["adapters"]["dspy_examples"]["files"]["dpo"]).exists()
    assert Path(finetune["adapters"]["trl_dpo"]["files"]["dpo"]).exists()
    assert Path(finetune["learning_adapters"]["thinkingtrees_dspy"]["prepared"]["files"]["sft"]).exists()

    for path_text in result["files"].values():
        assert Path(path_text).exists()

    tree_records = load_tree_records(result["files"]["tree_records"])
    assert len(tree_records) == 1
    assert tree_records[0].tree_id == tree.doc_id
    assert tree_records[0].root_label == tree.document_score
    assert any(
        node.state is not None and node.state.kind == "manifesto_policy"
        for node in tree_records[0].nodes
    )

    dataset = json.loads(Path(result["files"]["dataset"]).read_text(encoding="utf-8"))
    units = dataset["units"]
    assert len(units) == tree.num_chunks
    assert all(row["tree_id"] == tree.doc_id for row in units)
    assert all(row["doc_id"] == tree.doc_id for row in units)
    assert all(row["node_id"] for row in units)
    assert all(row["unit_id"] == make_unit_id(tree.doc_id, row["node_id"]) for row in units)
    assert any(row["unit_type"] == "root" and row["left_child_id"] and row["right_child_id"] for row in units)
    assert any(row["unit_type"] == "qsentence" and row["parent_id"] for row in units)


def test_scheduled_sampling_rate_zero_is_noop(tmp_path: Path, monkeypatch) -> None:
    """rate=0 must reproduce legacy gold-children prompts exactly."""
    tree = _build_single_tree_grid(tmp_path)
    family = ManifestoQSentenceDSPyFamily(
        config=ManifestoQSentenceDSPyFamilyConfig(
            leaf_size_tokens=128,
            lm_context_window_tokens=4096,
            max_completion_tokens=256,
            prompt_template_overhead_tokens=512,
            g_scheduled_sampling_rate=0.0,
        )
    )

    def boom(self, *, g_program, trees):  # must never be called at rate 0
        raise AssertionError("scheduled sampling generated states at rate 0")

    monkeypatch.setattr(
        ManifestoQSentenceDSPyFamily, "_generate_all_node_states_batched", boom
    )

    baseline = family._qsentence_g_records([tree])
    scheduled = family._qsentence_g_records([tree], g_program=object(), iteration=5)
    assert [r["prompt"] for r in baseline] == [r["prompt"] for r in scheduled]
    assert all(r["metadata"]["scheduled_sampling_rate"] == 0.0 for r in scheduled)


def test_scheduled_sampling_substitutes_generated_child_states(tmp_path: Path, monkeypatch) -> None:
    """rate=1.0 must feed g's OWN generated child state into every merge prompt."""
    tree = _build_single_tree_grid(tmp_path)
    family = ManifestoQSentenceDSPyFamily(
        config=ManifestoQSentenceDSPyFamilyConfig(
            leaf_size_tokens=128,
            lm_context_window_tokens=4096,
            max_completion_tokens=256,
            prompt_template_overhead_tokens=512,
            g_scheduled_sampling_rate=1.0,
        )
    )
    marker = "GENERATED_STATE_MARKER_42"

    def fake_states_batched(self, *, g_program, trees):
        # every node of every tree "generates" the recognizable marker. The
        # batched path returns one {node_id: state} dict per input tree.
        return [{str(nid): marker for nid in t.nodes} for t in trees]

    monkeypatch.setattr(
        ManifestoQSentenceDSPyFamily,
        "_generate_all_node_states_batched",
        fake_states_batched,
    )

    records = family._qsentence_g_records([tree], g_program=object(), iteration=0)
    merge_records = [r for r in records if not r["metadata"]["is_leaf"]]
    assert merge_records, "expected at least one merge node"
    # Every merge prompt must carry the generated marker (rate=1 substitutes all
    # available children), and none should still be gold-only.
    assert all(marker in r["prompt"] for r in merge_records)
    assert all(r["metadata"]["scheduled_sampling_rate"] == 1.0 for r in merge_records)
    assert any(r["metadata"]["used_generated_children"] for r in merge_records)
    # Leaf prompts are raw text spans and must NOT contain child-state markers.
    leaf_records = [r for r in records if r["metadata"]["is_leaf"]]
    assert all(marker not in r["prompt"] for r in leaf_records)


def test_scheduled_sampling_rate_ramps_with_iteration() -> None:
    family = ManifestoQSentenceDSPyFamily(
        config=ManifestoQSentenceDSPyFamilyConfig(
            leaf_size_tokens=128,
            lm_context_window_tokens=4096,
            max_completion_tokens=256,
            prompt_template_overhead_tokens=512,
            g_scheduled_sampling_rate=0.8,
            g_scheduled_sampling_rate_start=0.1,
            g_scheduled_sampling_ramp_per_iter=0.3,
        )
    )
    assert family._scheduled_sampling_rate(iteration=0) == pytest.approx(0.1)
    assert family._scheduled_sampling_rate(iteration=1) == pytest.approx(0.4)
    assert family._scheduled_sampling_rate(iteration=2) == pytest.approx(0.7)
    # capped at 0.8
    assert family._scheduled_sampling_rate(iteration=10) == pytest.approx(0.8)


def test_qsentence_training_record_summary_persists_selected_node_mask(tmp_path: Path) -> None:
    from scripts import build_manifesto_qsentence_dspy_labeled_grid as cli

    corpus_csv = tmp_path / "manifesto_corpus_df.csv"
    mpds_csv = tmp_path / "manifesto_maindataset.csv"
    grid = tmp_path / "grid"
    _write_synthetic_qsentence_csv(corpus_csv)
    _write_synthetic_mpds_csv(mpds_csv)
    assert cli.main(
        [
            "--corpus-csv",
            str(corpus_csv),
            "--mpds-csv",
            str(mpds_csv),
            "--leaf-qsentences",
            "1",
            "--train-n",
            "1",
            "--val-n",
            "0",
            "--test-n",
            "0",
            "--output-dir",
            str(grid),
        ]
    ) == 0

    tree = load_labeled_trees(grid / "leafq001" / "labeled_trees.jsonl")[0]
    family = ManifestoQSentenceDSPyFamily(
        config=ManifestoQSentenceDSPyFamilyConfig(
            leaf_size_tokens=128,
            lm_context_window_tokens=4096,
            max_completion_tokens=256,
            prompt_template_overhead_tokens=512,
            max_train_records=2,
            record_sample_seed=17,
        )
    )
    pre_cap = family._qsentence_g_records([tree])
    selected = family._cap_qsentence_records(pre_cap, role="g")
    out = tmp_path / "artifacts"

    family._write_qsentence_training_record_artifacts(
        output_dir=out,
        iteration=2,
        role="g",
        records=selected,
        pre_cap_records=pre_cap,
    )

    summary = json.loads(
        (out / "g_qs_training_records_summary_iter_02.json").read_text(encoding="utf-8")
    )
    assert summary["record_cap"]["applied"] is True
    assert summary["record_cap"]["pre_cap_count"] == len(pre_cap)
    assert summary["record_cap"]["post_cap_count"] == len(selected) == 2
    assert summary["record_cap"]["selection_policy"] == (
        "deterministic_uniform_without_replacement_over_qsentence_node_records"
    )
    assert summary["objective"]["gold_label_kind"] == "observed_gold_g_completion"
    assert summary["local_law_contract"]["estimand"] == "training_subset_only_not_ipw_estimator"

    keys = [
        json.loads(line)
        for line in (out / "g_qs_selected_record_keys_iter_02.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(keys) == 2
    assert all(row["doc_id"] == tree.doc_id for row in keys)
    assert all(row["node_id"] for row in keys)


def test_sampled_supervision_contract_records_leaf_window_propensity(tmp_path: Path) -> None:
    from scripts import build_manifesto_qsentence_dspy_labeled_grid as cli
    from scripts import run_manifesto_qsentence_sampled_supervision as sampled

    corpus_csv = tmp_path / "manifesto_corpus_df.csv"
    mpds_csv = tmp_path / "manifesto_maindataset.csv"
    grid = tmp_path / "grid"
    _write_synthetic_qsentence_csv(corpus_csv)
    _write_synthetic_mpds_csv(mpds_csv)
    assert cli.main(
        [
            "--corpus-csv",
            str(corpus_csv),
            "--mpds-csv",
            str(mpds_csv),
            "--leaf-qsentences",
            "1",
            "--train-n",
            "1",
            "--val-n",
            "0",
            "--test-n",
            "0",
            "--output-dir",
            str(grid),
        ]
    ) == 0

    tree = load_labeled_trees(grid / "leafq001" / "labeled_trees.jsonl")[0]
    example = sampled._sample_example(
        tree,
        sample_leaf_count=1,
        dims=("rile",),
        rng=random.Random(123),
    )
    assert example is not None
    assert example.sample_unit == "leaf_window"
    assert example.total_leaf_count == len(tree.get_leaves())
    assert example.leaf_inclusion_probability == 1.0 / len(tree.get_leaves())

    prompt = json.loads(example.prompt)
    assert prompt["sampling"]["sample_unit"] == "leaf_window"
    assert prompt["sampling"]["leaf_inclusion_probability"] == example.leaf_inclusion_probability
    assert "not individual-qsentence sampling" in prompt["sampling"]["qsentence_inclusion_note"]

    mask = sampled._sample_mask_row(example, split="train", index=0)
    assert mask["sampling_scheme"] == "uniform_without_replacement_over_leaf_windows"
    assert mask["leaf_inclusion_probability"] == example.leaf_inclusion_probability
    assert mask["sample_node_ids"] == example.sample_node_ids
