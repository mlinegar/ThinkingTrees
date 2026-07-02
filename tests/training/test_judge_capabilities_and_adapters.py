from __future__ import annotations

import json
from typing import Dict

import pytest

from src.core.logged_supervision import ObservationUnitKind, SamplingMetadata
from src.core.preference_supervision import preference_supervision_metadata
from src.training.judge_optimization import JudgeOptimizationConfig, JudgeOptimizer
from src.training.preference.judge_capabilities import invoke_comparative_judgment_sync
from src.training.preference.optimizer_adapters import (
    build_dpo_training_records,
    build_group_grpo_training_records,
    build_reward_model_training_records,
    coerce_treepo_preference_dataset,
    prepare_binary_optimizer_dataset,
)
from src.training.preference.types import (
    ComparativeCandidate,
    ComparativeDataset,
    ComparativeJudgmentRecord,
)


def _compact_state(scores: Dict[str, float], *, total_non_header: int = 1) -> str:
    return json.dumps(
        {
            "cmp_state": {"compact_targets": scores},
            "total_non_header": int(total_non_header),
        },
        sort_keys=True,
    )


def _make_qsentence_labeled_tree():
    from src.tasks.manifesto.span_targets import COMPACT_TARGET_DIMENSIONS
    from src.tree.labeled import LabeledNode, LabeledTree

    def scores(rile: float, domain_1: float, domain_2: float) -> Dict[str, float]:
        out = {dim: 0.0 for dim in COMPACT_TARGET_DIMENSIONS}
        out["rile"] = float(rile)
        out["domain_1"] = float(domain_1)
        out["domain_2"] = float(domain_2)
        return out

    left_scores = scores(1.0, 1.0, 0.0)
    right_scores = scores(0.0, 0.0, 1.0)
    root_scores = scores(0.5, 0.5, 0.5)
    tree = LabeledTree(
        doc_id="11110_202001",
        document_text="Cut taxes for workers.\nProtect public healthcare.",
        document_score=0.5,
        metadata={
            "split": "train",
            "label_source": "manifesto_qsentence_cmp_annotations_v1",
        },
        label_source="manifesto_qsentence_cmp_annotations_v1",
    )
    for node in (
        LabeledNode(
            node_id="leaf_0",
            doc_id=tree.doc_id,
            level=0,
            text="Cut taxes for workers.",
            score=left_scores["rile"],
            dimension_scores=left_scores,
            metadata={
                "target_summary": _compact_state(left_scores),
                "teacher_summary": _compact_state(left_scores),
                "target_dimension_scores_0_1": left_scores,
                "total_qsentences": 1,
                "total_non_header_qsentences": 1,
                "leaf_qsentences": 1,
                "qsentence_start_index": 0,
                "qsentence_end_index": 1,
                "cmp_counts": {"104": 1},
                "domain_counts": {"domain_1": 1},
                "rile_raw": 100.0,
                "rile_norm": 1.0,
            },
        ),
        LabeledNode(
            node_id="leaf_1",
            doc_id=tree.doc_id,
            level=0,
            text="Protect public healthcare.",
            score=right_scores["rile"],
            dimension_scores=right_scores,
            metadata={
                "target_summary": _compact_state(right_scores),
                "teacher_summary": _compact_state(right_scores),
                "target_dimension_scores_0_1": right_scores,
                "total_qsentences": 1,
                "total_non_header_qsentences": 1,
                "leaf_qsentences": 1,
                "qsentence_start_index": 1,
                "qsentence_end_index": 2,
                "cmp_counts": {"202": 1},
                "domain_counts": {"domain_2": 1},
                "rile_raw": -100.0,
                "rile_norm": 0.0,
            },
        ),
        LabeledNode(
            node_id="root_0",
            doc_id=tree.doc_id,
            level=1,
            text=tree.document_text,
            score=root_scores["rile"],
            dimension_scores=root_scores,
            left_child_id="leaf_0",
            right_child_id="leaf_1",
            metadata={
                "target_summary": _compact_state(root_scores, total_non_header=2),
                "teacher_summary": _compact_state(root_scores, total_non_header=2),
                "target_dimension_scores_0_1": root_scores,
                "total_qsentences": 2,
                "total_non_header_qsentences": 2,
                "leaf_qsentences": 1,
                "qsentence_start_index": 0,
                "qsentence_end_index": 2,
                "cmp_counts": {"104": 1, "202": 1},
                "domain_counts": {"domain_1": 1, "domain_2": 1},
                "rile_raw": 0.0,
                "rile_norm": 0.5,
            },
        ),
    ):
        tree.add_node(node)
    return tree


def _make_comparative_record(
    *,
    record_id: str = "cmp1",
    example_id: str = "doc1",
    num_candidates: int = 3,
) -> ComparativeJudgmentRecord:
    candidates = []
    for index in range(1, num_candidates + 1):
        score = float(num_candidates - index + 1)
        candidates.append(
            ComparativeCandidate(
                candidate_id=f"C{index}",
                response=f"summary {index}",
                rank=index,
                response_signal_value=score,
                metadata={"generation_config": {"temperature": 0.1 * index}},
            )
        )
    return ComparativeJudgmentRecord(
        record_id=record_id,
        source_example_id=example_id,
        original_text=f"original {example_id}",
        rubric="rubric",
        reference_score=0.75,
        law_type="sufficiency",
        candidates=candidates,
        sampling=SamplingMetadata(
            joint_propensity=0.25,
            unit_kind=ObservationUnitKind.PAIR,
        ),
        preference_supervision=preference_supervision_metadata(
            application_name="test_collection",
            law_type="sufficiency",
            response_signal_name="judge_score",
            response_signal_min=1.0,
            response_signal_max=float(num_candidates),
        ).with_updates(preference_family="groupwise"),
        aggregate_sample_weight=4.0,
        metadata={"confidence": 0.9, "reasoning": "ordered by utility"},
    )


def test_invoke_comparative_judgment_sync_falls_back_to_pairwise() -> None:
    score_map: Dict[str, float] = {
        "summary alpha": 3.0,
        "summary beta": 2.0,
        "summary gamma": 1.0,
    }

    class PairwiseOnlyJudge:
        def compare(
            self,
            *,
            context: str,
            original_text: str,
            summary_a: str,
            summary_b: str,
            law_type: str = "sufficiency",
        ) -> dict[str, object]:
            del context, original_text, law_type
            score_a = score_map[summary_a]
            score_b = score_map[summary_b]
            return {
                "preferred": "A" if score_a >= score_b else "B",
                "confidence": 0.9,
                "score_estimate_a": score_a,
                "score_estimate_b": score_b,
                "response_signal_name": "judge_score",
            }

    result = invoke_comparative_judgment_sync(
        PairwiseOnlyJudge(),
        context="rubric",
        original_text="original",
        candidate_summaries=["summary alpha", "summary beta", "summary gamma"],
        law_type="sufficiency",
    )

    assert result.ordered_candidate_ids == ["C1", "C2", "C3"]
    assert result.candidate_scores["C1"] == pytest.approx(3.0)
    assert result.candidate_scores["C2"] == pytest.approx(2.0)
    assert result.raw_payload["pairwise_fallback"] is True


def test_optimizer_adapters_handle_comparative_records_directly() -> None:
    dataset = ComparativeDataset([_make_comparative_record()])

    binary_dataset = prepare_binary_optimizer_dataset(
        dataset,
        projection="adjacent",
        keep_existing=False,
    )
    assert len(binary_dataset.pairs) == 2
    assert [pair.preferred for pair in binary_dataset.pairs] == ["A", "A"]

    dpo_rows = build_dpo_training_records(dataset, projection="adjacent")
    assert len(dpo_rows) == 2
    assert dpo_rows[0]["chosen"] == "summary 1"
    assert dpo_rows[0]["rejected"] == "summary 2"

    reward_rows = build_reward_model_training_records(dataset, projection="adjacent")
    assert len(reward_rows) == 2
    assert reward_rows[0]["chosen"] == "summary 1"
    assert reward_rows[0]["chosen_score"] == pytest.approx(3.0)
    assert reward_rows[0]["rejected_score"] == pytest.approx(2.0)

    grpo_rows = build_group_grpo_training_records(dataset)
    assert len(grpo_rows) == 1
    assert grpo_rows[0]["responses"] == ["summary 1", "summary 2", "summary 3"]
    assert grpo_rows[0]["ranks"] == [1, 2, 3]
    assert grpo_rows[0]["reference_score"] == pytest.approx(0.75)
    assert grpo_rows[0]["original_text"] == "original doc1"


def test_optimizer_adapters_accept_treepo_preference_dataset(tmp_path) -> None:
    from treepo.methods.preference import Candidate, PreferenceDataset, PreferenceRecord

    dataset = PreferenceDataset(
        (
            PreferenceRecord(
                record_id="pref1",
                unit_id="doc1",
                unit_type="qsentence",
                context="Choose the better summary.",
                candidates=(
                    Candidate(id="a", value="summary a", score=0.2),
                    Candidate(id="b", value="summary b", score=0.9, preferred=True),
                ),
                target="g",
                weight=2.0,
                propensity=0.5,
                metadata={"treepo": {"sample_weight_source": "test"}},
            ),
        )
    )

    dpo_rows = build_dpo_training_records(dataset)
    assert len(dpo_rows) == 1
    assert dpo_rows[0]["chosen"] == "summary b"
    assert dpo_rows[0]["rejected"] == "summary a"
    assert dpo_rows[0]["sample_weight"] == pytest.approx(4.0)
    assert dpo_rows[0]["metadata"]["treepo"]["sample_weight_source"] == "test"

    reward_rows = build_reward_model_training_records(dataset)
    assert len(reward_rows) == 1
    assert reward_rows[0]["chosen"] == "summary b"
    assert reward_rows[0]["chosen_score"] == pytest.approx(0.9)
    assert reward_rows[0]["rejected_score"] == pytest.approx(0.2)

    grpo_rows = build_group_grpo_training_records(dataset)
    assert len(grpo_rows) == 1
    assert grpo_rows[0]["responses"] == ["summary b", "summary a"]
    assert grpo_rows[0]["ranks"] == [1, 2]

    path = tmp_path / "preference_dataset.json"
    dataset.save(path)
    assert build_dpo_training_records(path)[0]["chosen"] == "summary b"


def test_optimizer_adapters_accept_treepo_preference_dataset_and_hf_rows(tmp_path) -> None:
    from treepo.methods.preference import Candidate, PreferenceDataset, PreferenceRecord

    dataset = PreferenceDataset(
        [
            PreferenceRecord(
                record_id="score1",
                unit_id="node1",
                unit_type="qsentence",
                target="g",
                context="Summarize this qsentence.",
                candidates=(
                    Candidate(id="specific", value="specific evidence", score=1.0),
                    Candidate(id="generic", value="generic text", score=0.1),
                ),
                metadata={"law_type": "c1_leaf"},
            ),
            PreferenceRecord(
                record_id="rank1",
                unit_id="root1",
                unit_type="root",
                target="f",
                context="Score this root document.",
                candidates=(
                    Candidate(id="best", value="RILE score: 2", rank=1),
                    Candidate(id="ok", value="RILE score: 1", rank=2),
                    Candidate(id="bad", value="RILE score: -2", rank=3),
                ),
                metadata={"law_type": "root_label"},
            ),
        ]
    )

    assert coerce_treepo_preference_dataset(dataset) is dataset
    dpo_rows = build_dpo_training_records(dataset)
    assert len(dpo_rows) == 3
    assert dpo_rows[0]["chosen"] == "specific evidence"
    reward_rows = build_reward_model_training_records(dataset)
    assert reward_rows[0]["chosen_score"] == pytest.approx(1.0)
    grpo_rows = build_group_grpo_training_records(dataset)
    assert len(grpo_rows) == 2
    assert grpo_rows[1]["responses"] == ["RILE score: 2", "RILE score: 1", "RILE score: -2"]

    hf_dataset = dataset.to_hf_dataset_dict()
    assert len(build_dpo_training_records(hf_dataset)) == 3
    path = tmp_path / "preference_dataset.json"
    dataset.save(path)
    assert len(build_reward_model_training_records(path)) == 3


def test_manifesto_qsentence_preferences_flow_to_trl_records() -> None:
    from src.ctreepo.treepo_bridge.manifesto_preferences import (
        build_manifesto_qsentence_preferences,
    )

    tree = _make_qsentence_labeled_tree()
    preferences = build_manifesto_qsentence_preferences([tree], mode="ranked")
    assert all(row["tree_id"] == tree.doc_id for row in preferences.units)
    assert all(row["doc_id"] == tree.doc_id for row in preferences.units)
    assert all(row["node_id"] for row in preferences.units)
    assert all(row["unit_id"] == f"{tree.doc_id}:{row['node_id']}" for row in preferences.units)
    assert any(row["left_child_id"] and row["right_child_id"] for row in preferences.units)

    dpo_rows = build_dpo_training_records(preferences)
    reward_rows = build_reward_model_training_records(preferences)
    grpo_rows = build_group_grpo_training_records(preferences.to_hf_dataset_dict())

    assert len(preferences) == tree.num_chunks
    assert len(dpo_rows) == tree.num_chunks * 2
    assert len(reward_rows) == tree.num_chunks * 2
    assert len(grpo_rows) == tree.num_chunks
    assert json.loads(dpo_rows[0]["chosen"])["kind"] == "manifesto_policy"
    assert "Convert this Manifesto Project quasi-sentence span" in dpo_rows[0]["prompt"]
    assert any(row["metadata"]["unit_type"] == "qsentence" for row in dpo_rows)
    assert any(row["metadata"]["unit_type"] == "root" for row in grpo_rows)
    assert all(row["metadata"]["tree_id"] == tree.doc_id for row in dpo_rows)
    assert all(row["metadata"]["doc_id"] == tree.doc_id for row in dpo_rows)
    assert all(row["metadata"]["node_id"] for row in dpo_rows)
    assert all(row["metadata"]["law_type"] == "qsentence_cmp_state" for row in dpo_rows)
    assert any(len(row["responses"]) == 3 for row in grpo_rows)


def test_judge_optimizer_accepts_comparative_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    comparative_dataset = ComparativeDataset(
        [_make_comparative_record(record_id="cmp_many", example_id="doc_many", num_candidates=11)]
    )

    class FakeJudge:
        use_dspy_predictor = True

        def forward(
            self,
            *,
            context: str,
            original_text: str,
            summary_a: str,
            summary_b: str,
            law_type: str = "sufficiency",
        ) -> dict[str, object]:
            del context, original_text, summary_a, summary_b, law_type
            return {"preferred": "A", "confidence": 0.9}

    class FakeGEPA:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def compile(self, module, trainset):
            assert len(trainset) >= 8
            return module

    import src.training.judge_optimization as judge_optimization_module

    monkeypatch.setattr(judge_optimization_module.dspy, "GEPA", FakeGEPA)

    optimizer = JudgeOptimizer(
        config=JudgeOptimizationConfig(
            budget="light",
            num_threads=1,
            test_split=0.2,
            use_propensity_weighting=False,
        )
    )

    optimized_judge, results = optimizer.optimize(
        comparative_dataset,
        use_oracle_as_ground_truth=False,
        initial_judge=FakeJudge(),
    )

    assert isinstance(optimized_judge, FakeJudge)
    assert results["total_comparative_input"] == 1
    assert results["total_pairs_input"] == 10
    assert results["baseline"]["accuracy"] == pytest.approx(1.0)
