from __future__ import annotations

import threading
from pathlib import Path

from src.core.logged_supervision import ObservationUnitKind, SamplingMetadata
from src.core.preference_supervision import preference_supervision_metadata
from src.training.preference.types import ComparativeCandidate, ComparativeJudgmentRecord
from src.training.tournament_loop import ToTConfig, TournamentOfTournamentsTrainer


def test_comparative_record_projects_to_adjacent_pairs() -> None:
    record = ComparativeJudgmentRecord(
        record_id="cmp_projection",
        source_example_id="doc_projection",
        original_text="original text",
        rubric="rubric",
        reference_score=0.4,
        law_type="sufficiency",
        candidates=[
            ComparativeCandidate(candidate_id="C1", response="summary a", rank=1, response_signal_value=0.9),
            ComparativeCandidate(candidate_id="C2", response="summary b", rank=2, response_signal_value=0.7),
            ComparativeCandidate(candidate_id="C3", response="summary c", rank=3, response_signal_value=0.1),
        ],
        sampling=SamplingMetadata(joint_propensity=0.5, unit_kind=ObservationUnitKind.PAIR),
        preference_supervision=preference_supervision_metadata(
            law_type="sufficiency",
            response_signal_name="oracle_relative_utility",
        ).with_updates(preference_family="groupwise"),
        metadata={"reasoning": "oracle ranking", "confidence": 0.9},
    )

    pairs = record.to_preference_pairs(projection="adjacent")
    assert len(pairs) == 2
    assert [pair.summary_a for pair in pairs] == ["summary a", "summary b"]
    assert [pair.summary_b for pair in pairs] == ["summary b", "summary c"]
    assert all(pair.preferred == "A" for pair in pairs)
    assert all(pair.preference_supervision.preference_family == "pairwise" for pair in pairs)
    assert all(
        pair.preference_supervision.metadata["source_record_id"] == "cmp_projection"
        for pair in pairs
    )


def test_tournament_of_tournaments_collects_comparative_dataset_with_listwise_judge(
    tmp_path: Path,
) -> None:
    counter_lock = threading.Lock()
    counter = {"n": 0}

    def summarizer(content: str, rubric: str) -> str:
        with counter_lock:
            counter["n"] += 1
            idx = counter["n"]
        return f"candidate_{idx}: {content[:10]} | {rubric[:8]}"

    class ListwiseJudge:
        judge_backend = "tot_listwise"

        def rank_candidates(
            self,
            *,
            context: str,
            original_text: str,
            candidate_summaries,
            law_type: str = "sufficiency",
        ):
            assert len(candidate_summaries) == 3
            return {
                "ordered_candidate_ids": ["C2", "C1", "C3"],
                "candidate_scores": {"C1": 0.4, "C2": 0.9, "C3": 0.1},
                "reasoning": "C2 is best.",
                "confidence": 0.85,
                "response_signal_name": "listwise_candidate_score",
            }

    trainer = TournamentOfTournamentsTrainer(
        summarizer=summarizer,
        oracle_predict=lambda text: float(len(str(text))) / 100.0,
        initial_judge=ListwiseJudge(),
        config=ToTConfig(
            max_iterations=1,
            min_iterations=1,
            n_samples_per_iteration=1,
            k_candidates=3,
            shuffle_samples_each_iteration=False,
        ),
        output_dir=tmp_path,
    )
    trainer._current_dspy_judge = ListwiseJudge()

    dataset = trainer._build_trees_and_collect_supervision(
        samples=[{"text": "document text", "doc_id": "doc_1", "reference_score": 0.5}],
        rubric="rubric text",
        iteration=1,
    )

    assert len(dataset.comparative_judgments) == 1
    assert dataset.comparative_judgments[0].source_example_id == "doc_1"

    projected = dataset.project_binary(projection="adjacent")
    assert len(projected.comparisons) == 2
    assert all(
        pair.pair_id.endswith(("proj:adjacent:000", "proj:adjacent:001"))
        for pair in projected.comparisons
    )
