#!/usr/bin/env python3
"""CPU-only walkthrough: one scalar supervision source, many optimizer views."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.logged_supervision import ObservationUnitKind, SamplingMetadata
from src.core.supervision_metadata import judgment_supervision_metadata
from src.training.supervision import ResponseJudgment, SupervisionDataset


def _dataset() -> SupervisionDataset:
    judgments = []
    for candidate_id, response, score in (
        ("candidate_a", "Summary A", 1.0),
        ("candidate_b", "Summary B", 2.0),
        ("candidate_c", "Summary C", 3.0),
    ):
        judgments.append(
            ResponseJudgment(
                judgment_id=f"toy:{candidate_id}",
                source_example_id="toy_example",
                original_text="Original text",
                rubric="Higher score means better overall response.",
                response=response,
                response_id=candidate_id,
                reference_score=1.0,
                law_type="sufficiency",
                source_doc_id="toy_doc",
                truth_label_source="oracle",
                sampling=SamplingMetadata(
                    document_propensity=1.0,
                    unit_propensity=1.0,
                    label_propensity=1.0,
                    unit_kind=ObservationUnitKind.PAIR,
                    supports_ipw_estimation=True,
                ),
                supervision_metadata=judgment_supervision_metadata(
                    application_name="tutorial_scalar_comparative_binary_bridge",
                    supervision_channel_name="full_document_supervision",
                    supervision_signal_name="document_level_target",
                    response_signal_name="score",
                    law_type="sufficiency",
                    response_signal_min=1.0,
                    response_signal_max=3.0,
                ),
                response_signal_value=score,
                candidate_features=[score],
            )
        )
    return SupervisionDataset(response_judgments=judgments)


def run_example() -> dict[str, object]:
    supervision = _dataset()
    comparative_dataset = supervision.to_comparative_dataset()
    adjacent_binary = supervision.project_binary(projection="adjacent")
    top_binary = supervision.project_binary(projection="winner_vs_runner_up")

    comparative_record = comparative_dataset.records[0]
    return {
        "example": "scalar_comparative_binary_bridge",
        "supervision_summary": supervision.summary(),
        "scalar_scores": {
            judgment.response_id: judgment.response_signal_value
            for judgment in supervision.response_judgments
        },
        "comparative_view": {
            "n_records": len(comparative_dataset.records),
            "ordered_candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "rank": candidate.rank,
                    "score": candidate.response_signal_value,
                }
                for candidate in comparative_record.candidates
            ],
        },
        "binary_adjacent_view": {
            "n_pairs": len(adjacent_binary.comparisons),
            "pairs": [
                {
                    "pair_id": pair.pair_id,
                    "preferred": pair.preferred,
                    "score_estimate_a": pair.score_estimate_a,
                    "score_estimate_b": pair.score_estimate_b,
                }
                for pair in adjacent_binary.comparisons
            ],
        },
        "binary_winner_vs_runner_up_view": {
            "n_pairs": len(top_binary.comparisons),
            "pairs": [
                {
                    "pair_id": pair.pair_id,
                    "preferred": pair.preferred,
                    "score_estimate_a": pair.score_estimate_a,
                    "score_estimate_b": pair.score_estimate_b,
                }
                for pair in top_binary.comparisons
            ],
        },
        "optimizer_exports": {
            "scalar_reward_records": len(supervision.to_scalar_reward_records()),
            "group_grpo_records": len(supervision.to_group_grpo_records()),
            "dpo_records_adjacent": len(supervision.to_dpo_records(projection="adjacent")),
            "dpo_records_winner_vs_runner_up": len(
                supervision.to_dpo_records(projection="winner_vs_runner_up")
            ),
        },
    }


def main() -> int:
    print(json.dumps(run_example(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
