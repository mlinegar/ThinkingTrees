#!/usr/bin/env python3
"""CPU-only walkthrough: turn scored attempts into grouped comparative supervision."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.feedback import FeedbackDimension, FeedbackRequest, FeedbackResponse
from src.training.supervision import SupervisionDataset


def _request(request_id: str, candidate_text: str) -> FeedbackRequest:
    return FeedbackRequest(
        request_id=request_id,
        text_a=candidate_text,
        original_text="Summarize the committee memo while preserving the budget decision.",
        rubric="Higher scores mean better preservation of the budget decision.",
        node_id="memo_1",
        source_doc_id="memo_1",
        law_type="sufficiency",
        dimensions=[FeedbackDimension(kind="scalar", name="quality", scale=(1.0, 5.0))],
    )


def run_example() -> dict[str, object]:
    supervision = SupervisionDataset(
        response_judgments=[
            FeedbackResponse.from_human_scalar_feedback(
                request_id="cand_1",
                score=2.0,
                dimension_name="quality",
                reasoning="Drops the key budget decision.",
            ).to_response_judgment(
                _request("cand_1", "Candidate A: mentions the deadline but drops the budget vote."),
                response_id="candidate_a",
            ),
            FeedbackResponse.from_human_scalar_feedback(
                request_id="cand_2",
                score=4.0,
                dimension_name="quality",
                reasoning="Mostly preserves the important decision.",
            ).to_response_judgment(
                _request("cand_2", "Candidate B: keeps the vote and the budget number."),
                response_id="candidate_b",
            ),
            FeedbackResponse.from_human_scalar_feedback(
                request_id="cand_3",
                score=5.0,
                dimension_name="quality",
                reasoning="Best overall preservation.",
            ).to_response_judgment(
                _request("cand_3", "Candidate C: keeps the vote, amount, and who supported it."),
                response_id="candidate_c",
            ),
        ]
    )
    comparative_dataset = supervision.to_comparative_dataset()
    binary_projection = supervision.project_binary(projection="adjacent")
    record = comparative_dataset.records[0]
    ordered_candidates = [
        {
            "candidate_id": candidate.candidate_id,
            "rank": candidate.rank,
            "score": candidate.response_signal_value,
        }
        for candidate in record.candidates
    ]
    return {
        "example": "grouped_comparative_from_scalar_scores",
        "n_response_judgments": len(supervision.response_judgments),
        "n_grouped_records": len(comparative_dataset.records),
        "n_binary_projection_records": len(binary_projection.comparisons),
        "ordered_candidates": ordered_candidates,
        "projected_preferences": [
            {
                "pair_id": pair.pair_id,
                "preferred": pair.preferred,
                "score_estimate_a": pair.score_estimate_a,
                "score_estimate_b": pair.score_estimate_b,
            }
            for pair in binary_projection.comparisons
        ],
    }


def main() -> int:
    print(json.dumps(run_example(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
