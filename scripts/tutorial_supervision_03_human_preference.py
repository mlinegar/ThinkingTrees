#!/usr/bin/env python3
"""CPU-only walkthrough: submit human preference and export canonical supervision."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preference_collection import PreferenceDimension, PreferenceRequest, PreferenceStore


def run_example() -> dict[str, object]:
    store = PreferenceStore()

    pair_request = PreferenceRequest(
        request_id="pair_req",
        text_a="Summary A keeps the vote but misses the total amount.",
        text_b="Summary B keeps the vote and the amount.",
        original_text="Budget committee memo with vote and approved amount.",
        rubric="Prefer summaries that preserve the final budget decision.",
        law_type="sufficiency",
        source_doc_id="budget_memo",
    )
    scalar_request = PreferenceRequest(
        request_id="scalar_req",
        text_a="Single-attempt summary for a second memo.",
        original_text="Second memo with a single candidate summary.",
        rubric="Rate overall faithfulness from 1 to 5.",
        law_type="merge",
        source_doc_id="memo_2",
        dimensions=[PreferenceDimension(kind="scalar", name="faithfulness", scale=(1.0, 5.0))],
    )

    store.enqueue(pair_request)
    store.enqueue(scalar_request)

    store.submit_human_pairwise_preference(
        "pair_req",
        preferred="B",
        reasoning="B preserves both the vote and the amount.",
        confidence=1.0,
        score_estimate_a=2.0,
        score_estimate_b=5.0,
    )
    store.submit_human_scalar_preference(
        "scalar_req",
        score=4.0,
        dimension_name="faithfulness",
        reasoning="Solid single-attempt summary.",
    )

    supervision = store.to_supervision_dataset()
    binary_projection = store.to_binary_projection_dataset(projection="adjacent")
    return {
        "example": "human_preference_store_export",
        "completed_items": store.get_statistics()["completed"],
        "supervision_summary": supervision.summary(),
        "binary_projection_count": len(binary_projection.comparisons),
        "binary_truth_sources": sorted(
            {pair.truth_label_source for pair in binary_projection.comparisons}
        ),
    }


def main() -> int:
    print(json.dumps(run_example(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
