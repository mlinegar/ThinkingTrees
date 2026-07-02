#!/usr/bin/env python3
"""LLM-score q-sentence CHUNKS (coarse leaves) on Benoit expert dimensions.

This is the "light LLM supervision over aggregations of q-sentences" step: apply
the same Benoit scoring rubric the full-doc baseline uses (`get_benoit_scoring_
context`) to each leaf CHUNK (e.g. leaf=16 q-sentences), giving f a real per-leaf
target instead of the broadcast doc label. Output: per-(doc_id, node_id, dim)
normalized [0,1] score, cached to JSON, consumed by the relabel+train step.

Scores are produced on the LLM's native 1-7 Benoit scale and normalized
(x-1)/6 -> [0,1] (Pearson vs expert means is scale-invariant downstream).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tasks.manifesto.benoit_scoring_contexts import get_benoit_scoring_context
from src.tasks.manifesto.dimensions import PolicyDimension
from src.tasks.manifesto.qsentence_chunks import collect_chunks

LOGGER = logging.getLogger(__name__)

DIM_TO_POLICY = {
    "economic": PolicyDimension.ECONOMIC,
    "social": PolicyDimension.SOCIAL,
    "immigration": PolicyDimension.IMMIGRATION,
    "eu": PolicyDimension.EU,
    "environment": PolicyDimension.ENVIRONMENT,
    "decentralization": PolicyDimension.DECENTRALIZATION,
}


def _parse_score(text: str) -> Optional[float]:
    """Extract the first 1-7 number from the model output; normalize to [0,1]."""
    import re

    m = re.search(r"-?\d+(?:\.\d+)?", str(text or ""))
    if not m:
        return None
    raw = float(m.group(0))
    raw = max(1.0, min(7.0, raw))
    return (raw - 1.0) / 6.0



def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid-dir", type=Path, default=Path("outputs/benoit_qsentence_grid_full"))
    p.add_argument("--leaf", type=int, default=16)
    p.add_argument("--dimensions", default="economic")
    p.add_argument("--model", default="openai/nvidia/Gemma-4-31B-IT-NVFP4")
    p.add_argument(
        "--api-base",
        default="http://localhost:8010/v1,http://localhost:8011/v1,http://localhost:8012/v1,http://localhost:8013/v1",
    )
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--max-concurrent", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-threads", type=int, default=256)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument(
        "--force-score",
        action="store_true",
        help="Forbid NA: every chunk gets a 1-7 (4=neutral/irrelevant). Default allows NA (off-topic chunks unscored).",
    )
    p.add_argument(
        "--node-levels",
        choices=("leaves", "merges", "all"),
        default="leaves",
        help="Which tree nodes to score: 'leaves' (chunks), 'merges' (intermediate"
        "+root span nodes), or 'all'. 'merges' adds holistic LLM span scores at"
        " every internal level for node-level g supervision.",
    )
    p.add_argument("--output", type=Path, default=Path("outputs/benoit_chunk_scores"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    import dspy
    from src.config.dspy_config import configure_dspy
    from src.core.dspy_batch_client import BatchedDSPyLM

    lm = BatchedDSPyLM(
        model=str(args.model),
        api_base=str(args.api_base),
        api_key=str(args.api_key),
        temperature=0.0,
        max_tokens=int(args.max_tokens),
        cache=False,
        max_concurrent=int(args.max_concurrent),
        batch_size=int(args.batch_size),
        batch_timeout=0.02,
    )
    configure_dspy(lm=lm)

    if bool(args.force_score):
        class BenoitChunkSignature(dspy.Signature):
            """Predict the expert 1-7 score for one policy dimension from a text chunk.

            Use only the provided text and task context. ALWAYS output one number
            1-7; never output NA. If the chunk is neutral or not about this
            dimension, output 4."""

            dimension: str = dspy.InputField(desc="Policy dimension name.")
            task_context: str = dspy.InputField(desc="Dimension scale and scoring guidance.")
            document: str = dspy.InputField(desc="A chunk of manifesto text.")
            score: str = dspy.OutputField(desc="Single numeric score 1-7 (4 if neutral/irrelevant). Never NA.")
    else:
        class BenoitChunkSignature(dspy.Signature):
            """Predict the expert 1-7 score for one policy dimension from a text chunk.

            Use only the provided text and task context. Output one number 1-7,
            or NA if the chunk is not about this dimension."""

            dimension: str = dspy.InputField(desc="Policy dimension name.")
            task_context: str = dspy.InputField(desc="Dimension scale and scoring guidance.")
            document: str = dspy.InputField(desc="A chunk of manifesto text.")
            score: str = dspy.OutputField(desc="Single numeric score from 1 to 7, or NA.")

    predictor = dspy.Predict(BenoitChunkSignature)

    chunks = collect_chunks(Path(args.grid_dir), int(args.leaf), str(args.node_levels))
    LOGGER.info("collected %d leaf=%d nodes (%s)", len(chunks), int(args.leaf), str(args.node_levels))
    suffix = "" if str(args.node_levels) == "leaves" else f"_{args.node_levels}"
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    dims = [d.strip() for d in str(args.dimensions).replace(";", ",").split(",") if d.strip()]
    for dim in dims:
        if dim not in DIM_TO_POLICY:
            raise ValueError(f"unknown dim {dim!r}; allowed {list(DIM_TO_POLICY)}")
        context = get_benoit_scoring_context(DIM_TO_POLICY[dim])
        scores: Dict[str, float] = {}
        n_fail = 0

        def _score_one(item: Tuple[str, str, str]) -> Tuple[str, Optional[float]]:
            doc_id, node_id, text = item
            try:
                with dspy.context(lm=lm):
                    res = predictor(dimension=dim, task_context=context, document=text)
                return f"{doc_id}|{node_id}", _parse_score(getattr(res, "score", ""))
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("score failed %s|%s: %s", doc_id, node_id, exc)
                return f"{doc_id}|{node_id}", None

        with ThreadPoolExecutor(max_workers=int(args.num_threads)) as ex:
            futures = [ex.submit(_score_one, c) for c in chunks]
            for i, fut in enumerate(as_completed(futures)):
                key, val = fut.result()
                if val is None:
                    n_fail += 1
                else:
                    scores[key] = val
                if (i + 1) % 2000 == 0:
                    LOGGER.info("[%s] %d/%d scored (%d fail)", dim, i + 1, len(chunks), n_fail)

        path = out_dir / f"leafq{int(args.leaf):03d}_{dim}{suffix}.json"
        path.write_text(json.dumps(scores) + "\n", encoding="utf-8")
        LOGGER.info("[%s] wrote %d scores (%d fail) -> %s", dim, len(scores), n_fail, path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
