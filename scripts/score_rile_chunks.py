#!/usr/bin/env python3
"""LLM-score q-sentence leaves on RILE (left-right ideological position) for the
rile LLM-recreation arm.

RILE is NOT a Benoit 1-7 policy-INTENSITY dimension — it's a left/right composite
(MPDS publishes doc rile in [-100,+100], normalized to [0,1] via (r+100)/200, where
0.5 = neutral). So each q-sentence gets a left-right LEAN score in [0,1]:
  0.0 = strongly LEFT (welfare expansion, peace, internationalism, labour, etc.)
  0.5 = neutral / not ideological / non-political
  1.0 = strongly RIGHT (free market, military, nationalism, law-and-order, etc.)
The doc-level rile is recovered by g composing these leaf leans (same as the gold
path, where leaf rile comes from CMP codes). Mirrors score_benoit_chunks.py's batched
DSPy structure (gemma-4-31B fleet, ports 8010-8013); writes the same scores JSON shape
({doc_id|node_id: float}) the relabel script consumes.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tasks.manifesto.qsentence_chunks import collect_chunks  # noqa: E402

LOGGER = logging.getLogger("score_rile_chunks")


def _parse_rile(text: str) -> Optional[float]:
    """Extract a 0-1 left-right score from model output; clamp to [0,1]."""
    m = re.search(r"-?\d+(?:\.\d+)?", str(text or ""))
    if not m:
        return None
    return max(0.0, min(1.0, float(m.group(0))))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid-dir", type=Path, required=True)
    p.add_argument("--leaf", type=int, default=1)
    p.add_argument("--model", default="openai/nvidia/Gemma-4-31B-IT-NVFP4")
    p.add_argument(
        "--api-base",
        default="http://localhost:8010/v1,http://localhost:8011/v1,"
        "http://localhost:8012/v1,http://localhost:8013/v1",
    )
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--max-concurrent", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-threads", type=int, default=256)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--output", type=Path, default=Path("outputs/mpds_rile_llmseg_scores"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    import dspy
    from src.config.dspy_config import configure_dspy
    from src.core.dspy_batch_client import BatchedDSPyLM

    lm = BatchedDSPyLM(
        model=str(args.model), api_base=str(args.api_base), api_key=str(args.api_key),
        temperature=0.0, max_tokens=int(args.max_tokens), cache=False,
        max_concurrent=int(args.max_concurrent), batch_size=int(args.batch_size),
        batch_timeout=0.02,
    )
    configure_dspy(lm=lm)

    class RileChunkSignature(dspy.Signature):
        """Rate the LEFT-RIGHT ideological position of one quasi-sentence from a
        political manifesto, on a 0.0-1.0 scale.

        0.0 = strongly LEFT (e.g. welfare-state expansion, public services,
        peace/anti-military, internationalism, labour/unions, equality,
        environmental protection, civil rights).
        0.5 = neutral, non-ideological, or not political.
        1.0 = strongly RIGHT (e.g. free markets, lower taxes, military strength,
        nationalism/sovereignty, law-and-order, traditional morality, limited
        government).
        Use only the text. ALWAYS output one number 0.0-1.0; never output NA.
        Output 0.5 if neutral or not political."""

        document: str = dspy.InputField(desc="A single manifesto quasi-sentence.")
        score: str = dspy.OutputField(desc="Single number 0.0-1.0 (0.5 if neutral/non-political). Never NA.")

    predictor = dspy.Predict(RileChunkSignature)

    chunks = collect_chunks(Path(args.grid_dir), int(args.leaf), "leaves")
    LOGGER.info("collected %d leaf=%d nodes (rile)", len(chunks), int(args.leaf))
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores: Dict[str, float] = {}
    n_fail = 0

    def _score_one(item: Tuple[str, str, str]) -> Tuple[str, Optional[float]]:
        doc_id, node_id, text = item
        try:
            with dspy.context(lm=lm):
                res = predictor(document=text)
            return f"{doc_id}|{node_id}", _parse_rile(getattr(res, "score", ""))
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
                LOGGER.info("[rile] %d/%d scored (%d fail)", i + 1, len(chunks), n_fail)

    path = out_dir / f"leafq{int(args.leaf):03d}_rile.json"
    path.write_text(json.dumps(scores) + "\n", encoding="utf-8")
    LOGGER.info("[rile] wrote %d scores (%d fail) -> %s", len(scores), n_fail, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
