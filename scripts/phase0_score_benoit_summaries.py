#!/usr/bin/env python3
"""
Phase 0 variant: score Benoit's own anonymized summaries from data_masked.csv
using our DimensionScorer, then correlate against expert means.

Isolates our **scoring** step from our **summarization** step — the summaries
here are Benoit's GPT-4o outputs, not ours. This is also a clean slot for
C-TreePO ablations later: "score trained on Benoit summaries + expert means"
is literally the f-learned-on-summaries-and-root cell of the ablation matrix.

Usage:
    python scripts/phase0_score_benoit_summaries.py \\
        --port 8010 --dimension economic --max-n 50 \\
        --output-dir outputs/phase0_score_benoit_summaries
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config.dspy_config import configure_dspy, create_local_engine_lm
from src.config.local_inference import resolve_local_inference_config
from src.tasks.manifesto.corpus_metrics import compute_corpus_pearson_r
from src.tasks.manifesto.dimensions import BENOIT_DIMENSIONS, PolicyDimension
from src.tasks.manifesto.dimension_scorer import DimensionScorer
from src.tasks.manifesto.benoit_scoring_contexts import get_benoit_scoring_context
from src.tasks.manifesto.expert_benchmarks import (
    load_benoit_expert_means,
    load_benoit_masked_summaries,
)

logger = logging.getLogger(__name__)

_DIM_FROM_NAME = {d.value: d for d in PolicyDimension}

_PUBLISHED = {  # Benoit 2026 Figure 1 headline r values (proprietary 18-score ensemble)
    PolicyDimension.ECONOMIC: 0.87,
    PolicyDimension.SOCIAL: 0.92,
    PolicyDimension.IMMIGRATION: 0.89,
    PolicyDimension.EU: 0.91,
    PolicyDimension.ENVIRONMENT: 0.82,
    PolicyDimension.DECENTRALIZATION: 0.49,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--dimension", type=str, default="economic", choices=sorted(_DIM_FROM_NAME))
    p.add_argument("--max-n", type=int, default=None, help="Cap on summaries to score (default: all available)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--use-benoit-rubric", action="store_true",
                   help="Use Benoit's exact scoring rubric (SystemMessage from data_masked.csv) "
                        "instead of our DimensionScorer's default. True apples-to-apples replication.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs") / f"phase0_score_benoit_summaries_{datetime.now(timezone.utc):%Y%m%d_%H%M}")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dim = _DIM_FROM_NAME[args.dimension]
    spec = BENOIT_DIMENSIONS[dim]

    logger.info("Loading Benoit anonymized summaries for dimension=%s", dim.value)
    summaries = load_benoit_masked_summaries(dimension=dim)
    logger.info("Loaded %d summaries", len(summaries))

    logger.info("Loading Benoit expert means for dimension=%s", dim.value)
    experts = load_benoit_expert_means(dim)
    logger.info("Loaded %d expert-mean rows", len(experts))
    # data_experts 'manifesto' is e.g. "Austria - AU 2006 - AU 2006 Green.txt";
    # summaries 'manifesto_stem' strips the .txt. Align.
    experts_lookup = {
        str(row.manifesto).removesuffix(".txt"): float(row.expert_mean)
        for row in experts.itertuples()
    }
    summaries["benoit_manifesto_key"] = summaries["manifesto_stem"]
    summaries["expert_mean"] = summaries["manifesto_stem"].map(experts_lookup)
    scorable = summaries.dropna(subset=["expert_mean"]).reset_index(drop=True)
    logger.info("Summaries with matched expert means: %d", len(scorable))

    if args.max_n is not None and args.max_n < len(scorable):
        scorable = scorable.sample(n=args.max_n, random_state=0).reset_index(drop=True)
        logger.info("Subsampled to %d (seed=0)", len(scorable))

    logger.info("Configuring LM on port %d (T=%g, max_tokens=%d)", args.port, args.temperature, args.max_tokens)
    local_inference = resolve_local_inference_config(args)
    lm = create_local_engine_lm(**local_inference.dspy_kwargs(cache=True))
    configure_dspy(lm=lm)
    logger.info("Using LM: %s", getattr(lm, "model", "unknown"))

    scorer = DimensionScorer(spec, use_cot=False)
    benoit_ctx = get_benoit_scoring_context(dim) if args.use_benoit_rubric else None
    if benoit_ctx is not None:
        logger.info("Using Benoit's exact scoring rubric (%d chars) for %s", len(benoit_ctx), dim.value)
    rows: list[dict] = []
    per_path = args.output_dir / "per_summary.jsonl"
    t0 = time.time()

    with per_path.open("w") as fp:
        for i, src in enumerate(scorable.itertuples()):
            try:
                if benoit_ctx is not None:
                    result = scorer(summary=src.summary, task_context=benoit_ctx)
                else:
                    result = scorer(summary=src.summary)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Scorer failed on %s: %s", src.manifesto_stem, exc)
                result = {"score": None, "reasoning": f"ERROR: {exc}"}
            row = {
                "manifesto_stem": src.manifesto_stem,
                "expert_mean": float(src.expert_mean),
                "benoit_score": float(src.benoit_score) if pd.notna(src.benoit_score) else None,
                "our_score": result["score"],
                "our_is_na": result["score"] is None,
                "our_reasoning": (result.get("reasoning") or "")[:600],
            }
            rows.append(row)
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            fp.flush()
            if (i + 1) % 10 == 0:
                logger.info("Scored %d/%d (%.1fs elapsed)", i + 1, len(scorable), time.time() - t0)

    logger.info("Done. Computing correlations.")

    def _r(preds, truths):
        return compute_corpus_pearson_r(preds, truths)

    ours_vs_expert = _r([r["our_score"] for r in rows], [r["expert_mean"] for r in rows])
    benoit_vs_expert = _r([r["benoit_score"] for r in rows], [r["expert_mean"] for r in rows])
    ours_vs_benoit = _r([r["our_score"] for r in rows], [r["benoit_score"] for r in rows])

    published = _PUBLISHED[dim]
    report = {
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": getattr(lm, "model", "unknown"),
            "dimension": dim.value,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "n_scored": len(rows),
            "elapsed_seconds": round(time.time() - t0, 1),
        },
        "reference": {
            "benoit_figure1_proprietary_ensemble": published,
        },
        "ours_vs_expert": ours_vs_expert.as_dict(),
        "benoit_single_vs_expert": benoit_vs_expert.as_dict(),
        "ours_vs_benoit_single": ours_vs_benoit.as_dict(),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))

    def _line(label, rep):
        logger.info(
            "%-30s r=%+.3f  n=%d  ci=[%+.3f, %+.3f]",
            label, rep.pearson_r, rep.n, rep.pearson_ci_low, rep.pearson_ci_high,
        )
    _line("Ours vs expert_mean", ours_vs_expert)
    _line("Benoit single vs expert_mean", benoit_vs_expert)
    _line("Ours vs Benoit single score", ours_vs_benoit)
    logger.info("Reference: Benoit 18-score ensemble r = %.2f (Figure 1)", published)
    logger.info("Outputs in %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
