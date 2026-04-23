#!/usr/bin/env python3
"""
Phase 0 pilot: one Benoit-comparable Pearson-r number on the Economic
dimension, with a small manifesto subset and a single open-weight LLM.

Replicates Benoit et al. (2026 AJPS) §4.2 (Figure 1 Economic = .87 baseline,
Table 6 open-weight = .84-.87). Does NOT use the C-TreePO tree — this is
the flat summarize-then-scale baseline that Phase A aims to reproduce.

Usage:
    python scripts/phase0_economic_pilot.py \\
        --port 8000 \\
        --countries 51 41 31 22 \\
        --min-year 2010 \\
        --max-year 2019 \\
        --max-manifestos 50 \\
        --output-dir outputs/phase0_economic

Outputs under <output-dir>/:
    per_manifesto.jsonl      one row per manifesto: id, party, year,
                             ches_expert_mean_1_7, llm_score_1_7, summary,
                             reasoning, is_na
    report.json              corpus-level CorrelationReport + run metadata
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config.dspy_config import configure_dspy, create_vllm_lm, create_vllm_lm_multi
from src.core.protocols import format_merge_input
from src.preprocessing.chunker import chunk_for_ops
from src.tasks.manifesto import ManifestoDataset
from src.tasks.manifesto.corpus_metrics import compute_corpus_pearson_r
from src.tasks.manifesto.dimensions import (
    BENOIT_DIMENSIONS,
    PolicyDimension,
    get_preservation_rubric,
)
from src.tasks.manifesto.dimension_scorer import DimensionScorer
from src.tasks.manifesto.expert_benchmarks import (
    load_benoit_expert_means,
    load_benoit_mp_crosswalk,
)
from src.tasks.manifesto.pipeline import UnifiedManifestoG

logger = logging.getLogger(__name__)
_DIM_FROM_NAME = {d.value: d for d in PolicyDimension}

_BENOIT_FIGURE1 = {
    PolicyDimension.ECONOMIC: 0.87,
    PolicyDimension.SOCIAL: 0.92,
    PolicyDimension.IMMIGRATION: 0.89,
    PolicyDimension.EU: 0.91,
    PolicyDimension.ENVIRONMENT: 0.82,
    PolicyDimension.DECENTRALIZATION: 0.49,
}

_BENOIT_TABLE3_UPPER = {
    PolicyDimension.ECONOMIC: 0.88,
    PolicyDimension.SOCIAL: 0.91,
    PolicyDimension.IMMIGRATION: 0.88,
    PolicyDimension.EU: 0.95,
    PolicyDimension.ENVIRONMENT: 0.84,
    PolicyDimension.DECENTRALIZATION: 0.78,
}


def _benoit_published_reference(dim: PolicyDimension) -> dict:
    return {
        "figure1_proprietary_ensemble": _BENOIT_FIGURE1[dim],
        "table3_expert_upper_bound": _BENOIT_TABLE3_UPPER[dim],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8000, help="vLLM server port (single endpoint).")
    p.add_argument("--ports", type=int, nargs="+", default=None,
                   help="Multiple vLLM server ports for round-robin load-balancing "
                        "(overrides --port). e.g. --ports 8010 8011.")
    p.add_argument("--model", type=str, default=None, help="Model name (auto-detected if None)")
    p.add_argument("--countries", type=int, nargs="+", default=[51, 41, 31, 22],
                   help="CMP country codes (default: UK=51, Germany=41, France=31, Netherlands=22)")
    p.add_argument("--min-year", type=int, default=2010)
    p.add_argument("--max-year", type=int, default=2019)
    p.add_argument("--max-manifestos", type=int, default=50,
                   help="Upper bound on number of manifestos to score")
    p.add_argument("--chunk-chars", type=int, default=8000,
                   help="Char budget per summarization chunk (Benoit-ish: ~one summary fits)")
    # Defaults centralized in src/tasks/manifesto/pipeline_config.py — target
    # ~50-100 concurrent requests to a single TP=4 Gemma-4 server.
    from src.tasks.manifesto.pipeline_config import (
        DEFAULT_MANIFESTO_WORKERS, DEFAULT_SUMMARY_WORKERS,
    )
    p.add_argument("--manifesto-workers", type=int, default=DEFAULT_MANIFESTO_WORKERS,
                   help="parallel manifestos to process concurrently")
    p.add_argument("--summary-workers", type=int, default=DEFAULT_SUMMARY_WORKERS,
                   help="parallel leaf/merge calls within one manifesto")
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs") / f"phase0_economic_{datetime.now(timezone.utc):%Y%m%d_%H%M}")
    p.add_argument("--seed", type=int, default=0, help="Python RNG seed for manifesto subsampling")
    p.add_argument("--mp-data-dir", type=Path, default=None,
                   help="Override ManifestoDataset data_dir. Default uses ManifestoDataset's default path.")
    p.add_argument("--dimension", type=str, default="economic", choices=sorted(_DIM_FROM_NAME),
                   help="Benoit policy dimension to score on the 1-7 scale.")
    p.add_argument("--mode", choices=["tree", "concat", "flat"], default="tree",
                   help="'tree' (default): chunk+summarize+merge+score. "
                        "'concat': chunk+summarize, concat leaf summaries, no merges. "
                        "'flat': skip chunking/summarization; score truncated raw text.")
    p.add_argument("--flat-chars-cap", type=int, default=24000,
                   help="Max chars fed to scorer in --mode flat (truncation).")
    import os
    env_cap = os.environ.get("MANIFESTO_MAX_TOKENS")
    p.add_argument("--max-tokens", type=int, default=int(env_cap) if env_cap else None,
                   help="Cap on LLM output tokens per call (summarizer/merger/scorer). "
                        "Defaults to $MANIFESTO_MAX_TOKENS if set, else vLLM auto-sized. "
                        "Benoit's 300-400 word summaries fit in ~1024 tokens; 8192 gives headroom.")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def _summarize_manifesto(
    text: str,
    g: UnifiedManifestoG,
    *,
    chunk_chars: int,
    rubric: str,
    mode: str = "tree",
    flat_chars_cap: int = 24000,
    max_workers: int = 16,
) -> str:
    """Produce a summary string per manifesto under the requested mode.

    modes:
      - 'tree' (default): chunk -> summarize leaves -> pairwise merge up to root.
      - 'concat': chunk -> summarize leaves -> join with blank lines. No merges.
      - 'flat': skip chunking and summarization entirely. Truncate the raw
                manifesto text to `flat_chars_cap` chars and use that directly
                as the "summary" for the scoring step.
    """
    if mode == "flat":
        # Direct scoring on the raw text (truncated to fit in context). This
        # is the "no tree, no summary" baseline — closest to Benoit's flat
        # approach if they had skipped summarization.
        return text[:flat_chars_cap]

    from concurrent.futures import ThreadPoolExecutor

    chunks = chunk_for_ops(text, max_chars=chunk_chars, strategy="axis")
    if not chunks:
        raise ValueError("chunk_for_ops returned empty list")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        summaries = list(pool.map(
            lambda c: g(content=c.text, rubric=rubric), chunks
        ))

        if mode == "concat":
            # No merge — concatenate leaf summaries with clear separators so
            # the scorer can still find distinct content blocks. Tests whether
            # the merge operation (C3 law) carries signal beyond concatenation.
            return "\n\n---\n\n".join(summaries)

        # default 'tree' mode: pairwise merge up to the root
        while len(summaries) > 1:
            pairs = []
            carry = None
            for i in range(0, len(summaries), 2):
                if i + 1 < len(summaries):
                    pairs.append((summaries[i], summaries[i + 1]))
                else:
                    carry = summaries[i]
            merged = list(pool.map(
                lambda p: g(content=format_merge_input(p[0], p[1]), rubric=rubric), pairs
            ))
            if carry is not None:
                merged.append(carry)
            summaries = merged
    return summaries[0]


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Configuring DSPy with vLLM on port %d", args.port)
    lm_kwargs = {"model": args.model, "temperature": 0.0, "cache": True}
    if args.max_tokens is not None:
        lm_kwargs["max_tokens"] = args.max_tokens
        logger.info("Using max_tokens=%d for LM outputs", args.max_tokens)
    if args.ports:
        logger.info("Load-balancing across vLLM ports %s", args.ports)
        lm = create_vllm_lm_multi(ports=args.ports, **lm_kwargs)
    else:
        lm = create_vllm_lm(port=args.port, **lm_kwargs)
    configure_dspy(lm=lm)
    logger.info("Using LM: %s", getattr(lm, "model", "unknown"))

    logger.info("Loading MP manifestos: countries=%s years=%d-%d data_dir=%s",
                args.countries, args.min_year, args.max_year, args.mp_data_dir)
    dataset = ManifestoDataset(
        data_dir=args.mp_data_dir,
        countries=args.countries,
        min_year=args.min_year,
        max_year=args.max_year,
        require_text=True,
    )
    all_ids = dataset.get_all_ids()
    logger.info("Dataset has %d manifestos matching filters", len(all_ids))

    dim = _DIM_FROM_NAME[args.dimension]
    rubric = get_preservation_rubric(dim)
    spec = BENOIT_DIMENSIONS[dim]
    logger.info("Loading Benoit expert means + MP crosswalk for dimension=%s", dim.value)
    expert_df = load_benoit_expert_means(dim)
    crosswalk = load_benoit_mp_crosswalk()
    joined = expert_df.merge(
        crosswalk[["manifesto", "party", "year"]], on="manifesto", how="inner"
    )
    if args.min_year is not None:
        joined = joined[joined["year"] >= args.min_year]
    if args.max_year is not None:
        joined = joined[joined["year"] <= args.max_year]
    expert_lookup = {
        (int(row.party), int(row.year)): (float(row.expert_mean), row.manifesto)
        for row in joined.itertuples()
    }
    logger.info("Expert lookup: %d (party, year) keys", len(expert_lookup))

    g = UnifiedManifestoG(use_cot=False)
    scorer = DimensionScorer(spec, use_cot=False)

    per_manifesto_path = args.output_dir / "per_manifesto.jsonl"
    already_scored: dict[str, dict] = {}
    if per_manifesto_path.exists():
        for line in per_manifesto_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            mid = row.get("manifesto_id")
            if mid:
                already_scored[mid] = row
        logger.info("Resume: %d manifestos already in %s", len(already_scored), per_manifesto_path)
    rows: list[dict] = list(already_scored.values())
    scored = len(already_scored)
    t0 = time.time()

    def _work_one(manifesto_id: str) -> dict | None:
        sample = dataset.get_sample(manifesto_id)
        if sample is None or not sample.text:
            return None
        key = (int(sample.party_id), int(sample.year))
        if key not in expert_lookup:
            return None
        expert_value, benoit_manifesto_key = expert_lookup[key]
        try:
            summary = _summarize_manifesto(
                sample.text, g,
                chunk_chars=args.chunk_chars, rubric=rubric,
                mode=args.mode, flat_chars_cap=args.flat_chars_cap,
            )
            result = scorer.forward(summary)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed on %s: %s", manifesto_id, exc)
            return None
        return {
            "manifesto_id": manifesto_id,
            "benoit_manifesto_key": benoit_manifesto_key,
            "party_id": sample.party_id,
            "party_abbrev": sample.party_abbrev,
            "country_name": sample.country_name,
            "year": sample.year,
            "benoit_expert_mean": expert_value,
            "llm_score_1_7": result["score"],
            "is_na": result["score"] is None,
            "summary": summary,
            "reasoning": result.get("reasoning", ""),
        }

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock

    # Submit every un-done ID; _work_one returns None fast for non-scorable
    # ones (wrong party, no expert match). max_manifestos caps SUCCESSFUL
    # rows, not submission attempts.
    pending = [m for m in all_ids if m not in already_scored]
    logger.info("Processing %d un-done IDs with manifesto_workers=%d "
                "(scoring cap: %d total rows)",
                len(pending), args.manifesto_workers, args.max_manifestos)
    write_lock = Lock()
    with per_manifesto_path.open("a" if already_scored else "w") as fp, \
         ThreadPoolExecutor(max_workers=args.manifesto_workers) as outer:
        futures = {outer.submit(_work_one, mid): mid for mid in pending}
        for fut in as_completed(futures):
            row = fut.result()
            if row is None:
                continue
            with write_lock:
                rows.append(row)
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                fp.flush()
                scored += 1
                if scored % 5 == 0:
                    logger.info("Scored %d/%d manifestos (%.1fs elapsed)",
                                scored, args.max_manifestos, time.time() - t0)
                if scored >= args.max_manifestos:
                    break

    logger.info("Done scoring. Computing correlation report.")
    pred = [r["llm_score_1_7"] for r in rows]
    true = [r["benoit_expert_mean"] for r in rows]
    report = compute_corpus_pearson_r(pred, true, pred_rescaled=pred)

    summary_out = {
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": getattr(lm, "model", "unknown"),
            "dimension": dim.value,
            "countries": args.countries,
            "min_year": args.min_year,
            "max_year": args.max_year,
            "max_manifestos": args.max_manifestos,
            "chunk_chars": args.chunk_chars,
            "mode": args.mode,
            "flat_chars_cap": args.flat_chars_cap if args.mode == "flat" else None,
            "scored": scored,
            "elapsed_seconds": round(time.time() - t0, 1),
        },
        "benoit_reference": _benoit_published_reference(dim),
        "report": report.as_dict(),
    }
    (args.output_dir / "report.json").write_text(json.dumps(summary_out, indent=2))

    logger.info("Pearson r = %.3f (n=%d, 95%% CI [%.3f, %.3f])",
                report.pearson_r, report.n, report.pearson_ci_low, report.pearson_ci_high)
    logger.info("Benoit baseline: Figure 1 Economic = .87; Table 6 open-weight = .84-.86")
    logger.info("Outputs written to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
