#!/usr/bin/env python3
"""
Phase 2b/2c: combined pipeline with one shared unified g across all 6 dims.

One `UnifiedManifestoG` with `JOINT_RUBRIC` produces a single summary per
manifesto (via parallel chunk → summarize → merge). That same summary is
then scored on all 6 dimensions by `JointDimensionScorer`. Inference cost
drops ~6× vs running 6 per-dim pipelines.

**Two-pass ordering (KV-cache friendly).** We run all summarization first,
then iterate the 6 dim rubrics on the outside so every scoring call inside
one dim-pass shares the same ~1.1K-token SYSTEM prefix. vLLM's prefix
cache (`enable_prefix_caching=True`) then skips prefill for the rubric on
calls 2..N of each dim-pass, saving ~1.5M tokens across a full 229-manifesto
run.

Writes:
  * summaries.jsonl       one row per manifesto (pass-1 output).
  * scores.jsonl          one row per (manifesto_id, dim) (pass-2 output).
  * per_manifesto.jsonl   reconciled row-per-manifesto schema, identical
                          to the pre-split output so downstream aggregators
                          (comparison_table, rescore_variants) keep working.
  * report.json           per-dim Pearson r + macro + Benoit reference.

Usage:
    python scripts/phase2_combined_pipeline.py \\
        --port 8010 \\
        --mp-data-dir data/raw/manifesto_corpus_benoit \\
        --countries 11 12 13 14 21 22 23 31 32 33 34 35 41 42 51 53 54 56 \\
                    61 62 64 81 82 83 86 87 88 92 93 94 95 96 97 \\
        --min-year 1989 --max-year 2019 \\
        --output-dir outputs/phase2/combined_pipeline
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config.dspy_config import configure_dspy, create_local_engine_lm
from src.config.local_inference import resolve_local_inference_config
from src.core.protocols import format_merge_input
from src.preprocessing.chunker import chunk_for_ops
from src.tasks.manifesto import ManifestoDataset
from src.tasks.manifesto.corpus_metrics import compute_corpus_pearson_r
from src.tasks.manifesto.dimensions import (
    BENOIT_DIMENSIONS,
    PolicyDimension,
    get_joint_rubric,
)
from src.tasks.manifesto.expert_benchmarks import (
    load_benoit_expert_means,
    load_benoit_mp_crosswalk,
)
from src.tasks.manifesto.joint_scorer import JointDimensionScorer
from src.tasks.manifesto.pipeline import UnifiedManifestoG

logger = logging.getLogger(__name__)

_ORDER = [
    PolicyDimension.ECONOMIC,
    PolicyDimension.SOCIAL,
    PolicyDimension.IMMIGRATION,
    PolicyDimension.EU,
    PolicyDimension.ENVIRONMENT,
    PolicyDimension.DECENTRALIZATION,
]

_BENOIT_FIGURE1 = {
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
    p.add_argument("--ports", type=int, nargs="+", default=None,
                   help="Multi-endpoint load-balancing (overrides --port).")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--mp-data-dir", type=Path,
                   default=project_root / "data" / "raw" / "manifesto_corpus_benoit")
    p.add_argument("--countries", type=int, nargs="*", default=None)
    p.add_argument("--min-year", type=int, default=1989)
    p.add_argument("--max-year", type=int, default=2019)
    p.add_argument("--max-manifestos", type=int, default=1000)
    p.add_argument("--chunk-chars", type=int, default=24000)
    # Defaults centralized in src/tasks/manifesto/pipeline_config.py — target
    # ~50-100 concurrent requests to a single TP=4 Gemma-4 server.
    from src.tasks.manifesto.pipeline_config import (
        DEFAULT_MANIFESTO_WORKERS, DEFAULT_SUMMARY_WORKERS, DEFAULT_SCORING_WORKERS,
    )
    p.add_argument("--summary-workers", type=int, default=DEFAULT_SUMMARY_WORKERS,
                   help="parallel leaf/merge calls within one manifesto")
    p.add_argument("--scoring-workers", type=int, default=DEFAULT_SCORING_WORKERS,
                   help="parallel scoring calls across dims for one summary")
    p.add_argument("--manifesto-workers", type=int, default=DEFAULT_MANIFESTO_WORKERS,
                   help="parallel manifestos to process concurrently")
    p.add_argument("--optimized-program", type=Path, default=None,
                   help="Optional path to a saved optimized JointDimensionScorer JSON "
                        "(from scripts/phase2_joint_optimize.py).")
    import os
    env_cap = os.environ.get("MANIFESTO_MAX_TOKENS")
    p.add_argument("--max-tokens", type=int, default=int(env_cap) if env_cap else None,
                   help="Cap on LLM output tokens per call (summarizer/merger/scorer). "
                        "Defaults to $MANIFESTO_MAX_TOKENS if set, else vLLM auto-sized.")
    p.add_argument("--output-dir", type=Path,
                   default=project_root / "outputs" / "phase2" /
                   f"combined_pipeline_{datetime.now(timezone.utc):%Y%m%d_%H%M}")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _summarize(
    text: str,
    g: UnifiedManifestoG,
    rubric: str,
    *,
    chunk_chars: int,
    max_workers: int,
) -> str:
    chunks = chunk_for_ops(text, max_chars=chunk_chars, strategy="axis")
    if not chunks:
        raise ValueError("chunk_for_ops returned empty")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        summaries = list(pool.map(lambda c: g(content=c.text, rubric=rubric), chunks))
        while len(summaries) > 1:
            pairs, carry = [], None
            for i in range(0, len(summaries), 2):
                if i + 1 < len(summaries):
                    pairs.append((summaries[i], summaries[i + 1]))
                else:
                    carry = summaries[i]
            merged = list(
                pool.map(
                    lambda p: g(content=format_merge_input(p[0], p[1]), rubric=rubric),
                    pairs,
                )
            )
            if carry is not None:
                merged.append(carry)
            summaries = merged
    return summaries[0]


def _score_one_dim_all_summaries(
    summaries: list[dict],
    dim: PolicyDimension,
    scorer: JointDimensionScorer,
    *,
    max_workers: int,
    already_scored_keys: set[tuple[str, str]],
    scores_fp,
    scores_lock,
) -> int:
    """Score every summary through one dim-pass (streams to scores_fp).

    Returns the number of NEW (manifesto_id, dim) rows appended to the
    jsonl during this pass. Parallelizes across manifestos so the shared
    SYSTEM rubric stays hot in vLLM's prefix cache.
    """
    spec = BENOIT_DIMENSIONS[dim]
    dim_value = dim.value
    pending = [
        s for s in summaries
        if (s["manifesto_id"], dim_value) not in already_scored_keys
    ]
    if not pending:
        return 0

    new_count = 0

    def _one(summary_row: dict) -> Optional[dict]:
        try:
            out = scorer(summary=summary_row["summary"], dimension_spec=spec)
        except Exception as e:  # noqa: BLE001
            return {
                "manifesto_id": summary_row["manifesto_id"],
                "dim": dim_value,
                "pred": None,
                "reasoning": f"ERROR: {e}",
            }
        return {
            "manifesto_id": summary_row["manifesto_id"],
            "dim": dim_value,
            "pred": out.get("score"),
            "reasoning": out.get("reasoning", ""),
        }

    from concurrent.futures import as_completed
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_one, s) for s in pending]
        for fut in as_completed(futures):
            row = fut.result()
            if row is None:
                continue
            with scores_lock:
                scores_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                scores_fp.flush()
                new_count += 1
    return new_count


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    local_inference = resolve_local_inference_config({**vars(args), "temperature": 0.0})
    if local_inference.max_tokens is not None:
        logger.info("Using max_tokens=%d for LM outputs", args.max_tokens)
    logger.info("Configuring LM on %s port(s) %s", local_inference.engine, list(local_inference.ports))
    lm = create_local_engine_lm(**local_inference.dspy_kwargs(cache=True))
    configure_dspy(lm=lm)
    logger.info("LM: %s", getattr(lm, "model", "unknown"))

    dataset = ManifestoDataset(
        data_dir=args.mp_data_dir,
        countries=args.countries,
        min_year=args.min_year,
        max_year=args.max_year,
        require_text=True,
    )
    all_ids = dataset.get_all_ids()
    logger.info("Dataset has %d manifestos matching filters", len(all_ids))

    # Expert lookups per dim, joined via Benoit manifesto crosswalk
    crosswalk = load_benoit_mp_crosswalk()
    expert_by_dim: dict[PolicyDimension, dict[tuple[int, int], float]] = {}
    expert_by_dim_1_7: dict[PolicyDimension, dict[tuple[int, int], float]] = {}
    for dim in _ORDER:
        experts = load_benoit_expert_means(dim)
        joined = experts.merge(crosswalk[["manifesto", "party", "year"]], on="manifesto", how="inner")
        expert_by_dim[dim] = {
            (int(r.party), int(r.year)): float(r.expert_mean)
            for r in joined.itertuples()
        }
        expert_by_dim_1_7[dim] = {
            (int(r.party), int(r.year)): float(r.expert_mean_1_7)
            for r in joined.itertuples()
        }
    # Any manifesto with at least one dim label is scorable; we'll fill per-dim
    scorable_keys = set()
    for mapping in expert_by_dim.values():
        scorable_keys.update(mapping.keys())
    logger.info("Scorable (party, year) keys across any dim: %d", len(scorable_keys))

    # Build / load scorer
    scorer = JointDimensionScorer(use_cot=False)
    if args.optimized_program is not None and args.optimized_program.exists():
        scorer.load(str(args.optimized_program))
        logger.info("Loaded optimized scorer from %s", args.optimized_program)

    g = UnifiedManifestoG(use_cot=False)
    rubric = get_joint_rubric()

    # --------------------------------------------------------------
    # Resume: load any prior progress from three files (in preference).
    # summaries.jsonl and scores.jsonl are the new canonical streaming
    # outputs; per_manifesto.jsonl is the legacy reconciled file that
    # we still emit at the end and also accept as an input when an old
    # run is being resumed.
    # --------------------------------------------------------------
    per_path = args.output_dir / "per_manifesto.jsonl"
    summaries_path = args.output_dir / "summaries.jsonl"
    scores_path = args.output_dir / "scores.jsonl"

    summaries_by_id: dict[str, dict] = {}
    scores_by_key: dict[tuple[str, str], dict] = {}

    # Load prior summaries streaming file
    if summaries_path.exists():
        for line in summaries_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("manifesto_id") and r.get("summary"):
                summaries_by_id[r["manifesto_id"]] = r
        logger.info("Resume[summaries.jsonl]: %d prior summaries", len(summaries_by_id))

    # Load prior scores streaming file
    if scores_path.exists():
        for line in scores_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            mid, dim_v = r.get("manifesto_id"), r.get("dim")
            if mid and dim_v:
                scores_by_key[(mid, dim_v)] = r
        logger.info("Resume[scores.jsonl]: %d prior (manifesto, dim) scores", len(scores_by_key))

    # Legacy migration: if an old per_manifesto.jsonl exists from a pre-
    # two-pass run, seed both streaming files from it (keep any in-flight
    # progress).
    if per_path.exists() and not (summaries_by_id or scores_by_key):
        logger.info("Legacy resume: migrating from %s", per_path)
        for line in per_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            mid = r.get("manifesto_id")
            if not mid:
                continue
            if r.get("summary"):
                summaries_by_id[mid] = {
                    "manifesto_id": mid,
                    "party_id": r.get("party_id"),
                    "party_abbrev": r.get("party_abbrev"),
                    "country_name": r.get("country_name"),
                    "year": r.get("year"),
                    "summary": r["summary"],
                    "expert_means": r.get("expert_means", {}),
                }
            preds = r.get("predictions") or {}
            reasoning = r.get("reasoning") or {}
            for dim_v, pred in preds.items():
                scores_by_key[(mid, dim_v)] = {
                    "manifesto_id": mid,
                    "dim": dim_v,
                    "pred": pred,
                    "reasoning": reasoning.get(dim_v, ""),
                }
        # Write migration forward into the new files so subsequent resumes
        # see them without re-reading per_manifesto.jsonl.
        with summaries_path.open("w") as fp:
            for row in summaries_by_id.values():
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        with scores_path.open("w") as fp:
            for row in scores_by_key.values():
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("Legacy migration done: %d summaries, %d scores rewritten",
                    len(summaries_by_id), len(scores_by_key))

    from concurrent.futures import as_completed
    from threading import Lock

    t0 = time.time()

    # --------------------------------------------------------------
    # Pass 1: summarize all manifestos that don't yet have a summary.
    # Outer MW threads, each internally fanning leaves/merges on SW.
    # --------------------------------------------------------------
    def _summarize_one(manifesto_id: str) -> Optional[dict]:
        sample = dataset.get_sample(manifesto_id)
        if sample is None or not sample.text:
            return None
        key = (int(sample.party_id), int(sample.year))
        if key not in scorable_keys:
            return None
        try:
            summary = _summarize(
                sample.text, g, rubric,
                chunk_chars=args.chunk_chars,
                max_workers=args.summary_workers,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Summarize failed on %s: %s", manifesto_id, exc)
            return None
        return {
            "manifesto_id": manifesto_id,
            "party_id": sample.party_id,
            "party_abbrev": sample.party_abbrev,
            "country_name": sample.country_name,
            "year": sample.year,
            "summary": summary,
            "expert_means": {
                dim.value: expert_by_dim[dim].get(key) for dim in _ORDER
            },
            "expert_means_1_7": {
                dim.value: expert_by_dim_1_7[dim].get(key) for dim in _ORDER
            },
        }

    # Cap on total produced summaries (matches old --max-manifestos).
    needed = max(0, args.max_manifestos - len(summaries_by_id))
    pending_for_summary = [m for m in all_ids if m not in summaries_by_id][: needed * 8]
    # Heuristic * 8 because most un-done IDs fail the scorable_keys check; the
    # inner loop breaks as soon as `args.max_manifestos` is reached.

    logger.info(
        "Pass 1 (summarize): %d pending IDs with manifesto_workers=%d "
        "(cap %d summaries total; already have %d)",
        len(pending_for_summary), args.manifesto_workers,
        args.max_manifestos, len(summaries_by_id),
    )

    summaries_lock = Lock()
    with summaries_path.open("a" if summaries_by_id else "w") as fp, \
         ThreadPoolExecutor(max_workers=args.manifesto_workers) as outer:
        futures = {outer.submit(_summarize_one, mid): mid for mid in pending_for_summary}
        for fut in as_completed(futures):
            row = fut.result()
            if row is None:
                continue
            with summaries_lock:
                summaries_by_id[row["manifesto_id"]] = row
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                fp.flush()
                if len(summaries_by_id) % 5 == 0:
                    logger.info("Summarized %d/%d (%.1fs)",
                                len(summaries_by_id), args.max_manifestos,
                                time.time() - t0)
                if len(summaries_by_id) >= args.max_manifestos:
                    # Cancel remaining in-flight futures; they won't be written
                    # but the underlying LM calls will finish on their own.
                    break

    summary_rows = list(summaries_by_id.values())
    logger.info("Pass 1 done: %d summaries. Starting Pass 2 (score-by-dim).",
                len(summary_rows))

    # --------------------------------------------------------------
    # Pass 2: iterate dims on the outside so each dim's ~1.1K-token
    # SYSTEM rubric stays hot in vLLM's prefix cache across all 229
    # scoring calls in that pass.
    # --------------------------------------------------------------
    scores_lock = Lock()
    with scores_path.open("a" if scores_by_key else "w") as scores_fp:
        for dim in _ORDER:
            t_dim = time.time()
            new_count = _score_one_dim_all_summaries(
                summary_rows,
                dim,
                scorer,
                max_workers=args.scoring_workers,
                already_scored_keys=set(scores_by_key.keys()),
                scores_fp=scores_fp,
                scores_lock=scores_lock,
            )
            # Refresh scores_by_key with what we just streamed so subsequent
            # dim passes don't re-read the file.
            if new_count:
                # Tail-read the rows we just appended (scores_fp was flushed).
                # Cheaper: re-scan the whole file once per dim (tiny I/O).
                scores_by_key.clear()
                for line in scores_path.read_text().splitlines():
                    if not line.strip():
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    mid, dv = r.get("manifesto_id"), r.get("dim")
                    if mid and dv:
                        scores_by_key[(mid, dv)] = r
            logger.info(
                "Pass 2 [%s]: +%d scores (%.1fs; total %d/%d done)",
                dim.value, new_count, time.time() - t_dim,
                sum(1 for k in scores_by_key if k[1] == dim.value),
                len(summary_rows),
            )

    # --------------------------------------------------------------
    # Reconcile: emit the legacy per_manifesto.jsonl schema for
    # downstream aggregators (comparison_table, rescore_variants, etc.).
    # --------------------------------------------------------------
    rows: list[dict] = []
    for row in summary_rows:
        mid = row["manifesto_id"]
        preds = {d.value: None for d in _ORDER}
        reasons = {d.value: "" for d in _ORDER}
        for dim in _ORDER:
            entry = scores_by_key.get((mid, dim.value))
            if entry is not None:
                preds[dim.value] = entry.get("pred")
                reasons[dim.value] = entry.get("reasoning", "") or ""
        rows.append({
            "manifesto_id": mid,
            "party_id": row.get("party_id"),
            "party_abbrev": row.get("party_abbrev"),
            "country_name": row.get("country_name"),
            "year": row.get("year"),
            "summary": row["summary"],
            "predictions": preds,
            "reasoning": reasons,
            "expert_means": row.get("expert_means", {}),
            "expert_means_1_7": row.get("expert_means_1_7", {}),
        })
    with per_path.open("w") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    scored = len(rows)
    logger.info("Done scoring. Reconciled %d rows into per_manifesto.jsonl", scored)
    per_dim = {}
    for dim in _ORDER:
        preds, truths = [], []
        for r in rows:
            p = r["predictions"].get(dim.value)
            e = r["expert_means"].get(dim.value)
            if p is None or e is None:
                continue
            preds.append(p)
            truths.append(e)
        if len(preds) < 4:
            # compute_corpus_pearson_r computes a Fisher CI that needs n-3 > 0;
            # fall through with no-CI report rather than crash on tiny samples.
            per_dim[dim.value] = {"n": len(preds), "pearson_r": None}
            continue
        rep = compute_corpus_pearson_r(preds, truths)
        per_dim[dim.value] = rep.as_dict()
        logger.info("%-18s r=%+.3f n=%d ci=[%+.3f,%+.3f]",
                    dim.value, rep.pearson_r, rep.n,
                    rep.pearson_ci_low, rep.pearson_ci_high)

    macro = sum(v["pearson_r"] for v in per_dim.values() if v.get("pearson_r") is not None)
    n_dims = sum(1 for v in per_dim.values() if v.get("pearson_r") is not None)
    macro_avg = macro / n_dims if n_dims else None

    report = {
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": getattr(lm, "model", "unknown"),
            "n_scored": scored,
            "elapsed_seconds": round(time.time() - t0, 1),
            "chunk_chars": args.chunk_chars,
            "summary_workers": args.summary_workers,
            "scoring_workers": args.scoring_workers,
            "optimized_program": str(args.optimized_program) if args.optimized_program else None,
        },
        "benoit_figure1_reference": {d.value: _BENOIT_FIGURE1[d] for d in _ORDER},
        "per_dim": per_dim,
        "macro_pearson_r": macro_avg,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    logger.info("macro avg r across dims = %+.3f", macro_avg if macro_avg else float("nan"))
    logger.info("Outputs in %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
