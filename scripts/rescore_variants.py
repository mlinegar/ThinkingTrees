#!/usr/bin/env python3
"""
Rescore existing cached summaries at a different (temperature, n-samples)
config, without re-running the summarizer. Reuses summaries from
per_manifesto.jsonl files produced by phase0_economic_pilot.py (per-dim)
or phase2_combined_pipeline.py (combined).

Per-dim schema:
    {manifesto_id, summary, llm_score_1_7, benoit_expert_mean, ...}
Combined schema:
    {manifesto_id, summary, predictions: {dim: score}, expert_means: {dim: mean}, ...}

Output is written with the same schema under a mirrored path, prefixed
with outputs/rescore/T{T}_N{N}/..., so the comparison-table resolvers can
find it by just swapping the root.

Usage:
    # Per-dim: rescore all chunk_sweep economic cells at T=0.2 N=3
    for c in 64000 32000 16000 8000; do
      python scripts/rescore_variants.py \\
        --mode per-dim --dimension economic \\
        --input-dir outputs/chunk_sweep/economic_c${c} \\
        --output-dir outputs/rescore/T0.2_N3/chunk_sweep/economic_c${c} \\
        --temperature 0.2 --n-samples 3 \\
        --ports 8010 8011 8012 8013
    done

    # Combined: rescore all combined chunks at T=0.2 N=3
    for c in 64000 32000 16000 8000 24000; do
      python scripts/rescore_variants.py \\
        --mode combined \\
        --input-dir outputs/phase3/combined_c${c} \\
        --output-dir outputs/rescore/T0.2_N3/phase3/combined_c${c} \\
        --temperature 0.2 --n-samples 3 \\
        --ports 8010 8011 8012 8013
    done
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

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config.dspy_config import configure_dspy, create_vllm_lm, create_vllm_lm_multi
from src.tasks.manifesto.corpus_metrics import compute_corpus_pearson_r
from src.tasks.manifesto.dimensions import BENOIT_DIMENSIONS, PolicyDimension
from src.tasks.manifesto.dimension_scorer import DimensionScorer
from src.tasks.manifesto.joint_scorer import JointDimensionScorer
from src.tasks.manifesto.resume_utils import load_resume_rows

logger = logging.getLogger(__name__)
_DIM_FROM_NAME = {d.value: d for d in PolicyDimension}
_ORDER = [
    PolicyDimension.ECONOMIC, PolicyDimension.SOCIAL, PolicyDimension.IMMIGRATION,
    PolicyDimension.EU, PolicyDimension.ENVIRONMENT, PolicyDimension.DECENTRALIZATION,
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--ports", type=int, nargs="+", default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--mode", choices=["per-dim", "combined"], required=True)
    p.add_argument("--dimension", default=None, choices=sorted(_DIM_FROM_NAME),
                   help="Required for --mode per-dim.")
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Source dir with per_manifesto.jsonl.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write rescored per_manifesto.jsonl + report.json.")
    p.add_argument("--temperature", type=float, required=True)
    p.add_argument("--n-samples", type=int, required=True,
                   help="Number of scoring calls per summary to average.")
    p.add_argument("--max-workers", type=int, default=16,
                   help="Parallel scoring calls within one manifesto's N-sample loop.")
    import os
    env_cap = os.environ.get("MANIFESTO_MAX_TOKENS")
    p.add_argument("--max-tokens", type=int, default=int(env_cap) if env_cap else None)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _configure_lm(args: argparse.Namespace):
    lm_kwargs = {"model": args.model, "temperature": args.temperature, "cache": False}
    if args.max_tokens is not None:
        lm_kwargs["max_tokens"] = args.max_tokens
    if args.ports:
        lm = create_vllm_lm_multi(ports=args.ports, **lm_kwargs)
    else:
        lm = create_vllm_lm(port=args.port, **lm_kwargs)
    configure_dspy(lm=lm)
    return lm


def _read_rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        raise SystemExit(f"Input {path} not found.")
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def _mean_score(scorer, summary: str, dim_spec, n: int, pool: ThreadPoolExecutor
                ) -> tuple[Optional[float], list[Optional[float]]]:
    """Run scorer N times on `summary` in parallel; return (mean_score, per_sample_scores).
    NA samples are excluded from the mean; returns None if all NA or empty."""
    def _one(_):
        try:
            if isinstance(scorer, JointDimensionScorer):
                result = scorer(summary=summary, dimension_spec=dim_spec)
            else:
                result = scorer(summary=summary)
            s = result.get("score")
            return float(s) if s is not None else None
        except Exception as e:  # noqa: BLE001
            logger.warning("score call failed: %s", e)
            return None
    samples = list(pool.map(_one, range(n)))
    usable = [s for s in samples if s is not None]
    if not usable:
        return None, samples
    return sum(usable) / len(usable), samples


def _run_per_dim(args: argparse.Namespace) -> int:
    if args.dimension is None:
        raise SystemExit("--mode per-dim requires --dimension")
    dim = _DIM_FROM_NAME[args.dimension]
    spec = BENOIT_DIMENSIONS[dim]
    scorer = DimensionScorer(spec, use_cot=False)

    rows_in = _read_rows(args.input_dir / "per_manifesto.jsonl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "per_manifesto.jsonl"
    already, resuming = load_resume_rows(out_path, log_label=f"per-dim {dim.value}")
    rows_out: list[dict] = list(already.values())
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool, \
         out_path.open("a" if resuming else "w") as fp:
        for i, r in enumerate(rows_in):
            mid = r.get("manifesto_id")
            if mid is not None and str(mid) in already:
                continue
            summary = r.get("summary")
            expert = r.get("benoit_expert_mean")
            if not summary:
                continue
            mean_s, samples = _mean_score(scorer, summary, spec, args.n_samples, pool)
            r_out = dict(r)
            r_out["llm_score_1_7_T0"] = r.get("llm_score_1_7")  # preserve baseline
            r_out["llm_score_1_7"] = mean_s
            r_out["score_samples"] = samples
            r_out["rescore_T"] = args.temperature
            r_out["rescore_N"] = args.n_samples
            rows_out.append(r_out)
            fp.write(json.dumps(r_out, ensure_ascii=False) + "\n")
            fp.flush()
            if (i + 1) % 25 == 0:
                logger.info("[per-dim %s T%g N%d] %d/%d (%.1fs)",
                            dim.value, args.temperature, args.n_samples,
                            i + 1, len(rows_in), time.time() - t0)

    preds = [r.get("llm_score_1_7") for r in rows_out]
    truths = [r.get("benoit_expert_mean") for r in rows_out]
    try:
        rep = compute_corpus_pearson_r(preds, truths)
        logger.info("r=%+.3f n=%d CI[%+.3f,%+.3f] (%.0fs)",
                    rep.pearson_r, rep.n, rep.pearson_ci_low, rep.pearson_ci_high,
                    time.time() - t0)
        report = rep.as_dict()
    except ValueError as e:
        logger.warning("correlation failed: %s", e)
        report = {"pearson_r": None, "n": 0, "error": str(e)}

    (args.output_dir / "report.json").write_text(json.dumps({
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "per-dim",
            "dimension": dim.value,
            "temperature": args.temperature,
            "n_samples": args.n_samples,
            "input_dir": str(args.input_dir),
            "n_rows": len(rows_out),
            "elapsed_seconds": round(time.time() - t0, 1),
        },
        "report": report,
    }, indent=2))
    return 0


def _run_combined(args: argparse.Namespace) -> int:
    scorer = JointDimensionScorer(use_cot=False)
    rows_in = _read_rows(args.input_dir / "per_manifesto.jsonl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "per_manifesto.jsonl"
    already, resuming = load_resume_rows(out_path, log_label="combined")
    rows_out: list[dict] = list(already.values())
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool, \
         out_path.open("a" if resuming else "w") as fp:
        for i, r in enumerate(rows_in):
            mid = r.get("manifesto_id")
            if mid is not None and str(mid) in already:
                continue
            summary = r.get("summary")
            if not summary:
                continue
            new_preds = {}
            new_samples = {}
            for dim in _ORDER:
                spec = BENOIT_DIMENSIONS[dim]
                mean_s, samples = _mean_score(scorer, summary, spec, args.n_samples, pool)
                new_preds[dim.value] = mean_s
                new_samples[dim.value] = samples
            r_out = dict(r)
            r_out["predictions_T0"] = r.get("predictions")
            r_out["predictions"] = new_preds
            r_out["score_samples"] = new_samples
            r_out["rescore_T"] = args.temperature
            r_out["rescore_N"] = args.n_samples
            rows_out.append(r_out)
            fp.write(json.dumps(r_out, ensure_ascii=False) + "\n")
            fp.flush()
            if (i + 1) % 10 == 0:
                logger.info("[combined T%g N%d] %d/%d (%.1fs)",
                            args.temperature, args.n_samples,
                            i + 1, len(rows_in), time.time() - t0)

    per_dim = {}
    for dim in _ORDER:
        preds = [r.get("predictions", {}).get(dim.value) for r in rows_out]
        truths = [r.get("expert_means", {}).get(dim.value) for r in rows_out]
        try:
            rep = compute_corpus_pearson_r(preds, truths)
            per_dim[dim.value] = rep.as_dict()
        except ValueError:
            per_dim[dim.value] = {"pearson_r": None, "n": 0}

    macros = [v["pearson_r"] for v in per_dim.values() if v.get("pearson_r") is not None]
    macro = sum(macros) / len(macros) if macros else None

    (args.output_dir / "report.json").write_text(json.dumps({
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "combined",
            "temperature": args.temperature,
            "n_samples": args.n_samples,
            "input_dir": str(args.input_dir),
            "n_rows": len(rows_out),
            "elapsed_seconds": round(time.time() - t0, 1),
        },
        "per_dim": per_dim,
        "macro_pearson_r": macro,
    }, indent=2))
    logger.info("macro r=%+.3f (%.0fs)", macro if macro is not None else float("nan"),
                time.time() - t0)
    return 0


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    _configure_lm(args)
    if args.mode == "per-dim":
        return _run_per_dim(args)
    return _run_combined(args)


if __name__ == "__main__":
    sys.exit(main())
