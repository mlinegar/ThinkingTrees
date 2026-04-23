#!/usr/bin/env python3
"""
Score Benoit's anonymized summaries using the *exact* LangChain-style chat
protocol captured in data_masked.csv: raw SystemMessage (the per-dim rubric)
+ HumanMessage ("Analyze the following political text:\\n\\n<SUMMARY>"), with
the model responding with a bare integer (or "NA"). No DSPy wrapper, no
structured-output coercion.

This isolates one remaining explanation for the 0.07 macro gap between our
Gemma-3-27B scorer-only numbers (r=0.731) and Benoit's Table 6 Gemma-3 column
(r~0.79): whether DSPy's JSON-field wrapping changes how Gemma-3 reasons
about the rubric vs. Benoit's native two-message format.

Usage:
    python scripts/score_benoit_raw_prompt.py \\
        --port 8020 --dimension economic \\
        --output-dir outputs/gemma3/scorer_raw_benoit/economic
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.tasks.manifesto.corpus_metrics import compute_corpus_pearson_r
from src.tasks.manifesto.dimensions import PolicyDimension
from src.tasks.manifesto.benoit_scoring_contexts import get_benoit_scoring_context
from src.tasks.manifesto.expert_benchmarks import (
    load_benoit_expert_means,
    load_benoit_masked_summaries,
)
from src.tasks.manifesto.resume_utils import load_resume_rows

logger = logging.getLogger(__name__)

_DIM_FROM_NAME = {d.value: d for d in PolicyDimension}

_PUBLISHED = {
    PolicyDimension.ECONOMIC: 0.87,
    PolicyDimension.SOCIAL: 0.92,
    PolicyDimension.IMMIGRATION: 0.89,
    PolicyDimension.EU: 0.91,
    PolicyDimension.ENVIRONMENT: 0.82,
    PolicyDimension.DECENTRALIZATION: 0.49,
}

_HUMAN_TEMPLATE = "Analyze the following political text:\n\n{summary}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8020)
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--model", type=str, default=None,
                   help="Model id. If omitted, auto-discovered from /v1/models.")
    p.add_argument("--dimension", type=str, default="economic", choices=sorted(_DIM_FROM_NAME))
    p.add_argument("--max-n", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=8,
                   help="Cap on completion tokens. Benoit ran ~2; 8 gives headroom for 'NA'.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def _discover_model(client) -> str:
    models = client.models.list().data
    if not models:
        raise RuntimeError("vLLM server returned no models.")
    return models[0].id


_INT_RE = re.compile(r"([1-7])")


def _parse_response(text: str) -> tuple[float | None, str]:
    """Return (score, normalized_text). 'NA' (any case) -> None."""
    stripped = (text or "").strip()
    if not stripped:
        return None, ""
    upper = stripped.upper()
    if upper.startswith("NA") or upper == "N/A":
        return None, stripped
    m = _INT_RE.search(stripped)
    if m is None:
        return None, stripped
    return float(m.group(1)), stripped


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from openai import OpenAI

    dim = _DIM_FROM_NAME[args.dimension]
    system_rubric = get_benoit_scoring_context(dim)
    logger.info("Using Benoit exact rubric for %s (%d chars)", dim.value, len(system_rubric))

    logger.info("Loading Benoit anonymized summaries for dimension=%s", dim.value)
    summaries = load_benoit_masked_summaries(dimension=dim)
    logger.info("Loaded %d summaries", len(summaries))

    experts = load_benoit_expert_means(dim)
    experts_lookup = {
        str(row.manifesto).removesuffix(".txt"): float(row.expert_mean)
        for row in experts.itertuples()
    }
    summaries["expert_mean"] = summaries["manifesto_stem"].map(experts_lookup)
    scorable = summaries.dropna(subset=["expert_mean"]).reset_index(drop=True)
    logger.info("Summaries with matched expert means: %d", len(scorable))

    if args.max_n is not None and args.max_n < len(scorable):
        scorable = scorable.sample(n=args.max_n, random_state=0).reset_index(drop=True)
        logger.info("Subsampled to %d (seed=0)", len(scorable))

    base_url = f"http://{args.host}:{args.port}/v1"
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    model = args.model or _discover_model(client)
    logger.info("vLLM: %s  model=%s  T=%g  max_tokens=%d",
                base_url, model, args.temperature, args.max_tokens)

    per_path = args.output_dir / "per_summary.jsonl"
    already, resuming = load_resume_rows(per_path, key_field="manifesto_stem",
                                         log_label=f"raw-prompt {dim.value}")
    rows: list[dict] = list(already.values())
    t0 = time.time()

    with per_path.open("a" if resuming else "w") as fp:
        for i, src in enumerate(scorable.itertuples()):
            if src.manifesto_stem in already:
                continue
            messages = [
                {"role": "system", "content": system_rubric},
                {"role": "user", "content": _HUMAN_TEMPLATE.format(summary=src.summary)},
            ]
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    extra_body={"seed": 0, "top_p": 1.0},
                )
                raw = resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                logger.exception("API call failed on %s: %s", src.manifesto_stem, exc)
                raw = ""
            score, normalized = _parse_response(raw)
            row = {
                "manifesto_stem": src.manifesto_stem,
                "expert_mean": float(src.expert_mean),
                "benoit_score": float(src.benoit_score) if pd.notna(src.benoit_score) else None,
                "our_score": score,
                "our_is_na": score is None,
                "our_raw_response": normalized[:200],
            }
            rows.append(row)
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            fp.flush()
            if (i + 1) % 20 == 0:
                logger.info("Scored %d/%d (%.1fs)", i + 1, len(scorable), time.time() - t0)

    logger.info("Done scoring. Computing correlations.")

    def _r(preds, truths):
        return compute_corpus_pearson_r(preds, truths)

    ours_vs_expert = _r([r["our_score"] for r in rows], [r["expert_mean"] for r in rows])
    benoit_vs_expert = _r([r["benoit_score"] for r in rows], [r["expert_mean"] for r in rows])
    ours_vs_benoit = _r([r["our_score"] for r in rows], [r["benoit_score"] for r in rows])

    n_na = sum(1 for r in rows if r["our_is_na"])
    report = {
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "dimension": dim.value,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "prompt_style": "raw_langchain_benoit",
            "n_scored": len(rows),
            "n_na": n_na,
            "elapsed_seconds": round(time.time() - t0, 1),
        },
        "reference": {"benoit_figure1_proprietary_ensemble": _PUBLISHED[dim]},
        "ours_vs_expert": ours_vs_expert.as_dict(),
        "benoit_single_vs_expert": benoit_vs_expert.as_dict(),
        "ours_vs_benoit_single": ours_vs_benoit.as_dict(),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))

    def _line(label, rep):
        logger.info("%-32s r=%+.3f  n=%d  ci=[%+.3f,%+.3f]",
                    label, rep.pearson_r, rep.n, rep.pearson_ci_low, rep.pearson_ci_high)
    _line("Ours (raw-prompt) vs expert", ours_vs_expert)
    _line("Benoit single vs expert",    benoit_vs_expert)
    _line("Ours vs Benoit single",      ours_vs_benoit)
    logger.info("NA count: %d / %d  (%.1f%%)", n_na, len(rows), 100 * n_na / max(len(rows), 1))
    logger.info("Outputs in %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
