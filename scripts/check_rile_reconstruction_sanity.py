#!/usr/bin/env python
"""Step 0 gate: gold local CMP codes must additively reconstruct global RILE.

Aggregates per-quasi-sentence gold codes per manifesto (the local signal),
rolls them up with `targets_from_counts`, and correlates the resulting root
RILE against the published MPDS `rile` (the global signal). This guards two
things before any GPU/LLM spend:

1. The reconstruction premise — global RILE is a near-exact additive rollup
   of local codes (Step 0, 2026-06-09: Pearson 0.9975 / MAE 0.49).
2. The normalization convention — `total_non_header` is the repo standard;
   the legacy all-quasi-sentence convention is reported alongside for
   comparison (0.9944 / 1.35). Re-run this after any change to
   `span_targets.py` / `rile_codes.py`.

Usage:
    ./venv/bin/python scripts/check_rile_reconstruction_sanity.py
    # report-only, no gate:
    ./venv/bin/python scripts/check_rile_reconstruction_sanity.py --min-pearson 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.tasks.manifesto.rile_reconstruction import (
    gold_counts_per_manifesto as _gold_counts_per_manifesto,
    pearson_or_nan as _pearson,
    published_rile as _published_rile,
)
from src.tasks.manifesto.span_targets import targets_from_counts

DEFAULT_CORPUS_CSV = (
    REPO_ROOT / "data" / "raw" / "manifesto_project_full" / "manifesto_corpus_df.csv"
)
DEFAULT_MPDS_CSV = (
    REPO_ROOT / "data" / "raw" / "manifesto_corpus_benoit" / "manifesto_maindataset.csv"
)



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-csv", type=Path, default=DEFAULT_CORPUS_CSV)
    parser.add_argument("--mpds-csv", type=Path, default=DEFAULT_MPDS_CSV)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument(
        "--min-pearson",
        type=float,
        default=0.99,
        help="Gate: fail unless the standard (non_header) convention reaches "
        "this Pearson vs published MPDS rile. Pass 0 to report only.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    gold = _gold_counts_per_manifesto(args.corpus_csv, args.chunksize)
    published = _published_rile(args.mpds_csv)

    results: dict[str, dict[str, float]] = {}
    for convention in ("non_header", "all"):
        pred: list[float] = []
        obs: list[float] = []
        for mid, counts in gold.items():
            if mid not in published:
                continue
            pred.append(
                float(targets_from_counts(counts, denominator=convention)["rile_raw"])
            )
            obs.append(published[mid])
        n = len(pred)
        pearson = _pearson(pred, obs) if n >= 2 else float("nan")
        mae = sum(abs(p - o) for p, o in zip(pred, obs)) / max(1, n)
        results[convention] = {"n": n, "pearson": pearson, "mae_rile_points": mae}

    print(f"coded manifestos with gold counts: {len(gold)}")
    print(f"{'convention':<14} {'n':>6} {'pearson':>10} {'MAE (RILE pts)':>16}")
    for convention, row in results.items():
        print(
            f"{convention:<14} {row['n']:>6} {row['pearson']:>10.4f} "
            f"{row['mae_rile_points']:>16.2f}"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json_out}")

    standard = results["non_header"]
    if standard["pearson"] < args.min_pearson:
        print(
            f"FAIL: non_header Pearson {standard['pearson']:.4f} "
            f"< gate {args.min_pearson}"
        )
        return 1
    print(f"PASS: non_header Pearson {standard['pearson']:.4f} >= {args.min_pearson}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
