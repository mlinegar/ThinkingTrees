#!/usr/bin/env python3
"""
Zero-inference sanity check: reproduce Benoit Figure 1 correlations from
their own replication archive. Confirms our expert-join + ensemble-mean +
Pearson-r pipeline matches theirs exactly, independent of any LLM we run
ourselves.

Published numbers (Benoit 2026 Figure 1, p. 8):
    Economic .87  Social .92  Immigration .89
    European Union .91  Environment .82  Decentralization .49

Usage:
    python scripts/reproduce_benoit_figure1.py [--kind reported|openweight|replication]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.tasks.manifesto.corpus_metrics import compute_corpus_pearson_r
from src.tasks.manifesto.dimensions import BENOIT_DIMENSIONS, PolicyDimension
from src.tasks.manifesto.expert_benchmarks import (
    benoit_ensemble_mean,
    load_benoit_expert_means,
    load_benoit_llm_scores,
)


PUBLISHED_FIGURE1 = {
    PolicyDimension.ECONOMIC: 0.87,
    PolicyDimension.SOCIAL: 0.92,
    PolicyDimension.IMMIGRATION: 0.89,
    PolicyDimension.EU: 0.91,
    PolicyDimension.ENVIRONMENT: 0.82,
    PolicyDimension.DECENTRALIZATION: 0.49,
}

_ORDER = [
    PolicyDimension.ECONOMIC,
    PolicyDimension.SOCIAL,
    PolicyDimension.IMMIGRATION,
    PolicyDimension.EU,
    PolicyDimension.ENVIRONMENT,
    PolicyDimension.DECENTRALIZATION,
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", choices=["reported", "openweight", "replication"], default="reported")
    p.add_argument("--dataverse-dir", type=Path, default=None)
    p.add_argument("--output-json", type=Path, default=None)
    args = p.parse_args()

    llm = load_benoit_llm_scores(kind=args.kind, dataverse_dir=args.dataverse_dir)
    ensemble = benoit_ensemble_mean(llm)

    results = []
    print(f"=== Benoit Figure 1 reproduction (kind={args.kind}) ===")
    print(f"{'Dimension':<20} {'Published':>10} {'Ours':>8} {'Δ':>7} {'n':>5} {'95% CI':>18}")

    for dim in _ORDER:
        code = BENOIT_DIMENSIONS[dim].benoit_issue_code
        experts = load_benoit_expert_means(dim, dataverse_dir=args.dataverse_dir)
        merged = ensemble[ensemble["issue"] == code].merge(
            experts[["manifesto", "expert_mean"]], on="manifesto", how="left"
        )
        report = compute_corpus_pearson_r(
            merged["score_llm_mean"].tolist(),
            merged["expert_mean"].tolist(),
        )
        published = PUBLISHED_FIGURE1[dim] if args.kind == "reported" else None
        delta = None if published is None else round(report.pearson_r - published, 3)

        row = {
            "dimension": dim.value,
            "issue_code": code,
            "published": published,
            "ours": round(report.pearson_r, 3),
            "delta": delta,
            "n": report.n,
            "ci_low": round(report.pearson_ci_low, 3),
            "ci_high": round(report.pearson_ci_high, 3),
        }
        results.append(row)
        pub = f"{published:.2f}" if published is not None else "  —  "
        dlt = f"{delta:+.3f}" if delta is not None else "   —  "
        print(
            f"{dim.value:<20} {pub:>10} {row['ours']:>8.3f} {dlt:>7} "
            f"{row['n']:>5} [{row['ci_low']:>5.2f}, {row['ci_high']:>5.2f}]"
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"kind": args.kind, "rows": results}, indent=2))
        print(f"\nWrote {args.output_json}")

    if args.kind == "reported":
        max_delta = max(abs(r["delta"]) for r in results if r["delta"] is not None)
        if max_delta > 0.01:
            print(f"\nWARN: max |delta| vs published = {max_delta:.3f} (threshold 0.01)")
            return 1
        print(f"\nOK: all deltas within 0.01 of published values (max |delta| = {max_delta:.3f}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
