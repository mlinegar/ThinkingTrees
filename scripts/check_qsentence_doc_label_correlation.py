#!/usr/bin/env python3
"""How correlated are quasi-sentence labels with doc-level labels? (zero-inference)

Two questions, both answered from gold data only (no model calls):

A. Rollup validity — do per-quasi-sentence CMP codes, aggregated per document,
   reproduce the PUBLISHED doc-level MPDS measures?
   - rile vs published `rile` (Step 0 re-check; expect ~0.9975)
   - domain_1..7 shares vs MPDS `per###`-derived domain shares
   - pooled per-category check (gold share vs per###/100)

B. Construct ceiling — how much of the BENOIT EXPERT MEANS (six dimensions,
   doc level) can gold local codes explain at all? Per dimension: Pearson of
   each compact feature (rile + domain shares) with expert_mean_1_7, plus OLS
   in-sample R^2 and 5-fold CV R^2 from the 8-feature compact vector. This is
   the ceiling for ANY local->global method supervised by CMP codes; if it is
   low, the gap is construct mismatch, not method failure.

Caveats recorded in the output: gold domain shares use the `total_non_header`
denominator while MPDS per### percentages use MPDS's own denominator; the
MPDS side sums 3-digit per-columns only (4-digit handbook-v5 subcodes are
normalized into their parents on the gold side).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.tasks.manifesto.rile_reconstruction import (  # noqa: E402
    gold_counts_per_manifesto as _gold_counts_per_manifesto,
    pearson_or_nan as _pearson,
)
from src.tasks.manifesto.span_targets import (  # noqa: E402
    CMP_DOMAIN_KEYS,
    targets_from_counts,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_CORPUS = PROJECT_ROOT / "data/raw/manifesto_project_full/manifesto_corpus_df.csv"
DEFAULT_MPDS = PROJECT_ROOT / "data/raw/manifesto_corpus_benoit/manifesto_maindataset.csv"

BENOIT_DIMENSIONS = (
    "economic",
    "social",
    "immigration",
    "eu",
    "environment",
    "decentralization",
)
COMPACT_FEATURES = ("rile",) + tuple(CMP_DOMAIN_KEYS)


def _gold_targets(corpus_csv: Path) -> Dict[str, Dict[str, float]]:
    counts = _gold_counts_per_manifesto(corpus_csv, chunksize=500_000)
    out: Dict[str, Dict[str, float]] = {}
    for mid, counter in counts.items():
        target = targets_from_counts(counter)
        if not target.get("total_non_header"):
            continue
        flat = {
            k: float(v) for k, v in target.items() if isinstance(v, (int, float))
        }
        flat.update({k: float(v) for k, v in (target.get("compact") or {}).items()})
        out[str(mid)] = flat
    return out


def _mpds_frame(mpds_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(mpds_csv, low_memory=False)
    df = df.dropna(subset=["party", "date"])
    df["manifesto_id"] = df.apply(
        lambda r: f"{int(r['party'])}_{int(r['date'])}", axis=1
    )
    return df.set_index("manifesto_id")


def _per_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if re.fullmatch(r"per\d{3}", c)]


def _part_a(
    gold: Mapping[str, Mapping[str, float]], mpds: pd.DataFrame
) -> Dict[str, Any]:
    per_cols = _per_columns(mpds)
    shared = [mid for mid in gold if mid in mpds.index]
    LOGGER.info("Part A overlap: %d coded manifestos with MPDS rows", len(shared))

    result: Dict[str, Any] = {"n_overlap": len(shared)}

    rile_pairs = [
        (gold[mid]["rile_raw"], float(mpds.at[mid, "rile"]))
        for mid in shared
        if pd.notna(mpds.at[mid, "rile"])
    ]
    xs, ys = zip(*rile_pairs)
    result["rile"] = {
        "n": len(xs),
        "pearson": _pearson(list(xs), list(ys)),
        "mae": float(np.mean(np.abs(np.array(xs) - np.array(ys)))),
    }

    domains: Dict[str, Any] = {}
    for key in CMP_DOMAIN_KEYS:
        digit = key.split("_")[1]
        cols = [c for c in per_cols if c[3] == digit]
        pairs = []
        for mid in shared:
            published = mpds.loc[mid, cols].astype(float)
            if published.isna().all():
                continue
            pairs.append((gold[mid][key], float(published.fillna(0.0).sum()) / 100.0))
        if len(pairs) >= 4:
            xs, ys = zip(*pairs)
            domains[key] = {
                "n": len(xs),
                "n_per_columns": len(cols),
                "pearson": _pearson(list(xs), list(ys)),
                "mae": float(np.mean(np.abs(np.array(xs) - np.array(ys)))),
            }
    result["domains"] = domains

    per_category: List[Dict[str, Any]] = []
    for col in per_cols:
        code = col[3:]
        pairs = []
        for mid in shared:
            value = mpds.at[mid, col]
            if pd.isna(value):
                continue
            counter_share = gold[mid].get(f"_share_{code}")
            pairs.append((counter_share, float(value) / 100.0))
        # gold per-category shares are attached lazily below; skip if absent
        pairs = [(a, b) for a, b in pairs if a is not None]
        if len(pairs) >= 50:
            xs, ys = zip(*pairs)
            if float(np.std(xs)) > 0 and float(np.std(ys)) > 0:
                per_category.append(
                    {"code": code, "n": len(xs), "pearson": _pearson(list(xs), list(ys))}
                )
    if per_category:
        pearsons = [item["pearson"] for item in per_category]
        result["per_category"] = {
            "n_categories": len(per_category),
            "pearson_mean": float(np.mean(pearsons)),
            "pearson_median": float(np.median(pearsons)),
            "pearson_min": float(np.min(pearsons)),
            "worst5": sorted(per_category, key=lambda i: i["pearson"])[:5],
        }
    return result


def _attach_category_shares(
    corpus_csv: Path, gold: Dict[str, Dict[str, float]]
) -> None:
    """Add per-category gold shares (non_header denominator) as _share_<code>."""
    counts = _gold_counts_per_manifesto(corpus_csv, chunksize=500_000)
    for mid, counter in counts.items():
        target = gold.get(str(mid))
        if not target:
            continue
        denominator = float(target.get("total_non_header") or 0.0)
        if denominator <= 0:
            continue
        for code, count in counter.items():
            if isinstance(code, str) and code.isdigit() and len(code) == 3:
                target[f"_share_{code}"] = float(count) / denominator


def _part_b(
    gold: Mapping[str, Mapping[str, float]], mpds: pd.DataFrame
) -> Dict[str, Any]:
    from src.tasks.manifesto.dimensions import PolicyDimension
    from src.tasks.manifesto.expert_benchmarks import (
        load_benoit_expert_means,
        load_benoit_mp_crosswalk,
    )

    crosswalk = load_benoit_mp_crosswalk()

    # The crosswalk has (party, year) but manifesto_id needs the YYYYMM date;
    # resolve via MPDS, skipping ambiguous party-years (two elections in one
    # year for the same party).
    party_year_to_mid: Dict[Tuple[int, int], List[str]] = {}
    for mid, row in mpds.iterrows():
        try:
            party = int(row["party"])
            date = int(row["date"])
        except (TypeError, ValueError):
            continue
        party_year_to_mid.setdefault((party, date // 100), []).append(str(mid))

    def _mid_for(row: Mapping[str, Any]) -> Optional[str]:
        try:
            key = (int(row["party"]), int(row["year"]))
        except (TypeError, ValueError):
            return None
        candidates = party_year_to_mid.get(key) or []
        return candidates[0] if len(candidates) == 1 else None

    out: Dict[str, Any] = {}
    for dim_name in BENOIT_DIMENSIONS:
        dimension = PolicyDimension(dim_name)
        experts = load_benoit_expert_means(dimension)
        merged = experts.merge(crosswalk, on="manifesto", how="inner")
        rows: List[Tuple[List[float], float]] = []
        for _, row in merged.iterrows():
            mid = _mid_for(row)
            if mid is None or mid not in gold:
                continue
            expert = row.get("expert_mean_1_7")
            if pd.isna(expert):
                continue
            features = [float(gold[mid][feat]) for feat in COMPACT_FEATURES]
            rows.append((features, float(expert)))
        if len(rows) < 12:
            out[dim_name] = {"n": len(rows), "note": "insufficient overlap"}
            continue
        X = np.array([r[0] for r in rows])
        y = np.array([r[1] for r in rows])

        per_feature = {
            feat: _pearson(list(X[:, i]), list(y))
            for i, feat in enumerate(COMPACT_FEATURES)
            if float(np.std(X[:, i])) > 0
        }

        Xd = np.column_stack([np.ones(len(y)), X])
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        in_sample_r2 = 1.0 - float(
            np.sum((y - Xd @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
        )

        rng = np.random.default_rng(42)
        order = rng.permutation(len(y))
        folds = np.array_split(order, 5)
        sse, sst = 0.0, 0.0
        for fold in folds:
            mask = np.ones(len(y), dtype=bool)
            mask[fold] = False
            beta_cv, *_ = np.linalg.lstsq(Xd[mask], y[mask], rcond=None)
            preds = Xd[fold] @ beta_cv
            sse += float(np.sum((y[fold] - preds) ** 2))
            sst += float(np.sum((y[fold] - y[mask].mean()) ** 2))
        cv_r2 = 1.0 - sse / sst if sst > 0 else None

        out[dim_name] = {
            "n": len(y),
            "per_feature_pearson": {
                k: round(v, 4) for k, v in sorted(
                    per_feature.items(), key=lambda kv: -abs(kv[1])
                )
            },
            "ols_in_sample_r2": round(in_sample_r2, 4),
            "ols_cv5_r2": round(cv_r2, 4) if cv_r2 is not None else None,
        }
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-csv", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--mpds-csv", type=Path, default=DEFAULT_MPDS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "qsentence_doc_label_correlation",
    )
    parser.add_argument("--skip-per-category", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")

    LOGGER.info("Aggregating gold q-sentence codes per manifesto (chunked corpus read)...")
    gold = _gold_targets(args.corpus_csv)
    LOGGER.info("Gold targets for %d coded manifestos", len(gold))
    if not args.skip_per_category:
        LOGGER.info("Attaching per-category gold shares (second corpus pass)...")
        _attach_category_shares(args.corpus_csv, gold)

    mpds = _mpds_frame(args.mpds_csv)
    part_a = _part_a(gold, mpds)
    part_b = _part_b(gold, mpds)

    payload = {
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "corpus_csv": str(args.corpus_csv),
        "mpds_csv": str(args.mpds_csv),
        "conventions": {
            "gold_denominator": "total_non_header",
            "mpds_domain_share": "sum of 3-digit per### columns / 100 (MPDS denominator)",
            "expert_scale": "expert_mean_1_7",
        },
        "part_a_rollup_validity": part_a,
        "part_b_expert_construct_ceiling": part_b,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "correlation_report.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    LOGGER.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
