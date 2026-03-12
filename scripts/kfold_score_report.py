#!/usr/bin/env python3
"""
Stratified, tail-weighted k-fold evaluation for *_score_report.jsonl artifacts.

This is a lightweight analysis utility (no LLM calls). It helps answer:
- Are we systematically worse in the tails (far from neutral)?
- How stable are metrics across folds if we stratify by score (or distance)?
- What do tail-weighted metrics look like (so optimization can target them)?

Example:
  ./venv/bin/python scripts/kfold_score_report.py \\
    --report outputs/manifesto_overnight_20260222_084204/test_score_report.jsonl \\
    --k 5 --stratify dist --weighting power --alpha 3.0 --gamma 2.0
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _as_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _quantile_edges(values: Sequence[float], bins: int) -> List[float]:
    if not values:
        return [0.0, 1.0]
    bins = max(1, int(bins))
    xs = sorted(float(v) for v in values)
    n = len(xs)
    edges: List[float] = []
    for i in range(bins + 1):
        q = i / bins
        idx = int(round(q * (n - 1)))
        idx = max(0, min(n - 1, idx))
        edges.append(xs[idx])
    for i in range(1, len(edges)):
        if edges[i] < edges[i - 1]:
            edges[i] = edges[i - 1]
    return edges


def _assign_bin(value: float, edges: Sequence[float]) -> int:
    if not edges or len(edges) < 2:
        return 0
    lo = 0
    hi = len(edges) - 2
    while lo <= hi:
        mid = (lo + hi) // 2
        if value < edges[mid + 1]:
            hi = mid - 1
        else:
            lo = mid + 1
    return max(0, min(len(edges) - 2, lo))


def _make_stratified_folds(
    strat_values: Sequence[float],
    *,
    k: int,
    bins: int,
    seed: int,
) -> List[List[int]]:
    k = max(2, int(k))
    bins = max(2, int(bins))
    edges = _quantile_edges(strat_values, bins=bins)

    by_bin: Dict[int, List[int]] = defaultdict(list)
    for idx, val in enumerate(strat_values):
        by_bin[_assign_bin(float(val), edges)].append(idx)

    rng = random.Random(int(seed))
    for bucket in by_bin.values():
        rng.shuffle(bucket)

    folds: List[List[int]] = [[] for _ in range(k)]
    fold_offset = 0
    for bin_idx in sorted(by_bin.keys()):
        bucket = by_bin[bin_idx]
        if not bucket:
            continue
        for j, idx in enumerate(bucket):
            folds[(fold_offset + j) % k].append(idx)
        fold_offset = (fold_offset + len(bucket)) % k

    for fold in folds:
        fold.sort()
    return folds


def _corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = sum((x - mx) ** 2 for x in xs)
    deny = sum((y - my) ** 2 for y in ys)
    if denx <= 0.0 or deny <= 0.0:
        return None
    return float(num / math.sqrt(denx * deny))


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> Optional[float]:
    if not values or not weights or len(values) != len(weights):
        return None
    total_w = float(sum(weights))
    if total_w <= 0.0:
        return None
    return float(sum(v * w for v, w in zip(values, weights)) / total_w)


def _weights_from_dist(
    dists: Sequence[float],
    *,
    weighting: str,
    alpha: float,
    gamma: float,
) -> List[float]:
    weighting = str(weighting or "none").strip().lower()
    if weighting == "none":
        return [1.0 for _ in dists]
    out: List[float] = []
    for dist in dists:
        scaled = max(0.0, min(1.0, float(dist) / 0.5))
        if weighting == "linear":
            out.append(1.0 + float(alpha) * scaled)
        elif weighting == "power":
            out.append(1.0 + float(alpha) * (scaled ** float(gamma)))
        else:
            out.append(1.0)
    return out


def _fold_metrics(
    preds: Sequence[float],
    acts: Sequence[float],
    *,
    neutral: float,
    weighting: str,
    alpha: float,
    gamma: float,
    tail_threshold: float,
) -> Dict[str, Any]:
    abs_err = [abs(p - a) for p, a in zip(preds, acts)]
    dist = [abs(a - neutral) for a in acts]
    weights = _weights_from_dist(dist, weighting=weighting, alpha=alpha, gamma=gamma)

    mae = float(sum(abs_err) / max(1, len(abs_err)))
    wmae = _weighted_mean(abs_err, weights)
    within10 = float(sum(1 for e in abs_err if e <= 0.10) / max(1, len(abs_err))) * 100.0
    within5 = float(sum(1 for e in abs_err if e <= 0.05) / max(1, len(abs_err))) * 100.0
    same_side = 0
    for pred, actual in zip(preds, acts):
        pred_delta = float(pred) - float(neutral)
        actual_delta = float(actual) - float(neutral)
        # Strict metric: exact-neutral predictions always count as wrong.
        if abs(pred_delta) <= 1e-9:
            continue
        if pred_delta * actual_delta > 0.0:
            same_side += 1
    same_side_pct = (float(same_side) / max(1, len(abs_err))) * 100.0
    corr_dist_err = _corr(dist, abs_err)

    tail_abs_err = [e for d, e in zip(dist, abs_err) if d >= float(tail_threshold)]
    tail_mae = float(sum(tail_abs_err) / len(tail_abs_err)) if tail_abs_err else None

    return {
        "n": len(abs_err),
        "mae": mae,
        "weighted_mae": wmae,
        "within_5pct": within5,
        "within_10pct": within10,
        "same_side_of_neutral_pct": same_side_pct,
        "corr_dist_err": corr_dist_err,
        "tail_threshold": float(tail_threshold),
        "tail_n": len(tail_abs_err),
        "tail_mae": tail_mae,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Tail-weighted stratified k-fold metrics for score_report.jsonl.")
    parser.add_argument("--report", type=Path, required=True, help="Path to *_score_report.jsonl")
    parser.add_argument("--k", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument(
        "--stratify",
        type=str,
        default="dist",
        choices=["dist", "actual"],
        help="Stratify folds by `dist=|actual-neutral|` or by `actual` (default: dist).",
    )
    parser.add_argument("--bins", type=int, default=10, help="Stratification bins (quantile edges, default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for folds (default: 42)")

    parser.add_argument("--neutral", type=float, default=0.5, help="Neutral point in normalized scale (default: 0.5)")
    parser.add_argument(
        "--weighting",
        type=str,
        default="none",
        choices=["none", "linear", "power"],
        help="Per-example error weight as function of |actual-neutral| (default: none).",
    )
    parser.add_argument("--alpha", type=float, default=2.0, help="Weight strength (default: 2.0)")
    parser.add_argument("--gamma", type=float, default=2.0, help="Power exponent for --weighting power (default: 2.0)")
    parser.add_argument(
        "--tail-threshold",
        type=float,
        default=0.15,
        help="Compute tail MAE for |actual-neutral| >= threshold (default: 0.15)",
    )
    args = parser.parse_args()

    rows = _load_jsonl(Path(args.report))
    parsed: List[Tuple[str, float, float]] = []
    for r in rows:
        doc_id = str(r.get("doc_id") or "").strip()
        p = _as_float(r.get("predicted"))
        a = _as_float(r.get("actual"))
        if not doc_id or p is None or a is None:
            continue
        parsed.append((doc_id, float(p), float(a)))

    if len(parsed) < 4:
        raise SystemExit("Not enough rows with doc_id/predicted/actual")

    preds_all = [p for _, p, _ in parsed]
    acts_all = [a for _, _, a in parsed]

    neutral = max(0.0, min(1.0, float(args.neutral)))
    strat_values = [abs(a - neutral) if args.stratify == "dist" else a for a in acts_all]

    folds = _make_stratified_folds(
        strat_values,
        k=int(args.k),
        bins=int(args.bins),
        seed=int(args.seed),
    )

    fold_summaries: List[Dict[str, Any]] = []
    for fold_idx, idxs in enumerate(folds):
        preds = [preds_all[j] for j in idxs]
        acts = [acts_all[j] for j in idxs]
        summary = _fold_metrics(
            preds,
            acts,
            neutral=neutral,
            weighting=str(args.weighting),
            alpha=float(args.alpha),
            gamma=float(args.gamma),
            tail_threshold=float(args.tail_threshold),
        )
        summary["fold"] = fold_idx
        fold_summaries.append(summary)

    def _avg(key: str) -> Optional[float]:
        vals = [s.get(key) for s in fold_summaries if s.get(key) is not None]
        if not vals:
            return None
        return float(sum(float(v) for v in vals) / len(vals))

    overall = _fold_metrics(
        preds_all,
        acts_all,
        neutral=neutral,
        weighting=str(args.weighting),
        alpha=float(args.alpha),
        gamma=float(args.gamma),
        tail_threshold=float(args.tail_threshold),
    )

    report_path = Path(args.report)
    print(
        f"report={report_path} rows={len(parsed)} k={len(folds)} stratify={args.stratify} bins={args.bins} seed={args.seed}"
    )
    print(
        f"weighting={args.weighting} neutral={neutral:.3f} alpha={float(args.alpha):g} gamma={float(args.gamma):g} tail_threshold={float(args.tail_threshold):.3f}"
    )
    print("")
    print("fold  n   mae    wmae   within10  same-side  tail_n  tail_mae  corr(dist,err)")
    print("----  --- ------ ------ --------  ---------  ------  --------  -------------")
    for s in fold_summaries:
        corr_val = s.get("corr_dist_err")
        corr_str = "n/a" if corr_val is None else f"{float(corr_val):+.3f}"
        tail_mae = s.get("tail_mae")
        tail_mae_str = "n/a" if tail_mae is None else f"{float(tail_mae):.4f}"
        print(
            f"{int(s['fold']):>4}  {int(s['n']):>3}  {float(s['mae']):.4f}  {float(s['weighted_mae']):.4f}  "
            f"{float(s['within_10pct']):>7.1f}%  {float(s['same_side_of_neutral_pct']):>8.1f}%  "
            f"{int(s['tail_n']):>6}  {tail_mae_str:>8}  {corr_str:>13}"
        )

    print("----  --- ------ ------ --------  ---------  ------  --------  -------------")
    avg_tail_mae = _avg("tail_mae")
    avg_tail_mae_str = "n/a" if avg_tail_mae is None else f"{float(avg_tail_mae):.4f}"
    avg_corr = _avg("corr_dist_err")
    avg_corr_str = "n/a" if avg_corr is None else f"{float(avg_corr):+.3f}"
    print(
        f"avg   {int(_avg('n') or 0):>3}  {float(_avg('mae') or 0.0):.4f}  {float(_avg('weighted_mae') or 0.0):.4f}  "
        f"{float(_avg('within_10pct') or 0.0):>7.1f}%  {float(_avg('same_side_of_neutral_pct') or 0.0):>8.1f}%  "
        f"{int(_avg('tail_n') or 0):>6}  {avg_tail_mae_str:>8}  {avg_corr_str:>13}"
    )
    print("")
    overall_tail_mae = overall.get("tail_mae")
    overall_tail_mae_str = "n/a" if overall_tail_mae is None else f"{float(overall_tail_mae):.4f}"
    print(
        f"overall n={overall['n']} mae={overall['mae']:.4f} weighted_mae={overall['weighted_mae']:.4f} "
        f"within10={overall['within_10pct']:.1f}% same-side={overall['same_side_of_neutral_pct']:.1f}% "
        f"tail(n={overall['tail_n']} mae={overall_tail_mae_str})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
