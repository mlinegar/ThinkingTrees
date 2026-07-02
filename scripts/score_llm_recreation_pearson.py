#!/usr/bin/env python3
"""Score the LLM-recreation A/B: g-stage (iter_02) root-pred-vs-expert Pearson r per
doc, TEST split, 3-seed mean+-pstdev, gold vs llm, for every dim under
outputs/llm_recreation/<dim>/{gold,llm}/seed_<sd>/fno/leafq<LEAF>/prediction_records/
iter_02_post_eval.jsonl. leaf=16 for Benoit dims, leaf=1 for rile.
"""
from __future__ import annotations
import json, statistics, sys
from pathlib import Path

ROOT = Path("outputs/llm_recreation")
SEEDS = ["101", "202", "303"]
# dim -> leaf dir
LEAF = {"rile": "leafq001"}  # everything else defaults to leafq016


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return 0.0  # constant predictor -> collapse
    return sxy / (sxx ** 0.5 * syy ** 0.5)


def _seed_r(dim, path, sd):
    leaf = LEAF.get(dim, "leafq016")
    f = ROOT / dim / path / f"seed_{sd}" / "fno" / leaf / "prediction_records" / "iter_02_post_eval.jsonl"
    if not f.exists():
        return None, None
    rows = [json.loads(l) for l in f.open() if l.strip()]
    rows = [r for r in rows if str(r.get("split", "")).lower() == "test"]
    if not rows:
        return None, None
    xs = [float(r["prediction"]) for r in rows]
    ys = [float(r["expert_score"]) for r in rows]
    return _pearson(xs, ys), len(rows)


def main():
    dims = sys.argv[1:] or sorted(p.name for p in ROOT.iterdir() if p.is_dir())
    print(f"{'dim':<16}{'path':<6}{'3-seed Pearson':<20}{'per-seed':<32}{'n_test'}")
    print("-" * 90)
    results = {}
    for dim in dims:
        results[dim] = {}
        ntest = None
        for path in ("gold", "llm"):
            rs, ns = [], []
            for sd in SEEDS:
                r, n = _seed_r(dim, path, sd)
                if r is not None:
                    rs.append(r); ns.append(n)
            if not rs:
                print(f"{dim:<16}{path:<6}(no results yet)")
                continue
            ntest = ns[0]
            m = statistics.mean(rs)
            s = statistics.pstdev(rs) if len(rs) > 1 else 0.0
            results[dim][path] = (m, s, rs, ntest)
            per = "[" + ", ".join(f"{r:+.3f}" for r in rs) + "]"
            print(f"{dim:<16}{path:<6}{m:+.3f} ± {s:.3f}{'':<7}{per:<32}{ntest}")
        # ratio line
        if "gold" in results[dim] and "llm" in results[dim]:
            g = results[dim]["gold"][0]; l = results[dim]["llm"][0]
            frac = (l / g * 100) if g != 0 else float("nan")
            print(f"{'':<16}{'->':<6}llm/gold = {frac:.0f}%  (delta {l-g:+.3f})")
        print()
    return results


if __name__ == "__main__":
    main()
