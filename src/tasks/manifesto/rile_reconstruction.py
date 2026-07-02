from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.tasks.manifesto.span_targets import normalize_cmp_code


def gold_counts_per_manifesto(corpus_csv: str | Path, chunksize: int) -> dict[str, Counter]:
    import pandas as pd

    counts: dict[str, Counter] = {}
    rows_seen: Counter = Counter()
    for chunk in pd.read_csv(
        corpus_csv,
        usecols=["manifesto_id", "cmp_code"],
        chunksize=chunksize,
        low_memory=False,
    ):
        for mid, code in zip(chunk["manifesto_id"], chunk["cmp_code"]):
            mid = str(mid)
            rows_seen[mid] += 1
            normalized = normalize_cmp_code(code)
            if normalized is None:
                continue
            counts.setdefault(mid, Counter())[normalized] += 1
    return {mid: counter for mid, counter in counts.items() if rows_seen[mid] > 1}


def published_rile(mpds_csv: str | Path) -> dict[str, float]:
    import pandas as pd

    df = pd.read_csv(mpds_csv, usecols=["party", "date", "rile"])
    df = df.dropna(subset=["party", "date", "rile"])
    return {
        f"{int(party)}_{int(date)}": float(rile)
        for party, date, rile in zip(df["party"], df["date"], df["rile"])
    }


def pearson_or_nan(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2 or len(ys) != n:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return cov / (vx**0.5 * vy**0.5)


__all__ = ["gold_counts_per_manifesto", "pearson_or_nan", "published_rile"]
