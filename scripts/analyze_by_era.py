#!/usr/bin/env python3
"""
Cross-era breakdown. Splits existing full-pipeline results by manifesto
election year and recomputes Pearson r per era bucket. Tests whether the
method holds up across 3 decades of political manifesto style.

Usage:
    python scripts/analyze_by_era.py
    python scripts/analyze_by_era.py --out-md paper/ctreepo/tables/era_breakdown.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.tasks.manifesto import ManifestoDataset
from src.tasks.manifesto.corpus_metrics import compute_corpus_pearson_r

_DIM_ORDER = ["economic", "social", "immigration", "eu", "environment", "decentralization"]
_ERAS = [("1989-1999", 1989, 2000), ("2000-2009", 2000, 2010), ("2010-2019", 2010, 2020)]


def _iter_rows(path: Path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _era_bucket(year: int) -> str:
    for label, lo, hi in _ERAS:
        if lo <= year < hi:
            return label
    return "other"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-md", type=Path, default=None)
    args = p.parse_args()

    lines: list[str] = []
    lines.append("# Pearson r by election era (per-dim full pipeline at chunk=24K)\n")
    lines.append("Splits the 215-mfesto test set by election-year era. Benoit's "
                 "data goes back to 1989; this checks whether scoring quality holds "
                 "up as political rhetoric evolved.\n")

    header = "|Dimension|" + "|".join(label for label, _, _ in _ERAS) + "|all|"
    sep = "|---|" + "|".join("---:" for _ in _ERAS) + "|---:|"
    lines.append(header)
    lines.append(sep)

    for dim in _DIM_ORDER:
        path = project_root / "outputs" / "overnight_benoit" / "full_pipeline" / dim / "per_manifesto.jsonl"
        if not path.exists():
            lines.append(f"|{dim}|" + "|".join("—" for _ in _ERAS) + "|—|")
            continue

        by_era = {label: [] for label, _, _ in _ERAS}
        all_pairs = []
        for r in _iter_rows(path):
            year = r.get("year")
            pred = r.get("llm_score_1_7")
            truth = r.get("benoit_expert_mean")
            if year is None or pred is None or truth is None:
                continue
            era = _era_bucket(int(year))
            if era in by_era:
                by_era[era].append((float(pred), float(truth)))
                all_pairs.append((float(pred), float(truth)))

        cells = []
        for label, _, _ in _ERAS:
            pairs = by_era[label]
            if len(pairs) < 4:
                cells.append(f"— (n={len(pairs)})")
                continue
            try:
                rep = compute_corpus_pearson_r(
                    [p for p, _ in pairs], [t for _, t in pairs]
                )
                cells.append(f"{rep.pearson_r:+.3f} (n={rep.n})")
            except ValueError:
                cells.append("—")
        all_cell = "—"
        if len(all_pairs) >= 4:
            try:
                rep = compute_corpus_pearson_r(
                    [p for p, _ in all_pairs], [t for _, t in all_pairs]
                )
                all_cell = f"{rep.pearson_r:+.3f} (n={rep.n})"
            except ValueError:
                pass
        lines.append(f"|{dim}|" + "|".join(cells) + f"|{all_cell}|")

    out = "\n".join(lines) + "\n"
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(out)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
