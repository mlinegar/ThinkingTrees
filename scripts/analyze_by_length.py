#!/usr/bin/env python3
"""
Post-hoc manifesto-length bucket analysis.

Reads existing per_manifesto.jsonl from full-pipeline runs (per-dim and
combined) and recomputes Pearson r within 4 buckets of manifesto length:

    <20K chars  — short
    20–50K      — medium
    50–100K     — long
    >100K       — very long

Expected: flat baseline should crash on the longest bucket (can't process
the tail), while tree/concat should hold up. This is the ablation that
shows the tree earns its keep on long documents.

Usage:
    python scripts/analyze_by_length.py
    python scripts/analyze_by_length.py --out-md paper/ctreepo/tables/length_buckets.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.tasks.manifesto import ManifestoDataset
from src.tasks.manifesto.corpus_metrics import compute_corpus_pearson_r

_DIM_ORDER = ["economic", "social", "immigration", "eu", "environment", "decentralization"]
_BUCKETS = [
    ("<20K", 0, 20_000),
    ("20-50K", 20_000, 50_000),
    ("50-100K", 50_000, 100_000),
    (">100K", 100_000, 10**9),
]


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


def _length_bucket(chars: int) -> str:
    for label, lo, hi in _BUCKETS:
        if lo <= chars < hi:
            return label
    return ">100K"


def _load_sources(ds: ManifestoDataset) -> dict[str, list[tuple[str, Path, str]]]:
    """Return labeled list of (source_name, per_manifesto_path, schema)
    for per-dim full_pipeline and combined_pipeline runs."""
    sources = []
    # Per-dim full_pipeline (chunk=24K, n≈215, 6 dims)
    for dim in _DIM_ORDER:
        p = project_root / "outputs" / "overnight_benoit" / "full_pipeline" / dim / "per_manifesto.jsonl"
        if p.exists():
            sources.append((f"tree-{dim}", p, "per-dim"))
    # Flat × 24K
    for dim in _DIM_ORDER:
        p = project_root / "outputs" / "ablations" / "flat" / dim / "per_manifesto.jsonl"
        if p.exists():
            sources.append((f"flat-{dim}", p, "per-dim"))
    # Concat × 16K
    for dim in _DIM_ORDER:
        p = project_root / "outputs" / "ablations" / "concat" / dim / "per_manifesto.jsonl"
        if p.exists():
            sources.append((f"concat-{dim}", p, "per-dim"))
    # Combined pipeline (24K)
    p = project_root / "outputs" / "phase2" / "combined_pipeline" / "per_manifesto.jsonl"
    if p.exists():
        sources.append(("combined", p, "combined"))
    return sources


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-md", type=Path, default=None)
    args = p.parse_args()

    ds = ManifestoDataset(data_dir="data/raw/manifesto_corpus_benoit", require_text=True)
    text_len_cache: dict[str, int] = {}

    def _len_of(mid: str) -> int:
        if mid in text_len_cache:
            return text_len_cache[mid]
        s = ds.get_sample(mid)
        n = len(s.text) if (s and s.text) else 0
        text_len_cache[mid] = n
        return n

    sources = _load_sources(ds)
    lines: list[str] = []
    lines.append("# Pearson r by manifesto length bucket\n")
    lines.append("Buckets: <20K / 20-50K / 50-100K / >100K chars. "
                 "Tests whether the tree + summarization earn their keep on long manifestos "
                 "(where flat-baseline truncation loses the tail).\n")
    header = "|Source|" + "|".join(label for label, _, _ in _BUCKETS) + "|all|"
    sep = "|---|" + "|".join("---:" for _ in _BUCKETS) + "|---:|"

    # Separate per-dim and combined
    for schema in ("per-dim", "combined"):
        lines.append(f"## {'Per-dimension runs' if schema == 'per-dim' else 'Combined pipeline (per-dim cells)'}\n")
        lines.append(header)
        lines.append(sep)
        for src_name, src_path, src_schema in sources:
            if src_schema != schema:
                continue
            if schema == "per-dim":
                row_data = {label: [] for label, _, _ in _BUCKETS}
                all_pairs = []
                for r in _iter_rows(src_path):
                    mid = r.get("manifesto_id")
                    if not mid:
                        continue
                    pred = r.get("llm_score_1_7")
                    truth = r.get("benoit_expert_mean")
                    if pred is None or truth is None:
                        continue
                    bucket = _length_bucket(_len_of(mid))
                    row_data[bucket].append((float(pred), float(truth)))
                    all_pairs.append((float(pred), float(truth)))

                cells = []
                for label, _, _ in _BUCKETS:
                    pairs = row_data[label]
                    if len(pairs) < 4:
                        cells.append(f"— (n={len(pairs)})")
                        continue
                    preds = [p for p, _ in pairs]
                    truths = [t for _, t in pairs]
                    try:
                        rep = compute_corpus_pearson_r(preds, truths)
                        cells.append(f"{rep.pearson_r:+.2f} (n={rep.n})")
                    except ValueError:
                        cells.append("—")
                all_cell = "—"
                if len(all_pairs) >= 4:
                    try:
                        rep = compute_corpus_pearson_r(
                            [p for p, _ in all_pairs], [t for _, t in all_pairs]
                        )
                        all_cell = f"{rep.pearson_r:+.2f} (n={rep.n})"
                    except ValueError:
                        pass
                lines.append(f"|{src_name}|" + "|".join(cells) + f"|{all_cell}|")
            else:
                # Combined: has per-dim predictions + expert_means dicts;
                # show 6 rows (one per dim)
                for dim in _DIM_ORDER:
                    row_data = {label: [] for label, _, _ in _BUCKETS}
                    all_pairs = []
                    for r in _iter_rows(src_path):
                        mid = r.get("manifesto_id")
                        if not mid:
                            continue
                        pred = (r.get("predictions") or {}).get(dim)
                        truth = (r.get("expert_means") or {}).get(dim)
                        if pred is None or truth is None:
                            continue
                        bucket = _length_bucket(_len_of(mid))
                        row_data[bucket].append((float(pred), float(truth)))
                        all_pairs.append((float(pred), float(truth)))
                    cells = []
                    for label, _, _ in _BUCKETS:
                        pairs = row_data[label]
                        if len(pairs) < 3:
                            cells.append(f"— (n={len(pairs)})")
                            continue
                        try:
                            rep = compute_corpus_pearson_r(
                                [p for p, _ in pairs], [t for _, t in pairs]
                            )
                            cells.append(f"{rep.pearson_r:+.2f} (n={rep.n})")
                        except ValueError:
                            cells.append("—")
                    all_cell = "—"
                    if len(all_pairs) >= 4:
                        try:
                            rep = compute_corpus_pearson_r(
                                [p for p, _ in all_pairs], [t for _, t in all_pairs]
                            )
                            all_cell = f"{rep.pearson_r:+.2f} (n={rep.n})"
                        except ValueError:
                            pass
                    lines.append(f"|{src_name}-{dim}|" + "|".join(cells) + f"|{all_cell}|")
        lines.append("")

    out = "\n".join(lines) + "\n"
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(out)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
