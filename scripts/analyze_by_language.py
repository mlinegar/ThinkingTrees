#!/usr/bin/env python3
"""
Post-hoc language breakdown of our full-pipeline results.

Reads existing per_manifesto.jsonl from phase0 full_pipeline and chunk_sweep
runs and regroups by (manifesto_country -> language). Reports Pearson r per
language per dimension so we can claim the method works across Benoit's
21-language corpus.

Usage:
    python scripts/analyze_by_language.py
    python scripts/analyze_by_language.py --dim environment
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

# MP country code -> (name, primary manifesto language)
_COUNTRY_LANG = {
    11: ("Sweden", "sv"), 12: ("Norway", "no"), 13: ("Denmark", "da"),
    14: ("Finland", "fi"), 21: ("Iceland", "is"), 22: ("Netherlands", "nl"),
    23: ("Belgium", "nl/fr"), 31: ("France", "fr"), 32: ("Italy", "it"),
    33: ("Spain", "es"), 34: ("Greece", "el"), 35: ("Portugal", "pt"),
    41: ("Germany", "de"), 42: ("Austria", "de"), 43: ("Switzerland", "de/fr/it"),
    51: ("UK", "en"), 53: ("Ireland", "en"), 54: ("Malta", "mt"),
    56: ("Cyprus", "el"), 61: ("USA", "en"), 62: ("Canada", "en/fr"),
    63: ("Australia", "en"), 64: ("New Zealand", "en"),
    81: ("Albania", "sq"), 82: ("Czech", "cs"), 83: ("Slovakia", "sk"),
    86: ("Hungary", "hu"), 87: ("Poland", "pl"), 88: ("Bulgaria", "bg"),
    92: ("Romania", "ro"), 93: ("Estonia", "et"), 94: ("Lithuania", "lt"),
    95: ("Latvia", "lv"), 96: ("Slovenia", "sl"), 97: ("Croatia", "hr"),
}


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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", default=None,
                   help="Restrict to one dimension (default: all 6).")
    p.add_argument("--min-n", type=int, default=5,
                   help="Drop language cells with fewer than this many mfestos.")
    p.add_argument("--out-md", type=Path, default=None)
    args = p.parse_args()

    ds = ManifestoDataset(data_dir="data/raw/manifesto_corpus_benoit", require_text=True)

    dims = [args.dim] if args.dim else _DIM_ORDER

    lines = []
    lines.append(f"# Pearson r by language (full-pipeline runs at chunk_chars=24K)\n")

    for dim in dims:
        path = project_root / "outputs" / "overnight_benoit" / "full_pipeline" / dim / "per_manifesto.jsonl"
        rows = list(_iter_rows(path))
        if not rows:
            lines.append(f"### {dim}: no data at {path}\n")
            continue
        # Group by language
        by_lang: dict[str, list[tuple[float, float]]] = {}
        for r in rows:
            mid = r.get("manifesto_id")
            if mid is None:
                continue
            sample = ds.get_sample(mid)
            if sample is None:
                continue
            country_info = _COUNTRY_LANG.get(int(sample.country_code))
            if country_info is None:
                continue
            lang = country_info[1]
            pred = r.get("llm_score_1_7")
            expert = r.get("benoit_expert_mean")
            if pred is None or expert is None:
                continue
            by_lang.setdefault(lang, []).append((float(pred), float(expert)))

        # Aggregate
        lines.append(f"## {dim}\n")
        lines.append("|language|n|Pearson r|")
        lines.append("|---|---:|---:|")
        all_pred, all_true = [], []
        ranked = []
        for lang, pairs in by_lang.items():
            if len(pairs) < args.min_n:
                continue
            preds = [p for p, _ in pairs]
            truths = [t for _, t in pairs]
            all_pred.extend(preds)
            all_true.extend(truths)
            try:
                rep = compute_corpus_pearson_r(preds, truths)
                ranked.append((lang, len(pairs), rep.pearson_r))
            except ValueError:
                continue
        ranked.sort(key=lambda x: -x[1])
        for lang, n, r in ranked:
            lines.append(f"|{lang}|{n}|{r:+.3f}|")
        if all_pred:
            rep_all = compute_corpus_pearson_r(all_pred, all_true)
            lines.append(f"|**all (pooled)**|**{rep_all.n}**|**{rep_all.pearson_r:+.3f}**|")
        lines.append("")

    out = "\n".join(lines) + "\n"
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(out)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
