#!/usr/bin/env python3
"""
Secondary benchmark: Pearson r of our predictions vs Benoit's MP-derived
logit scores (Lowe et al. 2011 transformation of hand-coded quasi-sentence
counts). Benoit's own paper uses these as a secondary validation — the
"MP hand-coding" baseline they compare against in Table 7.

Sources the logit scores from data_mp.rda. Mapping to our 6 dims:
    economic       -> logplaneco   (public-services / taxation axis)
    social         -> loglibcons   (liberal/conservative social axis)
    immigration    -> logimmig
    eu             -> logeu
    environment    -> logenv
    decentralization -> logdecent

Usage:
    python scripts/analyze_vs_mp_logit.py
    python scripts/analyze_vs_mp_logit.py --out-md paper/ctreepo/tables/vs_mp_logit.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pyreadr

from src.tasks.manifesto.corpus_metrics import compute_corpus_pearson_r

_DIM_ORDER = ["economic", "social", "immigration", "eu", "environment", "decentralization"]
_DIM_TO_MP_LOGIT = {
    "economic": "logplaneco",
    "social": "loglibcons",
    "immigration": "logimmig",
    "eu": "logeu",
    "environment": "logenv",
    "decentralization": "logdecent",
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
    p.add_argument("--out-md", type=Path, default=None)
    args = p.parse_args()

    mp = pyreadr.read_r("data/examples/benoit_dataverse/data_mp.rda")["data_mp"]
    mp["_key"] = mp["party"].astype("Int64").astype(str) + "_" + mp["date"].astype("Int64").astype(str)

    lines: list[str] = []
    lines.append("# Pearson r vs MP-derived logit scores (Benoit's secondary benchmark)\n")
    lines.append("Compares our single-shot predictions against MP hand-coded quasi-sentence "
                 "counts transformed via Lowe et al. 2011 logit. This is the alternate "
                 "ground truth Benoit reports alongside expert surveys — shows what our "
                 "method captures vs what the MP coders saw in the same text.\n")

    header = "|Source|" + "|".join(d for d in _DIM_ORDER) + "|macro|"
    sep = "|---|" + "|".join("---:" for _ in _DIM_ORDER) + "|---:|"
    lines.append(header)
    lines.append(sep)

    def _run_source(label: str, base_path: Path, schema: str) -> list[str]:
        """Return one row of the table."""
        cells = []
        macro_r = []
        for dim in _DIM_ORDER:
            logit_col = _DIM_TO_MP_LOGIT[dim]
            mp_lookup = {}
            for r in mp[["_key", logit_col]].dropna().itertuples():
                mp_lookup[r._1] = float(getattr(r, logit_col))

            # Per-dim source: outputs/overnight_benoit/full_pipeline/{dim}/...
            # Combined source: outputs/phase2/combined_pipeline/...
            if schema == "per-dim":
                path = base_path / dim / "per_manifesto.jsonl"
            else:
                path = base_path / "per_manifesto.jsonl"
            preds, truths = [], []
            for row in _iter_rows(path):
                mid = row.get("manifesto_id")
                if mid is None or mid not in mp_lookup:
                    continue
                if schema == "per-dim":
                    pred = row.get("llm_score_1_7")
                else:
                    pred = (row.get("predictions") or {}).get(dim)
                if pred is None:
                    continue
                preds.append(float(pred))
                truths.append(mp_lookup[mid])

            if len(preds) < 4:
                cells.append(f"— (n={len(preds)})")
                continue
            try:
                rep = compute_corpus_pearson_r(preds, truths)
                cells.append(f"{rep.pearson_r:+.2f} (n={rep.n})")
                macro_r.append(rep.pearson_r)
            except ValueError:
                cells.append("—")

        if macro_r:
            cells.append(f"{sum(macro_r)/len(macro_r):+.3f}")
        else:
            cells.append("—")
        return cells

    # Per-dim full-pipeline (chunk=24K)
    cells = _run_source(
        "per-dim tree (24K)",
        project_root / "outputs" / "overnight_benoit" / "full_pipeline",
        "per-dim",
    )
    lines.append("|per-dim tree (24K)|" + "|".join(cells) + "|")

    # Flat (24K truncation)
    cells = _run_source(
        "flat (24K truncation)",
        project_root / "outputs" / "ablations" / "flat",
        "per-dim",
    )
    lines.append("|flat (24K trunc)|" + "|".join(cells) + "|")

    # Concat 16K
    cells = _run_source(
        "concat (16K)",
        project_root / "outputs" / "ablations" / "concat",
        "per-dim",
    )
    lines.append("|concat (16K)|" + "|".join(cells) + "|")

    # Combined pipeline (24K)
    cells = _run_source(
        "combined (24K)",
        project_root / "outputs" / "phase2" / "combined_pipeline",
        "combined",
    )
    lines.append("|combined (24K)|" + "|".join(cells) + "|")

    lines.append("")
    lines.append("Benoit Table 7 comparison context: MP hand-coded logit scores place only "
                 "38% of coalition positions inside member-party ranges (vs 64% for their "
                 "LLM ensemble) — MP and experts disagree for structural reasons, esp. on "
                 "Decentralization. Our r values here show what *of* the MP signal we capture.")

    out = "\n".join(lines) + "\n"
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(out)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
