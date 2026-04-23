#!/usr/bin/env python3
"""
Aggregate outputs/chunk_sweep/{dim}_c{chars}/report.json into a single
table: rows = chunk_chars, cols = per-dim Pearson r.

Designed for the C-TreePO "chunk-invariance" figure: if r stays flat as
leaf size shrinks, the local laws preserve oracle info across compression
ratios.

Usage:
    python scripts/aggregate_chunk_sweep.py
    python scripts/aggregate_chunk_sweep.py --out-md outputs/chunk_sweep/summary.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


_DIM_ORDER = ["economic", "social", "immigration", "eu", "environment", "decentralization"]
_DIR_RE = re.compile(r"^([a-z_]+)_c(\d+)$")


def _load_r(report_path: Path) -> Optional[dict]:
    if not report_path.exists():
        return None
    try:
        doc = json.loads(report_path.read_text())
    except json.JSONDecodeError:
        return None
    return doc.get("report")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=project_root / "outputs" / "chunk_sweep")
    p.add_argument("--out-md", type=Path, default=None)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    if not args.root.exists():
        print(f"No chunk_sweep at {args.root}", file=sys.stderr)
        return 1

    # Index: (dim, chunk_chars) -> r-dict
    cells: dict[tuple[str, int], dict] = {}
    for sub in sorted(args.root.iterdir()):
        if not sub.is_dir():
            continue
        m = _DIR_RE.match(sub.name)
        if not m:
            continue
        dim, chars = m.group(1), int(m.group(2))
        r = _load_r(sub / "report.json")
        if r is not None:
            cells[(dim, chars)] = r

    if not cells:
        print("No completed chunk_sweep runs found yet.")
        return 0

    dims = sorted({d for d, _ in cells.keys()}, key=lambda x: _DIM_ORDER.index(x) if x in _DIM_ORDER else 999)
    chars_list = sorted({c for _, c in cells.keys()}, reverse=True)

    lines: list[str] = []
    lines.append(f"# Chunk-size sweep (r vs chunk_chars)\n")
    lines.append("chunk_chars ≈ 4 × input_tokens. Smaller chunks = deeper trees = more LLM calls per manifesto.\n")
    header = "|chunk_chars|" + "|".join(d.capitalize()[:13] for d in dims) + "|"
    sep = "|---|" + "|".join("---:" for _ in dims) + "|"
    lines.append(header)
    lines.append(sep)
    for c in chars_list:
        row = [f"{c:,}"]
        for d in dims:
            cell = cells.get((d, c))
            if cell is None:
                row.append("—")
                continue
            r = cell.get("pearson_r")
            n = cell.get("n")
            if r is None:
                row.append("—")
            else:
                row.append(f"{r:+.3f} (n={n})")
        lines.append("|" + "|".join(row) + "|")

    md_out = "\n".join(lines) + "\n"
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md_out)
    print(md_out)

    if args.out_json:
        json_payload = {
            "cells": [
                {"dimension": d, "chunk_chars": c, "report": r}
                for (d, c), r in sorted(cells.items())
            ]
        }
        args.out_json.write_text(json.dumps(json_payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
