#!/usr/bin/env python3
"""
Aggregate outputs/phase3/combined_c{chunk}/report.json into a
chunk_chars × dimension Pearson-r table.

This shows whether the combined pipeline (one shared summarizer with
JOINT_RUBRIC, one scorer for all 6 dims) preserves r across shrinking
leaf sizes — the cleanest C-TreePO oracle-preservation claim.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

_DIM_ORDER = ["economic", "social", "immigration", "eu", "environment", "decentralization"]
_DIR_RE = re.compile(r"^combined_c(\d+)(?:_.*)?$")

_BENOIT_FIG1 = {
    "economic": 0.87, "social": 0.92, "immigration": 0.89,
    "eu": 0.91, "environment": 0.82, "decentralization": 0.49,
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=project_root / "outputs" / "phase3")
    p.add_argument("--out-md", type=Path, default=None)
    args = p.parse_args()

    cells: dict[int, dict[str, dict]] = {}
    macros: dict[int, float] = {}
    for sub in sorted(args.root.iterdir() if args.root.exists() else []):
        if not sub.is_dir():
            continue
        m = _DIR_RE.match(sub.name)
        if not m:
            continue
        chunk = int(m.group(1))
        rp = sub / "report.json"
        if not rp.exists():
            continue
        try:
            doc = json.loads(rp.read_text())
        except json.JSONDecodeError:
            continue
        per_dim = doc.get("per_dim", {})
        cells[chunk] = per_dim
        macros[chunk] = doc.get("macro_pearson_r")

    if not cells:
        print("No combined_c* reports found yet.")
        return 0

    chunks = sorted(cells.keys(), reverse=True)

    def cell(c, d):
        rep = cells[c].get(d)
        if rep is None or rep.get("pearson_r") is None:
            return "—"
        r = rep["pearson_r"]; n = rep.get("n")
        return f"{r:+.3f} (n={n})"

    lines = ["# Phase 3 combined pipeline × chunk_chars (r per dimension)\n"]
    lines.append("One shared summarizer (JOINT_RUBRIC) + one scorer, all 6 dims from one summary. Benoit Fig 1 reference in the last row for context.\n")
    header = "|chunk_chars|" + "|".join(d.capitalize()[:13] for d in _DIM_ORDER) + "|Macro|"
    sep = "|---|" + "|".join("---:" for _ in _DIM_ORDER) + "|---:|"
    lines.append(header)
    lines.append(sep)
    for c in chunks:
        row = [f"{c:,}"]
        for d in _DIM_ORDER:
            row.append(cell(c, d))
        macro = macros.get(c)
        row.append(f"{macro:+.3f}" if macro is not None else "—")
        lines.append("|" + "|".join(row) + "|")
    # Benoit reference
    lines.append("|**Benoit Fig 1**|" + "|".join(f"{_BENOIT_FIG1[d]:.2f}" for d in _DIM_ORDER) + "|—|")

    out = "\n".join(lines) + "\n"
    print(out)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
