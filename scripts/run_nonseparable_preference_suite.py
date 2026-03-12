#!/usr/bin/env python3
"""Run non-separable preference DGP separation suite."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tree.nonseparable_preference_suite import (  # noqa: E402
    NonseparableSuiteConfig,
    run_nonseparable_preference_suite,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run non-separable preference separation suite.")
    p.add_argument("--n-replicates", type=int, default=80)
    p.add_argument("--n-pairs-per-replicate", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--beta", type=float, default=3.0)
    p.add_argument("--and-left-threshold", type=int, default=3)
    p.add_argument("--and-right-threshold", type=int, default=3)
    p.add_argument("--and-count-max", type=int, default=7)
    p.add_argument("--dgp2-vocab-size", type=int, default=6)
    p.add_argument("--dgp2-seq-len", type=int, default=24)
    p.add_argument("--dgp2-lambda", type=float, default=2.0)
    p.add_argument("--hard-regime", action="store_true")
    p.add_argument("--effect-gate", type=float, default=0.05)
    p.add_argument("--strong-effect-gate", type=float, default=0.10)
    p.add_argument("--bound-tolerance", type=float, default=0.02)
    p.add_argument(
        "--json-summary",
        type=str,
        default="outputs/nonseparable_preference_suite_summary.json",
    )
    p.add_argument(
        "--csv-summary",
        type=str,
        default="outputs/nonseparable_preference_suite_summary.csv",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON summary to stdout.")
    return p.parse_args()


def _write_csv(path: Path, rows: List[dict]) -> None:
    if len(rows) == 0:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    cfg = NonseparableSuiteConfig(
        n_replicates=int(args.n_replicates),
        n_pairs_per_replicate=int(args.n_pairs_per_replicate),
        seed=int(args.seed),
        beta=float(args.beta),
        and_left_threshold=int(args.and_left_threshold),
        and_right_threshold=int(args.and_right_threshold),
        and_count_max=int(args.and_count_max),
        dgp2_vocab_size=int(args.dgp2_vocab_size),
        dgp2_seq_len=int(args.dgp2_seq_len),
        dgp2_lambda=float(args.dgp2_lambda),
        hard_regime=bool(args.hard_regime),
        effect_gate=float(args.effect_gate),
        strong_effect_gate=float(args.strong_effect_gate),
        bound_tolerance=float(args.bound_tolerance),
    )
    result = run_nonseparable_preference_suite(cfg)
    payload = result.to_dict()

    rows: List[dict] = []
    for dgp in payload["dgps"]:
        dgp_name = str(dgp["name"])
        for arm in dgp["arms"]:
            rows.append({"row_type": "arm", "dgp": dgp_name, **arm})
        for sep in dgp["separation_checks"]:
            rows.append({"row_type": "separation", "dgp": dgp_name, **sep})

    json_path = Path(args.json_summary)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = Path(args.csv_summary)
    _write_csv(csv_path, rows)

    print(f"wrote_json | {json_path}")
    print(f"wrote_csv | {csv_path}")
    for dgp in payload["dgps"]:
        name = str(dgp["name"])
        strong = int(bool(dgp["strong_separation_pass"]))
        flagged = len(dgp["flagged_cells"])
        print(f"dgp={name} | strong_separation={strong} | flagged_cells={flagged}")
    if bool(args.json):
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
