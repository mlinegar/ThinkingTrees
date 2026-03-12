#!/usr/bin/env python3
"""Run SFM-style privacy-constrained sketch comparison experiments."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tree.private_sfm_comparison import (
    ComparisonRow,
    SFMComparisonConfig,
    run_sfm_style_comparison,
)


def _parse_int_csv(s: str) -> Tuple[int, ...]:
    out = tuple(int(float(x.strip())) for x in s.split(",") if x.strip())
    if len(out) == 0:
        raise ValueError("expected non-empty int CSV")
    return out


def _parse_float_csv(s: str) -> Tuple[float, ...]:
    out = tuple(float(x.strip()) for x in s.split(",") if x.strip())
    if len(out) == 0:
        raise ValueError("expected non-empty float CSV")
    return out


def _apply_preset(args: argparse.Namespace) -> None:
    preset = str(getattr(args, "preset", "default"))
    if preset == "default":
        return
    if preset == "small":
        args.n_values = "100,300,1000,3000"
        args.n_trials = 80
        args.merge_counts = "1,2,8"
        args.epsilons = "0.5,1,2"
        args.ridge_train_samples = 1000
        return
    if preset == "large":
        args.n_values = "100,300,1000,3000,10000,30000,100000,300000"
        args.n_trials = 800
        args.merge_counts = "1,2,8,32,64"
        args.epsilons = "0.25,0.5,1,2,4"
        args.ridge_train_samples = 12000
        return
    raise ValueError(f"unsupported preset: {preset!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare private/non-private distinct-count estimators under matched "
            "memory (B x P bits) and privacy budget epsilon."
        )
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=("default", "small", "large"),
        default="default",
        help="Optional experiment-size preset that overwrites core grid arguments.",
    )
    parser.add_argument("--n-values", type=str, default="100,300,1000,3000,10000,30000,100000")
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--merge-counts", type=str, default="1,2,8,32")
    parser.add_argument("--universe-size", type=int, default=20000000)
    parser.add_argument("--epsilons", type=str, default="0.5,1,2,4")
    parser.add_argument("--buckets", type=int, default=4096)
    parser.add_argument("--levels", type=int, default=24)
    parser.add_argument("--n-min-est", type=int, default=1)
    parser.add_argument("--n-max-est", type=int, default=2000000)
    parser.add_argument(
        "--no-hll",
        action="store_true",
        help="Disable non-private HLL baseline rows.",
    )
    parser.add_argument(
        "--no-ours-ridge",
        action="store_true",
        help="Disable learned ridge decoder rows.",
    )
    parser.add_argument("--ridge-train-samples", type=int, default=4000)
    parser.add_argument("--ridge-l2", type=float, default=1e-3)
    parser.add_argument(
        "--disable-theory-floor",
        action="store_true",
        help="Disable theory floor columns (HLL asymptotic RRMSE floor).",
    )
    parser.add_argument(
        "--enable-ipw",
        action="store_true",
        help="Enable IPW audit estimation over bounded scalar loss labels.",
    )
    parser.add_argument(
        "--ipw-audit-rates",
        type=str,
        default="0.1,0.25,0.5,1.0",
        help="CSV of audit inclusion-rate targets when IPW is enabled.",
    )
    parser.add_argument("--ipw-delta", type=float, default=0.05)
    parser.add_argument(
        "--ipw-sampling-scheme",
        type=str,
        choices=("uniform", "prediction_stratified"),
        default="prediction_stratified",
    )
    parser.add_argument("--ipw-propensity-floor", type=float, default=0.01)
    parser.add_argument(
        "--ipw-violation-abs-rel-threshold",
        type=float,
        default=0.10,
        help="Violation indicator threshold on absolute relative error.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--json-summary",
        type=str,
        default="outputs/sfm_comparison_summary.json",
    )
    parser.add_argument(
        "--csv-summary",
        type=str,
        default="outputs/sfm_comparison_summary.csv",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also emit JSON payload to stdout.",
    )
    return parser.parse_args()


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if len(rows) == 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _rows_to_dicts(rows: Sequence[ComparisonRow]) -> List[dict]:
    return [asdict(r) for r in rows]


def _fmt_opt(x: Optional[float], prec: int = 3) -> str:
    if x is None:
        return "na"
    return f"{float(x):.{int(prec)}f}"


def main() -> int:
    args = parse_args()
    _apply_preset(args)

    cfg = SFMComparisonConfig(
        n_values=_parse_int_csv(args.n_values),
        n_trials=int(args.n_trials),
        merge_counts=_parse_int_csv(args.merge_counts),
        universe_size=int(args.universe_size),
        epsilons=_parse_float_csv(args.epsilons),
        buckets=int(args.buckets),
        levels=int(args.levels),
        n_min_est=int(args.n_min_est),
        n_max_est=int(args.n_max_est),
        include_hll_non_private=not bool(args.no_hll),
        include_ours_ridge_sym=not bool(args.no_ours_ridge),
        ridge_train_samples=int(args.ridge_train_samples),
        ridge_l2=float(args.ridge_l2),
        include_theory_floor=not bool(args.disable_theory_floor),
        enable_ipw=bool(args.enable_ipw),
        ipw_audit_rates=_parse_float_csv(args.ipw_audit_rates),
        ipw_delta=float(args.ipw_delta),
        ipw_sampling_scheme=str(args.ipw_sampling_scheme),
        ipw_propensity_floor=float(args.ipw_propensity_floor),
        ipw_violation_abs_rel_threshold=float(args.ipw_violation_abs_rel_threshold),
        seed=int(args.seed),
    )

    summary = run_sfm_style_comparison(cfg)
    rows = _rows_to_dicts(summary.rows)

    payload = {
        "config": summary.config,
        "rows": rows,
    }
    jpath = Path(args.json_summary)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    cpath = Path(args.csv_summary)
    _write_csv(cpath, rows)

    if args.json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")

    if cfg.enable_ipw:
        print(
            "method | n | merge | eps | audit_rate | rrmse | gap_to_floor | "
            "mean_abs_rel | ipw_pref | true_pref | ipw_pref_ci | ipw_n_eff"
        )
    else:
        print(
            "method | n | merge | eps | eps_eff | rrmse | gap_to_floor | mean_abs_rel | "
            "rel_eff_vs_sfm_sym | ch_calib_l1"
        )
    key = lambda r: (
        int(r["merge_count"]),
        int(r["n"]),
        str(r["method"]),
        -1.0 if r["epsilon"] is None else float(r["epsilon"]),
        -1.0 if r["ipw_audit_rate"] is None else float(r["ipw_audit_rate"]),
    )
    for r in sorted(rows, key=key):
        if cfg.enable_ipw:
            print(
                f"{r['method']} | {int(r['n'])} | {int(r['merge_count'])} | "
                f"{_fmt_opt(r['epsilon'], 2)} | {_fmt_opt(r['ipw_audit_rate'], 2)} | "
                f"{float(r['rrmse']):.5f} | {_fmt_opt(r['rrmse_gap_to_theory_floor'], 5)} | "
                f"{float(r['mean_abs_rel_error']):.5f} | "
                f"{_fmt_opt(r['ipw_preference_loss'], 5)} | {_fmt_opt(r['true_preference_loss'], 5)} | "
                f"[{_fmt_opt(r['ipw_preference_ci_low'], 5)}, {_fmt_opt(r['ipw_preference_ci_high'], 5)}] | "
                f"{_fmt_opt(r['ipw_effective_sample_size'], 2)}"
            )
        else:
            print(
                f"{r['method']} | {int(r['n'])} | {int(r['merge_count'])} | "
                f"{_fmt_opt(r['epsilon'], 2)} | {_fmt_opt(r['epsilon_effective'], 3)} | "
                f"{float(r['rrmse']):.5f} | {_fmt_opt(r['rrmse_gap_to_theory_floor'], 5)} | "
                f"{float(r['mean_abs_rel_error']):.5f} | "
                f"{_fmt_opt(r['rel_eff_vs_sfm_sym'], 3)} | {_fmt_opt(r['channel_calibration_l1'], 4)}"
            )

    print(f"wrote_json | {jpath}")
    print(f"wrote_csv | {cpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
