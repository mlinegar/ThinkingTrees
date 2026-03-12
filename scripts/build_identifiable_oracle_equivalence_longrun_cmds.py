#!/usr/bin/env python3
"""Build command packs for the identifiable oracle-equivalence long run.

This script emits:
- equivalence commands
- scale commands
- pilot commands (prefix subset for a ~20-minute smoke)
- plot/report/gate commands
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Dict, List


def _call(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def _read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [ln.rstrip("\n") for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _write(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _seed_text(n: int) -> str:
    return " ".join(str(i) for i in range(max(0, int(n))))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build long-run oracle-equivalence command lists.")
    p.add_argument("--run-id", type=str, default="20260303_longrun_equiv_v1")
    p.add_argument("--python-bin", type=str, default="venv/bin/python")
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--figures-root", type=Path, default=None)

    p.add_argument("--out-cmds-equiv", type=Path, default=None)
    p.add_argument("--out-cmds-scale", type=Path, default=None)
    p.add_argument("--out-cmds-pilot", type=Path, default=None)
    p.add_argument("--out-plot-cmds", type=Path, default=None)
    p.add_argument("--out-meta", type=Path, default=None)

    p.add_argument("--segment-test-docs", type=int, default=5000)
    p.add_argument("--ctree-test-books", type=int, default=5000)
    p.add_argument("--markov-test-docs", type=int, default=2000)
    p.add_argument("--markov-n-epochs", type=int, default=12)
    p.add_argument("--torch-threads", type=int, default=1)

    p.add_argument("--pilot-cmd-count", type=int, default=240)
    p.add_argument("--target-main-jobs", type=int, default=48)
    p.add_argument("--target-pilot-minutes", type=int, default=20)
    return p.parse_args()


def _keep_coupled_ctree(cmds: List[str]) -> List[str]:
    pat = re.compile(
        r"--eval-leaf-query-rate\s+([0-9eE+.\-]+)\s+--eval-internal-query-rate\s+([0-9eE+.\-]+)"
    )
    out: List[str] = []
    for cmd in cmds:
        m = pat.search(cmd)
        if m is None:
            continue
        ql = float(m.group(1))
        qi = float(m.group(2))
        if abs(float(ql) - float(qi)) <= 1e-12:
            out.append(cmd)
    return out


def main() -> int:
    args = _parse_args()
    run_id = str(args.run_id).strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    python_bin = str(args.python_bin)
    skip_flag = "--skip-existing" if bool(args.skip_existing) else "--no-skip-existing"

    output_root = args.output_root or Path(f"outputs/identifiable_zero_suite_{run_id}")
    figures_root = args.figures_root or (output_root / "figures")
    out_cmds_equiv = args.out_cmds_equiv or Path(f"logs/identifiable_zero_suite_{run_id}_cmds_equiv.txt")
    out_cmds_scale = args.out_cmds_scale or Path(f"logs/identifiable_zero_suite_{run_id}_cmds_scale.txt")
    out_cmds_pilot = args.out_cmds_pilot or Path(f"logs/identifiable_zero_suite_{run_id}_cmds_pilot.txt")
    out_plot_cmds = args.out_plot_cmds or Path(f"logs/identifiable_zero_suite_{run_id}_plot_cmds.txt")
    out_meta = args.out_meta or Path(f"logs/identifiable_zero_suite_{run_id}_meta.json")

    output_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    seeds = _seed_text(12)
    segment_train = "100 200 500 1000 2000 4000 8000 12000"
    segment_audit = "0.01 0.02 0.05 0.1 0.2 0.5 1.0"

    ctree_train = "256 512 1024 2048 4096"
    ctree_cal = "0.01 0.02 0.05 0.1"
    ctree_qinfer = "0 0.05 0.1 0.2 0.35 0.5 0.75 1.0"
    ctree_leaf_frontier = "0 0.1 0.25 0.5 0.75 1.0"
    ctree_internal_frontier = "0 0.05 0.1 0.2 0.35 0.5 0.75 1.0"

    markov_train_scale = "100 200 500 1000 2000 4000 8000"
    markov_train_equiv = "1000 4000 8000"
    markov_audit = "0 0.01 0.02 0.05 0.1 0.2 0.5 1.0"
    markov_lqr = "0 0.05 0.1 0.25 0.5 1.0"
    markov_qinfer = "0 0.1 0.25 0.5 0.75 1.0"

    tmp_segment = out_meta.with_name(out_meta.stem + "_tmp_segment_scale.txt")
    tmp_ctree_scale = out_meta.with_name(out_meta.stem + "_tmp_ctree_scale.txt")
    tmp_ctree_equiv_raw = out_meta.with_name(out_meta.stem + "_tmp_ctree_equiv_raw.txt")
    tmp_markov_scale = out_meta.with_name(out_meta.stem + "_tmp_markov_scale.txt")
    tmp_markov_equiv = out_meta.with_name(out_meta.stem + "_tmp_markov_equiv.txt")

    # Segment (scale pack).
    _call(
        [
            python_bin,
            "-u",
            "scripts/build_segment_lda_ops_weight_recovery_cmds.py",
            "--out-cmds",
            str(tmp_segment),
            "--output-root",
            str(output_root / "segment_lda_ops_weight_recovery" / "scale"),
            "--train-docs",
            segment_train,
            "--test-docs",
            str(int(args.segment_test_docs)),
            "--audit-fractions",
            segment_audit,
            "--topic-phi-docs",
            "0",
            "--topic-phi-estimators",
            "true embedding_spectral",
            "--topic-processes",
            "segments",
            "--lambda-multipliers",
            "1.0",
            "--seeds",
            seeds,
            "--topic-source",
            "infer",
            "--feature-inference",
            "hard",
            "--run-all-feature-modes",
            skip_flag,
        ]
    )

    # C-TreePO scale/frontier pack.
    _call(
        [
            python_bin,
            "-u",
            "scripts/build_segmented_lda_ctreepo_cmds.py",
            "--out-cmds",
            str(tmp_ctree_scale),
            "--output-root",
            str(output_root / "segmented_lda_ctreepo" / "scale"),
            "--train-docs",
            ctree_train,
            "--seeds",
            seeds,
            "--calibration-rates",
            ctree_cal,
            "--eval-leaf-rates",
            ctree_leaf_frontier,
            "--eval-internal-rates",
            ctree_internal_frontier,
            "--topic-phi-estimator",
            "spectral_numpy",
            "--topic-phi-docs",
            "0",
            "--n-books-test",
            str(int(args.ctree_test_books)),
            "--eval-internal-query-design",
            "risk",
            skip_flag,
        ]
    )

    # C-TreePO coupled q_infer equivalence pack.
    _call(
        [
            python_bin,
            "-u",
            "scripts/build_segmented_lda_ctreepo_cmds.py",
            "--out-cmds",
            str(tmp_ctree_equiv_raw),
            "--output-root",
            str(output_root / "segmented_lda_ctreepo" / "equivalence"),
            "--train-docs",
            ctree_train,
            "--seeds",
            seeds,
            "--calibration-rates",
            ctree_cal,
            "--eval-leaf-rates",
            ctree_qinfer,
            "--eval-internal-rates",
            ctree_qinfer,
            "--topic-phi-estimator",
            "spectral_numpy",
            "--topic-phi-docs",
            "0",
            "--n-books-test",
            str(int(args.ctree_test_books)),
            "--eval-internal-query-design",
            "risk",
            skip_flag,
        ]
    )

    # Markov scale pack (broader, lighter q_infer trials).
    _call(
        [
            python_bin,
            "-u",
            "scripts/build_markov_changepoint_ops_count_cmds.py",
            "--out-cmds",
            str(tmp_markov_scale),
            "--output-root",
            str(output_root / "markov_changepoint_ops_count" / "scale"),
            "--train-docs",
            markov_train_scale,
            "--test-docs",
            str(int(args.markov_test_docs)),
            "--audit-fractions",
            markov_audit,
            "--model-family",
            "additive neural",
            "--c3-audit-strategies",
            "uniform",
            "--leaf-query-rates",
            markov_lqr,
            "--include-root-query",
            "true false",
            "--root-weights",
            "1.0",
            "--schedule-consistency-weights",
            "0.0",
            "--seeds",
            seeds,
            "--n-epochs",
            str(int(args.markov_n_epochs)),
            "--device",
            "cpu",
            "--torch-threads",
            str(int(args.torch_threads)),
            "--eval-guidance-qs",
            markov_qinfer,
            "--eval-guidance-trials",
            "3",
            "--eval-guidance-include-root",
            skip_flag,
        ]
    )

    # Markov equivalence pack (narrower, higher q_infer trials).
    _call(
        [
            python_bin,
            "-u",
            "scripts/build_markov_changepoint_ops_count_cmds.py",
            "--out-cmds",
            str(tmp_markov_equiv),
            "--output-root",
            str(output_root / "markov_changepoint_ops_count" / "equivalence"),
            "--train-docs",
            markov_train_equiv,
            "--test-docs",
            str(int(args.markov_test_docs)),
            "--audit-fractions",
            markov_audit,
            "--model-family",
            "additive neural",
            "--c3-audit-strategies",
            "uniform",
            "--leaf-query-rates",
            markov_lqr,
            "--include-root-query",
            "true false",
            "--root-weights",
            "1.0",
            "--schedule-consistency-weights",
            "0.0",
            "--seeds",
            seeds,
            "--n-epochs",
            str(int(args.markov_n_epochs)),
            "--device",
            "cpu",
            "--torch-threads",
            str(int(args.torch_threads)),
            "--eval-guidance-qs",
            markov_qinfer,
            "--eval-guidance-trials",
            "8",
            "--eval-guidance-include-root",
            skip_flag,
        ]
    )

    segment_cmds = _read_lines(tmp_segment)
    ctree_scale_cmds = _read_lines(tmp_ctree_scale)
    ctree_equiv_cmds = _keep_coupled_ctree(_read_lines(tmp_ctree_equiv_raw))
    markov_scale_cmds = _read_lines(tmp_markov_scale)
    markov_equiv_cmds = _read_lines(tmp_markov_equiv)

    scale_cmds = [*segment_cmds, *ctree_scale_cmds, *markov_scale_cmds]
    equiv_cmds = [*ctree_equiv_cmds, *markov_equiv_cmds]
    pilot_cmds = [*equiv_cmds, *scale_cmds][: int(max(0, args.pilot_cmd_count))]

    _write(out_cmds_scale, scale_cmds)
    _write(out_cmds_equiv, equiv_cmds)
    _write(out_cmds_pilot, pilot_cmds)

    plot_cmds = [
        " ".join(
            [
                python_bin,
                "-u",
                "scripts/check_oracle_equivalence_invariants.py",
                "--output-root",
                f"'{output_root}'",
                "--ceiling-threshold",
                "1e-8",
                "--hard-guided-threshold",
                "1e-12",
                "--output-json",
                f"'{figures_root}/oracle_equivalence_invariants.json'",
            ]
        ),
        " ".join(
            [
                python_bin,
                "-u",
                "scripts/report_identifiable_zero_suite_publication_clean.py",
                "--output-root",
                f"'{output_root}'",
                "--emit-pdf",
            ]
        ),
    ]
    _write(out_plot_cmds, plot_cmds)

    meta: Dict[str, object] = {
        "run_id": run_id,
        "python_bin": python_bin,
        "skip_existing": bool(args.skip_existing),
        "output_root": str(output_root),
        "figures_root": str(figures_root),
        "target_policy": {
            "pilot_minutes": int(args.target_pilot_minutes),
            "main_jobs": int(args.target_main_jobs),
            "pilot_cmd_count": int(args.pilot_cmd_count),
        },
        "cmd_files": {
            "equiv": str(out_cmds_equiv),
            "scale": str(out_cmds_scale),
            "pilot": str(out_cmds_pilot),
            "plot": str(out_plot_cmds),
        },
        "counts": {
            "segment_scale": int(len(segment_cmds)),
            "ctree_scale": int(len(ctree_scale_cmds)),
            "ctree_equiv_coupled": int(len(ctree_equiv_cmds)),
            "markov_scale": int(len(markov_scale_cmds)),
            "markov_equiv": int(len(markov_equiv_cmds)),
            "scale_total": int(len(scale_cmds)),
            "equiv_total": int(len(equiv_cmds)),
            "pilot_total": int(len(pilot_cmds)),
        },
        "grid_spec": {
            "segment": {
                "train_docs": segment_train,
                "audit_fraction": segment_audit,
                "topic_phi_estimator": "true embedding_spectral",
                "lambda_multiplier": "1.0",
                "seeds": seeds,
            },
            "ctree": {
                "train_docs": ctree_train,
                "calibration_leaf_query_rate": ctree_cal,
                "equivalence_q_infer": ctree_qinfer,
                "frontier_leaf_rates": ctree_leaf_frontier,
                "frontier_internal_rates": ctree_internal_frontier,
                "seeds": seeds,
            },
            "markov": {
                "train_docs_scale": markov_train_scale,
                "train_docs_equiv": markov_train_equiv,
                "audit_fraction": markov_audit,
                "leaf_query_rate": markov_lqr,
                "include_root_query": "true false",
                "eval_guidance_qs": markov_qinfer,
                "eval_guidance_trials_scale": 3,
                "eval_guidance_trials_equiv": 8,
                "seeds": seeds,
            },
        },
    }
    out_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
