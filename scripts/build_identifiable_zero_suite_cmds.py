#!/usr/bin/env python3
"""Build command lists for an identifiable-only simulation suite.

Goal:
- keep only families/settings where zero-error oracle ceilings are attainable,
- focus on Segment-LDA + Segmented-LDA C-TreePO,
- optionally include Markov in additive family only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Dict, List


def _call_builder(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def _read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _seed_str(n: int) -> str:
    return " ".join(str(i) for i in range(max(0, int(n))))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build identifiable-only simulation command lists.")
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--profile", choices=["smoke", "paper", "walk_long"], default="walk_long")
    p.add_argument("--python-bin", type=str, default="venv/bin/python")

    p.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Default: outputs/identifiable_zero_suite_<run_id>",
    )
    p.add_argument(
        "--figures-root",
        type=str,
        default="",
        help="Default: <output-root>/figures",
    )
    p.add_argument("--out-cmds", type=str, default="")
    p.add_argument("--out-plot-cmds", type=str, default="")
    p.add_argument("--out-meta", type=str, default="")

    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--include-markov",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, include Markov additive-family runs.",
    )
    p.add_argument(
        "--include-embedding-estimator",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, include Segment-LDA estimator=embedding_spectral in addition to true.",
    )

    p.add_argument("--segment-test-docs", type=int, default=4000)
    p.add_argument("--ctree-test-books", type=int, default=4000)
    p.add_argument("--markov-test-docs", type=int, default=2000)
    p.add_argument("--markov-n-epochs", type=int, default=10)
    p.add_argument("--torch-threads", type=int, default=1)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_id = str(args.run_id).strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    output_root = (
        Path(args.output_root)
        if str(args.output_root).strip()
        else Path(f"outputs/identifiable_zero_suite_{run_id}")
    )
    figures_root = (
        Path(args.figures_root)
        if str(args.figures_root).strip()
        else (output_root / "figures")
    )
    out_cmds = (
        Path(args.out_cmds) if str(args.out_cmds).strip() else Path(f"logs/identifiable_zero_suite_{run_id}_cmds.txt")
    )
    out_plot_cmds = (
        Path(args.out_plot_cmds)
        if str(args.out_plot_cmds).strip()
        else Path(f"logs/identifiable_zero_suite_{run_id}_plot_cmds.txt")
    )
    out_meta = (
        Path(args.out_meta) if str(args.out_meta).strip() else Path(f"logs/identifiable_zero_suite_{run_id}_meta.json")
    )

    python_bin = str(args.python_bin)
    profile = str(args.profile)
    skip_flag = "--skip-existing" if bool(args.skip_existing) else "--no-skip-existing"

    output_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)
    out_cmds.parent.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Profile grids
    # ----------------------------
    if profile == "smoke":
        seg_train = "200 500"
        seg_audit = "0.1 0.2 0.5 1.0"
        seg_lam = "0 1.0"
        seg_seeds = _seed_str(2)

        ctree_train = "128 256"
        ctree_cal = "0 0.1 1.0"
        ctree_leaf = "0 1.0"
        ctree_int = "0 0.5 1.0"
        ctree_seeds = _seed_str(2)

        markov_train = "200 500"
        markov_audit = "0.1 0.2 0.5 1.0"
        markov_seeds = _seed_str(2)
    elif profile == "paper":
        seg_train = "100 200 500 1000 2000"
        seg_audit = "0.02 0.05 0.1 0.2 0.5 1.0"
        seg_lam = "0 0.25 1.0"
        seg_seeds = _seed_str(8)

        ctree_train = "64 128 256 512 1024"
        ctree_cal = "0 0.02 0.05 0.1 0.25 0.5 1.0"
        ctree_leaf = "0 0.5 1.0"
        ctree_int = "0 0.05 0.1 0.25 0.5 1.0"
        ctree_seeds = _seed_str(8)

        markov_train = "100 200 500 1000 2000"
        markov_audit = "0.02 0.05 0.1 0.2 0.5 1.0"
        markov_seeds = _seed_str(8)
    else:  # walk_long
        seg_train = "100 200 500 1000 2000 4000"
        seg_audit = "0.01 0.02 0.05 0.1 0.2 0.5 1.0"
        seg_lam = "0 0.25 1.0"
        seg_seeds = _seed_str(16)

        ctree_train = "64 128 256 512 1024"
        ctree_cal = "0 0.02 0.05 0.1 0.25 0.5 1.0"
        ctree_leaf = "0 0.25 0.5 1.0"
        ctree_int = "0 0.05 0.1 0.25 0.5 1.0"
        ctree_seeds = _seed_str(16)

        markov_train = "100 200 500 1000 2000 4000"
        markov_audit = "0.01 0.02 0.05 0.1 0.2 0.5 1.0"
        markov_seeds = _seed_str(16)
    ctree_focus_train = max(int(x) for x in str(ctree_train).split())

    # ----------------------------
    # Segment-LDA OPS (identifiable estimators only)
    # ----------------------------
    seg_out = output_root / "segment_lda_ops_weight_recovery"
    seg_cmds = out_cmds.with_name(out_cmds.stem + "_segment_lda_ops.txt")
    seg_estimators = ["true"]
    if bool(args.include_embedding_estimator):
        seg_estimators.append("embedding_spectral")
    seg_estimators_str = " ".join(seg_estimators)

    _call_builder(
        [
            python_bin,
            "-u",
            "scripts/build_segment_lda_ops_weight_recovery_cmds.py",
            "--out-cmds",
            str(seg_cmds),
            "--output-root",
            str(seg_out),
            "--train-docs",
            seg_train,
            "--test-docs",
            str(int(args.segment_test_docs)),
            "--audit-fractions",
            seg_audit,
            "--topic-phi-docs",
            "0",
            "--topic-phi-estimators",
            seg_estimators_str,
            "--topic-processes",
            "segments",
            "--lambda-multipliers",
            seg_lam,
            "--seeds",
            seg_seeds,
            "--topic-source",
            "infer",
            "--feature-inference",
            "hard",
            "--run-all-feature-modes",
            skip_flag,
        ]
    )

    # ----------------------------
    # Segmented-LDA C-TreePO
    # ----------------------------
    ctree_out = output_root / "segmented_lda_ctreepo"
    ctree_cmds = out_cmds.with_name(out_cmds.stem + "_segmented_lda_ctreepo.txt")

    _call_builder(
        [
            python_bin,
            "-u",
            "scripts/build_segmented_lda_ctreepo_cmds.py",
            "--out-cmds",
            str(ctree_cmds),
            "--output-root",
            str(ctree_out),
            "--train-docs",
            ctree_train,
            "--seeds",
            ctree_seeds,
            "--calibration-rates",
            ctree_cal,
            "--eval-leaf-rates",
            ctree_leaf,
            "--eval-internal-rates",
            ctree_int,
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

    # ----------------------------
    # Optional Markov additive-only
    # ----------------------------
    cmd_sources: Dict[str, Path] = {
        "segment_lda_ops": seg_cmds,
        "segmented_lda_ctreepo": ctree_cmds,
    }
    markov_out = output_root / "markov_changepoint_ops_count"
    if bool(args.include_markov):
        markov_cmds = out_cmds.with_name(out_cmds.stem + "_markov_additive.txt")
        _call_builder(
            [
                python_bin,
                "-u",
                "scripts/build_markov_changepoint_ops_count_cmds.py",
                "--out-cmds",
                str(markov_cmds),
                "--output-root",
                str(markov_out),
                "--train-docs",
                markov_train,
                "--test-docs",
                str(int(args.markov_test_docs)),
                "--audit-fractions",
                markov_audit,
                "--model-family",
                "additive",
                "--c3-audit-strategies",
                "uniform",
                "--leaf-query-rates",
                "1.0",
                "--root-weights",
                "1.0",
                "--schedule-consistency-weights",
                "0.0",
                "--seeds",
                markov_seeds,
                "--n-epochs",
                str(int(args.markov_n_epochs)),
                "--device",
                "cpu",
                "--torch-threads",
                str(int(args.torch_threads)),
                skip_flag,
            ]
        )
        cmd_sources["markov_additive"] = markov_cmds

    # ----------------------------
    # Merge sim commands
    # ----------------------------
    all_cmds: List[str] = []
    counts: Dict[str, int] = {}
    for key, path in cmd_sources.items():
        lines = _read_lines(path)
        counts[key] = int(len(lines))
        all_cmds.extend(lines)
    _write_text(out_cmds, "\n".join(all_cmds) + ("\n" if all_cmds else ""))

    # ----------------------------
    # Plot/report commands
    # ----------------------------
    plot_cmds: List[str] = []

    plot_cmds.append(
        " ".join(
            [
                python_bin,
                "-u",
                "scripts/plot_segment_lda_ops_weight_recovery_grid.py",
                "--input-glob",
                f"'{seg_out}/**/*seed_*.json'",
                "--audit-strategy",
                "random",
                "--topic-phi-estimator",
                "true",
                "--output-figure",
                f"'{figures_root}/segment_lda_ops_weight_recovery_grid_true.png'",
                "--output-json",
                f"'{figures_root}/segment_lda_ops_weight_recovery_grid_true_report.json'",
            ]
        )
    )
    plot_cmds.append(
        " ".join(
            [
                python_bin,
                "-u",
                "scripts/plot_segment_lda_oracle_gap_focus.py",
                "--input-glob",
                f"'{seg_out}/**/*seed_*.json'",
                "--topic-phi-estimator",
                "true",
                "--aggregate",
                "median",
                "--output-figure",
                f"'{figures_root}/segment_lda_oracle_gap_focus_true.png'",
                "--output-json",
                f"'{figures_root}/segment_lda_oracle_gap_focus_true_report.json'",
            ]
        )
    )
    plot_cmds.append(
        " ".join(
            [
                python_bin,
                "-u",
                "scripts/plot_segment_lda_ops_weight_recovery_ceilings.py",
                "--input-glob",
                f"'{seg_out}/**/*seed_*.json'",
                "--audit-strategy",
                "random",
                "--topic-phi-estimator",
                "true",
                "--aggregate",
                "median",
                "--band",
                "p10_p90",
                "--output-figure",
                f"'{figures_root}/segment_lda_ops_weight_recovery_ceilings_true.png'",
                "--output-json",
                f"'{figures_root}/segment_lda_ops_weight_recovery_ceilings_true_report.json'",
            ]
        )
    )
    plot_cmds.append(
        " ".join(
            [
                python_bin,
                "-u",
                "scripts/plot_ctreepo_guidance_frontier.py",
                "--input-glob",
                f"'{ctree_out}/**/*.json'",
                "--train-docs",
                str(int(ctree_focus_train)),
                "--aggregate",
                "median",
                "--output-figure",
                f"'{figures_root}/ctreepo_guidance_frontier_focus_train{int(ctree_focus_train)}.png'",
                "--output-json",
                f"'{figures_root}/ctreepo_guidance_frontier_focus_train{int(ctree_focus_train)}_report.json'",
            ]
        )
    )
    if bool(args.include_embedding_estimator):
        plot_cmds.append(
            " ".join(
                [
                    python_bin,
                    "-u",
                    "scripts/plot_segment_lda_ops_weight_recovery_ceilings.py",
                    "--input-glob",
                    f"'{seg_out}/**/*seed_*.json'",
                    "--audit-strategy",
                    "random",
                    "--topic-phi-estimator",
                    "embedding_spectral",
                    "--aggregate",
                    "median",
                    "--band",
                    "p10_p90",
                    "--output-figure",
                    f"'{figures_root}/segment_lda_ops_weight_recovery_ceilings_embedding_spectral.png'",
                    "--output-json",
                    f"'{figures_root}/segment_lda_ops_weight_recovery_ceilings_embedding_spectral_report.json'",
                ]
            )
        )

    plot_cmds.append(
        " ".join(
            [
                python_bin,
                "-u",
                "scripts/plot_segmented_lda_ctreepo_phase.py",
                "--input-glob",
                f"'{ctree_out}/**/*.json'",
                "--metric",
                "decomposition_total_root_l1_mean",
                "--aggregate",
                "median",
                "--output-figure",
                f"'{figures_root}/segmented_lda_ctreepo_phase.png'",
                "--output-json",
                f"'{figures_root}/segmented_lda_ctreepo_phase_report.json'",
            ]
        )
    )
    plot_cmds.append(
        " ".join(
            [
                python_bin,
                "-u",
                "scripts/plot_segmented_lda_ctreepo_ceilings.py",
                "--input-glob",
                f"'{ctree_out}/**/*.json'",
                "--aggregate",
                "median",
                "--band",
                "p10_p90",
                "--output-figure",
                f"'{figures_root}/segmented_lda_ctreepo_ceilings.png'",
                "--output-json",
                f"'{figures_root}/segmented_lda_ctreepo_ceilings_report.json'",
            ]
        )
    )
    plot_cmds.append(
        " ".join(
            [
                python_bin,
                "-u",
                "scripts/plot_ctreepo_guidance_frontier.py",
                "--input-glob",
                f"'{ctree_out}/**/*.json'",
                "--aggregate",
                "median",
                "--output-figure",
                f"'{figures_root}/ctreepo_guidance_frontier.png'",
                "--output-json",
                f"'{figures_root}/ctreepo_guidance_frontier_report.json'",
            ]
        )
    )

    markov_glob = f"{markov_out}/**/*seed_*.json" if bool(args.include_markov) else f"{output_root}/_none_markov/**/*.json"
    plot_cmds.append(
        " ".join(
            [
                python_bin,
                "-u",
                "scripts/plot_full_budget_gap_suite.py",
                "--markov-glob",
                f"'{markov_glob}'",
                "--segment-glob",
                f"'{seg_out}/**/*seed_*.json'",
                "--ctree-glob",
                f"'{ctree_out}/**/*.json'",
                "--aggregate",
                "median",
                "--output-figure",
                f"'{figures_root}/full_budget_gap_suite.png'",
                "--output-json",
                f"'{figures_root}/full_budget_gap_suite_report.json'",
            ]
        )
    )
    plot_cmds.append(
        " ".join(
            [
                python_bin,
                "-u",
                "scripts/report_identifiable_zero_suite.py",
                "--output-root",
                f"'{output_root}'",
                "--emit-pdf",
            ]
        )
    )
    _write_text(out_plot_cmds, "\n".join(plot_cmds) + ("\n" if plot_cmds else ""))

    meta = {
        "run_id": run_id,
        "profile": profile,
        "python_bin": python_bin,
        "skip_existing": bool(args.skip_existing),
        "include_markov": bool(args.include_markov),
        "include_embedding_estimator": bool(args.include_embedding_estimator),
        "output_root": str(output_root),
        "figures_root": str(figures_root),
        "cmds_file": str(out_cmds),
        "plot_cmds_file": str(out_plot_cmds),
        "counts_by_family": counts,
        "n_sim_commands_total": int(len(all_cmds)),
        "n_plot_commands_total": int(len(plot_cmds)),
        "builder_cmd_files": {k: str(v) for k, v in cmd_sources.items()},
        "segment_test_docs": int(args.segment_test_docs),
        "ctree_test_books": int(args.ctree_test_books),
    }
    _write_text(out_meta, json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
