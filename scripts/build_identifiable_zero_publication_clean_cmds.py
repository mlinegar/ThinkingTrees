#!/usr/bin/env python3
"""Build the reduced identifiable-zero command packs used by the clean publication report.

This builder matches the fixed slices consumed by
`report_identifiable_zero_suite_publication_clean.py`:

- Segment-LDA OPS: train_docs=12000, lambda=1.0
- Segmented-LDA C-TreePO: train_docs=4096, coupled q_infer grid
- Markov OPS-count: train_docs=8000, leaf_query_rate=1.0, include_root_query=true

The output is split into CPU and GPU command files so the Markov neural lane can
be queued behind existing GPU work without blocking the CPU-valid lanes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import re
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
        if abs(ql - qi) <= 1e-12:
            out.append(cmd)
    return out


def _keep_explicit_task_weight(cmds: List[str], *, task_weight: float) -> List[str]:
    token = f"--task-objective-weight {float(task_weight)}"
    return [cmd for cmd in cmds if token in cmd]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build reduced identifiable-zero publication-clean command lists.")
    p.add_argument("--python-bin", type=str, default="venv/bin/python")
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--figures-root", type=Path, default=None)

    p.add_argument("--out-cpu-cmds", type=Path, required=True)
    p.add_argument("--out-gpu-cmds", type=Path, required=True)
    p.add_argument("--out-plot-cmds", type=Path, required=True)
    p.add_argument("--out-meta", type=Path, required=True)

    p.add_argument("--segment-test-docs", type=int, default=5000)
    p.add_argument("--ctree-test-books", type=int, default=5000)
    p.add_argument("--markov-test-docs", type=int, default=2000)
    p.add_argument("--markov-n-epochs", type=int, default=12)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--n-seeds", type=int, default=12)
    p.add_argument("--markov-additive-device", type=str, default="cpu")
    p.add_argument("--markov-neural-device", type=str, default="auto")
    p.add_argument("--markov-local-law-weight", type=float, default=0.2)
    p.add_argument("--markov-task-objective-weight", type=float, default=1.0)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    python_bin = str(args.python_bin)
    output_root = Path(args.output_root)
    figures_root = args.figures_root or (output_root / "figures")
    skip_flag = "--skip-existing" if bool(args.skip_existing) else "--no-skip-existing"
    seeds = _seed_text(int(args.n_seeds))

    output_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)
    args.out_meta.parent.mkdir(parents=True, exist_ok=True)

    segment_audit = "0.01 0.02 0.05 0.1 0.2 0.5 1.0"
    ctree_cal = "0.01 0.02 0.05 0.1"
    ctree_qinfer = "0 0.05 0.1 0.2 0.35 0.5 0.75 1.0"
    markov_audit = "0 0.01 0.02 0.05 0.1 0.2 0.5 1.0"
    markov_qinfer = "0 0.1 0.25 0.5 0.75 1.0"

    tmp_segment = args.out_meta.with_name(args.out_meta.stem + "_tmp_segment.txt")
    tmp_ctree_raw = args.out_meta.with_name(args.out_meta.stem + "_tmp_ctree_raw.txt")
    tmp_markov_add = args.out_meta.with_name(args.out_meta.stem + "_tmp_markov_add.txt")
    tmp_markov_neu = args.out_meta.with_name(args.out_meta.stem + "_tmp_markov_neu.txt")

    _call(
        [
            python_bin,
            "-u",
            "scripts/build_segment_lda_ops_weight_recovery_cmds.py",
            "--out-cmds",
            str(tmp_segment),
            "--output-root",
            str(output_root / "segment_lda_ops_weight_recovery" / "publication_clean"),
            "--train-docs",
            "12000",
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

    _call(
        [
            python_bin,
            "-u",
            "scripts/build_segmented_lda_ctreepo_cmds.py",
            "--out-cmds",
            str(tmp_ctree_raw),
            "--output-root",
            str(output_root / "segmented_lda_ctreepo" / "equivalence"),
            "--train-docs",
            "4096",
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

    for fam, device, out_path in [
        ("additive", str(args.markov_additive_device), tmp_markov_add),
        ("neural", str(args.markov_neural_device), tmp_markov_neu),
    ]:
        # Keep the neural single-family pack in a disjoint subtree so it does not
        # overwrite the already-populated additive outputs when emitted separately.
        _call(
            [
                python_bin,
                "-u",
                "scripts/build_markov_changepoint_ops_count_cmds.py",
                "--out-cmds",
                str(out_path),
                "--output-root",
                str(
                    output_root
                    / "markov_changepoint_ops_count"
                    / "equivalence"
                    / (f"family_{fam}" if fam == "neural" else "")
                ),
                "--train-docs",
                "8000",
                "--test-docs",
                str(int(args.markov_test_docs)),
                "--audit-fractions",
                markov_audit,
                "--model-family",
                fam,
                "--c3-audit-strategies",
                "uniform",
                "--leaf-query-rates",
                "1.0",
                "--include-root-query",
                "true",
                "--local-law-weights",
                str(float(args.markov_local_law_weight)),
                "--task-objective-weights",
                str(float(args.markov_task_objective_weight)),
                "--c1-relative-weights",
                "0.0",
                "--c2-relative-weights",
                "0.0",
                "--c3-relative-weights",
                "1.0",
                "--root-weights",
                "1.0",
                "--schedule-consistency-weights",
                "0.0",
                "--seeds",
                seeds,
                "--n-epochs",
                str(int(args.markov_n_epochs)),
                "--device",
                device,
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
    ctree_cmds = _keep_coupled_ctree(_read_lines(tmp_ctree_raw))
    markov_add_cmds = _keep_explicit_task_weight(
        _read_lines(tmp_markov_add),
        task_weight=float(args.markov_task_objective_weight),
    )
    markov_neu_cmds = _keep_explicit_task_weight(
        _read_lines(tmp_markov_neu),
        task_weight=float(args.markov_task_objective_weight),
    )

    cpu_cmds = [*segment_cmds, *ctree_cmds, *markov_add_cmds]
    gpu_cmds = [*markov_neu_cmds]

    _write(args.out_cpu_cmds, cpu_cmds)
    _write(args.out_gpu_cmds, gpu_cmds)

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
    _write(args.out_plot_cmds, plot_cmds)

    meta: Dict[str, object] = {
        "output_root": str(output_root),
        "figures_root": str(figures_root),
        "skip_existing": bool(args.skip_existing),
        "cmd_files": {
            "cpu": str(args.out_cpu_cmds),
            "gpu": str(args.out_gpu_cmds),
            "plot": str(args.out_plot_cmds),
        },
        "counts": {
            "segment": int(len(segment_cmds)),
            "ctree_coupled": int(len(ctree_cmds)),
            "markov_additive": int(len(markov_add_cmds)),
            "markov_neural": int(len(markov_neu_cmds)),
            "cpu_total": int(len(cpu_cmds)),
            "gpu_total": int(len(gpu_cmds)),
        },
        "fixed_slices": {
            "segment": {
                "train_docs": 12000,
                "lambda_multiplier": 1.0,
                "audit_fraction": segment_audit,
                "topic_phi_estimators": "true embedding_spectral",
                "seeds": seeds,
            },
            "ctree": {
                "train_docs": 4096,
                "calibration_rates": ctree_cal,
                "q_infer_coupled": ctree_qinfer,
                "topic_phi_estimator": "spectral_numpy",
                "seeds": seeds,
            },
            "markov": {
                "train_docs": 8000,
                "audit_fraction": markov_audit,
                "q_infer": markov_qinfer,
                "leaf_query_rate": 1.0,
                "include_root_query": True,
                "task_objective_weight": float(args.markov_task_objective_weight),
                "local_law_weight": float(args.markov_local_law_weight),
                "c1_relative_weight": 0.0,
                "c2_relative_weight": 0.0,
                "c3_relative_weight": 1.0,
                "families": {
                    "additive": str(args.markov_additive_device),
                    "neural": str(args.markov_neural_device),
                },
                "seeds": seeds,
            },
        },
    }
    args.out_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
