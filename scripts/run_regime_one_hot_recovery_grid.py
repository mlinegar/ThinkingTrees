#!/usr/bin/env python3
"""Run the JAX regime-one-hot local-law recovery grid.

This targets the count-extraction failure in ``learned_local_laws``.  The main
grid compares the existing flat MLP summary against the weakly structured
``regime_transition_sum`` encoder.  If the structured leaf=64 row meets the
configured success threshold, the runner adds learned-merge seed follow-ups.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GridCell:
    phase: str
    summary_family: str
    hidden_dim: int
    leaf: int
    n_iter: int
    learning_rate: float
    law_architecture: str = "analytic"
    c2_merge_target: str = "theta"
    seed: int = 0

    @property
    def cell_id(self) -> str:
        return (
            f"{self.phase}"
            f"__family_{self.summary_family}"
            f"__hidden{self.hidden_dim}"
            f"__leaf{self.leaf}"
            f"__iter{self.n_iter}"
            f"__lr{self.learning_rate:g}"
            f"__arch_{self.law_architecture}"
            f"__c2_{self.c2_merge_target}"
            f"__seed{self.seed}"
        )


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw).replace(",", " ").split() if part.strip()]


def _parse_strings(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).replace(",", " ").split() if part.strip()]


def _ctreepo_bin() -> str:
    candidate = REPO / "venv" / "bin" / "ctreepo"
    return str(candidate) if candidate.exists() else "ctreepo"


def _cmd_for_cell(args: argparse.Namespace, cell: GridCell, cell_root: Path) -> list[str]:
    return [
        _ctreepo_bin(),
        "sim",
        "run",
        "contextual-sbijax",
        "--data-source",
        "markov",
        "--load-data-bundle",
        str(args.bundle),
        "--sbijax-trainer",
        "learned_local_laws",
        "--sbijax-method",
        "nasss",
        "--sbijax-package-theta",
        "markov_exact_sketch",
        "--sbijax-input-encoding",
        "regime_one_hot",
        "--local-law-summary-family",
        str(cell.summary_family),
        "--law-architecture",
        str(cell.law_architecture),
        "--c2-merge-target",
        str(cell.c2_merge_target),
        "--learned-merge-hidden-dim",
        str(cell.hidden_dim),
        "--learned-decoder-hidden-dim",
        str(cell.hidden_dim),
        "--train-docs",
        str(args.train_docs),
        "--val-docs",
        str(args.val_docs),
        "--test-docs",
        str(args.test_docs),
        "--fragment-len",
        str(cell.leaf),
        "--context-samples-per-doc",
        "1",
        "--response-signature-contexts",
        str(args.response_signature_contexts),
        "--response-signature-slices",
        str(args.response_signature_slices),
        "--embedding-dim",
        "32",
        "--state-dim",
        "25",
        "--hidden-dim",
        str(cell.hidden_dim),
        "--learning-rate",
        str(cell.learning_rate),
        "--lr-schedule",
        "cosine",
        "--n-iter",
        str(cell.n_iter),
        "--batch-size",
        str(args.batch_size),
        "--local-law-weight",
        "1.0",
        "--local-law-leaf-weight",
        "1.0",
        "--local-law-merge-weight",
        "1.0",
        "--local-law-idempotence-weight",
        "1.0",
        "--local-law-contextual-weight",
        "1.0",
        "--local-law-package-weight",
        "0.0",
        "--seed",
        str(cell.seed),
        "--output-root",
        str(cell_root),
    ]


def _read_metric_row(cell: GridCell, cell_root: Path, *, status: str, elapsed: float = 0.0, error: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {
        **asdict(cell),
        "cell_id": cell.cell_id,
        "status": status,
        "elapsed_sec": round(float(elapsed), 3),
        "output_root": str(cell_root),
    }
    summary_path = cell_root / "summary.json"
    if not summary_path.exists():
        if error:
            row["error"] = error
        return row
    with summary_path.open() as fh:
        data = json.load(fh)
    test = dict(data.get("diagnostics", {}).get("test", {}) or {})
    final = {}
    history = list(data.get("history") or [])
    if history:
        final = dict(history[-1])
    row.update(
        {
            "contextual_mae": test.get("contextual_mae"),
            "contextual_raw_mae": test.get("contextual_raw_mae"),
            "theta_mae": test.get("theta_mae"),
            "theta_count_raw_mae": test.get("theta_count_raw_mae"),
            "theta_first_regime_accuracy": test.get("theta_first_regime_accuracy"),
            "theta_last_regime_accuracy": test.get("theta_last_regime_accuracy"),
            "eps_leaf": test.get("eps_leaf"),
            "eps_merge": test.get("eps_merge"),
            "eps_idemp": test.get("eps_idemp"),
            "pred_truth_corr": test.get("pred_truth_corr"),
            "best_iteration": final.get("best_iteration"),
            "best_val_law_score": final.get("best_val_law_score"),
        }
    )
    return row


def _write_outputs(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "summary.json").open("w") as fh:
        json.dump(rows, fh, indent=2, allow_nan=True)
    if not rows:
        return
    preferred = [
        "cell_id",
        "status",
        "phase",
        "summary_family",
        "hidden_dim",
        "leaf",
        "n_iter",
        "learning_rate",
        "law_architecture",
        "c2_merge_target",
        "seed",
        "contextual_raw_mae",
        "theta_count_raw_mae",
        "theta_mae",
        "theta_first_regime_accuracy",
        "theta_last_regime_accuracy",
        "eps_leaf",
        "eps_merge",
        "eps_idemp",
        "best_iteration",
        "elapsed_sec",
        "error",
        "output_root",
    ]
    fields = [field for field in preferred if any(field in row for row in rows)]
    fields.extend(sorted({key for row in rows for key in row.keys() if key not in set(fields)}))
    with (output_root / "grid_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (output_root / "grid_report.md").open("w") as fh:
        fh.write("# Regime-One-Hot Recovery Grid\n\n")
        fh.write(f"Rows: {len(rows)}\n\n")
        fh.write("| cell | family | hidden | leaf | iter | lr | arch | seed | count raw MAE | theta MAE | first/last | eps merge |\n")
        fh.write("| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            def fmt(key: str) -> str:
                value = row.get(key)
                if value is None or value == "":
                    return ""
                try:
                    return f"{float(value):.6g}"
                except Exception:
                    return str(value)

            first = row.get("theta_first_regime_accuracy")
            last = row.get("theta_last_regime_accuracy")
            edge = "" if first is None or last is None else f"{float(first):.3f}/{float(last):.3f}"
            fh.write(
                f"| {row.get('cell_id')} | {row.get('summary_family')} | "
                f"{row.get('hidden_dim')} | {row.get('leaf')} | {row.get('n_iter')} | "
                f"{row.get('learning_rate')} | {row.get('law_architecture')} | {row.get('seed')} | "
                f"{fmt('theta_count_raw_mae')} | {fmt('theta_mae')} | {edge} | "
                f"{fmt('eps_merge')} |\n"
            )


def _run_cell(args: argparse.Namespace, cell: GridCell, gpu: int, output_root: Path) -> dict[str, Any]:
    cell_root = output_root / "cells" / cell.cell_id
    if (cell_root / "summary.json").exists():
        return _read_metric_row(cell, cell_root, status="completed_cached")
    cell_root.mkdir(parents=True, exist_ok=True)
    with (cell_root / "cell.json").open("w") as fh:
        json.dump({**asdict(cell), "gpu": gpu}, fh, indent=2)
    cmd = _cmd_for_cell(args, cell, cell_root)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.xla_mem_fraction)
    env["PYTHONUNBUFFERED"] = "1"
    started = time.time()
    log_path = cell_root / "job.log"
    with log_path.open("w") as log_fh:
        proc = subprocess.run(
            cmd,
            cwd=REPO,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.time() - started
    status = "completed" if proc.returncode == 0 and (cell_root / "summary.json").exists() else "failed"
    error = "" if status == "completed" else f"returncode={proc.returncode}; summary_exists={(cell_root / 'summary.json').exists()}"
    row = _read_metric_row(cell, cell_root, status=status, elapsed=elapsed, error=error)
    with (cell_root / "result.json").open("w") as fh:
        json.dump(row, fh, indent=2, allow_nan=True)
    return row


def _run_cells(args: argparse.Namespace, cells: list[GridCell], output_root: Path, rows: list[dict[str, Any]]) -> None:
    gpus = _parse_ints(args.gpus)
    work: queue.Queue[GridCell] = queue.Queue()
    for cell in cells:
        work.put(cell)
    lock = threading.Lock()

    def worker(gpu: int) -> None:
        while True:
            try:
                cell = work.get_nowait()
            except queue.Empty:
                return
            print(f"[gpu{gpu}] start {cell.cell_id}", flush=True)
            row = _run_cell(args, cell, gpu, output_root)
            print(
                f"[gpu{gpu}] done {cell.cell_id} status={row.get('status')} "
                f"count_mae={row.get('theta_count_raw_mae')}",
                flush=True,
            )
            with lock:
                rows.append(row)
                _write_outputs(output_root, rows)
            work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=True) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def _completed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("status", "")).startswith("completed")]


def _best_structured_by_leaf(rows: list[dict[str, Any]], leaf: int) -> dict[str, Any] | None:
    candidates = [
        row
        for row in _completed(rows)
        if row.get("summary_family") == "regime_transition_sum"
        and int(row.get("leaf", -1)) == int(leaf)
        and row.get("theta_count_raw_mae") is not None
        and row.get("law_architecture") == "analytic"
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(row.get("theta_count_raw_mae") or 1e30))


def _main_cells(args: argparse.Namespace) -> list[GridCell]:
    cells = []
    for family in _parse_strings(args.summary_families):
        learning_rates = (
            _parse_strings(args.structured_learning_rates)
            if family == "regime_transition_sum"
            else _parse_strings(args.mlp_learning_rates)
        )
        for hidden in _parse_ints(args.hidden_dims):
            for leaf in _parse_ints(args.leaves):
                for learning_rate in learning_rates:
                    cells.append(
                        GridCell(
                            phase="main",
                            summary_family=family,
                            hidden_dim=hidden,
                            leaf=leaf,
                            n_iter=int(args.n_iter),
                            learning_rate=float(learning_rate),
                        )
                    )
    return cells


def _long_followup_cells(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[GridCell]:
    cells: list[GridCell] = []
    for leaf in [32, 64, 128]:
        best = _best_structured_by_leaf(rows, leaf)
        if best is None:
            continue
        cells.append(
            GridCell(
                phase="long",
                summary_family="regime_transition_sum",
                hidden_dim=int(best["hidden_dim"]),
                leaf=int(best["leaf"]),
                n_iter=int(args.long_n_iter),
                learning_rate=float(best["learning_rate"]),
            )
        )
    return cells


def _structured_leaf64_succeeded(rows: list[dict[str, Any]], threshold: float) -> bool:
    best = _best_structured_by_leaf(rows, 64)
    if best is None:
        return False
    return (
        float(best.get("theta_count_raw_mae") or 1e30) < float(threshold)
        and float(best.get("theta_mae") or 1e30) < 1e-3
        and float(best.get("eps_merge") or 1e30) < 1e-3
    )


def _learned_merge_cells(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[GridCell]:
    if not _structured_leaf64_succeeded(rows, float(args.count_success_threshold)):
        print("skip learned-merge followup: structured leaf=64 success criteria not met", flush=True)
        return []
    cells: list[GridCell] = []
    for leaf in [32, 64]:
        best = _best_structured_by_leaf(rows, leaf)
        if best is None:
            continue
        for seed in [0, 1, 2]:
            cells.append(
                GridCell(
                    phase="learned_merge",
                    summary_family="regime_transition_sum",
                    hidden_dim=int(best["hidden_dim"]),
                    leaf=int(best["leaf"]),
                    n_iter=int(args.n_iter),
                    learning_rate=float(best["learning_rate"]),
                    law_architecture="learned_merge",
                    c2_merge_target="self_consistency",
                    seed=seed,
                )
            )
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the JAX regime-one-hot local-law recovery grid."
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO / "outputs" / f"regime_one_hot_recovery_grid_{_utc_stamp()}"),
    )
    parser.add_argument(
        "--bundle",
        default="outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json",
    )
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--summary-families", default="mlp regime_transition_sum")
    parser.add_argument("--hidden-dims", default="64 128 256")
    parser.add_argument("--leaves", default="1 2 4 8 16 32 64 128")
    parser.add_argument("--train-docs", type=int, default=10240)
    parser.add_argument("--val-docs", type=int, default=1024)
    parser.add_argument("--test-docs", type=int, default=1024)
    parser.add_argument("--n-iter", type=int, default=300)
    parser.add_argument("--long-n-iter", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mlp-learning-rates", default="0.0003")
    parser.add_argument("--structured-learning-rates", default="0.003 0.01")
    parser.add_argument("--response-signature-contexts", type=int, default=16)
    parser.add_argument("--response-signature-slices", type=int, default=8)
    parser.add_argument("--count-success-threshold", type=float, default=0.05)
    parser.add_argument("--xla-mem-fraction", type=float, default=0.35)
    parser.add_argument("--main-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "regime_one_hot_recovery_grid.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "args": vars(args),
    }
    with (output_root / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)

    rows: list[dict[str, Any]] = []
    phases = [("main", _main_cells(args))]
    if args.dry_run:
        for name, cells in phases:
            print(f"{name}: {len(cells)} cells")
            for cell in cells:
                print(cell.cell_id)
        return 0

    for phase_name, cells in phases:
        print(f"==> {phase_name}: {len(cells)} cells", flush=True)
        _run_cells(args, cells, output_root, rows)
    if not args.main_only:
        long_cells = _long_followup_cells(args, rows)
        print(f"==> long followups: {len(long_cells)} cells", flush=True)
        _run_cells(args, long_cells, output_root, rows)
        merge_cells = _learned_merge_cells(args, rows)
        print(f"==> learned-merge followups: {len(merge_cells)} cells", flush=True)
        _run_cells(args, merge_cells, output_root, rows)
    _write_outputs(output_root, rows)
    print(f"summary: {output_root / 'grid_summary.csv'}", flush=True)
    print(f"report:  {output_root / 'grid_report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
