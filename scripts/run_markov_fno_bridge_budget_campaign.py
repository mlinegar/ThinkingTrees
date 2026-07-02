#!/usr/bin/env python3
"""Run a wall-clock-budgeted CleanUnifiedNO Markov local-law campaign.

The point of this runner is breadth under a real overnight budget. It favors
short law-first probes across leaf sizes and FNO hyperparameters, then spends
remaining time on longer follow-ups for the best cells. The objective under
test is the decoded local-law bridge, not direct merge-state supervision.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CampaignCell:
    phase: str
    objective: str
    leaf_tokens: int
    channels: int
    g_n_modes: int
    epochs: int
    batch_size: int
    train_docs: int
    eval_docs: int
    seed: int = 0
    doc_tokens: int = 0
    load_data_bundle: str = ""
    benchmark: str = "recoverable_v5_t2048"
    markov_law_leaf_weight: float = 1.0
    markov_law_merge_weight: float = 1.0
    markov_law_idempotence_weight: float = 0.1
    markov_law_readout: str = "flatten"
    label: str = ""

    @property
    def cell_id(self) -> str:
        label = f"_{self.label}" if self.label else ""
        weight_bits = ""
        if self.objective == "markov_local_laws_fno":
            weight_bits = (
                f"_lw{self.markov_law_leaf_weight:g}"
                f"_mw{self.markov_law_merge_weight:g}"
                f"_iw{self.markov_law_idempotence_weight:g}"
                f"_{self.markov_law_readout}"
            )
        return (
            f"{self.phase}{label}"
            f"__obj_{self.objective}"
            f"__leaf{self.leaf_tokens}"
            f"__ch{self.channels}"
            f"__gm{self.g_n_modes}"
            f"__ep{self.epochs}"
            f"__bs{self.batch_size}"
            f"__n{self.train_docs}"
            f"{weight_bits}"
            f"__seed{self.seed}"
        ).replace(".", "p")


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _parse_gpus(raw: str) -> list[int]:
    gpus = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not gpus:
        raise ValueError("at least one GPU id is required")
    return gpus


def _batch_for_leaf(leaf_tokens: int) -> int:
    # Aggressive but conservative enough for full-trace law diagnostics on 96GB GPUs.
    if leaf_tokens >= 2048:
        return 16
    if leaf_tokens >= 256:
        return 32
    if leaf_tokens >= 128:
        return 1024
    if leaf_tokens >= 64:
        return 512
    if leaf_tokens >= 32:
        return 256
    if leaf_tokens >= 16:
        return 128
    return 64


def _train_docs_for_leaf(leaf_tokens: int, default_train_docs: int) -> int:
    # Keep the very deep trees in the campaign without letting them consume it.
    if leaf_tokens <= 4:
        return min(default_train_docs, 2048)
    if leaf_tokens <= 16:
        return min(default_train_docs, 4096)
    return default_train_docs


def _base_cell(
    *,
    phase: str,
    objective: str,
    leaf_tokens: int,
    channels: int,
    g_n_modes: int,
    epochs: int,
    train_docs: int,
    eval_docs: int,
    seed: int,
    load_data_bundle: str,
    label: str = "",
) -> CampaignCell:
    return CampaignCell(
        phase=phase,
        objective=objective,
        leaf_tokens=leaf_tokens,
        channels=channels,
        g_n_modes=g_n_modes,
        epochs=epochs,
        batch_size=_batch_for_leaf(leaf_tokens),
        train_docs=_train_docs_for_leaf(leaf_tokens, train_docs),
        eval_docs=eval_docs,
        seed=seed,
        load_data_bundle=load_data_bundle,
        label=label,
    )


def _phase1_cells(args: argparse.Namespace) -> list[CampaignCell]:
    cells: list[CampaignCell] = []
    # Law-first breadth. Keep leaf=2/4 present, but capped in docs/epochs.
    for leaf in [128, 64, 32, 16, 4, 2]:
        for channels in [64, 128]:
            for modes in [8, 16]:
                epochs = 6 if leaf >= 16 else 3
                cells.append(
                    _base_cell(
                        phase="p1_wide_laws",
                        objective="markov_local_laws_fno",
                        leaf_tokens=leaf,
                        channels=channels,
                        g_n_modes=modes,
                        epochs=epochs,
                        train_docs=args.train_docs,
                        eval_docs=args.eval_docs,
                        seed=args.seed,
                        load_data_bundle=args.load_data_bundle,
                    )
                )

    # Small set of root/contextual controls for interpreting law cells.
    for objective in ["root", "contextual_none"]:
        for leaf in [128, 64, 16, 2]:
            cells.append(
                _base_cell(
                    phase="p1_controls",
                    objective=objective,
                    leaf_tokens=leaf,
                    channels=128,
                    g_n_modes=16,
                    epochs=6 if leaf >= 16 else 3,
                    train_docs=args.train_docs,
                    eval_docs=args.eval_docs,
                    seed=args.seed,
                    load_data_bundle=args.load_data_bundle,
                )
            )

    # Direct witness is diagnostic only: capacity/readout sanity, not the theorem bridge.
    for leaf in [64, 16, 2]:
        cells.append(
            _base_cell(
                phase="p1_witness_diag",
                objective="markov_node_witness",
                leaf_tokens=leaf,
                channels=128,
                g_n_modes=16,
                epochs=3,
                train_docs=args.train_docs,
                eval_docs=args.eval_docs,
                seed=args.seed,
                load_data_bundle=args.load_data_bundle,
            )
        )
    return cells


def _phase2_cells(
    args: argparse.Namespace,
    completed_rows: list[dict[str, Any]],
) -> list[CampaignCell]:
    law_rows = [
        row
        for row in completed_rows
        if row.get("status") == "completed"
        and row.get("objective") == "markov_local_laws_fno"
        and row.get("test_root_mae") is not None
    ]
    law_rows.sort(key=lambda row: float(row.get("test_root_mae") or 1e30))

    selected: list[dict[str, Any]] = []
    seen_leaf: set[int] = set()
    for row in law_rows:
        leaf = int(row["leaf_tokens"])
        if leaf not in seen_leaf:
            selected.append(row)
            seen_leaf.add(leaf)
        if len(selected) >= 4:
            break
    for row in law_rows:
        if row not in selected:
            selected.append(row)
        if len(selected) >= 6:
            break

    cells: list[CampaignCell] = []
    for row in selected:
        leaf = int(row["leaf_tokens"])
        epochs = 24 if leaf >= 32 else 12
        cells.append(
            _base_cell(
                phase="p2_followup",
                objective="markov_local_laws_fno",
                leaf_tokens=leaf,
                channels=int(row["channels"]),
                g_n_modes=int(row["g_n_modes"]),
                epochs=epochs,
                train_docs=args.train_docs,
                eval_docs=args.eval_docs,
                seed=args.seed,
                load_data_bundle=args.load_data_bundle,
                label="best",
            )
        )

    if selected:
        best = selected[0]
        for idem in [0.0, 1.0]:
            cells.append(
                replace(
                    _base_cell(
                        phase="p2_idemp_ablation",
                        objective="markov_local_laws_fno",
                        leaf_tokens=int(best["leaf_tokens"]),
                        channels=int(best["channels"]),
                        g_n_modes=int(best["g_n_modes"]),
                        epochs=12,
                        train_docs=args.train_docs,
                        eval_docs=args.eval_docs,
                        seed=args.seed,
                        load_data_bundle=args.load_data_bundle,
                        label=f"idem{idem:g}",
                    ),
                    markov_law_idempotence_weight=idem,
                )
            )
        cells.append(
            replace(
                _base_cell(
                    phase="p2_readout_ablation",
                    objective="markov_local_laws_fno",
                    leaf_tokens=int(best["leaf_tokens"]),
                    channels=int(best["channels"]),
                    g_n_modes=int(best["g_n_modes"]),
                    epochs=12,
                    train_docs=args.train_docs,
                    eval_docs=args.eval_docs,
                    seed=args.seed,
                    load_data_bundle=args.load_data_bundle,
                    label="convpool",
                ),
                markov_law_readout="conv_pool",
            )
        )
    return cells


def _phase3_cells(args: argparse.Namespace) -> list[CampaignCell]:
    cells: list[CampaignCell] = []
    for objective in ["root", "markov_local_laws_fno"]:
        for leaf in [2048, 256]:
            for modes in [8, 16]:
                cells.append(
                    CampaignCell(
                        phase="p3_t2048_stress",
                        objective=objective,
                        leaf_tokens=leaf,
                        channels=128,
                        g_n_modes=modes,
                        epochs=8 if leaf == 2048 else 6,
                        batch_size=_batch_for_leaf(leaf),
                        train_docs=args.train_docs,
                        eval_docs=args.eval_docs,
                        seed=args.seed,
                        doc_tokens=2048,
                        load_data_bundle="",
                    )
                )
    return cells


def _cmd_for_cell(cell: CampaignCell, gpu: int, cell_root: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "probe_clean_unified_no.py"),
        "--benchmark",
        cell.benchmark,
        "--leaf-tokens",
        str(cell.leaf_tokens),
        "--train-docs",
        str(cell.train_docs),
        "--eval-docs",
        str(cell.eval_docs),
        "--epochs",
        str(cell.epochs),
        "--batch-size",
        str(cell.batch_size),
        "--channels",
        str(cell.channels),
        "--g-n-modes",
        str(cell.g_n_modes),
        "--g-n-layers",
        "2",
        "--scorer-n-modes",
        "16",
        "--scorer-n-layers",
        "2",
        "--lr",
        "0.0001",
        "--optimizer",
        "adamw",
        "--weight-decay",
        "0.01",
        "--lr-schedule",
        "cosine",
        "--grad-clip",
        "1.0",
        "--leaf-pool",
        "sum",
        "--diagnostic-baselines",
        "none",
        "--seed",
        str(cell.seed),
        "--device",
        "cuda",
        "--gpu",
        str(gpu),
        "--training-objective",
        cell.objective,
        "--output-root",
        str(cell_root),
    ]
    if cell.objective == "root":
        cmd.append("--root-only")
    if cell.load_data_bundle:
        cmd.extend(["--load-data-bundle", cell.load_data_bundle])
    elif cell.doc_tokens:
        cmd.extend(["--doc-tokens", str(cell.doc_tokens)])
    if cell.objective == "markov_local_laws_fno":
        cmd.extend(
            [
                "--markov-law-leaf-weight",
                str(cell.markov_law_leaf_weight),
                "--markov-law-merge-weight",
                str(cell.markov_law_merge_weight),
                "--markov-law-idempotence-weight",
                str(cell.markov_law_idempotence_weight),
                "--markov-law-readout",
                cell.markov_law_readout,
            ]
        )
    return cmd


def _nested_get(payload: dict[str, Any], path: list[str]) -> Any:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _summary_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    law_test = _nested_get(
        summary, ["markov_local_law_fno_diagnostics", "splits", "test"]
    )
    witness_test = _nested_get(
        summary, ["markov_node_witness_diagnostics", "splits", "test"]
    )
    row: dict[str, Any] = {
        "best_val_root_mae": summary.get("best_val_root_mae"),
        "best_val_epoch": summary.get("best_val_epoch"),
        "test_root_mae": summary.get("test_root_mae"),
        "n_leaves_per_doc": summary.get("n_leaves_per_doc"),
        "target_scale": summary.get("target_scale"),
    }
    if isinstance(law_test, dict):
        for prefix, block in [("law_leaf", law_test.get("leaf")), ("law_merge", law_test.get("merge")), ("law_root", law_test.get("root"))]:
            if isinstance(block, dict):
                row[f"{prefix}_theta_mae"] = block.get("theta_mae")
                row[f"{prefix}_first_acc"] = block.get("theta_first_regime_accuracy")
                row[f"{prefix}_last_acc"] = block.get("theta_last_regime_accuracy")
                row[f"{prefix}_count_mae"] = _nested_get(block, ["count_diagnostics", "root_mae"])
                row[f"{prefix}_rounded_count_exact"] = block.get("rounded_count_exact_rate")
                row[f"{prefix}_full_exact"] = block.get("full_witness_exact_rate")
                row[f"{prefix}_eps_idemp_range"] = block.get("eps_idemp_range")
    if isinstance(witness_test, dict):
        for prefix, block in [("witness_leaf", witness_test.get("leaf")), ("witness_merge", witness_test.get("merge")), ("witness_root", witness_test.get("root"))]:
            if isinstance(block, dict):
                row[f"{prefix}_theta_mae"] = block.get("theta_mae")
                row[f"{prefix}_first_acc"] = block.get("theta_first_regime_accuracy")
                row[f"{prefix}_last_acc"] = block.get("theta_last_regime_accuracy")
                row[f"{prefix}_count_mae"] = _nested_get(block, ["count_diagnostics", "root_mae"])
                row[f"{prefix}_full_exact"] = block.get("full_witness_exact_rate")
    return row


def _write_rows(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "campaign_summary.json", "w") as fh:
        json.dump(rows, fh, indent=2, allow_nan=True)
    if not rows:
        return
    preferred = [
        "cell_id",
        "status",
        "phase",
        "objective",
        "leaf_tokens",
        "channels",
        "g_n_modes",
        "epochs",
        "batch_size",
        "train_docs",
        "gpu",
        "elapsed_sec",
        "best_val_root_mae",
        "test_root_mae",
        "law_leaf_theta_mae",
        "law_merge_theta_mae",
        "law_leaf_first_acc",
        "law_leaf_last_acc",
        "law_merge_first_acc",
        "law_merge_last_acc",
        "law_leaf_count_mae",
        "law_merge_count_mae",
        "witness_leaf_theta_mae",
        "witness_merge_theta_mae",
        "error",
        "output_root",
    ]
    fields = [field for field in preferred if any(field in row for row in rows)]
    fields.extend(
        sorted({key for row in rows for key in row.keys() if key not in set(fields)})
    )
    with open(output_root / "campaign_summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with open(output_root / "campaign_report.md", "w") as fh:
        completed = sum(1 for row in rows if row.get("status") == "completed")
        fh.write("# Markov FNO Local-Law Bridge Budget Campaign\n\n")
        fh.write(f"Cells completed: {completed} / {len(rows)}\n\n")
        fh.write("| cell | objective | leaf | ch | modes | ep | bs | test root MAE | law leaf theta | law merge theta | leaf first/last | merge first/last |\n")
        fh.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            def fmt(key: str, digits: int = 4) -> str:
                val = row.get(key)
                return "" if val is None else f"{float(val):.{digits}f}"

            leaf_acc = (
                ""
                if row.get("law_leaf_first_acc") is None or row.get("law_leaf_last_acc") is None
                else f"{float(row['law_leaf_first_acc']):.3f}/{float(row['law_leaf_last_acc']):.3f}"
            )
            merge_acc = (
                ""
                if row.get("law_merge_first_acc") is None or row.get("law_merge_last_acc") is None
                else f"{float(row['law_merge_first_acc']):.3f}/{float(row['law_merge_last_acc']):.3f}"
            )
            fh.write(
                f"| {row.get('cell_id')} | {row.get('objective')} | {row.get('leaf_tokens')} "
                f"| {row.get('channels')} | {row.get('g_n_modes')} | {row.get('epochs')} "
                f"| {row.get('batch_size')} | {fmt('test_root_mae')} | {fmt('law_leaf_theta_mae')} "
                f"| {fmt('law_merge_theta_mae')} | {leaf_acc} | {merge_acc} |\n"
            )


def _run_cell(
    cell: CampaignCell,
    gpu: int,
    output_root: Path,
    deadline: float,
) -> dict[str, Any]:
    cell_root = output_root / "cells" / cell.cell_id
    cell_root.mkdir(parents=True, exist_ok=True)
    with open(cell_root / "cell.json", "w") as fh:
        json.dump({**asdict(cell), "gpu": gpu}, fh, indent=2)
    cmd = _cmd_for_cell(cell, gpu, cell_root)
    log_path = cell_root / "job.log"
    row: dict[str, Any] = {**asdict(cell), "cell_id": cell.cell_id, "gpu": gpu, "output_root": str(cell_root)}
    remaining = deadline - time.time()
    if remaining < 120:
        row.update({"status": "skipped_deadline", "elapsed_sec": 0.0})
        return row
    started = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        with open(log_path, "w") as log_fh:
            proc = subprocess.run(
                cmd,
                cwd=REPO,
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                timeout=max(60.0, remaining - 30.0),
                check=False,
            )
        row["returncode"] = proc.returncode
        row["elapsed_sec"] = round(time.time() - started, 3)
        summary_path = cell_root / "summary.json"
        if proc.returncode == 0 and summary_path.exists():
            with open(summary_path) as fh:
                summary = json.load(fh)
            row.update(_summary_metrics(summary))
            row["status"] = "completed"
        else:
            row["status"] = "failed"
            row["error"] = f"returncode={proc.returncode}; summary_exists={summary_path.exists()}"
    except subprocess.TimeoutExpired:
        row["status"] = "timeout"
        row["elapsed_sec"] = round(time.time() - started, 3)
        row["error"] = "cell exceeded remaining campaign budget"
    except Exception as exc:  # pragma: no cover - defensive launcher path.
        row["status"] = "failed"
        row["elapsed_sec"] = round(time.time() - started, 3)
        row["error"] = repr(exc)
    with open(cell_root / "result.json", "w") as fh:
        json.dump(row, fh, indent=2, allow_nan=True)
    return row


def _run_cells_parallel(
    *,
    phase_name: str,
    cells: list[CampaignCell],
    gpus: list[int],
    output_root: Path,
    deadline: float,
    rows: list[dict[str, Any]],
    rows_lock: threading.Lock,
) -> None:
    if not cells:
        return
    print(f"==> {phase_name}: {len(cells)} cells on GPUs {gpus}", flush=True)
    work: queue.Queue[CampaignCell] = queue.Queue()
    for cell in cells:
        work.put(cell)

    def worker(gpu: int) -> None:
        while time.time() < deadline - 120:
            try:
                cell = work.get_nowait()
            except queue.Empty:
                return
            print(f"[gpu{gpu}] start {cell.cell_id}", flush=True)
            row = _run_cell(cell, gpu, output_root, deadline)
            print(
                f"[gpu{gpu}] done {cell.cell_id} status={row.get('status')} "
                f"test_root_mae={row.get('test_root_mae')}",
                flush=True,
            )
            with rows_lock:
                rows.append(row)
                _write_rows(output_root, rows)
            work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=True) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an 8-hour-ish staged Markov FNO local-law bridge campaign."
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO / "outputs" / f"markov_fno_bridge_budget_campaign_{_utc_stamp()}"),
    )
    parser.add_argument("--budget-hours", type=float, default=8.0)
    parser.add_argument("--gpus", default="0,2,3")
    parser.add_argument(
        "--load-data-bundle",
        default="outputs/_bundles/markov_hazard_panels/paper_hazard_panel_v1_t128/seed_0/base_bundle.json",
    )
    parser.add_argument("--train-docs", type=int, default=10240)
    parser.add_argument("--eval-docs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-t2048", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    gpus = _parse_gpus(args.gpus)
    deadline = time.time() + float(args.budget_hours) * 3600.0
    rows: list[dict[str, Any]] = []
    rows_lock = threading.Lock()
    manifest = {
        "schema_version": "markov_fno_bridge_budget_campaign.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "output_root": str(output_root),
        "budget_hours": args.budget_hours,
        "deadline_utc": datetime.fromtimestamp(deadline, UTC).isoformat(),
        "gpus": gpus,
        "args": vars(args),
    }
    with open(output_root / "campaign_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    phase1 = _phase1_cells(args)
    _run_cells_parallel(
        phase_name="phase 1 wide law/control probes",
        cells=phase1,
        gpus=gpus,
        output_root=output_root,
        deadline=deadline,
        rows=rows,
        rows_lock=rows_lock,
    )

    phase2 = _phase2_cells(args, rows)
    _run_cells_parallel(
        phase_name="phase 2 best-cell followups and ablations",
        cells=phase2,
        gpus=gpus,
        output_root=output_root,
        deadline=deadline,
        rows=rows,
        rows_lock=rows_lock,
    )

    if not args.skip_t2048:
        phase3 = _phase3_cells(args)
        _run_cells_parallel(
            phase_name="phase 3 t2048 composition stress",
            cells=phase3,
            gpus=gpus,
            output_root=output_root,
            deadline=deadline,
            rows=rows,
            rows_lock=rows_lock,
        )

    _write_rows(output_root, rows)
    print(f"summary: {output_root / 'campaign_summary.csv'}", flush=True)
    print(f"report:  {output_root / 'campaign_report.md'}", flush=True)


if __name__ == "__main__":
    main()
