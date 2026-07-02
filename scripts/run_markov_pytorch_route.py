#!/usr/bin/env python3
"""Self-contained PyTorch Markov route around CleanUnifiedNO."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from markov_route_contract import normalize_pytorch_summary, write_route_outputs
except ModuleNotFoundError:  # pragma: no cover - used when imported as a package in tests.
    from scripts.markov_route_contract import normalize_pytorch_summary, write_route_outputs


REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PytorchRouteCell:
    objective: str
    leaf_tokens: int
    channels: int
    n_modes: int
    epochs: int
    batch_size: int
    seed: int

    @property
    def cell_id(self) -> str:
        return (
            f"{self.objective}"
            f"__leaf{self.leaf_tokens}"
            f"__ch{self.channels}"
            f"__m{self.n_modes}"
            f"__ep{self.epochs}"
            f"__bs{self.batch_size}"
            f"__seed{self.seed}"
        )


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _parse_ints(raw: str) -> list[int]:
    return [int(part) for part in str(raw).replace(",", " ").split() if part]


def _parse_strings(raw: str) -> list[str]:
    return [part for part in str(raw).replace(",", " ").split() if part]


def _cells(args: argparse.Namespace) -> list[PytorchRouteCell]:
    cells: list[PytorchRouteCell] = []
    for objective in _parse_strings(args.objectives):
        for leaf in _parse_ints(args.leaves):
            for channels in _parse_ints(args.channels):
                for n_modes in _parse_ints(args.n_modes):
                    for seed in _parse_ints(args.seeds):
                        cells.append(
                            PytorchRouteCell(
                                objective=objective,
                                leaf_tokens=leaf,
                                channels=channels,
                                n_modes=n_modes,
                                epochs=int(args.epochs),
                                batch_size=int(args.batch_size),
                                seed=seed,
                            )
                        )
    return cells


def _objective_args(cell: PytorchRouteCell) -> list[str]:
    if cell.objective == "root":
        return ["--training-objective", "root", "--root-only"]
    if cell.objective == "contextual_none":
        return [
            "--training-objective",
            "contextual_sufficiency",
            "--context-samples-per-doc",
            "1",
            "--contextual-loss-weight",
            "1.0",
            "--infomax-loss-weight",
            "0.0",
            "--contextual-dependence-objective",
            "none",
            "--root-only",
        ]
    if cell.objective == "markov_node_witness":
        return ["--training-objective", "markov_node_witness"]
    if cell.objective == "markov_local_laws_fno":
        return ["--training-objective", "markov_local_laws_fno"]
    raise ValueError(f"unknown PyTorch route objective: {cell.objective!r}")


def _cmd_for_cell(args: argparse.Namespace, cell: PytorchRouteCell, cell_root: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "probe_clean_unified_no.py"),
        "--benchmark",
        str(args.benchmark),
        "--load-data-bundle",
        str(args.bundle),
        "--doc-tokens",
        str(args.doc_tokens),
        "--leaf-tokens",
        str(cell.leaf_tokens),
        "--train-docs",
        str(args.train_docs),
        "--eval-docs",
        str(args.eval_docs),
        "--epochs",
        str(cell.epochs),
        "--batch-size",
        str(cell.batch_size),
        "--channels",
        str(cell.channels),
        "--g-n-modes",
        str(cell.n_modes),
        "--g-n-layers",
        str(args.g_n_layers),
        "--scorer-n-modes",
        str(args.scorer_n_modes),
        "--scorer-n-layers",
        str(args.scorer_n_layers),
        "--lr",
        str(args.lr),
        "--optimizer",
        str(args.optimizer),
        "--weight-decay",
        str(args.weight_decay),
        "--lr-schedule",
        str(args.lr_schedule),
        "--grad-clip",
        str(args.grad_clip),
        "--leaf-pool",
        str(args.leaf_pool),
        "--diagnostic-baselines",
        str(args.diagnostic_baselines),
        "--markov-witness-readout",
        str(args.markov_witness_readout),
        "--markov-law-readout",
        str(args.markov_law_readout),
        "--markov-law-idempotence-weight",
        str(args.markov_law_idempotence_weight),
        "--seed",
        str(cell.seed),
        "--device",
        str(args.device),
        "--output-root",
        str(cell_root),
    ]
    if args.gpu is not None:
        cmd.extend(["--gpu", str(args.gpu)])
    cmd.extend(_objective_args(cell))
    return cmd


def _row(cell: PytorchRouteCell, cell_root: Path, elapsed: float | None = None) -> dict[str, Any]:
    summary_path = cell_root / "summary.json"
    if not summary_path.exists():
        return {
            "cell_id": cell.cell_id,
            "status": "missing_summary",
            "route": "pytorch",
            "objective": cell.objective,
            "output_root": str(cell_root),
            "elapsed_sec": elapsed,
        }
    with summary_path.open() as fh:
        summary = json.load(fh)
    return normalize_pytorch_summary(
        summary,
        cell_id=cell.cell_id,
        objective=cell.objective,
        output_root=str(cell_root),
        elapsed_sec=elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the self-contained PyTorch CleanUnifiedNO Markov route."
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO / "outputs" / f"markov_pytorch_route_{_utc_stamp()}"),
    )
    parser.add_argument("--benchmark", default="paper_hazard_panel_v1")
    parser.add_argument(
        "--bundle",
        default=(
            "outputs/_bundles/markov_hazard_panels/"
            "paper_hazard_panel_v1_t128/seed_0/base_bundle.json"
        ),
    )
    parser.add_argument("--objectives", default="root,markov_node_witness,markov_local_laws_fno")
    parser.add_argument("--doc-tokens", type=int, default=128)
    parser.add_argument("--leaves", default="16")
    parser.add_argument("--channels", default="32")
    parser.add_argument("--n-modes", default="8")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--train-docs", type=int, default=64)
    parser.add_argument("--eval-docs", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--g-n-layers", type=int, default=2)
    parser.add_argument("--scorer-n-modes", type=int, default=8)
    parser.add_argument("--scorer-n-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--lr-schedule", default="cosine", choices=["none", "cosine"])
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--leaf-pool", default="sum", choices=["sum", "mean"])
    parser.add_argument("--diagnostic-baselines", default="none")
    parser.add_argument("--markov-witness-readout", default="flatten", choices=["flatten", "conv_pool"])
    parser.add_argument("--markov-law-readout", default="flatten", choices=["flatten", "conv_pool"])
    parser.add_argument("--markov-law-idempotence-weight", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cells = _cells(args)
    manifest = {
        "schema_version": "markov_pytorch_route.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "route": "pytorch",
        "output_root": str(output_root),
        "args": vars(args),
        "cells": [asdict(cell) for cell in cells],
    }
    if args.dry_run:
        write_route_outputs(output_root, [], title="Markov PyTorch Route", manifest=manifest)
        print(json.dumps(manifest, indent=2))
        return

    rows: list[dict[str, Any]] = []
    if args.aggregate_only:
        rows = [_row(cell, output_root / cell.cell_id) for cell in cells]
        write_route_outputs(output_root, rows, title="Markov PyTorch Route", manifest=manifest)
        print(f"summary: {output_root / 'grid_summary.csv'}")
        return

    for idx, cell in enumerate(cells, start=1):
        cell_root = output_root / cell.cell_id
        if (cell_root / "summary.json").exists():
            rows.append(_row(cell, cell_root, elapsed=0.0))
            write_route_outputs(output_root, rows, title="Markov PyTorch Route", manifest=manifest)
            print(f"[{idx}/{len(cells)}] skip {cell.cell_id}", flush=True)
            continue
        cmd = _cmd_for_cell(args, cell, cell_root)
        print(f"[{idx}/{len(cells)}] start {cell.cell_id}", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=REPO)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            rows.append(
                {
                    **asdict(cell),
                    "cell_id": cell.cell_id,
                    "status": f"failed:{proc.returncode}",
                    "route": "pytorch",
                    "backend": "pytorch",
                    "elapsed_sec": round(elapsed, 3),
                    "output_root": str(cell_root),
                }
            )
            write_route_outputs(output_root, rows, title="Markov PyTorch Route", manifest=manifest)
            if not args.keep_going:
                raise SystemExit(proc.returncode)
            continue
        rows.append(_row(cell, cell_root, elapsed=round(elapsed, 3)))
        write_route_outputs(output_root, rows, title="Markov PyTorch Route", manifest=manifest)
        print(f"[{idx}/{len(cells)}] done {cell.cell_id}", flush=True)

    write_route_outputs(output_root, rows, title="Markov PyTorch Route", manifest=manifest)
    print(f"summary: {output_root / 'grid_summary.csv'}")
    print(f"report:  {output_root / 'grid_report.md'}")


if __name__ == "__main__":
    main()
