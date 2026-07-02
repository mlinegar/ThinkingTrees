#!/usr/bin/env python3
"""Self-contained JAX Markov route: sbijax baselines plus internal JAX FNO cells."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from markov_route_contract import normalize_jax_summary, write_route_outputs
except ModuleNotFoundError:  # pragma: no cover - used when imported as a package in tests.
    from scripts.markov_route_contract import normalize_jax_summary, write_route_outputs


REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class JaxRouteCell:
    objective: str
    input_encoding: str
    leaf_tokens: int
    hidden_dim: int
    n_iter: int
    learning_rate: float
    seed: int
    n_modes: int = 8
    n_layers: int = 2
    pooling_mode: str = "sum"

    @property
    def cell_id(self) -> str:
        return (
            f"{self.objective}"
            f"__enc_{self.input_encoding}"
            f"__leaf{self.leaf_tokens}"
            f"__h{self.hidden_dim}"
            f"__iter{self.n_iter}"
            f"__lr{self.learning_rate:g}"
            f"__seed{self.seed}"
        )


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _parse_ints(raw: str) -> list[int]:
    return [int(part) for part in str(raw).replace(",", " ").split() if part]


def _parse_strings(raw: str) -> list[str]:
    return [part for part in str(raw).replace(",", " ").split() if part]


def _ctreepo_bin() -> str:
    candidate = REPO / "venv" / "bin" / "ctreepo"
    return str(candidate) if candidate.exists() else "ctreepo"


def _objective_args(cell: JaxRouteCell) -> list[str]:
    if cell.objective == "sbijax_package_nass":
        return ["--sbijax-trainer", "package", "--sbijax-method", "nass"]
    if cell.objective == "sbijax_package_nasss":
        return ["--sbijax-trainer", "package", "--sbijax-method", "nasss"]
    if cell.objective == "jax_fno_node_witness":
        return [
            "--sbijax-trainer",
            "learned_local_laws",
            "--sbijax-method",
            "nasss",
            "--local-law-summary-family",
            "jax_fno",
            "--law-architecture",
            "analytic",
            "--c2-merge-target",
            "theta",
            "--local-law-supervision-mode",
            "dense_exact",
            "--local-law-leaf-weight",
            "1.0",
            "--local-law-merge-weight",
            "0.0",
            "--local-law-idempotence-weight",
            "0.0",
            "--local-law-contextual-weight",
            "0.0",
        ]
    if cell.objective == "jax_fno_local_laws":
        return [
            "--sbijax-trainer",
            "learned_local_laws",
            "--sbijax-method",
            "nasss",
            "--local-law-summary-family",
            "jax_fno",
            "--law-architecture",
            "learned_merge",
            "--c2-merge-target",
            "self_consistency",
            "--local-law-contextual-weight",
            "0.0",
        ]
    if cell.objective == "regime_transition_sum":
        return [
            "--sbijax-trainer",
            "learned_local_laws",
            "--sbijax-method",
            "nasss",
            "--local-law-summary-family",
            "regime_transition_sum",
            "--law-architecture",
            "analytic",
            "--c2-merge-target",
            "theta",
            "--local-law-contextual-weight",
            "0.0",
        ]
    raise ValueError(f"unknown JAX route objective: {cell.objective!r}")


def _cmd_for_cell(args: argparse.Namespace, cell: JaxRouteCell, cell_root: Path) -> list[str]:
    cmd = [
        _ctreepo_bin(),
        "sim",
        "run",
        "contextual-sbijax",
        "--data-source",
        "markov",
        "--load-data-bundle",
        str(args.bundle),
        "--sbijax-package-theta",
        "markov_exact_sketch",
        "--sbijax-input-encoding",
        str(cell.input_encoding),
        "--train-docs",
        str(args.train_docs),
        "--val-docs",
        str(args.val_docs),
        "--test-docs",
        str(args.test_docs),
        "--fragment-len",
        str(cell.leaf_tokens),
        "--context-samples-per-doc",
        "1",
        "--response-signature-contexts",
        str(args.response_signature_contexts),
        "--response-signature-slices",
        str(args.response_signature_slices),
        "--embedding-dim",
        str(args.embedding_dim),
        "--state-dim",
        "25",
        "--hidden-dim",
        str(cell.hidden_dim),
        "--learned-merge-hidden-dim",
        str(cell.hidden_dim),
        "--learned-decoder-hidden-dim",
        str(cell.hidden_dim),
        "--learning-rate",
        str(cell.learning_rate),
        "--lr-schedule",
        str(args.lr_schedule),
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
        "--local-law-summary-fno-n-modes",
        str(cell.n_modes),
        "--local-law-summary-fno-n-layers",
        str(cell.n_layers),
        "--local-law-summary-fno-pooling-mode",
        str(cell.pooling_mode),
        "--seed",
        str(cell.seed),
        "--output-root",
        str(cell_root),
    ]
    cmd.extend(_objective_args(cell))
    return cmd


def _default_cells(args: argparse.Namespace) -> list[JaxRouteCell]:
    objectives = _parse_strings(args.objectives)
    encodings = _parse_strings(args.input_encodings)
    leaves = _parse_ints(args.leaves)
    seeds = _parse_ints(args.seeds)
    cells: list[JaxRouteCell] = []
    for objective in objectives:
        for encoding in encodings:
            if objective == "regime_transition_sum" and encoding != "regime_one_hot":
                continue
            for leaf in leaves:
                modes = max(1, min(int(args.fno_n_modes), int(leaf) // 2 + 1))
                for seed in seeds:
                    cells.append(
                        JaxRouteCell(
                            objective=objective,
                            input_encoding=encoding,
                            leaf_tokens=leaf,
                            hidden_dim=int(args.hidden_dim),
                            n_iter=int(args.n_iter),
                            learning_rate=float(args.learning_rate),
                            seed=seed,
                            n_modes=modes,
                            n_layers=int(args.fno_n_layers),
                            pooling_mode=str(args.fno_pooling_mode),
                        )
                    )
    return cells


def _read_completed_row(cell: JaxRouteCell, cell_root: Path, elapsed: float | None = None) -> dict[str, Any]:
    summary_path = cell_root / "summary.json"
    if not summary_path.exists():
        return {
            "cell_id": cell.cell_id,
            "status": "missing_summary",
            "route": "jax",
            "objective": cell.objective,
            "output_root": str(cell_root),
            "elapsed_sec": elapsed,
        }
    with summary_path.open() as fh:
        summary = json.load(fh)
    row = normalize_jax_summary(
        summary,
        cell_id=cell.cell_id,
        objective=cell.objective,
        output_root=str(cell_root),
        elapsed_sec=elapsed,
    )
    if row.get("summary_family") is None and cell.objective.startswith("sbijax_package"):
        row["summary_family"] = "sbijax_package"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the self-contained JAX Markov route."
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO / "outputs" / f"markov_jax_route_{_utc_stamp()}"),
    )
    parser.add_argument(
        "--bundle",
        default=(
            "outputs/_bundles/markov_hazard_panels/"
            "paper_hazard_panel_v1_t128/seed_0/base_bundle.json"
        ),
    )
    parser.add_argument(
        "--objectives",
        default=(
            "sbijax_package_nass,sbijax_package_nasss,"
            "jax_fno_node_witness,jax_fno_local_laws,regime_transition_sum"
        ),
    )
    parser.add_argument("--input-encodings", default="regime_one_hot")
    parser.add_argument("--leaves", default="16")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--train-docs", type=int, default=64)
    parser.add_argument("--val-docs", type=int, default=32)
    parser.add_argument("--test-docs", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lr-schedule", default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--fno-n-modes", type=int, default=8)
    parser.add_argument("--fno-n-layers", type=int, default=2)
    parser.add_argument("--fno-pooling-mode", default="sum", choices=["sum", "mean"])
    parser.add_argument("--response-signature-contexts", type=int, default=8)
    parser.add_argument("--response-signature-slices", type=int, default=4)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--xla-mem-fraction", default="0.35")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cells = _default_cells(args)
    manifest = {
        "schema_version": "markov_jax_route.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "route": "jax",
        "output_root": str(output_root),
        "args": vars(args),
        "cells": [asdict(cell) for cell in cells],
        "external_dependency_policy": {
            "norax": "reference_only_not_required",
            "pardax": "reference_only_not_required",
        },
    }

    if args.dry_run:
        write_route_outputs(output_root, [], title="Markov JAX Route", manifest=manifest)
        print(json.dumps(manifest, indent=2))
        return

    rows: list[dict[str, Any]] = []
    if args.aggregate_only:
        for cell in cells:
            rows.append(_read_completed_row(cell, output_root / cell.cell_id))
        write_route_outputs(output_root, rows, title="Markov JAX Route", manifest=manifest)
        print(f"summary: {output_root / 'grid_summary.csv'}")
        return

    for idx, cell in enumerate(cells, start=1):
        cell_root = output_root / cell.cell_id
        if (cell_root / "summary.json").exists():
            row = _read_completed_row(cell, cell_root, elapsed=0.0)
            rows.append(row)
            write_route_outputs(output_root, rows, title="Markov JAX Route", manifest=manifest)
            print(f"[{idx}/{len(cells)}] skip {cell.cell_id}", flush=True)
            continue
        env = os.environ.copy()
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.xla_mem_fraction)
        if args.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        cmd = _cmd_for_cell(args, cell, cell_root)
        print(f"[{idx}/{len(cells)}] start {cell.cell_id}", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=REPO, env=env)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            row = {
                **asdict(cell),
                "cell_id": cell.cell_id,
                "status": f"failed:{proc.returncode}",
                "route": "jax",
                "backend": "jax",
                "elapsed_sec": round(elapsed, 3),
                "output_root": str(cell_root),
            }
            rows.append(row)
            write_route_outputs(output_root, rows, title="Markov JAX Route", manifest=manifest)
            if not args.keep_going:
                raise SystemExit(proc.returncode)
            continue
        rows.append(_read_completed_row(cell, cell_root, elapsed=round(elapsed, 3)))
        write_route_outputs(output_root, rows, title="Markov JAX Route", manifest=manifest)
        print(f"[{idx}/{len(cells)}] done {cell.cell_id}", flush=True)

    write_route_outputs(output_root, rows, title="Markov JAX Route", manifest=manifest)
    print(f"summary: {output_root / 'grid_summary.csv'}")
    print(f"report:  {output_root / 'grid_report.md'}")


if __name__ == "__main__":
    main()
