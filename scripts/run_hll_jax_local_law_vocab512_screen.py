#!/usr/bin/env python3
"""Vocab-512 HLL screen for explicit JAX f/g local laws."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from run_hll_jax_local_law_round4_overnight_grid import Cell, _command


REPO = Path(__file__).resolve().parents[1]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _weight_label(value: float) -> str:
    return f"w{float(value):g}".replace(".", "p").replace("-", "m")


def _cell_name(
    *,
    group: str,
    train_docs: int,
    fragment_len: int,
    summary_dim: int,
    estimate_weight: float,
) -> str:
    return (
        f"v512_{group}_n{int(train_docs)}_L{int(fragment_len):03d}_"
        f"dz{int(summary_dim)}_{_weight_label(float(estimate_weight))}_lm_mse"
    )


def _make_cell(
    *,
    group: str,
    fragment_len: int,
    summary_dim: int,
    estimate_weight: float,
    seed_offset: int = 0,
) -> Cell:
    train_docs = 102400
    seed = (
        512_000
        + int(train_docs)
        + 17 * int(fragment_len)
        + 31 * int(summary_dim)
        + 101 * int(round(float(estimate_weight) * 10.0))
        + int(seed_offset)
    )
    return Cell(
        name=_cell_name(
            group=group,
            train_docs=train_docs,
            fragment_len=int(fragment_len),
            summary_dim=int(summary_dim),
            estimate_weight=float(estimate_weight),
        ),
        group=group,
        train_docs=train_docs,
        val_docs=4096,
        test_docs=4096,
        n_iter=300,
        fragment_len=int(fragment_len),
        summary_dim=int(summary_dim),
        estimate_weight=float(estimate_weight),
        precision=8,
        hash_bits=64,
        vocab_size=512,
        doc_tokens=512,
        hidden_dim=128,
        batch_size=512,
        response_contexts=16,
        response_slices=8,
        seed=int(seed),
    )


def _screen_cells() -> list[Cell]:
    cells: list[Cell] = []
    for fragment_len in [16, 32, 64, 128, 256, 512]:
        for summary_dim in [64, 128]:
            cells.append(
                _make_cell(
                    group="main",
                    fragment_len=int(fragment_len),
                    summary_dim=int(summary_dim),
                    estimate_weight=1.0,
                )
            )
    for estimate_weight in [0.0, 0.1]:
        cells.append(
            _make_cell(
                group="est",
                fragment_len=512,
                summary_dim=128,
                estimate_weight=float(estimate_weight),
                seed_offset=5_000,
            )
        )
    return sorted(cells, key=lambda cell: (-cell.estimated_cost, cell.name))


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _write_manifest(output_root: Path, cells: list[Cell]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HLL JAX Local-Law Vocab-512 Screen",
        "",
        "Purpose: test whether the Round 4 raw-MAE U-shape was driven by the 128-token",
        "support ceiling by repeating the 102400-doc leaf grid at `vocab_size=512`.",
        "",
        "Cells:",
    ]
    for idx, cell in enumerate(cells):
        lines.append(
            f"- {idx:02d}: `{cell.name}` group={cell.group} "
            f"train={cell.train_docs} vocab={cell.vocab_size} leaf={cell.fragment_len} "
            f"dz={cell.summary_dim} weight={cell.estimate_weight:g} iter={cell.n_iter}"
        )
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_root / "cell_manifest.json").write_text(
        json.dumps([asdict(cell) for cell in cells], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _status_row(output_root: Path, cell: Cell) -> dict[str, Any]:
    cell_dir = output_root / cell.name
    row: dict[str, Any] = {
        **asdict(cell),
        "estimated_cost": f"{cell.estimated_cost:.6f}",
        "output_dir": str(cell_dir),
        "summary_exists": (cell_dir / "summary.json").exists(),
        "exit_code": "",
        "elapsed_seconds": "",
    }
    exit_path = cell_dir / "exit_code.txt"
    if exit_path.exists():
        row["exit_code"] = exit_path.read_text(encoding="utf-8").strip()
    elapsed_path = cell_dir / "elapsed_seconds.txt"
    if elapsed_path.exists():
        row["elapsed_seconds"] = elapsed_path.read_text(encoding="utf-8").strip()
    return row


def _write_status_csv(output_root: Path, rows: list[dict[str, Any]], *, name: str) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path = output_root / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})
    tmp.replace(path)


def _run_cell(output_root: Path, cell: Cell, *, force: bool) -> int:
    cell_dir = output_root / cell.name
    if (cell_dir / "summary.json").exists() and not force:
        print(f"[skip] {cell.name}: summary exists", flush=True)
        return 0
    cell_dir.mkdir(parents=True, exist_ok=True)
    cmd = _command(cell, cell_dir)
    (cell_dir / "command.json").write_text(json.dumps(cmd, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    env.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    start = time.time()
    print(f"[run] {cell.name}", flush=True)
    with (cell_dir / "run.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.time() - start
    (cell_dir / "exit_code.txt").write_text(f"{proc.returncode}\n", encoding="utf-8")
    (cell_dir / "elapsed_seconds.txt").write_text(f"{elapsed:.3f}\n", encoding="utf-8")
    print(f"[done] {cell.name} exit={proc.returncode} elapsed={elapsed:.1f}s", flush=True)
    return int(proc.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = (
        Path(str(args.output_root)).resolve()
        if str(args.output_root).strip()
        else (REPO / "outputs" / f"hll_jax_local_law_vocab512_screen_{_utc_stamp()}").resolve()
    )
    cells = _screen_cells()
    _write_manifest(output_root, cells)
    shard_index = int(args.shard_index)
    num_shards = max(1, int(args.num_shards))
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    shard_cells = [cell for idx, cell in enumerate(cells) if idx % num_shards == shard_index]
    if args.plan_only:
        print(
            json.dumps(
                _json_safe(
                    {
                        "output_root": str(output_root),
                        "num_cells": len(cells),
                        "shard_index": shard_index,
                        "num_shards": num_shards,
                        "shard_cells": [asdict(cell) for cell in shard_cells],
                        "shard_estimated_cost": sum(cell.estimated_cost for cell in shard_cells),
                        "total_estimated_cost": sum(cell.estimated_cost for cell in cells),
                    }
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not args.status_only:
        exit_codes = [_run_cell(output_root, cell, force=bool(args.force)) for cell in shard_cells]
    else:
        exit_codes = []
    _write_status_csv(
        output_root,
        [_status_row(output_root, cell) for cell in shard_cells],
        name=f"grid_status_shard_{shard_index:02d}.csv",
    )
    _write_status_csv(
        output_root,
        [_status_row(output_root, cell) for cell in cells],
        name="grid_status.csv",
    )
    return 1 if any(code != 0 for code in exit_codes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
