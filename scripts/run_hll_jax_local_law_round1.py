#!/usr/bin/env python3
"""Round 1 HLL screen for the JAX learned-local-laws lane.

This is the HLL analogue of the Markov local-law F/G screens: same
``probe_contextual_sbijax.py`` learned-local-laws path, but with
``package_theta=hll_register_sketch`` and HLL union/cardinality contexts.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "scripts" / "probe_contextual_sbijax.py"


@dataclass(frozen=True)
class Cell:
    name: str
    train_docs: int
    val_docs: int
    test_docs: int
    n_iter: int
    input_encoding: str
    summary_family: str
    law_architecture: str
    merge_family: str
    merge_loss: str
    precision: int = 8
    hash_bits: int = 64
    vocab_size: int = 512
    doc_tokens: int = 512
    fragment_len: int = 64
    hidden_dim: int = 128
    batch_size: int = 128
    response_contexts: int = 16
    response_slices: int = 8
    seed: int = 0


def _cells() -> list[Cell]:
    specs = [
        ("mlp_exact_mse", "one_hot_token_ids", "mlp", "analytic", "mlp", "mse"),
        (
            "mlp_learned_mse",
            "one_hot_token_ids",
            "mlp",
            "learned_merge",
            "mlp",
            "mse",
        ),
        (
            "mlp_learned_nasss",
            "one_hot_token_ids",
            "mlp",
            "learned_merge",
            "mlp",
            "nasss_jsd",
        ),
        (
            "fno_learned_nasss",
            "one_hot_token_ids",
            "jax_fno",
            "learned_merge",
            "mlp",
            "nasss_jsd",
        ),
        (
            "fno_fnog_nasss",
            "one_hot_token_ids",
            "jax_fno",
            "learned_merge",
            "fno_rep",
            "nasss_jsd",
        ),
        (
            "norm_mlp_learned_nasss",
            "normalized_token_ids",
            "mlp",
            "learned_merge",
            "mlp",
            "nasss_jsd",
        ),
    ]
    out: list[Cell] = []
    for train_docs, val_docs, test_docs, n_iter in [
        (1024, 256, 256, 120),
        (4096, 512, 512, 60),
    ]:
        for idx, (
            label,
            input_encoding,
            summary_family,
            law_architecture,
            merge_family,
            merge_loss,
        ) in enumerate(specs):
            out.append(
                Cell(
                    name=f"p8_n{train_docs}_{label}",
                    train_docs=train_docs,
                    val_docs=val_docs,
                    test_docs=test_docs,
                    n_iter=n_iter,
                    input_encoding=input_encoding,
                    summary_family=summary_family,
                    law_architecture=law_architecture,
                    merge_family=merge_family,
                    merge_loss=merge_loss,
                    seed=1000 + train_docs + idx,
                )
            )
    return out


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _load_summary(cell_dir: Path) -> dict[str, Any]:
    path = cell_dir / "summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _metric(payload: dict[str, Any], split: str, key: str) -> Any:
    diag = payload.get("diagnostics", {})
    if not isinstance(diag, dict):
        return None
    split_payload = diag.get(split, {})
    if not isinstance(split_payload, dict):
        return None
    return split_payload.get(key)


def _row_for_cell(output_root: Path, cell: Cell) -> dict[str, Any]:
    cell_dir = output_root / cell.name
    payload = _load_summary(cell_dir)
    row: dict[str, Any] = {
        **asdict(cell),
        "status": payload.get("status", "missing") if payload else "missing",
        "output_dir": str(cell_dir),
        "exit_code": None,
    }
    exit_path = cell_dir / "exit_code.txt"
    if exit_path.exists():
        row["exit_code"] = exit_path.read_text(encoding="utf-8").strip()
    for split in ("train", "val", "test"):
        for key in (
            "theta_mae",
            "hll_register_mae",
            "hll_estimate_raw_mae",
            "contextual_mae",
            "contextual_raw_mae",
            "eps_leaf",
            "eps_merge",
            "eps_idemp",
            "pred_truth_corr",
        ):
            row[f"{split}_{key}"] = _metric(payload, split, key)
    metric_summary = payload.get("metric_summary", {})
    if isinstance(metric_summary, dict):
        row["metric_summary_law_set_id"] = metric_summary.get("law_set_id")
        row["metric_summary_baseline_role"] = metric_summary.get("baseline_role")
    return row


def _write_grid_summary(output_root: Path, rows: list[dict[str, Any]], name: str) -> None:
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


def _write_readme(output_root: Path, cells: list[Cell]) -> None:
    lines = [
        "# HLL JAX Local-Law Round 1",
        "",
        "Package: `hll_register_sketch`.",
        "Context: HLL union/cardinality query.",
        "Objective: `learned_local_laws` with dense local-law supervision.",
        "",
        "Cells:",
    ]
    for idx, cell in enumerate(cells):
        lines.append(
            f"- {idx:02d}: `{cell.name}` n={cell.train_docs} "
            f"encoding={cell.input_encoding} summary={cell.summary_family} "
            f"arch={cell.law_architecture} merge={cell.merge_family} "
            f"loss={cell.merge_loss} iter={cell.n_iter}"
        )
    lines.extend(
        [
            "",
            "Per-shard summaries are `grid_summary_shard_*.csv`.",
            "Run with `--aggregate-only` after completion to write `grid_summary.csv`.",
        ]
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_root / "cell_manifest.json").write_text(
        json.dumps([asdict(cell) for cell in cells], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _command(cell: Cell, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(PROBE),
        "--data-source",
        "hll",
        "--sbijax-trainer",
        "learned_local_laws",
        "--sbijax-method",
        "nasss",
        "--sbijax-package-theta",
        "hll_register_sketch",
        "--sbijax-input-encoding",
        cell.input_encoding,
        "--train-docs",
        str(cell.train_docs),
        "--val-docs",
        str(cell.val_docs),
        "--test-docs",
        str(cell.test_docs),
        "--doc-tokens",
        str(cell.doc_tokens),
        "--fragment-len",
        str(cell.fragment_len),
        "--vocab-size",
        str(cell.vocab_size),
        "--hll-precision",
        str(cell.precision),
        "--hll-hash-bits",
        str(cell.hash_bits),
        "--context-samples-per-doc",
        "1",
        "--response-signature-contexts",
        str(cell.response_contexts),
        "--response-signature-slices",
        str(cell.response_slices),
        "--hidden-dim",
        str(cell.hidden_dim),
        "--learning-rate",
        "0.0003",
        "--lr-schedule",
        "cosine",
        "--n-iter",
        str(cell.n_iter),
        "--batch-size",
        str(cell.batch_size),
        "--local-law-supervision-mode",
        "dense_exact",
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
        "--local-law-summary-family",
        cell.summary_family,
        "--local-law-summary-fno-n-modes",
        "32",
        "--local-law-summary-fno-n-layers",
        "2",
        "--local-law-summary-fno-pooling-mode",
        "mean",
        "--law-architecture",
        cell.law_architecture,
        "--merge-family",
        cell.merge_family,
        "--local-law-merge-loss",
        cell.merge_loss,
        "--merge-nasss-n-slices",
        "16",
        "--merge-fno-n-modes",
        "32",
        "--merge-fno-n-layers",
        "2",
        "--merge-fno-hidden-channels",
        "32",
        "--decoder-head",
        "linear",
        "--seed",
        str(cell.seed),
        "--output-root",
        str(output_dir),
    ]


def _run_cell(output_root: Path, cell: Cell, *, force: bool) -> int:
    cell_dir = output_root / cell.name
    summary_path = cell_dir / "summary.json"
    if summary_path.exists() and not force:
        print(f"[skip] {cell.name}: summary exists", flush=True)
        return 0
    cell_dir.mkdir(parents=True, exist_ok=True)
    cmd = _command(cell, cell_dir)
    (cell_dir / "command.json").write_text(
        json.dumps(cmd, indent=2) + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
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
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(str(args.output_root)).resolve()
    cells = _cells()
    _write_readme(output_root, cells)
    shard_index = int(args.shard_index)
    num_shards = max(1, int(args.num_shards))
    shard_cells = [cell for idx, cell in enumerate(cells) if idx % num_shards == shard_index]
    if args.plan_only:
        payload = {
            "output_root": str(output_root),
            "num_cells": len(cells),
            "shard_index": shard_index,
            "num_shards": num_shards,
            "shard_cells": [asdict(cell) for cell in shard_cells],
        }
        print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
        return 0
    if not args.aggregate_only:
        exit_codes = [_run_cell(output_root, cell, force=bool(args.force)) for cell in shard_cells]
        shard_rows = [_row_for_cell(output_root, cell) for cell in shard_cells]
        _write_grid_summary(
            output_root,
            shard_rows,
            name=f"grid_summary_shard_{shard_index:02d}.csv",
        )
        if any(code != 0 for code in exit_codes):
            return 1
    all_rows = [_row_for_cell(output_root, cell) for cell in cells]
    _write_grid_summary(output_root, all_rows, name="grid_summary.csv")
    print(
        json.dumps(
            {
                "status": "ok",
                "output_root": str(output_root),
                "cells": len(cells),
                "shard_cells": len(shard_cells),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
