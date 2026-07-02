#!/usr/bin/env python3
"""Round 4 overnight HLL grid for explicit JAX f/g local laws."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
    group: str
    train_docs: int
    val_docs: int
    test_docs: int
    n_iter: int
    fragment_len: int
    summary_dim: int
    estimate_weight: float
    law_architecture: str = "learned_merge"
    merge_loss: str = "mse"
    state_decoder_head: str = "mlp"
    precision: int = 8
    hash_bits: int = 64
    vocab_size: int = 128
    doc_tokens: int = 512
    hidden_dim: int = 128
    batch_size: int = 512
    response_contexts: int = 16
    response_slices: int = 8
    seed: int = 0

    @property
    def estimated_cost(self) -> float:
        train_scale = float(self.train_docs) / 10240.0
        iter_scale = float(self.n_iter) / 100.0
        leaf_scale = max(1.0, float(self.doc_tokens) / float(max(1, self.fragment_len)))
        arch_scale = 1.25 if self.law_architecture == "fully_learned" else 1.0
        loss_scale = 1.15 if self.merge_loss != "mse" else 1.0
        return train_scale * iter_scale * (1.0 + 0.08 * leaf_scale) * arch_scale * loss_scale


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _weight_label(value: float) -> str:
    text = f"{float(value):g}".replace(".", "p").replace("-", "m")
    return f"w{text}"


def _name(
    *,
    group: str,
    train_docs: int,
    fragment_len: int,
    summary_dim: int,
    estimate_weight: float,
    law_architecture: str,
    merge_loss: str,
) -> str:
    arch = {
        "learned_merge": "lm",
        "fully_learned": "fl",
    }.get(str(law_architecture), str(law_architecture))
    loss = {
        "mse": "mse",
        "nasss_jsd": "nasss",
        "nass_jsd": "nass",
    }.get(str(merge_loss), str(merge_loss).replace("_", ""))
    return (
        f"{group}_n{int(train_docs)}_L{int(fragment_len):03d}_"
        f"dz{int(summary_dim)}_{_weight_label(float(estimate_weight))}_{arch}_{loss}"
    )


def _train_spec(train_docs: int) -> tuple[int, int, int, int]:
    if int(train_docs) >= 102400:
        return int(train_docs), 4096, 4096, 300
    if int(train_docs) >= 40960:
        return int(train_docs), 2048, 2048, 500
    return int(train_docs), 1024, 1024, 600


def _make_cell(
    *,
    group: str,
    train_docs: int,
    fragment_len: int,
    summary_dim: int,
    estimate_weight: float,
    law_architecture: str = "learned_merge",
    merge_loss: str = "mse",
    n_iter: int | None = None,
    seed_offset: int = 0,
) -> Cell:
    train_docs, val_docs, test_docs, default_iter = _train_spec(train_docs)
    resolved_iter = int(default_iter if n_iter is None else n_iter)
    name = _name(
        group=group,
        train_docs=train_docs,
        fragment_len=fragment_len,
        summary_dim=summary_dim,
        estimate_weight=estimate_weight,
        law_architecture=law_architecture,
        merge_loss=merge_loss,
    )
    seed = (
        40_000
        + int(train_docs)
        + 17 * int(fragment_len)
        + 31 * int(summary_dim)
        + 101 * int(round(float(estimate_weight) * 10.0))
        + int(seed_offset)
    )
    return Cell(
        name=name,
        group=group,
        train_docs=train_docs,
        val_docs=val_docs,
        test_docs=test_docs,
        n_iter=resolved_iter,
        fragment_len=int(fragment_len),
        summary_dim=int(summary_dim),
        estimate_weight=float(estimate_weight),
        law_architecture=str(law_architecture),
        merge_loss=str(merge_loss),
        seed=int(seed),
    )


def _overnight_cells() -> list[Cell]:
    cells: list[Cell] = []
    leaf_grid = [16, 32, 64, 128, 256, 512]
    train_grid = [10240, 40960, 102400]
    summary_grid = [64, 128]

    for train_docs in train_grid:
        for fragment_len in leaf_grid:
            for summary_dim in summary_grid:
                cells.append(
                    _make_cell(
                        group="main",
                        train_docs=train_docs,
                        fragment_len=fragment_len,
                        summary_dim=summary_dim,
                        estimate_weight=1.0,
                    )
                )

    for train_docs in [10240, 102400]:
        for fragment_len in [32, 64, 128, 512]:
            for estimate_weight in [0.0, 0.1]:
                cells.append(
                    _make_cell(
                        group="est",
                        train_docs=train_docs,
                        fragment_len=fragment_len,
                        summary_dim=128,
                        estimate_weight=estimate_weight,
                        seed_offset=5_000,
                    )
                )

    for train_docs in [40960, 102400]:
        for fragment_len in [64, 128]:
            cells.append(
                _make_cell(
                    group="full",
                    train_docs=train_docs,
                    fragment_len=fragment_len,
                    summary_dim=128,
                    estimate_weight=1.0,
                    law_architecture="fully_learned",
                    n_iter=240 if int(train_docs) >= 102400 else 360,
                    seed_offset=10_000,
                )
            )
            cells.append(
                _make_cell(
                    group="nasss",
                    train_docs=train_docs,
                    fragment_len=fragment_len,
                    summary_dim=128,
                    estimate_weight=1.0,
                    merge_loss="nasss_jsd",
                    n_iter=240 if int(train_docs) >= 102400 else 360,
                    seed_offset=15_000,
                )
            )

    for fragment_len in [64, 128]:
        cells.append(
            _make_cell(
                group="wide",
                train_docs=102400,
                fragment_len=fragment_len,
                summary_dim=256,
                estimate_weight=1.0,
                n_iter=240,
                seed_offset=20_000,
            )
        )

    return sorted(cells, key=lambda cell: (-cell.estimated_cost, cell.name))


def _smoke_cells() -> list[Cell]:
    return [
        Cell(
            name="smoke_n16_L008_dz12_w0p1_lm_mse",
            group="smoke",
            train_docs=16,
            val_docs=4,
            test_docs=4,
            n_iter=1,
            fragment_len=8,
            summary_dim=12,
            estimate_weight=0.1,
            precision=4,
            vocab_size=32,
            doc_tokens=32,
            hidden_dim=8,
            batch_size=8,
            response_contexts=2,
            response_slices=2,
            seed=123,
        )
    ]


def _cells(preset: str) -> list[Cell]:
    if str(preset) == "smoke":
        return _smoke_cells()
    if str(preset) == "overnight":
        return _overnight_cells()
    raise ValueError(f"unknown preset: {preset!r}")


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


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
        "one_hot_token_ids",
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
        "--local-law-hll-estimate-weight",
        str(cell.estimate_weight),
        "--local-law-summary-family",
        "mlp",
        "--local-law-explicit-state-decoder",
        "--local-law-summary-dim",
        str(cell.summary_dim),
        "--local-law-state-decoder-head",
        cell.state_decoder_head,
        "--law-architecture",
        cell.law_architecture,
        "--merge-family",
        "mlp",
        "--local-law-merge-loss",
        cell.merge_loss,
        "--merge-nasss-n-slices",
        "16",
        "--decoder-head",
        "linear",
        "--seed",
        str(cell.seed),
        "--output-root",
        str(output_dir),
    ]


def _load_summary(cell_dir: Path) -> dict[str, Any]:
    path = cell_dir / "summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _metric(payload: dict[str, Any], split: str, key: str) -> Any:
    split_payload = payload.get("diagnostics", {}).get(split, {})
    return split_payload.get(key) if isinstance(split_payload, dict) else None


def _history_metric(payload: dict[str, Any], key: str) -> Any:
    history = payload.get("history", [])
    if not isinstance(history, list):
        return None
    rows = [row for row in history if isinstance(row, dict) and int(row.get("iteration", -1)) >= 0]
    if not rows:
        return None
    return rows[-1].get(key)


def _row_for_cell(output_root: Path, cell: Cell) -> dict[str, Any]:
    cell_dir = output_root / cell.name
    payload = _load_summary(cell_dir)
    provenance = payload.get("provenance", {}) if isinstance(payload, dict) else {}
    params = payload.get("params", {}) if isinstance(payload, dict) else {}
    row: dict[str, Any] = {
        **asdict(cell),
        "estimated_cost": f"{cell.estimated_cost:.6f}",
        "status": payload.get("status", "missing") if payload else "missing",
        "output_dir": str(cell_dir),
        "exit_code": None,
        "paper_notation_factorization": (
            provenance.get("paper_notation_factorization")
            if isinstance(provenance, dict)
            else None
        ),
        "provenance_g_summary_dim": (
            provenance.get("g_summary_dim") if isinstance(provenance, dict) else None
        ),
        "provenance_f_state_decoder_kind": (
            provenance.get("f_state_decoder_kind") if isinstance(provenance, dict) else None
        ),
        "params_summary_dim_effective": (
            params.get("summary_dim_effective") if isinstance(params, dict) else None
        ),
    }
    exit_path = cell_dir / "exit_code.txt"
    if exit_path.exists():
        row["exit_code"] = exit_path.read_text(encoding="utf-8").strip()
    elapsed_path = cell_dir / "elapsed_seconds.txt"
    if elapsed_path.exists():
        row["elapsed_seconds"] = elapsed_path.read_text(encoding="utf-8").strip()
    for split in ("train", "val", "test"):
        for key in (
            "theta_mae",
            "hll_register_mae",
            "hll_estimate_raw_mae",
            "hll_estimate_norm_mae",
            "contextual_mae",
            "contextual_raw_mae",
            "eps_leaf",
            "eps_merge",
            "eps_idemp",
            "pred_truth_corr",
            "hll_merge_register_exact_frac",
            "hll_merge_register_rounded_mae",
            "hll_suff_r2",
            "hll_suff_probe_mae",
        ):
            row[f"{split}_{key}"] = _metric(payload, split, key)
    for key in (
        "train_hll_estimate_mse",
        "val_hll_estimate_mse",
        "train_l1_leaf_mse",
        "val_l1_leaf_mse",
        "train_l2_merge_mse",
        "val_l2_merge_mse",
        "train_idempotence_mse",
        "val_idempotence_mse",
    ):
        row[f"final_{key}"] = _history_metric(payload, key)
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


def _write_readme(output_root: Path, cells: list[Cell], *, preset: str) -> None:
    lines = [
        "# HLL JAX Local-Law Round 4 Overnight Grid",
        "",
        f"Preset: `{preset}`.",
        "",
        "Purpose: run the paper-notation explicit JAX f/g HLL lane overnight.",
        "",
        "All non-smoke cells use:",
        "",
        "- `g_phi(x or z_left,z_right) -> z`",
        "- `f_psi(z) -> theta_hat` via `--local-law-explicit-state-decoder`",
        "- HLL register sketch theta with precision p=8",
        "- estimate-aware HLL cardinality auxiliary, except explicit weight controls",
        "",
        "Cells:",
    ]
    for idx, cell in enumerate(cells):
        lines.append(
            f"- {idx:02d}: `{cell.name}` group={cell.group} "
            f"train={cell.train_docs} leaf={cell.fragment_len} dz={cell.summary_dim} "
            f"weight={cell.estimate_weight:g} arch={cell.law_architecture} "
            f"loss={cell.merge_loss} iter={cell.n_iter}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_root / "cell_manifest.json").write_text(
        json.dumps([asdict(cell) for cell in cells], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
    parser.add_argument(
        "--output-root",
        default="",
        help=(
            "Output root. Defaults to "
            "outputs/hll_jax_local_law_round4_overnight_grid_<UTC stamp>."
        ),
    )
    parser.add_argument("--preset", choices=["smoke", "overnight"], default="overnight")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = (
        Path(str(args.output_root)).resolve()
        if str(args.output_root).strip()
        else (
            REPO / "outputs" / f"hll_jax_local_law_round4_overnight_grid_{_utc_stamp()}"
        ).resolve()
    )
    cells = _cells(str(args.preset))
    _write_readme(output_root, cells, preset=str(args.preset))
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
                        "preset": str(args.preset),
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
    if not args.aggregate_only:
        exit_codes = [_run_cell(output_root, cell, force=bool(args.force)) for cell in shard_cells]
        _write_grid_summary(
            output_root,
            [_row_for_cell(output_root, cell) for cell in shard_cells],
            name=f"grid_summary_shard_{shard_index:02d}.csv",
        )
        if any(code != 0 for code in exit_codes):
            return 1
    _write_grid_summary(
        output_root,
        [_row_for_cell(output_root, cell) for cell in cells],
        name="grid_summary.csv",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "output_root": str(output_root),
                "preset": str(args.preset),
                "cells": len(cells),
                "shard_cells": len(shard_cells),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
