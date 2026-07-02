#!/usr/bin/env python3
"""Repair Round 4 HLL grid summary CSVs without loading huge JSON summaries."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import mmap
from pathlib import Path
import re
from typing import Any

from run_hll_jax_local_law_round4_overnight_grid import Cell, _cells, _write_grid_summary


SPLIT_METRIC_KEYS = (
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
)

FINAL_HISTORY_KEYS = (
    "train_hll_estimate_mse",
    "val_hll_estimate_mse",
    "train_l1_leaf_mse",
    "val_l1_leaf_mse",
    "train_l2_merge_mse",
    "val_l2_merge_mse",
    "train_idempotence_mse",
    "val_idempotence_mse",
)

_SCALAR = rb"(null|true|false|-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|\"(?:\\.|[^\"\\])*\")"
_STATUS_RE = re.compile(rb'"status"\s*:\s*("ok"|"error"|"failed"|"missing")')
_PROVENANCE_RE = re.compile(
    rb'"(paper_notation_factorization|g_summary_dim|f_state_decoder_kind|'
    rb'summary_dim_effective|local_law_summary_dim)"\s*:\s*' + _SCALAR
)
_SPLIT_RE = re.compile(rb'\n    "(train|val|test)"\s*:\s*\{')
_HISTORY_RE = re.compile(rb'\n  "history"\s*:')
_METRIC_RE = re.compile(
    rb'"('
    + b"|".join(re.escape(key.encode("utf-8")) for key in SPLIT_METRIC_KEYS)
    + rb')"\s*:\s*'
    + _SCALAR
)


def _decode_scalar(raw: bytes) -> Any:
    text = raw.decode("utf-8")
    if text == "null":
        return None
    if text == "true":
        return True
    if text == "false":
        return False
    if text.startswith('"'):
        return json.loads(text)
    try:
        value = float(text)
    except ValueError:
        return text
    if value.is_integer() and "." not in text and "e" not in text.lower():
        return int(value)
    return value


def _last_training_history(history_path: Path) -> dict[str, Any]:
    if not history_path.exists():
        return {}
    last: dict[str, Any] = {}
    with history_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and int(row.get("iteration", -1)) >= 0:
                last = row
    return last


def _history_value(history: dict[str, Any], key: str) -> Any:
    if key in history:
        return history.get(key)
    if key.endswith("_idempotence_mse"):
        return history.get(key.replace("_idempotence_mse", "_l3_idempotence_mse"))
    return None


def _summary_scalars(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "missing",
        "paper_notation_factorization": None,
        "provenance_g_summary_dim": None,
        "provenance_f_state_decoder_kind": None,
        "params_summary_dim_effective": None,
    }
    if not path.exists():
        return out
    if path.stat().st_size == 0:
        out["status"] = "empty"
        return out
    with path.open("rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            status = _STATUS_RE.search(mm)
            if status is not None:
                out["status"] = _decode_scalar(status.group(1))

            for match in _PROVENANCE_RE.finditer(mm):
                key = match.group(1).decode("utf-8")
                value = _decode_scalar(match.group(2))
                if key == "paper_notation_factorization":
                    out["paper_notation_factorization"] = value
                elif key in {"g_summary_dim", "local_law_summary_dim"}:
                    if out["provenance_g_summary_dim"] is None:
                        out["provenance_g_summary_dim"] = value
                elif key == "f_state_decoder_kind":
                    out["provenance_f_state_decoder_kind"] = value
                elif key == "summary_dim_effective":
                    out["params_summary_dim_effective"] = value

            split_matches = list(_SPLIT_RE.finditer(mm))
            history_match = _HISTORY_RE.search(mm)
            history_start = history_match.start() if history_match is not None else len(mm)
            for idx, split_match in enumerate(split_matches):
                split = split_match.group(1).decode("utf-8")
                if split not in {"train", "val", "test"}:
                    continue
                start = split_match.end()
                end = (
                    split_matches[idx + 1].start()
                    if idx + 1 < len(split_matches)
                    else history_start
                )
                if start >= end:
                    continue
                segment = mm[start:end]
                for metric_match in _METRIC_RE.finditer(segment):
                    key = metric_match.group(1).decode("utf-8")
                    out[f"{split}_{key}"] = _decode_scalar(metric_match.group(2))
        finally:
            mm.close()
    return out


def _row_for_cell(output_root: Path, cell: Cell) -> dict[str, Any]:
    cell_dir = output_root / cell.name
    row: dict[str, Any] = {
        **asdict(cell),
        "estimated_cost": f"{cell.estimated_cost:.6f}",
        "status": "missing",
        "output_dir": str(cell_dir),
        "exit_code": None,
        "paper_notation_factorization": None,
        "provenance_g_summary_dim": None,
        "provenance_f_state_decoder_kind": None,
        "params_summary_dim_effective": None,
    }
    exit_path = cell_dir / "exit_code.txt"
    if exit_path.exists():
        row["exit_code"] = exit_path.read_text(encoding="utf-8").strip()
    elapsed_path = cell_dir / "elapsed_seconds.txt"
    if elapsed_path.exists():
        row["elapsed_seconds"] = elapsed_path.read_text(encoding="utf-8").strip()

    row.update(_summary_scalars(cell_dir / "summary.json"))
    history = _last_training_history(cell_dir / "history.jsonl")
    for key in FINAL_HISTORY_KEYS:
        row[f"final_{key}"] = _history_value(history, key)
    return row


def _cells_from_manifest(output_root: Path) -> list[Cell]:
    path = output_root / "cell_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected list manifest at {path}")
    cells: list[Cell] = []
    field_names = set(Cell.__dataclass_fields__)
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"invalid manifest cell in {path}: {item!r}")
        kwargs = {key: item[key] for key in field_names if key in item}
        cells.append(Cell(**kwargs))
    return cells


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--preset", choices=["overnight", "smoke"], default="overnight")
    parser.add_argument(
        "--from-manifest",
        action="store_true",
        help="Read cell definitions from output-root/cell_manifest.json.",
    )
    parser.add_argument("--num-shards", type=int, default=4)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).resolve()
    cells = (
        _cells_from_manifest(output_root)
        if bool(args.from_manifest)
        else _cells(str(args.preset))
    )
    rows = [_row_for_cell(output_root, cell) for cell in cells]
    _write_grid_summary(output_root, rows, name="grid_summary.csv")
    num_shards = max(1, int(args.num_shards))
    for shard_index in range(num_shards):
        shard_rows = [row for idx, row in enumerate(rows) if idx % num_shards == shard_index]
        _write_grid_summary(
            output_root,
            shard_rows,
            name=f"grid_summary_shard_{shard_index:02d}.csv",
        )
    status_counts: dict[str, int] = {}
    exit_counts: dict[str, int] = {}
    for row in rows:
        status_counts[str(row.get("status"))] = status_counts.get(str(row.get("status")), 0) + 1
        exit_counts[str(row.get("exit_code"))] = exit_counts.get(str(row.get("exit_code")), 0) + 1
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "rows": len(rows),
                "status_counts": status_counts,
                "exit_counts": exit_counts,
            },
            sort_keys=True,
        )
    )
    return 0 if status_counts == {"ok": len(rows)} and exit_counts == {"0": len(rows)} else 1


if __name__ == "__main__":
    raise SystemExit(main())
