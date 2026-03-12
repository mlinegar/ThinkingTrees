#!/usr/bin/env python3
"""Replay a saved Markov OPS-count summary from its stored config."""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.core.markov_changepoint_ops_count import (
    OPSCountConfig,
    OPSCountSummary,
    run_markov_changepoint_ops_count_experiment,
)
from src.ctreepo.sim.cli.run_markov_changepoint_ops_count import _rows_from_summary


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if len(rows) == 0:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            fieldnames.append(str(key))
            seen.add(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a saved Markov OPS-count summary.")
    parser.add_argument("--summary-json", type=Path, required=True, help="Existing summary JSON to replay.")
    parser.add_argument(
        "--csv-summary",
        type=Path,
        default=None,
        help="Optional CSV output override. Defaults to the sibling .csv path.",
    )
    parser.add_argument(
        "--device",
        choices=["inherit", "cpu", "cuda"],
        default="inherit",
        help="Runtime override for the replayed run.",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="Optional CUDA device override when --device cuda is used.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="Optional torch thread override for the replayed run.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the replayed summary JSON to stdout as well.",
    )
    return parser.parse_args()


def _normalize_eval_guidance_qs(value: Any) -> tuple[float, ...]:
    if value is None:
        return tuple()
    if isinstance(value, (list, tuple)):
        return tuple(float(x) for x in value)
    if isinstance(value, str):
        items = []
        for raw in value.replace(",", " ").split():
            token = raw.strip()
            if token:
                items.append(float(token))
        return tuple(items)
    return tuple()


def _load_config(summary_path: Path) -> OPSCountConfig:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    raw_cfg = dict(data.get("config") or {})
    allowed = {field.name for field in fields(OPSCountConfig)}
    filtered = {key: value for key, value in raw_cfg.items() if key in allowed}
    filtered["eval_guidance_qs"] = _normalize_eval_guidance_qs(filtered.get("eval_guidance_qs"))
    return OPSCountConfig(**filtered)


def _apply_runtime_overrides(
    cfg: OPSCountConfig,
    *,
    device_mode: str,
    cuda_device: int | None,
    torch_threads: int | None,
) -> OPSCountConfig:
    cfg_dict = {field.name: getattr(cfg, field.name) for field in fields(OPSCountConfig)}
    if device_mode == "cpu":
        cfg_dict["use_cuda"] = False
        cfg_dict["cuda_device"] = None
    elif device_mode == "cuda":
        cfg_dict["use_cuda"] = True
        if cuda_device is not None:
            cfg_dict["cuda_device"] = int(cuda_device)
    if torch_threads is not None:
        cfg_dict["torch_threads"] = int(torch_threads)
    return OPSCountConfig(**cfg_dict)


def _default_csv_path(summary_path: Path) -> Path:
    return summary_path.with_suffix(".csv")


def main() -> int:
    args = _parse_args()
    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        raise SystemExit(f"summary does not exist: {summary_path}")
    csv_path = Path(args.csv_summary) if args.csv_summary is not None else _default_csv_path(summary_path)

    cfg = _load_config(summary_path)
    cfg = _apply_runtime_overrides(
        cfg,
        device_mode=str(args.device),
        cuda_device=(int(args.cuda_device) if args.cuda_device is not None else None),
        torch_threads=(int(args.torch_threads) if args.torch_threads is not None else None),
    )

    summary: OPSCountSummary = run_markov_changepoint_ops_count_experiment(cfg)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary.to_json(), encoding="utf-8")
    _write_csv(csv_path, _rows_from_summary(summary))

    if args.json:
        print(summary.to_json())
    else:
        payload = {
            "summary_json": str(summary_path),
            "csv_summary": str(csv_path),
            "device": "cuda" if bool(cfg.use_cuda) else "cpu",
            "torch_threads": int(cfg.torch_threads),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
