#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ctreepo.data.prep_common import ensure_repo_on_path  # noqa: E402

REPO_ROOT = ensure_repo_on_path()

from src.ctreepo.data.splits import SPLIT_SCHEMA_VERSION  # noqa: E402
from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (  # noqa: E402
    STICKY_STRUCTURAL_V2_CELL_SPECS,
    prepare_markov_full_doc_anchor_diagnostics_data,
)


def _default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / f"markov_sticky_benchmark_data_{timestamp}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize sticky Markov benchmark corpora and prepared tree data once, "
            "then write a manifest that downstream launchers and plotters can reuse."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
    )
    parser.add_argument(
        "--benchmark-names",
        nargs="*",
        default=["recoverable_v5_t128"],
        help="Standalone benchmark names to materialize, e.g. recoverable_v5_t128.",
    )
    parser.add_argument(
        "--hardness-grid",
        default="structural_core_v2_t128",
        help="Optional structural grid to materialize in one call.",
    )
    parser.add_argument(
        "--grid-cell-ids",
        nargs="*",
        default=[str(spec["cell_id"]) for spec in STICKY_STRUCTURAL_V2_CELL_SPECS],
        help=(
            "Optional explicit structural cell ids. Defaults to the full sticky v2 grid: "
            + ", ".join(str(spec["cell_id"]) for spec in STICKY_STRUCTURAL_V2_CELL_SPECS)
        ),
    )
    parser.add_argument(
        "--train-doc-counts",
        type=int,
        nargs="+",
        default=[1024, 4096, 10240, 20480],
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument(
        "--prepared-data-root",
        type=Path,
        default=None,
        help=(
            "Optional explicit prepared-data cache root. When omitted, the default "
            "shared prepared-data directory is used."
        ),
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
    )
    return parser.parse_args()


def _config_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if args.prepared_data_root is not None:
        overrides["prepared_data_root"] = str(Path(args.prepared_data_root).expanduser())
        overrides["prepared_data_allow_create"] = True
    return overrides


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "sticky_benchmark_data_manifest.json"
    report_path = args.output_dir / "report.md"

    config_overrides = _config_overrides(args)
    prepared_payloads: List[Dict[str, Any]] = []

    for benchmark_name in list(args.benchmark_names or []):
        payload = prepare_markov_full_doc_anchor_diagnostics_data(
            benchmark_name=str(benchmark_name),
            seeds=tuple(int(seed) for seed in list(args.seeds or [])),
            train_doc_counts=tuple(int(value) for value in list(args.train_doc_counts or [])),
            use_cuda=False,
            cuda_device=None,
            torch_threads=int(args.torch_threads),
            config_overrides=config_overrides,
        )
        prepared_payloads.extend(list(payload.get("prepared") or []))

    if str(args.hardness_grid or "").strip():
        payload = prepare_markov_full_doc_anchor_diagnostics_data(
            hardness_grid=str(args.hardness_grid),
            grid_cell_ids=tuple(str(value) for value in list(args.grid_cell_ids or [])),
            seeds=tuple(int(seed) for seed in list(args.seeds or [])),
            train_doc_counts=tuple(int(value) for value in list(args.train_doc_counts or [])),
            use_cuda=False,
            cuda_device=None,
            torch_threads=int(args.torch_threads),
            config_overrides=config_overrides,
        )
        prepared_payloads.extend(list(payload.get("prepared") or []))

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_names": [str(value) for value in list(args.benchmark_names or [])],
        "hardness_grid": str(args.hardness_grid or ""),
        "grid_cell_ids": [str(value) for value in list(args.grid_cell_ids or [])],
        "train_doc_counts": [int(value) for value in list(args.train_doc_counts or [])],
        "seeds": [int(value) for value in list(args.seeds or [])],
        "prepared_data_root_override": str(args.prepared_data_root or ""),
        "prepared": prepared_payloads,
        "shared_split_schema_version": SPLIT_SCHEMA_VERSION,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Sticky Markov Benchmark Data",
        "",
        f"Generated: `{manifest['generated_at']}`",
        "",
        f"- Benchmarks: `{', '.join(manifest['benchmark_names']) or '(none)'}`",
        f"- Structural grid: `{manifest['hardness_grid'] or '(none)'}`",
        f"- Structural cells: `{', '.join(manifest['grid_cell_ids']) or '(all available)'}`",
        f"- Train doc counts: `{', '.join(str(v) for v in manifest['train_doc_counts'])}`",
        f"- Seeds: `{', '.join(str(v) for v in manifest['seeds'])}`",
        "",
        "## Prepared Roots",
        "",
    ]
    for item in prepared_payloads:
        lines.append(
            "- "
            + f"`{str(item.get('benchmark') or '')}`"
            + f" → `{str(item.get('prepared_data_root') or '')}`"
        )
    lines.append("")
    lines.append(f"Manifest: `{manifest_path}`")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
