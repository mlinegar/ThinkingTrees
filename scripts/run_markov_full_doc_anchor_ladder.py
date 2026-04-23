#!/usr/bin/env python3
"""Run a staged buildout for the standalone full-document anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.core.full_doc_anchor_ladder import (  # noqa: E402
    render_full_doc_anchor_ladder_markdown,
    resolve_markov_full_doc_anchor_ladder,
    run_markov_full_doc_anchor_ladder,
    write_full_doc_anchor_ladder_csv,
)
from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (  # noqa: E402
    DEFAULT_DIAGNOSTIC_BASELINE_FAMILIES,
    DEFAULT_STRUCTURAL_CORE_BASELINE_FAMILIES,
    VALID_BASELINE_FAMILIES,
    VALID_HARDNESS_GRIDS,
    load_markov_full_doc_anchor_diagnostics_from_output_dir,
    render_full_doc_anchor_diagnostic_markdown,
    run_markov_full_doc_anchor_diagnostics,
    write_full_doc_anchor_diagnostic_csv,
)
from src.ctreepo.sim.core.full_doc_config_codec import (  # noqa: E402
    runtime_config_overrides_from_config_like,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep a staged buildout showing when the standalone full-doc anchor starts to work."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["ladder", "diagnostic"],
        default="ladder",
    )
    parser.add_argument(
        "--preset",
        choices=[
            "quick_buildout",
            "recoverable_budget_ladder",
            "recoverable_reproduction_ladder",
        ],
        default="quick_buildout",
    )
    parser.add_argument(
        "--benchmark",
        choices=["recoverable_v4", "recoverable", "demo_v1", "smoke"],
        default="recoverable_v4",
        help="Diagnostic benchmark. Used only with --mode diagnostic.",
    )
    parser.add_argument(
        "--hardness-grid",
        choices=list(VALID_HARDNESS_GRIDS),
        default="",
        help="Optional structural-hardening grid for diagnostic mode.",
    )
    parser.add_argument(
        "--grid-cell-ids",
        nargs="*",
        default=None,
        help="Optional subset of structural grid cell ids, e.g. r8_seg7to9.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Seed sweep for diagnostic mode. Defaults to 0 1 2 3 4.",
    )
    parser.add_argument(
        "--train-doc-counts",
        nargs="*",
        type=int,
        default=None,
        help="Train document counts for diagnostic mode. Defaults to the benchmark's 1x/2x/5x/10x sweep.",
    )
    parser.add_argument(
        "--baseline-families",
        nargs="*",
        choices=list(VALID_BASELINE_FAMILIES),
        default=None,
        help=(
            "Baseline families for diagnostic mode. Defaults to "
            "official_fno/official_fno_sumlen/cnn1d/palette_block_exact for "
            "structural_core_v1, otherwise the standard full-doc diagnostic set."
        ),
    )
    parser.add_argument(
        "--emit-confusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Emit per-split confusion matrices and per-class precision/recall in diagnostic mode.",
    )
    parser.add_argument(
        "--fno-pooling",
        choices=["mean", "sum"],
        default="mean",
        help="Full-doc FNO pooling mode for diagnostic mode.",
    )
    parser.add_argument(
        "--fno-concat-length-feature",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Concatenate normalized sequence length into the full-doc FNO head in diagnostic mode.",
    )
    parser.add_argument(
        "--fno-transition-channel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the adjacent-pair transition input channel for the full-doc FNO path in diagnostic mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/markov_full_doc_anchor_ladder",
    )
    parser.add_argument(
        "--json-summary",
        type=str,
        default="",
        help="Optional JSON output path. Defaults to <output-dir>/summary.json.",
    )
    parser.add_argument(
        "--csv-summary",
        type=str,
        default="",
        help="Optional CSV output path. Defaults to <output-dir>/summary.csv.",
    )
    parser.add_argument(
        "--markdown-summary",
        type=str,
        default="",
        help="Optional Markdown output path. Defaults to <output-dir>/summary.md.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cuda-device", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--skip-existing-stages",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--aggregate-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Aggregate saved diagnostic run JSONs from <output-dir> instead of rerunning models.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    json_summary = (
        Path(str(args.json_summary))
        if str(args.json_summary).strip()
        else output_dir / "summary.json"
    )
    csv_summary = (
        Path(str(args.csv_summary))
        if str(args.csv_summary).strip()
        else output_dir / "summary.csv"
    )
    markdown_summary = (
        Path(str(args.markdown_summary))
        if str(args.markdown_summary).strip()
        else output_dir / "summary.md"
    )
    if args.cpu:
        args.device = "cpu"
    use_cuda = args.device in ("auto", "cuda")
    if args.device == "auto":
        use_cuda = torch.cuda.is_available()

    if str(args.mode) == "diagnostic":
        if args.baseline_families:
            baseline_families = tuple(args.baseline_families)
        elif str(args.hardness_grid).strip().lower() == "structural_core_v1":
            baseline_families = DEFAULT_STRUCTURAL_CORE_BASELINE_FAMILIES
        else:
            baseline_families = DEFAULT_DIAGNOSTIC_BASELINE_FAMILIES
        if bool(args.aggregate_only):
            payload = load_markov_full_doc_anchor_diagnostics_from_output_dir(output_dir)
        else:
            payload = run_markov_full_doc_anchor_diagnostics(
                benchmark_name=str(args.benchmark),
                hardness_grid=str(args.hardness_grid),
                grid_cell_ids=tuple(args.grid_cell_ids or ()),
                seeds=tuple(args.seeds) if args.seeds else (0, 1, 2, 3, 4),
                train_doc_counts=tuple(args.train_doc_counts or ()),
                baseline_families=baseline_families,
                emit_confusion=bool(args.emit_confusion),
                output_dir=output_dir,
                use_cuda=bool(use_cuda),
                cuda_device=int(args.cuda_device) if args.cuda_device is not None else None,
                torch_threads=int(args.torch_threads),
                config_overrides=runtime_config_overrides_from_config_like(
                    {
                        "doc_sequence_fno_pooling": str(args.fno_pooling),
                        "doc_sequence_fno_concat_length_feature": bool(
                            args.fno_concat_length_feature
                        ),
                        "doc_sequence_fno_include_transition_channel": bool(
                            args.fno_transition_channel
                        ),
                    }
                ),
            )
        json_summary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        write_full_doc_anchor_diagnostic_csv(
            csv_summary,
            list(payload.get("aggregate_rows") or []),
        )
        markdown_summary.write_text(
            render_full_doc_anchor_diagnostic_markdown(payload),
            encoding="utf-8",
        )
        summary_stub = {
            "json_summary": str(json_summary),
            "csv_summary": str(csv_summary),
            "markdown_summary": str(markdown_summary),
            "runs_csv": str(payload.get("runs_csv", "")),
            "aggregate_csv": str(payload.get("aggregate_csv", "")),
            "heatmap_csv": str(payload.get("heatmap_csv", "")),
            "runs": len(payload.get("runs") or []),
            "aggregate_rows": len(payload.get("aggregate_rows") or []),
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(json.dumps(summary_stub, indent=2, sort_keys=True))
        return 0

    stages = resolve_markov_full_doc_anchor_ladder(preset=str(args.preset))
    payload = run_markov_full_doc_anchor_ladder(
        stage_specs=stages,
        output_dir=output_dir,
        use_cuda=bool(use_cuda),
        cuda_device=int(args.cuda_device) if args.cuda_device is not None else None,
        torch_threads=int(args.torch_threads),
        skip_existing=bool(args.skip_existing_stages),
        preset=str(args.preset),
    )
    rows = [
        dict(row)
        for row in json.loads(
            json.dumps(payload, sort_keys=True)
        ).get("stages", [])
    ]
    json_summary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_full_doc_anchor_ladder_csv(
        csv_summary,
        [
            {
                key: value
                for key, value in row.items()
                if not isinstance(value, (dict, list))
            }
            for row in rows
        ],
    )
    markdown_summary.write_text(
        render_full_doc_anchor_ladder_markdown(payload),
        encoding="utf-8",
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "json_summary": str(json_summary),
                    "csv_summary": str(csv_summary),
                    "markdown_summary": str(markdown_summary),
                    "stages": len(payload.get("stages") or []),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
