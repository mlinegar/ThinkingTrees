#!/usr/bin/env python3
"""Run a small controlled grid for the clean minimal unified-g/f lane.

This intentionally targets ``CleanUnifiedNO`` via ``probe_clean_unified_no.py``,
not the production ``FNOCountSketch(tree_model_version="unified_g")`` path.
Use it for paper-facing Markov smoke grids where the important question is
whether the reference contract survives leaf-size, mode, and capacity changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GridCell:
    cell_id: str
    objective: str
    leaf_tokens: int
    channels: int
    g_n_modes: int
    batch_size: int
    seed: int


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _parse_int_grid(raw: str) -> list[int]:
    values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError(f"empty integer grid: {raw!r}")
    return values


def _parse_str_grid(raw: str) -> list[str]:
    values = [str(part).strip() for part in str(raw).split(",") if str(part).strip()]
    if not values:
        raise ValueError(f"empty string grid: {raw!r}")
    return values


def _parse_batch_size_map(raw: str) -> dict[int, int]:
    out: dict[int, int] = {}
    text = str(raw or "").strip()
    if not text:
        return out
    for item in text.replace(",", ";").split(";"):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(
                "batch-size map entries must be leaf_tokens=batch_size; "
                f"got {item!r}"
            )
        key, value = item.split("=", 1)
        out[int(key.strip())] = int(value.strip())
    return out


def _iter_cells(
    *,
    objectives: Iterable[str],
    leaf_tokens_grid: Iterable[int],
    channels_grid: Iterable[int],
    g_n_modes_grid: Iterable[int],
    batch_size: int,
    batch_size_map: dict[int, int] | None = None,
    seeds: Iterable[int],
) -> list[GridCell]:
    cells: list[GridCell] = []
    resolved_batch_map = dict(batch_size_map or {})
    for objective in objectives:
        for leaf_tokens in leaf_tokens_grid:
            cell_batch_size = int(resolved_batch_map.get(int(leaf_tokens), int(batch_size)))
            for channels in channels_grid:
                for g_n_modes in g_n_modes_grid:
                    for seed in seeds:
                        cell_id = (
                            f"obj_{str(objective)}"
                            f"_leaf{int(leaf_tokens)}"
                            f"_ch{int(channels)}"
                            f"_gm{int(g_n_modes)}"
                            f"_bs{int(cell_batch_size)}"
                            f"_seed{int(seed)}"
                        )
                        cells.append(
                            GridCell(
                                cell_id=cell_id,
                                objective=str(objective),
                                leaf_tokens=int(leaf_tokens),
                                channels=int(channels),
                                g_n_modes=int(g_n_modes),
                                batch_size=int(cell_batch_size),
                                seed=int(seed),
                            )
                        )
    return cells


def _read_summary(cell_root: Path) -> dict[str, object]:
    with open(cell_root / "summary.json") as fh:
        return json.load(fh)


def _result_row(cell: GridCell, cell_root: Path, summary: dict[str, object]) -> dict[str, object]:
    history = list(summary.get("history") or [])
    final_history = dict(history[-1]) if history else {}
    args = dict(summary.get("args") or {})
    generation_meta = dict(summary.get("generation_meta") or {})
    exact_witness = dict(summary.get("exact_palette_block_witness") or {})
    exact_val = dict(exact_witness.get("val") or {})
    exact_test = dict(exact_witness.get("test") or {})
    exact_surface = dict(summary.get("exact_surface_contract") or {})
    exact_surface_diag = dict(exact_surface.get("diagnostics") or {})
    constants = dict(summary.get("constant_baselines") or {})
    constant_splits = dict(constants.get("splits") or {})
    val_constants = dict(constant_splits.get("val") or {})
    test_constants = dict(constant_splits.get("test") or {})
    val_split_median = dict(val_constants.get("split_median_predictor") or {})
    test_split_median = dict(test_constants.get("split_median_predictor") or {})
    learned_diag = dict(summary.get("learned_prediction_diagnostics") or {})
    learned_val = dict(learned_diag.get("val") or {})
    learned_test = dict(learned_diag.get("test") or {})
    baselines = dict(summary.get("diagnostic_baselines") or {})
    palette_ridge = dict(baselines.get("palette_block_bigram_ridge") or {})
    palette_ridge_test = dict(palette_ridge.get("test") or {})
    cnn1d = dict(baselines.get("cnn1d") or {})
    cnn1d_test = dict(cnn1d.get("test") or {})
    fno_vanilla = dict(baselines.get("fno_vanilla") or {})
    fno_vanilla_test = dict(fno_vanilla.get("test") or {})
    fno_transition = dict(baselines.get("fno_transition_channel") or {})
    fno_transition_test = dict(fno_transition.get("test") or {})
    boundary_ablation = dict(summary.get("boundary_supervision_ablation") or {})
    boundary_prediction = dict(boundary_ablation.get("prediction_diagnostics") or {})
    boundary_test = dict(boundary_prediction.get("test") or {})
    node_witness = dict(summary.get("markov_node_witness_diagnostics") or {})
    node_witness_test = dict(dict(node_witness.get("splits") or {}).get("test") or {})
    local_law = dict(summary.get("markov_local_law_fno_diagnostics") or {})
    local_law_test = dict(dict(local_law.get("splits") or {}).get("test") or {})
    law_leaf = dict(local_law_test.get("leaf") or {})
    law_merge = dict(local_law_test.get("merge") or {})
    law_root = dict(local_law_test.get("root") or {})
    witness_leaf = dict(node_witness_test.get("leaf") or {})
    witness_merge = dict(node_witness_test.get("merge") or {})
    return {
        "cell_id": cell.cell_id,
        "status": "completed",
        "objective": cell.objective,
        "leaf_tokens": cell.leaf_tokens,
        "doc_tokens": args.get("doc_tokens") or generation_meta.get("doc_tokens"),
        "expected_boundaries": generation_meta.get("expected_boundaries"),
        "n_leaves_per_doc": summary.get("n_leaves_per_doc"),
        "channels": cell.channels,
        "g_n_modes": cell.g_n_modes,
        "g_n_layers": args.get("g_n_layers"),
        "scorer_n_modes": args.get("scorer_n_modes"),
        "scorer_n_layers": args.get("scorer_n_layers"),
        "seed": cell.seed,
        "train_docs": args.get("train_docs"),
        "epochs": args.get("epochs"),
        "batch_size": cell.batch_size or args.get("batch_size"),
        "n_params_total": summary.get("n_params_total"),
        "n_params_g": summary.get("n_params_g"),
        "n_params_f": summary.get("n_params_f"),
        "best_val_root_mae": summary.get("best_val_root_mae"),
        "best_val_epoch": summary.get("best_val_epoch"),
        "test_root_mae": summary.get("test_root_mae"),
        "exact_witness_val_root_mae": exact_val.get("mae"),
        "exact_witness_test_root_mae": exact_test.get("mae"),
        "exact_witness_test_max_abs_error": exact_test.get("max_abs_error"),
        "exact_surface_status": exact_surface.get("status"),
        "exact_surface_test_root_mae": exact_surface_diag.get("root_mae"),
        "val_split_median_constant_root_mae": val_split_median.get("root_mae"),
        "test_split_median_constant_root_mae": test_split_median.get("root_mae"),
        "learned_val_pred_std": learned_val.get("pred_std"),
        "learned_val_pred_truth_corr": learned_val.get("pred_truth_corr"),
        "learned_test_pred_std": learned_test.get("pred_std"),
        "learned_test_pred_truth_corr": learned_test.get("pred_truth_corr"),
        "palette_ridge_test_root_mae": palette_ridge_test.get("root_mae"),
        "cnn1d_test_root_mae": cnn1d_test.get("root_mae"),
        "fno_vanilla_test_root_mae": fno_vanilla_test.get("root_mae"),
        "fno_transition_test_root_mae": fno_transition_test.get("root_mae"),
        "boundary_ablation_status": boundary_ablation.get("status"),
        "boundary_ablation_test_root_mae": boundary_test.get("root_mae"),
        "markov_law_status": local_law.get("status"),
        "markov_law_c2_merge_target": local_law.get("c2_merge_target"),
        "markov_law_test_leaf_theta_mae": law_leaf.get("theta_mae"),
        "markov_law_test_merge_theta_mae": law_merge.get("theta_mae"),
        "markov_law_test_root_theta_mae": law_root.get("theta_mae"),
        "markov_law_test_leaf_first_acc": law_leaf.get("theta_first_regime_accuracy"),
        "markov_law_test_leaf_last_acc": law_leaf.get("theta_last_regime_accuracy"),
        "markov_law_test_merge_first_acc": law_merge.get("theta_first_regime_accuracy"),
        "markov_law_test_merge_last_acc": law_merge.get("theta_last_regime_accuracy"),
        "markov_law_test_leaf_count_mae": dict(law_leaf.get("count_diagnostics") or {}).get("root_mae"),
        "markov_law_test_merge_count_mae": dict(law_merge.get("count_diagnostics") or {}).get("root_mae"),
        "markov_law_test_root_count_mae": dict(law_root.get("count_diagnostics") or {}).get("root_mae"),
        "markov_law_test_leaf_eps_idemp_range": law_leaf.get("eps_idemp_range"),
        "markov_law_test_merge_eps_idemp_range": law_merge.get("eps_idemp_range"),
        "node_witness_status": node_witness.get("status"),
        "node_witness_test_leaf_full_exact": witness_leaf.get("full_witness_exact_rate"),
        "node_witness_test_merge_full_exact": witness_merge.get("full_witness_exact_rate"),
        "final_train_loss": final_history.get("train_loss"),
        "final_lr": final_history.get("lr"),
        "output_root": str(cell_root),
    }


def _write_summary_files(output_root: Path, rows: list[dict[str, object]]) -> None:
    csv_path = output_root / "grid_summary.csv"
    if rows:
        preferred = [
            "cell_id",
            "status",
            "objective",
            "leaf_tokens",
            "doc_tokens",
            "expected_boundaries",
            "n_leaves_per_doc",
            "channels",
            "g_n_modes",
            "batch_size",
            "seed",
            "best_val_root_mae",
            "best_val_epoch",
            "test_root_mae",
            "markov_law_test_leaf_theta_mae",
            "markov_law_test_merge_theta_mae",
            "markov_law_test_leaf_first_acc",
            "markov_law_test_leaf_last_acc",
            "markov_law_test_merge_first_acc",
            "markov_law_test_merge_last_acc",
            "markov_law_test_leaf_count_mae",
            "markov_law_test_merge_count_mae",
            "node_witness_test_leaf_full_exact",
            "node_witness_test_merge_full_exact",
            "exact_witness_test_root_mae",
            "exact_surface_test_root_mae",
            "test_split_median_constant_root_mae",
            "learned_test_pred_std",
            "learned_test_pred_truth_corr",
            "palette_ridge_test_root_mae",
            "cnn1d_test_root_mae",
            "fno_vanilla_test_root_mae",
            "fno_transition_test_root_mae",
            "boundary_ablation_test_root_mae",
            "elapsed_sec",
            "output_root",
        ]
        fields = [field for field in preferred if any(field in row for row in rows)]
        seen = set(fields)
        extra_fields = sorted(
            {
                field
                for row in rows
                for field in row.keys()
                if field not in seen
            }
        )
        fields.extend(extra_fields)
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    report_path = output_root / "grid_report.md"
    with open(report_path, "w") as fh:
        fh.write("# Clean Unified NO Grid Report\n\n")
        fh.write(f"Cells completed: {sum(1 for r in rows if r.get('status') == 'completed')} / {len(rows)}\n\n")
        if rows:
            fh.write("| cell | objective | leaves | channels | g modes | batch | best val root MAE | test root MAE | law leaf theta MAE | law merge theta MAE | leaf first/last | merge first/last |\n")
            fh.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
            for row in rows:
                best_val = row.get("best_val_root_mae")
                test_mae = row.get("test_root_mae")
                law_leaf = row.get("markov_law_test_leaf_theta_mae")
                law_merge = row.get("markov_law_test_merge_theta_mae")
                leaf_first = row.get("markov_law_test_leaf_first_acc")
                leaf_last = row.get("markov_law_test_leaf_last_acc")
                merge_first = row.get("markov_law_test_merge_first_acc")
                merge_last = row.get("markov_law_test_merge_last_acc")
                best_val_text = "" if best_val is None else f"{float(best_val):.4f}"
                test_mae_text = "" if test_mae is None else f"{float(test_mae):.4f}"
                law_leaf_text = "" if law_leaf is None else f"{float(law_leaf):.4f}"
                law_merge_text = "" if law_merge is None else f"{float(law_merge):.4f}"
                leaf_acc_text = (
                    ""
                    if leaf_first is None or leaf_last is None
                    else f"{float(leaf_first):.3f}/{float(leaf_last):.3f}"
                )
                merge_acc_text = (
                    ""
                    if merge_first is None or merge_last is None
                    else f"{float(merge_first):.3f}/{float(merge_last):.3f}"
                )
                fh.write(
                    f"| {row.get('cell_id')} "
                    f"| {row.get('objective')} "
                    f"| {row.get('n_leaves_per_doc')} "
                    f"| {row.get('channels')} "
                    f"| {row.get('g_n_modes')} "
                    f"| {row.get('batch_size')} "
                    f"| {best_val_text} "
                    f"| {test_mae_text} "
                    f"| {law_leaf_text} "
                    f"| {law_merge_text} "
                    f"| {leaf_acc_text} "
                    f"| {merge_acc_text} |\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a compact grid of CleanUnifiedNO Markov probes."
    )
    parser.add_argument("--benchmark", default="recoverable_v5_t2048")
    parser.add_argument(
        "--load-data-bundle",
        default=None,
        help="Forwarded to probe_clean_unified_no.py for saved MarkovOPSDataBundle parity.",
    )
    parser.add_argument("--doc-tokens", type=int, default=0)
    parser.add_argument("--expected-boundaries", type=float, default=None)
    parser.add_argument(
        "--objectives",
        default="root",
        help=(
            "Comma-separated objectives: root, contextual_none, "
            "markov_node_witness, markov_local_laws_fno."
        ),
    )
    parser.add_argument("--leaf-tokens-grid", default="2048,256")
    parser.add_argument("--channels-grid", default="32")
    parser.add_argument("--g-n-modes-grid", default="8,16")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--train-docs", type=int, default=64)
    parser.add_argument(
        "--eval-docs",
        type=int,
        default=None,
        help="Optional cap for val/test docs passed to probe_clean_unified_no.py.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--batch-size-map",
        default="",
        help="Optional per-leaf override, e.g. '128=256;64=128;16=64'.",
    )
    parser.add_argument("--g-n-layers", type=int, default=2)
    parser.add_argument("--scorer-n-modes", type=int, default=16)
    parser.add_argument("--scorer-n-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", default="adamw", choices=["adam", "adamw"])
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--lr-schedule", default="cosine", choices=["none", "cosine"])
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--leaf-pool", default="sum", choices=["mean", "sum"])
    parser.add_argument("--root-only", action="store_true")
    parser.add_argument(
        "--diagnostic-baselines",
        default="palette_ridge",
        help=(
            "Comma-separated baselines forwarded to probe: none, palette_ridge, "
            "cnn1d, fno_vanilla, fno_transition, all."
        ),
    )
    parser.add_argument("--diagnostic-baseline-epochs", type=int, default=0)
    parser.add_argument("--diagnostic-baseline-batch-size", type=int, default=0)
    parser.add_argument("--diagnostic-baseline-lr", type=float, default=0.0)
    parser.add_argument("--palette-ridge-alpha", type=float, default=0.0)
    parser.add_argument("--run-boundary-supervision-ablation", action="store_true")
    parser.add_argument("--boundary-supervision-epochs", type=int, default=0)
    parser.add_argument("--boundary-supervision-weight", type=float, default=1.0)
    parser.add_argument("--boundary-supervision-lr", type=float, default=0.0)
    parser.add_argument("--markov-witness-weight", type=float, default=1.0)
    parser.add_argument("--markov-witness-count-weight", type=float, default=1.0)
    parser.add_argument("--markov-witness-edge-weight", type=float, default=1.0)
    parser.add_argument(
        "--markov-witness-readout",
        default="flatten",
        choices=["flatten", "conv_pool"],
    )
    parser.add_argument("--markov-law-weight", type=float, default=1.0)
    parser.add_argument("--markov-law-leaf-weight", type=float, default=1.0)
    parser.add_argument("--markov-law-merge-weight", type=float, default=1.0)
    parser.add_argument("--markov-law-idempotence-weight", type=float, default=0.1)
    parser.add_argument("--markov-law-count-weight", type=float, default=1.0)
    parser.add_argument("--markov-law-edge-weight", type=float, default=1.0)
    parser.add_argument(
        "--markov-law-readout",
        default="flatten",
        choices=["flatten", "conv_pool"],
    )
    parser.add_argument("--require-exact-contract-zero", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Rebuild grid_summary.csv/grid_report.md from an existing grid_manifest.json.",
    )
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    output_root = Path(
        args.output_root
        or REPO / "outputs" / f"clean_unified_no_grid_{_utc_stamp()}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        manifest_path = output_root / "grid_manifest.json"
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        rows: list[dict[str, object]] = []
        for payload in manifest.get("cells") or []:
            cell = GridCell(
                cell_id=str(payload["cell_id"]),
                objective=str(payload.get("objective", "root")),
                leaf_tokens=int(payload["leaf_tokens"]),
                channels=int(payload["channels"]),
                g_n_modes=int(payload["g_n_modes"]),
                batch_size=int(payload.get("batch_size", args.batch_size)),
                seed=int(payload["seed"]),
            )
            cell_root = output_root / cell.cell_id
            summary_path = cell_root / "summary.json"
            if not summary_path.exists():
                rows.append(
                    {
                        "cell_id": cell.cell_id,
                        "status": "missing_summary",
                        "objective": cell.objective,
                        "leaf_tokens": cell.leaf_tokens,
                        "channels": cell.channels,
                        "g_n_modes": cell.g_n_modes,
                        "batch_size": cell.batch_size,
                        "seed": cell.seed,
                        "output_root": str(cell_root),
                    }
                )
                continue
            row = _result_row(cell, cell_root, _read_summary(cell_root))
            rows.append(row)
        _write_summary_files(output_root, rows)
        print(f"aggregated {len(rows)} cells")
        print(f"summary: {output_root / 'grid_summary.csv'}")
        print(f"report:  {output_root / 'grid_report.md'}")
        return

    cells = _iter_cells(
        objectives=_parse_str_grid(args.objectives),
        leaf_tokens_grid=_parse_int_grid(args.leaf_tokens_grid),
        channels_grid=_parse_int_grid(args.channels_grid),
        g_n_modes_grid=_parse_int_grid(args.g_n_modes_grid),
        batch_size=int(args.batch_size),
        batch_size_map=_parse_batch_size_map(args.batch_size_map),
        seeds=_parse_int_grid(args.seeds),
    )
    manifest = {
        "schema_version": "clean_unified_no_grid.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "output_root": str(output_root),
        "args": vars(args),
        "cells": [asdict(cell) for cell in cells],
    }
    with open(output_root / "grid_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"CleanUnifiedNO grid: {len(cells)} cells")
    print(f"output_root={output_root}")
    if args.dry_run:
        print("dry run only; not launching cells")
        return

    rows: list[dict[str, object]] = []
    for idx, cell in enumerate(cells, start=1):
        cell_root = output_root / cell.cell_id
        cmd = [
            sys.executable,
            str(REPO / "scripts" / "probe_clean_unified_no.py"),
            "--benchmark",
            str(args.benchmark),
            "--leaf-tokens",
            str(cell.leaf_tokens),
            "--train-docs",
            str(args.train_docs),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(cell.batch_size),
            "--channels",
            str(cell.channels),
            "--g-n-modes",
            str(cell.g_n_modes),
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
            "--diagnostic-baseline-epochs",
            str(args.diagnostic_baseline_epochs),
            "--diagnostic-baseline-batch-size",
            str(args.diagnostic_baseline_batch_size),
            "--diagnostic-baseline-lr",
            str(args.diagnostic_baseline_lr),
            "--palette-ridge-alpha",
            str(args.palette_ridge_alpha),
            "--boundary-supervision-epochs",
            str(args.boundary_supervision_epochs),
            "--boundary-supervision-weight",
            str(args.boundary_supervision_weight),
            "--boundary-supervision-lr",
            str(args.boundary_supervision_lr),
            "--markov-witness-weight",
            str(args.markov_witness_weight),
            "--markov-witness-count-weight",
            str(args.markov_witness_count_weight),
            "--markov-witness-edge-weight",
            str(args.markov_witness_edge_weight),
            "--markov-witness-readout",
            str(args.markov_witness_readout),
            "--markov-law-weight",
            str(args.markov_law_weight),
            "--markov-law-leaf-weight",
            str(args.markov_law_leaf_weight),
            "--markov-law-merge-weight",
            str(args.markov_law_merge_weight),
            "--markov-law-idempotence-weight",
            str(args.markov_law_idempotence_weight),
            "--markov-law-count-weight",
            str(args.markov_law_count_weight),
            "--markov-law-edge-weight",
            str(args.markov_law_edge_weight),
            "--markov-law-readout",
            str(args.markov_law_readout),
            "--seed",
            str(cell.seed),
            "--device",
            str(args.device),
            "--output-root",
            str(cell_root),
        ]
        objective = str(cell.objective)
        if objective == "root":
            cmd.extend(["--training-objective", "root", "--root-only"])
        elif objective == "contextual_none":
            cmd.extend(
                [
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
            )
        elif objective == "markov_node_witness":
            cmd.extend(["--training-objective", "markov_node_witness"])
        elif objective == "markov_local_laws_fno":
            cmd.extend(["--training-objective", "markov_local_laws_fno"])
        else:
            raise ValueError(f"unknown objective cell: {objective!r}")
        if int(args.doc_tokens) > 0:
            cmd.extend(["--doc-tokens", str(args.doc_tokens)])
        if args.load_data_bundle:
            cmd.extend(["--load-data-bundle", str(args.load_data_bundle)])
        if args.expected_boundaries is not None:
            cmd.extend(["--expected-boundaries", str(args.expected_boundaries)])
        if args.eval_docs is not None:
            cmd.extend(["--eval-docs", str(args.eval_docs)])
        if args.gpu is not None:
            cmd.extend(["--gpu", str(args.gpu)])
        if args.root_only:
            cmd.append("--root-only")
        if args.run_boundary_supervision_ablation:
            cmd.append("--run-boundary-supervision-ablation")
        if args.require_exact_contract_zero:
            cmd.append("--require-exact-contract-zero")

        print(f"[{idx}/{len(cells)}] start {cell.cell_id}", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=REPO)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            row = {
                "cell_id": cell.cell_id,
                "status": f"failed:{proc.returncode}",
                "objective": cell.objective,
                "leaf_tokens": cell.leaf_tokens,
                "channels": cell.channels,
                "g_n_modes": cell.g_n_modes,
                "batch_size": cell.batch_size,
                "seed": cell.seed,
                "elapsed_sec": round(elapsed, 3),
                "output_root": str(cell_root),
            }
            rows.append(row)
            _write_summary_files(output_root, rows)
            if not args.keep_going:
                raise SystemExit(proc.returncode)
            continue

        summary = _read_summary(cell_root)
        row = _result_row(cell, cell_root, summary)
        row["elapsed_sec"] = round(elapsed, 3)
        rows.append(row)
        _write_summary_files(output_root, rows)
        print(
            f"[{idx}/{len(cells)}] done {cell.cell_id}: "
            f"best_val={float(row['best_val_root_mae']):.4f} "
            f"test={float(row['test_root_mae']):.4f} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    _write_summary_files(output_root, rows)
    print(f"summary: {output_root / 'grid_summary.csv'}")
    print(f"report:  {output_root / 'grid_report.md'}")


if __name__ == "__main__":
    main()
