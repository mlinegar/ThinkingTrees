#!/usr/bin/env python3
"""Prepare or launch the HLL root-plus-random-nonroot sampling grid.

This grid keeps the supplied-answer semantics simple: use the HLL formula
readout family, vary the schedule, and vary root/non-root oracle label rates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GPU_IDS = (0, 1, 2, 3)
EXPECTED_METRICS = (
    "root_mae",
    "root_rel_mae",
    "official_f_on_learned_root_mae",
    "official_f_on_learned_root_rel_mae",
    "merge_state_mae",
    "merge_state_root_mae",
    "merge_readout_root_mae",
    "official_merge_readout_root_mae",
    "train_observed_rows_end",
    "train_population_rows_end",
    "train_root_observed_rows_end",
    "train_root_population_rows_end",
    "train_nonroot_observed_rows_end",
    "train_nonroot_population_rows_end",
    "train_observed_rows_per_doc_end",
    "train_root_observed_rows_per_doc_end",
    "train_nonroot_observed_rows_per_doc_end",
    "train_max_ipw_weight_end",
    "train_effective_sample_size_end",
    "train_local_proxy_loss_end",
    "train_local_oracle_observed_ipw_loss_end",
    "train_local_ipw_correction_end",
    "train_local_corrected_loss_end",
    "train_discounted_root_weight_end",
    "train_discounted_nonroot_weight_end",
    "train_observed_mass_end",
    "train_population_mass_end",
    "learned_state_below_valid_frac",
    "learned_state_above_valid_frac",
    "learned_state_nonfinite_frac",
    "learned_root_state_below_valid_frac",
    "learned_root_state_above_valid_frac",
    "hll_readout_preclamp_above_one_frac",
    "hll_readout_postclamp_near_one_frac",
    "merge_carrier_state_norm_mean",
    "merge_projection_delta_norm_mean",
    "merge_projection_delta_to_carrier_norm_mean",
    "merge_projection_delta_to_carrier_norm_root_mean",
    "hll_merge_register_exact_frac",
    "hll_merge_register_rounded_mae",
    "hll_zero_scalar_bad_state_frac",
    "hll_within_tol_bad_state_frac",
    "hll_future_context_readout_mae",
)


@dataclass(frozen=True)
class GridCell:
    cell_id: str
    family: str
    description: str
    n_leaves: int
    root_label_share: float
    sampled_node_rate: float
    gpu_id: int
    estimated_row_work: int
    output_dir: str
    command: list[str]
    expected_metrics: list[str]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_int_csv(raw: str) -> list[int]:
    values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def _parse_float_csv(raw: str) -> list[float]:
    values = [float(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one float")
    for value in values:
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"sample rates must be finite and in [0, 1], got {value!r}")
    return values


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_first_csv(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return dict(rows[0]) if rows else None


def _flag_value(command: Sequence[str], flag: str, default: str = "") -> str:
    try:
        index = list(command).index(flag)
    except ValueError:
        return default
    if index + 1 >= len(command):
        return default
    return str(command[index + 1])


def _replace_flag_value(command: Sequence[str], flag: str, value: str) -> list[str]:
    items = [str(part) for part in command]
    try:
        index = items.index(flag)
    except ValueError as exc:
        raise ValueError(f"command missing required flag {flag}") from exc
    if index + 1 >= len(items):
        raise ValueError(f"command flag {flag} has no value")
    items[index + 1] = str(value)
    return items


def _assert_scalar_only_or_explicitly_anchored(args: argparse.Namespace) -> None:
    has_dense_regularizer = (
        float(args.state_loss_weight) > 0.0
        or float(args.exact_state_anchor_weight) > 0.0
    )
    if has_dense_regularizer and not bool(getattr(args, "allow_dense_regularizers", False)):
        raise ValueError(
            "sampled-node paper grids are scalar-only by default; pass "
            "--allow-dense-regularizers to run anchored diagnostic lanes with "
            "nonzero --state-loss-weight or --exact-state-anchor-weight"
        )


def _rate_tag(rate: float) -> str:
    return f"r{int(round(float(rate) * 1000)):03d}"


def _share_tag(share: float) -> str:
    return f"R{int(round(float(share) * 100)):03d}"


def _rate_label(rate: float) -> str:
    return f"{100.0 * float(rate):g}%"


def _base_command(args: argparse.Namespace, output_dir: Path, *, n_leaves: int) -> list[str]:
    command = [
        str(args.python_bin),
        "scripts/run_fno_mergeable_sketch_diagnostic.py",
        "--targets",
        "hll_register_space",
        "--n-train",
        str(args.n_train),
        "--n-val",
        str(args.n_val),
        "--n-leaves",
        str(n_leaves),
        "--min-tokens",
        str(args.min_tokens),
        "--max-tokens",
        str(args.max_tokens),
        "--universe-size",
        str(args.universe_size),
        "--precision",
        str(args.precision),
        "--zipf-alphas",
        str(args.zipf_alphas),
        "--batch-size",
        str(args.batch_size),
        "--rollout-min-docs-per-batch",
        str(args.rollout_min_docs_per_batch),
        "--rollout-max-docs-per-batch",
        str(args.rollout_max_docs_per_batch),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--grad-accum-steps",
        str(args.grad_accum_steps),
        "--target-transform",
        str(args.target_transform),
        "--state-normalization",
        "register_div64",
        "--hidden-channels",
        str(args.hidden_channels),
        "--head-hidden-dim",
        str(args.head_hidden_dim),
        "--n-modes",
        str(args.n_modes),
        "--n-layers",
        str(args.n_layers),
        "--f-learning-rate",
        str(args.f_learning_rate),
        "--g-learning-rate",
        str(args.g_learning_rate),
        "--eval-every-epochs",
        str(args.eval_every_epochs),
        "--progress-every-epochs",
        str(args.progress_every_epochs),
        "--progress-every-batches",
        str(args.progress_every_batches),
        "--readout-arch",
        "hll_formula",
        "--device",
        "cuda",
        "--seed",
        str(args.seed),
        "--schedule",
        str(args.schedule),
        "--objective-mode",
        "rollout_local_law",
        "--epochs",
        str(args.epochs),
        "--local-law-weight",
        str(args.local_law_weight),
        "--local-law-leaf-discount-gamma",
        str(args.local_law_leaf_discount_gamma),
        "--merge-output-constraint",
        str(args.merge_output_constraint),
        "--objective-loss-weight",
        str(args.objective_loss_weight),
        "--state-loss-weight",
        str(args.state_loss_weight),
        "--exact-state-anchor-weight",
        str(args.exact_state_anchor_weight),
        "--output-dir",
        str(output_dir),
    ]
    sample_cache_dir = getattr(args, "sample_cache_dir", None)
    if sample_cache_dir is not None:
        command.extend(["--sample-cache-dir", str(sample_cache_dir)])
    return command


def _make_cell(
    args: argparse.Namespace,
    *,
    output_root: Path,
    n_leaves: int,
    root_label_share: float,
    sampled_node_rate: float,
) -> GridCell:
    root_share = max(0.0, min(1.0, float(root_label_share)))
    node_rate = max(0.0, min(1.0, float(sampled_node_rate)))
    root_tag = _share_tag(root_share)
    rate_tag = _rate_tag(node_rate)
    schedule = str(args.schedule)
    schedule_tag = "" if schedule == "g" else f"_{schedule}"
    cell_id = f"sampled{schedule_tag}_{root_tag}_{rate_tag}_L{int(n_leaves)}"
    output_dir = output_root / cell_id
    command = _base_command(args, output_dir, n_leaves=int(n_leaves))
    if root_share >= 1.0 and node_rate <= 0.0:
        command.extend(["--oracle-observation-design", "root_only"])
        family = f"sampled{schedule_tag}_R100_r000"
        description = "Root-only scalar labels; no non-root node oracle labels."
    elif root_share >= 1.0:
        command.extend(
            [
                "--oracle-observation-design",
                "sampled_nodes",
                "--sampled-node-rate",
                f"{node_rate:.8g}",
            ]
        )
        family = f"sampled{schedule_tag}_R100_{rate_tag}"
        description = (
            f"All root labels plus random non-root scalar labels at rate "
            f"{_rate_label(node_rate)}."
        )
    else:
        command.extend(
            [
                "--oracle-observation-design",
                "sampled_root_nodes",
                "--root-label-share",
                f"{root_share:.8g}",
                "--sampled-node-rate",
                f"{node_rate:.8g}",
            ]
        )
        family = f"sampled{schedule_tag}_{root_tag}_{rate_tag}"
        description = (
            f"Root labels sampled at {_rate_label(root_share)} plus random non-root "
            f"scalar labels at {_rate_label(node_rate)}."
        )
    return GridCell(
        cell_id=cell_id,
        family=family,
        description=description,
        n_leaves=int(n_leaves),
        root_label_share=float(root_share),
        sampled_node_rate=float(node_rate),
        gpu_id=-1,
        estimated_row_work=(
            int(args.epochs)
            * max(1, len(schedule))
            * int(args.n_train)
            * max(1, int(2 * n_leaves - 1))
        ),
        output_dir=str(output_dir),
        command=command,
        expected_metrics=list(EXPECTED_METRICS),
    )


def _assign_gpus(cells: Sequence[GridCell], gpu_ids: Sequence[int]) -> list[GridCell]:
    loads = {int(gpu_id): 0 for gpu_id in gpu_ids}
    assigned: list[GridCell] = []
    for cell in sorted(
        cells,
        key=lambda item: (
            -int(item.estimated_row_work),
            -float(item.root_label_share),
            float(item.sampled_node_rate),
            item.cell_id,
        ),
    ):
        gpu_id = min(loads, key=lambda key: (loads[key], key))
        loads[gpu_id] += int(cell.estimated_row_work)
        assigned.append(
            GridCell(
                cell_id=cell.cell_id,
                family=cell.family,
                description=cell.description,
                n_leaves=cell.n_leaves,
                root_label_share=cell.root_label_share,
                sampled_node_rate=cell.sampled_node_rate,
                gpu_id=int(gpu_id),
                estimated_row_work=cell.estimated_row_work,
                output_dir=cell.output_dir,
                command=cell.command,
                expected_metrics=cell.expected_metrics,
            )
        )
    return sorted(
        assigned,
        key=lambda item: (
            item.gpu_id,
            -int(item.estimated_row_work),
            -float(item.root_label_share),
            float(item.sampled_node_rate),
            item.cell_id,
        ),
    )


def _load_source_cells(source_root: Path) -> list[dict[str, Any]]:
    manifest_path = source_root / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    cells = manifest.get("cells")
    return [dict(cell) for cell in cells] if isinstance(cells, list) else []


def _matching_resume_prefix(target: GridCell, source: Mapping[str, Any]) -> tuple[int, Path] | None:
    target_command = list(target.command)
    source_command = [str(item) for item in source.get("command", [])]
    if not target_command or not source_command:
        return None
    target_schedule = _flag_value(target_command, "--schedule")
    source_schedule = _flag_value(source_command, "--schedule")
    if not source_schedule or not target_schedule.startswith(source_schedule):
        return None
    if source_schedule == target_schedule:
        return None
    if int(source.get("n_leaves", -1)) != int(target.n_leaves):
        return None
    for flag in (
        "--targets",
        "--n-train",
        "--n-val",
        "--n-leaves",
        "--min-tokens",
        "--max-tokens",
        "--universe-size",
        "--precision",
        "--zipf-alphas",
        "--target-transform",
        "--state-normalization",
        "--hidden-channels",
        "--head-hidden-dim",
        "--n-modes",
        "--n-layers",
        "--readout-arch",
        "--merge-output-constraint",
        "--seed",
        "--objective-mode",
        "--oracle-observation-design",
        "--root-label-share",
        "--sampled-node-rate",
    ):
        if _flag_value(target_command, flag) != _flag_value(source_command, flag):
            return None
    checkpoint = (
        Path(str(source.get("output_dir", "")))
        / "hll_register_space"
        / f"stage_{len(source_schedule):02d}_{source_schedule[-1]}_model.pt"
    )
    if not checkpoint.exists():
        return None
    return len(source_schedule), checkpoint


def _patched_resume_cell(cell: GridCell, *, prefix: str, checkpoint: Path) -> GridCell:
    full_schedule = _flag_value(cell.command, "--schedule")
    suffix = full_schedule[len(prefix) :]
    command = _replace_flag_value(cell.command, "--schedule", suffix)
    command.extend(
        [
            "--init-checkpoint",
            str(checkpoint),
            "--schedule-prefix",
            prefix,
            "--stage-index-offset",
            str(len(prefix)),
        ]
    )
    epochs = int(_flag_value(command, "--epochs", "1"))
    estimated = epochs * max(1, len(suffix)) * int(_flag_value(command, "--n-train", "1")) * max(
        1,
        int(2 * cell.n_leaves - 1),
    )
    return GridCell(
        cell_id=cell.cell_id,
        family=cell.family,
        description=(
            f"{cell.description} Resumes from {prefix} checkpoint {checkpoint} "
            f"and trains suffix {suffix}."
        ),
        n_leaves=cell.n_leaves,
        root_label_share=cell.root_label_share,
        sampled_node_rate=cell.sampled_node_rate,
        gpu_id=cell.gpu_id,
        estimated_row_work=estimated,
        output_dir=cell.output_dir,
        command=command,
        expected_metrics=cell.expected_metrics,
    )


def apply_reuse_sources(
    cells: Sequence[GridCell],
    *,
    reuse_roots: Sequence[Path],
    gpu_ids: Sequence[int],
) -> list[GridCell]:
    source_cells: list[dict[str, Any]] = []
    for root in reuse_roots:
        source_cells.extend(_load_source_cells(Path(root).resolve()))
    if not source_cells:
        return list(cells)
    patched: list[GridCell] = []
    for cell in cells:
        candidates = [
            candidate
            for source in source_cells
            if (candidate := _matching_resume_prefix(cell, source)) is not None
        ]
        if candidates:
            prefix_len, checkpoint = max(candidates, key=lambda item: item[0])
            prefix = _flag_value(cell.command, "--schedule")[:prefix_len]
            patched.append(_patched_resume_cell(cell, prefix=prefix, checkpoint=checkpoint))
        else:
            patched.append(cell)
    return _assign_gpus(patched, gpu_ids)


def build_cells(args: argparse.Namespace, output_root: Path) -> list[GridCell]:
    _assert_scalar_only_or_explicitly_anchored(args)
    specs: list[GridCell] = []
    for leaves in _parse_int_csv(args.leaves):
        for root_share in _parse_float_csv(args.root_label_shares):
            for rate in _parse_float_csv(args.sample_rates):
                if (
                    bool(getattr(args, "skip_zero_label_cell", False))
                    and float(root_share) <= 0.0
                    and float(rate) <= 0.0
                ):
                    continue
                specs.append(
                    _make_cell(
                        args,
                        output_root=output_root,
                        n_leaves=leaves,
                        root_label_share=root_share,
                        sampled_node_rate=rate,
                    )
                )
    return _assign_gpus(specs, _parse_int_csv(args.gpu_ids))


def _summary_path(cell: GridCell) -> Path:
    return Path(cell.output_dir) / "summary.csv"


def _exit_path(cell: GridCell) -> Path:
    return Path(cell.output_dir) / "exit_code.txt"


def _status_for(cell: GridCell) -> str:
    if _summary_path(cell).exists():
        return "completed"
    if _exit_path(cell).exists():
        try:
            code = int(_exit_path(cell).read_text(encoding="utf-8").strip())
        except Exception:
            code = 1
        return "completed_without_summary" if code == 0 else "failed"
    return "pending"


def _write_runner(path: Path, *, gpu_id: int, cells: Sequence[GridCell]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        f"export CUDA_VISIBLE_DEVICES={int(gpu_id)}",
        "export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}",
        "failures=0",
        "",
    ]
    for cell in cells:
        out_dir = Path(cell.output_dir)
        log_path = out_dir / "run.log"
        exit_path = _exit_path(cell)
        lines.extend(
            [
                f"echo '[hll-sampled-rate-grid] start {cell.cell_id} on visible GPU {gpu_id}'",
                f"mkdir -p {shlex.quote(str(out_dir))}",
                f"if [ -f {shlex.quote(str(_summary_path(cell)))} ]; then",
                f"  echo '[hll-sampled-rate-grid] skip completed {cell.cell_id}'",
                "else",
                "  set +e",
                "  cmd=(",
            ]
        )
        for item in cell.command:
            lines.append(f"    {shlex.quote(str(item))}")
        lines.extend(
            [
                "  )",
                f"  \"${{cmd[@]}}\" > {shlex.quote(str(log_path))} 2>&1",
                "  code=$?",
                "  set -e",
                f"  echo \"$code\" > {shlex.quote(str(exit_path))}",
                "  if [ \"$code\" -ne 0 ]; then",
                f"    echo '[hll-sampled-rate-grid] failed {cell.cell_id} code='\"$code\"",
                "    failures=$((failures + 1))",
                "  else",
                f"    echo '[hll-sampled-rate-grid] done {cell.cell_id}'",
                "  fi",
                "fi",
                "",
            ]
        )
    lines.extend(
        [
            "if [ \"$failures\" -ne 0 ]; then",
            "  echo '[hll-sampled-rate-grid] runner finished with failures='\"$failures\"",
            "  exit 1",
            "fi",
            "echo '[hll-sampled-rate-grid] runner complete'",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def write_manifest_and_runners(args: argparse.Namespace, output_root: Path, cells: Sequence[GridCell]) -> dict[str, Any]:
    by_gpu: dict[int, list[GridCell]] = {}
    for cell in cells:
        by_gpu.setdefault(int(cell.gpu_id), []).append(cell)
    runners: list[dict[str, Any]] = []
    for gpu_id, gpu_cells in sorted(by_gpu.items()):
        runner_path = output_root / "runners" / f"gpu_{gpu_id}.sh"
        _write_runner(runner_path, gpu_id=gpu_id, cells=gpu_cells)
        runners.append(
            {
                "gpu_id": int(gpu_id),
                "runner_path": str(runner_path),
                "job_root": str(output_root / "launchers" / f"gpu_{gpu_id}"),
                "cell_ids": [cell.cell_id for cell in gpu_cells],
                "estimated_row_work": int(sum(cell.estimated_row_work for cell in gpu_cells)),
            }
        )

    manifest = {
        "schema_version": "hll_sampled_node_rate_grid.v1",
        "created_at": _utc_now(),
        "repo_root": str(REPO_ROOT),
        "output_root": str(output_root),
        "launched": False,
        "gpu_ids": _parse_int_csv(args.gpu_ids),
        "sample_cache_dir": str(args.sample_cache_dir),
        "dgp": {
            "target_kind": "hll_register_space",
            "precision": int(args.precision),
            "universe_size": int(args.universe_size),
            "min_tokens": int(args.min_tokens),
            "max_tokens": int(args.max_tokens),
            "zipf_alphas": str(args.zipf_alphas),
            "seed": int(args.seed),
            "n_train": int(args.n_train),
            "n_val": int(args.n_val),
        },
        "training": {
            "schedule": str(args.schedule),
            "epochs_per_stage": int(args.epochs),
            "batch_size": int(args.batch_size),
            "rollout_min_docs_per_batch": int(args.rollout_min_docs_per_batch),
            "rollout_max_docs_per_batch": int(args.rollout_max_docs_per_batch),
            "eval_batch_size": int(args.eval_batch_size),
            "readout_arch": "hll_formula",
            "target_transform": str(args.target_transform),
            "hidden_channels": int(args.hidden_channels),
            "head_hidden_dim": int(args.head_hidden_dim),
            "width_floor_policy": "diagnostic promotes hidden/head widths to at least max(128, 2*state_dim)",
            "merge_adapter": "induced_projection",
            "lean_merge_adapter": "merge(a,b)=g_theta(a+b); encode_leaf(x)=g_theta(x)",
            "lean_projection_target": "f*(x+y)=f*(g*(g*(x)+g*(y)))",
            "state_loss_weight": float(args.state_loss_weight),
            "exact_state_anchor_weight": float(args.exact_state_anchor_weight),
            "allow_dense_regularizers": bool(args.allow_dense_regularizers),
            "local_law_weight": float(args.local_law_weight),
            "local_law_leaf_discount_gamma": float(args.local_law_leaf_discount_gamma),
            "merge_output_constraint": str(args.merge_output_constraint),
            "objective_loss_weight": float(args.objective_loss_weight),
        },
        "sampling_grid": {
            "root_label_shares": _parse_float_csv(args.root_label_shares),
            "sample_rates": _parse_float_csv(args.sample_rates),
            "skip_zero_label_cell": bool(args.skip_zero_label_cell),
        },
        "cells": [asdict(cell) for cell in cells],
        "runners": runners,
        "notes": [
            "Root labels are sampled at the cell root_label_share. Non-root labels are sampled at the cell sampled_node_rate.",
            "Cells with root_label_share=1 and sampled_node_rate=0 are the root-only endpoint.",
            "Cells with root_label_share=1 and sampled_node_rate=1 are the dense sampled-node endpoint.",
            "Observed scalar target is f*(g* exact node); prediction is the current/supplied f_latest(g_theta rollout node).",
            "Default target transform is log1p_zscore so invalid HLL states are not hidden by linear01 output clipping.",
            "Local-law rows are discounted by local_law_leaf_discount_gamma^(root_depth - node_depth).",
            "The grid supports root-rate ablations first, then adding non-root node labels from the same root-rate baseline.",
            "state_loss_weight/exact_state_anchor_weight default to 0 so sampled-node runs use scalar-only IPW supervision.",
            "Nonzero dense state regularizers require --allow-dense-regularizers and should be treated as anchored diagnostics.",
            "All cells share --sample-cache-dir so exact trees and oracle node scores are persisted once per data-generation config.",
            "Default behavior is prelaunch only; use --launch to start long_job runners.",
        ],
    }
    _json_dump(output_root / "manifest.json", manifest)
    return manifest


def aggregate(output_root: Path) -> list[dict[str, Any]]:
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for item in manifest.get("cells", []):
        cell = GridCell(**item)
        summary = _read_first_csv(_summary_path(cell)) or {}
        row: dict[str, Any] = {
            "cell_id": cell.cell_id,
            "family": cell.family,
            "status": _status_for(cell),
            "n_leaves": int(cell.n_leaves),
            "root_label_share": float(cell.root_label_share),
            "sampled_node_rate": float(cell.sampled_node_rate),
            "gpu_id": int(cell.gpu_id),
            "estimated_row_work": int(cell.estimated_row_work),
            "output_dir": cell.output_dir,
            "summary_path": str(_summary_path(cell)),
        }
        for key in cell.expected_metrics:
            row[key] = summary.get(key, "")
        rows.append(row)

    columns = [
        "cell_id",
        "family",
        "status",
        "n_leaves",
        "root_label_share",
        "sampled_node_rate",
        "gpu_id",
        "estimated_row_work",
        *EXPECTED_METRICS,
        "output_dir",
        "summary_path",
    ]
    with (output_root / "grid_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})

    counts: dict[tuple[str, str], int] = {}
    family_rates: dict[str, tuple[float, float]] = {}
    for row in rows:
        key = (str(row["family"]), str(row["status"]))
        counts[key] = counts.get(key, 0) + 1
        family_rates[str(row["family"])] = (
            float(row.get("root_label_share", 1.0)),
            float(row["sampled_node_rate"]),
        )
    lines = [
        "# HLL Sampled Node-Rate Grid",
        "",
        "Prepared grid; no jobs are launched unless `--launch` is passed.",
        "",
        "| family | root rate | non-root rate | status | cells |",
        "|---|---:|---:|---|---:|",
    ]
    for (family, status), count in sorted(counts.items()):
        root_rate, node_rate = family_rates.get(family, (float("nan"), float("nan")))
        node_label = "none" if math.isfinite(node_rate) and node_rate <= 0.0 else _rate_label(node_rate)
        lines.append(f"| {family} | {_rate_label(root_rate)} | {node_label} | {status} | {count} |")
    lines.extend(
        [
            "",
            "## Launch",
            "",
            "```bash",
            f"./venv/bin/python scripts/run_hll_sampled_node_rate_grid.py --output-root {shlex.quote(str(output_root))} --launch",
            "```",
        ]
    )
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def launch_runners(args: argparse.Namespace, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    launched: list[dict[str, Any]] = []
    for runner in manifest.get("runners", []):
        gpu_id = int(runner["gpu_id"])
        runner_path = str(runner["runner_path"])
        job_root = str(runner["job_root"])
        name = f"hll_sampled_node_rate_grid_{Path(str(manifest['output_root'])).name}_gpu{gpu_id}"
        command = [
            str(args.python_bin),
            "scripts/long_job.py",
            "launch",
            "--name",
            name,
            "--job-root",
            job_root,
            "--cwd",
            str(REPO_ROOT),
            "--replace-existing",
            "--",
            runner_path,
        ]
        result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
        payload: dict[str, Any] = {
            "gpu_id": gpu_id,
            "runner_path": runner_path,
            "job_root": job_root,
            "returncode": int(result.returncode),
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        launched.append(payload)
        if result.returncode != 0:
            raise RuntimeError(
                f"failed to launch gpu runner {gpu_id}: {result.stderr.strip() or result.stdout.strip()}"
            )
    launch_path = Path(str(manifest["output_root"])) / "launched_runners.json"
    launch_path.write_text(json.dumps(launched, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = Path(str(manifest["output_root"])) / "manifest.json"
    updated = dict(manifest)
    updated["launched"] = True
    updated["launched_at"] = _utc_now()
    _json_dump(manifest_path, updated)
    return launched


def _print_dry_run(cells: Sequence[GridCell]) -> None:
    print(f"HLL sampled node-rate grid: {len(cells)} cells")
    for cell in cells:
        print(
            f"[gpu {cell.gpu_id}] {cell.cell_id} | "
            f"root={cell.root_label_share:g} nonroot={cell.sampled_node_rate:g} | L={cell.n_leaves}"
        )
        print("  " + shlex.join(cell.command))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--gpu-ids", default=",".join(str(x) for x in DEFAULT_GPU_IDS))
    parser.add_argument("--python-bin", default="./venv/bin/python")
    parser.add_argument("--sample-cache-dir", type=Path, default=None)
    parser.add_argument("--leaves", default="16,64,256")
    parser.add_argument(
        "--root-label-shares",
        default="1.0",
        help="Comma-separated root label probabilities. 1.0 reproduces root-only/dense endpoints.",
    )
    parser.add_argument("--sample-rates", default="0,0.01,0.03,0.10,1.0")
    parser.add_argument(
        "--skip-zero-label-cell",
        action="store_true",
        help="Skip cells with root_label_share=0 and sampled_node_rate=0 because they expose no oracle labels.",
    )
    parser.add_argument("--n-train", type=int, default=8192)
    parser.add_argument("--n-val", type=int, default=1024)
    parser.add_argument("--min-tokens", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--universe-size", type=int, default=512)
    parser.add_argument("--zipf-alphas", default="0.8,1.0,1.2")
    parser.add_argument("--precision", type=int, default=8)
    parser.add_argument(
        "--schedule",
        default="g",
        help="Training schedule over f/g stages, e.g. g, gf, gfgf, fgfgfg.",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--rollout-min-docs-per-batch", type=int, default=16)
    parser.add_argument("--rollout-max-docs-per-batch", type=int, default=0)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=65536,
        help="Validation/eval row batch target passed to the diagnostic runner.",
    )
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument(
        "--target-transform",
        choices=("linear01", "log1p_zscore", "zscore"),
        default="log1p_zscore",
        help=(
            "Scalar target transform for the HLL readout. log1p_zscore avoids "
            "linear01 clipping that can mask off-fiber HLL states."
        ),
    )
    parser.add_argument("--hidden-channels", type=int, default=512)
    parser.add_argument("--head-hidden-dim", type=int, default=512)
    parser.add_argument("--n-modes", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--f-learning-rate", default="1e-4")
    parser.add_argument("--g-learning-rate", default="1e-4")
    parser.add_argument("--local-law-weight", type=float, default=0.5)
    parser.add_argument("--local-law-leaf-discount-gamma", type=float, default=1.0)
    parser.add_argument("--merge-output-constraint", choices=("none", "unit_clamp"), default="none")
    parser.add_argument("--objective-loss-weight", type=float, default=1.0)
    parser.add_argument("--state-loss-weight", type=float, default=0.0)
    parser.add_argument("--exact-state-anchor-weight", type=float, default=0.0)
    parser.add_argument(
        "--allow-dense-regularizers",
        action="store_true",
        help=(
            "Allow nonzero state/exact-state auxiliary losses. The sampled "
            "paper grid defaults to scalar-only IPW supervision."
        ),
    )
    parser.add_argument("--eval-every-epochs", type=int, default=1)
    parser.add_argument("--progress-every-epochs", type=int, default=1)
    parser.add_argument("--progress-every-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reuse-from-root",
        type=Path,
        action="append",
        default=[],
        help="Existing sampled grid root used as a source of compatible stage-prefix checkpoints.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands only; do not write files.")
    parser.add_argument("--aggregate-only", action="store_true", help="Refresh grid_summary.csv/README.md for an existing manifest.")
    parser.add_argument("--launch", action="store_true", help="Launch one long_job runner per GPU after writing the manifest.")
    args = parser.parse_args(argv)
    if any(ch not in {"f", "g"} for ch in str(args.schedule)):
        raise ValueError("--schedule must be a string over {'f','g'}, e.g. g or fgfgfg")
    if int(args.epochs) <= 0:
        raise ValueError("--epochs must be positive")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")
    if int(args.rollout_min_docs_per_batch) <= 0:
        raise ValueError("--rollout-min-docs-per-batch must be positive")
    if int(args.rollout_max_docs_per_batch) < 0:
        raise ValueError("--rollout-max-docs-per-batch must be non-negative")
    if (
        int(args.rollout_max_docs_per_batch) > 0
        and int(args.rollout_max_docs_per_batch) < int(args.rollout_min_docs_per_batch)
    ):
        raise ValueError("--rollout-max-docs-per-batch must be 0 or at least --rollout-min-docs-per-batch")
    if float(args.state_loss_weight) < 0.0:
        raise ValueError("--state-loss-weight must be non-negative")
    if float(args.exact_state_anchor_weight) < 0.0:
        raise ValueError("--exact-state-anchor-weight must be non-negative")
    if float(args.local_law_leaf_discount_gamma) < 0.0:
        raise ValueError("--local-law-leaf-discount-gamma must be non-negative")
    _assert_scalar_only_or_explicitly_anchored(args)

    output_root = args.output_root
    if output_root is None:
        output_root = Path("outputs") / f"hll_sampled_node_rate_grid_{_utc_stamp()}"
    output_root = output_root.resolve()
    if args.sample_cache_dir is None:
        args.sample_cache_dir = output_root / "sample_cache"
    else:
        args.sample_cache_dir = Path(args.sample_cache_dir).resolve()

    if args.aggregate_only:
        rows = aggregate(output_root)
        print(f"aggregated {len(rows)} cells into {output_root / 'grid_summary.csv'}")
        return 0

    cells = build_cells(args, output_root)
    if args.dry_run:
        _print_dry_run(cells)
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    if args.reuse_from_root:
        cells = apply_reuse_sources(
            cells,
            reuse_roots=args.reuse_from_root,
            gpu_ids=_parse_int_csv(args.gpu_ids),
        )
    manifest = write_manifest_and_runners(args, output_root, cells)
    rows = aggregate(output_root)
    print(f"wrote manifest: {output_root / 'manifest.json'}")
    print(f"wrote pending summary: {output_root / 'grid_summary.csv'}")
    print(f"wrote launch notes: {output_root / 'README.md'}")
    print(f"prepared {len(rows)} cells across {len(manifest.get('runners', []))} GPU runners")
    if args.launch:
        launched = launch_runners(args, manifest)
        print(f"launched {len(launched)} GPU runner jobs")
    else:
        print("not launched; pass --launch with the same --output-root to start")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
