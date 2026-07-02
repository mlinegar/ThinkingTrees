#!/usr/bin/env python3
"""Prepare or launch the expanded HLL canonical f/g observation grid.

By default this writes a manifest plus one sequential runner per GPU, but does
not launch anything. Pass --launch when the prepared grid should start.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
UNIFIED_SRC = REPO_ROOT / "parallel" / "unified_g_v1" / "src"
TREEPO_SRC = REPO_ROOT / "treepo" / "src"
for _path in (REPO_ROOT, TREEPO_SRC, UNIFIED_SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.tree.hll import HLLConfig, HyperLogLogSketch
from unified_g_v1.sketch.classical_parity import ClassicalHLLParityConfig, generate_documents

DEFAULT_GPU_IDS = (0, 1, 2, 3)
EXPECTED_METRICS = (
    "root_mae",
    "root_rel_mae",
    "leaf_readout_mae",
    "internal_readout_mae",
    "root_readout_mae",
    "merge_state_mae",
    "merge_state_internal_mae",
    "merge_state_root_mae",
    "merge_readout_internal_mae",
    "merge_readout_root_mae",
    "official_f_on_learned_root_mae",
    "official_f_on_learned_root_rel_mae",
    "learned_f_on_exact_root_mae",
    "fstar_gstar_root_analytic_mae",
    "fstar_gstar_root_analytic_rel_mae",
    "hll_tree_flat_max_abs_diff",
    "hll_tree_flat_register_max_diff",
)


@dataclass(frozen=True)
class GridCell:
    cell_id: str
    family: str
    description: str
    n_leaves: int
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


def _parse_float_csv(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one float")
    return values


def _parse_schedule_csv(raw: str, *, allow_empty: bool = False) -> list[str]:
    values = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not values:
        if allow_empty:
            return []
        raise ValueError("expected at least one schedule")
    bad = [value for value in values if any(ch not in {"f", "g"} for ch in value)]
    if bad:
        raise ValueError(f"bad schedule(s) {bad}; schedules must be strings over {{'f','g'}}")
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


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _base_command(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    n_leaves: int,
    device: str = "cuda",
) -> list[str]:
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
        "linear01",
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
        str(device),
        "--seed",
        str(args.seed),
        "--no-identity-residual-init",
        "--local-law-weight",
        str(args.local_law_weight),
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


def _estimated_row_work(args: argparse.Namespace, n_leaves: int) -> int:
    # One stage sees f rows over all nodes and g rows over all merge rows.
    return int(args.epochs) * int(args.n_train) * max(1, int(3 * n_leaves - 2))


def _estimated_stage_work(args: argparse.Namespace, *, n_leaves: int, epochs: int, schedule: str) -> int:
    return int(epochs) * max(1, len(str(schedule))) * int(args.n_train) * max(1, int(2 * n_leaves - 1))


def _make_cell(
    args: argparse.Namespace,
    *,
    output_root: Path,
    cell_id: str,
    family: str,
    description: str,
    n_leaves: int,
    objective_mode: str,
    observation_design: str,
    schedule: str = "fgfg",
    epochs: int | None = None,
    extra: Sequence[str] = (),
) -> GridCell:
    output_dir = output_root / cell_id
    command = _base_command(args, output_dir, n_leaves=n_leaves)
    resolved_epochs = int(args.epochs if epochs is None else epochs)
    command.extend(
        [
            "--schedule",
            str(schedule),
            "--objective-mode",
            objective_mode,
            "--oracle-observation-design",
            observation_design,
            "--epochs",
            str(resolved_epochs),
            "--objective-loss-weight",
            str(args.objective_loss_weight),
        ]
    )
    command.extend(str(item) for item in extra)
    return GridCell(
        cell_id=cell_id,
        family=family,
        description=description,
        n_leaves=int(n_leaves),
        gpu_id=-1,
        estimated_row_work=_estimated_stage_work(
            args,
            n_leaves=int(n_leaves),
            epochs=resolved_epochs,
            schedule=str(schedule),
        ),
        output_dir=str(output_dir),
        command=command,
        expected_metrics=list(EXPECTED_METRICS),
    )


def _make_baseline_cell(output_root: Path, *, n_leaves: int) -> GridCell:
    return GridCell(
        cell_id=f"fstar_gstar_L{int(n_leaves)}",
        family="fstar_gstar",
        description="Package/native HLL f* after exact registerwise-max g* baseline; no training.",
        n_leaves=int(n_leaves),
        gpu_id=-1,
        estimated_row_work=0,
        output_dir=str(output_root / f"fstar_gstar_L{int(n_leaves)}"),
        command=[],
        expected_metrics=list(EXPECTED_METRICS),
    )


def _make_precompute_cell(args: argparse.Namespace, *, output_root: Path, n_leaves: int) -> GridCell:
    output_dir = output_root / f"precompute_L{int(n_leaves)}"
    command = _base_command(args, output_dir, n_leaves=int(n_leaves), device="cpu")
    command.extend(
        [
            "--schedule",
            "g",
            "--objective-mode",
            "exact_rows",
            "--oracle-observation-design",
            "root_only",
            "--epochs",
            "1",
            "--precompute-samples-only",
        ]
    )
    return GridCell(
        cell_id=f"precompute_L{int(n_leaves)}",
        family="precompute",
        description="Populate exact HLL tree/oracle sample cache only.",
        n_leaves=int(n_leaves),
        gpu_id=-1,
        estimated_row_work=int(args.n_train) * max(1, int(2 * n_leaves - 1)),
        output_dir=str(output_dir),
        command=command,
        expected_metrics=[],
    )


def _assign_gpus(cells: Sequence[GridCell], gpu_ids: Sequence[int]) -> list[GridCell]:
    loads = {int(gpu_id): 0 for gpu_id in gpu_ids}
    assigned: list[GridCell] = []
    for cell in sorted(cells, key=lambda item: (-int(item.estimated_row_work), item.cell_id)):
        if not cell.command:
            assigned.append(cell)
            continue
        gpu_id = min(loads, key=lambda key: (loads[key], key))
        loads[gpu_id] += int(cell.estimated_row_work)
        assigned.append(
            GridCell(
                cell_id=cell.cell_id,
                family=cell.family,
                description=cell.description,
                n_leaves=cell.n_leaves,
                gpu_id=int(gpu_id),
                estimated_row_work=cell.estimated_row_work,
                output_dir=cell.output_dir,
                command=cell.command,
                expected_metrics=cell.expected_metrics,
            )
        )
    return sorted(assigned, key=lambda item: (item.gpu_id, -item.estimated_row_work, item.cell_id))


def _known_f_exact_cell_id(schedule: str, leaves: int) -> str:
    return f"known_f_{schedule}_exact_L{int(leaves)}"


def _canonical_exact_cell_id(schedule: str, leaves: int) -> str:
    if schedule == "fgfg":
        return f"exact_formula_noid_L{int(leaves)}"
    return f"exact_{schedule}_formula_noid_L{int(leaves)}"


def _rollout_cell_id(prefix: str, schedule: str, leaves: int) -> str:
    if prefix.startswith("known_f_"):
        if schedule == "gf":
            return f"{prefix}_L{int(leaves)}"
        return f"{prefix}_{schedule}_L{int(leaves)}"
    if schedule == "fgfg":
        return f"{prefix}_formula_noid_L{int(leaves)}"
    return f"{prefix}_{schedule}_formula_noid_L{int(leaves)}"


def build_cells(args: argparse.Namespace, output_root: Path) -> list[GridCell]:
    exact_leaves = _parse_int_csv(args.exact_leaves)
    rollout_leaves = _parse_int_csv(args.rollout_leaves)
    budget_leaves = _parse_int_csv(args.budget_leaves)
    specs: list[GridCell] = []
    all_leaf_set = {*exact_leaves, *rollout_leaves}
    if bool(args.include_budgeted_mass):
        all_leaf_set.update(budget_leaves)
    all_leaves = sorted(all_leaf_set)

    if bool(args.precompute_only):
        return _assign_gpus(
            [_make_precompute_cell(args, output_root=output_root, n_leaves=leaf) for leaf in all_leaves],
            _parse_int_csv(args.gpu_ids),
        )

    if bool(args.include_baseline):
        specs.extend(_make_baseline_cell(output_root, n_leaves=leaf) for leaf in all_leaves)

    known_f_exact_schedules = _parse_schedule_csv(args.known_f_exact_schedules, allow_empty=True)
    canonical_exact_schedules = _parse_schedule_csv(args.canonical_exact_schedules, allow_empty=True)
    known_f_rollout_schedules = _parse_schedule_csv(args.known_f_rollout_schedules, allow_empty=True)
    canonical_rollout_schedules = _parse_schedule_csv(args.canonical_rollout_schedules, allow_empty=True)

    for leaves in exact_leaves:
        if bool(args.include_known_f):
            for schedule in known_f_exact_schedules:
                specs.append(
                    _make_cell(
                        args,
                        output_root=output_root,
                        cell_id=_known_f_exact_cell_id(schedule, leaves),
                        family=f"fixed_f_exact_{schedule}",
                        description=f"Supplied HLL readout f*, exact-row learned schedule {schedule}.",
                        n_leaves=leaves,
                        objective_mode="exact_rows",
                        observation_design="root_only",
                        schedule=schedule,
                        epochs=int(args.g_exact_epochs if schedule == "g" else args.gfgf_exact_epochs),
                    )
                )
        for schedule in canonical_exact_schedules:
            specs.append(
                _make_cell(
                    args,
                    output_root=output_root,
                    cell_id=_canonical_exact_cell_id(schedule, leaves),
                    family=f"canonical_exact_{schedule}",
                    description=f"Canonical learned f_theta,g_theta exact-row schedule {schedule}.",
                    n_leaves=leaves,
                    objective_mode="exact_rows",
                    observation_design="root_only",
                    schedule=schedule,
                    epochs=int(args.fgfg_exact_epochs),
                )
            )

    rollout_specs = [
        ("rollout_root", "root_only", canonical_rollout_schedules, (), "Rollout/local-law with root observations only."),
        ("rollout_dense", "dense_oracle", canonical_rollout_schedules, (), "Rollout/local-law with every node oracle-observed."),
    ]
    if bool(args.include_known_f):
        rollout_specs.extend(
            [
                (
                    "known_f_gf_rollout_root",
                    "root_only",
                    known_f_rollout_schedules,
                    (),
                    "Supplied HLL readout f*, rollout/local-law with root observations only.",
                ),
                (
                    "known_f_gf_rollout_dense",
                    "dense_oracle",
                    known_f_rollout_schedules,
                    (),
                    "Supplied HLL readout f*, rollout/local-law with every node oracle-observed.",
                ),
            ]
        )
    if bool(args.include_sampled1):
        rollout_specs.append(
            (
                "rollout_sampled1",
                "sampled_nodes",
                canonical_rollout_schedules,
                ("--sampled-node-rate", "1.0"),
                "Rollout/local-law through sampled-node code path with all non-root nodes sampled.",
            )
        )
    for prefix, observation, schedules, extra, description in rollout_specs:
        for leaves in rollout_leaves:
            for schedule in schedules:
                specs.append(
                    _make_cell(
                        args,
                        output_root=output_root,
                        cell_id=_rollout_cell_id(prefix, schedule, leaves),
                        family=f"{prefix}_{schedule}",
                        description=f"{description} Schedule {schedule}.",
                        n_leaves=leaves,
                        objective_mode="rollout_local_law",
                        observation_design=observation,
                        schedule=schedule,
                        epochs=int(args.rollout_epochs),
                        extra=extra,
                    )
                )

    if bool(args.include_budgeted_mass):
        for share in _parse_int_csv(args.root_label_shares):
            root_share = float(share) / 100.0
            for leaves in budget_leaves:
                specs.append(
                    _make_cell(
                        args,
                        output_root=output_root,
                        cell_id=f"budget_R{share}_formula_noid_L{leaves}",
                        family=f"budget_R{share}",
                        description=(
                            f"Budgeted-mass rollout with {share}% root-label share and "
                            "fixed total observation mass per document."
                        ),
                        n_leaves=leaves,
                        objective_mode="rollout_local_law",
                        observation_design="budgeted_mass",
                        schedule="fgfg",
                        epochs=int(args.rollout_epochs),
                        extra=(
                            "--root-label-share",
                            f"{root_share:.6g}",
                            "--mass-target-per-doc",
                            str(args.mass_target_per_doc),
                            "--local-label-pool",
                            str(args.local_label_pool),
                            "--local-label-allocation",
                            "span_mass",
                        ),
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


def _matching_resume_prefix_score(target: GridCell, source: Mapping[str, Any]) -> tuple[int, Path] | None:
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
    match_flags = (
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
        "--seed",
        "--objective-mode",
        "--oracle-observation-design",
    )
    for flag in match_flags:
        if _flag_value(target_command, flag) != _flag_value(source_command, flag):
            return None
    checkpoint = (
        Path(str(source.get("output_dir", "")))
        / "hll_register_space"
        / f"stage_{len(source_schedule):02d}_{source_schedule[-1]}_model.pt"
    )
    if not checkpoint.exists():
        return None
    return (len(source_schedule), checkpoint)


def _patched_resume_cell(target: GridCell, *, prefix: str, checkpoint: Path) -> GridCell:
    full_schedule = _flag_value(target.command, "--schedule")
    if not full_schedule.startswith(prefix) or full_schedule == prefix:
        raise ValueError(f"cannot resume {target.cell_id}: {prefix=} {full_schedule=}")
    suffix = full_schedule[len(prefix) :]
    command = _replace_flag_value(target.command, "--schedule", suffix)
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
    return GridCell(
        cell_id=target.cell_id,
        family=target.family,
        description=(
            f"{target.description} Resumes from {prefix} checkpoint "
            f"{checkpoint} and trains suffix {suffix}."
        ),
        n_leaves=target.n_leaves,
        gpu_id=target.gpu_id,
        estimated_row_work=_estimated_stage_work(
            argparse.Namespace(n_train=_flag_value(command, "--n-train", "1")),
            n_leaves=target.n_leaves,
            epochs=epochs,
            schedule=suffix,
        ),
        output_dir=target.output_dir,
        command=command,
        expected_metrics=target.expected_metrics,
    )


def _with_estimated_work(cell: GridCell, estimated_row_work: int) -> GridCell:
    return GridCell(
        cell_id=cell.cell_id,
        family=cell.family,
        description=cell.description,
        n_leaves=cell.n_leaves,
        gpu_id=cell.gpu_id,
        estimated_row_work=int(estimated_row_work),
        output_dir=cell.output_dir,
        command=cell.command,
        expected_metrics=cell.expected_metrics,
    )


def apply_reuse_sources(
    cells: Sequence[GridCell],
    *,
    output_root: Path,
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
        destination = Path(cell.output_dir)
        exact_source = next(
            (
                source
                for source in source_cells
                if str(source.get("cell_id", "")) == cell.cell_id
                and (Path(str(source.get("output_dir", ""))) / "summary.csv").exists()
            ),
            None,
        )
        if exact_source is not None and not destination.exists():
            shutil.copytree(Path(str(exact_source["output_dir"])), destination)

        if _summary_path(cell).exists() or not cell.command:
            patched.append(_with_estimated_work(cell, 0) if _summary_path(cell).exists() else cell)
            continue

        candidates: list[tuple[int, Path]] = []
        for source in source_cells:
            candidate = _matching_resume_prefix_score(cell, source)
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            patched.append(cell)
            continue
        prefix_len, checkpoint = max(candidates, key=lambda item: item[0])
        full_schedule = _flag_value(cell.command, "--schedule")
        prefix = full_schedule[:prefix_len]
        patched.append(_patched_resume_cell(cell, prefix=prefix, checkpoint=checkpoint))
    return _assign_gpus(patched, gpu_ids)


def _write_summary_csv(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(row.keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in columns})


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _write_baseline_summary(args: argparse.Namespace, cell: GridCell) -> None:
    if _summary_path(cell).exists():
        return
    hll_cfg = HLLConfig(precision=int(args.precision), hash_bits=64)
    cfg = ClassicalHLLParityConfig(
        precision=int(args.precision),
        n_leaves=int(cell.n_leaves),
        leaf_size=None,
        schedule="balanced",
        backend="native",
        n_val=int(args.n_train) + int(args.n_val),
        seed=int(args.seed),
        universe_size=int(args.universe_size),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        zipf_alphas=_parse_float_csv(args.zipf_alphas),
        oracle_kind="analytic",
    )
    docs = generate_documents(cfg)
    val_docs = docs[int(args.n_train) : int(args.n_train) + int(args.n_val)]
    root_abs: list[float] = []
    root_rel: list[float] = []
    leaf_abs: list[float] = []
    leaf_rel: list[float] = []
    tree_flat_est_diffs: list[float] = []
    tree_flat_register_diffs: list[float] = []
    for leaves, truth, flat in val_docs:
        root_registers: np.ndarray | None = None
        for leaf in leaves:
            leaf_sketch = HyperLogLogSketch.from_tokens(hll_cfg, list(leaf))
            leaf_truth = float(len(set(leaf)))
            leaf_est = float(leaf_sketch.estimate())
            leaf_err = abs(leaf_est - leaf_truth)
            leaf_abs.append(leaf_err)
            leaf_rel.append(leaf_err / max(1.0, abs(leaf_truth)))
            root_registers = (
                leaf_sketch.registers.copy()
                if root_registers is None
                else np.maximum(root_registers, leaf_sketch.registers)
            )
        if root_registers is None:
            raise RuntimeError(f"no HLL leaf registers produced for {cell.cell_id}")
        flat_sketch = HyperLogLogSketch.from_tokens(hll_cfg, list(flat))
        tree_est = float(HyperLogLogSketch.from_registers(hll_cfg, root_registers).estimate())
        flat_est = float(flat_sketch.estimate())
        err = abs(tree_est - float(truth))
        root_abs.append(err)
        root_rel.append(err / max(1.0, abs(float(truth))))
        tree_flat_est_diffs.append(abs(tree_est - flat_est))
        tree_flat_register_diffs.append(
            float(np.max(np.abs(root_registers.astype(int) - flat_sketch.registers.astype(int))))
        )

    row = {
        "target_kind": "hll_register_space",
        "schedule": "package_hll",
        "n_leaves": int(cell.n_leaves),
        "objective_mode": "exact_package",
        "observation": "analytic_truth",
        "readout_arch": "package_hll",
        "readout_kind": "hll_reference",
        "identity_residual_init": "False",
        "root_mae": 0.0,
        "root_rel_mae": 0.0,
        "fstar_gstar_target_mae": 0.0,
        "fstar_gstar_target_rel_mae": 0.0,
        "fstar_gstar_analytic_mae": _mean(root_abs),
        "fstar_gstar_analytic_rel_mae": _mean(root_rel),
        "fstar_gstar_root_analytic_mae": _mean(root_abs),
        "fstar_gstar_root_analytic_rel_mae": _mean(root_rel),
        "fstar_gstar_leaf_analytic_mae": _mean(leaf_abs),
        "fstar_gstar_leaf_analytic_rel_mae": _mean(leaf_rel),
        "hll_tree_flat_max_abs_diff": max(tree_flat_est_diffs) if tree_flat_est_diffs else float("nan"),
        "hll_tree_flat_mean_abs_diff": _mean(tree_flat_est_diffs),
        "hll_tree_flat_register_max_diff": max(tree_flat_register_diffs) if tree_flat_register_diffs else float("nan"),
        "precision": int(args.precision),
        "universe_size": int(args.universe_size),
        "min_tokens": int(args.min_tokens),
        "max_tokens": int(args.max_tokens),
        "zipf_alphas": str(args.zipf_alphas),
        "seed": int(args.seed),
        "n_train": int(args.n_train),
        "n_val": int(args.n_val),
        "baseline_kind": "fstar_gstar_tree_vs_flat",
    }
    if float(row["hll_tree_flat_max_abs_diff"]) != 0.0 or float(row["hll_tree_flat_register_max_diff"]) != 0.0:
        raise RuntimeError(
            f"HLL tree/flat mismatch for L={cell.n_leaves}: "
            f"estimate_diff={row['hll_tree_flat_max_abs_diff']} "
            f"register_diff={row['hll_tree_flat_register_max_diff']}"
        )
    _write_summary_csv(_summary_path(cell), row)


def write_baseline_summaries(args: argparse.Namespace, cells: Sequence[GridCell]) -> None:
    for cell in cells:
        if cell.family == "fstar_gstar":
            _write_baseline_summary(args, cell)


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
                f"echo '[hll-canonical-grid] start {cell.cell_id} on visible GPU {gpu_id}'",
                f"mkdir -p {shlex.quote(str(out_dir))}",
                f"if [ -f {shlex.quote(str(_summary_path(cell)))} ]; then",
                f"  echo '[hll-canonical-grid] skip completed {cell.cell_id}'",
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
                f"    echo '[hll-canonical-grid] failed {cell.cell_id} code='\"$code\"",
                "    failures=$((failures + 1))",
                "  else",
                f"    echo '[hll-canonical-grid] done {cell.cell_id}'",
                "  fi",
                "fi",
                "",
            ]
        )
    lines.extend(
        [
            "if [ \"$failures\" -ne 0 ]; then",
            "  echo '[hll-canonical-grid] runner finished with failures='\"$failures\"",
            "  exit 1",
            "fi",
            "echo '[hll-canonical-grid] runner complete'",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def write_manifest_and_runners(args: argparse.Namespace, output_root: Path, cells: Sequence[GridCell]) -> dict[str, Any]:
    by_gpu: dict[int, list[GridCell]] = {}
    for cell in cells:
        if not cell.command:
            continue
        by_gpu.setdefault(int(cell.gpu_id), []).append(cell)
    runners: list[dict[str, Any]] = []
    for gpu_id, gpu_cells in sorted(by_gpu.items()):
        runner_path = output_root / "runners" / f"gpu_{gpu_id}.sh"
        _write_runner(runner_path, gpu_id=gpu_id, cells=gpu_cells)
        runners.append(
            {
                "gpu_id": gpu_id,
                "runner_path": str(runner_path),
                "job_root": str(output_root / "launchers" / f"gpu_{gpu_id}"),
                "cell_ids": [cell.cell_id for cell in gpu_cells],
                "estimated_row_work": int(sum(cell.estimated_row_work for cell in gpu_cells)),
            }
        )

    manifest = {
        "schema_version": "hll_canonical_observation_grid.v2",
        "created_at": _utc_now(),
        "repo_root": str(REPO_ROOT),
        "output_root": str(output_root),
        "launched": False,
        "gpu_ids": _parse_int_csv(args.gpu_ids),
        "sample_cache_dir": str(args.sample_cache_dir),
        "precompute_only": bool(args.precompute_only),
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
            "batch_size": int(args.batch_size),
            "rollout_min_docs_per_batch": int(args.rollout_min_docs_per_batch),
            "rollout_max_docs_per_batch": int(args.rollout_max_docs_per_batch),
            "eval_batch_size": int(args.eval_batch_size),
            "epochs": int(args.epochs),
            "g_exact_epochs": int(args.g_exact_epochs),
            "gfgf_exact_epochs": int(args.gfgf_exact_epochs),
            "fgfg_exact_epochs": int(args.fgfg_exact_epochs),
            "rollout_epochs": int(args.rollout_epochs),
        },
        "cells": [asdict(cell) for cell in cells],
        "runners": runners,
        "notes": [
            "Default behavior is prelaunch only; use --launch to start long_job runners.",
            "This paper grid fixes the HLL DGP at min_tokens=max_tokens=1024 by default.",
            "fstar_gstar cells are exact package-HLL baselines and are computed at manifest-preparation time.",
            "known_f_* cells supply the package/formula HLL readout f* and learn g_theta.",
            "canonical_exact cells match the current red exact_formula_noid_L* family.",
            "rollout_root is the root-only observation family; canonical_exact is exact-row supervised and is not root-only.",
            "All cells use the current HLL recipe: p=8, state_dim=256, hll_formula readout, no identity residual init, fgfg schedule.",
            "All cells share --sample-cache-dir so exact trees and oracle node scores are persisted once per data-generation config.",
            "Batch size is held fixed for comparability; higher leaf counts naturally create more local-state batches per epoch.",
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
            "n_leaves": cell.n_leaves,
            "gpu_id": cell.gpu_id,
            "estimated_row_work": cell.estimated_row_work,
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

    lines = [
        "# HLL Canonical Observation Grid",
        "",
        "Prepared grid; no jobs are launched unless `--launch` is passed.",
        f"Token regime: `{manifest.get('dgp', {}).get('min_tokens')}` to `{manifest.get('dgp', {}).get('max_tokens')}` tokens/document.",
        f"Sample cache: `{manifest.get('sample_cache_dir', '')}`",
        "",
        "| family | status | cells |",
        "|---|---|---:|",
    ]
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row["family"]), str(row["status"]))
        counts[key] = counts.get(key, 0) + 1
    for (family, status), count in sorted(counts.items()):
        lines.append(f"| {family} | {status} | {count} |")
    lines.extend(
        [
            "",
            "## Launch",
            "",
            "```bash",
            f"./venv/bin/python scripts/run_hll_canonical_observation_grid.py --output-root {shlex.quote(str(output_root))} --launch",
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
        name = f"hll_canonical_observation_grid_{Path(str(manifest['output_root'])).name}_gpu{gpu_id}"
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
    print(f"HLL canonical observation grid: {len(cells)} cells")
    for cell in cells:
        print(f"[gpu {cell.gpu_id}] {cell.cell_id} | {cell.family} | L={cell.n_leaves}")
        if cell.command:
            print("  " + shlex.join(cell.command))
        else:
            print("  baseline summary generated during manifest preparation")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--gpu-ids", default=",".join(str(x) for x in DEFAULT_GPU_IDS))
    parser.add_argument("--python-bin", default="./venv/bin/python")
    parser.add_argument("--sample-cache-dir", type=Path, default=None)
    parser.add_argument("--exact-leaves", default="16,64,256")
    parser.add_argument("--rollout-leaves", default="16,64,256")
    parser.add_argument("--budget-leaves", default="16,64,256")
    parser.add_argument("--include-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-known-f", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-sampled1", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-budgeted-mass", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--known-f-exact-schedules",
        default="gfgf",
        help="Comma-separated exact-row schedules for fixed-HLL-readout lanes. Intermediate stages are reported from checkpoints.",
    )
    parser.add_argument(
        "--canonical-exact-schedules",
        default="fgfg",
        help="Comma-separated exact-row schedules for learned-readout canonical lanes.",
    )
    parser.add_argument(
        "--known-f-rollout-schedules",
        default="gf",
        help="Comma-separated rollout schedules for fixed-HLL-readout lanes.",
    )
    parser.add_argument(
        "--canonical-rollout-schedules",
        default="fgfg",
        help="Comma-separated rollout schedules for learned-readout canonical lanes.",
    )
    parser.add_argument(
        "--reuse-from-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Existing grid root to copy completed matching cells from and to use as a "
            "source of stage checkpoints for longer schedules."
        ),
    )
    parser.add_argument("--root-label-shares", default="0,50,100")
    parser.add_argument("--mass-target-per-doc", type=float, default=1.0)
    parser.add_argument("--local-label-pool", choices=("nonroot", "leaves", "internal"), default="nonroot")
    parser.add_argument("--n-train", type=int, default=8192)
    parser.add_argument("--n-val", type=int, default=1024)
    parser.add_argument("--min-tokens", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--universe-size", type=int, default=512)
    parser.add_argument("--zipf-alphas", default="0.8,1.0,1.2")
    parser.add_argument("--precision", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--g-exact-epochs", type=int, default=30)
    parser.add_argument("--gfgf-exact-epochs", type=int, default=20)
    parser.add_argument("--fgfg-exact-epochs", type=int, default=20)
    parser.add_argument("--rollout-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--rollout-min-docs-per-batch", type=int, default=16)
    parser.add_argument("--rollout-max-docs-per-batch", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=65536)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--hidden-channels", type=int, default=512)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--n-modes", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--f-learning-rate", default="1e-4")
    parser.add_argument("--g-learning-rate", default="1e-4")
    parser.add_argument("--local-law-weight", type=float, default=0.5)
    parser.add_argument("--state-loss-weight", type=float, default=1.0)
    parser.add_argument("--exact-state-anchor-weight", type=float, default=0.1)
    parser.add_argument("--objective-loss-weight", type=float, default=1.0)
    parser.add_argument("--eval-every-epochs", type=int, default=1)
    parser.add_argument("--progress-every-epochs", type=int, default=1)
    parser.add_argument("--progress-every-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precompute-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands only; do not write files.")
    parser.add_argument("--aggregate-only", action="store_true", help="Refresh grid_summary.csv/README.md for an existing manifest.")
    parser.add_argument("--launch", action="store_true", help="Launch one long_job runner per GPU after writing the manifest.")
    args = parser.parse_args(argv)

    if int(args.min_tokens) != int(args.max_tokens):
        raise ValueError("canonical HLL paper grid requires fixed document mass: --min-tokens must equal --max-tokens")
    if int(args.min_tokens) <= 0:
        raise ValueError("--min-tokens/--max-tokens must be positive")
    if int(args.n_train) <= 0 or int(args.n_val) <= 0:
        raise ValueError("--n-train and --n-val must be positive")
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

    output_root = args.output_root
    if output_root is None:
        output_root = Path("outputs") / f"hll_canonical_observation_grid_{_utc_stamp()}"
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
            output_root=output_root,
            reuse_roots=args.reuse_from_root,
            gpu_ids=_parse_int_csv(args.gpu_ids),
        )
    if not bool(args.precompute_only):
        write_baseline_summaries(args, cells)
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
