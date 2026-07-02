#!/usr/bin/env python3
"""Launch and summarize the HLL recoverability diagnostic grid.

This wrapper keeps two lanes separate:

* current_fno: the Lean-aligned FNO diagnostic over exact HLL register states.
* latent_sidecar: older learned-HLL parity tasks used only as capacity checks.

The launcher writes one sequential runner per GPU so a full grid can run without
oversubscribing every visible device with many simultaneous jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GPU_IDS = (0, 1, 2, 3)


@dataclass(frozen=True)
class GridCell:
    cell_id: str
    lane: str
    category: str
    description: str
    gpu_id: int
    output_dir: str
    command: list[str]
    expected_metrics: list[str]
    success_rule: str
    comparable_to_lean_objective: bool


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_int_csv(raw: str) -> list[int]:
    values: list[int] = []
    for item in str(raw).split(","):
        text = item.strip()
        if text:
            values.append(int(text))
    if not values:
        raise ValueError("expected at least one integer")
    return values


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _read_first_csv(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return dict(rows[0]) if rows else None


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _format_float(value: Any) -> str:
    number = _float_or_nan(value)
    if not math.isfinite(number):
        return ""
    return f"{number:.6g}"


def _fno_base_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    return [
        str(args.python_bin),
        "scripts/run_fno_mergeable_sketch_diagnostic.py",
        "--targets",
        "hll_register_space",
        "--n-train",
        str(args.current_n_train),
        "--n-val",
        str(args.current_n_val),
        "--batch-size",
        str(args.current_batch_size),
        "--target-transform",
        "linear01",
        "--state-normalization",
        "register_div64",
        "--hidden-channels",
        "512",
        "--head-hidden-dim",
        "256",
        "--n-modes",
        "32",
        "--n-layers",
        "2",
        "--f-learning-rate",
        "1e-4",
        "--g-learning-rate",
        "1e-4",
        "--local-law-weight",
        "0.5",
        "--state-loss-weight",
        "1.0",
        "--exact-state-anchor-weight",
        "0.1",
        "--eval-every-epochs",
        "1",
        "--no-identity-residual-init",
        "--device",
        "cuda",
        "--seed",
        str(args.seed),
        "--output-dir",
        str(output_dir),
    ]


def _fno_cell(
    *,
    args: argparse.Namespace,
    cell_id: str,
    category: str,
    description: str,
    output_dir: Path,
    gpu_id: int,
    schedule: str,
    objective_mode: str,
    n_leaves: int,
    epochs: int,
    success_rule: str,
    readout_arch: str = "fno_mlp",
    observation_design: str = "root_only",
    extra: Sequence[str] = (),
    objective_loss_weight: float = 1.0,
) -> GridCell:
    command = _fno_base_command(args, output_dir)
    command.extend(
        [
            "--schedule",
            schedule,
            "--objective-mode",
            objective_mode,
            "--n-leaves",
            str(n_leaves),
            "--epochs",
            str(epochs),
            "--readout-arch",
            readout_arch,
            "--oracle-observation-design",
            observation_design,
            "--objective-loss-weight",
            str(objective_loss_weight),
        ]
    )
    command.extend(str(item) for item in extra)
    return GridCell(
        cell_id=cell_id,
        lane="current_fno",
        category=category,
        description=description,
        gpu_id=int(gpu_id),
        output_dir=str(output_dir),
        command=command,
        expected_metrics=[
            "root_mae",
            "root_rel_mae",
            "learned_f_on_exact_root_mae",
            "official_f_on_learned_root_mae",
            "merge_state_mae",
            "train_root_loss_end",
            "train_local_loss_end",
            "train_observed_rows_end",
        ],
        success_rule=success_rule,
        comparable_to_lean_objective=True,
    )


def _latent_command(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    method: str,
    n_leaves: int,
    embedding_dim: int,
    summary_dim: int,
    state_dim: int | None,
    hidden_dim: int,
) -> list[str]:
    command = [
        str(args.python_bin),
        "scripts/run_classical_parity_benchmark.py",
        "--out",
        str(output_dir),
        "--methods",
        method,
        "--learned-precisions",
        "8",
        "--learned-leaf-counts",
        str(n_leaves),
        "--learned-seeds",
        str(args.seed),
        "--learned-oracle-kinds",
        "hll_reference",
        "--learned-n-train",
        str(args.latent_n_train),
        "--learned-n-val",
        str(args.latent_n_val),
        "--learned-min-tokens",
        "128",
        "--learned-max-tokens",
        "128",
        "--learned-universe-size",
        str(args.latent_universe_size),
        "--learned-n-epochs",
        str(args.latent_epochs),
        "--learned-batch-size",
        str(args.latent_batch_size),
        "--learned-lr",
        str(args.latent_learning_rate),
        "--learned-embedding-dim",
        str(embedding_dim),
        "--learned-summary-dim",
        str(summary_dim),
        "--learned-hidden-dim",
        str(hidden_dim),
        "--learned-local-law-weight",
        "0.9",
        "--learned-merge-state-weight",
        "100.0",
        "--learned-use-cuda",
        "--learned-cuda-device",
        "0",
        "--jobs",
        "1",
        "--tables-dir",
        str(output_dir / "tables"),
    ]
    if state_dim is not None:
        command.extend(["--learned-state-dim", str(state_dim)])
    return command


def _latent_cell(
    *,
    args: argparse.Namespace,
    cell_id: str,
    category: str,
    description: str,
    output_dir: Path,
    gpu_id: int,
    method: str,
    n_leaves: int,
    embedding_dim: int,
    summary_dim: int,
    state_dim: int | None,
    hidden_dim: int,
    success_rule: str,
    objective_loss_weight: float = 1.0,
) -> GridCell:
    return GridCell(
        cell_id=cell_id,
        lane="latent_sidecar",
        category=category,
        description=description,
        gpu_id=int(gpu_id),
        output_dir=str(output_dir),
        command=_latent_command(
            args=args,
            output_dir=output_dir,
            method=method,
            n_leaves=n_leaves,
            embedding_dim=embedding_dim,
            summary_dim=summary_dim,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
        ),
        expected_metrics=["val_mae", "root_mae", "root_rel_mae", "merge_state_mae"],
        success_rule=success_rule,
        comparable_to_lean_objective=False,
    )


def _round_robin_gpu(index: int, gpu_ids: Sequence[int]) -> int:
    return int(gpu_ids[index % len(gpu_ids)])


def build_cells(args: argparse.Namespace, output_root: Path) -> list[GridCell]:
    gpu_ids = _parse_int_csv(args.gpu_ids)
    cells: list[GridCell] = []

    current_root = output_root / "current_fno"
    fno_specs = [
        dict(
            cell_id="current_readout_f_L1_fno",
            category="readout_isolation",
            description="Exact-row f-only readout on single-leaf exact HLL registers.",
            output_dir=current_root / "readout_f_L1_fno",
            schedule="f",
            objective_mode="exact_rows",
            n_leaves=1,
            epochs=80,
            readout_arch="fno_mlp",
            success_rule="pass if root_mae < 1.0",
        ),
        dict(
            cell_id="current_readout_f_L4_fno",
            category="readout_isolation",
            description="Exact-row f-only readout on all exact HLL tree nodes for four leaves.",
            output_dir=current_root / "readout_f_L4_fno",
            schedule="f",
            objective_mode="exact_rows",
            n_leaves=4,
            epochs=80,
            readout_arch="fno_mlp",
            success_rule="pass if learned_f_on_exact_root_mae < 1.0",
        ),
        dict(
            cell_id="current_readout_f_L4_deep",
            category="readout_isolation",
            description="Deep-MLP non-FNO readout control on exact HLL tree nodes.",
            output_dir=current_root / "readout_f_L4_deep",
            schedule="f",
            objective_mode="exact_rows",
            n_leaves=4,
            epochs=80,
            readout_arch="deep_mlp",
            success_rule="pass if learned_f_on_exact_root_mae < 1.0",
        ),
        dict(
            cell_id="current_merge_g_L4_state",
            category="merge_isolation",
            description="Exact-row g-only state recovery with scalar loss disabled.",
            output_dir=current_root / "merge_g_L4_state",
            schedule="g",
            objective_mode="exact_rows",
            n_leaves=4,
            epochs=20,
            readout_arch="fno_mlp",
            objective_loss_weight=0.0,
            success_rule="pass if official_f_on_learned_root_mae < 0.1 and merge_state_mae < 0.1",
        ),
        dict(
            cell_id="current_historical_fgfg_L4_exact",
            category="historical_ceiling",
            description="Legacy exact-row fgfg ceiling using the historically successful recipe.",
            output_dir=current_root / "historical_fgfg_L4_exact",
            schedule="fgfg",
            objective_mode="exact_rows",
            n_leaves=4,
            epochs=20,
            readout_arch="fno_mlp",
            success_rule="pass if root_mae < 1.0",
        ),
        dict(
            cell_id="current_rollout_fgfg_L4_dense",
            category="corrected_rollout",
            description="Corrected rollout fgfg with every node oracle-observed.",
            output_dir=current_root / "rollout_fgfg_L4_dense_oracle",
            schedule="fgfg",
            objective_mode="rollout_local_law",
            observation_design="dense_oracle",
            n_leaves=4,
            epochs=20,
            readout_arch="fno_mlp",
            success_rule="pass if dense oracle root_mae < 1.0",
        ),
        dict(
            cell_id="current_rollout_fgfg_L4_sampled1",
            category="corrected_rollout",
            description="Corrected rollout through sampled-node metadata with all non-root nodes sampled.",
            output_dir=current_root / "rollout_fgfg_L4_sampled_nodes_1p0",
            schedule="fgfg",
            objective_mode="rollout_local_law",
            observation_design="sampled_nodes",
            extra=("--sampled-node-rate", "1.0"),
            n_leaves=4,
            epochs=20,
            readout_arch="fno_mlp",
            success_rule="pass if close to dense_oracle root_mae",
        ),
        dict(
            cell_id="current_rollout_fgfg_L4_root",
            category="corrected_rollout",
            description="Corrected rollout with root-only oracle observations.",
            output_dir=current_root / "rollout_fgfg_L4_root_only",
            schedule="fgfg",
            objective_mode="rollout_local_law",
            observation_design="root_only",
            n_leaves=4,
            epochs=20,
            readout_arch="fno_mlp",
            success_rule="diagnostic; should be finite and improve stage-over-stage",
        ),
    ]
    for share, label in ((1.0, "R100"), (0.5, "R50"), (0.0, "R0")):
        fno_specs.append(
            dict(
                cell_id=f"current_rollout_fgfg_L4_budget_{label}",
                category="budgeted_mass",
                description=f"Corrected rollout budgeted-mass design {label}.",
                output_dir=current_root / f"rollout_fgfg_L4_budget_{label.lower()}",
                schedule="fgfg",
                objective_mode="rollout_local_law",
                observation_design="budgeted_mass",
                extra=(
                    "--root-label-share",
                    f"{share:.1f}",
                    "--mass-target-per-doc",
                    "1.0",
                    "--local-label-pool",
                    "nonroot",
                    "--local-label-allocation",
                    "span_mass",
                ),
                n_leaves=4,
                epochs=20,
                readout_arch="fno_mlp",
                success_rule="pass if row/mass counts match the budget and losses are finite",
            )
        )

    for spec in fno_specs:
        gpu_id = _round_robin_gpu(len(cells), gpu_ids)
        cells.append(
            _fno_cell(
                args=args,
                gpu_id=gpu_id,
                extra=spec.get("extra", ()),
                observation_design=spec.get("observation_design", "root_only"),
                **{k: v for k, v in spec.items() if k not in {"extra", "observation_design"}},
            )
        )

    latent_root = output_root / "latent_sidecar"
    for method in ("learned_g_oracle_state", "learned_g"):
        for n_leaves in (1, 4, 8):
            gpu_id = _round_robin_gpu(len(cells), gpu_ids)
            cells.append(
                _latent_cell(
                    args=args,
                    cell_id=f"latent_{method}_L{n_leaves}_d256",
                    category=method,
                    description=f"{method} sidecar with register-width state/readout capacity.",
                    output_dir=latent_root / f"{method}_L{n_leaves}_d256",
                    gpu_id=gpu_id,
                    method=method,
                    n_leaves=n_leaves,
                    embedding_dim=256,
                    summary_dim=256,
                    state_dim=256,
                    hidden_dim=512,
                    success_rule="sidecar capacity diagnostic; not Lean-objective comparable",
                )
            )

    latent_caps = (
        (128, 256, 512, "d256"),
        (256, 512, 1024, "d512"),
    )
    for summary_dim, state_dim, hidden_dim, cap_label in latent_caps:
        for n_leaves in (1, 4, 8):
            gpu_id = _round_robin_gpu(len(cells), gpu_ids)
            cells.append(
                _latent_cell(
                    args=args,
                    cell_id=f"latent_learned_joint_L{n_leaves}_{cap_label}",
                    category="learned_joint",
                    description=f"Fully learned latent f/g sidecar with summary/state/hidden={summary_dim}/{state_dim}/{hidden_dim}.",
                    output_dir=latent_root / f"learned_joint_L{n_leaves}_{cap_label}",
                    gpu_id=gpu_id,
                    method="learned_joint",
                    n_leaves=n_leaves,
                    embedding_dim=summary_dim,
                    summary_dim=summary_dim,
                    state_dim=state_dim,
                    hidden_dim=hidden_dim,
                    success_rule="sidecar capacity diagnostic; not Lean-objective comparable",
                )
            )

    return cells


def _cell_summary_path(cell: GridCell) -> Path:
    out = Path(cell.output_dir)
    if cell.lane == "latent_sidecar":
        return out / "hll" / "summary.csv"
    return out / "summary.csv"


def _latest_stage_metrics(cell: GridCell) -> dict[str, Any]:
    path = Path(cell.output_dir) / "hll_register_space" / "stage_metrics.json"
    if not path.exists():
        path = Path(cell.output_dir) / "stage_metrics.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(payload, list) and payload:
        item = payload[-1]
        return dict(item) if isinstance(item, dict) else {}
    return {}


def _exit_code(cell: GridCell) -> int | None:
    path = Path(cell.output_dir) / "exit_code.txt"
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _status_for(cell: GridCell) -> str:
    if _cell_summary_path(cell).exists():
        return "completed"
    code = _exit_code(cell)
    if code is None:
        return "pending_or_running"
    return "failed" if code else "completed_without_summary"


def _rule_result(cell: GridCell, row: Mapping[str, Any], dense_root_mae: float | None = None) -> str:
    if _status_for(cell) != "completed":
        return ""
    root_mae = _float_or_nan(row.get("root_mae"))
    learned_f_exact = _float_or_nan(row.get("learned_f_on_exact_root_mae"))
    official = _float_or_nan(row.get("official_f_on_learned_root_mae"))
    merge_state = _float_or_nan(row.get("merge_state_mae"))
    if cell.category == "readout_isolation":
        score = learned_f_exact if math.isfinite(learned_f_exact) else root_mae
        return "pass" if math.isfinite(score) and score < 1.0 else "fail"
    if cell.category == "merge_isolation":
        return (
            "pass"
            if math.isfinite(official)
            and official < 0.1
            and math.isfinite(merge_state)
            and merge_state < 0.1
            else "fail"
        )
    if cell.category == "historical_ceiling":
        return "pass" if math.isfinite(root_mae) and root_mae < 1.0 else "fail"
    if cell.cell_id == "current_rollout_fgfg_L4_dense":
        return "pass" if math.isfinite(root_mae) and root_mae < 1.0 else "fail"
    if cell.cell_id == "current_rollout_fgfg_L4_sampled1":
        if dense_root_mae is None or not math.isfinite(root_mae):
            return "unknown"
        tolerance = max(0.1, 0.05 * max(1.0, abs(dense_root_mae)))
        return "pass" if abs(root_mae - dense_root_mae) <= tolerance else "fail"
    if cell.category == "budgeted_mass":
        observed = _float_or_nan(row.get("train_observed_rows_end"))
        local_loss = _float_or_nan(row.get("train_local_loss_end"))
        return "pass" if math.isfinite(observed) and observed >= 0.0 and math.isfinite(local_loss) else "fail"
    if cell.lane == "latent_sidecar":
        return "sidecar"
    return "diagnostic"


def aggregate(output_root: Path) -> tuple[list[dict[str, Any]], str]:
    manifest_path = output_root / "manifest.json"
    manifest = _load_json(manifest_path)
    cells = [GridCell(**item) for item in manifest.get("cells", [])]

    raw_rows: list[dict[str, Any]] = []
    dense_root_mae: float | None = None
    for cell in cells:
        summary = _read_first_csv(_cell_summary_path(cell)) or {}
        if cell.cell_id == "current_rollout_fgfg_L4_dense" and summary:
            dense_root_mae = _float_or_nan(summary.get("root_mae"))
        stage = _latest_stage_metrics(cell)
        row = {
            "cell_id": cell.cell_id,
            "lane": cell.lane,
            "category": cell.category,
            "description": cell.description,
            "status": _status_for(cell),
            "gpu_id": cell.gpu_id,
            "output_dir": cell.output_dir,
            "summary_path": str(_cell_summary_path(cell)),
            "exit_code": "" if _exit_code(cell) is None else _exit_code(cell),
            "comparable_to_lean_objective": cell.comparable_to_lean_objective,
            "success_rule": cell.success_rule,
            "root_mae": summary.get("root_mae", ""),
            "root_rel_mae": summary.get("root_rel_mae", ""),
            "val_mae": summary.get("val_mae", ""),
            "learned_f_on_exact_root_mae": summary.get("learned_f_on_exact_root_mae", ""),
            "official_f_on_learned_root_mae": summary.get("official_f_on_learned_root_mae", ""),
            "merge_state_mae": summary.get("merge_state_mae", ""),
            "train_root_loss_end": stage.get("train_root_loss_end", ""),
            "train_local_loss_end": stage.get("train_local_loss_end", ""),
            "train_objective_loss_end": stage.get("train_objective_loss_end", ""),
            "train_observed_rows_end": stage.get("train_observed_rows_end", ""),
            "train_population_rows_end": stage.get("train_population_rows_end", ""),
            "train_root_observed_rows_end": stage.get("train_root_observed_rows_end", ""),
            "train_root_population_rows_end": stage.get("train_root_population_rows_end", ""),
            "train_observed_mass_end": stage.get("train_observed_mass_end", ""),
            "train_population_mass_end": stage.get("train_population_mass_end", ""),
        }
        raw_rows.append(row)

    for row in raw_rows:
        cell = next(item for item in cells if item.cell_id == row["cell_id"])
        row["rule_result"] = _rule_result(cell, row, dense_root_mae=dense_root_mae)

    summary_path = output_root / "grid_summary.csv"
    columns = list(raw_rows[0].keys()) if raw_rows else []
    if columns:
        with summary_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            writer.writerows(raw_rows)

    report = _render_report(output_root, raw_rows, manifest)
    report_path = output_root / "grid_report.md"
    report_path.write_text(report, encoding="utf-8")
    return raw_rows, report


def _render_report(output_root: Path, rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get("status", ""))] = counts.get(str(row.get("status", "")), 0) + 1

    lines = [
        "# HLL Recoverability Grid",
        "",
        f"- Output root: `{output_root}`",
        f"- Generated: `{_utc_now()}`",
        f"- Cells: `{len(rows)}`",
        f"- Status counts: `{counts}`",
        f"- Sidecar latent rows are capacity diagnostics and are not Lean-objective comparable.",
        "",
        "## Current FNO",
        "",
        "| cell | category | status | rule | root MAE | f exact MAE | official-on-learned MAE | merge MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row.get("lane") != "current_fno":
            continue
        lines.append(
            "| {cell_id} | {category} | {status} | {rule_result} | {root_mae} | {f_exact} | {official} | {merge} |".format(
                cell_id=row.get("cell_id", ""),
                category=row.get("category", ""),
                status=row.get("status", ""),
                rule_result=row.get("rule_result", ""),
                root_mae=_format_float(row.get("root_mae")),
                f_exact=_format_float(row.get("learned_f_on_exact_root_mae")),
                official=_format_float(row.get("official_f_on_learned_root_mae")),
                merge=_format_float(row.get("merge_state_mae")),
            )
        )

    lines.extend(
        [
            "",
            "## Local-Law Rows",
            "",
            "| cell | observed rows | population rows | root observed | observed mass | local loss | root loss |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        if row.get("lane") != "current_fno":
            continue
        if row.get("category") not in {"corrected_rollout", "budgeted_mass"}:
            continue
        lines.append(
            "| {cell_id} | {obs} | {pop} | {root_obs} | {mass} | {local_loss} | {root_loss} |".format(
                cell_id=row.get("cell_id", ""),
                obs=_format_float(row.get("train_observed_rows_end")),
                pop=_format_float(row.get("train_population_rows_end")),
                root_obs=_format_float(row.get("train_root_observed_rows_end")),
                mass=_format_float(row.get("train_observed_mass_end")),
                local_loss=_format_float(row.get("train_local_loss_end")),
                root_loss=_format_float(row.get("train_root_loss_end")),
            )
        )

    lines.extend(
        [
            "",
            "## Latent Sidecar",
            "",
            "| cell | method | status | val MAE | root MAE | merge MAE |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        if row.get("lane") != "latent_sidecar":
            continue
        lines.append(
            "| {cell_id} | {category} | {status} | {val_mae} | {root_mae} | {merge} |".format(
                cell_id=row.get("cell_id", ""),
                category=row.get("category", ""),
                status=row.get("status", ""),
                val_mae=_format_float(row.get("val_mae")),
                root_mae=_format_float(row.get("root_mae")),
                merge=_format_float(row.get("merge_state_mae")),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- If readout isolation fails, the learned score head is already a bottleneck.",
            "- If merge isolation passes while learned readout fails, HLL register semantics are learnable but the neural readout is weak.",
            "- If exact-row fgfg passes while dense rollout fails, the rollout objective/training path is the issue.",
            "- If sampled-nodes 1.0 diverges from dense-oracle, sampled metadata/projection is wrong.",
            "- Budgeted-mass rows should show finite losses and the expected shift from root observations to local observations.",
            "",
            "## Manifest",
            "",
            f"- Manifest path: `{output_root / 'manifest.json'}`",
            f"- Runner count: `{len(manifest.get('runners', []))}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_runner(path: Path, *, repo_root: Path, gpu_id: int, cells: Sequence[GridCell]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(repo_root))}",
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(str(gpu_id))}",
        "failures=0",
        f"echo '[hll-grid] gpu={gpu_id} cells={len(cells)}'",
    ]
    for cell in cells:
        out_dir = Path(cell.output_dir)
        log_path = out_dir / "run.log"
        exit_path = out_dir / "exit_code.txt"
        summary_path = _cell_summary_path(cell)
        lines.extend(
            [
                "",
                f"echo '[hll-grid] start {cell.cell_id}'",
                f"mkdir -p {shlex.quote(str(out_dir))}",
                f"if [ -f {shlex.quote(str(exit_path))} ] && [ \"$(cat {shlex.quote(str(exit_path))})\" = \"0\" ] && [ -f {shlex.quote(str(summary_path))} ]; then",
                f"  echo '[hll-grid] skip completed {cell.cell_id}'",
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
                f"    echo '[hll-grid] failed {cell.cell_id} code='\"$code\"",
                "    failures=$((failures + 1))",
                "  else",
                f"    echo '[hll-grid] done {cell.cell_id}'",
                "  fi",
                "fi",
            ]
        )
    lines.extend(
        [
            "",
            "if [ \"$failures\" -ne 0 ]; then",
            "  echo '[hll-grid] runner finished with failures='\"$failures\"",
            "  exit 1",
            "fi",
            "echo '[hll-grid] runner complete'",
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
        _write_runner(runner_path, repo_root=REPO_ROOT, gpu_id=gpu_id, cells=gpu_cells)
        runners.append(
            {
                "gpu_id": gpu_id,
                "runner_path": str(runner_path),
                "job_root": str(output_root / "launchers" / f"gpu_{gpu_id}"),
                "cell_ids": [cell.cell_id for cell in gpu_cells],
            }
        )

    manifest = {
        "schema_version": "hll_recoverability_grid.v1",
        "created_at": _utc_now(),
        "repo_root": str(REPO_ROOT),
        "output_root": str(output_root),
        "gpu_ids": _parse_int_csv(args.gpu_ids),
        "cells": [asdict(cell) for cell in cells],
        "runners": runners,
        "notes": [
            "current_fno rows are claim-relevant Lean-objective diagnostics.",
            "latent_sidecar rows are capacity diagnostics and are not Lean-objective comparable.",
            "Runners execute their assigned cells sequentially on one CUDA-visible GPU.",
        ],
    }
    _json_dump(output_root / "manifest.json", manifest)
    return manifest


def launch_runners(args: argparse.Namespace, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    launched: list[dict[str, Any]] = []
    for runner in manifest.get("runners", []):
        gpu_id = int(runner["gpu_id"])
        runner_path = str(runner["runner_path"])
        job_root = str(runner["job_root"])
        name = f"hll_recoverability_grid_{Path(str(manifest['output_root'])).name}_gpu{gpu_id}"
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
        if result.returncode == 0:
            try:
                parsed = json.loads(result.stdout)
            except Exception:
                parsed = {}
            if isinstance(parsed, dict):
                payload.update(
                    {
                        "manifest_path": parsed.get("manifest_path"),
                        "status_command": parsed.get("status_command"),
                        "tail_command": parsed.get("tail_command"),
                    }
                )
        launched.append(payload)
        if result.returncode != 0:
            raise RuntimeError(
                f"failed to launch gpu runner {gpu_id}: {result.stderr.strip() or result.stdout.strip()}"
            )
    launch_path = Path(str(manifest["output_root"])) / "launched_runners.json"
    launch_path.write_text(json.dumps(launched, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return launched


def _print_dry_run(cells: Sequence[GridCell]) -> None:
    print(f"HLL recoverability grid: {len(cells)} cells")
    for cell in cells:
        print(
            f"[gpu {cell.gpu_id}] {cell.cell_id} | {cell.lane} | {cell.category} | {cell.description}"
        )
        print("  " + shlex.join(cell.command))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch and summarize the HLL recoverability mini grid.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root. Defaults to outputs/hll_recoverability_grid_<timestamp>.",
    )
    parser.add_argument("--gpu-ids", default=",".join(str(x) for x in DEFAULT_GPU_IDS))
    parser.add_argument("--python-bin", default="./venv/bin/python")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--current-n-train", type=int, default=8192)
    parser.add_argument("--current-n-val", type=int, default=1024)
    parser.add_argument("--current-batch-size", type=int, default=64)
    parser.add_argument("--latent-n-train", type=int, default=4096)
    parser.add_argument("--latent-n-val", type=int, default=512)
    parser.add_argument("--latent-universe-size", type=int, default=10_000)
    parser.add_argument("--latent-epochs", type=int, default=150)
    parser.add_argument("--latent-batch-size", type=int, default=16)
    parser.add_argument("--latent-learning-rate", type=float, default=1e-3)
    parser.add_argument("--dry-run", action="store_true", help="Print the planned cells and commands only.")
    parser.add_argument("--launch", action="store_true", help="Launch one sequential runner per GPU.")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Read an existing manifest and rebuild grid_summary.csv/grid_report.md.",
    )
    args = parser.parse_args(argv)

    output_root = args.output_root
    if output_root is None:
        output_root = Path("outputs") / f"hll_recoverability_grid_{_utc_stamp()}"
    output_root = output_root.resolve()

    if args.aggregate_only:
        rows, _report = aggregate(output_root)
        print(f"aggregated {len(rows)} cells into {output_root / 'grid_summary.csv'}")
        return 0

    cells = build_cells(args, output_root)
    if args.dry_run:
        _print_dry_run(cells)
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = write_manifest_and_runners(args, output_root, cells)
    rows, _report = aggregate(output_root)
    print(f"wrote manifest: {output_root / 'manifest.json'}")
    print(f"wrote pending summary: {output_root / 'grid_summary.csv'}")
    if args.launch:
        launched = launch_runners(args, manifest)
        print(f"launched {len(launched)} GPU runner jobs")
        for item in launched:
            if item.get("status_command"):
                print(item["status_command"])
    else:
        print("not launched; pass --launch to start the grid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
