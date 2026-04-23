#!/usr/bin/env python3
"""Focused high-capacity factored-theorem-surface search.

This runner keeps the latent tree merger generic while forcing all theorem and
root supervision to factor through one learned theorem feature. It screens a
small set of shared-feature variants at 128 docs, validates the best ones at
256 docs against a slotwise control, and only then promotes winners to 1024
and 5120.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_tree_neural_full_doc_mig as mig  # noqa: E402
from scripts import run_tree_neural_learning_push as lp  # noqa: E402


PHASE1_FULL_DENSE_LABEL = "internal_full_dense"
PHASE2_CONDITIONS = (
    ("root_only", 0.0, "count_only", "none", 0.0),
    ("leaf_dense", 1.0, "full_sketch", "none", 0.0),
    ("internal_count_dense", 1.0, "count_only", "count_only", 1.0),
    ("internal_full_dense", 1.0, "full_sketch", "full_sketch", 1.0),
    ("internal_full_r0p25", 0.25, "full_sketch", "full_sketch", 0.25),
)
PHASE3_CONDITIONS = (
    ("root_only", 0.0, "count_only", "none", 0.0),
    ("leaf_dense", 1.0, "full_sketch", "none", 0.0),
    ("internal_count_r0p25", 0.25, "count_only", "count_only", 0.25),
    ("internal_full_r0p25", 0.25, "full_sketch", "full_sketch", 0.25),
    ("internal_full_dense", 1.0, "full_sketch", "full_sketch", 1.0),
)
PHASE4_CONDITIONS = (
    ("internal_full_dense", 1.0, "full_sketch", "full_sketch", 1.0),
    ("internal_full_r0p25", 0.25, "full_sketch", "full_sketch", 0.25),
)
SHARED_FEATURE_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "label": "shared_feature_phi128_decodeheavy",
        "surface_mode": "shared_feature",
        "phi_dim": 128,
        "phi_hidden_dim": 256,
        "task_weight": 1.5,
        "phi_compose_weight": 0.25,
        "phi_contrastive_weight": 0.0,
    },
    {
        "label": "shared_feature_phi128_balanced",
        "surface_mode": "shared_feature",
        "phi_dim": 128,
        "phi_hidden_dim": 256,
        "task_weight": 1.0,
        "phi_compose_weight": 0.5,
        "phi_contrastive_weight": 0.05,
    },
    {
        "label": "shared_feature_phi192_decodeheavy",
        "surface_mode": "shared_feature",
        "phi_dim": 192,
        "phi_hidden_dim": 384,
        "task_weight": 1.5,
        "phi_compose_weight": 0.25,
        "phi_contrastive_weight": 0.0,
    },
    {
        "label": "shared_feature_phi192_balanced",
        "surface_mode": "shared_feature",
        "phi_dim": 192,
        "phi_hidden_dim": 384,
        "task_weight": 1.0,
        "phi_compose_weight": 0.5,
        "phi_contrastive_weight": 0.05,
    },
    {
        "label": "shared_feature_adapters_phi128",
        "surface_mode": "shared_feature_adapters",
        "phi_dim": 128,
        "phi_hidden_dim": 256,
        "task_weight": 1.5,
        "phi_compose_weight": 0.25,
        "phi_contrastive_weight": 0.0,
    },
    {
        "label": "shared_feature_adapters_phi192",
        "surface_mode": "shared_feature_adapters",
        "phi_dim": 192,
        "phi_hidden_dim": 384,
        "task_weight": 1.5,
        "phi_compose_weight": 0.25,
        "phi_contrastive_weight": 0.0,
    },
    # --- Fiber-primary variants: contrastive drives phi, task is auxiliary ---
    # Uses oracle_metric_name="markov" for theory-aligned continuous oracle
    # distances instead of discrete adapter equivalence classes.
    {
        "label": "fiber_primary_phi128",
        "surface_mode": "shared_feature",
        "phi_dim": 128,
        "phi_hidden_dim": 256,
        "task_weight": 0.25,
        "phi_compose_weight": 0.5,
        "phi_contrastive_weight": 2.0,
        "c2_mode": "fiber",
        "oracle_metric_name": "markov",
    },
    {
        "label": "fiber_primary_phi192",
        "surface_mode": "shared_feature",
        "phi_dim": 192,
        "phi_hidden_dim": 384,
        "task_weight": 0.25,
        "phi_compose_weight": 0.5,
        "phi_contrastive_weight": 2.0,
        "c2_mode": "fiber",
        "oracle_metric_name": "markov",
    },
)
CONTROL_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "label": "shared_bottleneck48_control",
        "surface_mode": "shared_bottleneck",
        "phi_dim": 48,
        "phi_hidden_dim": 256,
        "task_weight": 1.0,
        "phi_compose_weight": 1.0,
        "phi_contrastive_weight": 0.25,
    },
    {
        "label": "slotwise_control",
        "surface_mode": "slotwise",
        "phi_dim": 48,
        "phi_hidden_dim": 256,
        "task_weight": 1.0,
        "phi_compose_weight": 0.0,
        "phi_contrastive_weight": 0.0,
    },
)


def _parser() -> argparse.ArgumentParser:
    parser = lp._parser()
    parser.description = __doc__
    parser.set_defaults(
        output_root=f"outputs/tree_neural_shared_feature_push_{lp._timestamp()}",
        phase1_train_small=128,
        phase1_train_large=256,
        phase2_train_large=1024,
        phase2_train_x5=5120,
        phase2_seeds=(0, 1, 2, 3, 4),
        n_epochs=52,
        tree_stage1_epochs=12,
        tree_stage2_epochs=40,
    )
    parser.add_argument(
        "--mig-uuids",
        type=str,
        default="",
        help="Optional comma/space separated MIG UUID subset for this runner.",
    )
    parser.add_argument(
        "--phase4-seeds",
        nargs="*",
        type=int,
        default=(0, 1, 2),
        help="Seeds used for the promoted 5120-doc step.",
    )
    return parser


def _resolve_mig_uuids(args: argparse.Namespace) -> list[str]:
    raw = str(getattr(args, "mig_uuids", "") or "").strip()
    mig_uuids = mig._parse_mig_uuids(raw) if raw else mig._discover_mig_uuids()
    if not mig_uuids:
        raise RuntimeError("No MIG UUIDs discovered")
    return list(mig_uuids)


def _write_outputs(output_root: Path) -> None:
    payload = mig._write_summary_outputs(output_root)
    exact = mig._tree_neural_exact_sanity_summary(dict(payload or {}))
    (output_root / "tree_neural_exact_sanity_summary.json").write_text(
        json.dumps(exact, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / "tree_neural_exact_sanity_summary.md").write_text(
        mig._render_exact_sanity_summary_markdown(exact),
        encoding="utf-8",
    )


def _collect_all_runs(output_root: Path) -> list[dict[str, Any]]:
    summary_path = output_root / "summary.json"
    if not summary_path.exists():
        return []
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return [dict(run) for run in payload.get("runs") or []]


def _run_jobs(
    *,
    output_root: Path,
    jobs: Sequence[mig._JobSpec],
    torch_threads: int,
    use_cuda: bool,
    mig_uuids: Sequence[str],
    mode: str,
) -> Dict[str, Any]:
    manifest = {
        "mode": str(mode),
        "jobs": [asdict(job) for job in jobs],
    }
    return mig._run_job_batch(
        output_root=output_root,
        jobs=jobs,
        mig_uuids=tuple(str(uuid) for uuid in mig_uuids),
        resume_enabled=True,
        use_cuda=bool(use_cuda),
        torch_threads=int(torch_threads),
        manifest_payload=manifest,
    )


def _variant_label_from_config_label(config_label: str) -> str:
    return str(config_label).split("__", 1)[0]


def _is_control_variant(label: str) -> bool:
    return str(label).startswith("slotwise_control") or str(label).startswith(
        "shared_bottleneck48_control"
    )


def _make_surface_config(
    args: argparse.Namespace,
    *,
    train_doc_count: int,
    label: str,
    leaf_label_rate: float,
    leaf_supervision_kind: str,
    internal_supervision_kind: str,
    internal_label_rate: float,
    surface_mode: str,
    theorem_feature_dim: int,
    theorem_feature_hidden_dim: int,
    task_weight: float,
    phi_compose_weight: float,
    phi_contrastive_weight: float,
    c2_mode: str = "reconstruction",
    theorem_feature_adapter: str = "markov_count_sketch",
    oracle_metric_name: str = "",
) -> mig._RunConfigSpec:
    base = lp._make_slot_config(
        args,
        train_doc_count=int(train_doc_count),
        label=str(label),
        leaf_label_rate=float(leaf_label_rate),
        leaf_supervision_kind=str(leaf_supervision_kind),
        internal_supervision_kind=str(internal_supervision_kind),
        internal_label_rate=float(internal_label_rate),
        tree_summary_spec_root_mode="factored_theorem_readout",
        n_epochs=int(args.n_epochs),
        tree_training_schedule="two_stage",
        tree_stage1_epochs=int(args.tree_stage1_epochs),
        tree_stage2_epochs=int(args.tree_stage2_epochs),
        tree_task_objective_weight=float(task_weight),
    )
    return replace(
        base,
        tree_task_head_mode="theorem_feature_scalar",
        tree_theorem_surface_mode=str(surface_mode),
        tree_theorem_count_head_mode="scalar_mse",
        tree_theorem_feature_dim=int(theorem_feature_dim),
        tree_theorem_feature_hidden_dim=int(theorem_feature_hidden_dim),
        tree_phi_compose_weight=float(phi_compose_weight),
        tree_phi_contrastive_weight=float(phi_contrastive_weight),
        tree_phi_alignment_loss="cosine_mse",
        tree_checkpoint_metric="val_exact_sketch_direct",
        tree_stage1_checkpoint_metric="val_theorem_bootstrap_direct",
        tree_summary_spec_root_mode="factored_theorem_readout",
        tree_c2_mode=str(c2_mode),
        theorem_feature_adapter=str(theorem_feature_adapter),
        oracle_metric_name=str(oracle_metric_name),
    )


def _build_jobs_for_configs(
    args: argparse.Namespace,
    configs_by_train: Sequence[tuple[int, mig._RunConfigSpec]],
    *,
    seeds: Sequence[int],
    tuning_stage: str,
    axis_value: str,
) -> list[mig._JobSpec]:
    jobs: list[mig._JobSpec] = []
    for train_doc_count, config in configs_by_train:
        jobs.extend(
            mig._build_jobs_for_configs(
                families=(mig.EXACT_SANITY_FAMILY,),
                train_doc_counts=(int(train_doc_count),),
                benchmark=str(args.benchmark),
                hardness_grid="",
                grid_cell_ids=(),
                seeds=tuple(int(seed) for seed in seeds),
                job_granularity="family_train_seed",
                repeat_closed_form_controls=False,
                configs=(config,),
                tuning_stage=str(tuning_stage),
                study_name="shared_feature_push",
                study_axis="promotion_stage",
                axis_value=str(axis_value),
                selection_metric="exact_sketch_diagnostic_only",
            )
        )
    return jobs


def _phase1_configs(args: argparse.Namespace) -> list[tuple[int, mig._RunConfigSpec]]:
    configs: list[tuple[int, mig._RunConfigSpec]] = []
    train_doc_count = int(args.phase1_train_small)
    for variant in [*SHARED_FEATURE_VARIANTS, *CONTROL_VARIANTS]:
        configs.append(
            (
                train_doc_count,
                _make_surface_config(
                    args,
                    train_doc_count=train_doc_count,
                    label=str(variant["label"]),
                    leaf_label_rate=1.0,
                    leaf_supervision_kind="full_sketch",
                    internal_supervision_kind="full_sketch",
                    internal_label_rate=1.0,
                    surface_mode=str(variant["surface_mode"]),
                    theorem_feature_dim=int(variant["phi_dim"]),
                    theorem_feature_hidden_dim=int(variant["phi_hidden_dim"]),
                    task_weight=float(variant["task_weight"]),
                    phi_compose_weight=float(variant["phi_compose_weight"]),
                    phi_contrastive_weight=float(variant["phi_contrastive_weight"]),
                    c2_mode=str(variant.get("c2_mode", "reconstruction")),
                    theorem_feature_adapter=str(variant.get("theorem_feature_adapter", "markov_count_sketch")),
                    oracle_metric_name=str(variant.get("oracle_metric_name", "")),
                ),
            )
        )
    return configs


def _phase1_passes_gate(run: Mapping[str, Any]) -> bool:
    return (
        lp._direct_metric(run, "root_direct_count_mae", default=1e9) <= 0.70
        and lp._direct_metric(run, "leaf_direct_exact_match", default=0.0) >= 0.70
        and lp._direct_metric(run, "merge_direct_exact_match", default=0.0) >= 0.50
        and lp._direct_metric(run, "phi_merge_alignment", default=0.0) >= 0.90
    )


def _phase1_score(run: Mapping[str, Any]) -> float:
    root = lp._direct_metric(run, "root_direct_count_mae", default=1e9)
    leaf = lp._direct_metric(run, "leaf_direct_exact_match", default=0.0)
    merge = lp._direct_metric(run, "merge_direct_exact_match", default=0.0)
    phi_alignment = lp._direct_metric(run, "phi_merge_alignment", default=0.0)
    return float(
        root
        + max(0.0, 0.70 - leaf) * 2.0
        + max(0.0, 0.50 - merge) * 3.0
        + max(0.0, 0.90 - phi_alignment) * 2.0
    )


def _select_phase1_promotions(runs: Sequence[Mapping[str, Any]]) -> list[str]:
    candidates: list[dict[str, Any]] = []
    for run in runs:
        if str(run.get("tuning_stage", "")) != "phase1":
            continue
        label = str(run.get("config_label", "")).strip()
        if not label or _is_control_variant(label):
            continue
        if not _phase1_passes_gate(run):
            continue
        candidates.append(dict(run))
    ranked = sorted(candidates, key=_phase1_score)
    winners: list[str] = []
    for run in ranked:
        label = str(run.get("config_label", "")).strip()
        if label not in winners:
            winners.append(label)
        if len(winners) >= 2:
            break
    return winners


def _phase2_configs(
    args: argparse.Namespace,
    promoted_variants: Sequence[str],
) -> list[tuple[int, mig._RunConfigSpec]]:
    train_doc_count = int(args.phase1_train_large)
    variant_map = {
        str(variant["label"]): dict(variant)
        for variant in [*SHARED_FEATURE_VARIANTS, *CONTROL_VARIANTS]
    }
    configs: list[tuple[int, mig._RunConfigSpec]] = []
    for variant_label in promoted_variants:
        variant = variant_map[str(variant_label)]
        for condition_label, leaf_rate, leaf_kind, internal_kind, internal_rate in PHASE2_CONDITIONS:
            configs.append(
                (
                    train_doc_count,
                    _make_surface_config(
                        args,
                        train_doc_count=train_doc_count,
                        label=f"{variant_label}__{condition_label}_{train_doc_count}",
                        leaf_label_rate=float(leaf_rate),
                        leaf_supervision_kind=str(leaf_kind),
                        internal_supervision_kind=str(internal_kind),
                        internal_label_rate=float(internal_rate),
                        surface_mode=str(variant["surface_mode"]),
                        theorem_feature_dim=int(variant["phi_dim"]),
                        theorem_feature_hidden_dim=int(variant["phi_hidden_dim"]),
                        task_weight=float(variant["task_weight"]),
                        phi_compose_weight=float(variant["phi_compose_weight"]),
                        phi_contrastive_weight=float(variant["phi_contrastive_weight"]),
                        c2_mode=str(variant.get("c2_mode", "reconstruction")),
                        theorem_feature_adapter=str(variant.get("theorem_feature_adapter", "markov_count_sketch")),
                    oracle_metric_name=str(variant.get("oracle_metric_name", "")),
                    ),
                )
            )
    for control in CONTROL_VARIANTS:
        configs.append(
            (
                train_doc_count,
                _make_surface_config(
                    args,
                    train_doc_count=train_doc_count,
                    label=f"{control['label']}__{PHASE1_FULL_DENSE_LABEL}_{train_doc_count}",
                    leaf_label_rate=1.0,
                    leaf_supervision_kind="full_sketch",
                    internal_supervision_kind="full_sketch",
                    internal_label_rate=1.0,
                    surface_mode=str(control["surface_mode"]),
                    theorem_feature_dim=int(control["phi_dim"]),
                    theorem_feature_hidden_dim=int(control["phi_hidden_dim"]),
                    task_weight=float(control["task_weight"]),
                    phi_compose_weight=float(control["phi_compose_weight"]),
                    phi_contrastive_weight=float(control["phi_contrastive_weight"]),
                ),
            )
        )
    return configs


def _phase2_internal_full_dense_run(
    runs: Sequence[Mapping[str, Any]],
    *,
    config_prefix: str,
    train_doc_count: int,
) -> Mapping[str, Any] | None:
    target = f"{config_prefix}__{PHASE1_FULL_DENSE_LABEL}_{int(train_doc_count)}"
    for run in runs:
        if (
            str(run.get("tuning_stage", "")) == "phase2"
            and str(run.get("config_label", "")) == target
            and int(run.get("train_doc_count", 0)) == int(train_doc_count)
        ):
            return run
    return None


def _phase2_score(run: Mapping[str, Any]) -> float:
    root = lp._direct_metric(run, "root_direct_count_mae", default=1e9)
    leaf = lp._direct_metric(run, "leaf_direct_exact_match", default=0.0)
    merge = lp._direct_metric(run, "merge_direct_exact_match", default=0.0)
    probe_gap = lp._direct_metric(run, "phi_direct_probe_merge_gap", default=1e9)
    phi_alignment = lp._direct_metric(run, "phi_merge_alignment", default=0.0)
    return float(
        root
        + max(0.0, 0.88 - leaf) * 2.0
        + max(0.0, 0.75 - merge) * 3.0
        + max(0.0, probe_gap - 0.05) * 2.0
        + max(0.0, 0.90 - phi_alignment) * 1.5
    )


def _select_phase2_winner(runs: Sequence[Mapping[str, Any]]) -> str | None:
    train_doc_count = int(
        next(
            (
                run.get("train_doc_count")
                for run in runs
                if str(run.get("tuning_stage", "")) == "phase2"
            ),
            0,
        )
        or 0
    )
    if train_doc_count <= 0:
        return None
    slotwise_control = _phase2_internal_full_dense_run(
        runs,
        config_prefix="slotwise_control",
        train_doc_count=train_doc_count,
    )
    if slotwise_control is None:
        return None
    slotwise_leaf = lp._direct_metric(slotwise_control, "leaf_direct_exact_match", default=0.0)
    slotwise_merge = lp._direct_metric(slotwise_control, "merge_direct_exact_match", default=0.0)
    passing: list[dict[str, Any]] = []
    for variant in SHARED_FEATURE_VARIANTS:
        run = _phase2_internal_full_dense_run(
            runs,
            config_prefix=str(variant["label"]),
            train_doc_count=train_doc_count,
        )
        if run is None:
            continue
        root = lp._direct_metric(run, "root_direct_count_mae", default=1e9)
        leaf = lp._direct_metric(run, "leaf_direct_exact_match", default=0.0)
        merge = lp._direct_metric(run, "merge_direct_exact_match", default=0.0)
        if (
            root <= 0.45
            and leaf >= 0.88
            and merge >= 0.75
            and leaf >= (slotwise_leaf - 0.05)
            and merge >= (slotwise_merge - 0.05)
        ):
            passing.append(dict(run))
    if not passing:
        return None
    return str(min(passing, key=_phase2_score).get("config_label", "")).split("__", 1)[0]


def _phase3_configs(
    args: argparse.Namespace,
    *,
    winner_variant: str,
) -> list[tuple[int, mig._RunConfigSpec]]:
    train_doc_count = int(args.phase2_train_large)
    variant_map = {
        str(variant["label"]): dict(variant)
        for variant in [*SHARED_FEATURE_VARIANTS, *CONTROL_VARIANTS]
    }
    variant = variant_map[str(winner_variant)]
    configs: list[tuple[int, mig._RunConfigSpec]] = []
    for condition_label, leaf_rate, leaf_kind, internal_kind, internal_rate in PHASE3_CONDITIONS:
        configs.append(
            (
                train_doc_count,
                _make_surface_config(
                    args,
                    train_doc_count=train_doc_count,
                    label=f"{winner_variant}__{condition_label}_{train_doc_count}",
                    leaf_label_rate=float(leaf_rate),
                    leaf_supervision_kind=str(leaf_kind),
                    internal_supervision_kind=str(internal_kind),
                    internal_label_rate=float(internal_rate),
                    surface_mode=str(variant["surface_mode"]),
                    theorem_feature_dim=int(variant["phi_dim"]),
                    theorem_feature_hidden_dim=int(variant["phi_hidden_dim"]),
                    task_weight=float(variant["task_weight"]),
                    phi_compose_weight=float(variant["phi_compose_weight"]),
                    phi_contrastive_weight=float(variant["phi_contrastive_weight"]),
                    c2_mode=str(variant.get("c2_mode", "reconstruction")),
                    theorem_feature_adapter=str(variant.get("theorem_feature_adapter", "markov_count_sketch")),
                    oracle_metric_name=str(variant.get("oracle_metric_name", "")),
                ),
            )
        )
    return configs


def _phase4_configs(
    args: argparse.Namespace,
    *,
    winner_variant: str,
) -> list[tuple[int, mig._RunConfigSpec]]:
    train_doc_count = int(args.phase2_train_x5)
    variant_map = {
        str(variant["label"]): dict(variant)
        for variant in [*SHARED_FEATURE_VARIANTS, *CONTROL_VARIANTS]
    }
    variant = variant_map[str(winner_variant)]
    configs: list[tuple[int, mig._RunConfigSpec]] = []
    for condition_label, leaf_rate, leaf_kind, internal_kind, internal_rate in PHASE4_CONDITIONS:
        configs.append(
            (
                train_doc_count,
                _make_surface_config(
                    args,
                    train_doc_count=train_doc_count,
                    label=f"{winner_variant}__{condition_label}_{train_doc_count}",
                    leaf_label_rate=float(leaf_rate),
                    leaf_supervision_kind=str(leaf_kind),
                    internal_supervision_kind=str(internal_kind),
                    internal_label_rate=float(internal_rate),
                    surface_mode=str(variant["surface_mode"]),
                    theorem_feature_dim=int(variant["phi_dim"]),
                    theorem_feature_hidden_dim=int(variant["phi_hidden_dim"]),
                    task_weight=float(variant["task_weight"]),
                    phi_compose_weight=float(variant["phi_compose_weight"]),
                    phi_contrastive_weight=float(variant["phi_contrastive_weight"]),
                    c2_mode=str(variant.get("c2_mode", "reconstruction")),
                    theorem_feature_adapter=str(variant.get("theorem_feature_adapter", "markov_count_sketch")),
                    oracle_metric_name=str(variant.get("oracle_metric_name", "")),
                ),
            )
        )
    return configs


def _best_phase1_variant(runs: Sequence[Mapping[str, Any]]) -> str | None:
    candidates = [
        dict(run)
        for run in runs
        if str(run.get("tuning_stage", "")) == "phase1"
        and not _is_control_variant(str(run.get("config_label", "")))
    ]
    if not candidates:
        return None
    return str(min(candidates, key=_phase1_score).get("config_label", ""))


def _write_status(output_root: Path, payload: Mapping[str, Any]) -> None:
    (output_root / "shared_feature_push_status.json").write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    mig_uuids = _resolve_mig_uuids(args)

    phase1_jobs = _build_jobs_for_configs(
        args,
        _phase1_configs(args),
        seeds=(0,),
        tuning_stage="phase1",
        axis_value="phase1",
    )
    phase1_result = _run_jobs(
        output_root=output_root,
        jobs=phase1_jobs,
        torch_threads=int(args.torch_threads),
        use_cuda=bool(args.use_cuda),
        mig_uuids=mig_uuids,
        mode="shared_feature_phase1",
    )
    _write_outputs(output_root)
    all_runs = _collect_all_runs(output_root)
    promoted_variants = _select_phase1_promotions(all_runs)
    _write_status(
        output_root,
        {
            "output_root": str(output_root),
            "mig_uuids": list(mig_uuids),
            "phase1_completed_jobs": len(list(phase1_result.get("completed_jobs", ()))),
            "phase1_failed_jobs": len(list(phase1_result.get("failed_jobs", ()))),
            "phase1_promoted_variants": list(promoted_variants),
        },
    )

    winner_variant: str | None = None
    if promoted_variants:
        phase2_jobs = _build_jobs_for_configs(
            args,
            _phase2_configs(args, promoted_variants),
            seeds=(0,),
            tuning_stage="phase2",
            axis_value="phase2",
        )
        phase2_result = _run_jobs(
            output_root=output_root,
            jobs=phase2_jobs,
            torch_threads=int(args.torch_threads),
            use_cuda=bool(args.use_cuda),
            mig_uuids=mig_uuids,
            mode="shared_feature_phase2",
        )
        _write_outputs(output_root)
        all_runs = _collect_all_runs(output_root)
        winner_variant = _select_phase2_winner(all_runs)
        _write_status(
            output_root,
            {
                "output_root": str(output_root),
                "mig_uuids": list(mig_uuids),
                "phase1_promoted_variants": list(promoted_variants),
                "phase2_completed_jobs": len(
                    list(phase2_result.get("completed_jobs", ()))
                ),
                "phase2_failed_jobs": len(list(phase2_result.get("failed_jobs", ()))),
                "phase2_winner_variant": winner_variant or "",
            },
        )

    if winner_variant:
        phase3_jobs = _build_jobs_for_configs(
            args,
            _phase3_configs(args, winner_variant=winner_variant),
            seeds=tuple(int(seed) for seed in args.phase2_seeds),
            tuning_stage="phase3",
            axis_value="phase3",
        )
        phase3_result = _run_jobs(
            output_root=output_root,
            jobs=phase3_jobs,
            torch_threads=int(args.torch_threads),
            use_cuda=bool(args.use_cuda),
            mig_uuids=mig_uuids,
            mode="shared_feature_phase3",
        )
        _write_outputs(output_root)
        phase4_jobs = _build_jobs_for_configs(
            args,
            _phase4_configs(args, winner_variant=winner_variant),
            seeds=tuple(int(seed) for seed in args.phase4_seeds),
            tuning_stage="phase4",
            axis_value="phase4",
        )
        phase4_result = _run_jobs(
            output_root=output_root,
            jobs=phase4_jobs,
            torch_threads=int(args.torch_threads),
            use_cuda=bool(args.use_cuda),
            mig_uuids=mig_uuids,
            mode="shared_feature_phase4",
        )
        _write_outputs(output_root)
        _write_status(
            output_root,
            {
                "output_root": str(output_root),
                "mig_uuids": list(mig_uuids),
                "phase1_promoted_variants": list(promoted_variants),
                "phase2_winner_variant": winner_variant,
                "phase3_completed_jobs": len(
                    list(phase3_result.get("completed_jobs", ()))
                ),
                "phase3_failed_jobs": len(list(phase3_result.get("failed_jobs", ()))),
                "phase4_completed_jobs": len(
                    list(phase4_result.get("completed_jobs", ()))
                ),
                "phase4_failed_jobs": len(list(phase4_result.get("failed_jobs", ()))),
            },
        )
    else:
        fallback_variant = _best_phase1_variant(_collect_all_runs(output_root))
        if fallback_variant:
            fallback_configs = [
                config
                for config in _phase3_configs(args, winner_variant=fallback_variant)
                if config[1].label.endswith(f"{PHASE1_FULL_DENSE_LABEL}_{int(args.phase2_train_large)}")
            ]
            fallback_jobs = _build_jobs_for_configs(
                args,
                fallback_configs[:1],
                seeds=(0,),
                tuning_stage="phase3_fallback",
                axis_value="phase3_fallback",
            )
            _run_jobs(
                output_root=output_root,
                jobs=fallback_jobs,
                torch_threads=int(args.torch_threads),
                use_cuda=bool(args.use_cuda),
                mig_uuids=mig_uuids,
                mode="shared_feature_phase3_fallback",
            )
            _write_outputs(output_root)
            _write_status(
                output_root,
                {
                    "output_root": str(output_root),
                    "mig_uuids": list(mig_uuids),
                    "phase1_promoted_variants": list(promoted_variants),
                    "phase2_winner_variant": "",
                    "fallback_variant": str(fallback_variant),
                },
            )

    final_status = {
        "output_root": str(output_root),
        "mig_uuids": list(mig_uuids),
        "phase1_configs": len(_phase1_configs(args)),
        "phase1_promoted_variants": list(promoted_variants),
        "phase2_winner_variant": winner_variant or "",
    }
    print(json.dumps(final_status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
