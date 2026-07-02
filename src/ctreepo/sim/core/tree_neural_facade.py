from __future__ import annotations

"""Shared facade helpers for tree-neural full-doc runners."""

import hashlib
import subprocess
from typing import Any, List, Mapping, Sequence

from src.ctreepo.sim.core.run_config import (
    JobSpec,
    RunConfigSpec,
    config_mapping_for_run_config,
    run_config_from_mapping as _canonical_run_config_from_mapping,
    with_run_intent_overrides,
    write_run_config_spec,
)


CLOSED_FORM_CONTROL_FAMILIES = frozenset({"tree_ridge_leaf", "tree_doc_ridge"})


def job_priority(job: JobSpec, *, family_order: Mapping[str, int]) -> tuple[int, int, int, str, int]:
    is_control = 1 if str(job.family) in CLOSED_FORM_CONTROL_FAMILIES else 0
    min_seed = int(min(job.seeds)) if job.seeds else 0
    return (
        is_control,
        -int(job.train_doc_count),
        family_order.get(str(job.family), 0),
        str(job.config.label),
        min_seed,
    )


def build_jobs_for_configs(
    *,
    families: Sequence[str],
    train_doc_counts: Sequence[int],
    benchmark: str,
    hardness_grid: str,
    grid_cell_ids: Sequence[str],
    seeds: Sequence[int],
    job_granularity: str,
    repeat_closed_form_controls: bool,
    configs: Sequence[RunConfigSpec],
    tuning_stage: str = "",
    test_metrics_hidden_during_selection: bool = False,
    study_name: str = "",
    study_axis: str = "",
    axis_value: str = "",
    locked_tree_neural_config_label: str = "",
    selection_metric: str = "",
    budget_total_calls: int = 0,
    budget_total_calls_per_doc: float = 0.0,
    mass_target_per_doc: float = float("nan"),
    full_doc_budget_share: float = 1.0,
    doc_consumption_mode: str = "",
    local_split_mode: str = "",
    local_allocation_policy: str = "",
    package_semantics: str = "",
    depth_discount_gamma: float = 1.0,
) -> List[JobSpec]:
    jobs: List[JobSpec] = []
    family_list = [str(family) for family in families]
    family_order = {
        str(family): idx for idx, family in enumerate(family_list)
    }
    seed_values = [int(seed) for seed in seeds]
    for config in configs:
        effective_config = with_run_intent_overrides(
            config,
            budget_total_calls=int(budget_total_calls),
            budget_total_calls_per_doc=float(budget_total_calls_per_doc),
            mass_target_per_doc=float(mass_target_per_doc),
            full_doc_budget_share=float(full_doc_budget_share),
            doc_consumption_mode=str(doc_consumption_mode),
            local_split_mode=str(local_split_mode),
            local_allocation_policy=str(local_allocation_policy),
            package_semantics=str(package_semantics),
            depth_discount_gamma=float(depth_discount_gamma),
        )
        for train_doc_count in [int(value) for value in train_doc_counts]:
            for family in family_list:
                if bool(repeat_closed_form_controls) or str(family) not in CLOSED_FORM_CONTROL_FAMILIES:
                    job_seeds = list(seed_values)
                else:
                    job_seeds = [int(seed_values[0])]
                if str(job_granularity) == "family_train_seed":
                    for seed in job_seeds:
                        jobs.append(
                            JobSpec(
                                family=str(family),
                                train_doc_count=int(train_doc_count),
                                benchmark=str(benchmark),
                                hardness_grid=str(hardness_grid),
                                grid_cell_ids=tuple(str(cell) for cell in grid_cell_ids),
                                seeds=(int(seed),),
                                config=effective_config,
                                tuning_stage=str(tuning_stage),
                                test_metrics_hidden_during_selection=bool(
                                    test_metrics_hidden_during_selection
                                ),
                                study_name=str(study_name),
                                study_axis=str(study_axis),
                                axis_value=str(axis_value),
                                locked_tree_neural_config_label=str(
                                    locked_tree_neural_config_label
                                ),
                                selection_metric=str(selection_metric),
                            )
                        )
                    continue
                jobs.append(
                    JobSpec(
                        family=str(family),
                        train_doc_count=int(train_doc_count),
                        benchmark=str(benchmark),
                        hardness_grid=str(hardness_grid),
                        grid_cell_ids=tuple(str(cell) for cell in grid_cell_ids),
                        seeds=tuple(int(seed) for seed in job_seeds),
                        config=effective_config,
                        tuning_stage=str(tuning_stage),
                        test_metrics_hidden_during_selection=bool(
                            test_metrics_hidden_during_selection
                        ),
                        study_name=str(study_name),
                        study_axis=str(study_axis),
                        axis_value=str(axis_value),
                        locked_tree_neural_config_label=str(
                            locked_tree_neural_config_label
                        ),
                        selection_metric=str(selection_metric),
                    )
                )
    return sorted(jobs, key=lambda job: job_priority(job, family_order=family_order))


def job_output_dir_name(
    job_name: str,
    *,
    max_component_length: int = 180,
) -> str:
    name = str(job_name).strip() or "job"
    if len(name) <= int(max_component_length):
        return name
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
    suffix = f"__h_{digest}"
    prefix_budget = max(1, int(max_component_length) - len(suffix))
    prefix = name[:prefix_budget].rstrip("_") or name[:prefix_budget]
    return f"{prefix}{suffix}"


def parse_mig_uuids(value: str) -> List[str]:
    return [
        token.strip()
        for token in str(value or "").replace(",", " ").split()
        if token.strip()
    ]


def discover_mig_uuids() -> List[str]:
    result = subprocess.run(
        ["nvidia-smi", "-L"],
        capture_output=True,
        text=True,
        check=True,
    )
    uuids: List[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if "MIG" not in line or "UUID:" not in line:
            continue
        uuid = line.split("UUID:", 1)[1].rstrip(")").strip()
        if uuid:
            uuids.append(uuid)
    return uuids


def discover_scheduler_devices(args: Any) -> List[str]:
    mig_uuids = (
        parse_mig_uuids(args.mig_uuids)
        if str(getattr(args, "mig_uuids", "")).strip()
        else discover_mig_uuids()
    )
    if not mig_uuids:
        raise RuntimeError(
            "No MIG UUIDs discovered. Pass --mig-uuids explicitly or configure MIGs first."
        )
    return list(mig_uuids)


def run_config_from_mapping(mapping: Mapping[str, Any]) -> RunConfigSpec:
    normalized = dict(mapping)
    if normalized.get("tree_local_law_weight") not in {"", None}:
        normalized.setdefault("local_law_weight", normalized.get("tree_local_law_weight"))
    for root_key in ("root_share", "tree_task_objective_weight", "task_objective_weight"):
        if normalized.get(root_key) not in {"", None}:
            normalized.setdefault("root_share", normalized.get(root_key))
            break
    normalized.pop("task_objective_weight", None)
    normalized.pop("tree_local_law_weight", None)
    normalized.pop("tree_task_objective_weight", None)
    return _canonical_run_config_from_mapping(normalized)


__all__ = [
    "CLOSED_FORM_CONTROL_FAMILIES",
    "JobSpec",
    "RunConfigSpec",
    "build_jobs_for_configs",
    "config_mapping_for_run_config",
    "discover_mig_uuids",
    "discover_scheduler_devices",
    "job_output_dir_name",
    "job_priority",
    "parse_mig_uuids",
    "run_config_from_mapping",
    "with_run_intent_overrides",
    "write_run_config_spec",
]
