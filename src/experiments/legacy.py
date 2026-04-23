from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

from src.experiments.contracts import (
    ArtifactRef,
    BenchmarkRef,
    ExperimentSpec,
    PhaseSpec,
    TaskSpec,
    benchmark_ref_from_parts,
    default_phase_specs,
    method_ref_from_parts,
)
from src.runtime import contracts as runtime_contracts
from src.ctreepo.sim import manifest as sim_manifest


def runtime_run_spec_to_experiment(
    spec: runtime_contracts.RunSpec,
    *,
    launch_command: Sequence[str] = (),
    adapter_id: str = "runtime_eval",
) -> ExperimentSpec:
    benchmark = dict(spec.benchmark or {})
    benchmark_ref = benchmark_ref_from_parts(
        family=str(benchmark.get("family", "runtime_benchmark") or "runtime_benchmark"),
        scope=str(benchmark.get("name", "") or ""),
        dataset_id=str(benchmark.get("dataset_id", "") or ""),
        name=str(benchmark.get("name", "") or ""),
        metadata=benchmark,
    )
    model = dict(spec.model or {})
    method_ref = method_ref_from_parts(
        family="runtime_eval",
        variant=str(model.get("family", "") or ""),
        engine=str(model.get("engine", "") or ""),
        model=str(model.get("model", "") or ""),
        adapter=adapter_id,
        metadata=model,
    )
    phases = tuple(
        PhaseSpec(
            phase_id=str(phase.phase_id),
            phase_role=str(phase.phase_id),
            metadata={
                "tasks": list(phase.tasks),
                "lengths": list(phase.lengths),
                "seeds": list(phase.seeds),
                "num_samples": int(phase.num_samples),
                "split": str(phase.split),
                "modes": list(phase.modes),
            },
        )
        for phase in list(spec.phases or ())
    )
    units = runtime_contracts.expand_units(spec)
    tasks = tuple(
        TaskSpec(
            task_id=str(unit.unit_id),
            phase_id=str(unit.phase_id),
            task_kind="runtime_unit",
            benchmark_ref=benchmark_ref,
            method_ref=method_ref,
            resources={},
            metadata=unit.to_dict(),
        )
        for unit in units
    )
    artifacts = (
        ArtifactRef(
            artifact_id="units_jsonl",
            artifact_type="runtime_units_manifest",
            path=str(Path(spec.output_dir) / spec.run_id / "units.jsonl"),
            required=True,
        ),
    )
    return ExperimentSpec.create(
        adapter_id=adapter_id,
        output_root=str(Path(spec.output_dir) / spec.run_id),
        title=f"runtime_eval::{spec.run_id}",
        benchmark_refs=(benchmark_ref,),
        method_refs=(method_ref,),
        phases=phases,
        tasks=tasks,
        artifacts=artifacts,
        report_profiles=("runtime_eval_summary",),
        launch_command=launch_command,
        resume_command=launch_command,
        metadata={"legacy_runtime_run_id": str(spec.run_id)},
    )


def ctreepo_runs_to_experiment(
    runs: Iterable[sim_manifest.RunSpec],
    *,
    output_root: str,
    title: str = "",
    adapter_id: str = "ctreepo_sim",
    launch_command: Sequence[str] = (),
) -> ExperimentSpec:
    run_list = list(runs)
    phases = default_phase_specs({str(run.family or "execute") for run in run_list})
    tasks: list[TaskSpec] = []
    benchmark_refs: dict[str, BenchmarkRef] = {}
    method_refs: dict[str, object] = {}
    artifacts: list[ArtifactRef] = []
    for run in run_list:
        family = str(run.family or "ctreepo_sim")
        benchmark_ref = benchmark_refs.setdefault(
            family,
            benchmark_ref_from_parts(
                family="ctreepo_sim",
                scope=family,
                name=family,
                metadata=dict(run.config or {}),
            ),
        )
        method_ref = method_refs.setdefault(
            family,
            method_ref_from_parts(
                family=family,
                variant=str(run.config.get("variant", "") or ""),
                adapter=adapter_id,
                metadata=dict(run.config or {}),
            ),
        )
        tasks.append(
            TaskSpec(
                task_id=str(run.id),
                phase_id=family,
                task_kind="legacy_runspec_command",
                command=(str(run.command),),
                deps=tuple(str(item) for item in list(run.requires or [])),
                benchmark_ref=benchmark_ref,
                method_ref=method_ref,  # type: ignore[arg-type]
                resources=dict(run.resources or {}),
                metadata={
                    "family": family,
                    "legacy_outputs": dict(run.outputs or {}),
                    "legacy_config": dict(run.config or {}),
                },
            )
        )
        for key, value in dict(run.outputs or {}).items():
            artifacts.append(
                ArtifactRef(
                    artifact_id=f"{run.id}:{key}",
                    artifact_type=str(key),
                    path=str(value),
                    phase_id=family,
                    required=False,
                    metadata={"run_id": str(run.id)},
                )
            )
    return ExperimentSpec.create(
        adapter_id=adapter_id,
        output_root=str(output_root),
        title=title or "ctreepo_sim",
        benchmark_refs=tuple(benchmark_refs.values()),
        method_refs=tuple(method_refs.values()),  # type: ignore[arg-type]
        phases=phases,
        tasks=tuple(tasks),
        artifacts=tuple(artifacts),
        launch_command=launch_command,
        resume_command=launch_command,
        metadata={"legacy_runspec_count": len(run_list)},
    )
