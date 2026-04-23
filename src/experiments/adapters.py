from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.ctreepo.sim.manifest import read_manifest_jsonl
from src.experiments.contracts import (
    ExperimentSpec,
    benchmark_ref_from_parts,
    default_phase_specs,
    method_ref_from_parts,
)
from src.experiments.legacy import ctreepo_runs_to_experiment, runtime_run_spec_to_experiment
from src.experiments.markov_full_doc import method_ref_from_markov_full_doc_run
from src.experiments.registry import register_method_adapter
from src.runtime.contracts import RunPhaseSpec, RunSpec


def _strip_python(command: Sequence[str]) -> tuple[str, list[str]]:
    parts = [str(item) for item in list(command)]
    if not parts:
        raise ValueError("empty command")
    if parts[0].endswith("python") or parts[0].endswith("python3") or parts[0].endswith("pytest") or "python" in Path(parts[0]).name:
        if len(parts) < 2:
            raise ValueError("python command missing script path")
        return parts[1], parts[2:]
    return parts[0], parts[1:]


def _flag_value(args: Sequence[str], flag: str) -> str:
    items = [str(item) for item in list(args)]
    for idx, token in enumerate(items):
        if token == str(flag) and idx + 1 < len(items):
            return str(items[idx + 1]).strip()
        prefix = f"{flag}="
        if token.startswith(prefix):
            return str(token[len(prefix):]).strip()
    return ""


@register_method_adapter
class MarkovTreeAdapter:
    adapter_id = "markov_tree"
    aliases = ("markov", "tree_neural", "publication_bundle", "tradeoff_pipeline")

    def build_experiment_spec(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
    ) -> ExperimentSpec:
        script_name, argv = _strip_python(command)
        script_basename = Path(script_name).name
        output_root = _flag_value(argv, "--output-root")
        if not output_root:
            raise ValueError("markov_tree adapter requires --output-root")
        output_root_path = (cwd / output_root).resolve() if not Path(output_root).is_absolute() else Path(output_root).resolve()
        title = script_basename.replace(".py", "")
        phases = []
        packages: list[str] = []
        benchmarks: list[dict[str, Any]] = []
        if script_basename == "run_markov_optimization_tradeoff_pipeline.py":
            mod = importlib.import_module("scripts.run_markov_optimization_tradeoff_pipeline")
            args = mod._parse_args(argv)
            devices = mod._resolve_devices(args)
            plan = mod.build_run_plan(args, devices=devices)
            phase_counts = dict(plan.get("phase_task_counts", {}) or {})
            phases = [
                phase_name
                for phase_name in phase_counts.keys()
            ]
            recovery = dict(plan.get("resolved_selection", {}) or {})
            packages = [str(item) for item in list(recovery.get("supervision_recovery_packages") or ())]
            structural_cell = str(recovery.get("supervision_recovery_structural_cell", "") or "")
            benchmarks = [
                {"family": "markov_full_doc", "scope": "recoverable_v4", "name": "recoverable_v4"},
                {"family": "markov_full_doc", "scope": "structural_core_v1", "cell": structural_cell, "name": f"structural_core_v1::{structural_cell}" if structural_cell else "structural_core_v1"},
            ]
        elif script_basename == "run_markov_publication_bundle.py":
            mod = importlib.import_module("scripts.run_markov_publication_bundle")
            args = mod._parse_args(argv)
            migs = mod._resolved_mig_uuids(args)
            plan = mod.build_publication_run_plan(args, mig_uuids=migs, output_root=output_root_path)
            phases = list(plan.get("resolved_selection", {}).get("phases") or [])
            tradeoff = dict(plan.get("resolved_selection", {}).get("tradeoff") or {})
            structural_cell = str(tradeoff.get("supervision_recovery_structural_cell", "") or "")
            benchmarks = [
                {"family": "markov_full_doc", "scope": "recoverable_v4", "name": "recoverable_v4"},
                {"family": "markov_full_doc", "scope": "structural_core_v1", "cell": structural_cell, "name": f"structural_core_v1::{structural_cell}" if structural_cell else "structural_core_v1"},
            ]
        else:
            phases = ["screen", "locked", "report"]
        benchmark_refs = tuple(
            benchmark_ref_from_parts(
                family=str(item.get("family", "markov_full_doc")),
                scope=str(item.get("scope", "") or ""),
                cell=str(item.get("cell", "") or ""),
                name=str(item.get("name", "") or ""),
            )
            for item in benchmarks
        )
        method_refs = (
            method_ref_from_markov_full_doc_run(
                family="tree_neural",
                variant="family_default",
                adapter=self.adapter_id,
                metadata={"packages": packages},
            ),
            method_ref_from_markov_full_doc_run(
                family="official_fno",
                variant="family_default",
                adapter=self.adapter_id,
                metadata={"packages": packages},
            ),
        )
        return ExperimentSpec.create(
            adapter_id=self.adapter_id,
            output_root=str(output_root_path),
            title=title,
            benchmark_refs=benchmark_refs,
            method_refs=method_refs,
            phases=default_phase_specs(phases),
            report_profiles=("tradeoff", "publication_bundle", "supervision_recovery"),
            launch_command=command,
            resume_command=command,
            metadata={"legacy_script": script_basename},
        )

    def collect_artifacts(self, output_root: Path) -> Mapping[str, Any]:
        candidates = {
            "pipeline_summary_json": output_root / "pipeline_summary.json",
            "tradeoff_report_summary_json": output_root / "tradeoff_report" / "summary.json",
            "tradeoff_report_pdf": output_root / "tradeoff_report" / "report.pdf",
            "supervision_recovery_summary_json": output_root / "supervision_recovery" / "summary.json",
        }
        return {key: str(path) for key, path in candidates.items() if path.exists()}


@register_method_adapter
class RuntimeEvalAdapter:
    adapter_id = "runtime_eval"
    aliases = ("runtime", "runtime_evaluation")

    def build_experiment_spec(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
    ) -> ExperimentSpec:
        script_name, argv = _strip_python(command)
        if Path(script_name).name != "run_runtime_eval.py":
            raise ValueError("runtime_eval adapter expects run_runtime_eval.py")
        if not argv or argv[0] != "init":
            raise ValueError("runtime_eval adapter expects the init subcommand")
        config_path = _flag_value(argv[1:], "--config")
        output_dir = _flag_value(argv[1:], "--output-dir") or "outputs/evals"
        run_id = _flag_value(argv[1:], "--run-id") or ""
        mod = importlib.import_module("scripts.run_runtime_eval")
        spec = mod._load_run_spec(
            Path(config_path).resolve(),
            output_dir=(cwd / output_dir).resolve() if not Path(output_dir).is_absolute() else Path(output_dir).resolve(),
            run_id=(run_id or None),
        )
        return runtime_run_spec_to_experiment(
            spec,
            launch_command=command,
            adapter_id=self.adapter_id,
        )

    def collect_artifacts(self, output_root: Path) -> Mapping[str, Any]:
        candidates = {
            "metrics_json": output_root / "metrics.json",
            "merged_steps_jsonl": output_root / "steps.jsonl",
            "merged_predictions_jsonl": output_root / "predictions.jsonl",
            "units_jsonl": output_root / "units.jsonl",
        }
        return {key: str(path) for key, path in candidates.items() if path.exists()}


@register_method_adapter
class CTreePOSimAdapter:
    adapter_id = "ctreepo_sim"
    aliases = ("ctreepo", "suite_manifest")

    def build_experiment_spec(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
    ) -> ExperimentSpec:
        _script_name, argv = _strip_python(command)
        manifest_path = _flag_value(argv, "--manifest") or _flag_value(argv, "--runspec-manifest")
        if not manifest_path:
            raise ValueError("ctreepo_sim adapter requires --manifest or --runspec-manifest")
        manifest = Path(manifest_path).expanduser()
        if not manifest.is_absolute():
            manifest = (cwd / manifest).resolve()
        runs = read_manifest_jsonl(manifest)
        return ctreepo_runs_to_experiment(
            runs,
            output_root=str(manifest.parent),
            title=manifest.stem,
            adapter_id=self.adapter_id,
            launch_command=command,
        )

    def collect_artifacts(self, output_root: Path) -> Mapping[str, Any]:
        return {}


@register_method_adapter
class TreePOTrainingAdapter:
    adapter_id = "treepo_training"
    aliases = ("treepo", "training_pipeline", "train_neural_operators", "train_ctreepo")

    def build_experiment_spec(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
    ) -> ExperimentSpec:
        script_name, argv = _strip_python(command)
        script_basename = Path(script_name).name
        output_root = _flag_value(argv, "--output-dir")
        if not output_root:
            raise ValueError("treepo_training adapter requires --output-dir")
        output_root_path = (cwd / output_root).resolve() if not Path(output_root).is_absolute() else Path(output_root).resolve()
        task_name = _flag_value(argv, "--task") or "manifesto_rile"
        benchmark_ref = benchmark_ref_from_parts(
            family="treepo_task",
            scope=str(task_name),
            name=str(task_name),
        )
        method_refs = []
        if script_basename in {"train_neural_operators.py"}:
            method_refs.extend(
                [
                    method_ref_from_parts(
                        family="ctreepo",
                        variant="local_law_training",
                        adapter=self.adapter_id,
                    ),
                    method_ref_from_parts(
                        family="mergeable_sketch",
                        variant="embedding_sketch_training",
                        adapter=self.adapter_id,
                    ),
                ]
            )
        elif script_basename in {"train_ctreepo.py"}:
            method_refs.append(
                method_ref_from_parts(
                    family="ctreepo",
                    variant="local_law_training",
                    adapter=self.adapter_id,
                )
            )
        else:
            method_refs.extend(
                [
                    method_ref_from_parts(
                        family="llm_prompt_optimization",
                        variant="training_pipeline",
                        adapter=self.adapter_id,
                    ),
                    method_ref_from_parts(
                        family="embedding_proxy",
                        variant="training_pipeline",
                        adapter=self.adapter_id,
                    ),
                    method_ref_from_parts(
                        family="ctreepo",
                        variant="training_pipeline",
                        adapter=self.adapter_id,
                    ),
                    method_ref_from_parts(
                        family="mergeable_sketch",
                        variant="training_pipeline",
                        adapter=self.adapter_id,
                    ),
                    method_ref_from_parts(
                        family="generator_finetune",
                        variant="training_pipeline",
                        adapter=self.adapter_id,
                    ),
                ]
            )
        return ExperimentSpec.create(
            adapter_id=self.adapter_id,
            output_root=str(output_root_path),
            title=script_basename.replace(".py", ""),
            benchmark_refs=(benchmark_ref,),
            method_refs=tuple(method_refs),
            phases=default_phase_specs(("train", "eval", "aggregate", "report")),
            report_profiles=("runtime_eval_summary",),
            launch_command=command,
            resume_command=command,
            metadata={"legacy_script": script_basename, "task": task_name},
        )

    def collect_artifacts(self, output_root: Path) -> Mapping[str, Any]:
        candidates = {
            "summary_json": output_root / "summary.json",
            "final_stats_json": output_root / "final_stats.json",
            "score_report_pdf": output_root / "score_report.pdf",
            "optimizer_audit_manifest_json": output_root / "optimizer_audit_manifest.json",
            "ctreepo_training_result_json": output_root / "ctreepo" / "training_result.json",
            "ctreepo_best_model": output_root / "ctreepo" / "best.pt",
            "mergeable_metrics_json": output_root / "mergeable_sketch" / "metrics.json",
        }
        return {key: str(path) for key, path in candidates.items() if path.exists()}


@register_method_adapter
class ReportOnlyAdapter:
    adapter_id = "report_only"
    aliases = ("report",)

    def build_experiment_spec(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
    ) -> ExperimentSpec:
        output_root = _flag_value(command, "--output-root")
        if not output_root:
            raise ValueError("report_only adapter requires --output-root")
        output_root_path = (cwd / output_root).resolve() if not Path(output_root).is_absolute() else Path(output_root).resolve()
        return ExperimentSpec.create(
            adapter_id=self.adapter_id,
            output_root=str(output_root_path),
            title="report_only",
            phases=default_phase_specs(("report",)),
            report_profiles=("tradeoff", "publication_bundle", "runtime_eval_summary", "supervision_recovery"),
            launch_command=command,
            resume_command=command,
        )

    def collect_artifacts(self, output_root: Path) -> Mapping[str, Any]:
        return {}
