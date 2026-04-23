#!/usr/bin/env python3
"""Fixed-bundle memory bisect for stage-1 tree-neural training.

This script treats the saved `recoverable_v4` bundle as the canonical repro
target and compares four cases on the same fixed train/val/test split:

1. old stable slotwise control;
2. current shared-feature-adapters with legacy exact checkpoint selection;
3. current shared-feature-adapters with cheap stage-1 selection only; and
4. current shared-feature-adapters with cheap selection plus the streaming exact
   evaluator.

Each case runs in its own subprocess so host-RAM measurements are not polluted
by the previous run's allocator state.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import os
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_tree_neural_full_doc_mig as mig  # noqa: E402
from scripts import run_tree_neural_teacher_first_push as tfpush  # noqa: E402
from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (  # noqa: E402
    _base_config_for_benchmark,
    _bundle_with_fixed_eval_splits,
    _doc_root_targets,
    _materialize_base_bundle,
    resolve_full_doc_diagnostic_benchmark,
)
from src.ctreepo.sim.core.markov_changepoint_ops_count import (  # noqa: E402
    _build_objective_summary,
    OPSCountConfig,
)
from src.ctreepo.sim.core.markov_neural_operator_baselines import (  # noqa: E402
    _class_setup,
    _eval_fno_exact_sketch_direct_metrics,
    _eval_fno_exact_sketch_direct_metrics_legacy,
    _prepare_fno_count_docs,
    FNOCountSketch,
    train_fno_tree,
)


@dataclass(frozen=True)
class _HarnessCase:
    name: str
    description: str
    kind: str
    stage1_checkpoint_metric: str
    exact_evaluator_mode: str


@dataclass
class _MemoryTimelineRecorder:
    device: torch.device
    timeline: List[Dict[str, Any]]

    def record(self, event: str, payload: Mapping[str, Any]) -> None:
        entry: Dict[str, Any] = {
            "event": str(event),
            "timestamp_s": float(time.time()),
            "rss_bytes": int(_rss_bytes()),
        }
        for key, value in dict(payload).items():
            entry[str(key)] = value
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                entry["cuda_memory_allocated_bytes"] = int(
                    torch.cuda.memory_allocated(self.device)
                )
                entry["cuda_memory_reserved_bytes"] = int(
                    torch.cuda.memory_reserved(self.device)
                )
                entry["cuda_max_memory_allocated_bytes"] = int(
                    torch.cuda.max_memory_allocated(self.device)
                )
            except Exception:
                pass
        self.timeline.append(entry)


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _rss_bytes() -> int:
    return _pid_rss_bytes(None)


def _pid_rss_bytes(pid: int | None) -> int:
    status_path = (
        Path("/proc/self/status")
        if pid is None
        else Path("/proc") / str(int(pid)) / "status"
    )
    if status_path.exists():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    statm_path = (
        Path("/proc/self/statm")
        if pid is None
        else Path("/proc") / str(int(pid)) / "statm"
    )
    if statm_path.exists():
        fields = statm_path.read_text(encoding="utf-8").split()
        if len(fields) >= 2:
            return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))  # type: ignore[name-defined]
    return 0


def _case_specs() -> tuple[_HarnessCase, ...]:
    return (
        _HarnessCase(
            name="slotwise_control_legacy_exact",
            description="Old stable slotwise exact-sanity control with the legacy evaluator.",
            kind="control",
            stage1_checkpoint_metric="val_exact_sketch_direct",
            exact_evaluator_mode="legacy",
        ),
        _HarnessCase(
            name="shared_feature_adapters_exact_selection_legacy",
            description="Current shared_feature_adapters stage-1 with legacy exact checkpoint selection.",
            kind="shared_feature_adapters",
            stage1_checkpoint_metric="val_theorem_bootstrap_direct",
            exact_evaluator_mode="legacy",
        ),
        _HarnessCase(
            name="shared_feature_adapters_cheap_selection_legacy",
            description="Cheap stage-1 root-MAE selection, but legacy exact evaluator on the restored best checkpoint.",
            kind="shared_feature_adapters",
            stage1_checkpoint_metric="val_root_mae",
            exact_evaluator_mode="legacy",
        ),
        _HarnessCase(
            name="shared_feature_adapters_cheap_selection_streaming",
            description="Cheap stage-1 root-MAE selection plus the streaming exact evaluator.",
            kind="shared_feature_adapters",
            stage1_checkpoint_metric="val_root_mae",
            exact_evaluator_mode="streaming",
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode")

    def _add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--output-root",
            type=str,
            default=f"outputs/tree_neural_fixed_bundle_memory_bisect_{_timestamp()}",
        )
        subparser.add_argument("--benchmark", type=str, default="recoverable_v4")
        subparser.add_argument("--train-docs", type=int, default=128)
        subparser.add_argument("--seed", type=int, default=0)
        subparser.add_argument("--state-dim", type=int, default=128)
        subparser.add_argument("--hidden-dim", type=int, default=512)
        subparser.add_argument("--batch-size", type=int, default=64)
        subparser.add_argument("--lr", type=float, default=5e-4)
        subparser.add_argument("--weight-decay", type=float, default=0.0)
        subparser.add_argument("--tree-local-law-weight", type=float, default=0.8)
        subparser.add_argument("--tree-join-bit-weight", type=float, default=1.0)
        subparser.add_argument("--stage1-epochs", type=int, default=2)
        subparser.add_argument("--tree-theorem-count-dim", type=int, default=8)
        subparser.add_argument("--tree-theorem-first-dim", type=int, default=8)
        subparser.add_argument("--tree-theorem-last-dim", type=int, default=8)
        subparser.add_argument("--exact-selection-doc-limit", type=int, default=0)
        subparser.add_argument("--final-exact-doc-limit", type=int, default=0)
        subparser.add_argument("--phi-pair-calibration-max-nodes", type=int, default=512)
        subparser.add_argument(
            "--use-cuda",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        subparser.add_argument("--cuda-device", type=int, default=None)
        subparser.add_argument("--torch-threads", type=int, default=1)

    run_parser = subparsers.add_parser("run")
    _add_common(run_parser)
    worker_parser = subparsers.add_parser("case_worker")
    _add_common(worker_parser)
    worker_parser.add_argument("--case-name", type=str, required=True)
    return parser


def _common_runner_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        benchmark=str(args.benchmark),
        phase1_train_docs=int(args.train_docs),
        phase2_train_docs=int(args.train_docs),
        phase1_seeds=(int(args.seed),),
        phase2_seeds=(int(args.seed),),
        surrogate_labels=(),
        state_dim=int(args.state_dim),
        hidden_dim=int(args.hidden_dim),
        n_epochs=int(args.stage1_epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        tree_local_law_weight=float(args.tree_local_law_weight),
        tree_task_objective_weight=None,
        tree_join_bit_weight=float(args.tree_join_bit_weight),
        stage1_epochs=int(args.stage1_epochs),
        stage2_epochs=0,
        tree_theorem_count_dim=int(args.tree_theorem_count_dim),
        tree_theorem_first_dim=int(args.tree_theorem_first_dim),
        tree_theorem_last_dim=int(args.tree_theorem_last_dim),
        root_search_labels=tuple(),
        stage1_root_weight_grid=tuple(),
        use_cuda=bool(args.use_cuda),
        torch_threads=int(args.torch_threads),
    )


def _build_case_config(args: argparse.Namespace, case: _HarnessCase) -> mig._RunConfigSpec:
    common_args = _common_runner_args(args)
    if case.kind == "control":
        base = mig._slot_exact_sanity_config(
            common_args,
            train_doc_count=int(args.train_docs),
            config_label=str(case.name),
            leaf_label_rate=1.0,
            leaf_supervision_kind="full_sketch",
            internal_supervision_kind="full_sketch",
            internal_label_rate=1.0,
            tree_summary_spec_root_mode="factored_theorem_readout",
        )
        return replace(
            base,
            label=str(case.name),
            n_epochs=int(args.stage1_epochs),
            tree_training_schedule="single_stage",
            tree_stage1_epochs=0,
            tree_stage2_epochs=0,
            tree_checkpoint_metric="val_exact_sketch_direct",
            tree_stage1_checkpoint_metric=str(case.stage1_checkpoint_metric),
            tree_theorem_surface_mode="slotwise",
            tree_task_head_mode="theorem_feature_scalar",
            tree_summary_spec_root_mode="factored_theorem_readout",
        )
    variant = next(
        item
        for item in tfpush.SURROGATE_VARIANTS
        if str(item["label"]) == "teacherfirst_shared_feature_adapters_phi128"
    )
    variant_payload = dict(variant)
    variant_payload["label"] = str(case.name)
    variant_payload["stage1_checkpoint_metric"] = str(case.stage1_checkpoint_metric)
    return tfpush._make_stage1_config(
        common_args,
        train_doc_count=int(args.train_docs),
        variant=variant_payload,
    )


def _resolve_benchmark_bundle(
    *,
    args: argparse.Namespace,
) -> tuple[OPSCountConfig, Any, str]:
    benchmark = resolve_full_doc_diagnostic_benchmark(str(args.benchmark))
    base_config = _base_config_for_benchmark(
        benchmark=benchmark,
        train_docs=int(args.train_docs),
        use_cuda=bool(args.use_cuda),
        cuda_device=args.cuda_device,
        torch_threads=int(args.torch_threads),
        seed=int(args.seed),
    )
    base_bundle, base_source = _materialize_base_bundle(
        benchmark=benchmark,
        required_train_docs=int(args.train_docs),
        output_dir=Path(str(args.output_root)),
    )
    bundle, bundle_source = _bundle_with_fixed_eval_splits(
        base_bundle=base_bundle,
        base_source=base_source,
        train_doc_count=int(args.train_docs),
    )
    return base_config, bundle, bundle_source


def _merged_training_config(
    *,
    args: argparse.Namespace,
    case_config: mig._RunConfigSpec,
) -> OPSCountConfig:
    base_config, _bundle, _bundle_source = _resolve_benchmark_bundle(args=args)
    merged = {**asdict(base_config), **asdict(case_config)}
    merged["use_cuda"] = bool(args.use_cuda)
    merged["cuda_device"] = args.cuda_device
    merged["torch_threads"] = int(args.torch_threads)
    merged["seed"] = int(args.seed)
    return OPSCountConfig(**merged)


def _build_model_and_docs(
    *,
    config: OPSCountConfig,
    bundle: Any,
    device: torch.device,
) -> tuple[FNOCountSketch, Sequence[Any], Sequence[Any], Sequence[Any], Mapping[int, int]]:
    train_docs = tuple(bundle.train_docs)
    val_docs = tuple(bundle.val_docs)
    test_docs = tuple(bundle.test_docs)
    leaf_tokens = int(config.fixed_leaf_tokens)
    train_fno_docs = _prepare_fno_count_docs(train_docs, leaf_tokens=leaf_tokens)
    val_fno_docs = _prepare_fno_count_docs(val_docs, leaf_tokens=leaf_tokens)
    test_fno_docs = _prepare_fno_count_docs(test_docs, leaf_tokens=leaf_tokens)
    train_y = _doc_root_targets(train_docs)
    val_y = _doc_root_targets(val_docs)
    test_y = _doc_root_targets(test_docs)
    class_target_max, root_class_values, root_class_index, _ = _class_setup(
        train_y,
        val_y,
        test_y,
    )
    target_scale = (
        float(class_target_max)
        if str(config.tree_root_supervision_kind) == "count_ce"
        else float(max(1.0, np.max(train_y)))
    )
    tree_leaf_fno_width = (
        int(config.tree_leaf_fno_width)
        if config.tree_leaf_fno_width is not None
        else int(config.fno_width)
    )
    tree_leaf_fno_n_modes = (
        int(config.tree_leaf_fno_n_modes)
        if config.tree_leaf_fno_n_modes is not None
        else int(config.fno_n_modes)
    )
    tree_leaf_fno_n_layers = (
        int(config.tree_leaf_fno_n_layers)
        if config.tree_leaf_fno_n_layers is not None
        else int(config.fno_n_layers)
    )
    model = FNOCountSketch(
        vocab_size=int(config.vocab_size),
        leaf_tokens=int(config.fixed_leaf_tokens),
        state_dim=int(config.state_dim),
        hidden_dim=int(config.hidden_dim),
        target_scale=float(target_scale),
        n_regimes=int(config.n_regimes),
        doc_sequence_class_values=root_class_values,
        fno_width=int(tree_leaf_fno_width),
        fno_n_modes=int(tree_leaf_fno_n_modes),
        fno_n_layers=int(tree_leaf_fno_n_layers),
        root_supervision_kind=str(config.tree_root_supervision_kind),
        root_count_class_values=root_class_values,
        endpoint_loss_scale=float(config.endpoint_loss_scale),
        aligned_sketch_surface=str(config.aligned_sketch_surface),
        summary_spec_name=str(config.summary_spec_name),
        slot_count=int(config.slot_count),
        join_bit_weight=float(config.tree_join_bit_weight),
        task_head_mode=str(config.tree_task_head_mode),
        theorem_surface_mode=str(config.tree_theorem_surface_mode),
        theorem_count_head_mode=str(config.tree_theorem_count_head_mode),
        theorem_count_ordinal_weight=float(config.tree_theorem_count_ordinal_weight),
        theorem_count_scalar_aux_weight=float(
            config.tree_theorem_count_scalar_aux_weight
        ),
        theorem_count_threshold_balance=bool(
            config.tree_theorem_count_threshold_balance
        ),
        theorem_feature_dim=int(config.tree_theorem_feature_dim),
        theorem_feature_hidden_dim=int(config.tree_theorem_feature_hidden_dim),
        phi_alignment_loss=str(config.tree_phi_alignment_loss),
        c2_mode=str(config.tree_c2_mode),
        theorem_feature_adapter=str(config.theorem_feature_adapter),
        theorem_pair_same_threshold=config.theorem_pair_same_threshold,
        theorem_pair_diff_threshold=config.theorem_pair_diff_threshold,
        summary_spec_root_mode=str(config.tree_summary_spec_root_mode),
        theorem_count_dim=int(config.tree_theorem_count_dim),
        theorem_first_dim=int(config.tree_theorem_first_dim),
        theorem_last_dim=int(config.tree_theorem_last_dim),
    ).to(device=device)
    return model, train_fno_docs, val_fno_docs, test_fno_docs, root_class_index


def _resolve_exact_evaluator(case: _HarnessCase):
    if str(case.exact_evaluator_mode) == "legacy":
        return _eval_fno_exact_sketch_direct_metrics_legacy
    return _eval_fno_exact_sketch_direct_metrics


def _run_case_worker(args: argparse.Namespace) -> int:
    case = next(spec for spec in _case_specs() if spec.name == str(args.case_name))
    case_dir = Path(str(args.output_root)) / str(case.name)
    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result_payload: Dict[str, Any] = {
        "case_name": str(case.name),
        "description": str(case.description),
        "status": "running",
    }
    timeline: List[Dict[str, Any]] = []
    try:
        if int(args.torch_threads) > 0:
            torch.set_num_threads(int(args.torch_threads))
        if bool(args.use_cuda) and torch.cuda.is_available():
            device = (
                torch.device(f"cuda:{int(args.cuda_device)}")
                if args.cuda_device is not None
                else torch.device("cuda")
            )
            if args.cuda_device is not None:
                torch.cuda.set_device(int(args.cuda_device))
            torch.cuda.reset_peak_memory_stats(device)
        else:
            device = torch.device("cpu")
        recorder = _MemoryTimelineRecorder(device=device, timeline=timeline)
        case_config = _build_case_config(args, case)
        base_config, bundle, bundle_source = _resolve_benchmark_bundle(args=args)
        case_overrides = {
            key: value
            for key, value in asdict(case_config).items()
            if value is not None
        }
        merged_config = {
            **asdict(base_config),
            **case_overrides,
            "use_cuda": bool(args.use_cuda),
            "cuda_device": args.cuda_device,
            "torch_threads": int(args.torch_threads),
            "seed": int(args.seed),
        }
        allowed_fields = set(getattr(OPSCountConfig, "__annotations__", {}).keys())
        config = OPSCountConfig(
            **{
                key: value
                for key, value in merged_config.items()
                if str(key) in allowed_fields
            }
        )
        model, train_fno_docs, val_fno_docs, test_fno_docs, root_class_index = _build_model_and_docs(
            config=config,
            bundle=bundle,
            device=device,
        )
        objective = _build_objective_summary(config)
        recorder.record(
            "worker_setup_complete",
            {
                "bundle_source": str(bundle_source),
                "n_train_docs": int(len(train_fno_docs)),
                "n_val_docs": int(len(val_fno_docs)),
                "n_test_docs": int(len(test_fno_docs)),
            },
        )
        train_result = train_fno_tree(
            model=model,
            train_docs=train_fno_docs,
            val_docs=val_fno_docs,
            device=device,
            n_epochs=int(config.n_epochs),
            batch_size=int(config.batch_size),
            lr=float(config.lr),
            weight_decay=float(config.weight_decay),
            root_weight=float(objective["optimization_root_weight"]),
            c1_weight=float(objective["local_law_c1_weight"]),
            c2_weight=float(objective["local_law_c2_weight"]),
            c3_weight=float(objective["local_law_c3_weight"]),
            root_class_index=root_class_index,
            doc_sequence_class_index=root_class_index,
            internal_supervision_kind=str(config.internal_supervision_kind),
            internal_label_rate=float(config.internal_label_rate),
            leaf_exact_supervision=bool(config.leaf_exact_supervision),
            leaf_supervision_kind=str(config.leaf_supervision_kind),
            leaf_label_rate=float(config.leaf_label_rate),
            phi_compose_weight=float(config.tree_phi_compose_weight),
            phi_contrastive_weight=float(config.tree_phi_contrastive_weight),
            checkpoint_metric=str(config.tree_checkpoint_metric),
            tree_training_schedule=str(config.tree_training_schedule),
            tree_stage1_epochs=int(config.tree_stage1_epochs),
            tree_stage2_epochs=int(config.tree_stage2_epochs),
            tree_stage1_checkpoint_metric=str(config.tree_stage1_checkpoint_metric),
            tree_stage1_artifact_dir=str(config.tree_stage1_artifact_dir),
            tree_stage1_root_weight=float(config.tree_stage1_root_weight),
            exact_metric_evaluator=_resolve_exact_evaluator(case),
            exact_metric_selection_doc_limit=int(args.exact_selection_doc_limit),
            exact_metric_selection_interval=1,
            exact_metric_phi_pair_calibration_max_nodes=int(
                args.phi_pair_calibration_max_nodes
            ),
            exact_metric_final_doc_limit=int(args.final_exact_doc_limit),
            memory_probe=recorder.record,
            seed=int(args.seed),
        )
        selection_metric_name = str(train_result.get("selection_metric_name", ""))
        best_val_mae = float(train_result.get("best_val_mae", float("nan")))
        if (
            str(config.tree_training_schedule) == "two_stage"
            and int(config.tree_stage2_epochs) <= 0
        ):
            stage1_summary = dict(train_result.get("stage1_result_summary", {}) or {})
            if str(stage1_summary.get("selection_metric_name", "")).strip():
                selection_metric_name = str(stage1_summary.get("selection_metric_name", ""))
            stage1_best_metric = stage1_summary.get("best_metric_value", float("nan"))
            if isinstance(stage1_best_metric, (int, float)):
                best_val_mae = float(stage1_best_metric)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        peak_rss_bytes = max(
            [int(entry.get("rss_bytes", 0)) for entry in timeline] or [0]
        )
        result_payload.update(
            {
                "status": "ok",
                "bundle_source": str(bundle_source),
                "config_label": str(case_config.label),
                "tree_stage1_checkpoint_metric": str(
                    case_config.tree_stage1_checkpoint_metric
                ),
                "tree_training_schedule": str(case_config.tree_training_schedule),
                "exact_evaluator_mode": str(case.exact_evaluator_mode),
                "peak_rss_bytes": int(peak_rss_bytes),
                "peak_rss_gib": float(peak_rss_bytes / float(1024**3)),
                "elapsed_s": float(time.time() - started),
                "selection_metric_name": str(selection_metric_name),
                "best_val_mae": float(best_val_mae),
                "best_exact_metrics_split": str(
                    train_result.get("best_exact_metrics_split", "")
                ),
                "best_exact_metrics": dict(train_result.get("best_exact_metrics", {}) or {}),
            }
        )
    except Exception as exc:
        result_payload.update(
            {
                "status": "error",
                "elapsed_s": float(time.time() - started),
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "peak_rss_bytes": int(
                    max([int(entry.get("rss_bytes", 0)) for entry in timeline] or [0])
                ),
            }
        )
    timeline_path = case_dir / "memory_timeline.json"
    result_path = case_dir / "result.json"
    timeline_path.write_text(json.dumps(timeline, indent=2), encoding="utf-8")
    result_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    return 0 if result_payload.get("status") == "ok" else 1


def _worker_command(args: argparse.Namespace, case: _HarnessCase) -> List[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "case_worker",
        "--output-root",
        str(args.output_root),
        "--benchmark",
        str(args.benchmark),
        "--train-docs",
        str(int(args.train_docs)),
        "--seed",
        str(int(args.seed)),
        "--state-dim",
        str(int(args.state_dim)),
        "--hidden-dim",
        str(int(args.hidden_dim)),
        "--batch-size",
        str(int(args.batch_size)),
        "--lr",
        str(float(args.lr)),
        "--weight-decay",
        str(float(args.weight_decay)),
        "--tree-local-law-weight",
        str(float(args.tree_local_law_weight)),
        "--tree-join-bit-weight",
        str(float(args.tree_join_bit_weight)),
        "--stage1-epochs",
        str(int(args.stage1_epochs)),
        "--tree-theorem-count-dim",
        str(int(args.tree_theorem_count_dim)),
        "--tree-theorem-first-dim",
        str(int(args.tree_theorem_first_dim)),
        "--tree-theorem-last-dim",
        str(int(args.tree_theorem_last_dim)),
        "--exact-selection-doc-limit",
        str(int(args.exact_selection_doc_limit)),
        "--final-exact-doc-limit",
        str(int(args.final_exact_doc_limit)),
        "--phi-pair-calibration-max-nodes",
        str(int(args.phi_pair_calibration_max_nodes)),
        "--torch-threads",
        str(int(args.torch_threads)),
        "--case-name",
        str(case.name),
    ]
    if bool(args.use_cuda):
        cmd.append("--use-cuda")
    else:
        cmd.append("--no-use-cuda")
    if args.cuda_device is not None:
        cmd.extend(["--cuda-device", str(int(args.cuda_device))])
    return cmd


def _render_summary_markdown(summary: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Fixed-Bundle Memory Bisect",
        "",
        "| Case | Status | Peak RSS (GiB) | Selection Metric | Exact Evaluator |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in summary:
        peak = row.get("peak_rss_gib", float("nan"))
        peak_text = (
            f"{float(peak):.2f}"
            if isinstance(peak, (int, float)) and math.isfinite(float(peak))
            else "nan"
        )
        lines.append(
            "| {case} | {status} | {peak} | {metric} | {evaluator} |".format(
                case=str(row.get("case_name", "")),
                status=str(row.get("status", "")),
                peak=str(peak_text),
                metric=str(row.get("selection_metric_name", "")),
                evaluator=str(row.get("exact_evaluator_mode", "")),
            )
        )
    return "\n".join(lines) + "\n"


def _run(args: argparse.Namespace) -> int:
    output_root = Path(str(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, Any]] = []
    for case in _case_specs():
        proc = subprocess.Popen(
            _worker_command(args, case),
            cwd=str(REPO_ROOT),
        )
        observed_peak_rss_bytes = 0
        while True:
            observed_peak_rss_bytes = max(
                int(observed_peak_rss_bytes),
                int(_pid_rss_bytes(proc.pid)),
            )
            returncode = proc.poll()
            if returncode is not None:
                break
            time.sleep(1.0)
        case_dir = output_root / str(case.name)
        result_path = case_dir / "result.json"
        row: Dict[str, Any] = {
            "case_name": str(case.name),
            "description": str(case.description),
            "returncode": int(returncode),
            "exact_evaluator_mode": str(case.exact_evaluator_mode),
            "observed_peak_rss_bytes": int(observed_peak_rss_bytes),
            "observed_peak_rss_gib": float(observed_peak_rss_bytes / float(1024**3)),
        }
        if result_path.exists():
            row.update(json.loads(result_path.read_text(encoding="utf-8")))
        else:
            row["status"] = "missing_result"
        summary_rows.append(row)
    summary_json = output_root / "memory_bisect_summary.json"
    summary_md = output_root / "memory_bisect_summary.md"
    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    summary_md.write_text(_render_summary_markdown(summary_rows), encoding="utf-8")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    if not argv or str(argv[0]) not in {"run", "case_worker"}:
        argv = ["run", *argv]
    parser = _parser()
    args = parser.parse_args(argv)
    if str(args.mode) == "case_worker":
        return _run_case_worker(args)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
