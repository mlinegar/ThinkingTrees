#!/usr/bin/env python3
"""Teacher-first surrogate tournament push.

This runner turns the new stage-1 artifact path into a practical
"tournament of tournaments":

1. train a small bracket of stage-1 surrogate candidates;
2. freeze each surrogate as an artifact;
3. run a downstream stage-2 bracket against each artifact; and
4. rank surrogates by the best downstream decomposition-aware result.

The outer tournament optimizes the surrogate choice. The inner tournament
optimizes the downstream summary relative to that frozen surrogate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.core.tree_neural_config_recipes import (  # noqa: E402
    resolved_tree_batch_pack_mode,
    slot_exact_sanity_config,
)
from scripts import run_tree_neural_full_doc_mig as mig  # noqa: E402
from src.ctreepo.sim.core.tree_neural_execution import (  # noqa: E402
    run_job_batch,
    job_completion_keys,
    load_completed_run_keys,
    write_summary_outputs,
)
from src.ctreepo.sim.core.tree_neural_exact_sanity import (  # noqa: E402
    EXACT_SANITY_FAMILY,
    render_exact_sanity_summary_markdown,
    tree_neural_exact_sanity_summary,
)
from src.ctreepo.sim.core.tree_neural_facade import (  # noqa: E402
    build_jobs_for_configs,
    JobSpec,
    RunConfigSpec,
    discover_mig_uuids,
    job_output_dir_name,
    parse_mig_uuids,
    run_config_from_mapping,
)

GROUPED_STAGE2_SUMMARY_FILENAME = "grouped_stage2_summary.json"
GROUPED_STAGE2_COMPLETION_GRACE_S = 5.0



SURROGATE_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "label": "teacherfirst_shared_feature_phi128",
        "surface_mode": "shared_feature",
        "phi_dim": 128,
        "phi_hidden_dim": 256,
        "phi_compose_weight": 0.5,
        "phi_contrastive_weight": 0.1,
        "c2_mode": "reconstruction",
        "oracle_metric_name": "",
    },
    {
        "label": "teacherfirst_shared_feature_phi192",
        "surface_mode": "shared_feature",
        "phi_dim": 192,
        "phi_hidden_dim": 384,
        "phi_compose_weight": 0.5,
        "phi_contrastive_weight": 0.1,
        "c2_mode": "reconstruction",
        "oracle_metric_name": "",
    },
    {
        "label": "teacherfirst_shared_feature_adapters_phi128",
        "surface_mode": "shared_feature_adapters",
        "phi_dim": 128,
        "phi_hidden_dim": 256,
        "phi_compose_weight": 0.25,
        "phi_contrastive_weight": 0.0,
        "c2_mode": "reconstruction",
        "oracle_metric_name": "",
    },
    {
        "label": "teacherfirst_shared_feature_adapters_phi192",
        "surface_mode": "shared_feature_adapters",
        "phi_dim": 192,
        "phi_hidden_dim": 384,
        "phi_compose_weight": 0.25,
        "phi_contrastive_weight": 0.0,
        "c2_mode": "reconstruction",
        "oracle_metric_name": "",
    },
    {
        "label": "teacherfirst_scorefiber_s1_f15",
        "surface_mode": "factorized_score_fiber",
        "phi_dim": 16,
        "phi_hidden_dim": 128,
        "phi_compose_weight": 0.25,
        "phi_contrastive_weight": 0.1,
        "c2_mode": "fiber",
        "oracle_metric_name": "",
        "theorem_feature_adapter": "markov_score_endpoints",
        "score_dim": 1,
        "fiber_dim": 15,
        "aux_dim": 0,
        "score_merge_mode": "gated_affine",
    },
    {
        "label": "teacherfirst_scorefiber_s1_f31",
        "surface_mode": "factorized_score_fiber",
        "phi_dim": 32,
        "phi_hidden_dim": 256,
        "phi_compose_weight": 0.25,
        "phi_contrastive_weight": 0.1,
        "c2_mode": "fiber",
        "oracle_metric_name": "",
        "theorem_feature_adapter": "markov_score_endpoints",
        "score_dim": 1,
        "fiber_dim": 31,
        "aux_dim": 0,
        "score_merge_mode": "gated_affine",
    },
    {
        "label": "teacherfirst_fiber_primary_phi128",
        "surface_mode": "shared_feature",
        "phi_dim": 128,
        "phi_hidden_dim": 256,
        "phi_compose_weight": 0.5,
        "phi_contrastive_weight": 2.0,
        "c2_mode": "fiber",
        "oracle_metric_name": "markov",
    },
    {
        "label": "teacherfirst_fiber_primary_phi192",
        "surface_mode": "shared_feature",
        "phi_dim": 192,
        "phi_hidden_dim": 384,
        "phi_compose_weight": 0.5,
        "phi_contrastive_weight": 2.0,
        "c2_mode": "fiber",
        "oracle_metric_name": "markov",
    },
)

DEFAULT_ROOT_WEIGHT_EXPANSIONS: Mapping[str, tuple[float, ...]] = {
    "teacherfirst_shared_feature_adapters_phi128": (0.25, 0.5, 1.0),
    "teacherfirst_shared_feature_phi192": (0.5,),
    "teacherfirst_scorefiber_s1_f15": (0.5,),
    "teacherfirst_scorefiber_s1_f31": (0.5,),
}

STAGE2_JUDGE_CONDITIONS: tuple[dict[str, Any], ...] = (
    {
        "label": "leaf_dense",
        "leaf_label_rate": 1.0,
        "leaf_supervision_kind": "full_sketch",
        "internal_supervision_kind": "none",
        "internal_label_rate": 0.0,
    },
    {
        "label": "internal_count_dense",
        "leaf_label_rate": 1.0,
        "leaf_supervision_kind": "count_only",
        "internal_supervision_kind": "count_only",
        "internal_label_rate": 1.0,
    },
    {
        "label": "internal_full_dense",
        "leaf_label_rate": 1.0,
        "leaf_supervision_kind": "full_sketch",
        "internal_supervision_kind": "full_sketch",
        "internal_label_rate": 1.0,
    },
    {
        "label": "internal_full_r0p25",
        "leaf_label_rate": 0.25,
        "leaf_supervision_kind": "full_sketch",
        "internal_supervision_kind": "full_sketch",
        "internal_label_rate": 0.25,
    },
)


def _screen_doc_limit_for_train_docs(train_doc_count: int) -> int:
    train_docs = int(train_doc_count)
    if train_docs <= 128:
        return 16
    if train_docs <= 512:
        return 32
    if train_docs <= 1024:
        return 64
    return 128


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=str,
        default=f"outputs/tree_neural_teacher_first_push_{_timestamp()}",
    )
    parser.add_argument("--benchmark", type=str, default="recoverable_v4")
    parser.add_argument("--phase1-train-docs", type=int, default=128)
    parser.add_argument("--phase2-train-docs", type=int, default=256)
    parser.add_argument("--phase1-seeds", nargs="*", type=int, default=(0,))
    parser.add_argument("--phase2-seeds", nargs="*", type=int, default=(0, 1))
    parser.add_argument("--surrogate-labels", nargs="*", type=str, default=())
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--local-law-weight", dest="tree_local_law_weight", type=float, default=0.8)
    parser.add_argument("--tree-join-bit-weight", type=float, default=1.0)
    parser.add_argument("--stage1-epochs", type=int, default=12)
    parser.add_argument("--stage2-epochs", type=int, default=20)
    parser.add_argument("--tree-theorem-count-dim", type=int, default=8)
    parser.add_argument("--tree-theorem-first-dim", type=int, default=8)
    parser.add_argument("--tree-theorem-last-dim", type=int, default=8)
    parser.add_argument(
        "--root-search-labels",
        nargs="*",
        type=str,
        default=tuple(DEFAULT_ROOT_WEIGHT_EXPANSIONS.keys()),
    )
    parser.add_argument(
        "--stage1-root-weight-grid",
        nargs="*",
        type=float,
        default=(),
    )
    parser.add_argument("--promote-top-k", type=int, default=2)
    parser.add_argument(
        "--tree-stage1-eval-mode",
        type=str,
        default="per_epoch",
        choices=("per_epoch", "end_only"),
    )
    parser.add_argument("--tree-stage1-screen-doc-limit", type=int, default=0)
    parser.add_argument("--tree-stage1-final-exact-doc-limit", type=int, default=0)
    parser.add_argument("--tree-batch-pack-mode", type=str, default="")
    parser.add_argument("--batch-token-budget", type=int, default=0)
    parser.add_argument("--batch-node-budget", type=int, default=0)
    parser.add_argument(
        "--batch-autotune",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--eval-workers-per-mig", type=int, default=0)
    parser.add_argument(
        "--group-stage2-conditions",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--grouped-stage2-worker-manifest",
        type=str,
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--grouped-stage2-worker-output-dir",
        type=str,
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--grouped-stage2-worker-job-name",
        type=str,
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--mig-uuids", type=str, default="")
    parser.add_argument(
        "--use-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser


def _resolved_probe_dims(args: argparse.Namespace) -> tuple[int, int, int]:
    requested = (
        int(args.tree_theorem_count_dim),
        int(args.tree_theorem_first_dim),
        int(args.tree_theorem_last_dim),
    )
    state_dim = max(1, int(args.state_dim))
    if sum(requested) <= state_dim:
        return requested
    base = max(1, state_dim // 3)
    dims = [base, base, base]
    remainder = max(0, state_dim - sum(dims))
    idx = 0
    while remainder > 0:
        dims[idx % 3] += 1
        remainder -= 1
        idx += 1
    return int(dims[0]), int(dims[1]), int(dims[2])


def _weight_label(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def _resolved_root_weight_search(
    args: argparse.Namespace,
) -> Mapping[str, tuple[float, ...]]:
    enabled_labels = {
        str(label).strip()
        for label in getattr(args, "root_search_labels", ())
        if str(label).strip()
    }
    cli_grid = tuple(
        float(weight)
        for weight in getattr(args, "stage1_root_weight_grid", ())
    )
    resolved: Dict[str, tuple[float, ...]] = {}
    for label, weights in DEFAULT_ROOT_WEIGHT_EXPANSIONS.items():
        if label not in enabled_labels:
            continue
        raw_weights = cli_grid if cli_grid else tuple(float(weight) for weight in weights)
        positive = sorted({float(weight) for weight in raw_weights if float(weight) > 0.0})
        if positive:
            resolved[str(label)] = tuple(positive)
    return resolved


def _build_surrogate_variants(args: argparse.Namespace) -> tuple[dict[str, Any], ...]:
    root_weight_search = _resolved_root_weight_search(args)
    selected_labels = {
        str(label).strip()
        for label in getattr(args, "surrogate_labels", ())
        if str(label).strip()
    }
    variants: List[dict[str, Any]] = []
    for base in SURROGATE_VARIANTS:
        base_label = str(base["label"])
        if selected_labels and base_label not in selected_labels:
            continue
        base_variant = dict(base)
        base_variant["stage1_root_weight"] = 0.0
        base_variant["stage1_checkpoint_metric"] = "val_root_mae"
        variants.append(base_variant)
        for weight in root_weight_search.get(base_label, ()):
            variant = dict(base)
            variant["stage1_root_weight"] = float(weight)
            variant["stage1_checkpoint_metric"] = "val_root_mae"
            variant["label"] = f"{base_label}_root{_weight_label(weight)}"
            variants.append(variant)
    return tuple(variants)


def _base_args(args: argparse.Namespace) -> argparse.Namespace:
    theorem_count_dim, theorem_first_dim, theorem_last_dim = _resolved_probe_dims(args)
    return argparse.Namespace(
        benchmark=str(args.benchmark),
        train_doc_counts=(int(args.phase1_train_docs),),
        seeds=tuple(int(seed) for seed in args.phase1_seeds),
        job_granularity="family_train_seed",
        resume=True,
        mig_uuids="",
        state_dim=int(args.state_dim),
        hidden_dim=int(args.hidden_dim),
        n_epochs=int(args.stage1_epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        tree_local_law_weight=float(args.tree_local_law_weight),
        tree_task_objective_weight=None,
        tree_c1_relative_weight=1.0,
        tree_c2_relative_weight=1.0,
        tree_c3_relative_weight=1.0,
        tree_checkpoint_metric="val_exact_sketch_direct",
        tree_stage1_checkpoint_metric="val_root_mae",
        tree_stage1_eval_mode=str(
            getattr(args, "tree_stage1_eval_mode", "per_epoch")
        ),
        tree_stage1_screen_doc_limit=int(
            getattr(args, "tree_stage1_screen_doc_limit", 0)
        ),
        tree_stage1_final_exact_doc_limit=int(
            getattr(args, "tree_stage1_final_exact_doc_limit", 0)
        ),
        tree_batch_pack_mode=resolved_tree_batch_pack_mode(
            benchmark=str(getattr(args, "benchmark", "")),
            raw_value=getattr(args, "tree_batch_pack_mode", ""),
        ),
        tree_batch_token_budget=int(getattr(args, "batch_token_budget", 0)),
        tree_batch_node_budget=int(getattr(args, "batch_node_budget", 0)),
        tree_batch_autotune=bool(getattr(args, "batch_autotune", True)),
        tree_eval_workers_per_mig=int(getattr(args, "eval_workers_per_mig", 0)),
        tree_stage1_artifact_dir="",
        tree_stage1_root_weight=0.0,
        tree_join_bit_weight=float(args.tree_join_bit_weight),
        tree_training_schedule="two_stage",
        tree_stage1_epochs=int(args.stage1_epochs),
        tree_stage2_epochs=0,
        tree_task_head_mode="theorem_feature_scalar",
        tree_theorem_surface_mode="shared_feature",
        tree_theorem_count_head_mode="scalar_mse",
        tree_theorem_count_ordinal_weight=1.0,
        tree_theorem_count_scalar_aux_weight=0.25,
        tree_theorem_count_threshold_balance=True,
        tree_summary_spec_root_mode="factored_theorem_readout",
        tree_theorem_feature_dim=128,
        tree_theorem_feature_hidden_dim=256,
        tree_phi_compose_weight=0.5,
        tree_phi_contrastive_weight=0.1,
        tree_phi_alignment_loss="cosine_mse",
        tree_c2_mode="reconstruction",
        oracle_metric_name="",
        theorem_feature_adapter="markov_count_sketch",
        theorem_pair_same_threshold=None,
        theorem_pair_diff_threshold=None,
        tree_theorem_count_dim=int(theorem_count_dim),
        tree_theorem_first_dim=int(theorem_first_dim),
        tree_theorem_last_dim=int(theorem_last_dim),
        leaf_supervision_kind="full_sketch",
        doc_sequence_train_fraction=0.0,
        torch_threads=int(args.torch_threads),
        use_cuda=bool(args.use_cuda),
    )


def _make_stage1_config(
    args: argparse.Namespace,
    *,
    train_doc_count: int,
    variant: Mapping[str, Any],
) -> RunConfigSpec:
    base = slot_exact_sanity_config(
        _base_args(args),
        train_doc_count=int(train_doc_count),
        config_label=str(variant["label"]),
        leaf_label_rate=1.0,
        leaf_supervision_kind="full_sketch",
        internal_supervision_kind="full_sketch",
        internal_label_rate=1.0,
        tree_summary_spec_root_mode="factored_theorem_readout",
    )
    return replace(
        base,
        label=str(variant["label"]),
        n_epochs=int(args.stage1_epochs),
        tree_training_schedule="two_stage",
        tree_stage1_epochs=int(args.stage1_epochs),
        tree_stage2_epochs=0,
        tree_stage1_root_weight=float(variant.get("stage1_root_weight", 0.0)),
        tree_task_head_mode="theorem_feature_scalar",
        tree_theorem_surface_mode=str(variant["surface_mode"]),
        tree_summary_spec_root_mode="factored_theorem_readout",
        tree_theorem_feature_dim=int(variant["phi_dim"]),
        tree_theorem_feature_hidden_dim=int(variant["phi_hidden_dim"]),
        tree_theorem_score_dim=int(variant.get("score_dim", 0)),
        tree_theorem_fiber_dim=int(variant.get("fiber_dim", 0)),
        tree_theorem_aux_dim=int(variant.get("aux_dim", 0)),
        tree_score_merge_mode=str(variant.get("score_merge_mode", "gated_affine")),
        tree_phi_compose_weight=float(variant["phi_compose_weight"]),
        tree_phi_contrastive_weight=float(variant["phi_contrastive_weight"]),
        tree_c2_mode=str(variant.get("c2_mode", "reconstruction")),
        oracle_metric_name=str(variant.get("oracle_metric_name", "")),
        theorem_feature_adapter=str(variant.get("theorem_feature_adapter", "markov_count_sketch")),
        tree_stage1_checkpoint_metric=str(
            variant.get("stage1_checkpoint_metric", "val_root_mae")
        ),
        leaf_label_rate=1.0,
        leaf_supervision_kind="full_sketch",
        internal_supervision_kind="full_sketch",
        internal_label_rate=1.0,
    )


def _build_phase_jobs(
    *,
    args: argparse.Namespace,
    configs_by_train: Sequence[tuple[int, RunConfigSpec]],
    seeds: Sequence[int],
    tuning_stage: str,
    study_axis: str,
) -> List[JobSpec]:
    jobs: List[JobSpec] = []
    for train_doc_count, config in configs_by_train:
        jobs.extend(
            build_jobs_for_configs(
                families=(EXACT_SANITY_FAMILY,),
                train_doc_counts=(int(train_doc_count),),
                benchmark=str(args.benchmark),
                hardness_grid="",
                grid_cell_ids=(),
                seeds=tuple(int(seed) for seed in seeds),
                job_granularity="family_train_seed",
                repeat_closed_form_controls=True,
                configs=(config,),
                tuning_stage=str(tuning_stage),
                study_name="teacher_first_tournament",
                study_axis=str(study_axis),
                axis_value=str(config.label),
                selection_metric="teacher_first_total_bound",
            )
        )
    return jobs


def _run_phase(
    *,
    output_root: Path,
    jobs: Sequence[JobSpec],
    args: argparse.Namespace,
    manifest_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    if bool(args.use_cuda):
        mig_uuids = (
            parse_mig_uuids(args.mig_uuids)
            if str(args.mig_uuids).strip()
            else discover_mig_uuids()
        )
        if not mig_uuids:
            raise RuntimeError("No MIG UUIDs discovered")
    else:
        mig_uuids = ["cpu0"]
    result = run_job_batch(
        output_root=output_root,
        jobs=jobs,
        mig_uuids=mig_uuids,
        resume_enabled=True,
        use_cuda=bool(args.use_cuda),
        torch_threads=int(args.torch_threads),
        manifest_payload=dict(manifest_payload),
    )
    try:
        payload = write_summary_outputs(output_root)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"no completed diagnostic runs were written under {output_root}; "
            "inspect worker logs for stage failures"
        ) from exc
    return {
        "result": result,
        "payload": payload,
        "runs": [dict(run) for run in payload.get("runs") or []],
    }


def _direct_metric(run: Mapping[str, Any], key: str, default: float = float("nan")) -> float:
    value = run.get(key)
    if value not in {"", None}:
        return float(value)
    direct = dict(
        ((run.get("exact_sketch_diagnostics") or {}).get("direct_selection_metrics") or {}).get(
            "test",
            {},
        )
        or {}
    )
    fallback = direct.get(key, default)
    return float(fallback)


def _stage2_rank_key(run: Mapping[str, Any]) -> tuple[float, float, float, float]:
    bound = _direct_metric(run, "teacher_first_total_bound", default=float("inf"))
    substitution = _direct_metric(run, "stage1_substitution_cost", default=float("inf"))
    root_mae = _direct_metric(run, "test_root_mae", default=float("inf"))
    transport = _direct_metric(run, "stage2_transport_budget", default=float("inf"))
    return (
        float(bound if bound == bound else float("inf")),
        float(substitution if substitution == substitution else float("inf")),
        float(root_mae if root_mae == root_mae else float("inf")),
        float(transport if transport == transport else float("inf")),
    )


def _summary_rank_key(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row.get("mean_teacher_first_total_bound", float("inf"))),
        float(row.get("mean_stage1_substitution_cost", float("inf"))),
        float(row.get("mean_test_root_mae", float("inf"))),
        float(row.get("mean_stage2_transport_budget", float("inf"))),
    )


def _dominates_summary(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    left_key = _summary_rank_key(left)
    right_key = _summary_rank_key(right)
    return all(a <= b for a, b in zip(left_key, right_key)) and any(
        a < b for a, b in zip(left_key, right_key)
    )


def _pareto_frontier_labels(
    summary_rows: Sequence[Mapping[str, Any]],
) -> List[str]:
    frontier: List[str] = []
    for row in summary_rows:
        label = str(row.get("candidate_label", ""))
        if not label:
            continue
        dominated = False
        for other in summary_rows:
            other_label = str(other.get("candidate_label", ""))
            if not other_label or other_label == label:
                continue
            if _dominates_summary(other, row):
                dominated = True
                break
        if not dominated:
            frontier.append(label)
    return frontier


def _stage1_run_index(
    runs: Sequence[Mapping[str, Any]],
) -> Dict[tuple[str, int], Dict[str, Any]]:
    index: Dict[tuple[str, int], Dict[str, Any]] = {}
    for run in runs:
        label = str(run.get("config_label", "")).strip()
        seed = int(run.get("seed", 0))
        if not label:
            continue
        current = index.get((label, seed))
        if current is None or _stage2_rank_key(run) < _stage2_rank_key(current):
            index[(label, seed)] = dict(run)
    return index


def _make_stage2_config(
    *,
    base_config: RunConfigSpec,
    condition: Mapping[str, Any],
    artifact_dir: str,
    stage2_epochs: int,
    train_doc_count: int,
) -> RunConfigSpec:
    condition_label = str(condition["label"])
    return replace(
        base_config,
        label=f"{base_config.label}__{condition_label}__judge_t{int(train_doc_count)}",
        n_epochs=int(stage2_epochs),
        tree_training_schedule="two_stage",
        tree_stage1_epochs=0,
        tree_stage2_epochs=int(stage2_epochs),
        tree_stage1_artifact_dir=str(artifact_dir),
        leaf_label_rate=float(condition["leaf_label_rate"]),
        leaf_supervision_kind=str(condition["leaf_supervision_kind"]),
        internal_supervision_kind=str(condition["internal_supervision_kind"]),
        internal_label_rate=float(condition["internal_label_rate"]),
    )


def _build_stage2_jobs(
    *,
    args: argparse.Namespace,
    stage1_runs: Sequence[Mapping[str, Any]],
    base_configs: Mapping[str, RunConfigSpec],
    stage2_epochs: int | None = None,
) -> List[JobSpec]:
    stage1_by_label_seed = _stage1_run_index(stage1_runs)
    configs_by_train: List[tuple[int, RunConfigSpec]] = []
    target_train_docs = int(args.phase2_train_docs)
    resolved_stage2_epochs = int(
        args.stage2_epochs if stage2_epochs is None else stage2_epochs
    )
    for label, base_config in base_configs.items():
        stage1_run = stage1_by_label_seed.get((str(label), int(args.phase1_seeds[0])))
        if stage1_run is None:
            continue
        artifact_dir = str(stage1_run.get("tree_stage1_artifact_dir", "")).strip()
        if not artifact_dir:
            continue
        for condition in STAGE2_JUDGE_CONDITIONS:
            configs_by_train.append(
                (
                    target_train_docs,
                    _make_stage2_config(
                        base_config=base_config,
                        condition=condition,
                        artifact_dir=artifact_dir,
                        stage2_epochs=int(resolved_stage2_epochs),
                        train_doc_count=target_train_docs,
                    ),
                )
            )
    return _build_phase_jobs(
        args=args,
        configs_by_train=configs_by_train,
        seeds=tuple(int(seed) for seed in args.phase2_seeds),
        tuning_stage="stage2_judge",
        study_axis="stage2_judge_config",
    )


def _job_from_mapping(mapping: Mapping[str, Any]) -> JobSpec:
    config_mapping = dict(mapping.get("config") or {})
    job_kwargs = {
        "family": str(mapping.get("family", "")),
        "train_doc_count": int(mapping.get("train_doc_count", 0)),
        "benchmark": str(mapping.get("benchmark", "")),
        "hardness_grid": str(mapping.get("hardness_grid", "")),
        "grid_cell_ids": tuple(str(cell) for cell in mapping.get("grid_cell_ids", ()) or ()),
        "seeds": tuple(int(seed) for seed in mapping.get("seeds", ()) or ()),
        "config": run_config_from_mapping(config_mapping),
        "tuning_stage": str(mapping.get("tuning_stage", "")),
        "test_metrics_hidden_during_selection": bool(
            mapping.get("test_metrics_hidden_during_selection", False)
        ),
        "study_name": str(mapping.get("study_name", "")),
        "study_axis": str(mapping.get("study_axis", "")),
        "axis_value": str(mapping.get("axis_value", "")),
        "locked_tree_neural_config_label": str(
            mapping.get("locked_tree_neural_config_label", "")
        ),
        "selection_metric": str(mapping.get("selection_metric", "")),
        "budget_total_calls": int(mapping.get("budget_total_calls", 0)),
        "budget_total_calls_per_doc": float(mapping.get("budget_total_calls_per_doc", 0.0)),
        "full_doc_budget_share": float(mapping.get("full_doc_budget_share", 1.0)),
        "doc_consumption_mode": str(mapping.get("doc_consumption_mode", "")),
        "local_split_mode": str(mapping.get("local_split_mode", "")),
        "local_allocation_policy": str(mapping.get("local_allocation_policy", "")),
    }
    known_fields = set(JobSpec.__dataclass_fields__)
    return JobSpec(
        **{key: value for key, value in job_kwargs.items() if key in known_fields}
    )


def _worker_namespace_for_job(
    *,
    job: JobSpec,
    output_dir: Path,
    use_cuda: bool,
    torch_threads: int,
) -> argparse.Namespace:
    job_dict = asdict(job)
    config_dict = dict(job_dict.pop("config"))
    payload: Dict[str, Any] = {
        **job_dict,
        **config_dict,
        "output_dir": str(output_dir),
        "use_cuda": bool(use_cuda),
        "torch_threads": int(torch_threads),
        "job_name": str(job.job_name),
        "config_label": str(job.config.label),
    }
    return argparse.Namespace(**payload)


def _grouped_stage2_job_name(
    *,
    candidate_label: str,
    benchmark: str,
    family: str,
    train_doc_count: int,
    seed: int,
) -> str:
    return (
        f"{benchmark}__{family}__train_{int(train_doc_count)}"
        f"__stage_grouped_stage2__cfg_{str(candidate_label)}__seed_{int(seed)}"
    )


def _build_grouped_stage2_jobs(
    jobs: Sequence[JobSpec],
) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, int, int, str, str], List[JobSpec]] = {}
    for job in jobs:
        config_label = str(job.config.label).strip()
        candidate_label = config_label.split("__", 1)[0]
        if not candidate_label:
            continue
        seed = int(job.seeds[0]) if job.seeds else 0
        key = (
            candidate_label,
            seed,
            int(job.train_doc_count),
            str(job.benchmark),
            str(job.family),
        )
        grouped.setdefault(key, []).append(job)
    manifests: List[Dict[str, Any]] = []
    for (
        candidate_label,
        seed,
        train_doc_count,
        benchmark,
        family,
    ), condition_jobs in sorted(grouped.items()):
        manifests.append(
            {
                "candidate_label": str(candidate_label),
                "seed": int(seed),
                "train_doc_count": int(train_doc_count),
                "benchmark": str(benchmark),
                "family": str(family),
                "job_name": _grouped_stage2_job_name(
                    candidate_label=str(candidate_label),
                    benchmark=str(benchmark),
                    family=str(family),
                    train_doc_count=int(train_doc_count),
                    seed=int(seed),
                ),
                "jobs": [asdict(job) for job in condition_jobs],
            }
        )
    return manifests


def _grouped_stage2_summary_path(output_dir: Path) -> Path:
    return output_dir / GROUPED_STAGE2_SUMMARY_FILENAME


def _load_grouped_stage2_summary(output_dir: Path) -> Dict[str, Any] | None:
    summary_path = _grouped_stage2_summary_path(output_dir)
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    if not isinstance(payload.get("condition_results"), Sequence):
        return None
    return dict(payload)


def _run_grouped_stage2_worker(
    *,
    manifest_path: Path,
    output_dir: Path,
    job_name: str,
    use_cuda: bool,
    torch_threads: int,
) -> Dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_results: List[Dict[str, Any]] = []
    worker_start_s = time.perf_counter()
    for job_mapping in manifest.get("jobs", []):
        job = _job_from_mapping(job_mapping)
        condition_output_dir = output_dir / "conditions" / job_output_dir_name(
            job.job_name
        )
        condition_output_dir.mkdir(parents=True, exist_ok=True)
        condition_start_s = time.perf_counter()
        payload = mig._worker_payload(
            _worker_namespace_for_job(
                job=job,
                output_dir=condition_output_dir,
                use_cuda=bool(use_cuda),
                torch_threads=int(torch_threads),
            )
        )
        condition_results.append(
            {
                "job_name": str(job.job_name),
                "config_label": str(job.config.label),
                "seed": int(job.seeds[0]) if job.seeds else 0,
                "output_dir": str(condition_output_dir),
                "elapsed_s": float(time.perf_counter() - condition_start_s),
                "payload": dict(payload),
            }
        )
    grouped_payload = {
        "job_name": str(job_name),
        "manifest": dict(manifest),
        "condition_results": condition_results,
        "elapsed_s": float(time.perf_counter() - worker_start_s),
    }
    _grouped_stage2_summary_path(output_dir).write_text(
        json.dumps(grouped_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return grouped_payload


def _run_grouped_stage2_phase(
    *,
    output_root: Path,
    grouped_jobs: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    manifest_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    job_root = output_root / "jobs"
    job_root.mkdir(parents=True, exist_ok=True)
    (output_root / "mig_job_manifest.json").write_text(
        json.dumps(dict(manifest_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    completed_run_keys = load_completed_run_keys(output_root)
    pending: List[Dict[str, Any]] = []
    skipped_jobs: List[Dict[str, Any]] = []
    for grouped_job in grouped_jobs:
        condition_jobs = [
            _job_from_mapping(mapping)
            for mapping in list(grouped_job.get("jobs", ()) or ())
        ]
        required_keys = set()
        for job in condition_jobs:
            required_keys.update(job_completion_keys(job))
        if required_keys and required_keys.issubset(completed_run_keys):
            skipped_jobs.append(
                {
                    "job_name": str(grouped_job.get("job_name", "")),
                    "reason": "already_completed",
                }
            )
            continue
        pending.append(dict(grouped_job))

    available_tokens = (
        parse_mig_uuids(str(args.mig_uuids))
        if bool(args.use_cuda) and str(args.mig_uuids).strip()
        else (discover_mig_uuids() if bool(args.use_cuda) else ["cpu0"])
    )
    if bool(args.use_cuda) and not available_tokens:
        raise RuntimeError("No MIG UUIDs discovered")

    active: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    completed: List[Dict[str, Any]] = []
    stop_requested = False
    force_terminate_requested = False

    def _request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested, force_terminate_requested
        if not stop_requested:
            stop_requested = True
            print(
                f"received signal {int(signum)}; pausing grouped stage-2 launch queue",
                flush=True,
            )
            return
        if force_terminate_requested:
            return
        force_terminate_requested = True
        for entry in active:
            proc = entry.get("proc")
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    continue

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    while pending or active:
        while pending and available_tokens and not stop_requested:
            token = available_tokens.pop(0)
            grouped_job = pending.pop(0)
            job_output_dir = job_root / job_output_dir_name(
                str(grouped_job["job_name"])
            )
            job_output_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = job_output_dir / "group_manifest.json"
            manifest_path.write_text(
                json.dumps(dict(grouped_job), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            log_path = job_output_dir / "worker.log"
            log_fh = open(log_path, "w", encoding="utf-8")
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--grouped-stage2-worker-manifest",
                str(manifest_path),
                "--grouped-stage2-worker-output-dir",
                str(job_output_dir),
                "--grouped-stage2-worker-job-name",
                str(grouped_job["job_name"]),
                "--torch-threads",
                str(int(args.torch_threads)),
            ]
            cmd.append("--use-cuda" if bool(args.use_cuda) else "--no-use-cuda")
            env = dict(os.environ)
            if bool(args.use_cuda):
                env["CUDA_VISIBLE_DEVICES"] = str(token)
            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                cwd=str(REPO_ROOT),
                env=env,
            )
            active.append(
                {
                    "proc": proc,
                    "token": token,
                    "log_fh": log_fh,
                    "log_path": str(log_path),
                    "job_name": str(grouped_job["job_name"]),
                    "output_dir": str(job_output_dir),
                    "summary_ready_at": None,
                }
            )
        if not active:
            break
        time.sleep(1.0)
        still_active: List[Dict[str, Any]] = []
        for entry in active:
            proc = entry["proc"]
            output_dir = Path(str(entry["output_dir"]))
            grouped_summary = _load_grouped_stage2_summary(output_dir)
            if grouped_summary is not None and entry.get("summary_ready_at") is None:
                entry["summary_ready_at"] = time.monotonic()
            code = proc.poll()
            if code is None:
                summary_ready_at = entry.get("summary_ready_at")
                if (
                    grouped_summary is not None
                    and summary_ready_at is not None
                    and (
                        time.monotonic() - float(summary_ready_at)
                        >= float(GROUPED_STAGE2_COMPLETION_GRACE_S)
                    )
                ):
                    try:
                        proc.terminate()
                    except ProcessLookupError:
                        pass
                    try:
                        proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        try:
                            proc.kill()
                        except ProcessLookupError:
                            pass
                        try:
                            proc.wait(timeout=2.0)
                        except subprocess.TimeoutExpired:
                            pass
                    code = proc.poll()
                else:
                    still_active.append(entry)
                    continue
            entry["log_fh"].close()
            available_tokens.append(entry["token"])
            returncode = int(code) if code is not None else 0
            record = {
                "job_name": str(entry["job_name"]),
                "returncode": int(returncode),
                "log_path": str(entry["log_path"]),
                "summary_path": str(_grouped_stage2_summary_path(output_dir)),
            }
            if grouped_summary is not None:
                print(
                    f"completed {entry['job_name']} summary={record['summary_path']}",
                    flush=True,
                )
                completed.append(record)
            elif int(returncode) == 0:
                completed.append(record)
            else:
                failed.append(record)
        active = still_active
        available_tokens.sort()

    result = {
        "completed_jobs": completed,
        "failed_jobs": failed,
        "skipped_jobs": skipped_jobs,
    }
    (output_root / "controller_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if failed:
        raise RuntimeError(
            f"grouped stage-2 worker failures under {output_root}: "
            + ", ".join(str(item["job_name"]) for item in failed)
        )
    try:
        payload = write_summary_outputs(output_root)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"no completed diagnostic runs were written under {output_root}; "
            "inspect grouped worker logs for stage failures"
        ) from exc
    return {
        "result": result,
        "payload": payload,
        "runs": [dict(run) for run in payload.get("runs") or []],
    }


def _run_stage2_phase(
    *,
    output_root: Path,
    args: argparse.Namespace,
    jobs: Sequence[JobSpec],
    grouped_conditions: bool,
    manifest_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    if not bool(grouped_conditions):
        return _run_phase(
            output_root=output_root,
            jobs=jobs,
            args=args,
            manifest_payload=manifest_payload,
        )
    grouped_jobs = _build_grouped_stage2_jobs(jobs)
    return _run_grouped_stage2_phase(
        output_root=output_root,
        grouped_jobs=grouped_jobs,
        args=args,
        manifest_payload={
            **dict(manifest_payload),
            "grouped_stage2_conditions": True,
            "grouped_jobs": grouped_jobs,
        },
    )


def _aggregate_candidate_summary(
    stage2_runs: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    by_candidate_seed: Dict[tuple[str, int], List[Dict[str, Any]]] = {}
    for run in stage2_runs:
        config_label = str(run.get("config_label", "")).strip()
        candidate_label = config_label.split("__", 1)[0]
        seed = int(run.get("seed", 0))
        by_candidate_seed.setdefault((candidate_label, seed), []).append(dict(run))

    summaries: List[Dict[str, Any]] = []
    by_candidate: Dict[str, List[Dict[str, Any]]] = {}
    for (candidate_label, _seed), runs in by_candidate_seed.items():
        best = min(runs, key=_stage2_rank_key)
        by_candidate.setdefault(candidate_label, []).append(best)

    for candidate_label, best_runs in sorted(by_candidate.items()):
        bounds = [_direct_metric(run, "teacher_first_total_bound", float("inf")) for run in best_runs]
        substitutions = [_direct_metric(run, "stage1_substitution_cost", float("inf")) for run in best_runs]
        root_maes = [_direct_metric(run, "test_root_mae", float("inf")) for run in best_runs]
        transports = [_direct_metric(run, "stage2_transport_budget", float("inf")) for run in best_runs]
        stage1_root_weights = [
            float(run.get("tree_stage1_root_weight", 0.0)) for run in best_runs
        ]
        checkpoint_metrics = sorted(
            {
                str(run.get("tree_stage1_checkpoint_metric", "")).strip()
                for run in best_runs
                if str(run.get("tree_stage1_checkpoint_metric", "")).strip()
            }
        )
        summary = {
            "candidate_label": str(candidate_label),
            "n_stage2_seed_wins": int(len(best_runs)),
            "mean_teacher_first_total_bound": float(sum(bounds) / max(1, len(bounds))),
            "mean_stage1_substitution_cost": float(
                sum(substitutions) / max(1, len(substitutions))
            ),
            "mean_test_root_mae": float(sum(root_maes) / max(1, len(root_maes))),
            "mean_stage2_transport_budget": float(
                sum(transports) / max(1, len(transports))
            ),
            "mean_stage1_root_weight": float(
                sum(stage1_root_weights) / max(1, len(stage1_root_weights))
            ),
            "stage1_checkpoint_metric": str(checkpoint_metrics[0]) if checkpoint_metrics else "",
            "best_stage2_run_labels": [str(run.get("config_label", "")) for run in best_runs],
        }
        summaries.append(summary)
    summaries.sort(key=_summary_rank_key)
    frontier_labels = set(_pareto_frontier_labels(summaries))
    for row in summaries:
        row["on_pareto_frontier"] = str(row.get("candidate_label", "")) in frontier_labels
    return summaries


def _render_summary_markdown(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    phase1_count: int,
    phase2_count: int,
    pareto_frontier: Sequence[str],
) -> str:
    lines = [
        "# Teacher-First Surrogate Tournament Summary",
        "",
        f"- phase1 runs: `{int(phase1_count)}`",
        f"- phase2 runs: `{int(phase2_count)}`",
        f"- pareto_frontier: `{', '.join(str(label) for label in pareto_frontier)}`",
        "",
        "## Candidate Ranking",
        "",
    ]
    for idx, row in enumerate(summary_rows, start=1):
        lines.extend(
            [
                f"### {idx}. `{str(row.get('candidate_label', ''))}`",
                f"- mean_teacher_first_total_bound: `{float(row.get('mean_teacher_first_total_bound', float('nan'))):.6g}`",
                f"- mean_stage1_substitution_cost: `{float(row.get('mean_stage1_substitution_cost', float('nan'))):.6g}`",
                f"- mean_test_root_mae: `{float(row.get('mean_test_root_mae', float('nan'))):.6g}`",
                f"- mean_stage2_transport_budget: `{float(row.get('mean_stage2_transport_budget', float('nan'))):.6g}`",
                f"- mean_stage1_root_weight: `{float(row.get('mean_stage1_root_weight', float('nan'))):.6g}`",
                f"- stage1_checkpoint_metric: `{str(row.get('stage1_checkpoint_metric', ''))}`",
                f"- on_pareto_frontier: `{bool(row.get('on_pareto_frontier', False))}`",
                f"- best_stage2_run_labels: `{', '.join(str(x) for x in row.get('best_stage2_run_labels', []))}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = _parser().parse_args()
    if str(getattr(args, "grouped_stage2_worker_manifest", "")).strip():
        payload = _run_grouped_stage2_worker(
            manifest_path=Path(str(args.grouped_stage2_worker_manifest)),
            output_dir=Path(str(args.grouped_stage2_worker_output_dir)),
            job_name=str(args.grouped_stage2_worker_job_name),
            use_cuda=bool(args.use_cuda),
            torch_threads=int(args.torch_threads),
        )
        print(
            json.dumps(
                {
                    "job_name": str(payload.get("job_name", "")),
                    "grouped_stage2_summary": str(
                        _grouped_stage2_summary_path(
                            Path(str(args.grouped_stage2_worker_output_dir))
                        )
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    output_root = Path(str(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    surrogate_variants = _build_surrogate_variants(args)

    stage1_configs = {
        str(variant["label"]): _make_stage1_config(
            args,
            train_doc_count=int(args.phase1_train_docs),
            variant=variant,
        )
        for variant in surrogate_variants
    }
    phase1_jobs = _build_phase_jobs(
        args=args,
        configs_by_train=[
            (int(args.phase1_train_docs), config) for config in stage1_configs.values()
        ],
        seeds=tuple(int(seed) for seed in args.phase1_seeds),
        tuning_stage="stage1_surrogate",
        study_axis="stage1_surrogate",
    )
    phase1 = _run_phase(
        output_root=output_root / "phase1",
        jobs=phase1_jobs,
        args=args,
        manifest_payload={
            "mode": "teacher_first_stage1_surrogate",
            "benchmark": str(args.benchmark),
            "jobs": [asdict(job) for job in phase1_jobs],
        },
    )

    phase2_jobs = _build_stage2_jobs(
        args=args,
        stage1_runs=phase1["runs"],
        base_configs=stage1_configs,
    )
    phase2 = _run_stage2_phase(
        output_root=output_root / "phase2",
        args=args,
        jobs=phase2_jobs,
        grouped_conditions=bool(args.group_stage2_conditions),
        manifest_payload={
            "mode": "teacher_first_stage2_judge",
            "benchmark": str(args.benchmark),
            "jobs": [asdict(job) for job in phase2_jobs],
            "grouped_stage2_conditions": bool(args.group_stage2_conditions),
        },
    )

    candidate_summary = _aggregate_candidate_summary(phase2["runs"])
    pareto_frontier = [
        str(row.get("candidate_label", ""))
        for row in candidate_summary
        if bool(row.get("on_pareto_frontier", False))
    ]
    promoted = [
        str(row.get("candidate_label", ""))
        for row in candidate_summary[: max(0, int(args.promote_top_k))]
    ]
    payload = {
        "benchmark": str(args.benchmark),
        "phase1_train_docs": int(args.phase1_train_docs),
        "phase2_train_docs": int(args.phase2_train_docs),
        "phase1_runs": int(len(phase1["runs"])),
        "phase2_runs": int(len(phase2["runs"])),
        "candidate_summary": candidate_summary,
        "pareto_frontier": pareto_frontier,
        "promoted_candidates": promoted,
        "surrogate_variants": [dict(variant) for variant in surrogate_variants],
        "stage2_conditions": [dict(condition) for condition in STAGE2_JUDGE_CONDITIONS],
    }
    (output_root / "teacher_first_tournament_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_root / "teacher_first_tournament_summary.md").write_text(
        _render_summary_markdown(
            candidate_summary,
            phase1_count=int(len(phase1["runs"])),
            phase2_count=int(len(phase2["runs"])),
            pareto_frontier=pareto_frontier,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
