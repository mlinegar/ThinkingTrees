from __future__ import annotations

"""Exact-sanity reporting helpers for tree-neural Markov runs."""

from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from src.ctreepo.sim.core.full_doc_anchor_diagnostics import FAIR_FNO_PARITY_CONFIG_LABEL

EXACT_SANITY_STUDY_NAME = "tree_neural_exact_sanity"
EXACT_SANITY_FAMILY = "tree_neural"
EXACT_SANITY_LEVELS = ("leaf", "merge", "root")
EXACT_SANITY_COMPONENT_METRICS = (
    "count_mae",
    "count_match_rate",
    "first_accuracy",
    "last_accuracy",
    "exact_summary_match_rate",
)
EXACT_SANITY_MERGE_CONSISTENCY_METRICS = (
    "merge_join_bit_accuracy",
    "merge_decoded_consistency_count_mae",
    "merge_decoded_consistency_first_accuracy",
    "merge_decoded_consistency_last_accuracy",
)
EXACT_SANITY_LAW_METRICS = (
    "root_mae",
    "leaf_mae",
    "c2_idempotence_mae",
    "merge_mae",
)


def _sanitize_label(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value).strip()).strip("_")
    return cleaned or "default"


def _format_float_label(value: float) -> str:
    text = f"{float(value):.6g}"
    return _sanitize_label(text.replace("-", "m").replace(".", "p"))



def nested_mapping_value(
    mapping: Mapping[str, Any],
    path: Sequence[str],
    *,
    default: Any = float("nan"),
) -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, Mapping):
            return default
        cur = cur.get(str(key))
    return cur if cur is not None else default


def finite_summary_stats(values: Sequence[Any]) -> Dict[str, Any]:
    arr = np.asarray([float(value) for value in values], dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "n": int(finite.size),
    }


def _exact_sanity_metric_summary(
    runs: Sequence[Mapping[str, Any]],
    path: Sequence[str],
) -> Dict[str, Any]:
    return finite_summary_stats(
        [nested_mapping_value(run, path) for run in runs]
    )


def _exact_sanity_condition_kind(run: Mapping[str, Any]) -> str:
    config_label = str(run.get("config_label", "")).strip()
    task_split_suffix = "_task_split_ablation"
    task_split_ablation = config_label.endswith(task_split_suffix)
    base_config_label = (
        config_label[: -len(task_split_suffix)]
        if task_split_ablation and len(config_label) > len(task_split_suffix)
        else config_label
    )
    exact_label_map = {
        FAIR_FNO_PARITY_CONFIG_LABEL: "legacy_fair_fno_root_only",
        "tree_neural_slot_align_v1_root_only": "slot_root_only",
        "tree_neural_slot_align_v1_leaf_sampled": "slot_leaf_sampled_r0p25",
        "tree_neural_slot_align_v1_leaf_dense": "slot_leaf_dense",
        "tree_neural_slot_align_v1_internal_count_r0p25": "slot_internal_count_only_r0p25",
        "tree_neural_slot_align_v1_internal_full_r0p25": "slot_internal_full_sketch_r0p25",
        "tree_neural_slot_align_v1_internal_count_dense": "slot_internal_count_only_dense",
        "tree_neural_slot_align_v1_internal_full_dense": "slot_internal_full_sketch_dense",
        "tree_neural_slot_align_v1_internal_full_r0p5": "slot_internal_full_sketch_r0p5",
    }
    if base_config_label in exact_label_map:
        base_kind = exact_label_map[base_config_label]
        return (
            f"{base_kind}__task_split_ablation"
            if task_split_ablation
            else base_kind
        )
    summary_spec_name = str(run.get("summary_spec_name", "")).strip()
    leaf_label_rate = float(run.get("leaf_label_rate", 1.0) or 0.0)
    internal_kind = str(run.get("internal_supervision_kind", "none")).strip() or "none"
    internal_rate = float(run.get("internal_label_rate", 0.0) or 0.0)
    if summary_spec_name != "markov_count_sketch":
        return config_label or "legacy_unknown"
    if internal_kind == "none" and leaf_label_rate <= 0.0:
        return "slot_root_only"
    if internal_kind == "none":
        rate_label = _format_float_label(float(leaf_label_rate))
        if rate_label == "1":
            return "slot_leaf_dense"
        return f"slot_leaf_sampled_r{rate_label}"
    rate_label = _format_float_label(float(internal_rate))
    if internal_kind == "count_only":
        if rate_label == "1":
            return "slot_internal_count_only_dense"
        return f"slot_internal_count_only_r{rate_label}"
    if internal_kind == "full_sketch":
        if rate_label == "1":
            base_kind = "slot_internal_full_sketch_dense"
        else:
            base_kind = f"slot_internal_full_sketch_r{rate_label}"
        return (
            f"{base_kind}__task_split_ablation"
            if task_split_ablation
            else base_kind
        )
    return config_label or "aligned_unknown"


def _exact_sanity_condition_id(run: Mapping[str, Any]) -> str:
    config_label = str(run.get("config_label", "")).strip()
    if config_label:
        return config_label
    return _exact_sanity_condition_kind(run)


def _exact_sanity_condition_title(condition_id: str) -> str:
    fixed_titles = {
        FAIR_FNO_PARITY_CONFIG_LABEL: "Legacy Fair-FNO Root-Only",
        "tree_neural_slot_align_v1_root_only": "Slot-Aligned Root-Only",
        "tree_neural_slot_align_v1_leaf_sampled": "Slot-Aligned Leaf Sampled @ 0.25",
        "tree_neural_slot_align_v1_leaf_dense": "Slot-Aligned Leaf Dense",
        "tree_neural_slot_align_v1_internal_count_r0p25": "Slot-Aligned Internal Count-Only @ 0.25",
        "tree_neural_slot_align_v1_internal_full_r0p25": "Slot-Aligned Internal Full-Sketch @ 0.25",
        "tree_neural_slot_align_v1_internal_count_dense": "Slot-Aligned Internal Count-Only Dense",
        "tree_neural_slot_align_v1_internal_full_dense": "Slot-Aligned Internal Full-Sketch Dense",
        "tree_neural_slot_align_v1_internal_full_r0p5": "Slot-Aligned Internal Full-Sketch @ 0.5",
        "tree_neural_slot_align_v1_internal_full_r0p25_task_split_ablation": (
            "Slot-Aligned Internal Full-Sketch @ 0.25 (Task-Split Ablation)"
        ),
        "tree_neural_slot_align_v1_balanced_full_r0p25": "Slot-Aligned Rebalanced Full-Sketch @ 0.25",
        "tree_neural_slot_align_v1_leaf_ep_count_r0p25": "Slot-Aligned Leaf Full-Sketch + Internal Count @ 0.25",
        "legacy_fair_fno_root_only": "Legacy Fair-FNO Root-Only",
        "slot_root_only": "Slot-Aligned Root-Only",
        "slot_leaf_sampled_r0p25": "Slot-Aligned Leaf Sampled @ 0.25",
        "slot_leaf_dense": "Slot-Aligned Leaf Dense",
        "slot_internal_count_only_r0p25": "Slot-Aligned Internal Count-Only @ 0.25",
        "slot_internal_full_sketch_r0p25": "Slot-Aligned Internal Full-Sketch @ 0.25",
        "slot_internal_full_sketch_r0p25__task_split_ablation": (
            "Slot-Aligned Internal Full-Sketch @ 0.25 (Task-Split Ablation)"
        ),
        "slot_internal_count_only_dense": "Slot-Aligned Internal Count-Only Dense",
        "slot_internal_full_sketch_dense": "Slot-Aligned Internal Full-Sketch Dense",
        "slot_internal_full_sketch_r0p5": "Slot-Aligned Internal Full-Sketch @ 0.5",
    }
    normalized = str(condition_id)
    if normalized in fixed_titles:
        return fixed_titles[normalized]
    if normalized.startswith("slot_leaf_sampled_r"):
        return f"Slot-Aligned Leaf Sampled @ {normalized.split('_r', 1)[1].replace('p', '.')}"
    if normalized.startswith("slot_internal_count_only_r"):
        return f"Slot-Aligned Internal Count-Only @ {normalized.split('_r', 1)[1].replace('p', '.')}"
    if normalized.startswith("slot_internal_full_sketch_r"):
        if normalized.endswith("__task_split_ablation"):
            base_id = normalized[: -len("__task_split_ablation")]
            return (
                f"{_exact_sanity_condition_title(base_id)} "
                "(Task-Split Ablation)"
            )
        return f"Slot-Aligned Internal Full-Sketch @ {normalized.split('_r', 1)[1].replace('p', '.')}"
    return normalized


def _exact_sanity_condition_summary(
    runs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not runs:
        return {}
    exemplar = dict(runs[0])
    failure_bucket_counts: Dict[str, int] = {}
    for run in runs:
        bucket = str(run.get("exact_sketch_failure_bucket", "")).strip()
        if bucket:
            failure_bucket_counts[bucket] = failure_bucket_counts.get(bucket, 0) + 1
    tree_neural: Dict[str, Any] = {}
    for split in ("train", "val", "test"):
        tree_neural[split] = {}
        for level in EXACT_SANITY_LEVELS:
            tree_neural[split][level] = {
                branch: {
                    metric: _exact_sanity_metric_summary(
                        runs,
                        (
                            "exact_sketch_diagnostics",
                            "tree_neural",
                            split,
                            level,
                            branch,
                            metric,
                        ),
                    )
                    for metric in EXACT_SANITY_COMPONENT_METRICS
                }
                for branch in ("direct", "probe")
            }
            if level == "merge":
                tree_neural[split][level]["decoded_consistency"] = {
                    metric: _exact_sanity_metric_summary(
                        runs,
                        (
                            "exact_sketch_diagnostics",
                            "tree_neural",
                            split,
                            level,
                            "decoded_consistency",
                            metric,
                        ),
                    )
                    for metric in EXACT_SANITY_MERGE_CONSISTENCY_METRICS
                }
    test_tree = dict(tree_neural.get("test") or {})
    test_leaf_probe = dict((test_tree.get("leaf") or {}).get("probe") or {})
    test_merge_probe = dict((test_tree.get("merge") or {}).get("probe") or {})
    test_root_direct = dict((test_tree.get("root") or {}).get("direct") or {})
    test_root_probe = dict((test_tree.get("root") or {}).get("probe") or {})
    test_merge_consistency = dict(
        ((test_tree.get("merge") or {}).get("decoded_consistency") or {})
    )
    condition_id = _exact_sanity_condition_id(exemplar)
    condition_kind = _exact_sanity_condition_kind(exemplar)
    return {
        "condition_id": condition_id,
        "condition_kind": condition_kind,
        "condition_title": _exact_sanity_condition_title(condition_id),
        "config_label": str(exemplar.get("config_label", "")),
        "n_runs": int(len(runs)),
        "seed_values": sorted(
            {int(run.get("seed", 0)) for run in runs if "seed" in run}
        ),
        "aligned_sketch_surface": str(exemplar.get("aligned_sketch_surface", "")),
        "weighting_scheme": str(exemplar.get("weighting_scheme", "")),
        "optimization_root_weight": float(
            exemplar.get("optimization_root_weight", float("nan"))
        ),
        "local_law_c1_weight": float(exemplar.get("local_law_c1_weight", float("nan"))),
        "local_law_c2_weight": float(exemplar.get("local_law_c2_weight", float("nan"))),
        "local_law_c3_weight": float(exemplar.get("local_law_c3_weight", float("nan"))),
        "summary_spec_name": str(exemplar.get("summary_spec_name", "")),
        "slot_count": int(exemplar.get("slot_count", 0)),
        "tree_theorem_count_dim": int(exemplar.get("tree_theorem_count_dim", 0)),
        "tree_theorem_first_dim": int(exemplar.get("tree_theorem_first_dim", 0)),
        "tree_theorem_last_dim": int(exemplar.get("tree_theorem_last_dim", 0)),
        "tree_theorem_count_head_mode": str(
            exemplar.get("tree_theorem_count_head_mode", "")
        ),
        "tree_theorem_count_ordinal_weight": float(
            exemplar.get("tree_theorem_count_ordinal_weight", 1.0)
        ),
        "tree_theorem_count_scalar_aux_weight": float(
            exemplar.get("tree_theorem_count_scalar_aux_weight", 0.25)
        ),
        "tree_theorem_count_threshold_balance": bool(
            exemplar.get("tree_theorem_count_threshold_balance", True)
        ),
        "leaf_supervision_kind": str(exemplar.get("leaf_supervision_kind", "")),
        "internal_supervision_kind": str(
            exemplar.get("internal_supervision_kind", "none")
        ),
        "internal_label_rate": float(exemplar.get("internal_label_rate", 0.0)),
        "leaf_exact_supervision": bool(exemplar.get("leaf_exact_supervision", False)),
        "leaf_label_rate": float(exemplar.get("leaf_label_rate", 1.0)),
        "tree_training_schedule": str(exemplar.get("tree_training_schedule", "")),
        "tree_stage1_epochs": int(exemplar.get("tree_stage1_epochs", 0)),
        "tree_stage2_epochs": int(exemplar.get("tree_stage2_epochs", 0)),
        "tree_root_supervision_kind": str(
            exemplar.get("tree_root_supervision_kind", "")
        ),
        "tree_checkpoint_metric": str(exemplar.get("tree_checkpoint_metric", "")),
        "tree_stage1_checkpoint_metric": str(
            exemplar.get("tree_stage1_checkpoint_metric", "")
        ),
        "tree_stage1_eval_mode": str(exemplar.get("tree_stage1_eval_mode", "")),
        "tree_stage1_screen_doc_limit": int(
            exemplar.get("tree_stage1_screen_doc_limit", 0)
        ),
        "tree_stage1_final_exact_doc_limit": int(
            exemplar.get("tree_stage1_final_exact_doc_limit", 0)
        ),
        "tree_stage1_artifact_dir": str(
            exemplar.get("tree_stage1_artifact_dir", "")
        ),
        "tree_stage1_root_weight": float(
            exemplar.get("tree_stage1_root_weight", 0.0)
        ),
        "tree_summary_spec_root_mode": str(
            exemplar.get("tree_summary_spec_root_mode", "")
        ),
        "failure_bucket_counts": dict(failure_bucket_counts),
        "failure_gap_scores": {
            "leaf_boundary_encoding_gap_score": finite_summary_stats(
                [run.get("exact_sketch_leaf_gap_score", float("nan")) for run in runs]
            ),
            "count_composition_gap_score": finite_summary_stats(
                [run.get("exact_sketch_merge_gap_score", float("nan")) for run in runs]
            ),
            "subtree_label_value_gap_score": finite_summary_stats(
                [
                    (
                        nested_mapping_value(
                            run,
                            (
                                "exact_sketch_diagnostics",
                                "failure_attribution",
                                "subtree_label_value_gap_score",
                            ),
                        )
                        if np.isfinite(
                            float(
                                nested_mapping_value(
                                    run,
                                    (
                                        "exact_sketch_diagnostics",
                                        "failure_attribution",
                                        "subtree_label_value_gap_score",
                                    ),
                                )
                            )
                        )
                        else nested_mapping_value(
                            run,
                            (
                                "exact_sketch_diagnostics",
                                "failure_attribution",
                                "internal_label_value_gap_score",
                            ),
                        )
                    )
                    for run in runs
                ]
            ),
            "legacy_readout_gap_score": finite_summary_stats(
                [run.get("exact_sketch_readout_gap_score", float("nan")) for run in runs]
            ),
        },
        "tree_neural": tree_neural,
        "acceptance_readout": {
            "test_probe_leaf_exact_summary_match_rate_mean": float(
                (test_leaf_probe.get("exact_summary_match_rate") or {}).get(
                    "mean",
                    float("nan"),
                )
            ),
            "test_probe_merge_exact_summary_match_rate_mean": float(
                (test_merge_probe.get("exact_summary_match_rate") or {}).get(
                    "mean",
                    float("nan"),
                )
            ),
            "test_direct_root_count_mae_mean": float(
                (test_root_direct.get("count_mae") or {}).get("mean", float("nan"))
            ),
            "test_task_root_mae_ablation_mean": float(
                _exact_sanity_metric_summary(
                    runs,
                    (
                        "exact_sketch_diagnostics",
                        "direct_selection_metrics",
                        "test",
                        "task_root_mae_ablation",
                    ),
                ).get("mean", float("nan"))
            ),
            "test_task_root_mae_mean": float(
                _exact_sanity_metric_summary(
                    runs,
                    (
                        "exact_sketch_diagnostics",
                        "direct_selection_metrics",
                        "test",
                        "task_root_mae",
                    ),
                ).get("mean", float("nan"))
            ),
            "test_probe_root_count_mae_mean": float(
                (test_root_probe.get("count_mae") or {}).get("mean", float("nan"))
            ),
            "test_merge_join_bit_accuracy_mean": float(
                (test_merge_consistency.get("merge_join_bit_accuracy") or {}).get(
                    "mean",
                    float("nan"),
                )
            ),
            "test_c2_on_range_exact_match_mean": float(
                _exact_sanity_metric_summary(
                    runs,
                    (
                        "exact_sketch_diagnostics",
                        "direct_selection_metrics",
                        "test",
                        "c2_on_range_exact_match",
                    ),
                ).get("mean", float("nan"))
            ),
            "test_theorem_bootstrap_direct_mean": float(
                _exact_sanity_metric_summary(
                    runs,
                    (
                        "exact_sketch_diagnostics",
                        "direct_selection_metrics",
                        "test",
                        "val_theorem_bootstrap_direct",
                    ),
                ).get("mean", float("nan"))
            ),
            "test_probe_merge_count_match_rate_mean": float(
                (test_merge_probe.get("count_match_rate") or {}).get(
                    "mean",
                    float("nan"),
                )
            ),
            "test_probe_merge_first_accuracy_mean": float(
                (test_merge_probe.get("first_accuracy") or {}).get(
                    "mean",
                    float("nan"),
                )
            ),
            "test_probe_merge_last_accuracy_mean": float(
                (test_merge_probe.get("last_accuracy") or {}).get(
                    "mean",
                    float("nan"),
                )
            ),
            "test_probe_merge_count_mae_mean": float(
                (test_merge_probe.get("count_mae") or {}).get(
                    "mean",
                    float("nan"),
                )
            ),
        },
    }


def _condition_acceptance_value(
    condition: Mapping[str, Any],
    key: str,
) -> float:
    return float(
        dict(condition.get("acceptance_readout") or {}).get(key, float("nan"))
    )


def tree_neural_exact_sanity_summary(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    all_runs = [
        dict(run)
        for run in list(payload.get("runs") or [])
        if str(run.get("study_name", "")).strip() == EXACT_SANITY_STUDY_NAME
    ]
    runs = [
        dict(run)
        for run in all_runs
        if str(run.get("baseline_family", "")) == EXACT_SANITY_FAMILY
        and isinstance(run.get("exact_sketch_diagnostics"), Mapping)
    ]
    if not runs:
        return {}

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(int(run.get("train_doc_count", 0)), []).append(run)

    groups: List[Dict[str, Any]] = []
    for train_doc_count in sorted(grouped):
        group_runs = list(grouped[train_doc_count])
        fno_reference_runs = [
            run
            for run in all_runs
            if int(run.get("train_doc_count", 0)) == int(train_doc_count)
            and str(run.get("baseline_family", "")) == "official_fno"
            and isinstance(run.get("root_summary_probe_audit"), Mapping)
        ]
        failure_bucket_counts: Dict[str, int] = {}
        for run in group_runs:
            bucket = str(run.get("exact_sketch_failure_bucket", "")).strip()
            if bucket:
                failure_bucket_counts[bucket] = failure_bucket_counts.get(bucket, 0) + 1
        runs_by_condition: Dict[str, List[Dict[str, Any]]] = {}
        for run in group_runs:
            runs_by_condition.setdefault(_exact_sanity_condition_id(run), []).append(run)
        exact_witness: Dict[str, Any] = {}
        for split in ("train", "val", "test"):
            exact_witness[split] = {
                "law_metrics": {
                    metric: _exact_sanity_metric_summary(
                        group_runs,
                        (
                            "exact_sketch_diagnostics",
                            "exact_witness",
                            split,
                            "law_metrics",
                            metric,
                        ),
                    )
                    for metric in EXACT_SANITY_LAW_METRICS
                }
            }
            for level in EXACT_SANITY_LEVELS:
                exact_witness[split][level] = {
                    "direct": {
                        metric: _exact_sanity_metric_summary(
                            group_runs,
                            (
                                "exact_sketch_diagnostics",
                                "exact_witness",
                                split,
                                level,
                                "direct",
                                metric,
                            ),
                        )
                        for metric in EXACT_SANITY_COMPONENT_METRICS
                    },
                    "probe_control": {
                        metric: _exact_sanity_metric_summary(
                            group_runs,
                            (
                                "exact_sketch_diagnostics",
                                "exact_witness",
                                split,
                                level,
                                "probe_control",
                                metric,
                            ),
                        )
                        for metric in EXACT_SANITY_COMPONENT_METRICS
                    },
                }
        conditions = [
            _exact_sanity_condition_summary(condition_runs)
            for _condition_id, condition_runs in sorted(runs_by_condition.items())
        ]
        condition_by_id = {
            str(condition.get("condition_id", "")): condition for condition in conditions
        }
        condition_by_kind = {
            str(condition.get("condition_kind", "")): condition for condition in conditions
        }
        exact_test_laws = exact_witness["test"]["law_metrics"]
        exact_witness_near_zero = all(
            abs(float(exact_test_laws[metric]["mean"])) <= 1e-9
            for metric in EXACT_SANITY_LAW_METRICS
            if np.isfinite(float(exact_test_laws[metric]["mean"]))
        )
        legacy_condition = condition_by_kind.get("legacy_fair_fno_root_only")
        slot_root_only = condition_by_kind.get("slot_root_only")
        slot_leaf_sampled = condition_by_kind.get("slot_leaf_sampled_r0p25")
        slot_leaf_dense = condition_by_kind.get("slot_leaf_dense")
        legacy_vs_slot_root_only: Dict[str, Any] = {}
        if legacy_condition is not None and slot_root_only is not None:
            legacy_vs_slot_root_only = {
                "merge_probe_exact_summary_match_rate_delta": float(
                    _condition_acceptance_value(
                        slot_root_only,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                    - _condition_acceptance_value(
                        legacy_condition,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                ),
                "leaf_probe_exact_summary_match_rate_delta": float(
                    _condition_acceptance_value(
                        slot_root_only,
                        "test_probe_leaf_exact_summary_match_rate_mean",
                    )
                    - _condition_acceptance_value(
                        legacy_condition,
                        "test_probe_leaf_exact_summary_match_rate_mean",
                    )
                ),
                "direct_root_count_mae_delta": float(
                    _condition_acceptance_value(
                        slot_root_only,
                        "test_direct_root_count_mae_mean",
                    )
                    - _condition_acceptance_value(
                        legacy_condition,
                        "test_direct_root_count_mae_mean",
                    )
                ),
                "slot_root_only_improves_over_legacy": bool(
                    np.isfinite(
                        _condition_acceptance_value(
                            slot_root_only,
                            "test_probe_merge_exact_summary_match_rate_mean",
                        )
                    )
                    and np.isfinite(
                        _condition_acceptance_value(
                            legacy_condition,
                            "test_probe_merge_exact_summary_match_rate_mean",
                        )
                    )
                    and np.isfinite(
                        _condition_acceptance_value(
                            slot_root_only,
                            "test_direct_root_count_mae_mean",
                        )
                    )
                    and np.isfinite(
                        _condition_acceptance_value(
                            legacy_condition,
                            "test_direct_root_count_mae_mean",
                        )
                    )
                    and _condition_acceptance_value(
                        slot_root_only,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                    >= _condition_acceptance_value(
                        legacy_condition,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                    and _condition_acceptance_value(
                        slot_root_only,
                        "test_direct_root_count_mae_mean",
                    )
                    <= _condition_acceptance_value(
                        legacy_condition,
                        "test_direct_root_count_mae_mean",
                    )
                ),
            }
        leaf_sampled_value: Dict[str, Any] = {}
        leaf_value_by_rate: Dict[str, Any] = {}
        for rate_label in ("0p25", "0p5", "0p75", "dense"):
            leaf_condition = condition_by_kind.get(
                "slot_leaf_dense"
                if rate_label == "dense"
                else f"slot_leaf_sampled_r{rate_label}"
            )
            if slot_root_only is None or leaf_condition is None:
                continue
            payload = {
                "merge_probe_exact_summary_match_rate_delta": float(
                    _condition_acceptance_value(
                        leaf_condition,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                    - _condition_acceptance_value(
                        slot_root_only,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                ),
                "leaf_probe_exact_summary_match_rate_delta": float(
                    _condition_acceptance_value(
                        leaf_condition,
                        "test_probe_leaf_exact_summary_match_rate_mean",
                    )
                    - _condition_acceptance_value(
                        slot_root_only,
                        "test_probe_leaf_exact_summary_match_rate_mean",
                    )
                ),
                "root_probe_count_mae_delta": float(
                    _condition_acceptance_value(
                        leaf_condition,
                        "test_probe_root_count_mae_mean",
                    )
                    - _condition_acceptance_value(
                        slot_root_only,
                        "test_probe_root_count_mae_mean",
                    )
                ),
                "leaf_rate_improves_over_root_only": bool(
                    np.isfinite(
                        _condition_acceptance_value(
                            leaf_condition,
                            "test_probe_merge_exact_summary_match_rate_mean",
                        )
                    )
                    and np.isfinite(
                        _condition_acceptance_value(
                            slot_root_only,
                            "test_probe_merge_exact_summary_match_rate_mean",
                        )
                    )
                    and np.isfinite(
                        _condition_acceptance_value(
                            leaf_condition,
                            "test_probe_root_count_mae_mean",
                        )
                    )
                    and np.isfinite(
                        _condition_acceptance_value(
                            slot_root_only,
                            "test_probe_root_count_mae_mean",
                        )
                    )
                    and _condition_acceptance_value(
                        leaf_condition,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                    >= _condition_acceptance_value(
                        slot_root_only,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                    and _condition_acceptance_value(
                        leaf_condition,
                        "test_probe_root_count_mae_mean",
                    )
                    <= _condition_acceptance_value(
                        slot_root_only,
                        "test_probe_root_count_mae_mean",
                    )
                ),
            }
            leaf_value_by_rate[rate_label] = payload
            if rate_label == "0p25":
                leaf_sampled_value = dict(payload)
        dense_leaf_value: Dict[str, Any] = {}
        if slot_leaf_sampled is not None and slot_leaf_dense is not None:
            dense_leaf_value = {
                "merge_probe_exact_summary_match_rate_delta": float(
                    _condition_acceptance_value(
                        slot_leaf_dense,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                    - _condition_acceptance_value(
                        slot_leaf_sampled,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                ),
                "root_probe_count_mae_delta": float(
                    _condition_acceptance_value(
                        slot_leaf_dense,
                        "test_probe_root_count_mae_mean",
                    )
                    - _condition_acceptance_value(
                        slot_leaf_sampled,
                        "test_probe_root_count_mae_mean",
                    )
                ),
                "leaf_dense_improves_over_leaf_sampled": bool(
                    np.isfinite(
                        _condition_acceptance_value(
                            slot_leaf_dense,
                            "test_probe_merge_exact_summary_match_rate_mean",
                        )
                    )
                    and np.isfinite(
                        _condition_acceptance_value(
                            slot_leaf_sampled,
                            "test_probe_merge_exact_summary_match_rate_mean",
                        )
                    )
                    and np.isfinite(
                        _condition_acceptance_value(
                            slot_leaf_dense,
                            "test_probe_root_count_mae_mean",
                        )
                    )
                    and np.isfinite(
                        _condition_acceptance_value(
                            slot_leaf_sampled,
                            "test_probe_root_count_mae_mean",
                        )
                    )
                    and _condition_acceptance_value(
                        slot_leaf_dense,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                    >= _condition_acceptance_value(
                        slot_leaf_sampled,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                    and _condition_acceptance_value(
                        slot_leaf_dense,
                        "test_probe_root_count_mae_mean",
                    )
                    <= _condition_acceptance_value(
                        slot_leaf_sampled,
                        "test_probe_root_count_mae_mean",
                    )
                ),
            }
        subtree_label_value_by_rate: Dict[str, Any] = {}
        for rate_label in ("0p25", "0p5", "0p75", "dense"):
            count_condition = condition_by_kind.get(
                "slot_internal_count_only_dense"
                if rate_label == "dense"
                else f"slot_internal_count_only_r{rate_label}"
            )
            full_condition = condition_by_kind.get(
                "slot_internal_full_sketch_dense"
                if rate_label == "dense"
                else f"slot_internal_full_sketch_r{rate_label}"
            )
            if count_condition is None or full_condition is None:
                continue
            subtree_label_value_by_rate[rate_label] = {
                "merge_probe_exact_summary_match_rate_delta": float(
                    _condition_acceptance_value(
                        full_condition,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                    - _condition_acceptance_value(
                        count_condition,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                ),
                "merge_join_bit_accuracy_delta": float(
                    _condition_acceptance_value(
                        full_condition,
                        "test_merge_join_bit_accuracy_mean",
                    )
                    - _condition_acceptance_value(
                        count_condition,
                        "test_merge_join_bit_accuracy_mean",
                    )
                ),
                "direct_root_count_mae_delta": float(
                    _condition_acceptance_value(
                        full_condition,
                        "test_direct_root_count_mae_mean",
                    )
                    - _condition_acceptance_value(
                        count_condition,
                        "test_direct_root_count_mae_mean",
                    )
                ),
                "full_sketch_improves_over_count_only": bool(
                    np.isfinite(
                        _condition_acceptance_value(
                            full_condition,
                            "test_probe_merge_exact_summary_match_rate_mean",
                        )
                    )
                    and np.isfinite(
                        _condition_acceptance_value(
                            count_condition,
                            "test_probe_merge_exact_summary_match_rate_mean",
                        )
                    )
                    and np.isfinite(
                        _condition_acceptance_value(
                            full_condition,
                            "test_direct_root_count_mae_mean",
                        )
                    )
                    and np.isfinite(
                        _condition_acceptance_value(
                            count_condition,
                            "test_direct_root_count_mae_mean",
                        )
                    )
                    and _condition_acceptance_value(
                        full_condition,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                    >= _condition_acceptance_value(
                        count_condition,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                    and _condition_acceptance_value(
                        full_condition,
                        "test_direct_root_count_mae_mean",
                    )
                    <= _condition_acceptance_value(
                        count_condition,
                        "test_direct_root_count_mae_mean",
                    )
                ),
            }
        root_mode_alignment_by_base_config: Dict[str, Any] = {}
        for condition in conditions:
            condition_id = str(condition.get("condition_id", ""))
            if not condition_id.endswith("_task_split_ablation"):
                continue
            base_condition_id = condition_id[: -len("_task_split_ablation")]
            primary_condition = condition_by_id.get(base_condition_id)
            if primary_condition is None:
                continue
            root_mode_alignment_by_base_config[base_condition_id] = {
                "aligned_primary_condition_id": base_condition_id,
                "theorem_primary_condition_id": base_condition_id,
                "task_split_ablation_condition_id": condition_id,
                "aligned_primary_root_mode": str(
                    primary_condition.get("tree_summary_spec_root_mode", "")
                ),
                "theorem_primary_root_mode": str(
                    primary_condition.get("tree_summary_spec_root_mode", "")
                ),
                "task_split_root_mode": str(
                    condition.get("tree_summary_spec_root_mode", "")
                ),
                "theorem_root_count_mae_delta": float(
                    _condition_acceptance_value(
                        primary_condition,
                        "test_direct_root_count_mae_mean",
                    )
                    - _condition_acceptance_value(
                        condition,
                        "test_direct_root_count_mae_mean",
                    )
                ),
                "task_root_mae_ablation_delta": float(
                    _condition_acceptance_value(
                        primary_condition,
                        "test_task_root_mae_ablation_mean",
                    )
                    - _condition_acceptance_value(
                        condition,
                        "test_task_root_mae_ablation_mean",
                    )
                ),
                "merge_probe_exact_summary_match_rate_delta": float(
                    _condition_acceptance_value(
                        primary_condition,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                    - _condition_acceptance_value(
                        condition,
                        "test_probe_merge_exact_summary_match_rate_mean",
                    )
                ),
                "aligned_primary_improves_or_matches_theorem_root": bool(
                    np.isfinite(
                        _condition_acceptance_value(
                            primary_condition,
                            "test_direct_root_count_mae_mean",
                        )
                    )
                    and np.isfinite(
                        _condition_acceptance_value(
                            condition,
                            "test_direct_root_count_mae_mean",
                        )
                    )
                    and _condition_acceptance_value(
                        primary_condition,
                        "test_direct_root_count_mae_mean",
                    )
                    <= _condition_acceptance_value(
                        condition,
                        "test_direct_root_count_mae_mean",
                    )
                ),
            }
        groups.append(
            {
                "train_doc_count": int(train_doc_count),
                "n_runs": int(len(group_runs)),
                "seed_values": sorted(
                    {int(run.get("seed", 0)) for run in group_runs if "seed" in run}
                ),
                "config_labels": sorted(
                    {
                        str(run.get("config_label", "")).strip()
                        for run in group_runs
                        if str(run.get("config_label", "")).strip()
                    }
                ),
                "failure_bucket_counts": dict(failure_bucket_counts),
                "failure_gap_scores": {
                    "leaf_boundary_encoding_gap_score": finite_summary_stats(
                        [
                            run.get("exact_sketch_leaf_gap_score", float("nan"))
                            for run in group_runs
                        ]
                    ),
                    "count_composition_gap_score": finite_summary_stats(
                        [
                            run.get("exact_sketch_merge_gap_score", float("nan"))
                            for run in group_runs
                        ]
                    ),
                    "subtree_label_value_gap_score": finite_summary_stats(
                        [
                            (
                                nested_mapping_value(
                                    run,
                                    (
                                        "exact_sketch_diagnostics",
                                        "failure_attribution",
                                        "subtree_label_value_gap_score",
                                    ),
                                )
                                if np.isfinite(
                                    float(
                                        nested_mapping_value(
                                            run,
                                            (
                                                "exact_sketch_diagnostics",
                                                "failure_attribution",
                                                "subtree_label_value_gap_score",
                                            ),
                                        )
                                    )
                                )
                                else nested_mapping_value(
                                    run,
                                    (
                                        "exact_sketch_diagnostics",
                                        "failure_attribution",
                                        "internal_label_value_gap_score",
                                    ),
                                )
                            )
                            for run in group_runs
                        ]
                    ),
                    "legacy_readout_gap_score": finite_summary_stats(
                        [
                            run.get("exact_sketch_readout_gap_score", float("nan"))
                            for run in group_runs
                        ]
                    ),
                },
                "exact_witness": exact_witness,
                "conditions": conditions,
                "full_doc_fno_reference": {
                    split: {
                        metric: _exact_sanity_metric_summary(
                            fno_reference_runs,
                            ("root_summary_probe_audit", split, metric),
                        )
                        for metric in EXACT_SANITY_COMPONENT_METRICS
                    }
                    for split in ("train", "val", "test")
                }
                if fno_reference_runs
                else {},
                "acceptance_readout": {
                    "exact_witness_test_laws_near_zero": bool(exact_witness_near_zero),
                    "legacy_vs_slot_root_only": legacy_vs_slot_root_only,
                    "leaf_sampled_value": leaf_sampled_value,
                    "leaf_value_by_rate": leaf_value_by_rate,
                    "dense_leaf_value": dense_leaf_value,
                    "subtree_label_value_by_rate": subtree_label_value_by_rate,
                    "root_mode_alignment_by_base_config": root_mode_alignment_by_base_config,
                },
            }
        )
    return {
        "study_name": EXACT_SANITY_STUDY_NAME,
        "benchmark": str(payload.get("benchmark", "")),
        "baseline_family": EXACT_SANITY_FAMILY,
        "primary_question": (
            "Can tree_neural fair root-only recover the Lean-style exact sketch "
            "(count, first, last) well enough that the local-law gap is attributable?"
        ),
        "paper_to_lean_local_law_mapping": {
            "C1": "L1",
            "C2": "L3",
            "C3": "L2",
        },
        "theorem_contract": dict(
            ((runs[0].get("exact_sketch_diagnostics") or {}).get("theorem_contract") or {})
        ),
        "groups": groups,
    }


def render_exact_sanity_summary_markdown(
    summary: Mapping[str, Any],
) -> str:
    groups = list(summary.get("groups") or [])
    lines = [
        "# Tree-Neural Exact-Sketch Sanity Summary",
        "",
        f"- benchmark: `{str(summary.get('benchmark', ''))}`",
        f"- baseline_family: `{str(summary.get('baseline_family', ''))}`",
        f"- study_name: `{str(summary.get('study_name', ''))}`",
        f"- primary_question: `{str(summary.get('primary_question', ''))}`",
        f"- paper_to_lean_local_law_mapping: `{dict(summary.get('paper_to_lean_local_law_mapping') or {})}`",
        f"- theorem_contract: `{dict(summary.get('theorem_contract') or {})}`",
    ]
    for group in groups:
        acceptance = dict(group.get("acceptance_readout") or {})
        exact_witness = dict(group.get("exact_witness") or {})
        fno_reference = dict(group.get("full_doc_fno_reference") or {})
        witness_test_laws = dict((exact_witness.get("test") or {}).get("law_metrics") or {})
        lines.extend(
            [
                "",
                f"## train_doc_count = {int(group.get('train_doc_count', 0))}",
                "",
                f"- n_runs: `{int(group.get('n_runs', 0))}`",
                f"- seeds: `{list(group.get('seed_values') or [])}`",
                f"- config_labels: `{list(group.get('config_labels') or [])}`",
                f"- failure_bucket_counts: `{dict(group.get('failure_bucket_counts') or {})}`",
                (
                    "- exact witness test laws near zero: "
                    f"`{bool(acceptance.get('exact_witness_test_laws_near_zero', False))}`"
                ),
                "",
                "### Exact Witness Test Laws",
                "",
                "| metric | mean | std |",
                "|---|---:|---:|",
            ]
        )
        for metric in EXACT_SANITY_LAW_METRICS:
            stats = dict(witness_test_laws.get(metric) or {})
            lines.append(
                "| "
                f"{metric} | "
                f"{float(stats.get('mean', float('nan'))):.6g} | "
                f"{float(stats.get('std', float('nan'))):.6g} |"
            )
        for condition in list(group.get("conditions") or []):
            condition = dict(condition)
            tree_test = dict((condition.get("tree_neural") or {}).get("test") or {})
            condition_acceptance = dict(condition.get("acceptance_readout") or {})
            lines.extend(
                [
                    "",
                    f"### {str(condition.get('condition_title', 'Condition'))}",
                    "",
                    f"- condition_id: `{str(condition.get('condition_id', ''))}`",
                    f"- condition_kind: `{str(condition.get('condition_kind', ''))}`",
                    f"- config_label: `{str(condition.get('config_label', ''))}`",
                    f"- aligned_sketch_surface: `{str(condition.get('aligned_sketch_surface', ''))}`",
                    f"- weighting_scheme: `{str(condition.get('weighting_scheme', ''))}`",
                    f"- optimization_root_weight: `{float(condition.get('optimization_root_weight', float('nan'))):.6g}`",
                    f"- local_law_c1_weight: `{float(condition.get('local_law_c1_weight', float('nan'))):.6g}`",
                    f"- local_law_c2_weight: `{float(condition.get('local_law_c2_weight', float('nan'))):.6g}`",
                    f"- local_law_c3_weight: `{float(condition.get('local_law_c3_weight', float('nan'))):.6g}`",
                    f"- summary_spec_name: `{str(condition.get('summary_spec_name', ''))}`",
                    f"- slot_count: `{int(condition.get('slot_count', 0))}`",
                    f"- tree_theorem_count_dim: `{int(condition.get('tree_theorem_count_dim', 0))}`",
                    f"- tree_theorem_first_dim: `{int(condition.get('tree_theorem_first_dim', 0))}`",
                    f"- tree_theorem_last_dim: `{int(condition.get('tree_theorem_last_dim', 0))}`",
                    f"- tree_theorem_count_head_mode: `{str(condition.get('tree_theorem_count_head_mode', ''))}`",
                    f"- tree_theorem_count_ordinal_weight: `{float(condition.get('tree_theorem_count_ordinal_weight', 1.0)):.6g}`",
                    f"- tree_theorem_count_scalar_aux_weight: `{float(condition.get('tree_theorem_count_scalar_aux_weight', 0.25)):.6g}`",
                    f"- tree_theorem_count_threshold_balance: `{bool(condition.get('tree_theorem_count_threshold_balance', True))}`",
                    f"- leaf_supervision_kind: `{str(condition.get('leaf_supervision_kind', ''))}`",
                    f"- internal_supervision_kind: `{str(condition.get('internal_supervision_kind', ''))}`",
                    f"- internal_label_rate: `{float(condition.get('internal_label_rate', 0.0)):.6g}`",
                    f"- leaf_exact_supervision: `{bool(condition.get('leaf_exact_supervision', False))}`",
                    f"- leaf_label_rate: `{float(condition.get('leaf_label_rate', 1.0)):.6g}`",
                    f"- tree_training_schedule: `{str(condition.get('tree_training_schedule', ''))}`",
                    f"- tree_stage1_epochs: `{int(condition.get('tree_stage1_epochs', 0))}`",
                    f"- tree_stage2_epochs: `{int(condition.get('tree_stage2_epochs', 0))}`",
                    f"- tree_root_supervision_kind: `{str(condition.get('tree_root_supervision_kind', ''))}`",
                    f"- tree_checkpoint_metric: `{str(condition.get('tree_checkpoint_metric', ''))}`",
                    f"- tree_stage1_checkpoint_metric: `{str(condition.get('tree_stage1_checkpoint_metric', ''))}`",
                    f"- tree_stage1_root_weight: `{float(condition.get('tree_stage1_root_weight', 0.0)):.6g}`",
                    f"- tree_task_head_mode: `{str(condition.get('tree_task_head_mode', ''))}`",
                    f"- tree_theorem_surface_mode: `{str(condition.get('tree_theorem_surface_mode', ''))}`",
                    f"- tree_summary_spec_root_mode: `{str(condition.get('tree_summary_spec_root_mode', ''))}`",
                    f"- failure_bucket_counts: `{dict(condition.get('failure_bucket_counts') or {})}`",
                    "",
                    "| level | branch | count_mae | count_match | first_acc | last_acc | exact_match |",
                    "|---|---|---:|---:|---:|---:|---:|",
                ]
            )
            for level in EXACT_SANITY_LEVELS:
                level_payload = dict(tree_test.get(level) or {})
                for branch in ("direct", "probe"):
                    branch_payload = dict(level_payload.get(branch) or {})
                    lines.append(
                        "| "
                        f"{level} | "
                        f"{branch} | "
                        f"{float((branch_payload.get('count_mae') or {}).get('mean', float('nan'))):.6g} | "
                        f"{float((branch_payload.get('count_match_rate') or {}).get('mean', float('nan'))):.6g} | "
                        f"{float((branch_payload.get('first_accuracy') or {}).get('mean', float('nan'))):.6g} | "
                        f"{float((branch_payload.get('last_accuracy') or {}).get('mean', float('nan'))):.6g} | "
                        f"{float((branch_payload.get('exact_summary_match_rate') or {}).get('mean', float('nan'))):.6g} |"
                    )
            merge_consistency = dict((tree_test.get("merge") or {}).get("decoded_consistency") or {})
            if merge_consistency:
                lines.extend(
                    [
                        "",
                        "| merge_consistency_metric | mean | std |",
                        "|---|---:|---:|",
                    ]
                )
                for metric in EXACT_SANITY_MERGE_CONSISTENCY_METRICS:
                    stats = dict(merge_consistency.get(metric) or {})
                    lines.append(
                        "| "
                        f"{metric} | "
                        f"{float(stats.get('mean', float('nan'))):.6g} | "
                        f"{float(stats.get('std', float('nan'))):.6g} |"
                    )
            lines.extend(
                [
                    "",
                    f"- test probe leaf exact summary match rate mean: `{float(condition_acceptance.get('test_probe_leaf_exact_summary_match_rate_mean', float('nan'))):.6g}`",
                    f"- test probe merge exact summary match rate mean: `{float(condition_acceptance.get('test_probe_merge_exact_summary_match_rate_mean', float('nan'))):.6g}`",
                    f"- test theorem root direct count mae mean: `{float(condition_acceptance.get('test_direct_root_count_mae_mean', float('nan'))):.6g}`",
                    f"- test task root mae ablation mean: `{float(condition_acceptance.get('test_task_root_mae_ablation_mean', float('nan'))):.6g}`",
                    f"- test probe root count mae mean: `{float(condition_acceptance.get('test_probe_root_count_mae_mean', float('nan'))):.6g}`",
                    f"- test merge join bit accuracy mean: `{float(condition_acceptance.get('test_merge_join_bit_accuracy_mean', float('nan'))):.6g}`",
                    f"- test C2/L3 on-range exact match mean: `{float(condition_acceptance.get('test_c2_on_range_exact_match_mean', float('nan'))):.6g}`",
                    f"- test theorem bootstrap direct mean: `{float(condition_acceptance.get('test_theorem_bootstrap_direct_mean', float('nan'))):.6g}`",
                ]
            )
        if fno_reference:
            ref_test = dict(fno_reference.get("test") or {})
            lines.extend(
                [
                    "",
                    "### Full-Doc FNO Root Probe Reference",
                    "",
                    "| metric | mean | std |",
                    "|---|---:|---:|",
                ]
            )
            for metric in EXACT_SANITY_COMPONENT_METRICS:
                stats = dict(ref_test.get(metric) or {})
                lines.append(
                    "| "
                    f"{metric} | "
                    f"{float(stats.get('mean', float('nan'))):.6g} | "
                    f"{float(stats.get('std', float('nan'))):.6g} |"
                )
        lines.extend(
            [
                "",
                "### Acceptance Readout",
                "",
                f"- legacy_vs_slot_root_only: `{dict(acceptance.get('legacy_vs_slot_root_only') or {})}`",
                f"- leaf_sampled_value: `{dict(acceptance.get('leaf_sampled_value') or {})}`",
                f"- leaf_value_by_rate: `{dict(acceptance.get('leaf_value_by_rate') or {})}`",
                f"- dense_leaf_value: `{dict(acceptance.get('dense_leaf_value') or {})}`",
                f"- subtree_label_value_by_rate: `{dict(acceptance.get('subtree_label_value_by_rate') or {})}`",
                f"- root_mode_alignment_by_base_config: `{dict(acceptance.get('root_mode_alignment_by_base_config') or {})}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "EXACT_SANITY_COMPONENT_METRICS",
    "EXACT_SANITY_FAMILY",
    "EXACT_SANITY_LAW_METRICS",
    "EXACT_SANITY_LEVELS",
    "EXACT_SANITY_MERGE_CONSISTENCY_METRICS",
    "EXACT_SANITY_STUDY_NAME",
    "finite_summary_stats",
    "nested_mapping_value",
    "render_exact_sanity_summary_markdown",
    "tree_neural_exact_sanity_summary",
]
