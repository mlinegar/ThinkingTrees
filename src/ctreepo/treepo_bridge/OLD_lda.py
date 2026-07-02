# OLD_: archived 2026-07-02; depends on the treepo method registry (register_method/list_methods), removed in the treepo 2026-06 minimization. The sim CLI sweeps under src/ctreepo/sim/cli cover LDA runs. Kept for reference; do not import or run.
"""LDA adapter for the standalone ``treepo`` package.

ThinkingTrees owns the LDA simulator. The standalone package owns the small
method/benchmark contracts. This module connects the two without moving LDA
into ``treepo``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


LDA_METHOD = "lda"
LDA_BENCHMARK = "lda"
LDA_SCORER = "lda"

_CONFIG_KEYS: tuple[str, ...] = (
    "n_topics",
    "vocab_size",
    "min_tokens",
    "max_tokens",
    "doc_topic_concentration",
    "topic_concentration",
    "emission_mode",
    "anchor_words_per_topic",
    "anchor_multiplier",
    "relevant_topics",
    "theta_scale",
    "zero_diagonal",
    "lambda_multiplier",
    "quadratic_utility_weight",
    "leaf_tokens",
    "train_docs",
    "test_docs",
    "inference_prior_mass",
    "inference_max_iter",
    "inference_tol",
    "full_hidden_dim",
    "full_n_layers",
    "state_dim",
    "supervise_all_balanced_nodes",
    "n_epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "device",
    "cuda_device",
    "torch_threads",
    "seed",
)

_METHOD_CONFIG_KEYS: tuple[str, ...] = (*_CONFIG_KEYS, "output_dir")

_DEFAULT_TASK_CONFIG: dict[str, Any] = {
    "n_topics": 4,
    "vocab_size": 64,
    "min_tokens": 64,
    "max_tokens": 64,
    "doc_topic_concentration": 0.6,
    "topic_concentration": 0.2,
    "emission_mode": "anchored",
    "anchor_words_per_topic": 4,
    "anchor_multiplier": 20.0,
    "relevant_topics": 2,
    "theta_scale": 1.0,
    "zero_diagonal": False,
    "quadratic_utility_weight": 1.0,
    "leaf_tokens": 8,
    "train_docs": 16,
    "test_docs": 8,
    "inference_prior_mass": 0.25,
    "inference_max_iter": 40,
    "inference_tol": 1e-6,
    "full_hidden_dim": 32,
    "full_n_layers": 1,
    "state_dim": 16,
    "supervise_all_balanced_nodes": True,
    "n_epochs": 2,
    "batch_size": 8,
    "lr": 3e-3,
    "weight_decay": 1e-5,
    "device": "cpu",
    "cuda_device": None,
    "torch_threads": 1,
}


def register_lda_method() -> str:
    """Register the ThinkingTrees LDA method with ``treepo``."""

    from treepo.methods import list_methods, register_method

    if LDA_METHOD not in set(list_methods()):
        register_method(
            LDA_METHOD,
            _run_lda_method,
            allowed_config_keys=_METHOD_CONFIG_KEYS,
        )
    return LDA_METHOD


def register_lda_benchmark() -> str:
    """Register the LDA benchmark with ``treepo``."""

    register_lda_method()
    from treepo.bench.tasks import (
        TaskBenchmarkSpec,
        list_task_benchmarks,
        register_task_benchmark,
    )

    if LDA_BENCHMARK not in set(list_task_benchmarks()):
        register_task_benchmark(
            TaskBenchmarkSpec(
                name=LDA_BENCHMARK,
                default_method=LDA_METHOD,
                default_scorer=LDA_SCORER,
                supported_scorers=(LDA_SCORER,),
                default_task_config=dict(_DEFAULT_TASK_CONFIG),
                allowed_task_config_keys=_CONFIG_KEYS,
                build_method_config=_build_lda_method_config,
            )
        )
    return LDA_BENCHMARK


def run_lda_benchmark(
    *,
    config: Mapping[str, Any],
    json_out: str | Path,
    csv_out: str | Path,
    print_json: bool = False,
) -> dict[str, object]:
    """Run the registered LDA benchmark through ``treepo.bench.runner``."""

    register_lda_benchmark()
    from treepo.bench.runner import run_single

    return run_single(
        experiment=LDA_BENCHMARK,
        config=dict(config),
        json_out=Path(json_out),
        csv_out=Path(csv_out),
        print_json=bool(print_json),
    )


def _build_lda_method_config(
    config: Any,
    task_config: Mapping[str, Any],
    scorer: str,
    output_dir: Path | None,
) -> dict[str, Any]:
    method_config = dict(task_config)
    method_config.update(dict(getattr(config, "method_config", {}) or {}))
    method_config.setdefault("seed", int(getattr(config, "seed", 0)))
    if output_dir is not None:
        method_config["output_dir"] = str(output_dir)
    return method_config


def _run_lda_method(config: Mapping[str, Any]) -> dict[str, Any]:
    from src.ctreepo.sim.core.lda_tree_recovery_learned import (
        LDATreeRecoveryLearnedConfig,
        run_lda_tree_recovery_learned_experiment,
    )

    summary = run_lda_tree_recovery_learned_experiment(
        LDATreeRecoveryLearnedConfig(**_config_kwargs(config))
    )
    payload = json.loads(summary.to_json())
    manifest_path: str | None = None
    if config.get("output_dir"):
        out = Path(str(config["output_dir"]))
        out.mkdir(parents=True, exist_ok=True)
        path = out / "lda_summary.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        manifest_path = str(path)
    return {
        "status": "success",
        "method": LDA_METHOD,
        "metrics": _metrics_from_payload(payload),
        "summary": _compact_summary(payload),
        "manifest_path": manifest_path,
        "artifacts": {"summary_json": manifest_path},
    }


def _config_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = {str(key): value for key, value in dict(config or {}).items()}
    payload.pop("output_dir", None)
    if "quadratic_utility_weight" in payload:
        payload["lambda_multiplier"] = payload.pop("quadratic_utility_weight")
    return payload


def _compact_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    learning = dict(payload.get("learning") or {})
    full_diag = dict(learning.get("full_doc_operator") or {})
    tree_diag = dict(learning.get("tree_svd_sketch") or {})
    return {
        "config": dict(payload.get("config") or {}),
        "full_doc_operator": {
            "device": str(full_diag.get("device") or ""),
            "train_loss_final": full_diag.get("train_loss_final"),
            "val_loss_final": full_diag.get("val_loss_final"),
        },
        "tree_svd_sketch": {
            "state_dim": tree_diag.get("state_dim"),
            "train_rank": tree_diag.get("train_rank"),
            "exact_train_manifold_representable": tree_diag.get(
                "exact_train_manifold_representable"
            ),
        },
    }


def _metrics_from_payload(payload: Mapping[str, Any]) -> dict[str, float]:
    methods = dict(payload.get("methods") or {})
    exact = dict((payload.get("exact_reference") or {}).get("exact_recovery") or {})
    learning = dict(payload.get("learning") or {})
    full_diag = dict(learning.get("full_doc_operator") or {})
    tree_diag = dict(learning.get("tree_svd_sketch") or {})
    full = dict(methods.get("full_doc_operator") or {})
    tree = dict(methods.get("tree_svd_sketch") or {})
    device = str(full_diag.get("device") or "")
    metrics = {
        "n_train": learning.get("train_docs_requested"),
        "n_test": learning.get("test_docs_requested"),
        "device_is_cuda": 1.0 if device.startswith("cuda") else 0.0,
        "exact_root_count_l1_mean": exact.get("root_count_l1_mean"),
        "exact_root_pi_l1_mean": exact.get("root_pi_l1_mean"),
        "exact_root_utility_abs_mean": exact.get("root_utility_abs_mean"),
        "full_doc_operator_pi_l1_to_true_mean": full.get("pi_l1_to_true_mean"),
        "full_doc_operator_pi_l1_to_full_mean": full.get("pi_l1_to_full_mean"),
        "full_doc_operator_utility_abs_to_true_mean": full.get(
            "utility_abs_to_true_mean"
        ),
        "full_doc_operator_utility_abs_to_full_mean": full.get(
            "utility_abs_to_full_mean"
        ),
        "full_doc_operator_log_likelihood_abs_to_full_mean": full.get(
            "log_likelihood_abs_to_full_mean"
        ),
        "tree_svd_sketch_count_l1_to_full_mean": tree.get("count_l1_to_full_mean"),
        "tree_svd_sketch_node_count_l1_mean": tree.get("node_count_l1_mean"),
        "tree_svd_sketch_pi_l1_to_true_mean": tree.get("pi_l1_to_true_mean"),
        "tree_svd_sketch_pi_l1_to_full_mean": tree.get("pi_l1_to_full_mean"),
        "tree_svd_sketch_utility_abs_to_true_mean": tree.get(
            "utility_abs_to_true_mean"
        ),
        "tree_svd_sketch_utility_abs_to_full_mean": tree.get(
            "utility_abs_to_full_mean"
        ),
        "tree_svd_train_rank": tree_diag.get("train_rank"),
        "tree_svd_explained_variance_ratio_train": tree_diag.get(
            "explained_variance_ratio_train"
        ),
    }
    return {key: _float_or_nan(value) for key, value in metrics.items()}


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


__all__ = [
    "LDA_BENCHMARK",
    "LDA_METHOD",
    "LDA_SCORER",
    "register_lda_benchmark",
    "register_lda_method",
    "run_lda_benchmark",
]
