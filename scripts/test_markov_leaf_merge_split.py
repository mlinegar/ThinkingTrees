#!/usr/bin/env python3
"""Split Markov tree-neural failure into leaf codec vs merge propagation.

This is a narrow diagnostic for the recoverable Markov setting:

1. learned leaves + learned merge
2. learned leaves + exact Markov merge over decoded leaves
3. exact theorem leaves + learned merge
4. exact theorem leaves + exact Markov merge

The third row is the missing isolation case for the current C-TreePO Markov
ladder: can the learned pipeline merger propagate the exact DGP state once the
leaf state bottleneck is removed?
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from src.ctreepo.sim.core.full_doc_anchor_diagnostics import (  # noqa: E402
    _load_fno_docs,
    prepare_markov_full_doc_anchor_diagnostics_data,
)
from src.ctreepo.sim.core.markov_neural_operator_baselines import (  # noqa: E402
    FNOCountSketch,
    _FNOCountDoc,
    _eval_fno_exact_sketch_direct_metrics,
    _set_global_seed,
    train_fno_tree,
)
from test_markov_exact_progression import (  # noqa: E402
    _evaluate_exact_leaf_merger,
    _leaf_summary_batch,
    _root_support_max,
    _validate_uniform_leaf_shape,
)


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _metric_subset(metrics: Mapping[str, Any], keys: Sequence[str]) -> Dict[str, Any]:
    return {str(key): _jsonable(metrics.get(str(key))) for key in keys if str(key) in metrics}


def _load_prepared_docs(
    *,
    benchmark: str,
    train_docs: int,
    seed: int,
) -> tuple[tuple[_FNOCountDoc, ...], tuple[_FNOCountDoc, ...], tuple[_FNOCountDoc, ...], Dict[str, Any]]:
    prepared_payload = prepare_markov_full_doc_anchor_diagnostics_data(
        benchmark_name=str(benchmark),
        seeds=(int(seed),),
        train_doc_counts=(int(train_docs),),
        use_cuda=False,
        torch_threads=1,
    )
    prepared = dict(prepared_payload["prepared"][0])
    train_all = _load_fno_docs(Path(str(prepared["train_fno_docs_json"])))
    val_docs = _load_fno_docs(Path(str(prepared["val_fno_docs_json"])))
    test_docs = _load_fno_docs(Path(str(prepared["test_fno_docs_json"])))
    return tuple(train_all[: int(train_docs)]), tuple(val_docs), tuple(test_docs), prepared


def _vocab_size(docs: Sequence[_FNOCountDoc]) -> int:
    max_token = 0
    for doc in docs:
        for leaf in doc.leaf_token_ids:
            for token in leaf:
                max_token = max(max_token, int(token))
    return int(max_token + 1)


def _root_class_values(docs: Sequence[_FNOCountDoc]) -> tuple[int, ...]:
    max_root = max((int(round(float(doc.root_count))) for doc in docs), default=0)
    return tuple(range(max(0, int(max_root)) + 1)) or (0,)


def _build_model(
    *,
    all_docs: Sequence[_FNOCountDoc],
    train_docs: Sequence[_FNOCountDoc],
    args: argparse.Namespace,
    device: torch.device,
) -> FNOCountSketch:
    _n_leaves, n_regimes = _validate_uniform_leaf_shape(all_docs)
    root_class_values = _root_class_values(all_docs)
    model = FNOCountSketch(
        vocab_size=_vocab_size(all_docs),
        leaf_tokens=int(len(all_docs[0].leaf_token_ids[0])),
        state_dim=int(args.state_dim),
        hidden_dim=int(args.hidden_dim),
        target_scale=float(_root_support_max(train_docs)),
        n_regimes=int(n_regimes),
        doc_sequence_class_values=root_class_values,
        root_count_class_values=root_class_values,
        fno_width=int(args.tree_leaf_fno_width),
        fno_n_modes=int(args.tree_leaf_fno_n_modes),
        fno_n_layers=int(args.tree_leaf_fno_n_layers),
        root_supervision_kind="mse",
        aligned_sketch_surface="",
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        join_bit_weight=1.0,
        endpoint_loss_scale=1.0,
        task_head_mode="theorem_feature_scalar",
        summary_spec_root_mode="factored_theorem_readout",
        theorem_surface_mode=str(args.theorem_surface_mode),
        theorem_count_head_mode="scalar_mse",
        theorem_count_ordinal_weight=1.0,
        theorem_count_scalar_aux_weight=0.25,
        theorem_count_threshold_balance=True,
        theorem_feature_dim=int(args.theorem_feature_dim),
        theorem_feature_hidden_dim=int(args.theorem_feature_hidden_dim),
        theorem_score_dim=int(args.theorem_score_dim),
        theorem_fiber_dim=int(args.theorem_fiber_dim),
        theorem_aux_dim=int(args.theorem_aux_dim),
        merge_hidden_dim=0,
        score_merge_mode="gated_affine",
        phi_alignment_loss="cosine_mse",
        theorem_feature_adapter="markov_count_sketch",
        theorem_count_dim=int(args.theorem_count_dim),
        theorem_first_dim=int(args.theorem_first_dim),
        theorem_last_dim=int(args.theorem_last_dim),
        c2_mode="reconstruction",
        tree_model_version=str(args.tree_model_version),
    ).to(device=device)
    return model


def _train_model(
    *,
    model: FNOCountSketch,
    train_docs: Sequence[_FNOCountDoc],
    val_docs: Sequence[_FNOCountDoc],
    args: argparse.Namespace,
    device: torch.device,
    output_root: Path,
) -> Dict[str, Any]:
    progress_dir = output_root / "training_progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress_log = output_root / "progress.jsonl"

    def progress_callback(payload: Mapping[str, Any]) -> None:
        progress_log.open("a", encoding="utf-8").write(
            json.dumps(_jsonable(dict(payload)), sort_keys=True) + "\n"
        )

    stage1_epochs = int(args.stage1_epochs)
    stage2_epochs = int(args.stage2_epochs)
    artifact_dir = str(args.stage1_artifact_dir or "").strip()
    if artifact_dir and bool(args.resume_stage1_artifact):
        stage1_epochs = 0

    prev_runtime_mode = os.environ.get("TT_TREE_BATCH_RUNTIME_MODE")
    os.environ["TT_TREE_BATCH_RUNTIME_MODE"] = "unified_v2"
    try:
        return train_fno_tree(
            model=model,
            train_docs=train_docs,
            val_docs=val_docs,
            device=device,
            n_epochs=max(1, int(stage1_epochs + stage2_epochs)),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            root_weight=float(args.root_weight),
            c1_weight=float(args.local_weight),
            c2_weight=float(args.local_weight),
            c3_weight=float(args.local_weight),
            internal_supervision_kind="full_sketch",
            internal_label_rate=1.0,
            max_internal_depth=0,
            leaf_exact_supervision=False,
            leaf_supervision_kind="full_sketch",
            leaf_label_rate=1.0,
            tree_local_weighting_mode="span_mass_ipw_sum",
            tree_supervision_source="rate",
            phi_compose_weight=0.0,
            phi_contrastive_weight=0.0,
            checkpoint_metric="val_exact_sketch_direct",
            tree_training_schedule="two_stage",
            tree_stage1_epochs=int(stage1_epochs),
            tree_stage2_epochs=int(stage2_epochs),
            tree_stage1_checkpoint_metric="val_theorem_bootstrap_direct",
            tree_stage1_eval_mode="per_epoch",
            tree_batch_pack_mode="fixed_fused",
            tree_batch_autotune=False,
            tree_stage1_artifact_dir=artifact_dir,
            tree_stage1_resume_if_available=bool(args.resume_stage1_artifact),
            tree_stage1_root_weight=0.0,
            exact_metric_selection_doc_limit=0,
            exact_metric_selection_interval=1,
            exact_metric_final_doc_limit=0,
            tree_exact_eval_max_docs=int(args.eval_batch_size),
            progress_callback=progress_callback,
            progress_snapshot_interval=int(args.progress_snapshot_interval),
            progress_snapshot_dir=progress_dir,
            grad_clip_norm=1.0,
            depth_discount_gamma=1.0,
            seed=int(args.seed),
        )
    finally:
        if prev_runtime_mode is None:
            os.environ.pop("TT_TREE_BATCH_RUNTIME_MODE", None)
        else:
            os.environ["TT_TREE_BATCH_RUNTIME_MODE"] = prev_runtime_mode


def _evaluate_split(
    *,
    model: FNOCountSketch,
    docs: Sequence[_FNOCountDoc],
    target_scale: int,
    n_regimes: int,
    device: torch.device,
    eval_batch_size: int,
) -> Dict[str, Any]:
    learned = _eval_fno_exact_sketch_direct_metrics(
        model,
        docs,
        device=device,
        pack_mode="fixed_fused",
        runtime_bucket_mode="leaf_count_auto_queue",
        max_docs=max(1, int(eval_batch_size)),
        phi_pair_calibration_max_nodes=512,
    )
    exact_leaf = _evaluate_exact_leaf_merger(
        model,
        docs,
        target_scale=int(target_scale),
        n_regimes=int(n_regimes),
        device=device,
    )
    exact_leaf_encoder = _evaluate_exact_summary_leaf_encoder(
        model=model,
        docs=docs,
        target_scale=int(target_scale),
        n_regimes=int(n_regimes),
        device=device,
        eval_batch_size=int(eval_batch_size),
    )
    exact_leaf_root_mae = float(exact_leaf.get("step1_root_mae", float("nan")))
    learned_root_mae = float(learned.get("root_direct_count_mae", float("nan")))
    learned_exact_projected = float(learned.get("exact_projected_root_mae", float("nan")))
    return {
        "learned_leaves_learned_merge": _metric_subset(
            learned,
            (
                "root_direct_count_mae",
                "merge_direct_exact_match",
                "merge_first_accuracy",
                "merge_last_accuracy",
                "merge_join_bit_accuracy",
                "leaf_direct_exact_match",
                "leaf_first_accuracy",
                "leaf_last_accuracy",
                "leaf_direct_count_mae",
                "phi_merge_alignment",
                "val_exact_sketch_direct",
            ),
        ),
        "learned_leaves_exact_merge": _metric_subset(
            learned,
            (
                "exact_projected_root_mae",
                "root_mae_oracle_counts_predicted_endpoints",
                "root_mae_predicted_counts_oracle_endpoints",
            ),
        ),
        "exact_leaves_learned_merge": _metric_subset(
            exact_leaf,
            (
                "step1_root_mae",
                "step1_merge_exact_summary_match_rate",
                "step1_count_only_root_mae",
                "step1_endpoint_only_root_mae",
                "merge_first_accuracy",
                "merge_last_accuracy",
                "merge_join_accuracy",
                "per_depth_merge_exact_summary_match_rate",
            ),
        ),
        "exact_leaf_encoder_direct": exact_leaf_encoder,
        "exact_leaves_exact_merge": {
            "root_mae": 0.0,
            "note": "DGP theorem state with exact Markov merge is exact by construction.",
        },
        "deltas": {
            "learned_merge_benefit_over_exact_leaf_oracle_merge": (
                learned_root_mae - learned_exact_projected
            ),
            "leaf_bottleneck_gap_exact_leaf_merge_vs_learned_leaf_exact_merge": (
                learned_exact_projected - exact_leaf_root_mae
            ),
            "merge_bottleneck_gap_exact_leaf_merge_vs_oracle": exact_leaf_root_mae,
        },
    }


@torch.inference_mode()
def _evaluate_exact_summary_leaf_encoder(
    *,
    model: FNOCountSketch,
    docs: Sequence[_FNOCountDoc],
    target_scale: int,
    n_regimes: int,
    device: torch.device,
    eval_batch_size: int,
) -> Dict[str, float]:
    model.eval()
    count_abs_sum = 0.0
    exact_sum = 0.0
    first_sum = 0.0
    last_sum = 0.0
    total = 0
    batch_size = max(1, int(eval_batch_size))
    for start in range(0, len(docs), batch_size):
        batch_docs = tuple(docs[start : start + batch_size])
        if not batch_docs:
            continue
        leaf_summary = _leaf_summary_batch(
            batch_docs,
            target_scale=float(target_scale),
            n_regimes=int(n_regimes),
            device=device,
        )
        bsz, n_leaves, summary_dim = leaf_summary.shape
        states = model.encode_summary(
            leaf_summary.reshape(int(bsz) * int(n_leaves), int(summary_dim))
        )
        pred_count = model.predict_count_from_state(states).reshape(int(bsz), int(n_leaves))
        _h, first_logits, last_logits = model._split_state(states)
        pred_first = torch.argmax(first_logits, dim=-1).reshape(int(bsz), int(n_leaves))
        pred_last = torch.argmax(last_logits, dim=-1).reshape(int(bsz), int(n_leaves))
        truth_count = torch.tensor(
            [[float(value) for value in doc.leaf_counts] for doc in batch_docs],
            device=device,
            dtype=pred_count.dtype,
        )
        truth_first = torch.tensor(
            [[int(value) for value in doc.leaf_first_regimes] for doc in batch_docs],
            device=device,
            dtype=torch.long,
        )
        truth_last = torch.tensor(
            [[int(value) for value in doc.leaf_last_regimes] for doc in batch_docs],
            device=device,
            dtype=torch.long,
        )
        count_abs_sum += float(torch.abs(pred_count - truth_count).sum().detach().cpu())
        first_hit = pred_first.eq(truth_first)
        last_hit = pred_last.eq(truth_last)
        exact_hit = (
            torch.round(pred_count).to(dtype=torch.long).eq(
                torch.round(truth_count).to(dtype=torch.long)
            )
            & first_hit
            & last_hit
        )
        first_sum += float(first_hit.to(dtype=torch.float32).sum().detach().cpu())
        last_sum += float(last_hit.to(dtype=torch.float32).sum().detach().cpu())
        exact_sum += float(exact_hit.to(dtype=torch.float32).sum().detach().cpu())
        total += int(bsz) * int(n_leaves)
    denom = float(max(1, int(total)))
    return {
        "leaf_count_mae": float(count_abs_sum / denom),
        "leaf_exact_match": float(exact_sum / denom),
        "leaf_first_accuracy": float(first_sum / denom),
        "leaf_last_accuracy": float(last_sum / denom),
        "n_leaf_nodes": float(total),
    }


def _compact_train_result(train_result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "epochs_completed": int(train_result.get("epochs_completed", 0) or 0),
        "best_epoch": int(train_result.get("best_epoch", 0) or 0),
        "selection_metric_name": str(train_result.get("selection_metric_name", "")),
        "best_val_mae": float(train_result.get("best_val_mae", float("nan"))),
        "stage1_result_summary": _jsonable(train_result.get("stage1_result_summary", {})),
        "stage2_result_summary": _jsonable(train_result.get("stage2_result_summary", {})),
        "training_component_loss_finals": _jsonable(
            train_result.get("training_component_loss_finals", {})
        ),
        "train": _jsonable(train_result.get("train", {})),
        "val": _jsonable(train_result.get("val", {})),
        "best_exact_metrics": _metric_subset(
            train_result.get("best_exact_metrics", {}) or {},
            (
                "root_direct_count_mae",
                "exact_projected_root_mae",
                "leaf_direct_exact_match",
                "merge_direct_exact_match",
                "merge_join_bit_accuracy",
                "val_exact_sketch_direct",
            ),
        ),
    }


def _diagnosis(summary: Mapping[str, Any]) -> Dict[str, str]:
    test = dict((summary.get("splits") or {}).get("test") or {})
    learned_exact = dict(test.get("learned_leaves_exact_merge") or {})
    exact_leaf = dict(test.get("exact_leaves_learned_merge") or {})
    exact_leaf_encoder = dict(test.get("exact_leaf_encoder_direct") or {})
    learned_leaf_mae = float(learned_exact.get("exact_projected_root_mae", float("nan")))
    exact_leaf_mae = float(exact_leaf.get("step1_root_mae", float("nan")))
    exact_encoder_leaf_mae = float(exact_leaf_encoder.get("leaf_count_mae", float("nan")))
    exact_encoder_leaf_match = float(
        exact_leaf_encoder.get("leaf_exact_match", float("nan"))
    )
    if np.isfinite(learned_leaf_mae) and np.isfinite(exact_leaf_mae):
        if (
            np.isfinite(exact_encoder_leaf_mae)
            and np.isfinite(exact_encoder_leaf_match)
            and (exact_encoder_leaf_mae > 0.5 or exact_encoder_leaf_match < 0.9)
        ):
            bucket = "summary_encoder_bottleneck"
            explanation = (
                "Exact theorem summaries are not faithfully mapped into the model state before "
                "merging, so g_theta is operating on a distorted exact-sketch surface."
            )
        elif exact_leaf_mae <= 0.25 and learned_leaf_mae >= 1.0:
            bucket = "leaf_state_bottleneck"
            explanation = (
                "The learned merge can propagate exact theorem leaves, but decoded learned leaves "
                "do not carry reliable endpoint state."
            )
        elif exact_leaf_mae >= 1.0:
            bucket = "merge_or_summary_encoder_bottleneck"
            explanation = (
                "Even exact theorem leaves fail through the learned merge path, so the bottleneck "
                "is in g_theta or the exact-summary-to-state encoder used by g_theta."
            )
        else:
            bucket = "mixed"
            explanation = (
                "Exact leaves improve the root path, but the learned merge is still not at the "
                "oracle theorem merge."
            )
    else:
        bucket = "unknown"
        explanation = "Insufficient finite metrics to assign a bottleneck."
    return {"bucket": bucket, "explanation": explanation}


def _render_markdown(summary: Mapping[str, Any]) -> str:
    run = dict(summary.get("run") or {})
    diag = dict(summary.get("diagnosis") or {})
    lines = [
        "# Markov Leaf/Merge Split Diagnostic",
        "",
        f"- Benchmark: `{run.get('benchmark')}`",
        f"- Train docs: `{run.get('train_docs')}`",
        f"- Device: `{run.get('device')}`",
        f"- Stage schedule: `{run.get('stage1_epochs')} + {run.get('stage2_epochs')}`",
        f"- Diagnosis: `{diag.get('bucket')}`",
        "",
        str(diag.get("explanation", "")),
        "",
        "| split | learned leaves + learned merge MAE | learned leaves + exact merge MAE | exact leaves + learned merge MAE | exact leaf merge exact rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for split_name in ("train", "val", "test"):
        split = dict((summary.get("splits") or {}).get(split_name) or {})
        learned_learned = dict(split.get("learned_leaves_learned_merge") or {})
        learned_exact = dict(split.get("learned_leaves_exact_merge") or {})
        exact_learned = dict(split.get("exact_leaves_learned_merge") or {})
        lines.append(
            "| {split} | {ll:.6g} | {le:.6g} | {el:.6g} | {rate:.6g} |".format(
                split=split_name,
                ll=float(learned_learned.get("root_direct_count_mae", float("nan"))),
                le=float(learned_exact.get("exact_projected_root_mae", float("nan"))),
                el=float(exact_learned.get("step1_root_mae", float("nan"))),
                rate=float(
                    exact_learned.get(
                        "step1_merge_exact_summary_match_rate",
                        float("nan"),
                    )
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Test Details",
            "",
            "```json",
            json.dumps((summary.get("splits") or {}).get("test", {}), indent=2, sort_keys=True),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="recoverable_v5_t128")
    parser.add_argument("--train-docs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--stage1-epochs", type=int, default=40)
    parser.add_argument("--stage2-epochs", type=int, default=40)
    parser.add_argument("--stage1-artifact-dir", default="")
    parser.add_argument("--resume-stage1-artifact", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--root-weight", type=float, default=0.2)
    parser.add_argument("--local-weight", type=float, default=0.26666666666666666)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--tree-leaf-fno-width", type=int, default=128)
    parser.add_argument("--tree-leaf-fno-n-modes", type=int, default=8)
    parser.add_argument("--tree-leaf-fno-n-layers", type=int, default=4)
    parser.add_argument("--tree-model-version", default="v2")
    parser.add_argument("--theorem-surface-mode", default="shared_feature")
    parser.add_argument("--theorem-feature-dim", type=int, default=48)
    parser.add_argument("--theorem-feature-hidden-dim", type=int, default=256)
    parser.add_argument("--theorem-score-dim", type=int, default=1)
    parser.add_argument("--theorem-fiber-dim", type=int, default=47)
    parser.add_argument("--theorem-aux-dim", type=int, default=0)
    parser.add_argument("--theorem-count-dim", type=int, default=0)
    parser.add_argument("--theorem-first-dim", type=int, default=0)
    parser.add_argument("--theorem-last-dim", type=int, default=0)
    parser.add_argument("--progress-snapshot-interval", type=int, default=10)
    parser.add_argument(
        "--output-root",
        default=str(REPO / "outputs" / f"markov_leaf_merge_split_{_timestamp()}"),
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _set_global_seed(int(args.seed))
    if bool(args.use_cuda) and torch.cuda.is_available():
        device = torch.device(f"cuda:{int(args.cuda_device)}")
    else:
        device = torch.device("cpu")

    train_docs, val_docs, test_docs, prepared = _load_prepared_docs(
        benchmark=str(args.benchmark),
        train_docs=int(args.train_docs),
        seed=int(args.seed),
    )
    all_docs = tuple(train_docs) + tuple(val_docs) + tuple(test_docs)
    n_leaves, n_regimes = _validate_uniform_leaf_shape(all_docs)
    target_scale = _root_support_max(train_docs)

    model = _build_model(all_docs=all_docs, train_docs=train_docs, args=args, device=device)
    train_result = _train_model(
        model=model,
        train_docs=train_docs,
        val_docs=val_docs,
        args=args,
        device=device,
        output_root=output_root,
    )

    splits = {
        "train": _evaluate_split(
            model=model,
            docs=train_docs,
            target_scale=int(target_scale),
            n_regimes=int(n_regimes),
            device=device,
            eval_batch_size=int(args.eval_batch_size),
        ),
        "val": _evaluate_split(
            model=model,
            docs=val_docs,
            target_scale=int(target_scale),
            n_regimes=int(n_regimes),
            device=device,
            eval_batch_size=int(args.eval_batch_size),
        ),
        "test": _evaluate_split(
            model=model,
            docs=test_docs,
            target_scale=int(target_scale),
            n_regimes=int(n_regimes),
            device=device,
            eval_batch_size=int(args.eval_batch_size),
        ),
    }
    summary: Dict[str, Any] = {
        "run": {
            "benchmark": str(args.benchmark),
            "train_docs": int(args.train_docs),
            "seed": int(args.seed),
            "device": str(device),
            "stage1_epochs": int(args.stage1_epochs),
            "stage2_epochs": int(args.stage2_epochs),
            "stage1_artifact_dir": str(args.stage1_artifact_dir or ""),
            "resume_stage1_artifact": bool(args.resume_stage1_artifact),
            "n_leaves": int(n_leaves),
            "n_regimes": int(n_regimes),
            "target_scale": int(target_scale),
            "tree_model_version": str(args.tree_model_version),
            "theorem_surface_mode": str(args.theorem_surface_mode),
            "root_class_values": list(_root_class_values(all_docs)),
            "vocab_size": int(_vocab_size(all_docs)),
            "prepared_data_root": str(prepared.get("prepared_data_root", "")),
        },
        "train_result": _compact_train_result(train_result),
        "splits": _jsonable(splits),
    }
    torch.save(model.state_dict(), output_root / "final_model_state.pt")
    summary["run"]["final_model_state"] = str(output_root / "final_model_state.pt")
    summary["diagnosis"] = _diagnosis(summary)

    summary_json = output_root / "leaf_merge_split_summary.json"
    summary_md = output_root / "leaf_merge_split_summary.md"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary_md.write_text(_render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary["diagnosis"], indent=2, sort_keys=True))
    print(str(summary_json))
    print(str(summary_md))


if __name__ == "__main__":
    main()
