#!/usr/bin/env python3
"""Run a fixed-world Segment-LDA OPS sweep for one (family, process, leaf_tokens, seed) bundle."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
from typing import List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ctreepo.sim.core.segment_lda_ops_weight_recovery import (  # noqa: E402
    SegmentLDAOpsWeightRecoveryConfig,
    SegmentLDAOpsWeightRecoverySummary,
    run_segment_lda_ops_weight_recovery_experiment_from_world,
    sample_segment_lda_ops_weight_recovery_world,
    segment_lda_ops_weight_recovery_world_cache_signature,
)


def _parse_items(text: str) -> List[str]:
    out: List[str] = []
    for raw in str(text).replace(",", " ").split():
        item = raw.strip()
        if item:
            out.append(item)
    return out


def _parse_ints(text: str) -> List[int]:
    return [int(x) for x in _parse_items(text)]


def _parse_floats(text: str) -> List[float]:
    return [float(x) for x in _parse_items(text)]


def _fmt_float(x: float) -> str:
    s = f"{float(x):.6g}"
    return s.replace("-", "m").replace(".", "p")


def _rows_from_summary(summary: SegmentLDAOpsWeightRecoverySummary) -> List[dict]:
    cfg = dict(summary.config)
    geom = dict(summary.training_geometry)
    wt = dict(summary.weight_truth)
    rows: List[dict] = []
    metrics = summary.metrics
    if not isinstance(metrics, dict):
        return rows
    for name, metric in metrics.items():
        if not isinstance(metric, dict):
            continue
        row = {
            "method": str(name),
            **{f"cfg_{k}": cfg.get(k) for k in cfg.keys()},
            **{f"train_{k}": geom.get(k) for k in geom.keys()},
            "relevant_topics": ",".join(str(x) for x in wt.get("relevant_topics", [])),
            "lambda_multiplier": wt.get("lambda_multiplier"),
        }
        row.update(metric)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            fieldnames.append(key)
            seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _output_base(
    *,
    output_root: Path,
    family: str,
    topic_process: str,
    leaf_tokens: int,
    lambda_multiplier: float,
    train_docs: int,
    audit_fraction: float,
    seed: int,
    topic_phi_estimator: str,
    topic_phi_docs: int,
) -> Path:
    parts = [
        family,
        f"proc_{topic_process}",
        f"leaf_{leaf_tokens}",
    ]
    if str(topic_phi_estimator) != "true" or int(topic_phi_docs) > 0:
        parts.append(f"phi_{topic_phi_estimator}")
        parts.append(f"phi_docs_{topic_phi_docs}")
    parts.extend(
        [
            f"lam_{_fmt_float(lambda_multiplier)}",
            f"train_{train_docs}",
            f"audit_{audit_fraction}",
            f"seed_{seed}",
        ]
    )
    return output_root.joinpath(*parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fixed-world Segment-LDA OPS batch sweep.")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument(
        "--family",
        type=str,
        choices=["whole_document_controls", "one_boundary_controls", "full_tree_sweeps"],
        required=True,
    )
    parser.add_argument("--topic-process", type=str, choices=["segments", "bag_of_words"], required=True)
    parser.add_argument("--leaf-tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)

    parser.add_argument("--train-docs-grid", type=str, required=True)
    parser.add_argument("--lambda-grid", type=str, required=True)
    parser.add_argument("--audit-fractions", type=str, required=True)
    parser.add_argument("--test-docs", type=int, default=1024)
    parser.add_argument("--topic-phi-estimators", type=str, default="true")
    parser.add_argument("--topic-phi-docs-grid", type=str, default="0")
    parser.add_argument("--topic-source", choices=["true", "infer"], default="infer")
    parser.add_argument("--feature-inference", choices=["hard", "soft"], default="hard")
    parser.add_argument("--run-all-feature-modes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--world-cache-dir", type=str, default="")
    parser.add_argument("--world-train-docs-capacity", type=int, default=0)
    parser.add_argument("--world-test-docs-capacity", type=int, default=0)
    parser.add_argument("--world-phi-extra-docs-capacity", type=int, default=-1)

    parser.add_argument("--n-topics", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--min-tokens", type=int, default=384)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--min-segments", type=int, default=2)
    parser.add_argument("--max-segments", type=int, default=6)
    parser.add_argument("--min-seg-len", type=int, default=48)
    parser.add_argument("--max-seg-len", type=int, default=256)
    parser.add_argument("--align-segments-to-leaves", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--doc-topic-concentration", type=float, default=0.6)
    parser.add_argument("--boundary-profile", type=str, default="uniform")
    parser.add_argument("--boundary-profile-strength", type=float, default=0.0)
    parser.add_argument("--boundary-profile-seed", type=int, default=-1)
    parser.add_argument("--segment-length-power", type=float, default=1.0)

    parser.add_argument("--topic-concentration", type=float, default=0.2)
    parser.add_argument("--emission-mode", type=str, choices=["anchored", "disjoint"], default="anchored")
    parser.add_argument("--anchor-words-per-topic", type=int, default=20)
    parser.add_argument("--anchor-multiplier", type=float, default=25.0)

    parser.add_argument("--relevant-topics", type=int, default=2)
    parser.add_argument("--theta-scale", type=float, default=1.0)
    parser.add_argument("--zero-diagonal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--oracle-noise-std", type=float, default=0.0)

    parser.add_argument("--audit-policy", type=str, default="fraction")
    parser.add_argument("--audit-fixed-nodes", type=int, default=0)
    parser.add_argument("--audit-scale", type=float, default=1.0)
    parser.add_argument("--audit-strategy", type=str, default="random")
    parser.add_argument("--oracle-cost-power", type=float, default=1.25)
    parser.add_argument("--oracle-cost-per-query", type=float, default=0.0)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)

    parser.add_argument("--tlda-delta", type=float, default=0.10)
    parser.add_argument("--tlda-rate-constant", type=float, default=1.0)
    parser.add_argument("--tlda-sigmaK-floor", type=float, default=1e-6)
    parser.add_argument("--topic-phi-permute", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--online-tensor-lda-burn-in-docs", type=int, default=0)
    parser.add_argument("--online-tensor-lda-batch-docs", type=int, default=32)
    parser.add_argument("--online-tensor-lda-passes", type=int, default=1)
    parser.add_argument("--online-tensor-lda-lr", type=float, default=0.1)
    parser.add_argument("--online-tensor-lda-grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--embedding-topic-svd-dim-extra", type=int, default=4)
    parser.add_argument("--embedding-topic-kmeans-inits", type=int, default=8)
    parser.add_argument("--embedding-topic-kmeans-max-iter", type=int, default=80)
    parser.add_argument("--embedding-topic-assignment-temperature", type=float, default=0.35)
    parser.add_argument("--embedding-topic-ppmi-shift", type=float, default=1.0)
    parser.add_argument("--neural-topic-base-estimator", type=str, default="tensor_lda")
    parser.add_argument("--neural-topic-seed-fraction", type=float, default=0.35)
    parser.add_argument("--neural-topic-hidden-dim", type=int, default=48)
    parser.add_argument("--neural-topic-steps", type=int, default=60)
    parser.add_argument("--neural-topic-lr", type=float, default=3e-3)
    parser.add_argument("--neural-topic-weight-decay", type=float, default=1e-4)
    parser.add_argument("--neural-topic-mix-samples", type=int, default=128)
    parser.add_argument("--neural-topic-mix-temperature", type=float, default=1.0)
    parser.add_argument("--neural-topic-operator-boost", type=float, default=1.4)
    parser.add_argument("--neural-topic-seed-llm-min-weight", type=float, default=0.2)
    parser.add_argument("--neural-topic-seed-llm-max-weight", type=float, default=0.55)
    parser.add_argument("--neural-topic-similarity-temperature", type=float, default=0.15)
    parser.add_argument("--neural-topic-ridge", type=float, default=1e-3)
    parser.add_argument("--violation-tau", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_docs_grid = _parse_ints(args.train_docs_grid)
    lambda_grid = _parse_floats(args.lambda_grid)
    audit_fractions = _parse_floats(args.audit_fractions)
    topic_phi_estimators = _parse_items(args.topic_phi_estimators)
    topic_phi_docs_grid = _parse_ints(args.topic_phi_docs_grid)

    if not train_docs_grid or not lambda_grid or not audit_fractions or not topic_phi_estimators or not topic_phi_docs_grid:
        raise ValueError("all grids must be non-empty")

    max_train_docs = max(train_docs_grid)
    max_phi_extra = 0
    for train_docs in train_docs_grid:
        for phi_docs in topic_phi_docs_grid:
            phi_effective = int(phi_docs) if int(phi_docs) > 0 else int(train_docs)
            max_phi_extra = max(max_phi_extra, max(0, int(phi_effective) - int(train_docs)))

    world_train_docs_capacity = int(args.world_train_docs_capacity) if int(args.world_train_docs_capacity) > 0 else int(max_train_docs)
    world_test_docs_capacity = int(args.world_test_docs_capacity) if int(args.world_test_docs_capacity) > 0 else int(args.test_docs)
    world_phi_extra_docs_capacity = (
        int(args.world_phi_extra_docs_capacity)
        if int(args.world_phi_extra_docs_capacity) >= 0
        else int(max_phi_extra)
    )

    base_cfg = SegmentLDAOpsWeightRecoveryConfig(
        n_topics=int(args.n_topics),
        vocab_size=int(args.vocab_size),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        min_segments=int(args.min_segments),
        max_segments=int(args.max_segments),
        min_seg_len=int(args.min_seg_len),
        max_seg_len=int(args.max_seg_len),
        leaf_tokens=int(args.leaf_tokens),
        align_segments_to_leaves=bool(args.align_segments_to_leaves),
        doc_topic_concentration=float(args.doc_topic_concentration),
        topic_process=str(args.topic_process),
        boundary_profile=str(args.boundary_profile),
        boundary_profile_strength=float(args.boundary_profile_strength),
        boundary_profile_seed=int(args.boundary_profile_seed),
        segment_length_power=float(args.segment_length_power),
        topic_concentration=float(args.topic_concentration),
        emission_mode=str(args.emission_mode),
        anchor_words_per_topic=int(args.anchor_words_per_topic),
        anchor_multiplier=float(args.anchor_multiplier),
        relevant_topics=int(args.relevant_topics),
        theta_scale=float(args.theta_scale),
        zero_diagonal=bool(args.zero_diagonal),
        lambda_multiplier=float(lambda_grid[0]),
        oracle_noise_std=float(args.oracle_noise_std),
        audit_policy=str(args.audit_policy),
        audit_fixed_nodes=int(args.audit_fixed_nodes),
        audit_fraction=float(audit_fractions[0]),
        audit_scale=float(args.audit_scale),
        audit_strategy=str(args.audit_strategy),
        oracle_cost_power=float(args.oracle_cost_power),
        oracle_cost_per_query=float(args.oracle_cost_per_query),
        ridge_lambda=float(args.ridge_lambda),
        topic_source=str(args.topic_source),
        feature_inference=str(args.feature_inference),
        topic_phi_estimator=str(topic_phi_estimators[0]),
        topic_phi_docs=int(topic_phi_docs_grid[0]),
        tlda_delta=float(args.tlda_delta),
        tlda_rate_constant=float(args.tlda_rate_constant),
        tlda_sigmaK_floor=float(args.tlda_sigmaK_floor),
        topic_phi_permute=bool(args.topic_phi_permute),
        online_tensor_lda_burn_in_docs=int(args.online_tensor_lda_burn_in_docs),
        online_tensor_lda_batch_docs=int(args.online_tensor_lda_batch_docs),
        online_tensor_lda_passes=int(args.online_tensor_lda_passes),
        online_tensor_lda_lr=float(args.online_tensor_lda_lr),
        online_tensor_lda_grad_clip_norm=float(args.online_tensor_lda_grad_clip_norm),
        embedding_topic_svd_dim_extra=int(args.embedding_topic_svd_dim_extra),
        embedding_topic_kmeans_inits=int(args.embedding_topic_kmeans_inits),
        embedding_topic_kmeans_max_iter=int(args.embedding_topic_kmeans_max_iter),
        embedding_topic_assignment_temperature=float(args.embedding_topic_assignment_temperature),
        embedding_topic_ppmi_shift=float(args.embedding_topic_ppmi_shift),
        neural_topic_base_estimator=str(args.neural_topic_base_estimator),
        neural_topic_seed_fraction=float(args.neural_topic_seed_fraction),
        neural_topic_hidden_dim=int(args.neural_topic_hidden_dim),
        neural_topic_steps=int(args.neural_topic_steps),
        neural_topic_lr=float(args.neural_topic_lr),
        neural_topic_weight_decay=float(args.neural_topic_weight_decay),
        neural_topic_mix_samples=int(args.neural_topic_mix_samples),
        neural_topic_mix_temperature=float(args.neural_topic_mix_temperature),
        neural_topic_operator_boost=float(args.neural_topic_operator_boost),
        neural_topic_seed_llm_min_weight=float(args.neural_topic_seed_llm_min_weight),
        neural_topic_seed_llm_max_weight=float(args.neural_topic_seed_llm_max_weight),
        neural_topic_similarity_temperature=float(args.neural_topic_similarity_temperature),
        neural_topic_ridge=float(args.neural_topic_ridge),
        run_all_feature_modes=bool(args.run_all_feature_modes),
        violation_tau=float(args.violation_tau),
        train_docs=int(world_train_docs_capacity),
        test_docs=int(world_test_docs_capacity),
        seed=int(args.seed),
    )
    world_cache_dir = Path(args.world_cache_dir) if str(args.world_cache_dir).strip() else (output_root / "world_cache")
    world_cache_dir.mkdir(parents=True, exist_ok=True)
    world_cache_sig = segment_lda_ops_weight_recovery_world_cache_signature(
        base_cfg,
        train_docs_capacity=int(world_train_docs_capacity),
        test_docs_capacity=int(world_test_docs_capacity),
        phi_extra_docs_capacity=int(world_phi_extra_docs_capacity),
    )
    cache_key = hashlib.sha256(json.dumps(world_cache_sig, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    world_cache_path = world_cache_dir / f"{cache_key}.pkl"

    lock_path = world_cache_dir / f"{cache_key}.lock"
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if world_cache_path.exists():
            with world_cache_path.open("rb") as handle:
                world = pickle.load(handle)
        else:
            world = sample_segment_lda_ops_weight_recovery_world(
                base_cfg,
                train_docs_capacity=int(world_train_docs_capacity),
                test_docs_capacity=int(world_test_docs_capacity),
                phi_extra_docs_capacity=int(world_phi_extra_docs_capacity),
            )
            tmp_path = world_cache_dir / f"{cache_key}.{os.getpid()}.tmp"
            with tmp_path.open("wb") as handle:
                pickle.dump(world, handle, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(world_cache_path)
    world_meta_path = (
        output_root
        / "worlds"
        / str(args.family)
        / f"proc_{args.topic_process}"
        / f"leaf_{int(args.leaf_tokens)}"
        / f"seed_{int(args.seed)}.json"
    )
    world_meta_path.parent.mkdir(parents=True, exist_ok=True)
    world_meta_path.write_text(
        json.dumps(
            {
                "family": str(args.family),
                "topic_process": str(args.topic_process),
                "leaf_tokens": int(args.leaf_tokens),
                "seed": int(args.seed),
                "train_docs_capacity": int(world_train_docs_capacity),
                "test_docs_capacity": int(world_test_docs_capacity),
                "phi_extra_docs_capacity": int(world_phi_extra_docs_capacity),
                "signature": dict(world.signature),
                "cache_signature": dict(world_cache_sig),
                "cache_path": str(world_cache_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    completed = 0
    skipped = 0
    for topic_phi_estimator in topic_phi_estimators:
        for topic_phi_docs in topic_phi_docs_grid:
            for train_docs in train_docs_grid:
                for audit_fraction in audit_fractions:
                    for lambda_multiplier in lambda_grid:
                        base = _output_base(
                            output_root=output_root,
                            family=str(args.family),
                            topic_process=str(args.topic_process),
                            leaf_tokens=int(args.leaf_tokens),
                            lambda_multiplier=float(lambda_multiplier),
                            train_docs=int(train_docs),
                            audit_fraction=float(audit_fraction),
                            seed=int(args.seed),
                            topic_phi_estimator=str(topic_phi_estimator),
                            topic_phi_docs=int(topic_phi_docs),
                        )
                        json_path = base.with_suffix(".json")
                        csv_path = base.with_suffix(".csv")
                        if bool(args.skip_existing) and json_path.exists() and csv_path.exists():
                            skipped += 1
                            continue

                        cfg = SegmentLDAOpsWeightRecoveryConfig(
                            **{
                                **asdict(base_cfg),
                                "lambda_multiplier": float(lambda_multiplier),
                                "audit_fraction": float(audit_fraction),
                                "train_docs": int(train_docs),
                                "topic_phi_estimator": str(topic_phi_estimator),
                                "topic_phi_docs": int(topic_phi_docs),
                            }
                        )
                        summary = run_segment_lda_ops_weight_recovery_experiment_from_world(cfg, world)
                        rows = _rows_from_summary(summary)
                        json_path.parent.mkdir(parents=True, exist_ok=True)
                        json_path.write_text(summary.to_json(), encoding="utf-8")
                        _write_csv(csv_path, rows)
                        completed += 1

    print(
        json.dumps(
            {
                "family": str(args.family),
                "topic_process": str(args.topic_process),
                "leaf_tokens": int(args.leaf_tokens),
                "seed": int(args.seed),
                "completed": int(completed),
                "skipped": int(skipped),
                "train_docs_capacity": int(world_train_docs_capacity),
                "test_docs_capacity": int(world_test_docs_capacity),
                "phi_extra_docs_capacity": int(world_phi_extra_docs_capacity),
                "world_cache_path": str(world_cache_path),
                "world_meta": str(world_meta_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
