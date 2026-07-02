#!/usr/bin/env python3
"""Evaluate existing Manifesto DSPy f/g ladder artifacts on a new split."""

from __future__ import annotations

import argparse
import copy
import logging
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ctreepo.manifesto_ladder_families import build_dspy_family as _build_dspy_family  # noqa: E402
from src.experiments.ladder_reporting import (  # noqa: E402
    summarize_ladder_grid,
    write_alternating_markdown_summary,
)
from src.experiments.script_io import (  # noqa: E402
    now_iso as _now_iso,
    read_json_object as _load_json,
    read_jsonl as _read_jsonl,
    write_json as _write_json,
)
from src.experiments.script_parse import parse_int_grid as _parse_int_grid  # noqa: E402
from src.tasks.manifesto.result_rows import (  # noqa: E402
    order_split_rows as _order_split_rows,
    row_manifesto_id as _row_manifesto_id,
)
from src.ctreepo.alternating import evaluate_iteration  # noqa: E402
from src.ctreepo.distillation import load_labeled_trees  # noqa: E402
from src.core.batch_transport import (  # noqa: E402
    DEFAULT_BATCH_MAX_CONCURRENT,
    DEFAULT_BATCH_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_BATCH_ROUTING_POLICY,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_TIMEOUT_SECONDS,
)
from src.tasks.manifesto.expert_scale import (  # noqa: E402
    EXPERT_SCALE_CHOICES,
    EXPERT_SCALE_NORMALIZED_1_7,
    expert_scale_metadata,
    raw_benoit_expert_from_row,
    resolve_benoit_expert_target,
)


LOGGER = logging.getLogger(__name__)


def _tree_doc_id(tree: Any) -> str:
    metadata = getattr(tree, "metadata", None) or {}
    return str(getattr(tree, "doc_id", None) or metadata.get("manifesto_id") or metadata.get("doc_id") or "")


def _clone_test_trees(
    trees: Sequence[Any],
    test_ids: Sequence[str],
    *,
    rows_by_id: Mapping[str, Mapping[str, Any]],
    dimension: str,
    expert_target_scale: str,
) -> List[Any]:
    by_id = {_tree_doc_id(tree): tree for tree in trees if _tree_doc_id(tree)}
    out: List[Any] = []
    for doc_id in test_ids:
        tree = by_id.get(str(doc_id))
        if tree is None:
            LOGGER.warning("Missing labeled tree for requested test doc_id=%s", doc_id)
            continue
        clone = copy.deepcopy(tree)
        metadata = dict(getattr(clone, "metadata", None) or {})
        metadata["original_split"] = metadata.get("split")
        metadata["split"] = "test"
        row = rows_by_id.get(str(doc_id))
        if row is not None:
            expert_score = resolve_benoit_expert_target(
                row,
                dimension=dimension,
                scale=expert_target_scale,
            )
            expert_raw = raw_benoit_expert_from_row(row, dimension=dimension)
            if expert_score is not None:
                metadata["original_expert_score_1_7"] = metadata.get("expert_score_1_7")
                metadata["expert_score_1_7"] = float(expert_score)
                dim_scores = dict(metadata.get("expert_dimension_scores_1_7") or {})
                dim_scores[str(dimension)] = float(expert_score)
                metadata["expert_dimension_scores_1_7"] = dim_scores
            metadata["expert_score_raw_benoit"] = expert_raw
            metadata.update(expert_scale_metadata(dimension=dimension, scale=expert_target_scale))
        clone.metadata = metadata
        out.append(clone)
    return out


def _history_row_for_eval(
    *,
    source_history: Mapping[str, Any],
    eval_iterations: Sequence[Mapping[str, Any]],
    leaf_size_tokens: int,
    n_eval_trees: int,
) -> Dict[str, Any]:
    return {
        "family": "dspy",
        "axis_kind": "leaf_size_tokens",
        "axis_value": int(leaf_size_tokens),
        "leaf_count": None,
        "leaf_size_tokens": int(leaf_size_tokens),
        "row_label": f"leaf{int(leaf_size_tokens):04d}tok",
        "max_iterations": int(source_history.get("max_iterations") or len(eval_iterations) - 1),
        "eval_split": "test",
        "train_split": str(source_history.get("train_split") or "train"),
        "n_train_trees": int(source_history.get("n_train_trees") or 0),
        "n_eval_trees": int(n_eval_trees),
        "source_iteration_history": str(source_history.get("source_iteration_history") or ""),
        "iterations": list(eval_iterations),
    }


def _build_family_args(args: argparse.Namespace, *, leaf_size_tokens: int) -> argparse.Namespace:
    return argparse.Namespace(
        dimension=str(args.dimension),
        dspy_optimizer="mipro",
        dspy_budget="medium",
        dspy_num_threads=int(args.dspy_num_threads),
        dspy_model=str(args.dspy_model),
        dspy_api_base=str(args.dspy_api_base),
        dspy_api_key=str(args.dspy_api_key),
        dspy_max_tokens=int(args.dspy_max_tokens),
        dspy_lm_transport=str(args.dspy_lm_transport),
        dspy_batch_max_concurrent=int(args.dspy_batch_max_concurrent),
        dspy_batch_size=int(args.dspy_batch_size),
        dspy_batch_timeout=float(args.dspy_batch_timeout),
        dspy_batch_request_timeout=float(args.dspy_batch_request_timeout),
        dspy_batch_await_response_timeout=args.dspy_batch_await_response_timeout,
        dspy_batch_routing_policy=str(args.dspy_batch_routing_policy),
        dspy_mipro_num_candidates=None,
        dspy_mipro_num_trials=None,
        dspy_mipro_max_bootstrapped_demos=None,
        dspy_mipro_max_labeled_demos=None,
        dspy_mipro_minibatch_size=35,
        dspy_mipro_minibatch_full_eval_steps=5,
        dspy_lm_context_tokens=int(args.dspy_lm_context_tokens),
        dspy_prompt_overhead_tokens=int(args.dspy_prompt_overhead_tokens),
        dspy_f_init_path=None,
        dspy_f_init_mode="pretuned_scorer",
        root_label_sources=(),
        root_label_target="expert",
        local_law_weight=1.0,
        node_weight_normalization="per_tree",
        embedding_model=str(args.embedding_model),
        target_min=float(args.target_min),
        target_max=float(args.target_max),
        leaf_size_tokens=int(leaf_size_tokens),
    )


def evaluate_leaf(args: argparse.Namespace, *, leaf_size_tokens: int, test_ids: Sequence[str]) -> Dict[str, Any]:
    source_root = Path(args.source_run_root)
    source_leaf_dir = source_root / "ladder" / "dspy" / f"leaf{int(leaf_size_tokens):04d}tok"
    source_history_path = source_leaf_dir / "iteration_history.json"
    labeled_trees_path = source_root / "teacher" / f"leaf{int(leaf_size_tokens):04d}tok" / "labeled_trees.jsonl"
    if not source_history_path.exists():
        raise FileNotFoundError(f"missing source iteration history: {source_history_path}")
    if not labeled_trees_path.exists():
        raise FileNotFoundError(f"missing labeled trees: {labeled_trees_path}")

    source_history = _load_json(source_history_path)
    source_history["source_iteration_history"] = str(source_history_path)
    source_rows = _read_jsonl(Path(args.source_results))
    rows_by_id = {_row_manifesto_id(row): row for row in source_rows if _row_manifesto_id(row)}
    trees = _clone_test_trees(
        load_labeled_trees(labeled_trees_path),
        test_ids,
        rows_by_id=rows_by_id,
        dimension=str(args.dimension),
        expert_target_scale=str(args.expert_target_scale),
    )
    if not trees:
        raise RuntimeError(f"no eval trees selected for leaf_size_tokens={leaf_size_tokens}")

    family = _build_dspy_family(_build_family_args(args, leaf_size_tokens=leaf_size_tokens), leaf_size_tokens=leaf_size_tokens)
    out_leaf_dir = Path(args.output_dir) / "dspy" / f"leaf{int(leaf_size_tokens):04d}tok"
    out_leaf_dir.mkdir(parents=True, exist_ok=True)

    eval_iterations: List[Dict[str, Any]] = []
    for it in source_history.get("iterations", []):
        stage_name = str(it.get("stage_name") or f"iter_{it.get('iteration')}")
        iteration = int(it.get("iteration") or 0)
        f_artifact = it.get("f_artifact") or "identity"
        g_artifact = it.get("g_artifact") or "identity"
        LOGGER.info(
            "Evaluating leaf=%s stage=%s n=%d",
            leaf_size_tokens,
            stage_name,
            len(trees),
        )
        metrics = evaluate_iteration(
            family=family,
            f=f_artifact,
            g=g_artifact,
            trees=trees,
            splits=("all", "test"),
            prediction_records_path=(
                out_leaf_dir / "prediction_records" / f"iter_{iteration:02d}_post_eval.jsonl"
            ),
        )
        row = {
            "iteration": iteration,
            "stage_name": stage_name,
            "stage_label": it.get("stage_label") or stage_name,
            "family": "dspy",
            "trained": "eval_only",
            "f_degree": it.get("f_degree"),
            "g_degree": it.get("g_degree"),
            "axis_kind": "leaf_size_tokens",
            "axis_value": int(leaf_size_tokens),
            "leaf_count": None,
            "leaf_size_tokens": int(leaf_size_tokens),
            "f_artifact": f_artifact,
            "g_artifact": g_artifact,
            "split_metrics": {name: asdict(value) for name, value in metrics.items()},
            "extra": {
                "eval_only": True,
                "source_iteration_history": str(source_history_path),
            },
        }
        eval_iterations.append(row)
        checkpoint = {
            "schema_version": 1,
            "created_at": _now_iso(),
            "phase": "post_eval",
            "iteration": iteration,
            "stage_name": stage_name,
            "stage_label": row["stage_label"],
            "family": "dspy",
            "trained": "eval_only",
            "axis_kind": "leaf_size_tokens",
            "axis_value": int(leaf_size_tokens),
            "leaf_count": None,
            "leaf_size_tokens": int(leaf_size_tokens),
            "f_artifact": f_artifact,
            "g_artifact": g_artifact,
            "split_metrics": row["split_metrics"],
            "error": None,
            "source_iteration_history": str(source_history_path),
        }
        checkpoint_path = out_leaf_dir / "step_checkpoints" / f"iter_{iteration:02d}_post_eval.json"
        checkpoint["checkpoint_path"] = str(checkpoint_path)
        _write_json(checkpoint_path, checkpoint)
        _write_json(out_leaf_dir / "step_checkpoints" / "latest.json", checkpoint)

    history = _history_row_for_eval(
        source_history=source_history,
        eval_iterations=eval_iterations,
        leaf_size_tokens=int(leaf_size_tokens),
        n_eval_trees=len(trees),
    )
    _write_json(out_leaf_dir / "iteration_history.json", history)
    return history


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dimension", default="environment")
    parser.add_argument("--leaf-size-tokens", default="256,512,1024,2048")
    parser.add_argument("--train-n", type=int, default=107)
    parser.add_argument("--val-n", type=int, default=30)
    parser.add_argument("--test-n", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--expert-target-scale",
        choices=EXPERT_SCALE_CHOICES,
        default=EXPERT_SCALE_NORMALIZED_1_7,
        help=(
            "Scale used for expert_score_1_7 during this eval-only relabeling pass. "
            "Use raw_benoit to reproduce older metrics from stored trees."
        ),
    )
    parser.add_argument("--dspy-model", default="openai/nvidia/Gemma-4-31B-IT-NVFP4")
    parser.add_argument("--dspy-api-base", default="http://localhost:8010/v1")
    parser.add_argument("--dspy-api-key", default="EMPTY")
    parser.add_argument("--dspy-max-tokens", type=int, default=0)
    parser.add_argument("--dspy-lm-context-tokens", type=int, required=True)
    parser.add_argument("--dspy-prompt-overhead-tokens", type=int, default=1500)
    parser.add_argument("--dspy-num-threads", type=int, default=128)
    parser.add_argument("--dspy-lm-transport", choices=["batch", "litellm"], default="batch")
    parser.add_argument("--dspy-batch-max-concurrent", type=int, default=DEFAULT_BATCH_MAX_CONCURRENT)
    parser.add_argument("--dspy-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dspy-batch-timeout", type=float, default=DEFAULT_BATCH_TIMEOUT_SECONDS)
    parser.add_argument("--dspy-batch-request-timeout", type=float, default=DEFAULT_BATCH_REQUEST_TIMEOUT_SECONDS)
    parser.add_argument("--dspy-batch-await-response-timeout", type=float, default=None)
    parser.add_argument("--dspy-batch-routing-policy", default=DEFAULT_BATCH_ROUTING_POLICY)
    parser.add_argument("--embedding-model", default="/mnt/data/models/google/embeddinggemma-300m")
    parser.add_argument("--target-min", type=float, default=1.0)
    parser.add_argument("--target-max", type=float, default=7.0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    rows = _read_jsonl(Path(args.source_results))
    rows_by_id = {_row_manifesto_id(row): row for row in rows if _row_manifesto_id(row)}
    split_ids = _order_split_rows(
        rows_by_id,
        train_n=int(args.train_n),
        val_n=int(args.val_n),
        test_n=int(args.test_n),
        seed=int(args.seed),
    )
    test_ids = list(split_ids.get("test", {}).keys())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir.parent / "eval_split_ids.json",
        {split: sorted(values) for split, values in split_ids.items()},
    )
    _write_json(
        output_dir.parent / "eval_manifest.json",
        {
            "created_at": _now_iso(),
            "mode": "eval_only_existing_dspy_ladder",
            "source_run_root": str(args.source_run_root),
            "source_results": str(args.source_results),
            "output_dir": str(output_dir),
            "dimension": str(args.dimension),
            "train_n": int(args.train_n),
            "val_n": int(args.val_n),
            "test_n": int(args.test_n),
            "expert_target_scale": str(args.expert_target_scale),
            "actual_split_sizes": {split: len(values) for split, values in split_ids.items()},
            "leaf_size_tokens": list(_parse_int_grid(args.leaf_size_tokens)),
        },
    )

    grid_rows = [
        evaluate_leaf(args, leaf_size_tokens=int(leaf), test_ids=test_ids)
        for leaf in _parse_int_grid(args.leaf_size_tokens)
    ]
    summary_rows = summarize_ladder_grid(grid_rows, eval_split="test")
    _write_json(output_dir.parent / "grid_summary.json", {"rows": summary_rows})
    write_alternating_markdown_summary(summary_rows, output_dir.parent / "grid_summary.md", eval_split="test")
    print(f"wrote {output_dir.parent / 'grid_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
