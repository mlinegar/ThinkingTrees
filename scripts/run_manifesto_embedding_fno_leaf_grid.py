#!/usr/bin/env python3
"""Run an embedding-coordinate FNO leaf-topology grid on existing LLM results.

The leaf grid changes only the tree topology.  The FNO's spatial grid is the
embedding coordinate axis, so a 1024-dim embedding is processed as a length-1024
1D signal at every leaf and merge state.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_manifesto_dimension_fit_existing_results import (
    _DIM_FROM_NAME,
    _get_text_for_row,
    _load_run_metadata,
    _make_embedding_client,
    _order_split_rows,
    _phase3_split_examples,
    _preload_transformers_for_local_embedding,
    _read_jsonl,
    _row_expert_score,
    _row_manifesto_id,
    _row_summary,
    _row_target_score,
    _row_teacher_score,
)
from src.ctreepo.distillation import build_labeled_tree_from_text, write_labeled_trees_jsonl
from src.ctreepo.embedding_fno import (
    EmbeddingFNOModelConfig,
    EmbeddingFNOObjectiveConfig,
    EmbeddingFNOTrainConfig,
)
from src.tasks.manifesto import ManifestoDataset
from src.tasks.manifesto.dimensions import get_preservation_rubric
from src.training.config_sections import (
    OptimizerConfig,
    RunConfig,
    RuntimeConfig,
    TestConfig,
    TrainConfig,
    ValidationConfig,
    config_to_dict,
)
from src.tree.contract_runner import RESOURCE_EMBEDDING, TreePOResourceSpec, fit_treepo_contract
from src.tree.labeled import LabeledTree
from src.tree.treepo_stack import TreePOContractSpec, TreePOModelSpec


LOGGER = logging.getLogger(__name__)


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_to_dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _parse_int_grid(value: str) -> Tuple[int, ...]:
    parts = [part.strip() for part in str(value or "").replace(";", ",").split(",")]
    grid = tuple(int(part) for part in parts if part)
    if not grid:
        raise ValueError("grid must contain at least one integer")
    if any(item <= 0 for item in grid):
        raise ValueError(f"grid entries must be positive: {grid!r}")
    return grid


def _build_trees_for_leaf_count(
    *,
    rows: Sequence[Mapping[str, Any]],
    split_ids: Mapping[str, Mapping[str, str]],
    dimension: str,
    target_source: str,
    target_leaves_per_doc: int,
    source_results: Path,
    source_report: Optional[Path],
    mp_data_dir: Optional[Path],
) -> Tuple[List[LabeledTree], Dict[str, Any]]:
    rows_by_id = {_row_manifesto_id(row): row for row in rows if _row_manifesto_id(row)}
    needs_dataset = any(not text for split in split_ids.values() for text in split.values())
    dataset = ManifestoDataset(data_dir=mp_data_dir, require_text=True) if needs_dataset else None
    trees: List[LabeledTree] = []
    skipped = {"missing_row": 0, "missing_text": 0, "missing_target_or_summary": 0}

    for split, id_to_text in split_ids.items():
        for manifesto_id, split_text in id_to_text.items():
            row = rows_by_id.get(str(manifesto_id))
            if row is None:
                skipped["missing_row"] += 1
                continue
            text = _get_text_for_row(row=row, split_texts={str(manifesto_id): split_text}, dataset=dataset)
            target = _row_target_score(row, dimension=dimension, target_source=target_source)
            summary = _row_summary(row)
            if not text.strip():
                skipped["missing_text"] += 1
                continue
            if target is None or not summary:
                skipped["missing_target_or_summary"] += 1
                continue

            teacher_score = _row_teacher_score(row, dimension=dimension)
            expert_score = _row_expert_score(row, dimension=dimension)
            tree = build_labeled_tree_from_text(
                doc_id=str(manifesto_id),
                text=text,
                document_score=float(target),
                split=str(split),
                score_fn=lambda _span, score=float(target): float(score),
                window_size=max(1, len(text)),
                target_leaves_per_doc=int(target_leaves_per_doc),
                label_source=f"existing_gemma4_{target_source}_projected_to_nodes",
                root_summary=summary,
                summary_source="existing_gemma_result_root",
                fill_missing_summaries_from_span=False,
                extra_metadata={
                    "dimension": str(dimension),
                    "target_source": str(target_source),
                    "teacher_score_1_7": teacher_score,
                    "expert_score_1_7": expert_score,
                    "source_results_path": str(source_results),
                    "source_report_path": str(source_report) if source_report else None,
                    "node_score_projection": "document_level_existing_llm_score_attached_to_each_replayed_tree_node",
                    "node_score_projection_reason": (
                        "Existing Gemma result rows provide document/root scores. "
                        "This grid compares model node predictions to that replayed "
                        "teacher target on every exact topology node."
                    ),
                },
            )
            trees.append(tree)

    return trees, {
        "total": len(trees),
        "train": sum(1 for tree in trees if tree.metadata.get("split") == "train"),
        "val": sum(1 for tree in trees if tree.metadata.get("split") == "val"),
        "test": sum(1 for tree in trees if tree.metadata.get("split") == "test"),
        "skipped": skipped,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dimension", choices=sorted(_DIM_FROM_NAME), default="economic")
    parser.add_argument(
        "--source-results",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "overnight_benoit" / "full_pipeline" / "economic" / "per_manifesto.jsonl",
    )
    parser.add_argument(
        "--source-report",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "overnight_benoit" / "full_pipeline" / "economic" / "report.json",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-source", choices=["teacher", "expert"], default="teacher")
    parser.add_argument("--leaf-grid", default="1,2,4,8,16")
    parser.add_argument("--split-source", choices=["phase3", "results-order"], default="phase3")
    parser.add_argument("--train-pool", choices=["expert-split", "openweight", "expert"], default="expert-split")
    parser.add_argument("--split-strategy", choices=["random", "label-stratified"], default="label-stratified")
    parser.add_argument("--train-n", type=int, default=80)
    parser.add_argument("--val-n", type=int, default=20)
    parser.add_argument("--test-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mp-data-dir", type=Path, default=None)

    parser.add_argument("--embedding-backend", choices=["local-hf", "vllm", "hashing"], default="local-hf")
    parser.add_argument("--embedding-model", type=str, default="/mnt/data/models/Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-url", type=str, default=None)
    parser.add_argument("--embedding-api-key", type=str, default="EMPTY")
    parser.add_argument("--embedding-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-max-length", type=int, default=1024)
    parser.add_argument("--embedding-device", default="auto")
    parser.add_argument("--hashing-embedding-dim", type=int, default=256)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--n-modes", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--head-hidden-dim", type=int, default=64)
    parser.add_argument("--root-weight", type=float, default=1.0)
    parser.add_argument("--leaf-weight", type=float, default=0.5)
    parser.add_argument("--merge-weight", type=float, default=0.5)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    output_dir = args.output_dir or PROJECT_ROOT / "outputs" / "manifesto_embedding_fno_leaf_grid" / _now_stamp()
    output_dir.mkdir(parents=True, exist_ok=True)
    _preload_transformers_for_local_embedding(args)

    dim = _DIM_FROM_NAME[args.dimension]
    rows = _read_jsonl(args.source_results)
    rows_by_id = {_row_manifesto_id(row): row for row in rows if _row_manifesto_id(row)}
    run_metadata = _load_run_metadata(args.source_report)
    if args.split_source == "phase3":
        split_ids = _phase3_split_examples(
            dimension=dim,
            train_n=int(args.train_n),
            val_n=int(args.val_n),
            test_n=int(args.test_n),
            seed=int(args.seed),
            split_strategy=str(args.split_strategy),
            train_pool=str(args.train_pool),
            mp_data_dir=args.mp_data_dir,
        )
    else:
        split_ids = _order_split_rows(
            rows_by_id,
            train_n=int(args.train_n),
            val_n=int(args.val_n),
            test_n=int(args.test_n),
            seed=int(args.seed),
        )
    _write_json(output_dir / "split_ids.json", {split: sorted(ids) for split, ids in split_ids.items()})
    embedding_client = _make_embedding_client(args)
    leaf_grid = _parse_int_grid(args.leaf_grid)
    train_cfg = TrainConfig(train_splits=("train",), epochs=int(args.epochs), batch_size=int(args.batch_size))
    val_cfg = ValidationConfig(val_splits=("val",), enabled=True, eval_every=1)
    test_cfg = TestConfig(test_splits=("test",), enabled=True)
    runtime_cfg = RuntimeConfig(device=str(args.device), bf16=False, gradient_checkpointing=False)

    aggregate_rows: List[Dict[str, Any]] = []
    run_entries: Dict[str, Any] = {}
    for leaf_count in leaf_grid:
        leaf_dir = output_dir / f"leaf_{int(leaf_count):03d}"
        summary_path = leaf_dir / "summary.json"
        if args.skip_existing and summary_path.exists():
            run_entries[str(leaf_count)] = json.loads(summary_path.read_text(encoding="utf-8"))
            continue
        leaf_dir.mkdir(parents=True, exist_ok=True)
        trees, tree_counts = _build_trees_for_leaf_count(
            rows=rows,
            split_ids=split_ids,
            dimension=dim.value,
            target_source=str(args.target_source),
            target_leaves_per_doc=int(leaf_count),
            source_results=args.source_results,
            source_report=args.source_report,
            mp_data_dir=args.mp_data_dir,
        )
        if not trees:
            raise SystemExit(f"No labeled trees built for leaf_count={leaf_count}")
        labeled_tree_path = write_labeled_trees_jsonl(leaf_dir / "labeled_trees.jsonl", trees)
        contract = TreePOContractSpec(
            contract_id=f"manifesto_{dim.value}_embedding_fno_leaf_{int(leaf_count)}",
            objective_kind="embedding_fno_node_distillation",
            state_semantics="embedding_coordinate_function",
            adapter_preference="embedding_fno_node_distillation",
            rubric=get_preservation_rubric(dim),
            oracle_scale_min=1.0,
            oracle_scale_max=7.0,
            metadata={
                "dimension": dim.value,
                "target_source": str(args.target_source),
                "target_leaves_per_doc": int(leaf_count),
                "fno_spatial_axis": "embedding_dimension",
                "topology_axis": "leaf_count",
                "source_results": str(args.source_results),
            },
        )
        model = TreePOModelSpec(kind="embedding_coordinate_fno_tree_operator", model="embedding_fno_tree_operator")
        fno_cfg = EmbeddingFNOTrainConfig(
            run=RunConfig(output_dir=leaf_dir / "fit", seed=int(args.seed)),
            train=train_cfg,
            validation=val_cfg,
            test=test_cfg,
            runtime=runtime_cfg,
            optimizer=OptimizerConfig(
                learning_rate=float(args.learning_rate),
                weight_decay=float(args.weight_decay),
                grad_clip_norm=float(args.grad_clip_norm),
            ),
            model=EmbeddingFNOModelConfig(
                hidden_channels=int(args.hidden_channels),
                n_modes=int(args.n_modes),
                n_layers=int(args.n_layers),
                head_hidden_dim=int(args.head_hidden_dim),
                target_min=1.0,
                target_max=7.0,
            ),
            objective=EmbeddingFNOObjectiveConfig(
                root_weight=float(args.root_weight),
                leaf_weight=float(args.leaf_weight),
                merge_weight=float(args.merge_weight),
            ),
        )
        result = fit_treepo_contract(
            contract=contract,
            model=model,
            run=replace(fno_cfg.run, output_dir=leaf_dir / "fit"),
            train=train_cfg,
            validation=val_cfg,
            test=test_cfg,
            data={"labeled_trees": trees, "embedding_fno_config": fno_cfg},
            output_dir=leaf_dir / "contract_fit",
            resources={RESOURCE_EMBEDDING: TreePOResourceSpec(kind="object", value=embedding_client)},
        )
        entry = {
            "leaf_count": int(leaf_count),
            "tree_counts": tree_counts,
            "labeled_trees": str(labeled_tree_path),
            "contract_result": result.to_dict(),
        }
        _write_json(summary_path, entry)
        run_entries[str(leaf_count)] = entry

        fit_metrics = result.metrics.get("fit_metrics", {})
        for split in ("train", "val", "test"):
            metrics = fit_metrics.get(split) or {}
            aggregate_rows.append(
                {
                    "leaf_count": int(leaf_count),
                    "split": split,
                    "tree_count": metrics.get("count_trees"),
                    "node_count": metrics.get("count_nodes"),
                    "node_mae_1_7": metrics.get("node_mae_1_7"),
                    "leaf_mae_1_7": metrics.get("leaf_mae_1_7"),
                    "merge_mae_1_7": metrics.get("merge_mae_1_7"),
                    "root_teacher_pearson": (metrics.get("root_teacher_report") or {}).get("pearson_r"),
                    "root_teacher_mae_1_7": (metrics.get("root_teacher_report") or {}).get("mae_1_7"),
                    "root_expert_pearson": (metrics.get("root_expert_report") or {}).get("pearson_r"),
                    "root_expert_mae_1_7": (metrics.get("root_expert_report") or {}).get("mae_1_7"),
                    "prediction_path": metrics.get("prediction_path"),
                }
            )
        LOGGER.info("Completed leaf_count=%d", int(leaf_count))

    with (output_dir / "aggregate_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in aggregate_rows:
            handle.write(json.dumps(config_to_dict(row), sort_keys=True) + "\n")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "dimension": dim.value,
        "source_results": str(args.source_results),
        "source_report": str(args.source_report),
        "run_metadata": run_metadata,
        "config": {
            "target_source": str(args.target_source),
            "leaf_grid": list(leaf_grid),
            "fno_spatial_axis": "embedding_dimension",
            "topology_axis": "leaf_count",
            "embedding_backend": str(args.embedding_backend),
            "embedding_model": str(args.embedding_model),
            "train_n": int(args.train_n),
            "val_n": int(args.val_n),
            "test_n": int(args.test_n),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "hidden_channels": int(args.hidden_channels),
            "n_modes": int(args.n_modes),
            "n_layers": int(args.n_layers),
        },
        "artifacts": {
            "split_ids": str(output_dir / "split_ids.json"),
            "aggregate_metrics": str(output_dir / "aggregate_metrics.jsonl"),
        },
        "runs": run_entries,
    }
    _write_json(output_dir / "manifest.json", manifest)
    LOGGER.info("Wrote embedding-FNO leaf grid to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
