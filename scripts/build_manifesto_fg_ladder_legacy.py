#!/usr/bin/env python3
"""DEPRECATED - legacy one-shot f/g ladder for manifesto teacher traces.

This script encodes the *wrong* semantics for fgfg:
- fgf / fgfg are trained as independent parallel fits from the same teacher
  traces rather than as iterations of an alternating optimization.
- The only supported embedding backend is hashing (a 256-dim lexical-hash
  bag-of-signs), so every trained student Pearson caps far below teacher.

Do NOT re-run. Use scripts/run_manifesto_fg_real_training_grid.py instead,
which implements the alternating loop (f_init, g_init) -> f1 -> g1 -> f2 ...
with the current student f providing the scoring signal for g training,
across three symmetric backend families (DSPy, TRL, FNO).

Kept for reference only. The original docstring follows.

---

Build a tree-aligned f/g composition ladder for manifesto teacher traces.

The ladder is intentionally explicit about composition:

* f: score an already available baseline g representation.
* fg: run f after teacher g on each artifact node.
* fgf: learn/export f from the fg node representation.
* fgfg: learn/export g from the same nodes and pair it with the learned f.

All training/export stages route through the contract runner and the
sectioned distillation config.  The script does not call a teacher; it consumes
already materialized LabeledTree artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_manifesto_dimension_fit_existing_results import (  # noqa: E402
    _DIM_FROM_NAME,
    _make_embedding_client,
    _preload_transformers_for_local_embedding,
    _read_jsonl,
    _row_expert_score,
    _row_manifesto_id,
    _row_summary,
    _row_teacher_score,
)
from src.ctreepo.distillation import (  # noqa: E402
    DistillationContractConfig,
    DistillationTrainConfig,
    FEmbeddingConfig,
    FLMConfig,
    GLMConfig,
    ScoreTargetConfig,
    SummaryTargetConfig,
    TRAIN_TARGET_F,
    TRAIN_TARGET_G,
    STUDENT_MODEL_EMBEDDING_RIDGE_PROXY,
    STUDENT_MODEL_LM_SCALAR_REGRESSION,
    STUDENT_MODEL_LM_SFT,
    SUPERVISION_SOURCE_LABELED_TREE_ARTIFACT,
    load_labeled_trees,
)
from src.tasks.manifesto.corpus_metrics import compute_corpus_pearson_r  # noqa: E402
from src.tasks.manifesto.dimensions import get_preservation_rubric  # noqa: E402
from src.training.config_sections import (  # noqa: E402
    RunConfig,
    TestConfig,
    TrainConfig,
    ValidationConfig,
    config_to_dict,
)
from src.tree.contract_runner import (  # noqa: E402
    RESOURCE_EMBEDDING,
    TreePOResourceSpec,
    fit_treepo_contract,
)
from src.tree.labeled import LabeledNode, LabeledTree  # noqa: E402
from src.tree.treepo_stack import TreePOContractSpec, TreePOModelSpec  # noqa: E402


LOGGER = logging.getLogger(__name__)

DEFAULT_SOURCE_RESULTS = (
    PROJECT_ROOT / "outputs" / "overnight_benoit" / "full_pipeline" / "economic" / "per_manifesto.jsonl"
)
DEFAULT_FG_GRID_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "manifesto_teacher_fg_leaf_grid"
    / "economic_gemma4_aligned_l1_2_4_8_16"
)
DEFAULT_F_BASELINE_TREES = (
    PROJECT_ROOT
    / "outputs"
    / "manifesto_teacher_fg_leaf_grid"
    / "economic_gemma4_f_existing_summary_leaf1"
    / "leaf_001"
    / "labeled_trees.jsonl"
)
DEFAULT_F_DOC_TREES = (
    PROJECT_ROOT
    / "outputs"
    / "manifesto_teacher_fg_leaf_grid"
    / "economic_gemma4_f_only_leaf1"
    / "leaf_001"
    / "labeled_trees.jsonl"
)


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_to_dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _parse_int_grid(value: Any) -> Tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        grid = tuple(int(item) for item in value)
    else:
        parts = [part.strip() for part in str(value or "").replace(";", ",").split(",")]
        grid = tuple(int(part) for part in parts if part)
    if not grid:
        raise ValueError("leaf grid must contain at least one integer")
    if any(item <= 0 for item in grid):
        raise ValueError(f"leaf grid entries must be positive: {grid!r}")
    return grid


def _parse_stages(value: str) -> Tuple[str, ...]:
    stages = tuple(
        part.strip().lower()
        for part in str(value or "").replace("+", ",").replace(";", ",").split(",")
        if part.strip()
    )
    allowed = {"f", "f_doc", "fdoc", "fg", "fgf", "fgfg"}
    unknown = [stage for stage in stages if stage not in allowed]
    if unknown:
        raise ValueError(f"unknown ladder stage(s): {unknown!r}")
    canonical = tuple("f_doc" if stage == "fdoc" else stage for stage in stages)
    return canonical or ("f", "f_doc", "fg", "fgf", "fgfg")


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    return converted


def _root_node(tree: LabeledTree) -> Optional[LabeledNode]:
    for level_ids in reversed(tree.levels or []):
        for node_id in level_ids:
            node = tree.get_node(str(node_id))
            if node is not None:
                return node
    return None


def _split_counts(trees: Sequence[LabeledTree]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for tree in trees:
        split = str((tree.metadata or {}).get("split") or "unknown")
        counts[split] = counts.get(split, 0) + 1
    counts["total"] = len(trees)
    return counts


def _score_metric(rows: Sequence[Mapping[str, Any]], *, pred_key: str, truth_key: str) -> Dict[str, Any]:
    preds: List[float] = []
    truths: List[float] = []
    for row in rows:
        pred = _safe_float(row.get(pred_key))
        truth = _safe_float(row.get(truth_key))
        if pred is None or truth is None:
            continue
        preds.append(pred)
        truths.append(truth)
    if len(preds) >= 4:
        payload = compute_corpus_pearson_r(preds, truths).as_dict()
    else:
        payload = {"n": len(preds), "pearson_r": None}
    if preds:
        payload["mae_1_7"] = float(sum(abs(p - t) for p, t in zip(preds, truths)) / len(preds))
        payload["mean_prediction_1_7"] = float(sum(preds) / len(preds))
        payload["mean_truth_1_7"] = float(sum(truths) / len(truths))
    else:
        payload["mae_1_7"] = None
        payload["mean_prediction_1_7"] = None
        payload["mean_truth_1_7"] = None
    return payload


def _root_score_rows(trees: Sequence[LabeledTree]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for tree in trees:
        root = _root_node(tree)
        if root is None:
            continue
        metadata = dict(tree.metadata or {})
        existing_teacher = _safe_float(
            metadata.get("teacher_score_1_7_existing_root", metadata.get("teacher_score_1_7"))
        )
        expert = _safe_float(metadata.get("expert_score_1_7"))
        rows.append(
            {
                "doc_id": tree.doc_id,
                "split": str(metadata.get("split") or ""),
                "root_score_1_7": _safe_float(root.score),
                "document_score_1_7": _safe_float(tree.document_score),
                "existing_teacher_score_1_7": existing_teacher,
                "expert_score_1_7": expert,
            }
        )
    return rows


def _source_result_metrics(path: Optional[Path], *, dimension: str) -> Dict[str, Any]:
    if path is None or not Path(path).exists():
        return {"available": False}
    rows = _read_jsonl(Path(path))
    metric_rows: List[Dict[str, Any]] = []
    for row in rows:
        teacher = _row_teacher_score(row, dimension=dimension)
        expert = _row_expert_score(row, dimension=dimension)
        metric_rows.append(
            {
                "doc_id": _row_manifesto_id(row),
                "teacher_score_1_7": teacher,
                "expert_score_1_7": expert,
            }
        )
    return {
        "available": True,
        "path": str(path),
        "teacher_vs_expert": _score_metric(
            metric_rows,
            pred_key="teacher_score_1_7",
            truth_key="expert_score_1_7",
        ),
    }


def _load_leaf_trees(fg_grid_dir: Path, leaf_count: int) -> Optional[List[LabeledTree]]:
    path = Path(fg_grid_dir) / f"leaf_{int(leaf_count):03d}" / "labeled_trees.jsonl"
    if not path.exists():
        return None
    return load_labeled_trees(path)


def _split_by_doc_from_grid(fg_grid_dir: Path, leaf_grid: Sequence[int]) -> Dict[str, str]:
    for leaf_count in leaf_grid:
        trees = _load_leaf_trees(fg_grid_dir, int(leaf_count))
        if trees:
            return {
                str(tree.doc_id): str((tree.metadata or {}).get("split") or "train")
                for tree in trees
            }
    return {}


def _build_baseline_summary_trees_from_source(
    source_results: Path,
    *,
    dimension: str,
    split_by_doc: Mapping[str, str],
) -> List[LabeledTree]:
    trees: List[LabeledTree] = []
    for row in _read_jsonl(source_results):
        doc_id = _row_manifesto_id(row)
        summary = _row_summary(row)
        score = _row_teacher_score(row, dimension=dimension)
        expert = _row_expert_score(row, dimension=dimension)
        if not doc_id or not summary or score is None:
            continue
        split = str(split_by_doc.get(doc_id) or "train")
        tree = LabeledTree(
            doc_id=doc_id,
            document_text=summary,
            document_score=float(score),
            label_source="existing_baseline_g_summary_score",
            metadata={
                "split": split,
                "dimension": dimension,
                "teacher_score_1_7": float(score),
                "expert_score_1_7": expert,
                "summary_representation": "existing_baseline_g_summary",
                "topology_axis": "baseline_root_summary",
            },
        )
        node = LabeledNode(
            node_id="node_l0_00000",
            doc_id=doc_id,
            level=0,
            text=summary,
            score=float(score),
            dimension_scores={dimension: float(score)},
            metadata={
                "is_leaf": True,
                "teacher_summary": summary,
                "target_summary": summary,
                "teacher_summary_source": "existing_baseline_g_summary",
                "teacher_score_1_7": float(score),
                "teacher_score_source": "existing_baseline_f_score",
                "g_training_role": "baseline_root",
                "f_input_kind": "summary_embedding",
            },
        )
        tree.add_node(node)
        trees.append(tree)
    return trees


def _load_f_baseline_trees(
    *,
    f_baseline_labeled_trees: Optional[Path],
    source_results: Optional[Path],
    dimension: str,
    split_by_doc: Mapping[str, str],
) -> Tuple[List[LabeledTree], Dict[str, Any]]:
    if f_baseline_labeled_trees is not None and Path(f_baseline_labeled_trees).exists():
        trees = load_labeled_trees(Path(f_baseline_labeled_trees))
        return trees, {
            "kind": "labeled_tree_artifact",
            "path": str(f_baseline_labeled_trees),
            "score_interpretation": "f(existing_baseline_g_summary)",
        }
    if source_results is None or not Path(source_results).exists():
        return [], {"kind": "missing"}
    trees = _build_baseline_summary_trees_from_source(
        Path(source_results),
        dimension=dimension,
        split_by_doc=split_by_doc,
    )
    return trees, {
        "kind": "source_result_projection",
        "path": str(source_results),
        "score_interpretation": "existing stored f(g0(document)) score",
    }


def _load_f_doc_trees(f_doc_labeled_trees: Optional[Path]) -> Tuple[List[LabeledTree], Dict[str, Any]]:
    if f_doc_labeled_trees is not None and Path(f_doc_labeled_trees).exists():
        trees = load_labeled_trees(Path(f_doc_labeled_trees))
        return trees, {
            "kind": "labeled_tree_artifact",
            "path": str(f_doc_labeled_trees),
            "score_interpretation": "f(raw_whole_document)",
        }
    return [], {
        "kind": "missing",
        "score_interpretation": "f(raw_whole_document)",
    }


def _base_contract(*, dimension: str, contract_id: str, metadata: Mapping[str, Any]) -> TreePOContractSpec:
    dim = _DIM_FROM_NAME[dimension]
    return TreePOContractSpec(
        contract_id=contract_id,
        objective_kind="labeled_tree_distillation",
        state_semantics="natural_language_summary_state",
        adapter_preference="labeled_tree_distillation",
        rubric=get_preservation_rubric(dim),
        oracle_scale_min=1.0,
        oracle_scale_max=7.0,
        metadata=dict(metadata),
    )


def _fit_stage(
    *,
    trees: Sequence[LabeledTree],
    dimension: str,
    stage_name: str,
    fit_name: str,
    output_dir: Path,
    train: TrainConfig,
    validation: ValidationConfig,
    test: TestConfig,
    train_target: str,
    student_model_class: str,
    include_identity_targets: bool,
    teacher_model_spec: Mapping[str, Any],
    embedding_client: Any = None,
    f_method: str = "ridge",
    ridge_lambda: float = 1.0,
    f_epochs: int = 25,
    f_learning_rate: float = 5e-3,
    f_weight_decay: float = 1e-4,
) -> Dict[str, Any]:
    stage_dir = Path(output_dir) / stage_name / fit_name
    run_dir = stage_dir / "fit"
    contract = DistillationContractConfig(
        train_targets=(train_target,),
        student_model_class=student_model_class,
        supervision_source=SUPERVISION_SOURCE_LABELED_TREE_ARTIFACT,
        teacher_model_spec=dict(teacher_model_spec),
    )
    kwargs: Dict[str, Any] = {}
    if train_target == TRAIN_TARGET_G:
        kwargs["summary_targets"] = SummaryTargetConfig(
            include_identity_targets=bool(include_identity_targets),
        )
        kwargs["g_lm"] = GLMConfig(run_trl_sft=False)
    else:
        kwargs["score_targets"] = ScoreTargetConfig(
            include_identity_targets=bool(include_identity_targets),
            target_min=1.0,
            target_max=7.0,
        )
        if student_model_class == STUDENT_MODEL_EMBEDDING_RIDGE_PROXY:
            kwargs["f_embedding"] = FEmbeddingConfig(
                method=str(f_method),
                ridge_lambda=float(ridge_lambda),
                epochs=int(f_epochs),
                learning_rate=float(f_learning_rate),
                weight_decay=float(f_weight_decay),
                model_id=f"manifesto_{dimension}_{stage_name}_{fit_name}",
            )
        elif student_model_class == STUDENT_MODEL_LM_SCALAR_REGRESSION:
            kwargs["f_lm"] = FLMConfig(run_trl_scalar_reward=False)

    distillation_config = DistillationTrainConfig(
        contract=contract,
        run=RunConfig(output_dir=run_dir),
        train=train,
        validation=validation,
        test=test,
        **kwargs,
    )
    resources: Dict[str, Any] = {}
    if embedding_client is not None:
        resources[RESOURCE_EMBEDDING] = TreePOResourceSpec(kind="object", value=embedding_client)
    result = fit_treepo_contract(
        contract=_base_contract(
            dimension=dimension,
            contract_id=f"manifesto_{dimension}_{stage_name}_{fit_name}",
            metadata={
                "dimension": dimension,
                "ladder_stage": stage_name,
                "fit_name": fit_name,
                "composition": stage_name,
            },
        ),
        model=TreePOModelSpec(kind="artifact_distillation", model=student_model_class),
        run=RunConfig(output_dir=run_dir),
        train=train,
        validation=validation,
        test=test,
        data={
            "labeled_trees": list(trees),
            "distillation_config": distillation_config,
        },
        output_dir=stage_dir / "contract",
        resources=resources,
    )
    payload = result.to_dict()
    payload["fit_output_dir"] = str(run_dir)
    return payload


def _stage_f(
    *,
    trees: Sequence[LabeledTree],
    source_results: Optional[Path],
    output_dir: Path,
    dimension: str,
    stage_key: str = "f",
    meaning: str = "score an existing baseline g representation",
    right_to_left_alias: str = "f_after_g0",
    left_to_right_alias: str = "g0_then_f",
    train: TrainConfig,
    validation: ValidationConfig,
    test: TestConfig,
    embedding_client: Any,
    args: argparse.Namespace,
    source_info: Mapping[str, Any],
) -> Dict[str, Any]:
    root_rows = _root_score_rows(trees)
    root_vs_expert = _score_metric(
        root_rows,
        pred_key="root_score_1_7",
        truth_key="expert_score_1_7",
    )
    metric_key = f"{stage_key}_root_vs_expert"
    stage: Dict[str, Any] = {
        "composition": {
            "name": str(stage_key),
            "meaning": str(meaning),
            "right_to_left_alias": str(right_to_left_alias),
            "left_to_right_alias": str(left_to_right_alias),
        },
        "source": dict(source_info),
        "tree_counts": _split_counts(trees),
        "metrics": {
            "source_result_teacher_vs_expert": _source_result_metrics(
                source_results,
                dimension=dimension,
            ),
            "root_vs_expert": root_vs_expert,
            metric_key: root_vs_expert,
        },
        "fits": {},
    }
    if stage_key == "f":
        stage["metrics"]["baseline_f_root_vs_expert"] = root_vs_expert
    if not trees:
        return stage

    teacher_spec = {
        "kind": "baseline_g_summary_scores",
        "source": dict(source_info),
        "dimension": dimension,
    }
    if not args.skip_f_lm_export:
        stage["fits"]["f_lm_records"] = _fit_stage(
            trees=trees,
            dimension=dimension,
            stage_name=stage_key,
            fit_name="f_lm_records",
            output_dir=output_dir,
            train=train,
            validation=validation,
            test=test,
            train_target=TRAIN_TARGET_F,
            student_model_class=STUDENT_MODEL_LM_SCALAR_REGRESSION,
            include_identity_targets=True,
            teacher_model_spec=teacher_spec,
        )
    if not args.skip_f_embedding_fit and embedding_client is not None:
        stage["fits"]["f_embedding_proxy"] = _fit_stage(
            trees=trees,
            dimension=dimension,
            stage_name=stage_key,
            fit_name="f_embedding_proxy",
            output_dir=output_dir,
            train=train,
            validation=validation,
            test=test,
            train_target=TRAIN_TARGET_F,
            student_model_class=STUDENT_MODEL_EMBEDDING_RIDGE_PROXY,
            include_identity_targets=True,
            teacher_model_spec=teacher_spec,
            embedding_client=embedding_client,
            f_method=args.f_method,
            ridge_lambda=args.ridge_lambda,
            f_epochs=args.f_epochs,
            f_learning_rate=args.f_learning_rate,
            f_weight_decay=args.f_weight_decay,
        )
    return stage


def _stage_fg_leaf(
    *,
    trees: Sequence[LabeledTree],
    leaf_count: int,
) -> Dict[str, Any]:
    root_rows = _root_score_rows(trees)
    node_count = sum(len(tree.nodes) for tree in trees)
    summary_count = sum(
        1
        for tree in trees
        for node in tree.nodes.values()
        if str((node.metadata or {}).get("teacher_summary") or "").strip()
    )
    return {
        "composition": {
            "name": "fg",
            "meaning": "teacher f after teacher g on the same tree nodes",
            "right_to_left_alias": "f_after_g",
            "left_to_right_alias": "g_then_f",
        },
        "leaf_count": int(leaf_count),
        "tree_counts": _split_counts(trees),
        "node_count": int(node_count),
        "nodes_with_teacher_summary": int(summary_count),
        "metrics": {
            "fg_root_vs_expert": _score_metric(
                root_rows,
                pred_key="root_score_1_7",
                truth_key="expert_score_1_7",
            ),
            "fg_root_vs_existing_teacher_root": _score_metric(
                root_rows,
                pred_key="root_score_1_7",
                truth_key="existing_teacher_score_1_7",
            ),
        },
    }


def _run_leaf_ladder(
    *,
    trees: Sequence[LabeledTree],
    leaf_count: int,
    output_dir: Path,
    dimension: str,
    train: TrainConfig,
    validation: ValidationConfig,
    test: TestConfig,
    embedding_client: Any,
    args: argparse.Namespace,
    stages: Sequence[str],
) -> Dict[str, Any]:
    leaf_key = f"leaf_{int(leaf_count):03d}"
    teacher_spec = {
        "kind": "teacher_fg_node_artifact",
        "dimension": dimension,
        "leaf_count": int(leaf_count),
        "source_artifact": str(Path(args.fg_grid_dir) / leaf_key / "labeled_trees.jsonl"),
    }
    payload: Dict[str, Any] = {
        "leaf_count": int(leaf_count),
        "artifact": str(Path(args.fg_grid_dir) / leaf_key / "labeled_trees.jsonl"),
        "fg": _stage_fg_leaf(trees=trees, leaf_count=leaf_count),
        "fgf": {
            "composition": {
                "name": "fgf",
                "meaning": "learn f from teacher g states and teacher f labels",
                "right_to_left_alias": "fit_f_on_f_after_g_traces",
            },
            "fits": {},
        },
        "fgfg": {
            "composition": {
                "name": "fgfg",
                "meaning": "export g and f students from one shared tree-indexed representation",
                "right_to_left_alias": "fit_g_and_f_on_f_after_g_traces",
            },
            "fits": {},
        },
    }
    needs_f_student = "fgf" in stages or "fgfg" in stages
    needs_g_student = "fgfg" in stages
    if needs_f_student and not args.skip_f_lm_export:
        payload["fgf"]["fits"]["f_lm_records"] = _fit_stage(
            trees=trees,
            dimension=dimension,
            stage_name=f"{leaf_key}_fgf",
            fit_name="f_lm_records",
            output_dir=output_dir,
            train=train,
            validation=validation,
            test=test,
            train_target=TRAIN_TARGET_F,
            student_model_class=STUDENT_MODEL_LM_SCALAR_REGRESSION,
            include_identity_targets=bool(args.include_identity_targets),
            teacher_model_spec=teacher_spec,
        )
    if needs_f_student and not args.skip_f_embedding_fit and embedding_client is not None:
        payload["fgf"]["fits"]["f_embedding_proxy"] = _fit_stage(
            trees=trees,
            dimension=dimension,
            stage_name=f"{leaf_key}_fgf",
            fit_name="f_embedding_proxy",
            output_dir=output_dir,
            train=train,
            validation=validation,
            test=test,
            train_target=TRAIN_TARGET_F,
            student_model_class=STUDENT_MODEL_EMBEDDING_RIDGE_PROXY,
            include_identity_targets=bool(args.include_identity_targets),
            teacher_model_spec=teacher_spec,
            embedding_client=embedding_client,
            f_method=args.f_method,
            ridge_lambda=args.ridge_lambda,
            f_epochs=args.f_epochs,
            f_learning_rate=args.f_learning_rate,
            f_weight_decay=args.f_weight_decay,
        )
    if needs_g_student and not args.skip_g_export:
        payload["fgfg"]["fits"]["g_sft_records"] = _fit_stage(
            trees=trees,
            dimension=dimension,
            stage_name=f"{leaf_key}_fgfg",
            fit_name="g_sft_records",
            output_dir=output_dir,
            train=train,
            validation=validation,
            test=test,
            train_target=TRAIN_TARGET_G,
            student_model_class=STUDENT_MODEL_LM_SFT,
            include_identity_targets=bool(args.include_identity_targets),
            teacher_model_spec=teacher_spec,
        )
        if "f_embedding_proxy" in payload["fgf"]["fits"]:
            payload["fgfg"]["fits"]["paired_f_embedding_proxy"] = payload["fgf"]["fits"][
                "f_embedding_proxy"
            ].get("fit_output_dir")
        if "f_lm_records" in payload["fgf"]["fits"]:
            payload["fgfg"]["fits"]["paired_f_lm_records"] = payload["fgf"]["fits"][
                "f_lm_records"
            ].get("fit_output_dir")
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build f/fg/fgf/fgfg manifesto distillation ladder from LabeledTree artifacts."
    )
    parser.add_argument("--dimension", choices=sorted(_DIM_FROM_NAME), default="economic")
    parser.add_argument("--source-results", type=Path, default=DEFAULT_SOURCE_RESULTS)
    parser.add_argument("--fg-grid-dir", type=Path, default=DEFAULT_FG_GRID_DIR)
    parser.add_argument("--f-baseline-labeled-trees", type=Path, default=DEFAULT_F_BASELINE_TREES)
    parser.add_argument("--f-doc-labeled-trees", type=Path, default=DEFAULT_F_DOC_TREES)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to outputs/manifesto_fg_ladder/<dimension>_<timestamp>.",
    )
    parser.add_argument("--leaf-grid", type=str, default="1,2,4,8,16")
    parser.add_argument("--stages", type=str, default="f,f_doc,fg,fgf,fgfg")
    parser.add_argument("--require-complete-grid", action="store_true")
    parser.add_argument("--skip-f-lm-export", action="store_true")
    parser.add_argument("--skip-f-embedding-fit", action="store_true")
    parser.add_argument("--skip-g-export", action="store_true")
    parser.add_argument("--include-identity-targets", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--embedding-backend", choices=["none", "hashing", "local-hf", "vllm"], default="hashing")
    parser.add_argument("--embedding-model", type=str, default="/mnt/data/models/Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-url", type=str, default=None)
    parser.add_argument("--embedding-api-key", type=str, default="EMPTY")
    parser.add_argument("--embedding-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-max-length", type=int, default=1024)
    parser.add_argument("--embedding-device", default="auto")
    parser.add_argument("--hashing-embedding-dim", type=int, default=256)
    parser.add_argument("--f-method", choices=["ridge", "linear_sgd"], default="ridge")
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--f-epochs", type=int, default=25)
    parser.add_argument("--f-learning-rate", type=float, default=5e-3)
    parser.add_argument("--f-weight-decay", type=float, default=1e-4)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    stages = _parse_stages(args.stages)
    leaf_grid = _parse_int_grid(args.leaf_grid)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "manifesto_fg_ladder" / f"{args.dimension}_{_now_stamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_client = None
    if not args.skip_f_embedding_fit and args.embedding_backend != "none":
        _preload_transformers_for_local_embedding(args)
        embedding_client = _make_embedding_client(args)

    train_cfg = TrainConfig(train_splits=("train",), epochs=1)
    val_cfg = ValidationConfig(val_splits=("val",), enabled=True)
    test_cfg = TestConfig(test_splits=("test",), enabled=True)

    split_by_doc = _split_by_doc_from_grid(args.fg_grid_dir, leaf_grid)
    f_trees, f_source_info = _load_f_baseline_trees(
        f_baseline_labeled_trees=args.f_baseline_labeled_trees,
        source_results=args.source_results,
        dimension=str(args.dimension),
        split_by_doc=split_by_doc,
    )
    f_doc_trees, f_doc_source_info = _load_f_doc_trees(args.f_doc_labeled_trees)

    manifest: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dimension": str(args.dimension),
        "composition_ladder": {
            "f": "score an existing baseline g representation",
            "f_doc": "score the raw whole document directly, with no explicit g summary in front",
            "fg": "teacher f after teacher g on artifact nodes",
            "fgf": "learn/export f from fg node states and labels",
            "fgfg": "learn/export g and f from the same tree-indexed representation",
            "notation": {
                "right_to_left": "fg means f after g",
                "left_to_right": "fg can also be read operationally as g then f",
            },
        },
        "config": {
            "source_results": str(args.source_results) if args.source_results else None,
            "fg_grid_dir": str(args.fg_grid_dir),
            "f_baseline_labeled_trees": (
                str(args.f_baseline_labeled_trees) if args.f_baseline_labeled_trees else None
            ),
            "f_doc_labeled_trees": str(args.f_doc_labeled_trees) if args.f_doc_labeled_trees else None,
            "leaf_grid": list(leaf_grid),
            "stages": list(stages),
            "embedding_backend": str(args.embedding_backend),
            "skip_f_embedding_fit": bool(args.skip_f_embedding_fit),
            "skip_f_lm_export": bool(args.skip_f_lm_export),
            "skip_g_export": bool(args.skip_g_export),
        },
        "f": {},
        "f_doc": {},
        "fg": {},
        "fgf": {},
        "fgfg": {},
        "leaves": {},
        "missing_leaves": [],
        "artifacts": {},
    }

    if "f" in stages:
        LOGGER.info("Building f stage from %s trees", len(f_trees))
        manifest["f"] = _stage_f(
            trees=f_trees,
            source_results=args.source_results,
            output_dir=output_dir,
            dimension=str(args.dimension),
            train=train_cfg,
            validation=val_cfg,
            test=test_cfg,
            embedding_client=embedding_client,
            args=args,
            source_info=f_source_info,
        )

    if "f_doc" in stages:
        LOGGER.info("Building f_doc stage from %s trees", len(f_doc_trees))
        manifest["f_doc"] = _stage_f(
            trees=f_doc_trees,
            source_results=args.source_results,
            output_dir=output_dir,
            dimension=str(args.dimension),
            stage_key="f_doc",
            meaning="score the raw whole document directly, without an explicit g summary state",
            right_to_left_alias="f_after_raw_doc",
            left_to_right_alias="raw_doc_then_f",
            train=train_cfg,
            validation=val_cfg,
            test=test_cfg,
            embedding_client=embedding_client,
            args=args,
            source_info=f_doc_source_info,
        )

    if any(stage in stages for stage in ("fg", "fgf", "fgfg")):
        for leaf_count in leaf_grid:
            leaf_key = f"leaf_{int(leaf_count):03d}"
            trees = _load_leaf_trees(args.fg_grid_dir, int(leaf_count))
            if trees is None:
                manifest["missing_leaves"].append(leaf_key)
                LOGGER.warning("Missing %s; skipping", leaf_key)
                continue
            LOGGER.info("Building ladder for %s from %s trees", leaf_key, len(trees))
            leaf_payload = _run_leaf_ladder(
                trees=trees,
                leaf_count=int(leaf_count),
                output_dir=output_dir,
                dimension=str(args.dimension),
                train=train_cfg,
                validation=val_cfg,
                test=test_cfg,
                embedding_client=embedding_client,
                args=args,
                stages=stages,
            )
            manifest["leaves"][leaf_key] = leaf_payload
            if "fg" in stages:
                manifest["fg"][leaf_key] = leaf_payload["fg"]
            if "fgf" in stages:
                manifest["fgf"][leaf_key] = leaf_payload["fgf"]
            if "fgfg" in stages:
                manifest["fgfg"][leaf_key] = leaf_payload["fgfg"]

    if manifest["missing_leaves"] and args.require_complete_grid:
        raise SystemExit(f"Missing required leaf artifacts: {manifest['missing_leaves']}")

    manifest_path = _write_json(output_dir / "fg_ladder_manifest.json", manifest)
    manifest["artifacts"]["manifest"] = str(manifest_path)
    _write_json(manifest_path, manifest)
    LOGGER.info("Wrote %s", manifest_path)
    print(json.dumps({"manifest": str(manifest_path), "missing_leaves": manifest["missing_leaves"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
