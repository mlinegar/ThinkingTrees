from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from src.preprocessing.chunker import chunk_for_ops
from src.tasks.manifesto import ManifestoDataset
from src.tasks.manifesto.expert_scale import (
    expert_scale_metadata,
    raw_benoit_expert_from_row,
)
from src.tasks.manifesto.result_rows import (
    get_text_for_row,
    row_expert_score,
    row_manifesto_id,
    row_summary,
    row_target_score,
    row_teacher_score,
)
from src.tree.labeled import LabeledNode, LabeledTree


def add_labeled_node(
    tree: LabeledTree,
    *,
    node_id: str,
    level: int,
    text: str,
    score: float,
    char_start: int,
    char_end: int,
    is_leaf: bool,
    label_source: str,
    left_child_id: Optional[str] = None,
    right_child_id: Optional[str] = None,
    teacher_summary: Optional[str] = None,
    summary_source: Optional[str] = None,
) -> None:
    metadata: Dict[str, Any] = {
        "char_start": int(char_start),
        "char_end": int(char_end),
        "node_id": str(node_id),
        "is_leaf": bool(is_leaf),
        "label_source": str(label_source),
        "g_training_role": "leaf" if is_leaf else "merge",
        "f_input_kind": "summary_embedding",
    }
    if teacher_summary:
        metadata["teacher_summary"] = str(teacher_summary)
        metadata["target_summary"] = str(teacher_summary)
        metadata["teacher_summary_source"] = str(summary_source or "existing_gemma_result_root")
        if is_leaf:
            metadata["teacher_leaf_summary"] = str(teacher_summary)
        else:
            metadata["teacher_merge_summary"] = str(teacher_summary)
    else:
        metadata["missing_teacher_summary"] = True
    tree.add_node(
        LabeledNode(
            node_id=str(node_id),
            doc_id=tree.doc_id,
            level=int(level),
            text=str(text),
            score=float(score),
            left_child_id=left_child_id,
            right_child_id=right_child_id,
            metadata=metadata,
        )
    )


def build_partial_labeled_tree(
    *,
    row: Mapping[str, Any],
    text: str,
    split: str,
    dimension: str,
    target_source: str,
    expert_target_scale: str,
    chunk_chars: int,
    source_results_path: Path,
) -> Optional[LabeledTree]:
    mid = row_manifesto_id(row)
    summary = row_summary(row)
    target = row_target_score(
        row,
        dimension=dimension,
        target_source=target_source,
        expert_scale=expert_target_scale,
    )
    teacher_score = row_teacher_score(row, dimension=dimension)
    expert_score = row_expert_score(row, dimension=dimension, expert_scale=expert_target_scale)
    expert_raw = raw_benoit_expert_from_row(row, dimension=dimension)
    if not mid or not text.strip() or not summary or target is None:
        return None

    chunks = chunk_for_ops(text, max_chars=int(chunk_chars), strategy="axis")
    if not chunks:
        return None
    tree = LabeledTree(
        doc_id=str(mid),
        document_text=str(text),
        document_score=float(target),
        label_source=f"existing_gemma4_{target_source}",
        metadata={
            "artifact_version": "manifesto_dimension_existing_results_v1",
            "split": str(split),
            "dimension": str(dimension),
            "target_source": str(target_source),
            "teacher_score_1_7": teacher_score,
            "expert_score_1_7": expert_score,
            "expert_score_raw_benoit": expert_raw,
            **expert_scale_metadata(dimension=dimension, scale=expert_target_scale),
            "leaf_size_chars": int(chunk_chars),
            "chunking_source": "src.preprocessing.chunker.chunk_for_ops(strategy='axis')",
            "topology_policy": {
                "kind": "existing_phase_fixed_char_windows",
                "leaf_size_chars": int(chunk_chars),
                "actual_leaves": int(len(chunks)),
            },
            "topology_replay": "exact_artifact_spans",
            "source_results_path": str(source_results_path),
            "partial_artifact": True,
            "partial_artifact_reason": "existing result rows contain root summaries/scores but not all node summaries",
            "paper_to_lean_local_law_mapping": {
                "leaf": "C1_sufficiency",
                "idempotence": "C2_idempotence",
                "merge": "C3_associativity",
            },
        },
    )
    current: list[tuple[str, int, int, int]] = []
    tree.levels = []
    for idx, chunk in enumerate(chunks):
        node_id = f"node_l0_{idx:05d}"
        start = int(getattr(chunk, "start_char", 0))
        end = int(getattr(chunk, "end_char", start + len(str(chunk.text))))
        add_labeled_node(
            tree,
            node_id=node_id,
            level=0,
            text=str(chunk.text),
            score=float(target),
            char_start=start,
            char_end=end,
            is_leaf=True,
            label_source=tree.label_source,
        )
        current.append((node_id, start, end, 0))

    level = 1
    sibling_triples: list[dict[str, str]] = []
    while len(current) > 1:
        next_level: list[tuple[str, int, int, int]] = []
        for pair_idx in range(0, len(current), 2):
            left = current[pair_idx]
            right = current[pair_idx + 1] if pair_idx + 1 < len(current) else left
            node_id = f"node_l{level}_{len(next_level):05d}"
            start = int(left[1])
            end = int(right[2])
            parent_text = text[start:end]
            is_root = len(current) <= 2
            add_labeled_node(
                tree,
                node_id=node_id,
                level=level,
                text=parent_text,
                score=float(target),
                char_start=start,
                char_end=end,
                is_leaf=False,
                label_source=tree.label_source,
                left_child_id=left[0],
                right_child_id=right[0],
                teacher_summary=summary if is_root else None,
                summary_source="existing_gemma_result_root" if is_root else None,
            )
            sibling_triples.append(
                {
                    "left_node_id": str(left[0]),
                    "right_node_id": str(right[0]),
                    "parent_node_id": str(node_id),
                }
            )
            next_level.append((node_id, start, end, level))
        current = next_level
        level += 1

    if len(chunks) == 1:
        only_id = "node_l0_00000"
        node = tree.get_node(only_id)
        if node is not None:
            node.metadata["teacher_summary"] = summary
            node.metadata["target_summary"] = summary
            node.metadata["teacher_summary_source"] = "existing_gemma_result_root"
            node.metadata.pop("missing_teacher_summary", None)

    tree.metadata["sibling_triples"] = sibling_triples
    tree.metadata["idempotence_pairs"] = []
    return tree


def build_labeled_trees(
    *,
    rows: Sequence[Mapping[str, Any]],
    split_ids: Mapping[str, Mapping[str, str]],
    dimension: str,
    target_source: str,
    expert_target_scale: str,
    chunk_chars: int,
    source_results_path: Path,
    mp_data_dir: Optional[Path],
) -> Tuple[list[LabeledTree], dict[str, Any]]:
    rows_by_id = {row_manifesto_id(row): row for row in rows if row_manifesto_id(row)}
    dataset: Optional[ManifestoDataset] = None
    trees: list[LabeledTree] = []
    skipped: dict[str, int] = {"missing_row": 0, "missing_text": 0, "missing_target_or_summary": 0}

    if any(not text for split in split_ids.values() for text in split.values()):
        dataset = ManifestoDataset(data_dir=mp_data_dir, require_text=True)

    for split, id_to_text in split_ids.items():
        for mid, split_text in id_to_text.items():
            row = rows_by_id.get(str(mid))
            if row is None:
                skipped["missing_row"] += 1
                continue
            text = get_text_for_row(row=row, split_texts={str(mid): split_text}, dataset=dataset)
            if not text.strip():
                skipped["missing_text"] += 1
                continue
            tree = build_partial_labeled_tree(
                row=row,
                text=text,
                split=str(split),
                dimension=dimension,
                target_source=target_source,
                expert_target_scale=expert_target_scale,
                chunk_chars=int(chunk_chars),
                source_results_path=source_results_path,
            )
            if tree is None:
                skipped["missing_target_or_summary"] += 1
                continue
            trees.append(tree)
    counts = {
        "total": len(trees),
        "train": sum(1 for tree in trees if tree.metadata.get("split") == "train"),
        "val": sum(1 for tree in trees if tree.metadata.get("split") == "val"),
        "test": sum(1 for tree in trees if tree.metadata.get("split") == "test"),
        "skipped": skipped,
    }
    return trees, counts


__all__ = ["add_labeled_node", "build_labeled_trees", "build_partial_labeled_tree"]
