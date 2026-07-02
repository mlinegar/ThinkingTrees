#!/usr/bin/env python3
"""Extract one scalar dimension from vector Manifesto labeled trees.

Joint all-six teacher runs store every node's scores in ``dimension_scores`` /
``teacher_dimension_scores_1_7`` and root expert labels in
``expert_dimension_scores_1_7``. The scalar DSPy f/g family expects the active
dimension in ``node.score`` and ``tree.metadata["expert_score_1_7"]``. This
script rewrites that view without any model calls.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ctreepo.distillation import load_labeled_trees, write_labeled_trees_jsonl
from src.tree.labeled import LabeledNode, LabeledTree
from src.training.config_sections import config_to_dict


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _parse_ints(value: str) -> list[int]:
    out = [int(part.strip()) for part in str(value).replace(";", ",").split(",") if part.strip()]
    if not out:
        raise ValueError("expected at least one integer leaf size")
    return out


def _root_node(tree: LabeledTree) -> Optional[LabeledNode]:
    for level_ids in reversed(tree.levels or []):
        for node_id in level_ids:
            node = tree.get_node(str(node_id))
            if node is not None:
                return node
    return None


def _score_from_mapping(mapping: Any, dimension: str) -> Optional[float]:
    if not isinstance(mapping, Mapping):
        return None
    return _safe_float(mapping.get(dimension))


def _node_dimension_score(node: LabeledNode, dimension: str) -> Optional[float]:
    score = _score_from_mapping(getattr(node, "dimension_scores", None), dimension)
    if score is not None:
        return score
    metadata = dict(node.metadata or {})
    for key in ("teacher_dimension_scores_1_7", "dimension_scores"):
        score = _score_from_mapping(metadata.get(key), dimension)
        if score is not None:
            return score
    return None


def _tree_expert_score(tree: LabeledTree, dimension: str) -> Optional[float]:
    metadata = dict(tree.metadata or {})
    for key in ("expert_dimension_scores_1_7", "expert_means"):
        score = _score_from_mapping(metadata.get(key), dimension)
        if score is not None:
            return score
    if str(metadata.get("dimension") or "") == dimension:
        return _safe_float(metadata.get("expert_score_1_7"))
    return None


def _extract_leaf(
    *,
    input_dir: Path,
    output_dir: Path,
    leaf_size: int,
    dimension: str,
    missing_policy: str,
) -> dict[str, Any]:
    source_path = input_dir / f"leaf{int(leaf_size):04d}tok" / "labeled_trees.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(f"missing source labeled trees: {source_path}")
    trees = load_labeled_trees(source_path)
    out_trees: list[LabeledTree] = []
    skipped = {"missing_root_teacher": 0, "missing_expert": 0}
    missing_nodes = 0
    total_nodes = 0
    for tree in trees:
        root = _root_node(tree)
        root_score = _node_dimension_score(root, dimension) if root is not None else None
        expert_score = _tree_expert_score(tree, dimension)
        if root_score is None and missing_policy == "skip-tree":
            skipped["missing_root_teacher"] += 1
            continue
        if expert_score is None and missing_policy == "skip-tree":
            skipped["missing_expert"] += 1
            continue

        fallback_score = root_score
        if fallback_score is None:
            fallback_score = _safe_float(tree.document_score)
        if fallback_score is None:
            fallback_score = 4.0

        for node in tree.nodes.values():
            total_nodes += 1
            score = _node_dimension_score(node, dimension)
            if score is None:
                missing_nodes += 1
                score = fallback_score
            metadata = dict(node.metadata or {})
            metadata["source_vector_dimension"] = dimension
            metadata["source_vector_score"] = node.score
            metadata["dimension"] = dimension
            metadata["teacher_score_1_7"] = float(score)
            node.metadata = metadata
            node.score = float(score)
            node.dimension_scores = {dimension: float(score)}

        tree_metadata = dict(tree.metadata or {})
        tree_metadata["source_vector_dimension"] = dimension
        tree_metadata["source_vector_document_score"] = tree.document_score
        tree_metadata["dimension"] = dimension
        tree_metadata["teacher_score_1_7"] = float(fallback_score)
        tree_metadata["expert_score_1_7"] = float(expert_score) if expert_score is not None else None
        tree_metadata["label_source"] = f"manifesto_{dimension}_from_joint_vector_teacher_v1"
        tree_metadata["source_vector_labeled_trees_path"] = str(source_path)
        tree.metadata = tree_metadata
        tree.document_score = float(fallback_score)
        tree.label_source = f"manifesto_{dimension}_from_joint_vector_teacher_v1"
        out_trees.append(tree)

    leaf_dir = output_dir / f"leaf{int(leaf_size):04d}tok"
    labeled_path = write_labeled_trees_jsonl(leaf_dir / "labeled_trees.jsonl", out_trees)
    summary = {
        "dimension": dimension,
        "leaf_size_tokens": int(leaf_size),
        "source_labeled_trees": str(source_path),
        "artifacts": {"labeled_trees": str(labeled_path)},
        "tree_counts": {
            "total": len(out_trees),
            "train": sum(1 for tree in out_trees if (tree.metadata or {}).get("split") == "train"),
            "val": sum(1 for tree in out_trees if (tree.metadata or {}).get("split") == "val"),
            "test": sum(1 for tree in out_trees if (tree.metadata or {}).get("split") == "test"),
            "skipped": skipped,
        },
        "node_counts": {
            "total": int(total_nodes),
            "missing_dimension_score_filled": int(missing_nodes),
        },
    }
    (leaf_dir / "summary.json").write_text(
        json.dumps(config_to_dict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dimension", required=True)
    parser.add_argument("--leaf-size-tokens", default="256,512,1024,2048,4096,8192")
    parser.add_argument(
        "--missing-policy",
        choices=("fill-neutral", "skip-tree"),
        default="skip-tree",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    leaf_sizes = _parse_ints(str(args.leaf_size_tokens))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = {
        f"tok_{leaf_size}": _extract_leaf(
            input_dir=Path(args.input_dir),
            output_dir=output_dir,
            leaf_size=int(leaf_size),
            dimension=str(args.dimension),
            missing_policy=str(args.missing_policy),
        )
        for leaf_size in leaf_sizes
    }
    manifest = {
        "dimension": str(args.dimension),
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "leaf_size_tokens": leaf_sizes,
        "runs": runs,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(config_to_dict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(config_to_dict(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
