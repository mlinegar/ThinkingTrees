#!/usr/bin/env python3
"""Combine six scalar Manifesto teacher-trace dirs into one vector trace dir.

Input layout:
  <dimension-root>/<dimension>/leaf2048tok/labeled_trees.jsonl

Output layout:
  <output-dir>/leaf2048tok/labeled_trees.jsonl

The resulting ``LabeledTree`` artifacts keep the shared summary topology from
the first dimension and attach ``dimension_scores`` for every node. Root expert
dimension labels are copied from each scalar tree's metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ctreepo.distillation import load_labeled_trees, write_labeled_trees_jsonl
from src.ctreepo.contracts import (
    LEAF_UNIT_TEXT_TOKEN,
    normalize_tree_bundle_manifest,
    tree_bundle_metadata,
)
from src.tree.labeled import LabeledNode, LabeledTree
from src.training.config_sections import config_to_dict


DIMS = (
    "economic",
    "social",
    "immigration",
    "eu",
    "environment",
    "decentralization",
)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(values: Iterable[float]) -> Optional[float]:
    finite = [float(value) for value in values if _safe_float(value) is not None]
    return float(sum(finite) / len(finite)) if finite else None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_to_dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_leaf_trees(root: Path, dim: str, leaf_size: int) -> Dict[str, LabeledTree]:
    path = root / dim / f"leaf{int(leaf_size):04d}tok" / "labeled_trees.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing labeled trees for dimension={dim}: {path}")
    trees = load_labeled_trees(path)
    return {str(tree.doc_id): tree for tree in trees}


def _root_node(tree: LabeledTree) -> Optional[LabeledNode]:
    for level_ids in reversed(tree.levels or []):
        for node_id in level_ids:
            node = tree.get_node(str(node_id))
            if node is not None:
                return node
    return None


def _expert_score(tree: LabeledTree) -> Optional[float]:
    metadata = tree.metadata or {}
    return _safe_float(metadata.get("expert_score_1_7"))


def _teacher_score_for_node(tree: LabeledTree, node_id: str) -> Optional[float]:
    node = tree.get_node(str(node_id))
    if node is None:
        return None
    return _safe_float(node.score)


def _combine_leaf(
    *,
    dimension_root: Path,
    output_dir: Path,
    leaf_size: int,
    dimensions: Sequence[str],
) -> Dict[str, Any]:
    per_dim = {
        dim: _read_leaf_trees(dimension_root, dim, int(leaf_size))
        for dim in dimensions
    }
    anchor_dim = dimensions[0]
    anchor_trees = per_dim[anchor_dim]
    doc_ids = sorted(anchor_trees)
    combined_trees = []
    skipped: Dict[str, int] = {"missing_doc": 0, "topology_mismatch": 0}
    for doc_id in doc_ids:
        dim_trees = {}
        missing = False
        for dim in dimensions:
            tree = per_dim[dim].get(doc_id)
            if tree is None:
                missing = True
                break
            dim_trees[dim] = tree
        if missing:
            skipped["missing_doc"] += 1
            continue

        anchor = dim_trees[anchor_dim]
        node_ids = set(anchor.nodes)
        if any(set(tree.nodes) != node_ids or tree.levels != anchor.levels for tree in dim_trees.values()):
            skipped["topology_mismatch"] += 1
            continue

        expert_scores = {
            dim: float(score)
            for dim, tree in dim_trees.items()
            if (score := _expert_score(tree)) is not None
        }
        for node_id, node in anchor.nodes.items():
            scores = {
                dim: float(score)
                for dim, tree in dim_trees.items()
                if (score := _teacher_score_for_node(tree, node_id)) is not None
            }
            node.dimension_scores = scores
            node.metadata["teacher_dimension_scores_1_7"] = scores
            node.metadata["teacher_score_1_7"] = _mean(scores.values())
            if node.metadata.get("dimension"):
                node.metadata["source_scalar_dimension"] = node.metadata.get("dimension")
            node.metadata["dimension"] = "combined"
            node.score = float(_mean(scores.values()) or 4.0)

        root = _root_node(anchor)
        teacher_scores = dict(root.dimension_scores or {}) if root is not None else {}
        anchor.document_score = float(_mean(teacher_scores.values()) or 4.0)
        anchor.metadata["dimension"] = "combined"
        anchor.metadata["teacher_dimension_scores_1_7"] = teacher_scores
        anchor.metadata["expert_dimension_scores_1_7"] = expert_scores
        anchor.metadata["teacher_score_1_7"] = anchor.document_score
        anchor.metadata["expert_score_1_7"] = _mean(expert_scores.values())
        anchor.metadata["combined_dimensions"] = list(dimensions)
        anchor.metadata["label_source"] = "manifesto_combined_dimension_teacher_fg_node_v1"
        anchor.metadata["source_dimension_trace_root"] = str(dimension_root)
        anchor.label_source = "manifesto_combined_dimension_teacher_fg_node_v1"
        combined_trees.append(anchor)

    leaf_dir = output_dir / f"leaf{int(leaf_size):04d}tok"
    labeled_path = write_labeled_trees_jsonl(leaf_dir / "labeled_trees.jsonl", combined_trees)
    summary = {
        "topology_axis": "size_tokens",
        "leaf_size_tokens": int(leaf_size),
        "dimensions": list(dimensions),
        "tree_counts": {
            "total": len(combined_trees),
            "train": sum(1 for tree in combined_trees if (tree.metadata or {}).get("split") == "train"),
            "val": sum(1 for tree in combined_trees if (tree.metadata or {}).get("split") == "val"),
            "test": sum(1 for tree in combined_trees if (tree.metadata or {}).get("split") == "test"),
            "skipped": skipped,
        },
        "node_count": sum(len(tree.nodes) for tree in combined_trees),
        "artifacts": {"labeled_trees": str(labeled_path)},
    }
    _write_json(leaf_dir / "summary.json", summary)
    return summary


def _parse_ints(value: str) -> list[int]:
    out = [int(part.strip()) for part in str(value).replace(";", ",").split(",") if part.strip()]
    if not out:
        raise ValueError("expected at least one integer")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--leaf-size-tokens", default="2048,4096,8192")
    parser.add_argument("--dimensions", default=",".join(DIMS))
    return parser.parse_args()


def _combined_tree_bundle_metadata(
    *,
    dimension_root: Path,
    dimensions: Sequence[str],
    leaf_sizes: Sequence[int],
) -> Dict[str, Any]:
    source_payload: Mapping[str, Any] = {}
    for dim in dimensions:
        path = Path(dimension_root) / str(dim) / "manifest.json"
        if not path.exists():
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            config = manifest.get("config") if isinstance(manifest, Mapping) else None
            source_payload = config if isinstance(config, Mapping) else manifest
            break
        except Exception:
            continue
    normalized = normalize_tree_bundle_manifest(source_payload)
    return tree_bundle_metadata(
        domain="manifesto_rile",
        leaf_unit=LEAF_UNIT_TEXT_TOKEN,
        source_kind=str(normalized.get("source_kind") or "raw_input"),
        dimension="combined",
        target_scale=str(normalized.get("target_scale") or "normalized_1_7"),
        leaf_policy={
            "topology_axis": "size_tokens",
            "leaf_size_tokens": [int(value) for value in leaf_sizes],
        },
        state_contract=str(normalized.get("state_contract") or "raw_concat"),
        external_state_producer=normalized.get("external_state_producer"),
        metadata={"component_dimensions": [str(dim) for dim in dimensions]},
    )


def main() -> int:
    args = parse_args()
    dims = [part.strip() for part in str(args.dimensions).replace(";", ",").split(",") if part.strip()]
    leaf_sizes = _parse_ints(args.leaf_size_tokens)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = {
        f"tok_{leaf_size}": _combine_leaf(
            dimension_root=args.dimension_root,
            output_dir=args.output_dir,
            leaf_size=int(leaf_size),
            dimensions=dims,
        )
        for leaf_size in leaf_sizes
    }
    manifest = {
        "dimension": "combined",
        "dimensions": dims,
        "dimension_root": str(args.dimension_root),
        "config": {
            "leaf_size_tokens": leaf_sizes,
            "topology_axis": "size_tokens",
            **_combined_tree_bundle_metadata(
                dimension_root=args.dimension_root,
                dimensions=dims,
                leaf_sizes=leaf_sizes,
            ),
        },
        "runs": runs,
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(config_to_dict(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
