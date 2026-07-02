#!/usr/bin/env python3
"""Build quasi-sentence-supervised Manifesto labeled trees for DSPy f/g.

Unlike the previous teacher-grid builders, this script does not call an LLM.
Labels come directly from Manifesto Project quasi-sentence ``cmp_code``
annotations.  Each node receives an exact aggregate target over its descendant
quasi-sentences; its ``teacher_summary`` is a compact policy-state rendering
used as the DSPy ``g`` target.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ctreepo.distillation import build_labeled_tree_from_text, load_labeled_trees, write_labeled_trees_jsonl
from src.ctreepo.manifesto_qsentence_runner import leafq_dir
from src.ctreepo.treepo_bridge.manifesto_finetune import (
    add_manifesto_finetune_args,
    export_manifesto_finetune_bundle_from_args,
    finetune_export_config,
)
from src.tasks.manifesto.script_utils import (
    append_jsonl,
    now_iso as _now_iso,
    parse_int_grid,
    write_json as _write_json,
)
from src.tasks.manifesto.span_annotations import (
    DEFAULT_QSENTENCE_CORPUS,
    PositionedQSentence,
    ReconstructedManifesto,
    indexed_manifesto_ids,
    load_manifesto_qsentences,
    qsentences_in_span,
    reconstruct_manifesto,
)
from src.tasks.manifesto.span_targets import (
    COMPACT_TARGET_DIMENSIONS,
    aggregate_cmp_codes,
    render_policy_state,
    targets_from_counts,
)
from src.tree.labeled import LabeledTree

LOGGER = logging.getLogger(__name__)



def _load_split_ids(path: Optional[Path]) -> Optional[dict[str, list[str]]]:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "train": [str(v) for v in payload.get("train", [])],
        "val": [str(v) for v in payload.get("val", [])],
        "test": [str(v) for v in payload.get("test", [])],
    }


def _make_split_ids(
    ids: Sequence[str],
    *,
    split_ids_path: Optional[Path],
    train_n: int,
    val_n: int,
    test_n: int,
    seed: int,
) -> dict[str, list[str]]:
    existing = _load_split_ids(split_ids_path)
    available = set(str(v) for v in ids)
    if existing is not None:
        return {
            split: [mid for mid in values if mid in available]
            for split, values in existing.items()
        }
    rng = random.Random(int(seed))
    shuffled = list(ids)
    rng.shuffle(shuffled)
    train = shuffled[: max(0, int(train_n))]
    val = shuffled[len(train) : len(train) + max(0, int(val_n))]
    test = shuffled[len(train) + len(val) : len(train) + len(val) + max(0, int(test_n))]
    return {"train": train, "val": val, "test": test}


def _mpds_rile_lookup(mpds_csv: Path) -> dict[str, float]:
    if not Path(mpds_csv).exists():
        return {}
    df = pd.read_csv(mpds_csv, low_memory=False)
    if "rile" not in df.columns:
        return {}
    df = df.dropna(subset=["party", "date", "rile"]).copy()
    df["manifesto_id"] = df["party"].astype(int).astype(str) + "_" + df["date"].astype(int).astype(str)
    return {str(row.manifesto_id): float(row.rile) for row in df.itertuples(index=False)}


def _rile_norm(raw_rile: float) -> float:
    return max(0.0, min(1.0, (float(raw_rile) + 100.0) / 200.0))


def _leaf_windows(reconstructed: ReconstructedManifesto, leaf_qsentences: int) -> list[tuple[int, int]]:
    q = list(reconstructed.qsentences)
    if not q:
        return [(0, len(reconstructed.text))]
    size = max(1, int(leaf_qsentences))
    windows: list[tuple[int, int]] = []
    for start_idx in range(0, len(q), size):
        end_idx = min(len(q), start_idx + size)
        start = int(q[start_idx].char_start)
        if end_idx < len(q):
            end = int(q[end_idx].char_start)
        else:
            end = len(reconstructed.text)
        windows.append((start, end))
    return windows


def _span_target(reconstructed: ReconstructedManifesto, *, start: int, end: int) -> dict[str, Any]:
    rows = qsentences_in_span(reconstructed, char_start=start, char_end=end)
    return aggregate_cmp_codes(item.item.cmp_code for item in rows)


def _qsentence_index_range(
    reconstructed: ReconstructedManifesto,
    *,
    start: int,
    end: int,
) -> tuple[Optional[int], Optional[int]]:
    idxs = [
        idx
        for idx, item in enumerate(reconstructed.qsentences)
        if int(item.char_start) >= int(start) and int(item.char_end) <= int(end)
    ]
    if not idxs:
        return None, None
    return min(idxs), max(idxs) + 1


def _decorate_tree(
    tree: LabeledTree,
    reconstructed: ReconstructedManifesto,
    *,
    split: str,
    leaf_qsentences: int,
    mpds_rile_raw: Optional[float],
) -> tuple[LabeledTree, list[dict[str, Any]]]:
    node_rows: list[dict[str, Any]] = []
    root_node_id = tree.levels[-1][-1] if tree.levels and tree.levels[-1] else None
    root_target: Optional[dict[str, Any]] = None
    for node in tree.nodes.values():
        meta = dict(node.metadata or {})
        start = int(meta.get("char_start", 0))
        end = int(meta.get("char_end", start + len(str(node.text or ""))))
        target = _span_target(reconstructed, start=start, end=end)
        compact = dict(target.get("compact") or {})
        node.score = float(compact.get("rile", 0.5))
        node.dimension_scores = {key: float(compact.get(key, 0.0)) for key in COMPACT_TARGET_DIMENSIONS}
        summary = render_policy_state(target)
        q_start, q_end = _qsentence_index_range(reconstructed, start=start, end=end)
        meta.update(
            {
                "teacher_summary": summary,
                "target_summary": summary,
                "teacher_summary_source": "manifesto_qsentence_cmp_state",
                "teacher_dimension_scores_1_7": dict(node.dimension_scores),
                "target_dimension_scores_0_1": dict(node.dimension_scores),
                "cmp_counts": dict(target.get("counts") or {}),
                "domain_counts": dict(target.get("domain_counts") or {}),
                "rile_raw": float(target.get("rile_raw", 0.0)),
                "rile_norm": float(compact.get("rile", 0.5)),
                "total_qsentences": int(target.get("total_items", 0) or 0),
                "total_non_header_qsentences": int(target.get("total_non_header", 0) or 0),
                "leaf_qsentences": int(leaf_qsentences),
                "qsentence_start_index": q_start,
                "qsentence_end_index": q_end,
            }
        )
        if int(node.level) == 0:
            meta["teacher_leaf_summary"] = summary
        else:
            meta["teacher_merge_summary"] = summary
        node.metadata = meta
        if node.node_id == root_node_id:
            root_target = target
        node_rows.append(
            {
                "doc_id": tree.doc_id,
                "split": split,
                "node_id": node.node_id,
                "level": int(node.level),
                "is_leaf": int(node.level) == 0,
                "char_start": start,
                "char_end": end,
                "leaf_qsentences": int(leaf_qsentences),
                "qsentence_start_index": q_start,
                "qsentence_end_index": q_end,
                "total_qsentences": int(target.get("total_items", 0) or 0),
                "total_non_header_qsentences": int(target.get("total_non_header", 0) or 0),
                "rile_raw": float(target.get("rile_raw", 0.0)),
                "rile_norm": float(compact.get("rile", 0.5)),
                "dimension_scores_0_1": dict(node.dimension_scores),
                "cmp_counts": dict(target.get("counts") or {}),
            }
        )
    root_compact = dict((root_target or {}).get("compact") or {})
    tree.document_score = float(root_compact.get("rile", 0.5))
    expert_scores = {}
    if mpds_rile_raw is not None:
        expert_scores["rile"] = _rile_norm(float(mpds_rile_raw))
    metadata = dict(tree.metadata or {})
    metadata.update(
        {
            "split": split,
            "label_source": "manifesto_qsentence_cmp_annotations_v1",
            "leaf_qsentences": int(leaf_qsentences),
            "topology_axis": "leaf_qsentences",
            "target_dimensions": list(COMPACT_TARGET_DIMENSIONS),
            "teacher_dimension_scores_1_7": root_compact,
            "target_dimension_scores_0_1": root_compact,
            "expert_dimension_scores_1_7": expert_scores,
            "mpds_rile_raw": mpds_rile_raw,
            "mpds_rile_norm": expert_scores.get("rile"),
            "qsents_per_doc": int(len(reconstructed.qsentences)),
        }
    )
    tree.metadata = metadata
    return tree, node_rows


def _build_tree(
    reconstructed: ReconstructedManifesto,
    *,
    split: str,
    leaf_qsentences: int,
    mpds_rile_raw: Optional[float],
) -> tuple[LabeledTree, list[dict[str, Any]]]:
    windows = _leaf_windows(reconstructed, int(leaf_qsentences))
    tree = build_labeled_tree_from_text(
        doc_id=reconstructed.manifesto_id,
        text=reconstructed.text,
        document_score=0.5,
        split=split,
        score_fn=lambda _span: 0.5,
        window_size=max(1, len(reconstructed.text)),
        explicit_char_windows=windows,
        label_source="manifesto_qsentence_cmp_annotations_v1",
        node_summary_fn=lambda span, context: render_policy_state(
            _span_target(
                reconstructed,
                start=int(context["char_start"]),
                end=int(context["char_end"]),
            )
        ),
        summary_source="manifesto_qsentence_cmp_state",
        extra_metadata={
            "leaf_qsentences": int(leaf_qsentences),
            "topology_axis": "leaf_qsentences",
            "target_dimensions": list(COMPACT_TARGET_DIMENSIONS),
        },
    )
    return _decorate_tree(
        tree,
        reconstructed,
        split=split,
        leaf_qsentences=int(leaf_qsentences),
        mpds_rile_raw=mpds_rile_raw,
    )


def _verify_tree(tree: LabeledTree) -> None:
    for node in tree.get_merge_nodes():
        left = tree.get_node(str(node.left_child_id)) if node.left_child_id else None
        right = tree.get_node(str(node.right_child_id)) if node.right_child_id else None
        if left is None or right is None:
            raise ValueError(f"{tree.doc_id}:{node.node_id} has missing children")
        counts = Counter(dict((left.metadata or {}).get("cmp_counts") or {}))
        # The generic binary-tree builder promotes an odd orphan by setting
        # left_child_id == right_child_id.  That topology is a pass-through, not
        # a duplicate quasi-sentence span, so exact additive checks must count
        # the child once.
        if right.node_id != left.node_id:
            counts.update(dict((right.metadata or {}).get("cmp_counts") or {}))
        expected = targets_from_counts(counts)
        actual = dict((node.metadata or {}).get("cmp_counts") or {})
        if dict(expected.get("counts") or {}) != actual:
            raise ValueError(f"{tree.doc_id}:{node.node_id} parent counts do not equal child sums")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-csv", type=Path, default=DEFAULT_QSENTENCE_CORPUS)
    parser.add_argument(
        "--mpds-csv",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "manifesto_corpus_benoit" / "manifesto_maindataset.csv",
    )
    parser.add_argument("--split-ids", type=Path, default=None)
    parser.add_argument("--manifesto-ids", nargs="*", default=None)
    parser.add_argument("--leaf-qsentences", default="1,2,4,8,16")
    parser.add_argument("--train-n", type=int, default=140)
    parser.add_argument("--val-n", type=int, default=30)
    parser.add_argument("--test-n", type=int, default=48)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    add_manifesto_finetune_args(
        parser,
        kind="qsentence",
        help_text="Write treepo PreferenceDataset/fine-tune adapter bundles per leaf row.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild leaf cells even if their labeled_trees.jsonl already "
        "exists. Tree building is expensive; the default reuses existing "
        "cells and only records them in the manifest.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=str(args.log_level).upper(), format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    requested_ids = [str(v) for v in args.manifesto_ids] if args.manifesto_ids else None
    grouped = load_manifesto_qsentences(args.corpus_csv, manifesto_ids=requested_ids)
    ids = indexed_manifesto_ids(grouped)
    if args.max_docs is not None:
        ids = ids[: int(args.max_docs)]
    split_ids = _make_split_ids(
        ids,
        split_ids_path=args.split_ids,
        train_n=int(args.train_n),
        val_n=int(args.val_n),
        test_n=int(args.test_n),
        seed=int(args.seed),
    )
    selected = set(split_ids["train"]) | set(split_ids["val"]) | set(split_ids["test"])
    if args.max_docs is not None:
        selected &= set(ids)
        split_ids = {split: [mid for mid in mids if mid in selected] for split, mids in split_ids.items()}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "split_ids.json", split_ids)
    mpds_rile = _mpds_rile_lookup(args.mpds_csv)

    leaf_grid = parse_int_grid(args.leaf_qsentences, name="leaf-qsentences grid")
    manifest_runs: dict[str, Any] = {}
    for leaf_q in leaf_grid:
        row_dir = leafq_dir(output_dir, int(leaf_q))
        trees_path = row_dir / "labeled_trees.jsonl"
        if trees_path.exists() and not args.rebuild:
            LOGGER.info(
                "leaf_qsentences=%d reusing existing %s (pass --rebuild to regenerate)",
                leaf_q,
                trees_path,
            )
            summary_path = row_dir / "summary.json"
            existing_summary: dict[str, Any] = {}
            if summary_path.exists():
                existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            reused_trees = load_labeled_trees(trees_path)
            finetune_bundle = export_manifesto_finetune_bundle_from_args(
                args=args,
                trees=reused_trees,
                output_dir=row_dir / "treepo_finetune",
                kind="qsentence",
            )
            manifest_runs[f"qsent_{int(leaf_q)}"] = {
                "leaf_qsentences": int(leaf_q),
                "reused_existing": True,
                "artifacts": {
                    "labeled_trees": str(trees_path),
                    "teacher_node_rows": str(row_dir / "teacher_node_rows.jsonl"),
                    "finetune_bundle": str(row_dir / "treepo_finetune") if finetune_bundle else None,
                },
                "finetune": finetune_bundle,
                **existing_summary,
            }
            continue
        trees: list[LabeledTree] = []
        node_rows: list[dict[str, Any]] = []
        skipped: Counter[str] = Counter()
        for split, mids in split_ids.items():
            for mid in mids:
                rows = grouped.get(mid) or []
                if not rows:
                    skipped["missing_annotations"] += 1
                    continue
                reconstructed = reconstruct_manifesto(mid, rows)
                tree, rows_out = _build_tree(
                    reconstructed,
                    split=split,
                    leaf_qsentences=int(leaf_q),
                    mpds_rile_raw=mpds_rile.get(mid),
                )
                _verify_tree(tree)
                trees.append(tree)
                node_rows.extend(rows_out)
        write_labeled_trees_jsonl(row_dir / "labeled_trees.jsonl", trees)
        append_jsonl(row_dir / "teacher_node_rows.jsonl", node_rows, append=False)
        finetune_bundle = export_manifesto_finetune_bundle_from_args(
            args=args,
            trees=trees,
            output_dir=row_dir / "treepo_finetune",
            kind="qsentence",
        )
        summary = {
            "created_at": _now_iso(),
            "leaf_qsentences": int(leaf_q),
            "topology_axis": "leaf_qsentences",
            "tree_counts": {
                "total": len(trees),
                "train": sum(1 for t in trees if (t.metadata or {}).get("split") == "train"),
                "val": sum(1 for t in trees if (t.metadata or {}).get("split") == "val"),
                "test": sum(1 for t in trees if (t.metadata or {}).get("split") == "test"),
                "skipped": dict(skipped),
            },
            "node_count": len(node_rows),
            "target_dimensions": list(COMPACT_TARGET_DIMENSIONS),
        }
        _write_json(row_dir / "summary.json", summary)
        manifest_runs[f"qsent_{int(leaf_q)}"] = {
            "leaf_qsentences": int(leaf_q),
            "artifacts": {
                "labeled_trees": str(row_dir / "labeled_trees.jsonl"),
                "teacher_node_rows": str(row_dir / "teacher_node_rows.jsonl"),
                "finetune_bundle": str(row_dir / "treepo_finetune") if finetune_bundle else None,
            },
            "finetune": finetune_bundle,
            **summary,
        }
        LOGGER.info("leaf_qsentences=%d wrote %d trees / %d nodes", leaf_q, len(trees), len(node_rows))
    manifest = {
        "created_at": _now_iso(),
        "label_source": "manifesto_qsentence_cmp_annotations_v1",
        "corpus_csv": str(args.corpus_csv),
        "mpds_csv": str(args.mpds_csv),
        "split_ids": str(output_dir / "split_ids.json"),
        "leaf_qsentences": list(leaf_grid),
        "target_dimensions": list(COMPACT_TARGET_DIMENSIONS),
        "finetune_export": finetune_export_config(args),
        "runs": manifest_runs,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
