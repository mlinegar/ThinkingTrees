#!/usr/bin/env python3
"""Relabel a leaf=16 Benoit q-sentence grid with LLM chunk scores for learned-g.

Per tree, for one Benoit dimension:
  * leaf node.score  = LLM chunk score [0,1]  (real per-chunk supervision; None if NA)
  * merge node.score = descendant-leaf mean of chunk scores (skips NA)  (mean-rollup)
  * ROOT node.score / document_score = doc EXPERT mean (normalized)  (g learns chunk->expert)

So f distills the LLM chunk scorer, intermediate g learns mean-aggregation, and the
root pulls g toward the holistic expert score. dimension_scores carries the single
dim so the existing ``--fno-target-dimension`` retarget works. Expert mean is also
stored in tree.metadata for the external comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ctreepo.distillation import load_labeled_trees, write_labeled_trees_jsonl

LOGGER = logging.getLogger(__name__)


def _expert_norm(raw: float, *, already_normalized: bool = False) -> float:
    # Benoit dims are 1-7 policy-intensity scores rescaled to [0,1] via /10.
    # RILE targets (doc_rile.json) are ALREADY in [0,1] (mpds_rile_norm), so the
    # /10 must be skipped or 0.41 would wrongly become 0.041.
    div = 1.0 if already_normalized else 10.0
    return max(0.0, min(1.0, float(raw) / div))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-grid", type=Path, default=Path("outputs/benoit_qsentence_grid_full"))
    p.add_argument("--leaf", type=int, default=16)
    p.add_argument("--chunk-scores", type=Path, required=True)
    p.add_argument(
        "--merge-scores",
        type=Path,
        default=None,
        help="LLM span scores for intermediate/merge nodes (from "
        "score_benoit_chunks --node-levels merges). Required for "
        "--merge-supervision llm_span.",
    )
    p.add_argument("--dim", required=True)
    p.add_argument(
        "--expert-targets",
        type=Path,
        default=Path("outputs/benoit_qsentence_targets/expert_means_raw.json"),
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--expert-already-normalized",
        action="store_true",
        help="Treat --expert-targets values as already in [0,1] (skip the Benoit "
        "/10 rescale). Use for RILE (doc_rile.json is mpds_rile_norm in [0,1]).",
    )
    p.add_argument(
        "--merge-supervision",
        choices=("none", "mean_rollup", "llm_span"),
        default="llm_span",
        help="Intermediate (non-root) merge targets. 'llm_span' (default) = the "
        "LLM's holistic score of that node's merged span (real node-level gold "
        "label at every scale; needs --merge-scores). 'mean_rollup' = mean of "
        "descendant chunk scores (teaches averaging). 'none' = unsupervised.",
    )
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    chunk_scores: Dict[str, float] = json.loads(Path(args.chunk_scores).read_text(encoding="utf-8"))
    merge_scores: Dict[str, float] = {}
    if str(args.merge_supervision) == "llm_span":
        if args.merge_scores is None or not Path(args.merge_scores).exists():
            raise SystemExit("--merge-supervision llm_span requires --merge-scores <file>")
        merge_scores = json.loads(Path(args.merge_scores).read_text(encoding="utf-8"))
    expert_raw: Dict[str, Dict[str, float]] = json.loads(
        Path(args.expert_targets).read_text(encoding="utf-8")
    )
    dim = str(args.dim)

    trees = load_labeled_trees(args.src_grid / f"leafq{int(args.leaf):03d}" / "labeled_trees.jsonl")
    out_trees: List[Any] = []
    n_drop = 0
    for tree in trees:
        doc_id = str(tree.doc_id)
        if doc_id not in expert_raw or dim not in expert_raw[doc_id]:
            n_drop += 1
            continue
        expert = _expert_norm(
            expert_raw[doc_id][dim],
            already_normalized=bool(args.expert_already_normalized),
        )
        # (sum, count) of non-NA descendant-leaf chunk scores, bottom-up
        agg: Dict[str, List[float]] = {}  # node_id -> [sum, count]
        leaf_ids = set(str(n.node_id) for n in tree.nodes.values() if int(n.level) == 0)
        for nid in leaf_ids:
            s = chunk_scores.get(f"{doc_id}|{nid}")
            agg[nid] = [float(s), 1.0] if s is not None else [0.0, 0.0]
        # levels bottom-up (level 0 leaves already set)
        for level in (tree.levels or [])[1:]:
            for nid in level:
                node = tree.get_node(nid)
                if node is None:
                    continue
                lc = str(node.left_child_id)
                rc = str(node.right_child_id or node.left_child_id)
                ls, ln = agg.get(lc, [0.0, 0.0])
                rs, rn = agg.get(rc, [0.0, 0.0])
                if rc == lc:
                    agg[str(nid)] = [ls, ln]
                else:
                    agg[str(nid)] = [ls + rs, ln + rn]
        root_id = str(tree.levels[-1][0]) if tree.levels and tree.levels[-1] else None
        for node in tree.nodes.values():
            nid = str(node.node_id)
            if int(node.level) == 0:
                s = chunk_scores.get(f"{doc_id}|{nid}")
                # Missing leaf (parse miss / NA): fall back to neutral 0.5 so no
                # node is left None (the eval path floats every node.score).
                val = float(s) if s is not None else 0.5
            elif nid == root_id:
                val = expert  # root supervised on the holistic expert mean
            elif str(args.merge_supervision) == "llm_span":
                s = merge_scores.get(f"{doc_id}|{nid}")
                if s is not None:
                    val = float(s)  # LLM holistic span score
                else:
                    # span unscored (NA / parse miss): fall back to mean-rollup so
                    # no node is left None (the eval path floats node.score).
                    tot, cnt = agg.get(nid, [0.0, 0.0])
                    val = (tot / cnt) if cnt > 0 else expert
            elif str(args.merge_supervision) == "mean_rollup":
                tot, cnt = agg.get(nid, [0.0, 0.0])
                val = (tot / cnt) if cnt > 0 else None  # mean-rollup over non-NA descendants
            else:
                val = None  # unsupervised intermediate merge: g learns the aggregation freely
            node.score = val
            node.dimension_scores = {dim: val} if val is not None else {}
        tree.document_score = expert
        md = dict(tree.metadata or {})
        md.update(
            {
                "target_dimensions": [dim],
                "expert_dimension_scores_0_1": {dim: expert},
                "label_source": f"benoit_chunk_llm_{Path(args.chunk_scores).stem}",
                "leaf_qsentences": int(args.leaf),
            }
        )
        tree.metadata = md
        out_trees.append(tree)

    out = Path(args.output_dir)
    (out / f"leafq{int(args.leaf):03d}").mkdir(parents=True, exist_ok=True)
    write_labeled_trees_jsonl(out / f"leafq{int(args.leaf):03d}" / "labeled_trees.jsonl", out_trees)
    # split_ids carried over from source so the ladder eval split matches
    src_split = args.src_grid / "split_ids.json"
    if src_split.exists():
        (out / "split_ids.json").write_text(src_split.read_text(encoding="utf-8"), encoding="utf-8")
    LOGGER.info(
        "Relabeled %d trees (dropped %d missing-expert) -> %s [dim=%s]",
        len(out_trees), n_drop, out, dim,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
