#!/usr/bin/env python
"""Build a MPDS supervision-ablation arm from the gold CMP-annotated q-sentence grid.

MPDS (manifesto_qsentence_dspy_labeled_grid) is GOLD all the way down: leaves carry
per-quasi-sentence CMP human codes (label_source manifesto_qsentence_cmp_annotations_v1)
that vary within a doc, merges/root carry the gold rollup. This lets us run the TRUE
global-vs-local supervision test that Benoit could not (Benoit has no gold per-sentence
labels -- only doc-level expert means broadcast to leaves).

Arms (which nodes keep their gold score; all others -> None = unsupervised, learned freely):
  * root          : ONLY the root keeps gold (pure global / doc-level supervision).
  * root_leaf     : root + leaves keep gold; merges unsupervised (g learns the merge).
                    = root + gold LOCAL (per-q-sentence) supervision.
  * root_leaf_merge: everything keeps gold (full supervision; reference/upper arm).

We do NOT invent labels or rollups -- we only NULL existing gold scores to ablate
supervision. node.score and node.dimension_scores are nulled together so the live
--fno-target-dimension retarget stays consistent.
"""
import argparse, json, math
from pathlib import Path

from src.ctreepo.distillation import load_labeled_trees, write_labeled_trees_jsonl


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-grid", required=True, help="e.g. outputs/manifesto_qsentence_dspy_labeled_grid")
    p.add_argument("--leaf", type=int, required=True)
    p.add_argument("--keep", choices=("root", "root_leaf", "root_leaf_merge"), required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--leaf-keep-frac",
        type=float,
        default=1.0,
        help=(
            "Dose-response: fraction of LEAVES that keep gold (rest -> None). "
            "Only meaningful with --keep root_leaf. Deterministic per-doc by "
            "leaf node order (no RNG, reproducible). 1.0 = all gold leaves."
        ),
    )
    args = p.parse_args(argv)

    src = Path(args.src_grid)
    trees = load_labeled_trees(src / f"leafq{int(args.leaf):03d}" / "labeled_trees.jsonl")
    frac = max(0.0, min(1.0, float(args.leaf_keep_frac)))
    out_trees = []
    n_nulled = 0
    n_kept = 0
    n_leaf_dropped = 0
    for tree in trees:
        root_id = str(tree.levels[-1][0]) if tree.levels and tree.levels[-1] else None
        # Deterministic per-doc leaf subset: take the first ceil(frac*n) leaves in
        # node order. No RNG -> identical grid across seeds (only training seed varies).
        leaf_ids = [str(n.node_id) for n in tree.nodes.values() if int(n.level) == 0]
        n_keep_leaf = int(math.ceil(frac * len(leaf_ids)))
        keep_leaf_set = set(leaf_ids[:n_keep_leaf])
        for node in tree.nodes.values():
            nid = str(node.node_id)
            is_root = nid == root_id
            is_leaf = int(node.level) == 0
            if is_root:
                keep = True
            elif is_leaf:
                keep = args.keep in ("root_leaf", "root_leaf_merge")
                if keep and frac < 1.0 and nid not in keep_leaf_set:
                    keep = False
                    n_leaf_dropped += 1
            else:  # merge
                keep = args.keep == "root_leaf_merge"
            if keep:
                n_kept += 1
            else:
                node.score = None
                node.dimension_scores = {}
                n_nulled += 1
        out_trees.append(tree)

    out = Path(args.output_dir)
    (out / f"leafq{int(args.leaf):03d}").mkdir(parents=True, exist_ok=True)
    write_labeled_trees_jsonl(out / f"leafq{int(args.leaf):03d}" / "labeled_trees.jsonl", out_trees)
    src_split = src / "split_ids.json"
    if src_split.exists():
        (out / "split_ids.json").write_text(src_split.read_text(encoding="utf-8"), encoding="utf-8")
    print(
        f"[{args.keep} frac={frac:g}] {len(out_trees)} trees | kept {n_kept} gold nodes | "
        f"nulled {n_nulled} (leaf-dropped {n_leaf_dropped}) -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
