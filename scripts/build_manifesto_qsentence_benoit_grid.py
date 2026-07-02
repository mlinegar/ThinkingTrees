#!/usr/bin/env python3
"""Build q-sentence labeled trees for the BENOIT expert dimensions.

Unlike ``build_manifesto_qsentence_dspy_labeled_grid.py`` (which labels every
node with an exact CMP-code aggregate), the Benoit expert means are DOC-LEVEL
only — there is no per-quasi-sentence Benoit label. We therefore BROADCAST the
document-level Benoit expert score (per dimension) to every node:

* every node target (``dimension_scores``) = the document's Benoit score for
  that dimension, normalized to [0,1] (raw 0-10 scale / 10);
* ``tree.document_score`` and ``tree.metadata.expert_dimension_scores`` carry
  the same doc-level Benoit vector (the external-expert metric target).

This makes the bundle structurally identical to the CMP q-sentence bundle, so
the existing ladder (``run_manifesto_qsentence_dspy_ladder.py``) and the
``--fno-target-dimension`` retarget read it unchanged. The interpretation: f
estimates the document's Benoit score from each chunk, g learns to aggregate
chunk estimates into the doc score (the q-sentence generalization of the
full-doc f-only baseline). Reconstruction quality is read at the ROOT
(prediction vs the doc-level Benoit expert mean = external_expert_pearson),
directly comparable to ``outputs/overnight_benoit/roundup.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

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
    now_iso as _now_iso,
    parse_int_grid,
    write_json as _write_json,
)
from src.tasks.manifesto.span_annotations import (
    DEFAULT_QSENTENCE_CORPUS,
    ReconstructedManifesto,
    indexed_manifesto_ids,
    load_manifesto_qsentences,
    reconstruct_manifesto,
)
from src.tree.labeled import LabeledTree

LOGGER = logging.getLogger(__name__)

BENOIT_DIMENSIONS = (
    "economic",
    "social",
    "immigration",
    "eu",
    "environment",
    "decentralization",
)
BENOIT_RAW_SCALE = 10.0  # Benoit expert means are on a 0-10 scale.



def _norm(raw: float) -> float:
    return max(0.0, min(1.0, float(raw) / BENOIT_RAW_SCALE))


def _load_benoit_targets(path: Path) -> Dict[str, Dict[str, float]]:
    """manifesto_id -> {benoit_dim: normalized [0,1] score} (only dims present)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, float]] = {}
    for mid, dims in raw.items():
        out[str(mid)] = {
            d: _norm(v) for d, v in dims.items() if d in BENOIT_DIMENSIONS and v is not None
        }
    return out


def _leaf_windows(reconstructed: ReconstructedManifesto, leaf_qsentences: int) -> List[tuple]:
    q = list(reconstructed.qsentences)
    if not q:
        return [(0, len(reconstructed.text))]
    size = max(1, int(leaf_qsentences))
    windows: List[tuple] = []
    for start_idx in range(0, len(q), size):
        end_idx = min(len(q), start_idx + size)
        start = int(q[start_idx].char_start)
        end = int(q[end_idx].char_start) if end_idx < len(q) else len(reconstructed.text)
        windows.append((start, end))
    return windows


def _build_tree(
    reconstructed: ReconstructedManifesto,
    *,
    split: str,
    leaf_qsentences: int,
    doc_targets: Mapping[str, float],
) -> LabeledTree:
    """Build a q-sentence tree and BROADCAST the doc Benoit vector to all nodes."""
    windows = _leaf_windows(reconstructed, int(leaf_qsentences))
    # primary scalar target = economic if present, else first available dim
    primary_dim = "economic" if "economic" in doc_targets else (
        next(iter(doc_targets)) if doc_targets else "economic"
    )
    primary = float(doc_targets.get(primary_dim, 0.5))
    tree = build_labeled_tree_from_text(
        doc_id=reconstructed.manifesto_id,
        text=reconstructed.text,
        document_score=primary,
        split=split,
        score_fn=lambda _span: primary,
        window_size=max(1, len(reconstructed.text)),
        explicit_char_windows=windows,
        label_source="manifesto_qsentence_benoit_expert_v1",
        node_summary_fn=lambda span, context: str(span or ""),
        summary_source="manifesto_qsentence_text",
        extra_metadata={
            "leaf_qsentences": int(leaf_qsentences),
            "topology_axis": "leaf_qsentences",
            "target_dimensions": list(BENOIT_DIMENSIONS),
        },
    )
    dim_vec = {d: float(doc_targets.get(d, 0.0)) for d in BENOIT_DIMENSIONS}
    for node in tree.nodes.values():
        node.score = primary
        # broadcast doc-level Benoit vector to every node (only dims with data)
        node.dimension_scores = {d: dim_vec[d] for d in doc_targets}
        meta = dict(node.metadata or {})
        meta.update(
            {
                "leaf_qsentences": int(leaf_qsentences),
                "benoit_dimension_scores_0_1": {d: dim_vec[d] for d in doc_targets},
                "label_source": "manifesto_qsentence_benoit_expert_v1",
            }
        )
        node.metadata = meta
    tree.document_score = primary
    md = dict(tree.metadata or {})
    md.update(
        {
            "split": split,
            "label_source": "manifesto_qsentence_benoit_expert_v1",
            "leaf_qsentences": int(leaf_qsentences),
            "topology_axis": "leaf_qsentences",
            "target_dimensions": list(doc_targets.keys()),
            "expert_dimension_scores_0_1": {d: dim_vec[d] for d in doc_targets},
            "benoit_dimensions_present": list(doc_targets.keys()),
            "qsents_per_doc": int(len(reconstructed.qsentences)),
        }
    )
    tree.metadata = md
    return tree


def _make_split(ids: Sequence[str], targets: Mapping[str, Mapping[str, float]], *, train_n, val_n, test_n, seed):
    import random

    eligible = [m for m in ids if m in targets and targets[m]]
    rng = random.Random(int(seed))
    shuffled = list(eligible)
    rng.shuffle(shuffled)
    train = shuffled[:train_n]
    val = shuffled[train_n : train_n + val_n]
    test = shuffled[train_n + val_n : train_n + val_n + test_n]
    return {"train": train, "val": val, "test": test}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-csv", type=Path, default=DEFAULT_QSENTENCE_CORPUS)
    p.add_argument(
        "--benoit-targets",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "benoit_qsentence_targets" / "expert_means_raw.json",
    )
    p.add_argument("--manifesto-ids", nargs="*", default=None)
    p.add_argument("--leaf-qsentences", default="1,8,16")
    p.add_argument("--train-n", type=int, default=140)
    p.add_argument("--val-n", type=int, default=29)
    p.add_argument("--test-n", type=int, default=48)
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, required=True)
    add_manifesto_finetune_args(
        p,
        kind="qsentence",
        help_text="Write treepo PreferenceDataset/fine-tune adapter bundles per leaf row.",
    )
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=str(args.log_level).upper(), format="%(asctime)s %(levelname)s | %(message)s")
    targets = _load_benoit_targets(args.benoit_targets)
    requested = [str(v) for v in args.manifesto_ids] if args.manifesto_ids else list(targets.keys())
    grouped = load_manifesto_qsentences(args.corpus_csv, manifesto_ids=requested)
    ids = indexed_manifesto_ids(grouped)
    ids = [m for m in ids if m in targets and targets[m]]
    if args.max_docs is not None:
        ids = ids[: int(args.max_docs)]
    split_ids = _make_split(
        ids, targets, train_n=args.train_n, val_n=args.val_n, test_n=args.test_n, seed=args.seed
    )
    selected = set(split_ids["train"]) | set(split_ids["val"]) | set(split_ids["test"])
    split_lookup = {m: s for s, mids in split_ids.items() for m in mids}

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "split_ids.json", split_ids)

    leaf_grid = list(parse_int_grid(args.leaf_qsentences, name="leaf-qsentences grid"))
    manifest_runs: Dict[str, Any] = {}
    for leaf_q in leaf_grid:
        row_dir = leafq_dir(out, int(leaf_q))
        trees_path = row_dir / "labeled_trees.jsonl"
        if trees_path.exists() and not args.rebuild:
            LOGGER.info("leafq%03d reuse existing (use --rebuild)", leaf_q)
            reused_trees = load_labeled_trees(trees_path)
            finetune_bundle = export_manifesto_finetune_bundle_from_args(
                args=args,
                trees=reused_trees,
                output_dir=row_dir / "treepo_finetune",
                kind="qsentence",
            )
            manifest_runs[f"leafq{int(leaf_q):03d}"] = {
                "trees_path": str(trees_path),
                "reused": True,
                "n_trees": len(reused_trees),
                "finetune": finetune_bundle,
            }
            continue
        row_dir.mkdir(parents=True, exist_ok=True)
        trees: List[LabeledTree] = []
        for mid in selected:
            recon = reconstruct_manifesto(mid, grouped[mid]) if mid in grouped else None
            if recon is None or not recon.qsentences:
                continue
            tree = _build_tree(
                recon,
                split=split_lookup.get(mid, "train"),
                leaf_qsentences=int(leaf_q),
                doc_targets=targets[mid],
            )
            trees.append(tree)
        write_labeled_trees_jsonl(trees_path, trees)
        finetune_bundle = export_manifesto_finetune_bundle_from_args(
            args=args,
            trees=trees,
            output_dir=row_dir / "treepo_finetune",
            kind="qsentence",
        )
        LOGGER.info("leafq%03d wrote %d trees -> %s", leaf_q, len(trees), trees_path)
        manifest_runs[f"leafq{int(leaf_q):03d}"] = {
            "trees_path": str(trees_path),
            "n_trees": len(trees),
            "finetune": finetune_bundle,
        }
    _write_json(
        out / "manifest.json",
        {
            "created_at": _now_iso(),
            "builder": "build_manifesto_qsentence_benoit_grid",
            "benoit_dimensions": list(BENOIT_DIMENSIONS),
            "raw_scale": BENOIT_RAW_SCALE,
            "leaf_grid": leaf_grid,
            "n_docs": len(selected),
            "split_counts": {k: len(v) for k, v in split_ids.items()},
            "finetune_export": finetune_export_config(args),
            "runs": manifest_runs,
        },
    )
    LOGGER.info("Benoit q-sentence grid complete: %s (%d docs)", out, len(selected))
    return 0


if __name__ == "__main__":
    sys.exit(main())
