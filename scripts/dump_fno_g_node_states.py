#!/usr/bin/env python3
"""Dump per-node g-states from a trained FNO run for the merge-by-level eval.

The FNO analog of ``scripts/dump_qsentence_g_node_states.py`` (which is
LLM/DSPy-specific). ``scripts/eval_qsentence_merge_by_level.py --g-states-jsonl``
needs g's predicted scalar at EVERY merge node, keyed by (doc_id, node_id), in the
generic ``{"doc_id":.., "node_id":.., "compact_targets":{dim:val}}`` schema. This
loads the trained ``EmbeddingCoordinateFNOTreeRegressor`` (``fno_state_g.pt``),
runs the level-synchronous ``_forward_tree_states`` over the labeled trees (NO LM
calls — embeddings + the tiny FNO model only), and emits one row per merge node
with ``compact_targets = {dimension: predict_normalized(node_state)}``.

Single-dim by construction (matches the run's ``--fno-target-dimension``); the
score is the FNO's scalar head over the node state. Feeds the SAME per-node eval
as the LLM arm, so both substrates are scored on one path.

Example::

    python scripts/dump_fno_g_node_states.py \
        --run-dir outputs/fno_economic_leaf8_gpu_.../fno \
        --leaf-qsentences 8 \
        --fg-grid-dir outputs/manifesto_qsentence_dspy_labeled_grid \
        --target-dimension domain_4 \
        --embedding-model /mnt/data/models/google/embeddinggemma-300m \
        --embedding-device cuda \
        --fno-extent --fno-extent-merge-init additive \
        --out-jsonl <run>/g_node_states_leaf8.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch

from src.ctreepo.distillation import load_labeled_trees
from src.ctreepo.embedding_fno import _forward_tree_states
from src.ctreepo.fno_family import FNOFamily, FNOFamilyConfig
from src.tree.labeled import LabeledTree


def _build_embedding_client(args: argparse.Namespace):
    from src.experiments.embedding_clients import LocalHFEmbeddingClient

    client = LocalHFEmbeddingClient(
        model=str(args.embedding_model),
        batch_size=int(args.embedding_batch_size),
        max_length=int(args.embedding_max_length),
        device=str(args.embedding_device),
    )
    if getattr(args, "embedding_cache_dir", None):
        from src.ctreepo.embedding_cache import DiskCachedEmbeddingClient

        client = DiskCachedEmbeddingClient(
            client,
            cache_dir=str(args.embedding_cache_dir),
            model_id=str(args.embedding_model),
        )
    return client


def _retarget(trees: Sequence[LabeledTree], dimension: str) -> int:
    # Reuse the public q-sentence retarget so node.score matches training.
    from src.ctreepo.manifesto_qsentence_runner import retarget_trees_to_dimension

    return retarget_trees_to_dimension(trees, dimension)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        required=True,
        help="The run's <root>/fno directory (contains leafq00N/iter_02_train_g/fno_state_g.pt).",
    )
    ap.add_argument("--leaf-qsentences", type=int, required=True)
    ap.add_argument("--fg-grid-dir", required=True)
    ap.add_argument("--target-dimension", required=True)
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--max-trees", type=int, default=0, help="0 = all trees in the split.")
    ap.add_argument("--out-jsonl", required=True)
    # Embedding (must match the run that trained the model).
    ap.add_argument("--embedding-model", default="/mnt/data/models/google/embeddinggemma-300m")
    ap.add_argument("--embedding-device", default="cuda")
    ap.add_argument("--embedding-batch-size", type=int, default=128)
    # Must match the ladder's --embedding-max-length (default 2048); a smaller
    # value trips LocalHFEmbeddingClient's no-truncation guard on long chunks.
    ap.add_argument("--embedding-max-length", type=int, default=2048)
    ap.add_argument(
        "--embedding-cache-dir",
        default=None,
        help="Shared disk embedding cache (same dir the training arms used) — the "
        "dump then loads embeddings from disk instead of recomputing them.",
    )
    # Architecture flags (must match the trained checkpoint so state_dict loads).
    # FNO architecture — MUST match the ladder defaults the model was trained with
    # (run_manifesto_qsentence_dspy_ladder.py), else the state_dict won't load.
    ap.add_argument("--fno-merge-mode", default="gated", choices=["mean", "gated", "maxpool", "mlp"])
    ap.add_argument("--fno-merge-gate-hidden-dim", type=int, default=64)
    ap.add_argument("--fno-hidden-channels", type=int, default=32)
    ap.add_argument("--fno-n-modes", type=int, default=64)
    ap.add_argument("--fno-n-layers", type=int, default=2)
    ap.add_argument("--fno-head-hidden-dim", type=int, default=64)
    ap.add_argument("--fno-extent", action="store_true")
    ap.add_argument("--fno-extent-merge-init", default="neutral", choices=["neutral", "additive"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    leaf_dir = Path(args.fg_grid_dir) / f"leafq{int(args.leaf_qsentences):03d}"
    trees = load_labeled_trees(leaf_dir / "labeled_trees.jsonl")
    if args.eval_split:
        trees = [
            t
            for t in trees
            if str((t.metadata or {}).get("split") or "") == args.eval_split
        ]
    if args.max_trees and len(trees) > args.max_trees:
        trees = trees[: args.max_trees]
    if not trees:
        print(f"no trees for split={args.eval_split} in {leaf_dir}", file=sys.stderr)
        return 1
    n_set = _retarget(trees, str(args.target_dimension))
    print(f"loaded {len(trees)} trees; retargeted {n_set} node scores -> {args.target_dimension}")

    g_path = (
        Path(args.run_dir)
        / f"leafq{int(args.leaf_qsentences):03d}"
        / "iter_02_train_g"
        / "fno_state_g.pt"
    )
    if not g_path.exists():
        print(f"missing g checkpoint: {g_path}", file=sys.stderr)
        return 1

    config = FNOFamilyConfig(
        merge_mode=str(args.fno_merge_mode),
        merge_gate_hidden_dim=int(args.fno_merge_gate_hidden_dim),
        hidden_channels=int(args.fno_hidden_channels),
        n_modes=int(args.fno_n_modes),
        n_layers=int(args.fno_n_layers),
        head_hidden_dim=int(args.fno_head_hidden_dim),
        extent_enabled=bool(args.fno_extent),
        extent_merge_init=str(args.fno_extent_merge_init),
        identity_init=True,
    )
    family = FNOFamily(
        config=config,
        embedding_client=_build_embedding_client(args),
        device=str(args.device),
    )
    prepared, embedding_dim = family._prepare(trees)
    model = family._ensure_model(embedding_dim)
    family._load_state(str(g_path))
    model.eval()

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with torch.no_grad(), open(out_path, "w") as fh:
        for item in prepared:
            states = _forward_tree_states(model, item, device=family.device)
            # Match the eval's key exactly: it reads ``rec.get("doc_id")`` straight
            # from labeled_trees.jsonl, so use the tree's doc_id attribute.
            doc_id = str(getattr(item.tree, "doc_id", "") or "")
            for node_id in item.node_order:
                node = item.tree.get_node(node_id)
                if node is None or int(node.level) == 0:
                    continue  # merge nodes only (leaves have no merge)
                state = states.get(str(node_id))
                if state is None:
                    continue
                val = float(model.predict_normalized(state).reshape(-1)[0].item())
                fh.write(
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "node_id": str(node_id),
                            "compact_targets": {str(args.target_dimension): val},
                        }
                    )
                    + "\n"
                )
                n_rows += 1
    print(f"wrote {n_rows} merge-node states -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
