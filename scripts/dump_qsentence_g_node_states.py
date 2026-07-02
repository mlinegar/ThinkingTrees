#!/usr/bin/env python3
"""Dump a trained g's per-node states for the merge-by-level evaluator.

The ladder eval only persists ROOT predictions, but
``scripts/eval_qsentence_merge_by_level.py --g-states-jsonl`` needs g's state at
EVERY node to score the learned merge per level (vs equal-average bar and
mass-weighted ceiling). This loads a trained g artifact, generates states
bottom-up over the labeled trees (the same ``_generate_all_node_states_resilient``
the auditor uses), and writes one JSONL row per (doc_id, node_id):

    {"doc_id":.., "node_id":.., "compact_targets": {dim: value, ...}}

Routes over the dgemma fleet (round_robin) with the GIL-tokenizer guard off, so
all 4 GPUs stay busy. Usage:

    python scripts/dump_qsentence_g_node_states.py \
        --g-artifact <run>/dspy/leafq008/iter_02_train_g/g_qsentence_dspy_iter_02.json \
        --fg-grid-dir outputs/manifesto_qsentence_dspy_labeled_grid \
        --leaf-qsentences 8 --eval-split test --max-trees 16 \
        --dspy-api-base http://localhost:8004/v1,...8007/v1 \
        --out-jsonl <run>/g_node_states_leaf8.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

from src.ctreepo.distillation import load_labeled_trees
from src.ctreepo.manifesto_qsentence_dspy_family import (
    ManifestoQSentenceDSPyFamily,
    ManifestoQSentenceDSPyFamilyConfig,
    parse_compact_scores_json,
)


def _build_family(args: argparse.Namespace) -> ManifestoQSentenceDSPyFamily:
    return ManifestoQSentenceDSPyFamily(
        config=ManifestoQSentenceDSPyFamilyConfig(
            optimizer="gepa",
            lm_config={
                "model": str(args.dspy_model),
                "api_base": str(args.dspy_api_base),
                "api_key": "EMPTY",
                "max_tokens": int(args.dspy_max_tokens),
            },
            lm_transport="batch",
            num_threads=int(args.dspy_num_threads),
            batch_max_concurrent=int(args.dspy_num_threads),
            batch_size=int(args.dspy_batch_size),
            batch_routing_policy=str(args.dspy_batch_routing_policy),
            # GIL tokenizer guard off -> all 4 fleet GPUs stay busy (see
            # feedback_gil_tokenizer_guard_starves_fleet).
            skip_lm_input_budget_check=True,
            batch_request_timeout=float(args.dspy_batch_request_timeout),
            batch_await_response_timeout=float(args.dspy_batch_await_response_timeout),
            leaf_size_tokens=int(args.leaf_size_tokens),
            lm_context_window_tokens=int(args.dspy_lm_context_tokens),
            max_completion_tokens=int(args.dspy_max_tokens),
            target_dimensions=("all" if args.target_dimensions == "all"
                               else tuple(args.target_dimensions.split(","))),
            strict_optimizer_errors=False,
        )
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g-artifact", required=True)
    ap.add_argument(
        "--f-artifact",
        default=None,
        help="Optional trained f program. When given, every g state is ALSO read "
             "THROUGH f and the f-readout is emitted per node (f_readout field). "
             "This lets the merge eval score g the way it is actually used (f reads "
             "g's state in ANY format), instead of rigid direct-parse of compact "
             "targets — matches the r~0.64 'g composes' result. f-readout is run "
             "even for states that don't direct-parse (the point of f's flexibility).",
    )
    ap.add_argument("--fg-grid-dir", required=True)
    ap.add_argument("--leaf-qsentences", type=int, required=True)
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--max-trees", type=int, default=16)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--target-dimensions", default="all")
    ap.add_argument("--leaf-size-tokens", type=int, default=512)
    ap.add_argument("--dspy-model", default="openai/RedHatAI/diffusiongemma-26B-A4B-it-NVFP4")
    ap.add_argument("--dspy-api-base", required=True)
    ap.add_argument("--dspy-max-tokens", type=int, default=1024)
    ap.add_argument("--dspy-num-threads", type=int, default=48)
    ap.add_argument("--dspy-batch-size", type=int, default=2)
    ap.add_argument("--dspy-batch-routing-policy", default="round_robin")
    ap.add_argument("--dspy-lm-context-tokens", type=int, default=32768)
    ap.add_argument("--dspy-batch-request-timeout", type=float, default=120.0)
    ap.add_argument("--dspy-batch-await-response-timeout", type=float, default=180.0)
    args = ap.parse_args(argv)

    leaf_dir = Path(args.fg_grid_dir) / f"leafq{int(args.leaf_qsentences):03d}"
    trees = load_labeled_trees(leaf_dir / "labeled_trees.jsonl")
    if args.eval_split:
        trees = [t for t in trees if str((t.metadata or {}).get("split") or "") == args.eval_split]
    if args.max_trees and len(trees) > args.max_trees:
        trees = trees[: args.max_trees]
    if not trees:
        print(f"no trees for split={args.eval_split} in {leaf_dir}", file=sys.stderr)
        return 1

    family = _build_family(args)
    g_program = family._load_g_program(str(args.g_artifact))
    f_program = family._load_f_program(str(args.f_artifact)) if args.f_artifact else None

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Generate g's state at EVERY node for ALL trees in ONE level-synchronous
    # batched pass. This is the canonical multi-tree path: within-tree node walks
    # are sequential (a merge needs its children), but each level's wave pools
    # nodes ACROSS trees through a ThreadPoolExecutor (workers=num_threads), so
    # the whole 4-GPU fleet stays saturated. A per-tree loop pinned one GPU
    # (~1.5 req/s) and deadlocked on long chains.
    state_by_tree = family._generate_all_node_states_batched(
        g_program=g_program, trees=trees
    )

    # Optional: read EVERY g state through f. f is a flexible LLM readout, so it
    # is run even on states that don't direct-parse. These f-calls are independent
    # (no tree dependency), so run them as one flat ThreadPoolExecutor batch to
    # saturate the fleet.
    f_readout_by_key: dict = {}
    if f_program is not None:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        items = [
            (str(tree.doc_id), str(node_id), state)
            for tree, states in zip(trees, state_by_tree)
            for node_id, state in states.items()
            if str(state or "").strip()
        ]
        print(f"[dump] reading {len(items)} g states through f ...", flush=True)
        workers = max(1, int(args.dspy_num_threads))

        def read_one(item):
            doc_id, node_id, state = item
            try:
                scores = family._apply_f_scores(f_program, response=state)
            except Exception:
                scores = {}
            return doc_id, node_id, scores

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(read_one, it) for it in items]
            for fut in as_completed(futs):
                doc_id, node_id, scores = fut.result()
                if scores:
                    f_readout_by_key[(doc_id, node_id)] = {
                        k: float(v) for k, v in scores.items()
                    }
                done += 1
                if done % 200 == 0:
                    print(f"[dump] f-readout {done}/{len(items)}", flush=True)
        print(f"[dump] f-readout done: {len(f_readout_by_key)}/{len(items)} scorable",
              flush=True)

    n_nodes = 0
    n_with_f = 0
    with out_path.open("w") as fh:
        for tree, states in zip(trees, state_by_tree):
            tree_valid = 0
            for node_id, state in states.items():
                parsed = parse_compact_scores_json(state)
                f_readout = f_readout_by_key.get((str(tree.doc_id), str(node_id)))
                # Keep a node if EITHER it direct-parses OR f read it (f's job is
                # to rescue off-schema states; dropping those defeats the point).
                if not parsed and not f_readout:
                    continue
                row = {
                    "doc_id": str(tree.doc_id),
                    "node_id": str(node_id),
                    "compact_targets": {k: float(v) for k, v in parsed.items()} if parsed else None,
                }
                if f_readout:
                    row["f_readout"] = f_readout
                    n_with_f += 1
                fh.write(json.dumps(row) + "\n")
                n_nodes += 1
                tree_valid += 1
            fh.flush()
            print(f"[dump] doc={tree.doc_id} nodes={len(states)} "
                  f"valid={tree_valid} cumulative={n_nodes}", flush=True)
    print(f"[dump] wrote {n_nodes} node states ({n_with_f} with f-readout) "
          f"from {len(trees)} trees -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
