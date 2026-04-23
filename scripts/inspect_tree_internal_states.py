#!/usr/bin/env python3
"""Inspect internal states at every stage of the tree computation.

For a concrete document, traces: tokens → leaf states → merge state → readout,
showing what the model has learned at each stage and where error accumulates.

Usage:
  venv/bin/python scripts/inspect_tree_internal_states.py --use-cuda
  venv/bin/python scripts/inspect_tree_internal_states.py --n-train 1024 --n-epochs 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.diffusion.markov_toy import (
    changepoint_count,
    encode_markov_path,
    merge_markov_sketch,
)
from src.tree.markov_changepoint_honesty_simulation import ChangepointMarkovDoc
from src.ctreepo.sim.core.markov_changepoint_ops_count import _oracle_count, _leaf_spans


def _generate_simple_docs(n_docs, n_tokens, n_regimes, vocab_size, seed):
    rng = np.random.RandomState(seed)
    docs = []
    for _ in range(n_docs):
        regimes = np.zeros(n_tokens, dtype=np.int64)
        regimes[0] = rng.randint(0, n_regimes)
        for t in range(1, n_tokens):
            if rng.random() < 0.3:
                regimes[t] = rng.randint(0, n_regimes)
            else:
                regimes[t] = regimes[t - 1]
        tokens = np.zeros(n_tokens, dtype=np.int64)
        for t in range(n_tokens):
            regime = int(regimes[t])
            lo = (regime * vocab_size) // n_regimes
            hi = ((regime + 1) * vocab_size) // n_regimes
            tokens[t] = rng.randint(lo, max(lo + 1, hi))
        boundaries = np.nonzero(regimes[:-1] != regimes[1:])[0].astype(np.int64)
        docs.append(ChangepointMarkovDoc(
            tokens=tuple(int(x) for x in tokens),
            token_regimes=tuple(int(x) for x in regimes),
            transition_regimes=tuple(int(x) for x in regimes[1:]),
            true_boundaries=tuple(int(x) for x in boundaries),
        ))
    return docs


def inspect_document(model, doc, leaf_tokens: int, doc_idx: int = 0):
    """Trace the full tree computation for one document."""
    device = next(model.parameters()).device
    model.eval()

    regimes = list(doc.token_regimes)
    n_tok = len(regimes)
    oracle_root = _oracle_count(doc, start=0, end=n_tok)
    spans = _leaf_spans(n_tok, leaf_tokens=leaf_tokens)
    n_leaves = len(spans)

    print(f"\n{'='*70}")
    print(f"Document {doc_idx}: {n_tok} tokens, {n_leaves} leaves, oracle_root_count={oracle_root}")
    print(f"  Regimes: {regimes}")
    print(f"  Boundaries: {list(doc.true_boundaries)}")
    print(f"{'='*70}")

    # --- Encode leaves ---
    token_ids = []
    for s, e in spans:
        ids = list(doc.tokens[s:e])
        pad_len = leaf_tokens - len(ids)
        token_ids.append(ids + [model.pad_id] * pad_len)

    with torch.no_grad():
        tokens_t = torch.tensor(token_ids, dtype=torch.long, device=device)
        leaf_states = model.encode_leaf_tokens_batch(tokens_t, device=device)

        leaf_infos = []
        for i, (s, e) in enumerate(spans):
            st = leaf_states[i:i+1]
            pred_count = model.predict_count_from_state(st).item()
            oracle_c = _oracle_count(doc, start=s, end=e)
            oracle_first = int(regimes[s])
            oracle_last = int(regimes[e - 1])

            # Extract theorem feature (phi)
            phi = model.theorem_feature_from_state(st)

            # Get first/last predictions
            first_surface = model._first_surface_from_state(st)
            last_surface = model._last_surface_from_state(st)
            if hasattr(model, 'first_endpoint_proj') and model.first_endpoint_proj is not None:
                first_logits = model.first_endpoint_proj(first_surface)
            else:
                first_logits = first_surface
            if hasattr(model, 'last_endpoint_proj') and model.last_endpoint_proj is not None:
                last_logits = model.last_endpoint_proj(last_surface)
            else:
                last_logits = last_surface
            pred_first = torch.argmax(first_logits, dim=-1).item()
            pred_last = torch.argmax(last_logits, dim=-1).item()
            first_probs = F.softmax(first_logits, dim=-1).squeeze()
            last_probs = F.softmax(last_logits, dim=-1).squeeze()

            info = {
                "span": (s, e),
                "state": st.squeeze().cpu().numpy(),
                "phi": phi.squeeze().cpu().numpy(),
                "pred_count": pred_count,
                "oracle_count": oracle_c,
                "pred_first": pred_first,
                "oracle_first": oracle_first,
                "pred_last": pred_last,
                "oracle_last": oracle_last,
                "first_probs": first_probs.cpu().numpy(),
                "last_probs": last_probs.cpu().numpy(),
            }
            leaf_infos.append(info)

            print(f"\n  Leaf {i} (tokens {s}:{e}): regimes={regimes[s:e]}")
            print(f"    Oracle:    count={oracle_c}, first={oracle_first}, last={oracle_last}")
            print(f"    Predicted: count={pred_count:.3f}, first={pred_first} (p={first_probs[pred_first]:.2f}), last={pred_last} (p={last_probs[pred_last]:.2f})")
            print(f"    C1 error:  |{pred_count:.3f} - {oracle_c}| = {abs(pred_count - oracle_c):.4f}")
            print(f"    State[128] norm={np.linalg.norm(info['state']):.3f}, first 5: [{', '.join(f'{v:.3f}' for v in info['state'][:5])}]")
            print(f"    Phi[{len(info['phi'])}] norm={np.linalg.norm(info['phi']):.3f}, first 5: [{', '.join(f'{v:.3f}' for v in info['phi'][:5])}]")

        # --- Merge (if >1 leaf) ---
        if n_leaves > 1:
            print(f"\n  --- Merge ---")
            # For 2-leaf case, one merge
            current = leaf_states
            merge_level = 0
            while current.shape[0] > 1:
                n = current.shape[0]
                n_pairs = n // 2
                if n_pairs > 0:
                    left = current[0:2*n_pairs:2]
                    right = current[1:2*n_pairs:2]
                    merged = model._merge_state_pairs(left, right)

                    for j in range(n_pairs):
                        left_idx = 2 * j
                        right_idx = 2 * j + 1
                        parent_st = merged[j:j+1]

                        pred_parent_count = model.predict_count_from_state(parent_st).item()
                        parent_phi = model.theorem_feature_from_state(parent_st)

                        # Oracle for merged span
                        if merge_level == 0:
                            left_span = spans[left_idx]
                            right_span = spans[right_idx]
                        else:
                            # Approximate for deeper merges
                            left_span = (0, n_tok // 2)
                            right_span = (n_tok // 2, n_tok)
                        merged_s = left_span[0]
                        merged_e = right_span[1]
                        oracle_merge_count = _oracle_count(doc, start=merged_s, end=merged_e)
                        oracle_join = 1 if regimes[left_span[1]-1] != regimes[right_span[0]] else 0

                        # Join bit prediction
                        left_st = current[left_idx:left_idx+1]
                        right_st = current[right_idx:right_idx+1]
                        join_logit = model.predict_join_logit_from_states(left_st, right_st)
                        join_prob = torch.sigmoid(join_logit).item()

                        # First/last for parent
                        first_s = model._first_surface_from_state(parent_st)
                        last_s = model._last_surface_from_state(parent_st)
                        if hasattr(model, 'first_endpoint_proj') and model.first_endpoint_proj is not None:
                            p_first_logits = model.first_endpoint_proj(first_s)
                        else:
                            p_first_logits = first_s
                        if hasattr(model, 'last_endpoint_proj') and model.last_endpoint_proj is not None:
                            p_last_logits = model.last_endpoint_proj(last_s)
                        else:
                            p_last_logits = last_s
                        p_first = torch.argmax(p_first_logits, dim=-1).item()
                        p_last = torch.argmax(p_last_logits, dim=-1).item()

                        oracle_first = int(regimes[merged_s])
                        oracle_last = int(regimes[merged_e - 1])

                        parent_state_np = parent_st.squeeze().cpu().numpy()
                        parent_phi_np = parent_phi.squeeze().cpu().numpy()

                        print(f"\n  Merge level {merge_level}, pair {j}: spans {merged_s}:{merged_e}")
                        print(f"    Oracle:    count={oracle_merge_count}, first={oracle_first}, last={oracle_last}, join={oracle_join}")
                        print(f"    Predicted: count={pred_parent_count:.3f}, first={p_first}, last={p_last}, join_prob={join_prob:.3f}")
                        print(f"    Merge algebra: pred_L({leaf_infos[left_idx]['pred_count']:.2f}) + pred_R({leaf_infos[right_idx]['pred_count']:.2f}) + join({join_prob:.2f}) = {leaf_infos[left_idx]['pred_count'] + leaf_infos[right_idx]['pred_count'] + join_prob:.3f}")
                        print(f"    Actual parent count: {pred_parent_count:.3f}")
                        print(f"    C3 algebra gap: {abs(pred_parent_count - (leaf_infos[left_idx]['pred_count'] + leaf_infos[right_idx]['pred_count'] + join_prob)):.4f}")
                        print(f"    Root error: |{pred_parent_count:.3f} - {oracle_merge_count}| = {abs(pred_parent_count - oracle_merge_count):.4f}")
                        print(f"    State[{len(parent_state_np)}] norm={np.linalg.norm(parent_state_np):.3f}")
                        print(f"    Phi[{len(parent_phi_np)}] norm={np.linalg.norm(parent_phi_np):.3f}")

                    if n % 2 == 1:
                        current = torch.cat([merged, current[-1:]], dim=0)
                    else:
                        current = merged
                merge_level += 1

            root_count = model.predict_count_from_state(current[0:1]).item()
        else:
            root_count = model.predict_count_from_state(leaf_states[0:1]).item()

        print(f"\n  --- Root ---")
        print(f"    Oracle root count: {oracle_root}")
        print(f"    Predicted root count: {root_count:.3f}")
        print(f"    Root MAE: {abs(root_count - oracle_root):.4f}")

        # --- Exact sketch comparison ---
        print(f"\n  --- Exact Sketch (Step 0 reference) ---")
        for i, (s, e) in enumerate(spans):
            segment = [str(r) for r in regimes[s:e]]
            sketch = encode_markov_path(segment)
            print(f"    Leaf {i}: exact=(count={sketch.changepoints}, first={sketch.start_state}, last={sketch.end_state})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-regimes", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=4)
    parser.add_argument("--n-tokens", type=int, default=8)
    parser.add_argument("--leaf-tokens", type=int, default=4)
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-inspect", type=int, default=3)
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Generate data
    train_docs = _generate_simple_docs(args.n_train, args.n_tokens, args.n_regimes, args.vocab_size, args.seed)
    inspect_docs = _generate_simple_docs(args.n_inspect, args.n_tokens, args.n_regimes, args.vocab_size, args.seed + 99)

    from src.ctreepo.sim.core.markov_neural_operator_baselines import (
        FNOCountSketch, _prepare_fno_count_docs, train_fno_tree,
    )

    target_scale = max(1, int(args.n_tokens * 0.3))
    fno_train = _prepare_fno_count_docs(train_docs, leaf_tokens=args.leaf_tokens)
    fno_val = _prepare_fno_count_docs(inspect_docs, leaf_tokens=args.leaf_tokens)

    # Build model — unified_g: one encode function for both leaves and merges
    fno_width = 64
    model = FNOCountSketch(
        state_dim=fno_width,
        hidden_dim=128,
        n_regimes=args.n_regimes,
        vocab_size=args.vocab_size,
        fno_width=fno_width,
        fno_n_modes=4,
        fno_n_layers=2,
        leaf_tokens=args.leaf_tokens,
        target_scale=float(target_scale),
        tree_model_version="unified_g",
        score_merge_mode="exact_projected_sketch",
        summary_spec_name="markov_count_sketch",
        slot_count=4,
        theorem_feature_dim=32,
        theorem_feature_hidden_dim=64,
        theorem_score_dim=1,
        theorem_fiber_dim=31,
        theorem_aux_dim=0,
        # score_merge_mode set above for unified_g
        join_bit_weight=1.0,
        c2_mode="reconstruction",
        root_supervision_kind="mse",
        task_head_mode="theorem_feature_scalar",
        summary_spec_root_mode="factored_theorem_readout",
        theorem_count_head_mode="scalar_mse",
        theorem_count_ordinal_weight=1.0,
        theorem_count_scalar_aux_weight=0.25,
        theorem_feature_adapter="markov_count_sketch",
        theorem_count_dim=8,
        theorem_first_dim=8,
        theorem_last_dim=8,
    ).to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"state_dim={model.state_dim}, fno_width={fno_width}")

    # Inspect BEFORE training (random init)
    print("\n" + "=" * 70)
    print("BEFORE TRAINING (random initialization)")
    print("=" * 70)
    for i, doc in enumerate(inspect_docs[:2]):
        inspect_document(model, doc, args.leaf_tokens, doc_idx=i)

    # Train
    print(f"\nTraining {args.n_epochs} epochs on {args.n_train} docs...")
    train_fno_tree(
        model=model, train_docs=fno_train, val_docs=fno_val,
        device=device, n_epochs=args.n_epochs, batch_size=16, lr=1e-3, weight_decay=0.0,
        c1_weight=1.0, c2_weight=1.0, c3_weight=1.0, root_weight=1.0,
        leaf_query_rate=1.0, leaf_label_rate=1.0, audit_fraction=1.0,
        internal_supervision_kind="full_sketch", internal_label_rate=1.0,
        leaf_supervision_kind="full_sketch", leaf_exact_supervision=True,
        tree_local_weighting_mode="fixed_k_hajek", checkpoint_metric="val_root_mae",
    )

    # Inspect AFTER training
    print("\n" + "=" * 70)
    print("AFTER TRAINING")
    print("=" * 70)
    for i, doc in enumerate(inspect_docs[:args.n_inspect]):
        inspect_document(model, doc, args.leaf_tokens, doc_idx=i)


if __name__ == "__main__":
    main()
