# FNO Tree Model Debugging Handoff (2026-03-16)

## Problem Statement

The `FNOCountSketch` model in `src/ctreepo/sim/core/markov_neural_operator_baselines.py` achieves MAE=1.17 on the Markov changepoint counting task — it predicts the mean count and learns nothing. The additive baseline (`AdditiveCountSketch`) achieves MAE=0.0000 on the same task.

**Task**: Count regime changepoints in a token sequence. Tokens are drawn from a Markov process with 4 regimes, each with a different distribution over 32 vocab items. Sequences have 4-8 segments (so 3-7 changepoints). The tree partitions the sequence into 4 leaf spans, merges bottom-up in a balanced binary tree, and predicts the total changepoint count at the root.

## Architecture

### AdditiveCountSketch (works perfectly)
- Pre-extracted features: `[count/scale, first_regime_onehot, last_regime_onehot]`
- Merge: `merged_count = left_count + right_count + f(left_last, right_first)` where f detects if regimes differ
- This is closed-form and trivially solves the task

### FNOCountSketch (broken)
- Leaf encoder: `token_ids → embedding → FNO1d → pool → proj → h` (state_dim=16)
- Endpoint projections: FNO output at first/last positions → `Linear(fno_width, n_regimes)` → first_reg, last_reg
- State: `[h, first_reg, last_reg]` (state_dim + 2*n_regimes = 24)
- Merger: `[left_h, right_h, left_last, right_first] → MLP → merged_h`
- Readout: `sigmoid(MLP(h)) * target_scale`
- All nodes use the same readout function `predict_norm_from_state`

### Key files
- Model + training: `src/ctreepo/sim/core/markov_neural_operator_baselines.py` (~lines 870-1900)
- Config + experiment dispatch: `src/ctreepo/sim/core/markov_changepoint_ops_count.py` (~line 691 for config, ~line 7453 for dispatch)
- Experiment scripts: `scripts/quick_fno_tree_labeling_study.py`, `scripts/quick_fno_tree_law_comparison.py`

## What Has Been Tried

### 1. Separate C1/C2/C3 law supervision (original approach)
- Root loss (50% weight) + C1 leaf loss + C2 idempotence + C3 merge smoothness
- Result: MAE=1.17. Root gets too much weight, merger nodes get ~4% each.

### 2. Endpoint projection fix
- Original bug: `encode_leaf_tokens` hardcoded `first_reg = zeros`, `last_reg = zeros` — merger was blind to boundaries.
- Fixed: project FNO output at first/last token positions through `Linear(fno_width, n_regimes)`.
- Result: MAE=1.17. Endpoint projections get ~10x weaker gradient than merger.
- Also tried raw embedding endpoints (not FNO output): MAE=1.19, slightly worse.

### 3. Unified IPW node-level supervision (flat)
- All tree nodes (4 leaf + 3 internal + 1 root = 8) in one pool with equal weight.
- Hajek-normalized MSE loss: `Σ(w_i * MSE_i) / Σ(w_i)`
- Result: MAE=1.18. Rebalanced weights but same fundamental issue.
- Code: `forward_doc_unified` (~line 1213), `train_fno_tree_ipw` (~line 1476)

### 4. Bottom-up residual decomposition
- At merge nodes, loss on residual: `MSE(g(merge) - g(left).detach() - g(right).detach(), boundary_correction/scale)`
- Boundary correction is 0 or 1 in Markov DGP — much easier target than 3-7.
- Without `.detach()`: training diverges (leaf MAE=2.78, best_epoch=1). Conflicting gradients from parent residuals corrupt leaf learning.
- With `.detach()`: stable training (best_epoch=6) but still MAE=1.20. Merge is chasing moving target.
- Code: `_balanced_merge_children_map` helper, `use_residual_decomposition` flag in `forward_doc_unified`

### 5. Rejected approaches
- **Endpoint regime classification head** (CrossEntropy on regime labels): User rejected — "NOT our leaf laws". Only the SCORE should be supervised.
- **Ad-hoc losses**: User rejected — "the entire point is that we should only supply the SCORE."

## Comparative results (same config: 128 train, 30 epochs, seed=42)

| Config | Test Root MAE | Leaf MAE | Merge MAE | Best Epoch |
|--------|--------------|----------|-----------|------------|
| flat_ipw | 1.1789 | 0.6071 | 1.0024 | 7 |
| residual_ipw (detached) | 1.1984 | 1.0348 | 0.9968 | 6 |
| Mean prediction baseline | ~1.17 | — | — | — |
| AdditiveCountSketch | 0.0000 | — | — | — |

## Root Cause Analysis

The FNO processes each leaf span independently. The model architecture has all the right pieces (FNO → endpoint projections → merger → readout), but:

1. **FNO can't distinguish regimes from token IDs alone** (or at least not with current capacity/training). The token distributions per regime likely overlap enough that a small FNO with 30 epochs of 128 documents can't learn to classify them.

2. **Endpoint projections receive only indirect gradient** through the merger, which itself receives only indirect gradient through the root loss. This is a long gradient path with several bottlenecks.

3. **The merger can't learn boundary detection** because its inputs (endpoint projections) are uninformative. If the FNO doesn't produce useful endpoint features, the merger has nothing to work with.

The additive model bypasses all of this because it receives pre-extracted regime one-hots as features. The FNO must discover regime identity from raw tokens — a fundamentally harder problem.

## Potential Next Directions

1. **Verify the FNO can learn anything about individual leaves**: Run a standalone task — given a single leaf's tokens, predict its changepoint count. If the FNO can't even do this in isolation, the tree structure is irrelevant.

2. **Increase model capacity / training**: The current setup uses fno_width=32, 2 layers, 30 epochs, 128 train docs. Maybe this is just too small. The doc-level FNO baseline (which processes the full sequence as one input) achieved near-zero error in some prior runs.

3. **Check if feature_mode matters**: The config uses `feature_mode="full"`. There may be other modes that provide more discriminative features.

4. **Curriculum or staged training**: First train the FNO leaf encoder to predict leaf counts (standalone), freeze it, then train the merger.

5. **Examine what the doc-level FNO baseline does differently**: It processes the full 64-token sequence with one FNO and gets low error. The tree FNO processes 4x16-token spans independently. Maybe 16 tokens isn't enough context for regime detection.

## Config Reference

```python
OPSCountConfig(
    n_regimes=4, vocab_size=32,
    min_tokens=64, max_tokens=64,
    min_segments=4, max_segments=8,
    min_seg_len=4, max_seg_len=16,
    fixed_leaf_tokens=16,
    train_docs=128, val_docs=32, test_docs=64,
    model_family="neural",
    state_dim=16, hidden_dim=32,
    n_epochs=30, batch_size=16, lr=1e-3,
    fno_width=32, fno_n_modes=8, fno_n_layers=2,
    use_unified_ipw=True,       # enables IPW path
    use_residual_decomposition=True,  # enables residual at merge nodes
    # target_scale = max(1, max_segments - 1) = 7
)
```

## User Constraints
- Must align with the CTreePO paper framework (flat per-node IPW, tree sampling policy q)
- Only the SCORE should be supervised — no auxiliary classification heads or ad-hoc losses
- Extensions to the paper are acceptable (residual decomposition was approved as an extension)
- The approach should generalize to the LLM setting where the "score" is a preference signal
