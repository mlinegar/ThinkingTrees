# Markov Tree Parity: Handoff Summary (2026-04-08)

## Goal

Show that a tree-structured approach (encode leaves independently, merge bottom-up) recovers the same answer as an FNO processing the full document — as guaranteed by the Lean formalization's `one_pass` theorem (L1 + L2 → root distortion = 0).

## The Markov Changepoint Setting

- **Oracle**: count regime transitions (changepoints) in a token sequence
- **Sufficient statistic**: `(count, first_regime, last_regime)`
- **Exact merge algebra**: `count(P) = count(L) + count(R) + 1[last(L) ≠ first(R)]`, `first(P) = first(L)`, `last(P) = last(R)`
- **Benchmark**: `recoverable_v4`, 4 regimes, 96 vocab, 128-token docs
- **Prepared data**: `/home/mlinegar/ThinkingTrees/outputs/parity_corpus_20260403_v4/` — pre-built FNO docs at various `leaf_tokens` (16, 32, 64, 128). Load with `_fno_doc_from_payload` from `full_doc_anchor_diagnostics.py`.

## What the Lean Proves

- `one_pass` (PreservationTheorems.lean): If L1 (leaf sufficiency) and L2 (merge consistency) hold, root distortion = 0
- `certifiedRegularizedObjective` (RegularizedObjective.lean): Licenses using oracle labels at internal nodes for training
- `MarkovMergeSupervision.lean`: Formalizes that full-sketch parent supervision recovers L2/C3

## The Unified-g Architecture (CURRENT — in progress)

**The key insight**: The Lean's `reduce` applies ONE function `g` at every tree level. Our old implementation had two separate functions (FNO leaf encoder vs merger MLP), violating this. The unified_g architecture uses a single `g = encode_summary`:

```
Leaf:   tokens → FNO → (pooled, first_tok_features, last_tok_features) → [3×fno_width summary] → g → state
Merge:  merge_summary_proj(left_state, right_state)                    → [3×fno_width summary] → g → state
                                                                                                  ↑
                                                                                             SAME g (shared weights)
```

### Critical Design Decisions

**Summary width = 3×fno_width (NO compression)**. The leaf summary IS the raw FNO features (pooled + first token + last token features concatenated). No projector, no bottleneck. This was critical — narrower summaries (5-dim, 64-dim) lost count information and training collapsed.

**The merge summary projector** maps `2×state_dim → 3×fno_width`. With production config (`state_dim=128, fno_width=128`), that's `256 → 384` — an expansion, not compression. The merge path must produce summaries at least as wide as 2× the leaf inputs to capture all information from both children.

**g = summary_encoder** is a 3-layer MLP: `3×fno_width → hidden_dim → hidden_dim → state_dim`. With production config: `384 → 512 → 512 → 128`. This is the core of the model — make it deep/wide enough.

### Architecture Details

- `tree_model_version = "unified_g"` in `src/tree/tree_model_v2.py`
- `score_merge_mode = "exact_projected_sketch"` enables the decode-compose-reencode merge path
- C2 reencode path: narrow summary (1+2×n_regimes dims) gets zero-padded to 3×fno_width before going through same g

### Batched Training

**USE THE PROPER BATCHED PIPELINE**, not ad-hoc `train_fno_tree` calls:

```bash
python scripts/run_tree_neural_full_doc_mig.py worker \
  --tree-model-version unified_g \
  --tree-score-merge-mode exact_projected_sketch \
  --fixed-leaf-tokens 64 \
  --train-doc-count 4096 \
  --benchmark recoverable_v4 \
  --gpu-runtime-data-mode resident \
  --tree-batch-pack-mode fixed_fused \
  --tree-batch-autotune \
  --prepared-data-root .../prepared_tree_data \
  --prepared-data-allow-create \
  --base-bundle-path .../bundles/bundle_train4096.pkl \
  --use-cuda \
  ...
```

The `--gpu-runtime-data-mode resident` pre-caches tree structures on GPU. The `--tree-batch-pack-mode fixed_fused` batches multiple docs per GPU call. DO NOT use the per-doc `train_fno_tree` loop — it's CPU-bottlenecked.

### Results: unified_g BEATS the FNO baseline

| Metric | unified_g (leaf64) | v2 baseline (leaf64) | official_fno (leaf128) |
|--------|-------------------|---------------------|----------------------|
| **test_root_mae** | **0.0232** | 0.0877 | 0.0410 |
| test_leaf_mae | 0.0514 | 0.0895 | — |
| test_merge_mae | 0.0259 | 0.0916 | — |
| test_c2_idempotence | 0.5821 | 0.4610 | — |
| best_epoch | 39 | — | — |

**Root MAE 0.023 beats the official FNO (0.041) by 1.8×.** The 2-leaf tree with unified_g is MORE accurate than the FNO processing the full document. And it's 3.8× better than the old v2 architecture (0.088).

Config: `leaf_tokens=64` (2 leaves per 128-token doc), `train_docs=4096`, 40 epochs (10 stage1 + 30 stage2), `state_dim=128`, `fno_width=128`, `hidden_dim=512`.

Output: `outputs/unified_g_leaf64_test/summary.json`

### Key Files Modified This Session

- `src/tree/tree_model_v2.py`: Added `"unified_g"` as valid `TreeModelVersion`
- `src/ctreepo/sim/core/markov_neural_operator_baselines.py`:
  - `__init__`: `unified_g_leaf_summary_proj` (None = pass-through), `unified_g_merge_summary_proj`, `unified_g_summary_dim`
  - `summary_encoder` (= g): 3-layer MLP for unified_g
  - `encode_leaf_tokens_batch`: unified_g path concatenates FNO features → `encode_summary`
  - `_merge_state_pairs`: unified_g routes to `merge_summary_proj` → `encode_summary`
  - `encode_summary`: dispatches wide (unified_g) vs narrow (C2 zero-padded) inputs
  - `_summary_spec_merge_consistency_terms`: Fixed detach bug, added oracle join BCE
  - C3 block in `forward_doc`: Oracle supervision + algebraic consistency with join BCE

## What Was Established This Session

### Local Law Alignment with Lean

- **C2 = L3**: Pure on-range reencode only (removed merge contamination that was in C2)
- **C3 = L2**: Oracle supervision at merges (Lean-licensed) + algebraic consistency as secondary. Join head gets direct BCE supervision. Child counts detached in algebra.
- **C1 = L1**: Unchanged (leaf supervision against oracle)

### Progression Test (4-step exact→neural)

| Step | What | Root MAE |
|------|------|----------|
| 0: Exact sketch + exact merge | Lean Theorem 1 | **0.000** |
| 1: Exact leaves → learned merge | Merger quality | 0.53-1.17 |
| 2: Learned encoder → exact merge | Encoder quality | 0.03-0.06 |
| 3: Full system | Both | 0.12-0.28 |

Encoder works. Merger was the bottleneck. Oracle C3 helped (0.28→0.12).

### Internal State Inspection

The merger learned shortcut functions, not the algebra. C3 algebra gap 0.28-1.44. This motivated the unified_g redesign.

### Merge Signal Lab (1024 docs, exact leaves)

`teacher_parent_count` (oracle count at merges) was best at 0.112 root MAE. `teacher_parent_full_sketch` achieved 97% merge exact match. `strict_c3` alone was terrible (0.96).

## Stage1 Artifact Cache Warning

Stage1 artifacts at `outputs/_stage1_artifacts/` are cached by config hash, NOT code hash. Code changes require manual deletion of relevant artifacts before re-running.

## What to Do Next

1. **Verify one-leaf coincidence**: Run unified_g at `leaf_tokens=128` (1 leaf = full doc). Should match or beat official_fno. This establishes the Theorem 1 base case.
2. **Check C3 algebra gap**: Use `scripts/inspect_tree_internal_states.py` (updated for unified_g) to trace internal states. Does the merge now follow the algebra, or still shortcut?
3. **Scale to deeper trees**: Test `leaf_tokens=32` (4 leaves, 3 merges) and `leaf_tokens=16` (8 leaves, 7 merges). The Lean guarantee should hold at any depth.
4. **Run with more seeds**: The current result is seed 0 only. Run seeds 0-2 to check stability.
5. **C2 idempotence**: Still 0.58 — investigate whether this matters for correctness or is just a diagnostic artifact with unified_g.

## Key File Paths

- **Model**: `src/ctreepo/sim/core/markov_neural_operator_baselines.py`
- **Tree version**: `src/tree/tree_model_v2.py`
- **Diagnostics**: `src/ctreepo/sim/core/full_doc_anchor_diagnostics.py`
- **Worker**: `scripts/run_tree_neural_full_doc_mig.py`
- **Launcher**: `scripts/run_markov_supervision_recovery_parity_grid.py`
- **Inspector**: `scripts/inspect_tree_internal_states.py`
- **Progression test**: `scripts/test_markov_exact_progression.py`
- **Lean**: `lean3/FormalProofs/OPT/{LocalLaws,PreservationTheorems,MarkovMergeSupervision}.lean`
- **Prepared data**: `outputs/parity_corpus_20260403_v4/benchmark_corpora/recoverable_v4/`
- **Memory**: `/home/mlinegar/.claude/projects/-home-mlinegar-ThinkingTrees/memory/MEMORY.md`
