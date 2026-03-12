# Tree + Topic-Model Simulation Suite

This suite should now be read in a strict order.

The end goal is:

1. show that the tree system can reproduce ordinary bag-of-words LDA exactly when it keeps the right mergeable statistic,
2. quantify the cost of compressing that statistic with learned/neural operators,
3. then move to local latent structure where leaves genuinely matter.

This doc is a single index for the simulation families that connect:

- **exact ordinary-LDA recovery**,
- **mergeable sketch compression**,
- **topic recovery** (Tensor-LDA-style upstream estimators / rates),
- **OPS semantics** (mergeable summaries + C1/C3 checks),
- **C-TreePO behaviors** (calibration + guidance/query budgets + selection-bias auditing),

in a way that yields interpretable grids/learning curves.

## Families (what to run)

1) **Tree-Relevant LDA Ladder** (main paper entry point)
- Overview: `docs/tree_relevant_lda_simulation_ladder.md`
- Stage-1 spec background: `docs/lda_tree_recovery_simulation_spec.md`
- Lean base case: `lean3/FormalProofs/OPT/BagOfWordsLDARecovery.lean`
- Lean local-mixture gap: `lean3/FormalProofs/OPT/LeafLocalMixtureUtilityGap.lean`
- Stage-1 runner: `scripts/run_lda_tree_utility_vector_simulation.py`
- Stage-1 sweep builder: `scripts/build_lda_tree_utility_vector_cmds.py`
- Stage-2 runner: `scripts/run_leaf_local_mixture_utility_simulation.py`
- Stage-2 sweep builder: `scripts/build_leaf_local_mixture_utility_cmds.py`
- Paper report: `scripts/report_lda_tree_methods_paper.py`
- Diagnostic legacy family:
  - exact runner: `scripts/run_lda_tree_recovery_simulation.py`
  - learned runner: `scripts/run_lda_tree_recovery_learned_simulation.py`
- Purpose:
  - Stage 1: exact ordinary-LDA recovery of task-relevant mergeable utility sketches,
  - Stage 1: compression of those sketches with exact ceilings,
  - Stage 2: local-mixture extension where pooled document summaries lose information and leaf-aware methods help.

This is now the first family in the paper story. The older `lda_tree_recovery*` family remains useful as a diagnostic appendix, not the main claim.

2) **Markov changepoint OPS-count** (scalar oracle, clean OPS semantics)
- Simulation: `src/tree/markov_changepoint_ops_count_simulation.py`
- Runner: `scripts/run_markov_changepoint_ops_count_simulation.py`
- Sweep builder: `scripts/build_markov_changepoint_ops_count_cmds.py`
- Plot grid: `scripts/plot_markov_changepoint_ops_count_grid.py`
- Doc: `docs/markov_cut_budget_guidance_vs_ops.md`

3) **Segment‑LDA OPS weight recovery** (topic unigram+bigram oracle, ridge recovery)
- Simulation: `src/tree/segment_lda_ops_weight_recovery_simulation.py`
- Runner: `scripts/run_segment_lda_ops_weight_recovery_simulation.py`
- Sweep builder: `scripts/build_segment_lda_ops_weight_recovery_cmds.py`
- Plot grid: `scripts/plot_segment_lda_ops_weight_recovery_grid.py`
- Plot lines: `scripts/plot_segment_lda_ops_weight_recovery_lines.py`
- Spec: `docs/segment_lda_ops_simulation_spec.md`

4) **Segmented‑LDA end‑to‑end C‑TreePO decomposition** (main “triangle chain” benchmark)
- Simulation: `src/tree/segmented_lda_ctreepo_simulation.py`
- Runner: `scripts/run_segmented_lda_ctreepo_simulation.py`
- Grid runner: `scripts/grid_segmented_lda_ctreepo_simulation.py`
- Sweep builder: `scripts/build_segmented_lda_ctreepo_cmds.py`
- Plot phase: `scripts/plot_segmented_lda_ctreepo_phase.py`
- Plot lines: `scripts/plot_segmented_lda_ctreepo_lines.py`
- Doc: `docs/segmented_lda_ctreepo_end_to_end.md`

5) **Traditional LDA “books” benchmark** (Tensor-LDA paper-style DGP + C-TreePO-style tree)
- Simulation: `src/tree/tensor_lda_book_weight_benchmark.py`
- Runner: `scripts/run_tensor_lda_book_weight_benchmark.py`
- Doc: `docs/tensor_lda_ctreepo_benchmark.md`

## Shared knobs (upstream topic estimation)

Both Segment‑LDA OPS weight recovery and Segmented‑LDA C‑TreePO support:

- `topic_phi_estimator ∈ {true, noisy_theory, tensor_lda, online_tensor_lda, spectral_numpy}`
- `topic_phi_docs` (effective unlabeled corpus size used to estimate `φ̂`)

Suggested workflow:

1) start with the exact bag-of-words LDA tree-recovery base family,
2) then use `Markov` and `Segment-LDA OPS` as mergeability/operator-learning stress tests,
3) then swap to `tensor_lda` and `online_tensor_lda` for algorithmic topic-estimation baselines,
4) use `spectral_numpy` as a fast “data-driven but crude” proxy when needed.

## Sweep workflow (xargs/nohup)

Example: Markov OPS-count honesty grid:

```bash
cd /home/mlinegar/ThinkingTrees
venv/bin/python scripts/build_markov_changepoint_ops_count_cmds.py \
  --device cpu \
  --out-cmds logs/markov_changepoint_ops_count_cmds.txt

JOBS=$(nproc)
nohup bash -lc "xargs -P $JOBS -I{} bash -lc \"{}\" < logs/markov_changepoint_ops_count_cmds.txt" \
  > logs/markov_changepoint_ops_count_sweep.log 2>&1 &

venv/bin/python scripts/plot_markov_changepoint_ops_count_grid.py \
  --layout honesty --aggregate median --normalize \
  --output-figure outputs/markov_changepoint_ops_count_honesty_grid.png
```

Example: Segmented‑LDA C‑TreePO sweep command list:

```bash
cd ThinkingTrees
venv/bin/python scripts/build_segmented_lda_ctreepo_cmds.py \
  --topic-phi-estimator noisy_theory \
  --out-cmds logs/segmented_lda_ctreepo_cmds.txt

JOBS=$(nproc)
nohup bash -lc "xargs -P $JOBS -I{} bash -lc \"{}\" < logs/segmented_lda_ctreepo_cmds.txt" \
  > logs/segmented_lda_ctreepo_sweep.log 2>&1 &
```

Example: Segment‑LDA OPS weight recovery sweep command list:

```bash
cd ThinkingTrees
venv/bin/python scripts/build_segment_lda_ops_weight_recovery_cmds.py \
  --out-cmds logs/segment_lda_ops_weight_recovery_cmds.txt

JOBS=$(nproc)
nohup bash -lc "xargs -P $JOBS -I{} bash -lc \"{}\" < logs/segment_lda_ops_weight_recovery_cmds.txt" \
  > logs/segment_lda_ops_weight_recovery_sweep.log 2>&1 &
```

## Plotting conventions

- Prefer plotting from per-run JSON outputs (`outputs/**/**/*.json`) so you can slice/filter later.
- Use the family-specific plotters listed above; they share the same “glob + filters + aggregate/bands” pattern.
