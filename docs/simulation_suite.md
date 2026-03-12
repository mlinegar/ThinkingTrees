# ThinkingTrees Simulation Suite

This doc is the "table of contents" for the simulation families in `ThinkingTrees/src/tree/`, plus the standard workflow (run -> sweep -> plot -> report).

## Conventions

- Per-run outputs are written as `seed_<n>.json` and `seed_<n>.csv` under `outputs/<sim_name>/<grid_path>/`.
- Sweep orchestration is xargs-friendly:
  - build commands to a text file in `logs/`
  - run with `xargs -P <workers> ...`
- Plot scripts consume the per-run `seed_*.json` outputs via `--input-glob`.
- Lean alignment is reported separately from figure generation:
  - `scripts/check_simulation_expectations.py` checks family-level qualitative expectations
  - `scripts/report_simulation_theory_alignment.py` maps those families and canonical suites onto the Lean theorem surface

## Core Suite (Recommended)

These families should now be read as a ladder, not a grab bag:

1. exact ordinary-LDA recovery through mergeable tree summaries,
2. learned mergeable compression against that exact ceiling,
3. structured latent-process extensions and C-TreePO-style auditing/calibration.

If a new simulation family does not fit somewhere on that ladder, it is probably not central.

### 1) Tree-Relevant LDA Ladder

Goal: first prove exact recovery of a mergeable task-relevant statistic in ordinary LDA, then compress it, then move to a local-mixture model where leaves matter statistically.

- overview: `docs/tree_relevant_lda_simulation_ladder.md`
- Stage-1 background spec: `docs/lda_tree_recovery_simulation_spec.md`
- Lean base case: `lean3/FormalProofs/OPT/BagOfWordsLDARecovery.lean`
- Lean local-mixture gap: `lean3/FormalProofs/OPT/LeafLocalMixtureUtilityGap.lean`
- Stage-1 runner: `scripts/run_lda_tree_utility_vector_simulation.py`
- Stage-1 sweep builder: `scripts/build_lda_tree_utility_vector_cmds.py`
- Stage-2 runner: `scripts/run_leaf_local_mixture_utility_simulation.py`
- Stage-2 sweep builder: `scripts/build_leaf_local_mixture_utility_cmds.py`
- paper report: `scripts/report_lda_tree_methods_paper.py`
- diagnostic legacy family:
  - exact runner: `scripts/run_lda_tree_recovery_simulation.py`
  - learned runner: `scripts/run_lda_tree_recovery_learned_simulation.py`

Main intended comparisons:

- Stage 1 full-document exact utility vs exact tree utility,
- count-compression ceiling vs direct utility-sketch compression,
- Stage 2 pooled-document wrong model vs leaf-aware ridge from exact leaf utility sketches,
- fixed-budget vs all-leaves-labeled resolution tradeoffs.

This is now the first paper-facing topic-model family. The older `lda_tree_recovery` learned family stays as an appendix diagnostic.

### 2) Markov Changepoint OPS Count (Merge Learning)

Goal: learn a merge operator for a nontrivial associative target (changepoint count), under node-label budgets and different audit strategies. This is the cleanest place to reason about associativity and schedule dependence.

- DGP: piecewise-constant hidden regime process; we observe `token_regimes` (so oracle labels are exact).
- Ground-truth baselines (computed on the same test set):
  - `exact`: oracle mergeable summary (endpoints + count), should be 0 error / 0 spread.
  - `undersupported`: count-only summary, associative but biased (misses join indicator).
  - `flip_R1/R2`: controlled non-mergeable resummary stress tests.
- Key metric: `schedule_spread_mean` ("spread") = for each doc, merge leaves with multiple schedules and compute `max(pred_root)-min(pred_root)`.

Run / sweep / plot / report:

- runner: `scripts/run_markov_changepoint_ops_count_simulation.py`
- sweep builder: `scripts/build_markov_changepoint_ops_count_cmds.py`
- plots:
  - grid: `scripts/plot_markov_changepoint_ops_count_grid.py` (use `--layout honesty` for learned vs baseline)
  - lines: `scripts/plot_markov_changepoint_ops_count_lines.py` (add `--include-flip-baselines` if desired)
  - ceilings: `scripts/plot_markov_changepoint_ops_count_ceilings.py` (explicit exact ceiling + undersupported floor + schedule-spread scatter)
- PDF report: `scripts/report_markov_changepoint_ops_count_run.py`

### 3) Segment-LDA OPS Weight Recovery (Tensor-LDA Bridge)

Goal: upstream topic recovery (oracle/noisy/TensorLDA/online-TensorLDA) -> downstream OPS/audit training -> root/merge/leaf + weight recovery. This is the main end-to-end pipeline for "Tensor-LDA -> ThinkingTrees".

- DGP:
  - `topic_process=segments` (piecewise-constant topic segments, aligned to leaves)
  - optional `topic_process=bag_of_words` baseline
- Upstream estimators: `topic_phi_estimator ∈ {true, noisy_theory, tensor_lda, online_tensor_lda}`
- Downstream: ridge regression on mergeable features; optional `--run-all-feature-modes` emits oracle vs inferred feature upper bounds:
  - `ridge_true_topics`
  - `ridge_infer_true_phi`
  - `ridge_infer_est_phi`

Run / sweep / plot:

- runner: `scripts/run_segment_lda_ops_weight_recovery_simulation.py`
- sweep builder: `scripts/build_segment_lda_ops_weight_recovery_cmds.py`
- plots:
  - grid: `scripts/plot_segment_lda_ops_weight_recovery_grid.py`
  - lines: `scripts/plot_segment_lda_ops_weight_recovery_lines.py`
  - ceilings: `scripts/plot_segment_lda_ops_weight_recovery_ceilings.py` (compares ridge vs ridge_* ceilings; run sims with `--run-all-feature-modes`)

### 4) Segmented-LDA C-TreePO Simulation (Audit/Calibration Decomposition)

Goal: C-TreePO-style policies (oracle proxy vs estimated/calibrated/budgeted) with explicit decomposition terms (`topic`, `calibration`, `guidance`, `slack`, etc).

Run / sweep / plot:

- runner: `scripts/run_segmented_lda_ctreepo_simulation.py`
- sweep builder: `scripts/build_segmented_lda_ctreepo_cmds.py`
- plots:
  - phase heatmap: `scripts/plot_segmented_lda_ctreepo_phase.py`
  - lines/bands: `scripts/plot_segmented_lda_ctreepo_lines.py`
  - ceilings/ablations: `scripts/plot_segmented_lda_ctreepo_ceilings.py` (policy gains + decomposition tightness)

## Additional Families (Existing)

These are still useful, but are not currently the primary TLDA -> C-TreePO bridge.

- Tensor-LDA book benchmark:
  - runner: `scripts/run_tensor_lda_book_weight_benchmark.py`
  - sim: `src/tree/tensor_lda_book_weight_benchmark.py`
- Guidance/budget Markov sims:
  - `scripts/run_markov_changepoint_cut_budget_simulation.py`
  - plots: `scripts/plot_markov_changepoint_cut_budget_guidance_grid.py`, `scripts/plot_markov_changepoint_cut_budget_scaling.py`
- Honesty / selection-bias demos:
  - ledger: `scripts/run_ledger_honesty_simulation.py`
  - markov boundary honesty: `scripts/run_markov_boundary_honesty_simulation.py`
  - markov boundary chunker honesty: `scripts/run_markov_boundary_chunker_honesty_simulation.py`
  - markov changepoint honesty: `scripts/run_markov_changepoint_honesty_simulation.py`
- Preference learning toy:
  - runner: `scripts/run_markov_changepoint_preference_simulation.py`
  - plots: `scripts/plot_markov_changepoint_preference_scaling.py`
- Earlier sketch/bigram families:
  - `scripts/run_learned_sketch_simulation.py` (+ `scripts/plot_learned_sketch_*.py`)
  - `scripts/run_bigram_score_guidance_simulation.py` (+ `scripts/plot_bigram_score_guidance_*.py`)
- IPW CI demo:
  - `scripts/run_ipw_ci_simulation.py`
- Mergeable validation (sketch sufficiency + budget retention ceilings):
  - sim: `src/tree/mergeable_ablation.py`
  - plots: `scripts/plot_mergeable_ceilings.py`, `scripts/plot_mergeable_complexity_ladder.py`, `scripts/plot_mergeable_chunk_quality_sweep.py`

## Typical "Big Run" Workflow

Example pattern (repeat per sim or per suite):

```bash
cd /home/mlinegar/ThinkingTrees

# 1) Build commands
venv/bin/python scripts/build_markov_changepoint_ops_count_cmds.py \
  --out-cmds logs/<name>_cmds.txt \
  --output-root outputs/<name> \
  --skip-existing true

# 2) Run (parallel)
nohup bash -lc 'cat logs/<name>_cmds.txt | xargs -I{} -P 24 bash -lc "{}"' \
  > logs/<name>.log 2>&1 \
  & echo $! > logs/<name>.pid

# 3) Report
venv/bin/python scripts/report_markov_changepoint_ops_count_run.py --input-root outputs/<name>
```
