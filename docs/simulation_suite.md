# ThinkingTrees Simulation Suite

This doc is the "table of contents" for the simulation families in `ThinkingTrees/src/tree/`, plus the standard workflow (run -> sweep -> plot -> report).

## Conventions

- Per-run outputs are written as `seed_<n>.json` and `seed_<n>.csv` under `outputs/<sim_name>/<grid_path>/`.
- Family sweeps are still xargs-friendly, but the newer curated suites prefer manifests + the shared resource queue:
  - build manifests / command files from Python
  - run with `scripts/run_simulation_resource_queue.py` or a suite `run` entrypoint
- Plot scripts consume the per-run `seed_*.json` outputs via `--input-glob`.
- Lean alignment is reported separately from figure generation:
  - `scripts/check_simulation_expectations.py` checks family-level qualitative expectations
  - `scripts/report_simulation_theory_alignment.py` maps those families and canonical suites onto the Lean theorem surface

## Curated Suite API

For the paper-facing experiment bundles, the canonical interface is now:

```bash
venv/bin/python -m src.ctreepo.cli sim suite <suite-name> build ...
venv/bin/python -m src.ctreepo.cli sim suite <suite-name> run --output-root <root> --jobs <n> --gpu-tokens auto
venv/bin/python -m src.ctreepo.cli sim suite <suite-name> report --output-root <root>
```

Current suite names:

- `identifiable-zero`
- `publication-ctreepo`
- `identifiable-zero-publication`
- `identifiable-zero-learnability`
- `law-stress`
- `cpu-megasweep`
- `simulation-buildout`
- `identifiable-zero-neural-operator`
- `identifiable-zero-lda-leafnoise`
- `identifiable-zero-dtm-lda`
- `lda-tree-recovery-progress`
- `learned-sketch-smoke`
- `markov-observed-token`

Common behavior:

- `build` emits `suite_meta.json`, `suite_cmds.txt`, and a manifest-first `suite_manifest.jsonl`
- `run` defaults to the shared resource queue when a suite manifest exists
- `report` is the canonical user-facing reporting entrypoint for paper/appendix/diagnostic suites
- `--rebuild` is the standard way to force manifest regeneration before execution
- `--fail-fast` remains available on the older suites when you explicitly want the legacy `cmds.txt` executor
- canonical suite metadata and bundle/theory-alignment mapping live in `src/ctreepo/sim/suite/registry.py`

## Validation Ladder

For the fastest repo-facing end-to-end check, use this ladder:

1. `learned-sketch-smoke`: the smallest mergeable-sketch + tiny learned operator validation.
2. `identifiable-zero-learnability --profile smoke --groups markov_baseline`: the real Markov end-to-end smoke.
3. `markov-observed-token --profile demo`: fixed-bundle observed-token comparison between root-only learning and sampled local labels.

This third step is now the canonical appendix-facing way to demonstrate the specific claim that the Markov changepoint task can be learned from observed tokens alone, without exposing latent regime/color labels.

Recommended commands:

```bash
cd /home/mlinegar/ThinkingTrees

venv/bin/python -m src.ctreepo.cli sim suite learned-sketch-smoke build \
  --output-root outputs/learned_sketch_smoke_<stamp>
venv/bin/python -m src.ctreepo.cli sim suite learned-sketch-smoke run \
  --output-root outputs/learned_sketch_smoke_<stamp> \
  --jobs 1 \
  --gpu-tokens none
venv/bin/python -m src.ctreepo.cli sim suite learned-sketch-smoke report \
  --output-root outputs/learned_sketch_smoke_<stamp> \
  --no-emit-pdf

venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-learnability build \
  --profile smoke \
  --groups markov_baseline \
  --output-root outputs/markov_learnability_smoke_<stamp>
venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-learnability run \
  --output-root outputs/markov_learnability_smoke_<stamp> \
  --groups markov_baseline \
  --jobs 1 \
  --gpu-tokens none
venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-learnability report \
  --output-root outputs/markov_learnability_smoke_<stamp> \
  --no-emit-pdf

venv/bin/python -m src.ctreepo.cli sim suite markov-observed-token build \
  --profile demo \
  --output-root outputs/markov_observed_token_<stamp>
venv/bin/python -m src.ctreepo.cli sim suite markov-observed-token run \
  --output-root outputs/markov_observed_token_<stamp> \
  --jobs 1 \
  --gpu-tokens none
venv/bin/python -m src.ctreepo.cli sim suite markov-observed-token report \
  --output-root outputs/markov_observed_token_<stamp> \
  --no-emit-pdf
```

Expected runtime on a normal local CPU is on the order of seconds for the learned-sketch smoke, well under a minute for the Markov smoke, and roughly tens of seconds for the observed-token Markov demo profile.

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
- Paper-facing hazard panels: use `docs/markov_hazard_panels.md` as the
  source of truth for the mixed `paper_hazard_panel_v1_t128` and
  `paper_hazard_panel_v1_t2048` corpora. These panels deliberately mix
  low/high switch density and 4/12-regime conditions so a single global mean
  transition-count predictor is exposed by condition-wise diagnostics.
- Ground-truth baselines (computed on the same test set):
  - `exact`: oracle mergeable summary (endpoints + count), should be 0 error / 0 spread.
  - `undersupported`: count-only summary, associative but biased (misses join indicator).
  - `flip_R1/R2`: controlled non-mergeable resummary stress tests.
- Key metric: `schedule_spread_mean` ("spread") = for each doc, merge leaves with multiple schedules and compute `max(pred_root)-min(pred_root)`.
- Contextual-sufficiency control layer: `docs/contextual_sbijax_optimize_to_zero_resolved_2026-05-05.md`
  records the optimize-to-zero resolution on the t128 hazard panel.
  `learned_local_laws` is the exact-zero trainer; package NASS/NASSS are
  approximate baselines. This is a diagnostic/control layer, not a paper LaTeX
  update.
- Post-resolution ablations:
  `docs/markov_contextual_sufficiency_ablation_handoff_2026-05-05.md` and
  `outputs/markov_contextual_ablation_grid_report_20260505.md` record the
  NASS/NASSS auxiliary grid, learned merge/decoder grid, and standalone
  `CleanUnifiedNO` general f/g grid. Current conclusion: local-law /
  Markov-sketch supervision selects the sufficient state; general f/g remains
  an open bridge.

Current paper prep / run path:

```bash
source venv/bin/activate
python scripts/prepare_markov_hazard_panel_data.py

python scripts/run_markov_optimization_tradeoff_pipeline.py \
  --config config/markov/tradeoff_pipeline.hazard_panel_paper.toml \
  --plan-only
```

The preparation step writes the seed-0 raw bundles under
`outputs/_bundles/markov_hazard_panels/{panel_id}/seed_0/base_bundle.json` and
the prepared tree/FNO caches under
`outputs/_prepared_data/markov_hazard_panels/{panel_id}/prepared_*`. The
standard paper train ladder is `[1024, 4096, 10240]`, and each prefix is
condition-balanced by construction.

Run / sweep / plot / report:

- runner: `scripts/run_markov_changepoint_ops_count_simulation.py`
- sweep builder: `scripts/build_markov_changepoint_ops_count_cmds.py`
- plots:
  - grid: `scripts/plot_markov_changepoint_ops_count_grid.py` (use `--layout honesty` for learned vs baseline)
  - lines: `scripts/plot_markov_changepoint_ops_count_lines.py` (add `--include-flip-baselines` if desired)
  - ceilings: `scripts/plot_markov_changepoint_ops_count_ceilings.py` (explicit exact ceiling + undersupported floor + schedule-spread scatter)
- Dedicated PDF report: archived. See `docs/markov_report_archive.md`.

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

### 5) Identifiable-Zero Suite (Shared API)

Goal: build the paper-facing identifiable-zero family from one typed policy instead of keeping the train-doc, audit-budget, and seed grids duplicated in a legacy builder script.

Key files:

- suite policy: `src/ctreepo/sim/suite/identifiable_zero_policy.py`
- suite builder / runner: `src/ctreepo/sim/suite/identifiable_zero.py`
- top-level CLI entrypoint: `venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero ...`
- retired migration stub: `scripts/build_identifiable_zero_suite_cmds.py`
- report: `venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero report ...`

What this centralizes:

- one policy for the segment-LDA, C-TreePO, and optional Markov train grids / audit grids / seeds
- one canonical `suite_meta.json` + `suite_manifest.jsonl` for the mixed-family paper bundle
- one canonical entrypoint instead of a second independent builder implementation

Recommended usage:

```bash
venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero build \
  --output-root outputs/identifiable_zero_suite_<stamp>

venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero run \
  --output-root outputs/identifiable_zero_suite_<stamp> \
  --jobs 64 \
  --gpu-tokens auto
```

### 6) Publication C-TreePO Suite (Shared API)

Goal: keep the publication C-TreePO comparison lanes in one resolved lane catalog instead of hand-maintaining each lane in shell loops.

Key files:

- suite policy: `src/ctreepo/sim/suite/publication_policy.py`
- publication lane catalog: `src/ctreepo/sim/suite/publication_lanes.py`
- suite builder / runner: `src/ctreepo/sim/suite/publication_ctreepo.py`
- top-level CLI entrypoint: `venv/bin/python -m src.ctreepo.cli sim suite publication-ctreepo ...`
- report / expectations:
  - `venv/bin/python -m src.ctreepo.cli sim suite publication-ctreepo report ...`
  - `venv/bin/python -m src.ctreepo.cli sim suite publication-ctreepo expectations ...`

What this centralizes:

- one policy for the publication train-doc, calibration-rate, eval-rate, and seed grids
- one lane catalog for `lda_direct`, `phi_base`, `neural_weak`, `neural_default`, and `neural_upper`
- one manifest-first build that records the resolved lane metadata in `suite_meta.json`

Recommended usage:

```bash
venv/bin/python -m src.ctreepo.cli sim suite publication-ctreepo build \
  --output-root outputs/identifiable_zero_publication_ctreepo_<stamp>

venv/bin/python -m src.ctreepo.cli sim suite publication-ctreepo run \
  --output-root outputs/identifiable_zero_publication_ctreepo_<stamp> \
  --jobs 64 \
  --gpu-tokens auto
```

### 7) Identifiable-Zero Learnability Suite (Shared API)

Goal: run the paper-facing Markov + segmented-LDA learnability slices from one shared experiment definition, with synchronized train-doc grids, label-rate grids, held-out sizes, seeds, and matched no-tree baselines.

Key files:

- suite policy: `src/ctreepo/sim/suite/learnability_policy.py`
- suite builder / runner: `src/ctreepo/sim/suite/identifiable_zero_learnability.py`
- top-level CLI entrypoint: `venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-learnability ...`
- retired migration stub: `scripts/run_identifiable_zero_learnability_overnight.sh`
- report: `venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-learnability report ...`

What this centralizes:

- one policy for `train_docs_grid`, `label_rate_grid`, `heldout_docs`, base seeds, hero seeds, and C-TreePO guidance rates
- one grouped manifest / command-file build for:
  - `markov_baseline`, `markov_hard`, `markov_hard_hero`
  - `ctree_baseline_lstsq`, `ctree_baseline_theta`
  - `ctree_hard_lstsq`, `ctree_hard_theta`, hero hard slices
  - `ctree_lda_lstsq`, `ctree_lda_theta`
- matched no-tree baselines carried through the canonical APIs:
  - Markov doc-level baseline (`--include-doc-level-baseline`)
  - C-TreePO full-document theta baseline (`--include-full-doc-theta-baseline`)

Recommended usage:

```bash
venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-learnability build \
  --output-root outputs/identifiable_zero_learnability_v1_demo

venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-learnability run \
  --output-root outputs/identifiable_zero_learnability_v1_demo \
  --jobs 64 \
  --gpu-tokens auto
```

Useful controls:

- `--groups "markov_baseline ctree_baseline_theta"` to run only a subset
- policy overrides such as `--train-docs-grid`, `--label-rate-grid`, `--heldout-docs`, `--base-seeds`, `--hero-seeds`
- `--hero/--no-hero`
- `--rebuild` on `run` when you want the suite manifest regenerated before execution

### 8) Identifiable-Zero Publication Packs (Shared API)

Goal: move the remaining paper-facing identifiable-zero publication-clean and longrun oracle-equivalence packs onto the same suite/policy/manifest contract, instead of keeping separate builder scripts with duplicated grids.

Key files:

- suite policy: `src/ctreepo/sim/suite/identifiable_zero_publication_policy.py`
- suite builder / runner: `src/ctreepo/sim/suite/identifiable_zero_publication.py`
- top-level CLI entrypoint: `venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-publication ...`
- retired migration stubs:
  - `scripts/build_identifiable_zero_publication_clean_cmds.py`
  - `scripts/build_identifiable_oracle_equivalence_longrun_cmds.py`
- publication-clean report:
  - `venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-publication report --profile publication_clean ...`

Profiles:

- `publication_clean`
  - grouped slices: `cpu`, `gpu`
  - centralizes the reduced publication-clean segment-LDA, C-TreePO, and Markov packs
- `longrun_equiv_v1`
  - grouped slices: `equiv`, `scale`, `pilot`
  - centralizes the longrun equivalence / scale study and derives a bounded pilot manifest from the same resolved run catalog

What this centralizes:

- one typed policy for the publication-clean segment-LDA, C-TreePO, and Markov sweep defaults
- one typed policy for the longrun equivalence / scale packs, including `pilot_cmd_count`, `target_main_jobs`, and `target_pilot_minutes`
- one canonical grouped manifest layout under `suite_groups/{cmds,manifests}`
- retired migration stubs for the old publication-clean and longrun builder scripts instead of separate builder implementations

Recommended usage:

```bash
venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-publication build \
  --profile publication_clean \
  --output-root outputs/identifiable_zero_publication_clean_<stamp>

venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-publication run \
  --profile publication_clean \
  --output-root outputs/identifiable_zero_publication_clean_<stamp> \
  --groups cpu \
  --jobs 48 \
  --gpu-tokens none
```

Longrun usage:

```bash
venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-publication build \
  --profile longrun_equiv_v1 \
  --output-root outputs/identifiable_zero_longrun_equiv_<stamp>

venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-publication run \
  --profile longrun_equiv_v1 \
  --output-root outputs/identifiable_zero_longrun_equiv_<stamp> \
  --groups "equiv scale" \
  --jobs 48 \
  --gpu-tokens auto
```

Useful controls:

- `--groups` to build or run only `cpu`, `gpu`, `equiv`, `scale`, or `pilot`
- `--n-seeds`, `--segment-test-docs`, `--ctree-test-books`, `--markov-test-docs`, `--markov-n-epochs` for reduced smoke/backfill variants

### 9) Unified Law-Stress Suite (Shared API)

Goal: build and run the cross-DGP Markov + LDA law-stress suites from one entrypoint, with grouped manifests per suite slice instead of separate shell wrappers.

Key files:

- suite policy: `src/ctreepo/sim/suite/law_stress_policy.py`
- builder library: `src/ctreepo/sim/suite/law_stress_builders.py`
- suite builder / runner: `src/ctreepo/sim/suite/law_stress.py`
- top-level CLI entrypoint: `venv/bin/python -m src.ctreepo.cli sim suite law-stress ...`
- retired migration stubs:
  - `scripts/build_markov_law_stress_suite_cmds.py`
  - `scripts/build_lda_law_stress_suite_cmds.py`
- unified report: `venv/bin/python -m src.ctreepo.cli sim suite law-stress report --family <markov|lda> ...`

Current grouped slices:

- Markov:
  - `markov_sanity_suite`
  - `markov_transition_map_suite`
  - `markov_mechanism_suite`
  - `markov_capacity_appendix_suite`
  - `markov_cross_dgp_suite`
  - `markov_weight_ablation_suite`
- LDA:
  - `lda_sanity_suite`
  - `lda_transition_map_suite`
  - `lda_mechanism_suite`

Recommended usage:

```bash
venv/bin/python -m src.ctreepo.cli sim suite law-stress build \
  --output-root outputs/law_stress_suite_<stamp> \
  --groups "markov_sanity_suite lda_sanity_suite"

venv/bin/python -m src.ctreepo.cli sim suite law-stress run \
  --output-root outputs/law_stress_suite_<stamp> \
  --jobs 32 \
  --gpu-tokens auto
```

Notes:

- `markov_mechanism_suite` still requires `--transition-summary <path>` because it is conditioned on the transition-map report output
- the grid defaults now live in `src/ctreepo/sim/suite/law_stress_policy.py`, which is shared by the Markov and LDA law-stress builders
- the old family builders are retired migration stubs; canonical builds now come from `ctreepo sim suite law-stress ...`

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
  - `venv/bin/python -m src.ctreepo.cli sim suite learned-sketch-smoke ...` (canonical repo-facing smoke path)
  - `scripts/run_learned_sketch_simulation.py` (+ `scripts/plot_learned_sketch_*.py`) for lower-level direct runs
  - `scripts/run_bigram_score_guidance_simulation.py` (+ `scripts/plot_bigram_score_guidance_*.py`)
- IPW CI demo:
  - `scripts/run_ipw_ci_simulation.py`
- Mergeable validation (sketch sufficiency + budget retention ceilings):
  - sim: `src/tree/mergeable_ablation.py`
  - plots: `scripts/plot_mergeable_ceilings.py`, `scripts/plot_mergeable_complexity_ladder.py`, `scripts/plot_mergeable_chunk_quality_sweep.py`

## Typical "Big Run" Workflow

Example pattern for a single family:

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
# Dedicated OPS-count PDF reporting is archived. See docs/markov_report_archive.md.
```

Recommended pattern for the shared learnability suite:

```bash
cd /home/mlinegar/ThinkingTrees

venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-learnability build \
  --output-root outputs/identifiable_zero_learnability_v1_<stamp>

venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-learnability run \
  --output-root outputs/identifiable_zero_learnability_v1_<stamp> \
  --jobs 64 \
  --gpu-tokens auto

venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-learnability report \
  --output-root outputs/identifiable_zero_learnability_v1_<stamp>
```

Tiny Markov-only smoke:

```bash
venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-learnability build \
  --profile smoke \
  --groups markov_baseline \
  --output-root outputs/markov_learnability_smoke_<stamp>

venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-learnability run \
  --output-root outputs/markov_learnability_smoke_<stamp> \
  --groups markov_baseline \
  --jobs 1 \
  --gpu-tokens none

venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-learnability report \
  --output-root outputs/markov_learnability_smoke_<stamp> \
  --no-emit-pdf
```

Recommended pattern for the unified law-stress suite:

```bash
cd /home/mlinegar/ThinkingTrees

venv/bin/python -m src.ctreepo.cli sim suite law-stress build \
  --output-root outputs/law_stress_suite_<stamp> \
  --groups "markov_sanity_suite lda_sanity_suite"

venv/bin/python -m src.ctreepo.cli sim suite law-stress run \
  --output-root outputs/law_stress_suite_<stamp> \
  --jobs 32 \
  --gpu-tokens auto

venv/bin/python -m src.ctreepo.cli sim suite law-stress report \
  --family markov \
  --output-root outputs/law_stress_suite_<stamp>
```

Recommended pattern for the publication-clean / longrun publication packs:

```bash
cd /home/mlinegar/ThinkingTrees

venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-publication build \
  --profile publication_clean \
  --output-root outputs/identifiable_zero_publication_clean_<stamp>

venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-publication run \
  --profile publication_clean \
  --output-root outputs/identifiable_zero_publication_clean_<stamp> \
  --groups cpu \
  --jobs 48 \
  --gpu-tokens none

venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-publication report \
  --profile publication_clean \
  --output-root outputs/identifiable_zero_publication_clean_<stamp>
```
