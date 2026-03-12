# OPS Simulation Visualizations: What They Test and How to Read Them

This note is meant to become paper text later. It explains what each simulation grid is *for*, which
theoretical claim it is probing, and how to choose metrics/visualizations so the story is coherent.

## What we want from “OPS-style” simulations

In the Lean formalization and the C-TreePO framing, an OPS simulation should make it easy to see:

1) **Approximation bias (sketch insufficiency):** even with infinite data, an underspecified sketch
   cannot be oracle-preserving.
2) **Estimation error (finite data/labels):** with a sufficient sketch family, root error should
   shrink as we increase training documents and/or oracle labels.
3) **Selection bias (adaptive sampling):** when we choose which internal nodes to label
   adaptively, naive estimates can be biased; IPW/DSL correct this.

These are “model-level” questions about mergeable summaries, not “where should we cut?” questions
(see `docs/markov_cut_budget_guidance_vs_ops.md`).

## Simulation families and their roles

### 1) Bigram score guidance grid (split-invariant target / identifiability)

Files:
- Generator + metrics: `src/tree/bigram_score_guidance_simulation.py`
- Runner: `scripts/run_bigram_score_guidance_simulation.py`
- Grid plotter: `scripts/plot_bigram_score_guidance_grid.py`

Oracle:
- `f⋆(x) = Σ_t w[token_t, token_{t+1}]` (or topic-bigram version).

What it tests:
- **Identifiability from leaf-only supervision:** if we only query leaves, we cannot learn the
  cross-leaf boundary term. Adding internal-node queries resolves that and can drive error to ~0.

Why it’s useful:
- It is the cleanest “boundary metadata is necessary” demonstration tied directly to
  `lean3/FormalProofs/OPT/BigramSketch.lean`.

What the grid should show:
- Root RMSE / weight recovery vs (train docs × extra internal labels per leaf).
- Oracle cost ratio vs accuracy (to communicate the price of guidance).

### 2) Markov changepoint OPS-count grid (OPS semantics / learned merge)

Files:
- Simulation: `src/tree/markov_changepoint_ops_count_simulation.py`
- Runner: `scripts/run_markov_changepoint_ops_count_simulation.py`
- Sweep builder: `scripts/build_markov_changepoint_ops_count_cmds.py`
- Grid plotter: `scripts/plot_markov_changepoint_ops_count_grid.py`
- Line plotter: `scripts/plot_markov_changepoint_ops_count_lines.py`
- Ceiling plotter (paper-figure shaped): `scripts/plot_markov_changepoint_ops_count_ceilings.py`

Oracle:
- `f⋆(x) = # changepoints` in a regime sequence (integer-valued).

What it tests:
- A learned sketch family with an explicit leaf encoder + merge operator, audited at leaves (C1)
  and merges (C3) under a realized tree reduction.
- Baselines like `exact` (0 distortion) and `undersupported` (bias floor) separate approximation
  bias from estimation.

Important nuance: “merge violation rate” depends on a tolerance `τ`

The simulation defines a thresholded violation event `|pred−true| > τ` and reports its mean as
`merge_violation_rate`. This is only meaningful if `τ` is set to a scale that matches how the
learned predictor outputs values.

If:
- `τ = 0.0`, and
- the learned model outputs real-valued counts (continuous),

then the event `pred == true` is measure-zero and the violation rate will be ~1.0 almost everywhere.
In that regime, a better plot is **Merge MAE**, which reflects magnitude directly.

Schedule spread is a different diagnostic (associativity / L3-ish)

`schedule_spread_mean` measures how much the learned root prediction changes when we change the
reduction schedule (balanced vs left-to-right vs right-to-left).

- This is *not required* for one-pass correctness under a fixed realized schedule.
- It *is* a useful “robustness / associativity” diagnostic, and it’s related to the multi-round
  story (L3 / on-range idempotence type failures).

What the grid should show (recommended default):
- **Root MAE** (document-level distortion).
- **Merge MAE** (C3 magnitude).
- **Schedule spread mean** (non-associativity / schedule dependence).
- Optional: add the audit-bias panels with:
  - `scripts/plot_markov_changepoint_ops_count_grid.py --include-bias-panels`

Paper-facing “honesty grid” (recommended central figure):
- Use:
  - `scripts/plot_markov_changepoint_ops_count_grid.py --layout honesty --baseline-sketch undersupported`
- This produces one figure that visually separates:
  1) **Estimation error**: learned sketch Root/Leaf/Merge MAE improves with more data + labels.
  2) **Approximation bias floor**: undersupported sketch Root/Leaf/Merge MAE does not improve with data/labels.
  3) **Selection bias**: audit estimator abs bias (naive vs IPW vs DSL) under risk-biased sampling.
- Recommended paper defaults: `--aggregate median --normalize`

Canonical sweep + plot recipe:
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

If you *want* to use CUDA, pick a free GPU and run fewer jobs in parallel, e.g.:
`CUDA_VISIBLE_DEVICES=2 JOBS=1 ...` (otherwise you'll typically hit CUDA OOM when other servers are running).

Learning-curve view (recommended companion figure):
- Plot Root/Merge MAE vs either:
  - `train_docs` (with separate lines for `audit_fraction`), or
  - total oracle queries (train) (a single x-axis combining docs × label budgets).
- Overlay baselines (`exact`, `undersupported`) to make the bias floor explicit.

Ceiling view (recommended “make the max obvious” companion):
- Use `scripts/plot_markov_changepoint_ops_count_ceilings.py` to plot:
  1) root error vs labels with both the **exact ceiling** and **undersupported floor** overlaid, and
  2) schedule spread vs root error (robustness / associativity diagnostic) colored by budget.

How to interpret “distinct regions”:
- A region with low root/merge MAE but high schedule spread indicates the model is using
  schedule-dependent computation: it can be accurate under the training schedule while being
  non-associative.
- A region with low schedule spread but worse root/merge MAE often corresponds to a nearly
  schedule-invariant (sometimes near-constant) predictor.

Target scaling matters for learnability

The learned sketch in `markov_changepoint_ops_count_simulation.py` trains on a normalized target
`count / target_scale` passed through a sigmoid. If `target_scale` is chosen to be much larger than
the true maximum count (e.g. scaling by token length when the generator’s count is bounded by
`max_segments - 1`), then targets become extremely small and SGD can drive the model into a
near-zero, sigmoid-saturated regime. In that regime:

- Root predictions collapse toward 0 for many seeds (root MAE ≈ mean true root count),
- “More training” (more docs or more steps) can make collapse more likely, producing non-monotone
  grids like “50 docs looks better than 100 docs.”

For paper-facing sweeps, ensure `target_scale` matches the oracle’s actual scale (for this DGP,
`max_segments - 1`).

Fix the evaluation set (so “more docs” means “better”, not “different test data”)

When sweeping `train_docs`, prefer generating the **test set independently** of `train_docs`
so that each curve point is evaluated on the same distribution *and* the same held-out sample
for a given seed. Otherwise, apparent non-monotonicity can be partly due to swapping out the
test documents.

### 3) Segment‑LDA OPS weight recovery grid (latent structure + mergeability)

Files:
- Simulation: `src/tree/segment_lda_ops_weight_recovery_simulation.py`
- Runner: `scripts/run_segment_lda_ops_weight_recovery_simulation.py`
- Grid plotter: `scripts/plot_segment_lda_ops_weight_recovery_grid.py`
- Ceiling plotter (uses `--run-all-feature-modes` outputs): `scripts/plot_segment_lda_ops_weight_recovery_ceilings.py`
- Spec note (paper text): `docs/segment_lda_ops_simulation_spec.md`

Oracle:
- `f⋆(span) = <θ, topic_counts(span)> + λ <W, topic_bigrams(span)>`

What it tests:
- Same “boundary metadata / bigram mergeability” structure as the bigram sim, but with **words
  emitted from latent topics** (LDA-ish) and explicit recovery objectives:
  - recover sparse topic weights `θ`,
  - recover the bigram direction and the scalar multiplier `λ`.

Why it’s useful:
- It bridges the toy mergeability tests to a more realistic “latent structure → learned sketch”
  story while staying aligned with the Lean sketches (unigram counts + bigram counts + boundary ids).

Ceilings (make the “upper bounds” explicit):
- When you run the sim with `--run-all-feature-modes`, the JSON includes:
  - `ridge_true_topics` (best-case downstream estimator under the same audit budget),
  - `ridge_infer_true_phi` (topic inference only),
  - `ridge_infer_est_phi` (topic inference + phi estimation),
  in addition to `ridge`.
- `scripts/plot_segment_lda_ops_weight_recovery_ceilings.py` plots these together with `exact` (absolute ceiling)
  and `undersupported` (approximation-bias floor), plus a “gap to ceiling” panel.

### 4) Segmented‑LDA C‑TreePO end-to-end decomposition (component gains + bound tightness)

Files:
- Simulation: `src/tree/segmented_lda_ctreepo_simulation.py`
- Runner: `scripts/run_segmented_lda_ctreepo_simulation.py`
- Sweep builder: `scripts/build_segmented_lda_ctreepo_cmds.py`
- Phase plotter: `scripts/plot_segmented_lda_ctreepo_phase.py`
- Lines plotter: `scripts/plot_segmented_lda_ctreepo_lines.py`
- Ceiling/ablation plotter (paper-figure shaped): `scripts/plot_segmented_lda_ctreepo_ceilings.py`

What it tests:
- The end-to-end chain (oracle proxy → topic estimation → calibration → budgeted guidance) and the corresponding
  triangle-inequality decomposition into **topic**, **calibration**, **guidance**, and **oracle-proxy** components.

What the ceiling figure should show:
- **Ablation gains vs ceiling:** root error curves for each policy family, with `oracle_tree` at 0 distortion.
- **Decomposition tightness:** scatter of realized total vs the decomposition upper bound with a y=x reference.

## Visualization design guidelines (paper-facing)

1) **Always label tolerances and scales.**
   - If you show a violation probability, include `τ` in the title/caption.
2) **Prefer magnitude plots when outputs are continuous.**
   - MAE/RMSE are stable and comparable across settings; thresholded rates are easy to misread.
3) **Make schedule spread explicit as “robustness”, not core OPS correctness.**
   - If the correctness claim is “under this realized schedule”, then spread is a separate axis.
4) **Show baselines (exact / undersupported) at least once.**
   - This makes the bias floor visible and connects directly to Lean examples like
     `lean3/FormalProofs/OPT/MarkovCountSketchExample.lean`.
5) **Be explicit about seed aggregation (mean vs median vs quantiles).**
   - Learned sketches can have heavy-tailed outcomes across seeds (some runs find a very accurate but
     highly non-associative merger; others collapse to a near-constant predictor).
   - In that regime, heatmaps of the *mean* can show “distinct regions” that are actually driven by
     a small number of outlier seeds.
   - For paper figures, prefer either:
     - a robust statistic (median), or
     - two complementary panels (e.g. median + std / IQR), or
     - a “success rate” heatmap (fraction of seeds with root error below a threshold).

## Concrete next improvements (if we want cleaner grids)

For Markov OPS-count:
- Consider setting `--violation-tau 1.0` for “>1 changepoint error” (more interpretable than `0.0`).
- Optionally add a training regularizer or mixed-schedule training if we want schedule spread to
  decrease monotonically with more C3 supervision (associativity pressure).

For Bigram guidance:
- Consider reporting “boundary-weight error” separately (the identifiable-vs-not-identifiable part),
  but the current grid is already a good paper figure.
