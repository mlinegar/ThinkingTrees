# Mergeable Method Validation Report

This document summarizes what we built, how it works, and what the simulations currently show.

It consolidates the results from:

- `scripts/plot_mergeable_k_m_phase.py`
- `scripts/plot_mergeable_chunk_quality_sweep.py`
- `scripts/plot_mergeable_nonlanguage_suite.py`
- `scripts/plot_mergeable_nonlanguage_coverage.py`
- `scripts/plot_mergeable_complexity_ladder.py`

## 1. Goal

Validate a repeated-aggregation method for non-additive targets under chunking and budget constraints, and separate:

1. merge-rule correctness,
2. sketch sufficiency (`m` vs `k`),
3. chunk/selection quality,
4. uncertainty calibration (coverage).

Primary target family:

- `P(count >= k | spike)` where `count` is number of threshold-exceeding events in a document.

## 2. Core Method

For each document:

1. chunk tokens (fixed or adaptive),
2. optionally select only a budgeted subset of chunks,
3. aggregate kept chunks with a merge-safe top-`m` sketch,
4. estimate non-additive event indicators from the sketch.

If sketch order satisfies `m >= k` and relevant chunks are retained, top-`m` is sufficient for `count >= k`.
If `m < k`, the estimator is information-limited (not just noisy).

## 3. Ablations We Implemented

- `one_pass` reference: full document in one chunk.
- `full_model` aligned adaptive chunking + top-proxy selection + merge-safe sketch.
- `naive_majority`, `naive_mean_of_means`.
- `right_rule_wrong_chunker`: merge-safe rule, but misspecified chunking/selection.
- sketch-order sweeps for `m<k`, `m=k`, `m>k`.
- chunk-quality sweeps over leaf size and budget.
- nonlanguage scenario suite (ICU, intrusion, manufacturing, ECG).
- empirical CI coverage sweeps (Wilson intervals).

## 4. What We Added in Code

Main simulation module:

- `src/tree/mergeable_ablation.py`
  - `run_chunk_quality_sweep`
  - `run_chunk_quality_coverage_sweep`
  - `default_nonlanguage_chunk_quality_scenarios`
  - dataclasses:
    - `ChunkQualitySweepSummary`
    - `ChunkQualityCoverageSummary`
    - `NonLanguageScenario`

Plot runners:

- `scripts/plot_mergeable_k_m_phase.py`
- `scripts/plot_mergeable_chunk_quality_sweep.py`
- `scripts/plot_mergeable_nonlanguage_suite.py`
- `scripts/plot_mergeable_nonlanguage_coverage.py`

Documentation:

- `docs/mergeable_ablation_examples.md` (expanded usage and interpretation)

## 5. Key Results

### 5.1 Sketch sufficiency and regime behavior (`m` vs `k`)

From `outputs/mergeable_k_m_phase_summary.json`:

- Naive baselines are high-bias:
  - majority: `0.4375`
  - mean-of-means: `0.5391`
- Exact-supported (`m=k`) merge-safe methods stay low (`~0.036` to `~0.038` mean abs bias).
- Unsupported penalty (`m<k`) is large:
  - for `k=5`, unsupported-minus-exact is about `+0.45`.
- Over-supported (`m>k`) is near exact (penalty close to `0`).

Interpretation:

- The sharp failure is primarily a sketch-capacity issue when `m<k`.

### 5.2 Budget and leaf granularity

From `outputs/mergeable_chunk_quality_sweep_summary.json` (`k=5`, `m=5`, fixed + top-proxy):

- Perfect token leaves reference:
  - `target_capture=1.00`
  - `spike_recall=1.00`
  - `spike_isolation=1.00`
- Fine leaves (`size=1`) can still fail when budget is too small:
  - `b<=4`: `target_capture=0.00`, `abs_bias≈0.20`
- Same fine leaves recover once budget is sufficient:
  - `b=6`: `target_capture=1.00`, `abs_bias≈0.036`
- Coarse leaves (`size=16`) can have `isolation=0.00` but still high capture once budget is enough.

Interpretation:

- Local leaf quality and global selection budget are separate levers.
- High-quality leaves alone do not guarantee correctness if retained mass is too small.

### 5.3 Nonlanguage generalization

From `outputs/mergeable_nonlanguage_suite_summary.json` (`n_replicates=40`, `docs_per_replicate=100`, seed `7`):

- `network_intrusion_bursts`:
  - aligned abs-bias `~0.25 -> ~0.043` as budget increases
  - misspecified remains worse at high budget (`~0.094`)
- `icu_alarm_stream`:
  - aligned `~0.094 -> ~0.028`
- `manufacturing_defect_line`:
  - aligned `~0.029 -> ~0.012`

Interpretation:

- Behavior is structural and not language-specific.
- Same mergeable logic transfers to generic ordered signals (time windows, sensor bins, flow windows).

### 5.4 Empirical CI coverage

From `outputs/mergeable_nonlanguage_coverage_summary.json` (`ci=95%`, `n_replicates=60`, `docs_per_replicate=120`, seed `7`):

- Aligned coverage improves strongly with budget:
  - ICU: `0.050 -> 0.950`
  - Intrusion: `0.000 -> 0.983`
  - ECG: `0.317 -> 0.967`
- Manufacturing is easy enough that coverage is near/above nominal even at low budget.

Interpretation:

- Low-budget undercoverage indicates structural misspecification/chunk loss.
- As chunk quality and retained mass improve, coverage approaches nominal.
- Overcoverage with wide intervals can appear when effective conditional sample sizes are small.

### 5.5 Complexity ladder: necessity and failure attribution

From `outputs/mergeable_complexity_ladder_summary.json`:

- Positive controls are stable across stages:
  - one-pass oracle aggregate abs-bias: `0.0272` (stage 1), `0.0348` (stage 2), `0.0370` (stage 4), `0.0261` (stage 5)
  - full model aggregate abs-bias: `0.0311`, `0.0389`, `0.0335`, `0.0268`
- Right rule + wrong chunker remains structurally biased:
  - `0.6085` (stage 1), `0.4706` (stage 2), `0.3902` (stage 4)
- Naive majority degrades as complexity rises:
  - `0.4257` (stage 2), `0.5849` (stage 4), `0.8206` (stage 5)

Target-specific stage-4 signatures isolate missing sufficient statistics:

- `full_model_missing_boundary_stat`:
  - low on non-boundary targets (`0.0329`, `0.0426`, `0.0292`)
  - high on boundary target (`0.6500`)
- `full_model_missing_three_stat`:
  - low on other targets (`0.0319`, `0.0424`, `0.0420`)
  - high on `P(>=3|spike)` (`0.2997`)

Generic-`k` sufficiency boundary (`m>=k`) appears directly:

- `full_model_limited_sketch` (top-3 sketch):
  - supported targets stay low-bias (`k=2: 0.0373`, `k=3: 0.0257`)
  - unsupported targets jump (`k=4: 0.0805`, `k=5: 0.1277`)

Interpretation:

- These are parameter-specific failures, not generic degradation.
- Missing representation causes failure exactly on the corresponding target.
- Wrong chunking/selection causes broad failure even with otherwise correct merge rules.

## 6. Why This Demonstrates “Works”

The method passes a strong validation pattern:

1. One-pass and sufficiently supported merge-safe configurations recover known targets with low bias.
2. Controlled ablations fail exactly where theory predicts:
   - `m<k` (insufficient sketch),
   - bad chunking/selection (lost relevant chunks),
   - naive non-merge-safe aggregators.
3. Recovery and coverage improve along expected levers:
   - larger budget,
   - better-aligned chunking,
   - sufficient sketch order.
4. Same behaviors hold across nonlanguage scenarios.

## 7. Reproducibility Commands

```bash
source venv/bin/activate

python3 scripts/plot_mergeable_k_m_phase.py \
  --spike-count-support 1,2,3,4,5 \
  --spike-count-probs 0.10,0.20,0.25,0.25,0.20 \
  --target-ks 2,3,4,5 \
  --sketch-orders 2,3,4,5,6 \
  --budget-values 1,2,3,4,5,6,8,10 \
  --budget-target-k 5 \
  --budget-sketch-order 5 \
  --n-replicates 80 \
  --docs-per-replicate 120 \
  --seed 13

python3 scripts/plot_mergeable_chunk_quality_sweep.py \
  --target-k 5 \
  --sketch-order 5 \
  --chunker fixed \
  --selector top-proxy \
  --chunk-sizes 1,2,4,8,16 \
  --chunk-budgets 1,2,3,4,6,8 \
  --n-replicates 80 \
  --docs-per-replicate 120 \
  --seed 13

python3 scripts/plot_mergeable_nonlanguage_suite.py \
  --target-k 5 \
  --sketch-order 5 \
  --chunk-sizes 1,2,4,8,16 \
  --chunk-budgets 1,2,3,4,6,8 \
  --n-replicates 40 \
  --docs-per-replicate 100 \
  --seed 7

python3 scripts/plot_mergeable_nonlanguage_coverage.py \
  --target-k 5 \
  --sketch-order 5 \
  --chunk-sizes 1,2,4,8,16 \
  --chunk-budgets 1,2,3,4,6,8 \
  --ci-level 0.95 \
  --n-replicates 60 \
  --docs-per-replicate 120 \
  --seed 7

python3 scripts/plot_mergeable_complexity_ladder.py \
  --p-spike-doc 0.62 \
  --p-two-spikes-given-spike 0.45 \
  --p-multi-given-two-spikes 0.35 \
  --p-boundary-given-spike 0.35 \
  --generic-target-ks 2,3,4,5 \
  --n-replicates 120 \
  --docs-per-replicate 160 \
  --seed 0
```

## 8. Validation Status

- `pytest tests/tree/test_mergeable_ablation.py -q` currently passes (21 tests).
- Tests include deterministic checks for:
  - perfect-token-leaf recovery,
  - budget-induced failure/recovery,
  - coverage collapse under severe chunk-loss settings.
