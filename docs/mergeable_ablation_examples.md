# Mergeable Ablation Examples

This note provides a minimal numeric setting for repeated tree aggregation ablations.

Goal: show where methods fail when we repeatedly merge chunk summaries.

## Core setup

- Token-level truth is known exactly.
- Default objective is non-additive: `spike-exists` (`1` iff any token exceeds threshold).
- We vary three components:
  - chunker policy (`fixed`, `adaptive-aligned`, `adaptive-misspecified`)
  - chunk selection under budget (`top-proxy`, `bottom-proxy`, etc.)
  - tree aggregator (`merge-safe-max`, `naive-majority`, `naive-mean-of-means`)

Implemented in `src/tree/mergeable_ablation.py`.

## Failure modes covered

1. Naive repeated aggregation failure:
   - `naive-majority` and `naive-mean-of-means` can disagree with true non-additive objective.
2. Right merge rule, wrong chunker:
   - `merge-safe-max` is correct for `spike-exists`, but misspecified adaptive chunking + bad selection can drop spike-containing chunks.
3. Order sensitivity:
   - `naive-mean-of-means` can produce different roots for left-to-right vs right-to-left merge trees.

## Run

```bash
source venv/bin/activate
python3 scripts/run_mergeable_ablation_simulation.py \
  --n-docs 240 \
  --n-tokens 32 \
  --seed 0 \
  --show-worked-examples
```

JSON output:

```bash
source venv/bin/activate
python3 scripts/run_mergeable_ablation_simulation.py --json --show-worked-examples
```

## Parameter Recovery

We can also specify a known DGP parameter and test recovery/bias directly.

Here the target parameter is:

- `theta = P(doc has spike)` from a known spike-mixture distribution.

Run:

```bash
source venv/bin/activate
python3 scripts/run_mergeable_param_recovery.py \
  --p-spike-doc 0.55 \
  --n-replicates 200 \
  --docs-per-replicate 200 \
  --seed 0
```

The table reports:

- `mean_est`: average recovered parameter
- `mean_bias`: bias relative to known `theta`
- `mean_abs_bias`
- `sample_target_bias`: method bias relative to finite-sample truth in each replicate
- `std_est`, `rmse`

### Two-Parameter Recovery (build-up to full model vs ablations)

Recover jointly:

- `p_spike = P(doc has >=1 spike)`
- `p_two|spike = P(doc has >=2 spikes | doc has >=1 spike)`

Run:

```bash
source venv/bin/activate
python3 scripts/run_mergeable_param_recovery.py \
  --two-param \
  --p-spike-doc 0.55 \
  --p-two-spikes-given-spike 0.20 \
  --n-replicates 200 \
  --docs-per-replicate 200 \
  --seed 0
```

Method ladder in this run:

1. `one_pass_oracle`: single-chunk oracle baseline (full document in one pass).
2. `full_model_aligned`: adaptive aligned chunking + proxy-top selection + merge-safe operators.
3. `naive_*` and `right_rule_wrong_chunker`: ablations that should show bias.

This directly supports “only full model (or one-pass oracle) gets it right” cases.

### Three-Parameter Recovery (adds boundary-conditioned target)

Recover jointly:

- `p_spike = P(doc has >=1 spike)`
- `p_two|spike = P(doc has >=2 spikes | doc has >=1 spike)`
- `p_boundary|spike = P(doc has boundary spike | doc has >=1 spike)`

Current toy DGP uses disjoint spike categories (`two-spike`, `boundary-single-spike`, `interior-single-spike`), so choose
`p_two|spike + p_boundary|spike <= 1`.

Run:

```bash
source venv/bin/activate
python3 scripts/run_mergeable_param_recovery.py \
  --three-param \
  --p-spike-doc 0.55 \
  --p-two-spikes-given-spike 0.20 \
  --p-boundary-given-spike 0.50 \
  --p-multi-given-two-spikes 0.40 \
  --boundary-span-tokens 4 \
  --n-replicates 200 \
  --docs-per-replicate 200 \
  --seed 0
```

This adds one key ablation:

- `full_model_missing_boundary_stat`: same chunker/selection as full model, but no merge-safe boundary sufficient statistic.
  It reuses the generic spike statistic for the boundary target, so the boundary parameter is not identifiable.

Interpretation goal:

- `one_pass_oracle` and `full_model_aligned` should track all three targets.
- `full_model_missing_boundary_stat` should fail specifically on `p_boundary|spike`.
- naive aggregators and wrong chunker should fail on multiple targets.

### Four-Parameter Recovery (adds 3+ spikes target)

Recover jointly:

- `p_spike = P(doc has >=1 spike)`
- `p_two|spike = P(doc has >=2 spikes | doc has >=1 spike)`
- `p_three+|spike = P(doc has >=3 spikes | doc has >=1 spike)`
- `p_boundary|spike = P(doc has boundary spike | doc has >=1 spike)`

In this toy DGP:

- `p_three+|spike = p_two|spike * p_multi|two`
- `p_multi|two` is controlled by `--p-multi-given-two-spikes`.

Run:

```bash
source venv/bin/activate
python3 scripts/run_mergeable_param_recovery.py \
  --four-param \
  --p-spike-doc 0.62 \
  --p-two-spikes-given-spike 0.28 \
  --p-multi-given-two-spikes 0.75 \
  --p-boundary-given-spike 0.42 \
  --n-replicates 200 \
  --docs-per-replicate 200 \
  --seed 0
```

Key new ablation:

- `full_model_missing_three_stat`: same chunking/selection, but no merge-safe third-order statistic.

Representative 4-parameter output (non-corner multi-spike setting):

- `full_model_aligned`: `p_three|spike_hat=0.1543` vs true `0.1575` (bias `-0.0032`)
- `full_model_missing_three_stat`: `p_three|spike_hat=0.4496` (bias `+0.2921`)

## Key outputs

- `mean_abs_error`: average absolute error in score.
- `label_error_rate`: 0/1 decision mismatch rate.
- `order_spread_mean`: average spread across merge orders.
- `order_flip_rate`: rate where merge-order changes the binary decision.

These metrics make ablations easy to compare side-by-side.

## Complexity Ladder

A practical build-up path from simple to complex:

1. One parameter:
   recover only `P(spike)` with `scripts/run_mergeable_param_recovery.py`.
2. Two parameters:
   add `p_two|spike` to verify second-order sufficient statistic (`merge-safe-second-max`).
3. Three parameters:
   add `p_boundary|spike` and stress target-definition mismatch via `--align-boundary-span`.
4. Four parameters:
   add `p_three+|spike` and verify need for third-order sufficient statistic (`merge-safe-third-max`).
5. Scenario generalization:
   run `scripts/run_mergeable_generalization_sweep.py` and compare frozen vs retuned runs.

This ladder keeps each added failure mode isolated, so when estimates move you can attribute why.

Run the whole ladder in one command:

```bash
source venv/bin/activate
python3 scripts/run_mergeable_complexity_ladder.py \
  --p-spike-doc 0.62 \
  --p-two-spikes-given-spike 0.45 \
  --p-multi-given-two-spikes 0.35 \
  --p-boundary-given-spike 0.35 \
  --n-replicates 120 \
  --docs-per-replicate 160 \
  --seed 0
```

Generate a compact ladder figure + failure-signature heatmap:

```bash
source venv/bin/activate
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

Outputs:

- `outputs/mergeable_complexity_ladder.png`
- `outputs/mergeable_complexity_ladder_summary.json`

## Generic k (and insufficiency)

Run explicit generic-k targets:

```bash
source venv/bin/activate
python3 scripts/run_mergeable_k_recovery.py \
  --spike-count-support 1,2,3,4,5 \
  --spike-count-probs 0.10,0.20,0.25,0.25,0.20 \
  --target-ks 2,3,4,5 \
  --n-replicates 120 \
  --docs-per-replicate 160 \
  --seed 0 \
  --show-counterexample \
  --counterexample-m 3
```

Sweep both sides (`m<k`, `m=k`, `m>k`) directly:

```bash
source venv/bin/activate
python3 scripts/run_mergeable_k_m_sweep.py \
  --spike-count-support 1,2,3,4,5 \
  --spike-count-probs 0.10,0.20,0.25,0.25,0.20 \
  --target-ks 2,3,4,5 \
  --sketch-orders 2,3,4,5,6 \
  --n-replicates 120 \
  --docs-per-replicate 160 \
  --seed 0
```

Plot phase behavior with naive baselines on the same chart:

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
```

Outputs:

- `outputs/mergeable_k_m_phase.png`
- `outputs/mergeable_k_m_phase_summary.json`

The figure now has three panels:

1. Mean abs bias vs `delta = m-k` (one-pass/full-model) with naive baselines.
2. Regime penalties (`unsupported-exact`, `oversup-exact`) by target `k`.
3. Budget sweep for fixed `(k,m)`, with 95% CI error bars on signed bias and dashed abs-bias curves.

CI note:

- For each point, we approximate a 95% CI for mean signed bias using
  `rmse^2 = var + bias^2`, `se = sqrt(var / n_replicates)`, and `bias ± 1.96*se`.
- This directly answers whether bias is statistically distinguishable from zero.

Representative values from the command above:

- Naive baselines (same plot): majority `0.4375`, mean-of-means `0.5391` mean abs bias.
- Exact-supported (`m=k`) merge-safe methods: about `0.036` to `0.038` mean abs bias.
- Unsupported (`m<k`) penalty grows quickly: for `k=5`, unsupported-minus-exact is about `+0.45`.
- Over-supported (`m>k`) is close to exact: penalty near zero (about `-0.005` to `+0.003`).
- Budget effect (example with `k=5,m=5`): full-model abs-bias drops from about `0.20` at budget `1`
  to about `0.04` by budgets `5-10` (small non-monotone noise after that).

## Chunk Quality / Leaf Granularity Lever

To isolate chunk quality itself, sweep leaf size and budget while keeping the merge-safe target fixed:

```bash
source venv/bin/activate
python3 scripts/plot_mergeable_chunk_quality_sweep.py \
  --spike-count-support 1,2,3,4,5 \
  --spike-count-probs 0.10,0.20,0.25,0.25,0.20 \
  --target-k 5 \
  --sketch-order 5 \
  --chunker fixed \
  --selector top-proxy \
  --chunk-sizes 1,2,4,8,16 \
  --chunk-budgets 1,2,3,4,6,8 \
  --n-replicates 80 \
  --docs-per-replicate 120 \
  --seed 13
```

Outputs:

- `outputs/mergeable_chunk_quality_sweep.png`
- `outputs/mergeable_chunk_quality_sweep_summary.json`

What this adds beyond bias:

- `mean_target_capture_rate`: among true `count>=k` docs, how often retained chunks still contain `>=k` spikes.
- `mean_spike_token_recall`: retained spike tokens / true spike tokens.
- `mean_spike_token_isolation`: retained spike tokens that appear in singleton leaves / true spike tokens.

Interpretation pattern:

- Perfect alignment reference (`perfect_token_leaves_all`) should have near-1 recall/isolation and near-0 bias.
- If leaf quality is good but budget is too small, target capture can still fail (selection bottleneck).
- As budget grows, the same leaf quality should recover target capture and reduce bias toward zero.

Representative run (`k=5`, `m=5`, fixed chunker, top-proxy selector):

- `perfect_token_leaves_all`: `target_capture=1.00`, `spike_recall=1.00`, `spike_isolation=1.00`.
- Fine leaves (`size=1`) with low budget (`b<=4`) can still fail for this target:
  `target_capture=0.00`, `abs_bias≈0.20`.
- Same fine leaves with enough budget (`b=6`): `target_capture=1.00`, `abs_bias≈0.036`.
- Coarse leaves (`size=16`) can have low isolation (`0.00`) but still good capture once budget is enough (`b>=2`),
  showing that capture and isolation are distinct diagnostics.

## Non-Language Scenario Suite

The same mergeable-sketch logic is not tied to language tokens. We can map
"token positions" to generic bins in any ordered signal:

- ICU alarms: bins are time slices in a patient trace.
- Network intrusion: bins are packet/window aggregates in a flow.
- Manufacturing defects: bins are sensor windows along a production line.
- ECG arrhythmia: bins are beat windows in a rhythm segment.

Run:

```bash
source venv/bin/activate
python3 scripts/plot_mergeable_nonlanguage_suite.py \
  --target-k 5 \
  --sketch-order 5 \
  --chunk-sizes 1,2,4,8,16 \
  --chunk-budgets 1,2,3,4,6,8 \
  --n-replicates 60 \
  --docs-per-replicate 120 \
  --seed 0
```

Outputs:

- `outputs/mergeable_nonlanguage_suite.png`
- `outputs/mergeable_nonlanguage_suite_summary.json`

Plot interpretation:

- Solid lines: best abs-bias at each budget (optimized over candidate leaf sizes).
- Dashed lines: best target-capture rate at each budget.
- In every scenario, aligned chunking should improve with budget and approach one-pass/perfect references.
- Misspecified chunking remains a high-bias ablation, showing this is not language-specific but a structural chunking/selection issue.

Representative outputs (`n_replicates=40`, `docs_per_replicate=100`, seed `7`):

- `network_intrusion_bursts`:
  aligned best abs-bias drops from `~0.25` at budget `1` to `~0.043` at budget `8`,
  while misspecified remains worse (`~0.094` at budget `8`).
- `icu_alarm_stream`:
  aligned improves from `~0.094` to `~0.028` as budget grows.
- `manufacturing_defect_line`:
  low-noise setting stays low-bias (`~0.029 -> ~0.012`), with misspecified still worse at low budget.

Note on references:

- `perfect_token_leaves_all` can still have nonzero `mean_abs_bias` because this metric includes finite-sample
  estimation noise (replicate-to-replicate fluctuation), not only structural chunking bias.

Coverage-focused companion plot:

```bash
source venv/bin/activate
python3 scripts/plot_mergeable_nonlanguage_coverage.py \
  --target-k 5 \
  --sketch-order 5 \
  --chunk-sizes 1,2,4,8,16 \
  --chunk-budgets 1,2,3,4,6,8 \
  --ci-level 0.95 \
  --n-replicates 120 \
  --docs-per-replicate 160 \
  --seed 0
```

Outputs:

- `outputs/mergeable_nonlanguage_coverage.png`
- `outputs/mergeable_nonlanguage_coverage_summary.json`

Interpretation:

- Curves near the nominal line (`0.95`) indicate calibrated uncertainty for that regime.
- Low-budget undercoverage flags structural misspecification/chunk-loss, not just variance.
- Persistent overcoverage with wide intervals can indicate overly conservative intervals
  (often from tiny effective conditional sample sizes).
- As budget and chunk alignment improve, coverage should move toward nominal.

When is a summary sketch insufficient?

- For target `P(count>=k | spike)`, a top-`m` merge-safe sketch is sufficient only if `m >= k`.
- If `m < k`, information is missing: two documents can share the same top-`m` sketch but differ on `count>=k`.
- The counterexample printed by `--show-counterexample` demonstrates this concretely (same sketch signature, different truth label).

Representative generic-k result (`support=[1,2,3,4,5]`, `probs=[0.10,0.20,0.25,0.25,0.20]`):

- For `k=3` with sketch order `m=3` (`m>=k`), limited-sketch and full-model are close
  (`abs_bias ≈ 0.04`).
- For `k=5` with sketch order `m=3` (`m<k`), limited-sketch becomes strongly biased
  (`abs_bias ≈ 0.51`), while full-model (`m=5`) stays low (`abs_bias ≈ 0.04`).
- In `m x k` sweeps, over-supported (`m>k`) behaves like exact-supported (`m=k`) up to sampling noise,
  while unsupported (`m<k`) shows the sharp bias jump.

## Other comparisons worth tracking (simple setting)

1. One-pass vs full-model gap:
   compare `one_pass_m*` against `full_model_m*` at matched `m,k`; this isolates chunking/selection error from sketch-capacity error.
2. Unsupported inflation ratio:
   report `(unsupported abs-bias)/(exact abs-bias)` by `k` to quantify how fast failure grows as target complexity rises.
3. Bias vs variance split:
   track `abs_bias`, `bias`, and `rmse` together to separate structural misspecification (bias) from finite-sample noise.
4. Coverage under confidence intervals:
   now that CI code is wired for bias, estimate empirical coverage in each regime (`m<k`, `m=k`, `m>k`) to connect theory and practice.

Example phase summary (`target_ks=2,3,4,5`, `sketch_orders=2,3,4,5,6`):

- one-pass:
  - `k=5`: unsupported mean abs bias `0.4856`, exact `0.0351`, over-supported `0.0324`
  - `k=4`: unsupported `0.3527`, exact `0.0423`, over-supported `0.0428`
- full-model:
  - `k=5`: unsupported `0.4785`, exact `0.0380`, over-supported `0.0335`
  - `k=4`: unsupported `0.3460`, exact `0.0448`, over-supported `0.0469`

Interpretation:

- `m<k` is the true insufficiency regime (information-theoretic miss, not just noisy estimation).
- `m>=k` is sufficient for this target family; `m>k` does not buy much extra beyond variance-level changes.

## Generalization Stress Sweep

To test robustness (not just one DGP), run the built-in variable-length and adversarial sweep:

```bash
source venv/bin/activate
python3 scripts/run_mergeable_generalization_sweep.py \
  --n-replicates 120 \
  --docs-per-replicate 160 \
  --seed 0 \
  --csv outputs/mergeable_generalization_sweep.csv
```

Retuned variant (boundary statistic matched to each scenario’s target definition):

```bash
source venv/bin/activate
python3 scripts/run_mergeable_generalization_sweep.py \
  --n-replicates 120 \
  --docs-per-replicate 160 \
  --seed 0 \
  --align-boundary-span
```

Included scenarios:

- `baseline_balanced_fixed`: fixed-length calibration baseline.
- `variable_length_balanced`: same targets but wide document-length variation.
- `boundary_adversarial_concentrated`: short-doc-heavy concentrated DGP with high boundary mass and noisy proxy.
- `hard_noncorner_adversarial`: adversarial variable-length + noisy proxy but with non-corner target rates
  (`p_spike=0.62`, `p_two|spike=0.28`, `p_boundary|spike=0.42`) to avoid saturation artifacts.
- `multi_spike_noncorner`: non-corner, multi-spike-heavy setting with `p_two|spike=0.65` and
  explicit 3+ spike documents (`p_multi|two=0.75`).
- `long_tail_interior_shift`: long-doc-heavy shift with mostly interior spikes.

Reported for each `(scenario, method)`:

- per-parameter absolute bias for `p_spike`, `p_two|spike`, `p_boundary|spike`
- aggregate mean absolute bias (mean of those three)
- generalization gap vs baseline for the same method

Representative results from:

```bash
python3 scripts/run_mergeable_generalization_sweep.py --n-replicates 120 --docs-per-replicate 160 --seed 0
python3 scripts/run_mergeable_generalization_sweep.py --n-replicates 120 --docs-per-replicate 160 --seed 0 --align-boundary-span
```

- `hard_noncorner_adversarial` (frozen): `full_model_aligned=0.0915` vs `naive_majority=0.5630`, `right_rule_wrong_chunker=0.2235`.
- `hard_noncorner_adversarial` (retuned boundary span): `full_model_aligned=0.0466`, `one_pass_oracle=0.0475`, while naive methods remain `>0.56`.
- `multi_spike_noncorner` (frozen): `full_model_aligned=0.1331`, `one_pass_oracle=0.1303`,
  vs `naive_majority=0.4897`, `naive_mean_of_means=0.5839`.
- `multi_spike_noncorner` (retuned boundary span): `full_model_aligned=0.0697`,
  `one_pass_oracle=0.0658`, `right_rule_wrong_chunker=0.2272`.
- Method means across all 6 scenarios:
  - frozen: `full_model_aligned=0.0741`, `one_pass_oracle=0.0746`, `naive_majority=0.5260`
  - retuned: `full_model_aligned=0.0450`, `one_pass_oracle=0.0459`, `naive_majority=0.5260`

### What this does and does not capture

What it captures:

- repeated-merge failure modes (non-merge-safe operators),
- adaptive chunking/selection failures under budget,
- robustness under DGP shift (length distribution, concentration, boundary mass, proxy noise).

What it does not capture:

- semantic summarization errors from real LLMs,
- rich cross-document structure,
- learned adaptive chunk policies fit on real corpora.

So this suite is a necessary stress harness for the aggregation math, not a full end-to-end realism claim.
