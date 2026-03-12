# Mergeable Aggregation Under Adaptive Chunking: Why It Matters and What Works

## Abstract

Adaptive chunking is a practical way to reduce computation and labeling cost in long-sequence systems, but it introduces a core risk: if chunk boundaries and chunk selection are wrong, repeated aggregation can become biased for non-additive targets. We evaluate a merge-safe sketching approach for targets of the form `P(count >= k | spike)` under controlled simulation. The results show a clean pattern: when sketch order is sufficient (`m >= k`) and chunk retention is adequate, bias is low and uncertainty is well calibrated; when sketch order is insufficient (`m < k`) or chunking/selection is misspecified, failures are large and predictable. These effects persist beyond language-like settings, including ICU alarm streams, network intrusion bursts, manufacturing defect traces, and ECG rhythm windows.

## 1. Why This Is Important

Many real pipelines do repeated aggregation over partitioned data (documents, time windows, sensor traces, network flows). The operational gains are clear, but correctness can break in subtle ways:

1. Non-additive targets are not preserved by naive averaging or majority voting.
2. Adaptive chunking can drop critical local events if selection is budget-limited.
3. Confidence intervals can look precise while being miscalibrated in low-support regimes.

A practical method must therefore satisfy three constraints at once:

1. merge-safe representation for the target,
2. sufficient sketch capacity for the target order,
3. chunking/selection quality high enough to retain relevant evidence.

## 2. Setup

We use known synthetic DGPs so true targets are available exactly. The main family is:

- `theta_k = P(count >= k | spike)`, where `count` is number of threshold-exceeding events.

Pipeline per document:

1. chunk sequence (fixed or adaptive),
2. keep only a budgeted subset of chunks,
3. aggregate kept chunks with a merge-safe top-`m` sketch,
4. estimate `theta_k`.

Key theoretical axis:

- `m < k`: information-limited (insufficient sketch).
- `m = k`: sufficient.
- `m > k`: over-supported (typically similar to `m = k`).

## 3. Simulation Program

We run four complementary families:

1. **`m` vs `k` phase study** (`scripts/plot_mergeable_k_m_phase.py`)
   - isolates sketch sufficiency and compares against naive baselines.
2. **Chunk-quality study** (`scripts/plot_mergeable_chunk_quality_sweep.py`)
   - sweeps leaf granularity and budget; tracks both bias and capture diagnostics.
3. **Nonlanguage scenario suite** (`scripts/plot_mergeable_nonlanguage_suite.py`)
   - applies same method to ICU, intrusion, manufacturing, ECG analogs.
4. **Coverage study** (`scripts/plot_mergeable_nonlanguage_coverage.py`)
   - estimates empirical CI coverage via per-replicate Wilson intervals.

## 4. Main Results

### 4.1 Sufficiency dominates performance

From `outputs/mergeable_k_m_phase_summary.json`:

- Naive baselines are high bias:
  - majority `0.4375`, mean-of-means `0.5391` (mean abs bias).
- Merge-safe `m = k` is low bias (`~0.036` to `~0.038`).
- Unsupported penalty is large:
  - for `k = 5`, unsupported-minus-exact is about `+0.45`.
- Over-supported (`m > k`) is near exact-supported.

Interpretation: the primary failure boundary is sketch insufficiency, not random variation.

### 4.2 Chunk quality and budget are distinct levers

From `outputs/mergeable_chunk_quality_sweep_summary.json` (`k=5, m=5`):

- Perfect token leaves: capture/recall/isolation all `1.0`.
- Fine leaves with low budget can still fail:
  - `size=1`, `b<=4`: `target_capture=0.0`, `abs_bias≈0.20`.
- Same fine leaves with higher budget recover:
  - `size=1`, `b=6`: `target_capture=1.0`, `abs_bias≈0.036`.

Interpretation: good local leaves do not guarantee correctness unless enough of them are retained.

### 4.3 Not language-specific

From `outputs/mergeable_nonlanguage_suite_summary.json`:

- Network intrusion: aligned abs-bias `~0.25 -> ~0.043` (budget `1 -> 8`);
  misspecified remains worse (`~0.094` at budget `8`).
- ICU alarms: aligned `~0.094 -> ~0.028`.
- Manufacturing: aligned `~0.029 -> ~0.012`.

Interpretation: the mechanism is structural (ordered-signal chunking + repeated aggregation), not tied to text semantics.

### 4.4 Uncertainty can be repaired with better support

From `outputs/mergeable_nonlanguage_coverage_summary.json` (`95%` nominal):

- Aligned coverage improves sharply with budget:
  - ICU `0.050 -> 0.950`
  - Intrusion `0.000 -> 0.983`
  - ECG `0.317 -> 0.967`

Interpretation: undercoverage at low budget is a structural misspecification signal; as chunk support improves, coverage returns toward nominal.

### 4.5 Complexity ladder: what is necessary, and what fails when missing

From `outputs/mergeable_complexity_ladder_summary.json` (stages add one requirement at a time):

- Positive controls remain stable as complexity grows:
  - one-pass oracle aggregate abs-bias: `0.027 -> 0.035 -> 0.037 -> 0.026` (stages 1,2,4,5)
  - full model aggregate abs-bias: `0.031 -> 0.039 -> 0.034 -> 0.027`
- Right rule but wrong chunker fails even with merge-safe operators:
  - aggregate abs-bias: `0.609` (stage 1), `0.471` (stage 2), `0.390` (stage 4)
- Naive aggregators fail harder as target order increases:
  - naive majority aggregate abs-bias: `0.426` (stage 2), `0.585` (stage 4), `0.821` (stage 5)

Target-specific failure signatures (stage 4) isolate what is missing:

- Missing boundary statistic:
  - `full_model_missing_boundary_stat` has low bias on unrelated targets
    (`P(spike)=0.033`, `P(>=2|spike)=0.043`, `P(>=3|spike)=0.029`)
    but fails on boundary target (`P(boundary|spike)=0.650`).
- Missing third-order statistic:
  - `full_model_missing_three_stat` is low on other targets
    (`P(spike)=0.032`, `P(>=2|spike)=0.042`, `P(boundary|spike)=0.042`)
    but fails on `P(>=3|spike)` (`0.300`).

Generic-`k` stage shows sketch sufficiency (`m>=k`) as a hard requirement:

- `full_model_limited_sketch` supports `k=2,3` and stays low-bias
  (`0.037`, `0.026`),
- then loses support for `k=4,5` and bias jumps
  (`0.081`, `0.128`).

Interpretation: each ablation fails exactly the target component it cannot represent
or capture. This is the key necessity result:

1. need sufficient sketch order for each queried `k`,
2. need target-specific sufficient statistics (boundary and third-order),
3. need chunking/selection that retains evidence.

## 5. What This Establishes

The method "works" in the strongest practical sense:

1. It succeeds in the regimes where theory says it should (`m >= k`, adequate retention).
2. It fails in interpretable regimes where theory says it must (`m < k`, severe chunk loss).
3. It generalizes across nonlanguage domains with the same causal pattern.
4. It supports uncertainty diagnostics that distinguish variance from structural failure.

## 6. Practical Guidance

1. Match sketch order to target order (`m >= k`).
2. Monitor capture diagnostics (`target_capture`, spike recall) in addition to bias.
3. Treat coverage collapse as an operational alert for chunk/selection failure.
4. Increase budget before increasing model complexity when coverage is poor in high-spike regimes.

## 7. Limits and Next Steps

Current studies use synthetic-but-controlled DGPs with known truth; this is intentional for identifiability and ablation clarity. Next steps are:

1. attach the same diagnostics to real logged data,
2. calibrate CI behavior under realistic dependence structures,
3. extend targets beyond `count >= k` while preserving merge-safe sufficiency checks.

## 8. Reproducibility

All commands and artifacts are listed in:

- `docs/mergeable_method_validation_report.md`
- `docs/mergeable_ablation_examples.md`
