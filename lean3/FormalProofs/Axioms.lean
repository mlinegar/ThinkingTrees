import FormalProofs.OPT.PreferenceBounds
import FormalProofs.OPT.AuditBounds

/-!
# Assumption Registry

This file documents all modeling assumptions and assumption structures used in the formalization.

## Summary

### OPT Module: Expected-Lipschitz Interfaces (No Lean axioms)

| # | Assumption | Location | Purpose |
|---|------------|----------|---------|
| 1 | `ExpectedGroupLossLipschitz` | FormalProbability/DSL/RUM | Expected loss over groups is Lipschitz |

The abstract interface is justified by the **Random Utility Model** (McFadden 1974).
Under continuous noise, ranking ties have measure zero, so the expected loss is Lipschitz
even though the pointwise ranking function is discontinuous. Where the codebase already has
stronger first-principles proofs, we prefer those and export them separately.

The assumption is instantiated for specific loss functions:
- `ExpectedGRPOLossLipschitz` - GRPO-PL (Plackett-Luce ranking loss)
- `ExpectedGRPORLLossLipschitz` - GRPO-RL (PPO-style clipped surrogate)

### DSL Module: Assumption Structures

| Structure | Location | Purpose |
|-----------|----------|---------|
| `OracleAccess` | DSL/CoreDefinitions | Expert labels = oracle labels |
| `MEstimationAxioms` | DSL/AsymptoticTheory | M-estimation asymptotics |
| `MEstimatorConsistencyAssumption` / `MEstimatorAsymptoticNormalAssumption` | DSL/AsymptoticTheory | Decomposed M-estimation assumptions |
| `CoverageFromAsymptoticNormal` (`CoverageAxioms` alias) | DSL/AsymptoticTheory | CI coverage transfer from asymptotic normality |
| `CalibrationRMSEBound` (`CalibrationAxioms` alias) | DSL/JudgeCalibration | Calibration RMSE representativeness bound |
| `EmpiricalBernsteinAxioms` | DSL/IPWTheory | Compatibility wrapper for self-normalized concentration (event-based form also available) |
| `HonestyContract` (`HonestyAxioms` alias) | DSL/Honesty | Constructive honest sample splitting contract |
| `AdaptiveSamplingAssumptions` (`AdaptiveSamplingAxioms` alias) | DSL/Honesty | Predictable adaptive sampling with exploration floor |

Concrete first-principles routes already available in this lane:
- `DSL/ConcreteCoverage.lean` proves one-dimensional coverage directly from cdf
  convergence to the standard normal law and an explicit event equivalence, and
  also proves multivariate coordinatewise coverage from weak convergence of the
  full studentized vector by projecting to coordinates with the continuous
  mapping theorem.
- `DSL/AsymptoticCore.lean` now exposes the generic constructive interfaces
  `CoverageEventWitness`, `CoordinateCoverageLimitWitness`, and
  `NormalCoverageConstruction`, separating the event identity, limit law, and
  asymptotic-normality-to-coverage construction layers.
- `DSL/ConcreteCoverage.lean` proves
  `CoordinateCoverageLimitWitness.asymptoticCoverage` and
  `NormalCoverageConstruction.asymptoticCoverage`, so a caller can derive
  coverage from first principles without appealing to the blanket
  `CoverageFromAsymptoticNormal` assumption.
- `DSL/AsymptoticTheory.lean` now threads that concrete route into the DSL
  estimator surface via
  `DSL_valid_coverage_coordStdNormal_from_assumptions` /
  `DSL_valid_coverage_coordStdNormal`, so the coordinatewise Wald lane no
  longer needs `CoverageFromAsymptoticNormal`; standard-normal coordinates are
  derived from the `NormalLimit` witness after diagonal studentization with
  only positive diagonal variance assumptions. The `*_symm` wrappers package
  the common symmetric `[-z, z]` critical-value case.
- `DSL/AsymptoticTheory.lean` also now exposes the implementation-facing
  plug-in covariance route via
  `DSL_valid_coverage_pluginStdNormal_from_assumptions` /
  `DSL_valid_coverage_pluginStdNormal` and the matching valid-inference
  theorems, so callers can reason directly about studentization by an estimated
  diagonal covariance `V̂ₙ` rather than a population-only standard error.
- `Econometrics/OLS/AsymptoticOLS.lean` packages that route for 1D Wald
  coverage via `asymptotic_ci_coverage_from_tstat_cdf_to_stdNormal`.
- `DSL/JudgeCalibration.lean` now contains held-out calibration discharge
  lemmas: population RMSE or true-bias confidence events imply
  `CalibrationRMSEBound`, and those events can be pushed directly into the PMF
  surrogate-gap bounds.
- `DSL/TreeIPW.lean` now contains stopped-time wrappers that lift scheduled
  fixed-horizon event families into anytime-valid audit bounds for arbitrary
  stopping rules.
- `DSL/RuntimeCertificates.lean` packages existing validity theorems as
  soundness statements for checked runtime artifacts, so implementations can
  emit a certificate object and reuse the established `computeDSLBound` /
  local-law theorem surface rather than restating it.

### Econometrics Module: Assumption Structures

| Structure | Location | Purpose |
|-----------|----------|---------|
| `OLSAsymptoticAxioms` | Econometrics/OLS/AsymptoticOLS | LLN/CLT/Slutsky/delta-method package for OLS asymptotics |
| `ScoreLLNAssumption` / `ScoreCLTAssumption` | Econometrics/OLS/AsymptoticOLS | Decomposed score-process assumptions |
| `OLSConsistencyAssumption` / `OLSAsymptoticNormalAssumption` | Econometrics/OLS/AsymptoticOLS | Decomposed OLS limit assumptions |

## Soundness

All assumptions and assumption structures are **modeling choices**, not gaps in the proof:
- Each has rigorous mathematical justification from the statistics/econometrics literature
- The OPT expected-Lipschitz interface follows from the Random Utility Model (McFadden 1974)
- The DSL assumptions follow from M-estimation theory (Newey & McFadden 1994)
- The formalization is SOUND under these assumptions

---

## Interface: ExpectedGroupLossLipschitz

**Statement**: Expected loss over groups is Lipschitz in oracle distance.

```lean
def ExpectedGroupLossLipschitz {Strings A Y : Type*} [PseudoMetricSpace Y] {k : ℕ}
    (loss : Strings → (Fin k → A) → ℝ)
    (fstar : Strings → Y) (g : PMF (Fin k → A)) (L : ℝ≥0)
    (x z : Strings) : Prop :=
  |∑' group, (g group).toReal * loss x group -
   ∑' group, (g group).toReal * loss z group| ≤
  L * dist (fstar x) (fstar z)
```

**Location**: `FormalProbability/DSL/RUM.lean` (re-exported in `OPT/PreferenceBounds.lean`)

**Mathematical Justification**:
Under the Random Utility Model (McFadden 1974), scores are:
```
s(x, a) = u(x, a) + ε_a
```
where ε_a is continuous noise (e.g., Gumbel for Plackett-Luce).

The ranking function is discontinuous at ties (score crossings), but:
1. Ties have **measure zero** under continuous noise
2. By dominated convergence, the expected loss is continuous
3. With Lipschitz policy components, the expectation inherits Lipschitz

**When Safe**: Safe as an abstract interface when the expected loss is known to inherit
Lipschitz control from a continuous-noise choice model. For the fixed-ranker
Plackett-Luce lane, the repo also contains a direct first-principles discharge theorem,
so downstream certificates do not need to assume this separately there.

**Instantiations**:
- `ExpectedGRPOLossLipschitz` - For GRPO-PL (Plackett-Luce ranking loss)
- `ExpectedGRPORLLossLipschitz` - For GRPO-RL (PPO-style with clipping, KL penalty)

---

# DSL Module: Assumption Structures

The DSL module uses **structures** rather than Lean `axiom` declarations to bundle
assumptions. This makes them explicit parameters to theorems, which is cleaner
for a formalization that aims to be modular.

## OracleAccess

**Location**: `DSL/CoreDefinitions.lean`

**Statement**: When a document is sampled (R=1), the expert label matches the oracle:
```
doc.sampled = true → doc.d_mis_true = some (oracle doc.content)
```

**Mathematical Justification**:
This is the **design assumption** of DSL (Design-based Supervised Learning).
Expert coders are assumed to correctly label the "missing" variable when they
code a document. The oracle function `Content → Missing` represents the true
labeling rule that experts implement.

This is analogous to the "no measurement error" assumption in survey sampling:
when you measure something, you measure it correctly. In ML terms, the training
labels are assumed to be correct for the sampled subset.

**When Safe**: When expert coders follow a consistent labeling protocol.

---

## MEstimationAxioms

**Location**: `DSL/AsymptoticTheory.lean`

**Contents**:
- `consistent`: M-estimators converge in probability to true parameters
- `asymptotic_normal`: Centered/scaled estimators converge to N(0, V)
- decomposed interfaces:
  - `MEstimatorConsistencyAssumption`
  - `MEstimatorAsymptoticNormalAssumption`
  - constructor: `mkMEstimationAxioms`

**Mathematical Justification**:
Standard M-estimation theory from econometrics (Newey & McFadden 1994,
"Large Sample Estimation and Hypothesis Testing", Handbook of Econometrics).

Proving these from primitives would require formalizing:
1. Uniform laws of large numbers for dependent data
2. Central limit theorems for M-estimators
3. Delta method for smooth functionals

These are well-established results in the statistics literature.

**When Safe**: Under standard regularity conditions (identification, smoothness,
bounded moments).

---

## CoverageFromAsymptoticNormal (`CoverageAxioms` alias)

**Location**: `DSL/AsymptoticTheory.lean`

**Status**: Compatibility interface. The repo now also has a generic
constructive replacement:
- `CoverageEventWitness`
- `CoordinateCoverageLimitWitness`
- `NormalCoverageConstruction`

These live in `DSL/AsymptoticCore.lean` / `DSL/ConcreteCoverage.lean` and are
threaded into the estimator-level surface by
`DSL_valid_coverage_from_construction_from_assumptions` /
`DSL_valid_coverage_from_construction`.

**Statement**: Confidence intervals constructed from asymptotically normal
estimators achieve nominal coverage asymptotically.

**Mathematical Justification**:
If √n(β̂ - β) →d N(0, V), then the interval β̂ ± z_{α/2} × SE(β̂) covers β
with probability approaching 1-α.

This is the standard justification for Wald-type confidence intervals.

**When Safe**: When the asymptotic approximation is accurate (typically n ≥ 30
for well-behaved data, larger for heavy tails or sparse data).

**Concrete alternative already formalized**: for one-dimensional Wald intervals,
the repo has a first-principles route in `DSL/ConcreteCoverage.lean`
that derives coverage directly from:
- one-dimensional cdf convergence to the standard normal law; and
- multivariate weak convergence by coordinate projection plus boundary-null
  interval events.

`Econometrics/OLS/AsymptoticOLS.lean` instantiates the 1D route for
t-statistics.

The main DSL theorem surface also has a concrete coordinatewise Wald route via
`DSL_valid_coverage_coordStdNormal_from_assumptions` and
`DSL_valid_coverage_coordStdNormal`; what remains abstract is the fully generic
compatibility alias for callers who do not provide a construction witness. The
generic constructive interface itself is now formalized: one can specify the
event identity, limiting coordinate laws, and calibration data explicitly via
`NormalCoverageConstruction`, then derive coverage without a separate axiom. In
the coordinatewise Wald route, the only extra normalization input is positivity
of the diagonal variances together with an explicit event equivalence for the
diagonally studentized statistic, not a separate coordinate-law assumption.

There is now also a plug-in diagonal covariance route via
`DSL_valid_coverage_pluginStdNormal_from_assumptions` and
`DSL_valid_coverage_pluginStdNormal`: if a diagonal covariance estimator
converges in probability to a positive-diagonal limit and the plug-in
studentized interval event is identified explicitly, the Wald coverage theorem
no longer needs population-only standard errors at the API boundary.

---

## EmpiricalBernsteinAxioms

**Location**: `DSL/IPWTheory.lean`

**Status**: Compatibility interface. The TreePO theorem path now also exposes
direct event-based concentration assumptions (`empiricalBernstein_bound_ennreal_of_event`
and event-parameterized wrappers in `DSL/TreeIPW.lean`), so core validity
results no longer require this structure.

**Statement**: A self-normalized (empirical Bernstein) concentration bound
for weighted (Hajek) estimators. It provides a finite-sample confidence radius
based on the observed weighted variance and effective sample size.

**Mathematical Justification**:
Empirical Bernstein and Freedman-style inequalities provide tighter bounds than
Hoeffding by adapting to observed variance. For design-based weighting, the
same style of inequality is typically proven for self-normalized sums or
weighted means under appropriate boundedness and regularity assumptions.

**When Safe**: When the sampling design and weighting satisfy the standard
self-normalized concentration conditions (bounded outcomes, bounded weights,
and valid tail control).

---

## HonestyContract (`HonestyAxioms` alias)

**Location**: `DSL/Honesty.lean`

**Statement**: The training procedure depends only on the training split and
the evaluation estimator depends only on the evaluation split, with an explicit
split function. This captures the *honesty* condition used in causal forests
and sample-splitting inference.

**Mathematical Justification**:
Honest sample splitting ensures that evaluation is performed on data not used
to fit the model, so finite-sample bounds can be applied as if the model were
fixed. This prevents adaptive overfitting from invalidating inference.

**When Safe**: When the split is enforced by design and the evaluation step is
computed only on held-out data.

---

## CalibrationRMSEBound (`CalibrationAxioms` alias)

**Location**: `DSL/JudgeCalibration.lean`

**Status**: Compatibility interface. TreePO-level calibrated bounds now accept
the RMSE envelope directly (`h_rmse_upper`) with `*_from_axioms` wrappers for
backward compatibility.

**Concrete alternative already formalized**: `DSL/JudgeCalibration.lean`
contains direct discharge lemmas from held-out evidence:
- `CalibrationRMSEBound_of_abs_trueBias_le`
- `CalibrationRMSEBound_of_mem_biasConfidenceInterval`
- `calibrationRMSEBound_event_of_populationRMSE_event`
- `calibrationRMSEBound_event_of_biasConfidence_event`

These can then be fed directly into
`surrogate_bound_pmf_calibration2_event_of_rmse_event` and
`surrogate_bound_pmf_calibration2_event_of_biasConfidence_event`, so a concrete
implementation can certify calibration from stored held-out summaries rather
than postulating representativeness globally.

**Statement**: The population RMSE of judge error is bounded by the calibration
estimate:
```
sqrt( E[(judge - oracle)^2] ) ≤ absbiasUpperBound + judgeStd
```

**Mathematical Justification**:
This is a representativeness assumption for the calibration set: it asserts
that calibration samples reflect the population error distribution.

**When Safe**: When the calibration set is sampled from the same distribution
as the evaluation population, with sufficient size for stable error estimates.

---

## OLSAsymptoticAxioms

**Location**: `Econometrics/OLS/AsymptoticOLS.lean`

**Contents**:
- LLN for the OLS score (sample `x_i ε_i` averages)
- Multivariate CLT for the score
- Slutsky-based asymptotic normality of OLS
- Homoskedastic simplification to σ² Q⁻¹
- t-statistics normality and delta method
- decomposed interfaces:
  - `ScoreLLNAssumption`, `ScoreCLTAssumption`
  - `OLSConsistencyAssumption`, `OLSAsymptoticNormalAssumption`
  - `OLSAsymptoticNormalHomoskedasticAssumption`
  - `TStatNormalAssumption`, `DeltaMethodAssumption`
  - constructor: `mkOLSAsymptoticAxioms`

**Mathematical justification**:
These are standard large-sample results in econometrics (Wooldridge, Ch. 5).
They rely on i.i.d. sampling, finite moments, and identification so that
LLN/CLT and continuous mapping theorems apply.

**When Safe**: Under weak exogeneity, finite moments, and identification
with sufficiently large samples.

-/

namespace Axioms

/-! ## Re-exported Axioms with Documentation Aliases -/

/-- Unified axiom: Expected loss over groups is Lipschitz in oracle distance.

This is the main abstract expected-Lipschitz interface for preference learning
bounds. It is justified by the Random Utility Model (McFadden 1974). -/
abbrev expected_group_loss_lipschitz := @ExpectedGroupLossLipschitz

/-- GRPO Plackett-Luce expected loss is Lipschitz in oracle distance.

Instantiation of `ExpectedGroupLossLipschitz` for GRPO-PL. -/
abbrev grpo_pl_expected_lipschitz := @ExpectedGRPOLossLipschitz

/-- First-principles discharge of the GRPO-PL expected-Lipschitz interface for
the fixed-ranker Plackett-Luce path. -/
abbrev grpo_pl_expected_lipschitz_plackett_luce_fixed_ranker :=
  @ExpectedGRPOLossLipschitz_plackettLuce_fixed_ranker_all

/-- GRPO-RL (DeepSeek-R1 style) expected loss is Lipschitz in oracle distance.

Instantiation of `ExpectedGroupLossLipschitz` for GRPO-RL. -/
abbrev grpo_rl_expected_lipschitz := @ExpectedGRPORLLossLipschitz

/-- Finite-support first-principles discharge of the GRPO-RL expected-loss
Lipschitz interface from a primitive pointwise bound. -/
abbrev grpo_rl_expected_lipschitz_of_pointwise_finite :=
  @ExpectedGRPORLLossLipschitz_of_pointwise_finite

end Axioms
