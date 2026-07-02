import FormalProofs.OPT.PreferenceBounds
import FormalProofs.OPT.AuditBounds
import FormalProofs.OPT.InfluenceWeightedLocalLaws
import FormalProofs.DSL.Honesty
import FormalProofs.DSL.DocumentStructure
import FormalProofs.DSL.UnifiedLearningCertificate

/-!
# Assumption Registry

This file documents all modeling assumptions and assumption structures used in the formalization.

## Summary

### OPT Module: Expected-Lipschitz and Influence-Weighted Interfaces (No Lean axioms)

| # | Assumption | Location | Purpose |
|---|------------|----------|---------|
| 1 | `ExpectedGroupLossLipschitz` | FormalProbability/DSL/RUM | Expected loss over groups is Lipschitz |
| 2 | `InfluenceWeightedAuditOverlap` | OPT/InfluenceWeightedLocalLaws | Consequential local-law rows have enough logged audit probability |
| 3 | `RootErrorControlledByInfluenceMass` | OPT/InfluenceWeightedLocalLaws | Root error is bounded by influence-weighted local-law residual mass |

The abstract interface is justified by the **Random Utility Model** (McFadden 1974).
Under continuous noise, ranking ties have measure zero, so the expected loss is Lipschitz
even though the pointwise ranking function is discontinuous. Where the codebase already has
stronger first-principles proofs, we prefer those and export them separately.

The assumption is instantiated for specific loss functions:
- `ExpectedGRPOLossLipschitz` - GRPO-PL (Plackett-Luce ranking loss)
- `ExpectedGRPORLLossLipschitz` - GRPO-RL (PPO-style clipped surrogate)

The influence-weighted interfaces make the paper's finite-sample audit-overlap
condition explicit. They are theorem parameters, not global axioms: callers
must provide (i) a propagation inequality from local residual rows to root error
and (ii) bounded influence-to-propensity ratios for the audit design.

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
| `TopLevelIID` / `TopLevelExchangeable` | DSL/Honesty | IID/exchangeability applies to paired top-level observations `(X_i, Y_i*)` |
| `TopLevelSplit` / `ParentOf` | DSL/Honesty | Top-level unit split plus derived-row parent map |
| `DerivedRowHonestyContract` | DSL/Honesty | Derived rows inherit train/eval roles from top-level units |
| `KFoldHonestTraining` / `KFoldHonestEvaluation` | DSL/Honesty | Cross-fitted artifact and evaluation isolation by top-level fold |
| `UnifiedLearningHonesty` | DSL/Honesty | Three-layer chunker/g/oracle honesty plus frozen-artifact evaluation |
| `ChunkerObjectiveTerms` | DSL/Honesty | Instrumental chunker objective: loss + law mass + radius + cost + boundary regularization |
| `Span` / `AdmissiblePartition` | DSL/DocumentStructure | Finite ordered document spans and non-overlapping covering chunk partitions |
| `RunManifestContract` | DSL/DocumentStructure | Parent IDs, row IDs, artifact lineage, logged propensities, and influence weights |
| `UnifiedLearningErrorCertificate` | DSL/UnifiedLearningCertificate | Final reported estimate plus local-law, calibration, estimation, and clipping radii |
| `UnifiedLearningPaperAssumptions` | DSL/UnifiedLearningCertificate | Bundled top-level sampling, honesty, chunking, and manifest contract for the final theorem |
| `UnifiedLearningComponentEvidence` | DSL/UnifiedLearningCertificate | Component-radius provenance records for high-probability final certificates |
| `unified_learning_abs_gap_le_totalBound` | DSL/UnifiedLearningCertificate | Deterministic final-gap certificate for the unified learning procedure |
| `unified_learning_abs_gap_le_totalBound_from_influence` | DSL/UnifiedLearningCertificate | Same certificate with the local-law term supplied by the influence-weighted audit certificate |
| `unified_learning_final_paper_certificate` | DSL/UnifiedLearningCertificate | Final deterministic paper theorem with sampling, honesty, manifest, and influence-weighted local-law premises |
| `unified_learning_final_paper_certificate_high_prob` | DSL/UnifiedLearningCertificate | Final high-probability paper theorem from component-radius provenance |
| `unified_learning_certificate_high_prob` | DSL/UnifiedLearningCertificate | Union-bound high-probability certificate for the same decomposition |
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
- The OPT influence-weighted audit interface follows from an explicit design
  overlap condition plus a local-to-root propagation inequality
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

## InfluenceWeightedAuditOverlap and RootErrorControlledByInfluenceMass

**Location**: `OPT/InfluenceWeightedLocalLaws.lean`

**Status**: Explicit assumption structures / theorem parameters. These are not
Lean `axiom` declarations.

**Statement**:

For finite C1/C2/C3 audit rows `a`, local residuals `r(a)`, influence weights
`lambda(a)`, and logged audit propensities `pi(a)`, define the local-law mass

```text
sum_a lambda(a) * r(a).
```

The propagation interface requires root/document error to be bounded by this
mass. The audit-overlap interface requires `pi(a) > 0` for every row and bounds

```text
sum_a lambda(a)^2 / pi(a) <= Dlambda
lambda(a) / pi(a) <= Wlambda.
```

This is the formal "no adversarially hidden consequential needles" condition:
a needle may exist, but a root-relevant row cannot also have arbitrarily tiny
audit probability.

**Mathematical Justification**:
The condition is the design-based analogue of overlap/positivity in IPW
estimation. It is weaker than assuming information is IID within documents.
The document can be heterogeneous or needle-like; what matters for an
informative certificate is that influential rows have non-negligible logged
query probability and bounded residuals.

**When Safe**: When the audit policy logs propensities, has an exploration
floor on consequential row classes, and the chosen `lambda` weights genuinely
upper-bound root-error propagation.

**Lean certificate hooks**:
- `weightedOracleMass_le_proxy_plus_calibration`
- `rootError_le_of_influence_weighted_mass_upper`
- `rootError_le_proxy_estimate_plus_stat_plus_calibration`
- `InfluenceWeightedErrorCertificate.rootError_le_totalBound`

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

## TopLevelIID / TopLevelExchangeable

**Location**: `DSL/Honesty.lean`

**Statement**: IID or exchangeability applies to the paired top-level
observation sequence `(X_i, Y_i*)`. It does not assert that spans, nodes, or
local-law rows inside one top-level unit are IID.

**Mathematical Justification**:
The paper's document/case-level estimands average over source units. Once a
source unit is realized, its chunks, tree nodes, and audit rows are coupled by
the same text, topology, learned summaries, and oracle/readout residuals.

**When Safe**: When the declared `X_i` matches the estimand's sampling unit,
and linked siblings from the same source are grouped into the same top-level
fold.

---

## TopLevelSplit / ParentOf / DerivedRowHonestyContract

**Location**: `DSL/Honesty.lean`

**Statement**: The statistical train/eval split is over top-level units
(`X_i` in the paper-facing notation). Derived rows such as spans, tree nodes,
and local-law audit rows inherit their split role through a parent map
`ParentOf Case Row := Row → Case`.

**Mathematical Justification**:
Document or case-level claims require the IID/exchangeable object to be the
unit being split. Rows inside one top-level case are generally dependent:
their topology, summaries, residuals, and labels are all generated from the
same source. Treating those rows as independently split evaluation samples
would leak information across the train/eval boundary.

**When Safe**: When every derived row carries a stable parent top-level unit ID
and all sibling rows from that unit inherit the same top-level split role.

---

## KFoldHonestTraining / KFoldHonestEvaluation

**Location**: `DSL/Honesty.lean`

**Statement**: A fold-specific artifact for fold `k` is trained only on
top-level units outside fold `k`; the matching evaluation statistic for fold
`k` depends only on units inside fold `k`.

**Mathematical Justification**:
This is the cross-fitted version of honesty. It lets learned artifacts such as
chunkers, `g`, `f`, oracle views, and proxies be used on evaluation units while
still treating those artifacts as fixed relative to the held-out outcomes.

**When Safe**: When artifact lineage records the fold, split seed, parent unit
IDs, and the artifact was not updated from outcomes or residuals in its own
evaluation fold.

---

## UnifiedLearningHonesty and ChunkerObjectiveTerms

**Location**: `DSL/Honesty.lean`

**Statement**: `UnifiedLearningHonesty` bundles three-layer honesty for the
chunker, learned `g`, and oracle/readout training, plus final evaluation with a
frozen artifact bundle on the joint evaluation view. `ChunkerObjectiveTerms`
formalizes the chunker as an instrumental policy minimizing a weighted sum of
downstream loss, local-law residual mass, certificate radius, compute/query
cost, and boundary regularization.

**Mathematical Justification**:
Honest chunking may depend on learned `f/g` and other frozen artifacts. The
validity condition is that the artifacts used to choose boundaries for an
evaluation unit were trained out of fold, and that the unit's own held-out
label, residual, or local-law failure was not used before reporting that unit.

**When Safe**: When chunker updates use only chunker-train top-level units,
`g` updates use only `g`/summarizer-train units, oracle/readout updates use
only oracle-train units, and final metrics are computed with frozen artifacts
on the joint eval view.

---

## DocumentStructure and RunManifestContract

**Location**: `DSL/DocumentStructure.lean`

**Status**: Structural contract, not an axiom.

**Statement**: `Span` is a half-open ordered support interval.
`AdmissiblePartition` requires nonempty valid spans that are pairwise
non-overlapping and cover every document position. `RunManifestContract`
records the parent top-level unit for every theorem-facing row, checks that the
logged row and parent agree, and requires positive logged and effective
propensities. `ManifestRolesConsistent` and `ManifestSupportsValid` connect
manifest rows to the three-layer split and finite document support.

**Mathematical Justification**:
The IID/exchangeability claim is over top-level units.  The document-structure
layer prevents the paper from silently switching to chunks or nodes as if they
were independent observations: every span, node, and audit row is logged as a
derived object with a parent top-level unit and valid support.

**When Safe**: When chunk boundaries form an admissible partition of the
declared top-level unit, audit rows carry their parent unit IDs, and the run
manifest records split roles, artifact IDs, propensities, and influence weights
for every theorem-facing row.

---

## UnifiedLearningErrorCertificate

**Location**: `DSL/UnifiedLearningCertificate.lean`

**Status**: Theorem surface, not an axiom.

**Statement**: `UnifiedLearningErrorCertificate` stores the reported honest
estimate and four radii: local-law residual / transported distortion,
calibration, statistical estimation, and clipping/reporting. The theorem
`unified_learning_abs_gap_le_totalBound` proves the deterministic envelope:

```text
|target gap| <= |reported estimate|
              + local-law radius
              + calibration radius
              + estimation radius
              + clipping radius
```

`unified_learning_certificate_high_prob` gives the corresponding union-bound
event statement when each component has a high-probability failure bound.
`unified_learning_abs_gap_le_totalBound_from_influence` is the same final
certificate with the local-law radius supplied by
`InfluenceWeightedErrorCertificate.rootError_le_totalBound`.
`unified_learning_final_paper_certificate` and
`unified_learning_final_paper_certificate_high_prob` are the paper-facing
bundled theorem surfaces: their statements include top-level sampling,
three-layer honesty, admissible chunking/manifest contracts, and component
radius provenance.

**Mathematical Justification**:
Honesty and top-level splitting justify treating the reported estimate and
component bounds as held-out or cross-fitted quantities. The certificate file
then performs only deterministic triangle-inequality bookkeeping plus a union
bound over component failure events.

**When Safe**: When the component radii are produced from honest evaluation
units or valid out-of-fold artifacts, local-law residual mass is covered by the
influence-weighted audit design, and calibration/estimation/clipping terms are
reported separately.

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

/-! ## Influence-Weighted Local-Law Audit Interfaces -/

/-- Formal audit-overlap condition for influence-weighted local-law bounds. -/
abbrev influence_weighted_audit_overlap :=
  @FormalProofs.OPT.InfluenceWeightedAuditOverlap

/-- Local-to-root propagation condition for influence-weighted residual mass. -/
abbrev root_error_controlled_by_influence_mass :=
  @FormalProofs.OPT.RootErrorControlledByInfluenceMass

/-- Calibration transfer from proxy local-law residuals to true oracle residuals. -/
abbrev influence_weighted_calibration_transfer :=
  @FormalProofs.OPT.weightedOracleMass_le_proxy_plus_calibration

/-- One-shot root-error bound from a proxy estimate, statistical radius, and
calibration slack. -/
abbrev influence_weighted_proxy_root_error_bound :=
  @FormalProofs.OPT.rootError_le_proxy_estimate_plus_stat_plus_calibration

/-- Packaged finite-sample influence-weighted error certificate. -/
abbrev influence_weighted_error_certificate :=
  @FormalProofs.OPT.InfluenceWeightedErrorCertificate

end Axioms
