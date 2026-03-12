import FormalProofs.OPT.PreferenceBounds
import FormalProofs.OPT.AuditBounds

/-!
# Assumption Registry

This file documents all modeling assumptions and assumption structures used in the formalization.

## Summary

### OPT Module: RUM Assumption (No Axioms)

| # | Assumption | Location | Purpose |
|---|------------|----------|---------|
| 1 | `ExpectedGroupLossLipschitz` | FormalProbability/DSL/RUM | Expected loss over groups is Lipschitz |

This single assumption is justified by the **Random Utility Model** (McFadden 1974).
Under continuous noise, ranking ties have measure zero, so the expected loss is Lipschitz
even though the pointwise ranking function is discontinuous.

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

### Econometrics Module: Assumption Structures

| Structure | Location | Purpose |
|-----------|----------|---------|
| `OLSAsymptoticAxioms` | Econometrics/OLS/AsymptoticOLS | LLN/CLT/Slutsky/delta-method package for OLS asymptotics |
| `ScoreLLNAssumption` / `ScoreCLTAssumption` | Econometrics/OLS/AsymptoticOLS | Decomposed score-process assumptions |
| `OLSConsistencyAssumption` / `OLSAsymptoticNormalAssumption` | Econometrics/OLS/AsymptoticOLS | Decomposed OLS limit assumptions |

## Soundness

All assumptions and assumption structures are **modeling choices**, not gaps in the proof:
- Each has rigorous mathematical justification from the statistics/econometrics literature
- The OPT RUM assumption follows from the Random Utility Model (McFadden 1974)
- The DSL assumptions follow from M-estimation theory (Newey & McFadden 1994)
- The formalization is SOUND under these assumptions

---

## Assumption: ExpectedGroupLossLipschitz

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

**When Safe**: Always safe when using softmax/Plackett-Luce with temperature > 0.

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

**Statement**: Confidence intervals constructed from asymptotically normal
estimators achieve nominal coverage asymptotically.

**Mathematical Justification**:
If √n(β̂ - β) →d N(0, V), then the interval β̂ ± z_{α/2} × SE(β̂) covers β
with probability approaching 1-α.

This is the standard justification for Wald-type confidence intervals.

**When Safe**: When the asymptotic approximation is accurate (typically n ≥ 30
for well-behaved data, larger for heavy tails or sparse data).

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

This is the **single foundational axiom** for preference learning bounds.
Justified by the Random Utility Model (McFadden 1974). -/
abbrev expected_group_loss_lipschitz := @ExpectedGroupLossLipschitz

/-- GRPO Plackett-Luce expected loss is Lipschitz in oracle distance.

Instantiation of `ExpectedGroupLossLipschitz` for GRPO-PL. -/
abbrev grpo_pl_expected_lipschitz := @ExpectedGRPOLossLipschitz

/-- GRPO-RL (DeepSeek-R1 style) expected loss is Lipschitz in oracle distance.

Instantiation of `ExpectedGroupLossLipschitz` for GRPO-RL. -/
abbrev grpo_rl_expected_lipschitz := @ExpectedGRPORLLossLipschitz

end Axioms
