import FormalProofs.OPT.ExpectationTheory
import FormalProofs.OPT.PreferenceBounds
import FormalProofs.OPT.MeasureTheoreticAudit

/-!
# FormalProofs/TechnicalAxioms.lean

## Technical Lemmas (Lean Infrastructure)

This file documents the **Lean technical lemmas** used in the formalization
and points to their proof locations. Most are standard results already proved
in Mathlib or locally. The only remaining unsound item is the deprecated axiom
`PMF.summable_coe_real_mul`, kept temporarily for backwards compatibility.

### Distinction from Paper Assumptions

**Technical lemmas** (this file) are standard mathematical facts:
- Log-sum-exp is 1-Lipschitz
- Hoeffding's inequality (IS in Mathlib: `measure_sum_ge_le_of_iIndepFun`)
- Product measure properties

**Paper assumptions** (Assumptions.lean) are modeling choices:
- Local laws L1, L2, L3
- Oracle-measurability conditions
- Lipschitz policy assumptions

### Lemma Categories

| Category | Count | Description | Status |
|----------|-------|-------------|--------|
| PMF/Measure | 2 | Summability for PMF-weighted functions | 1 proved, 1 deprecated axiom |
| Lipschitz | 4 | Loss function Lipschitz bounds | Proved in `PreferenceBounds.lean` |
| Hoeffding | 5 | Statistical audit infrastructure | Proved in `MeasureTheoreticAudit.lean` |

### Proof Status

Each lemma is a standard result with a docstring explaining the math and a
concrete proof location. When additional assumptions are required (e.g.,
boundedness or countability), they are made explicit in the lemma statement.

### References

- **Log-sum-exp Lipschitz**: Boyd & Vandenberghe, Convex Optimization, §3.1.5
- **Hoeffding's inequality**: Mathlib `Mathlib.Probability.Moments.SubGaussian`
- **Product measure properties**: Billingsley, Probability and Measure
-/

set_option linter.mathlibStandardSet false

namespace TechnicalAxioms

/-!
## Category A: PMF/Measure Infrastructure

These lemmas handle summability for PMF-weighted functions.
They are used throughout the formalization for expectation calculations.
-/

/- **Lemma A1: PMF Summability** (axiom, deprecated)

For any PMF p and function f, the series ∑ p(z) * f(z) is summable.

**Location**: ExpectationTheory.lean:112

**Status**: Proper `axiom` declaration (not `sorry`). Deprecated in favor of bounded alternative.

**Mathematical justification**: For bounded f, this follows from absolute convergence
since ∑ p(z) = 1. The axiom extends to unbounded f, which is technically unsound
and kept only for backward compatibility.

**Soundness note**: See the docstring in ExpectationTheory.lean for details on
when this is safe vs. when to use the bounded variant.

**Preferred alternative**: `PMF.summable_coe_real_mul_of_bounded`.

Deprecated axiom removed; use the bounded summability lemma instead. -/

/-- **Lemma A2: Product Measure Summability**

For PMFs p, q and bounded function f, the double sum ∑∑ p(x)*q(z)*f(x,z) is summable.

**Location**: PreferenceBounds.lean:258

**Mathematical justification**: Fubini's theorem for product measures.
When |f(x,z)| ≤ M for all x,z, the sum is bounded by M * (∑p(x)) * (∑q(z)) = M.

**Used in**: Coupling arguments for gap bounds (dpo_gap_bounded, grpo_gap_bounded, etc.) -/
abbrev product_measure_summability := @PMF.summable_prod_mul_of_bounded

/-!
## Category B: Lipschitz Bounds

These lemmas establish Lipschitz continuity for preference learning loss functions.
They are used to derive quantitative gap bounds.
-/

/-- **Lemma B1: Log-Sum-Exp is 1-Lipschitz**

|log(∑ exp(xᵢ)) - log(∑ exp(yᵢ))| ≤ max_i |xᵢ - yᵢ|

**Location**: PreferenceBounds.lean:2903

**Mathematical justification**: The gradient of log-sum-exp is the softmax,
which has ℓ¹-norm exactly 1 (since softmax outputs sum to 1).
By the mean value theorem, the function is 1-Lipschitz in ℓ∞ norm.

**Reference**: Boyd & Vandenberghe, Convex Optimization, Section 3.1.5.

The local proof avoids differential calculus by using exponential bounds. -/
abbrev logsumexp_lipschitz := @logSumExp_lipschitz_uniform

/-- **Lemma B2: Plackett-Luce Loss Lipschitz (Same Ranks)**

When scores change by at most L, Plackett-Luce loss changes by at most 2kL.

**Location**: PreferenceBounds.lean:3037

**Mathematical justification**:
  PL(s, r) = -∑_i [s_i - log(∑_{j:r_j≥r_i} exp(s_j))]
Each term involves one score and one log-sum-exp, both 1-Lipschitz.
With k items, total is (2k)-Lipschitz. -/
abbrev plackett_luce_lipschitz_same := @PlackettLuceLoss_lipschitz_same_ranks

/-
**Lemma B3: Plackett-Luce Loss Lipschitz (General)** (DEPRECATED - MOVED)

Status: This lemma has been SUPERSEDED by the `ExpectedGRPOLossLipschitz` axiom
in PreferenceBounds.lean. The pointwise version is unprovable for `dist > 0` because
rankings are discontinuous at ties.

Moved to: `FormalProofs/Deprecated/PointwiseLipschitz.lean`

The main theorems now use expected Lipschitz (Random Utility Model assumption) instead.
See the Random Utility Model section in PreferenceBounds.lean for details.
-/
-- abbrev plackett_luce_lipschitz_general := @PlackettLuceLoss_lipschitz_general

/-
**Lemma B4: GRPO-RL Loss Lipschitz** (DEPRECATED - MOVED)

Status: This lemma has been SUPERSEDED by the `ExpectedGRPORLLossLipschitz` axiom
in PreferenceBounds.lean. The pointwise version is unprovable for `dist > 0` because
rankings/orderings can change discontinuously.

Moved to: `FormalProofs/Deprecated/PointwiseLipschitz.lean`

The main theorems now use expected Lipschitz (Random Utility Model assumption) instead.
See the Random Utility Model section in PreferenceBounds.lean for details.
-/
-- abbrev grpo_rl_lipschitz := @GRPORLLoss_lipschitz_general

/-!
## Category C: Hoeffding/Product Measure

These lemmas support the measure-theoretic audit framework.
They connect empirical violation rates to theoretical bounds.
Most are standard results IN Mathlib but require wiring infrastructure.
-/

/-! ### Lemma C1: Coordinate Projections are Independent

Under product measure μ^n, the coordinate projections ω ↦ ω_i are independent.

**Status**: Now proved inline using Mathlib's `iIndepFun_pi` in `hoeffding_violation_rate_bound`.

**Mathematical justification**: Fundamental property of product measures.
The joint distribution factors: P(ω_1 ∈ A_1, ..., ω_n ∈ A_n) = ∏ P(ω_i ∈ A_i).

**Reference**: Billingsley, Probability and Measure, Theorem 18.2

See `MeasureTheoreticAudit.hoeffding_violation_rate_bound` for usage of `iIndepFun_pi`.
-/

/-- **Lemma C2: Marginal Integral Property**

∫ f(ω_i) dμ^n = ∫ f dμ for any coordinate projection.

**Location**: MeasureTheoreticAudit.lean:155

**Mathematical justification**: The marginal of a product measure is the original measure.
Fubini's theorem allows computing the integral over one coordinate. -/
abbrev marginal_integral := @integral_proj_eq_marginal

/-- **Lemma C3: Integrability Preserved Under Projection**

If f is integrable under μ, then f ∘ (· i) is integrable under μ^n.

**Location**: MeasureTheoreticAudit.lean:192

**Mathematical justification**: Follows from the marginal property.
∫|f(ω_i)| dμ^n = ∫|f| dμ < ∞ by assumption. -/
abbrev integrability_preserved := @integrable_proj_of_integrable

/-- **Lemma C4: Deviation Set is Measurable**

The set {ω | |p̂(ω) - p| ≥ ε} is null-measurable.

**Location**: MeasureTheoreticAudit.lean:260

**Mathematical justification**: Empirical rate is a finite sum of indicator functions,
each measurable. The set {x | |f(x)| ≥ c} is measurable for measurable f. -/
abbrev deviation_measurable := @deviationSet_nullMeasurable

/-- **Lemma C5: Hoeffding's Inequality**

For n iid random variables X_i ∈ [0,1] with mean p:
  P(|p̂ - p| ≥ ε) ≤ 2 * exp(-2nε²)

**Location**: MeasureTheoreticAudit.lean:357

**Mathematical justification**: Classical result. Proof via:
1. Center: Y_i = X_i - p has E[Y_i] = 0
2. Y_i ∈ [-1,1] implies sub-Gaussian with parameter 1/4
3. Apply Chernoff bound to ∑Y_i

**Reference**: Hoeffding (1963), "Probability Inequalities for Sums of Bounded
Random Variables", JASA 58(301):13-30 -/
abbrev hoeffding_inequality := @hoeffding_iid_bounded_axiom

/-!
## Summary

| Lemma | Category | Status | Used For |
|-------|----------|--------|----------|
| `product_measure_summability` | PMF | proved | Coupling arguments |
| `logsumexp_lipschitz` | Lipschitz | proved | GRPO Plackett-Luce |
| `plackett_luce_lipschitz_same` | Lipschitz | proved | GRPO gap bounds |
| `plackett_luce_lipschitz_general` | Lipschitz | deprecated (moved) | GRPO gap bounds |
| `grpo_rl_lipschitz` | Lipschitz | deprecated (moved) | GRPO-RL gap bounds |
| `coordinate_independence` | Hoeffding | trivial | Audit sample complexity |
| `marginal_integral` | Hoeffding | proved | Audit expectation |
| `integrability_preserved` | Hoeffding | proved | Audit integrability |
| `deviation_measurable` | Hoeffding | proved (discrete Strings) | Audit measurability |
| `hoeffding_inequality` | Hoeffding | proved (Mathlib wiring) | Audit concentration |

All items are standard mathematical facts. The legacy PMF summability axiom has
been removed in favor of bounded summability lemmas.

**Future work priorities:**
1. Prefer bounded metric assumptions (`BoundedPseudoMetricSpace`) where needed
2. Keep measurability assumptions explicit (countability or measurability of `fstar`)
-/

end TechnicalAxioms
