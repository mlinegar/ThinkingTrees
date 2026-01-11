/-
# FormalProofs/DSL/LogisticRegression.lean

## Paper Reference: Section 3.1 (GLM Extensions), Appendix OA.5

This file formalizes DSL for logistic regression:
- DSL moment-based estimation
- Odds ratio estimation
- Standard errors and inference

### Key Difference from Linear Regression

For logistic regression, we cannot simply plug Ỹ into standard software.
Instead, we solve the DSL moment condition:

  (1/N) ∑ m̃(Dᵢ; β) = 0

where m̃ is the design-adjusted logistic moment:
  m̃(D; β) = m(D^obs, D̂^mis; β) - (R/π)(m(D^obs, D̂^mis; β) - m(D^obs, D^mis; β))
-/

import FormalProofs.DSL.DSLEstimator
import FormalProofs.DSL.AsymptoticTheory

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical
open MeasureTheory

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace DSL

/-!
## Logistic Regression Data
-/

/-- Data for DSL logistic regression -/
structure DSLLogisticData (n d : ℕ) where
  /-- Covariate matrix X (n × d) -/
  X : Fin n → Fin d → ℝ
  /-- Predicted binary outcomes Ŷ ∈ {0, 1} or probabilities -/
  Y_pred : Fin n → ℝ
  /-- True binary outcomes Y ∈ {0, 1} (for sampled units) -/
  Y_true : Fin n → ℝ
  /-- Sampling indicators R -/
  R : Fin n → SamplingIndicator
  /-- Sampling probabilities π -/
  π : Fin n → ℝ
  /-- Positivity condition -/
  h_π_pos : ∀ i, π i > 0

/-!
## DSL Logistic Moment
-/

/-- Standard logistic moment: (Y - expit(X'β)) · X -/
def logisticMomentAtPoint {d : ℕ}
    (Y : ℝ) (X : Fin d → ℝ) (β : Fin d → ℝ) : Fin d → ℝ :=
  fun j => (Y - expit (innerProduct X β)) * X j

/-- DSL logistic moment for a single observation -/
def DSLLogisticMoment {d : ℕ}
    (Y_pred Y_true : ℝ)
    (X : Fin d → ℝ)
    (R : SamplingIndicator)
    (π : ℝ)
    (β : Fin d → ℝ) : Fin d → ℝ :=
  let m_pred := logisticMomentAtPoint Y_pred X β
  let m_true := logisticMomentAtPoint Y_true X β
  fun j => designAdjustedOutcome (m_pred j) (m_true j) R π

/-- Sample average of DSL logistic moments -/
def sampleDSLLogisticMoment {n d : ℕ}
    (data : DSLLogisticData n d)
    (β : Fin d → ℝ) : Fin d → ℝ :=
  fun j => (∑ i : Fin n, DSLLogisticMoment (data.Y_pred i) (data.Y_true i)
    (data.X i) (data.R i) (data.π i) β j) / n

/-!
## Estimation via GMM
-/

/-- DSL logistic regression estimator.

    β̂_DSL solves: (1/N) ∑ m̃(Dᵢ; β) = 0

    This requires iterative optimization (e.g., Newton-Raphson). -/
def DSLLogisticEstimator {n d : ℕ}
    (data : DSLLogisticData n d)
    (initial_β : Fin d → ℝ)
    (max_iter : ℕ) : Fin d → ℝ :=
  -- Placeholder: actual implementation would use Newton-Raphson
  initial_β

/-- Objective function for GMM: ||sample moment||² -/
def gmmObjective {n d : ℕ}
    (data : DSLLogisticData n d)
    (β : Fin d → ℝ) : ℝ :=
  let m := sampleDSLLogisticMoment data β
  ∑ j : Fin d, (m j)^2

/-!
## Jacobian for Newton-Raphson
-/

/-- Derivative of expit: expit(x) · (1 - expit(x)) -/
def expitDerivative (x : ℝ) : ℝ :=
  expit x * (1 - expit x)

/-- Jacobian of logistic moment: ∂m/∂β -/
def logisticJacobian {d : ℕ}
    (X : Fin d → ℝ)
    (β : Fin d → ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  let p := expit (innerProduct X β)
  fun i j => -p * (1 - p) * X i * X j

/-- Sample Jacobian for DSL logistic regression -/
def sampleJacobian {n d : ℕ}
    (data : DSLLogisticData n d)
    (β : Fin d → ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  fun i j => (∑ k : Fin n, logisticJacobian (data.X k) β i j) / n

/-!
## Standard Errors
-/

/-- Sandwich variance for DSL logistic regression.

    V = S⁻¹ M S⁻¹' where:
    - S = E[∂m̃/∂β]
    - M = E[m̃ m̃'] -/
def sandwichVarianceLogistic {n d : ℕ}
    (data : DSLLogisticData n d)
    (β_hat : Fin d → ℝ)
    (S_inv : Matrix (Fin d) (Fin d) ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  -- Meat matrix: (1/n) ∑ m̃ᵢ m̃ᵢ'
  let meat : Matrix (Fin d) (Fin d) ℝ := fun i j =>
    (∑ k : Fin n,
      let m_k := DSLLogisticMoment (data.Y_pred k) (data.Y_true k)
        (data.X k) (data.R k) (data.π k) β_hat
      m_k i * m_k j) / n
  S_inv * meat * S_inv.transpose

/-- Standard errors for logistic regression coefficients -/
def standardErrorsLogistic {n d : ℕ}
    (V : Matrix (Fin d) (Fin d) ℝ)
    (N : ℕ) : Fin d → ℝ :=
  fun i => Real.sqrt (V i i / N)

/-!
## Odds Ratios
-/

/-- Odds ratio for coefficient j: exp(βⱼ) -/
def oddsRatio {d : ℕ} (β : Fin d → ℝ) (j : Fin d) : ℝ :=
  Real.exp (β j)

/-- Standard error for log odds ratio (= SE for β) -/
def seLogOddsRatio {d : ℕ} (SE_β : Fin d → ℝ) (j : Fin d) : ℝ :=
  SE_β j

/-- 95% CI for odds ratio -/
def oddsRatioCI95 {d : ℕ}
    (β : Fin d → ℝ)
    (SE_β : Fin d → ℝ)
    (j : Fin d) : ℝ × ℝ :=
  let log_lower := β j - 1.96 * SE_β j
  let log_upper := β j + 1.96 * SE_β j
  (Real.exp log_lower, Real.exp log_upper)

/-!
## Marginal Effects
-/

/-- Average marginal effect for variable j.

    AME_j = (1/N) ∑ expit'(Xᵢ'β) · βⱼ -/
def averageMarginalEffect {n d : ℕ}
    (data : DSLLogisticData n d)
    (β : Fin d → ℝ)
    (j : Fin d) : ℝ :=
  (∑ i : Fin n,
    let xb := innerProduct (data.X i) β
    expitDerivative xb * β j) / n

/-- Marginal effect at the mean.

    MEM_j = expit'(X̄'β) · βⱼ -/
def marginalEffectAtMean {n d : ℕ}
    (data : DSLLogisticData n d)
    (β : Fin d → ℝ)
    (j : Fin d) : ℝ :=
  let X_bar : Fin d → ℝ := fun k => (∑ i : Fin n, data.X i k) / n
  let xb := innerProduct X_bar β
  expitDerivative xb * β j

/-!
## Properties
-/

/-- Logistic moment function over (X, Y) pairs. -/
def logisticMomentPair {d : ℕ} : MomentFunction ((Fin d → ℝ) × ℝ) d :=
  fun D β => logisticMoment D.2 D.1 β

/-- DSL logistic regression is consistent.

    Under Assumption 1: β̂_DSL →p β* as N → ∞ -/
theorem DSL_logistic_consistent
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Con : Type*} {d : ℕ}
    (axioms :
      MEstimationAxioms Ω ((Fin d → ℝ) × ℝ × ℝ × SamplingIndicator × ℝ) μ d)
    (dbs : DesignBasedSampling (Fin d → ℝ) ℝ Con)
    (β_star : Fin d → ℝ)
    (reg : RegularityConditions
      ((Fin d → ℝ) × ℝ × ℝ × SamplingIndicator × ℝ) d)
    (h_unbiased :
      MomentUnbiased (DSLMomentFromData (logisticMomentPair (d := d))) axioms.E β_star)
    (data_seq : ℕ →
      Ω → List ((Fin d → ℝ) × ℝ × ℝ × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est :
      IsMEstimatorSeq (DSLMomentFromData (logisticMomentPair (d := d)))
        data_seq β_hat_seq)
    : ConvergesInProbability μ β_hat_seq (fun _ => β_star) := by
  exact DSL_consistent μ axioms dbs (logisticMomentPair (d := d)) β_star reg h_unbiased data_seq
    β_hat_seq h_est

/-- DSL confidence intervals for log-odds (or transformed odds ratios) are valid.

    This theorem exposes the generic CI coverage statement; users can
    instantiate `CI_seq` with `oddsRatioCI95` or a Wald CI on β. -/
theorem DSL_odds_ratio_valid_coverage
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Con : Type*} {d : ℕ}
    (axioms :
      MEstimationAxioms Ω ((Fin d → ℝ) × ℝ × ℝ × SamplingIndicator × ℝ) μ d)
    (coverage_axioms : CoverageAxioms μ d)
    (dbs : DesignBasedSampling (Fin d → ℝ) ℝ Con)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions
      ((Fin d → ℝ) × ℝ × ℝ × SamplingIndicator × ℝ) d)
    (h_unbiased :
      MomentUnbiased (DSLMomentFromData (logisticMomentPair (d := d))) axioms.E β_star)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ) (h_α : 0 < α ∧ α < 1)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    : AsymptoticCoverage μ CI_seq β_star α := by
  exact DSL_valid_coverage μ axioms coverage_axioms dbs (logisticMomentPair (d := d)) β_star V reg
    h_unbiased CI_seq α h_α centered_scaled_seq

/-!
## Binary Outcome with Binary Predictor
-/

/-- Special case: 2×2 contingency table.

    When both Y and X are binary, the odds ratio has a simple form. -/
structure BinaryBinaryData (n : ℕ) where
  X : Fin n → Bool       -- Binary predictor
  Y_pred : Fin n → Bool  -- Predicted outcome
  Y_true : Fin n → Bool  -- True outcome (for sampled)
  R : Fin n → SamplingIndicator
  π : Fin n → ℝ
  h_π_pos : ∀ i, π i > 0

/-- Cell counts after design adjustment -/
structure AdjustedContingencyTable where
  n11_tilde : ℝ  -- Ỹ = 1, X = 1
  n10_tilde : ℝ  -- Ỹ = 0, X = 1
  n01_tilde : ℝ  -- Ỹ = 1, X = 0
  n00_tilde : ℝ  -- Ỹ = 0, X = 0

/-- DSL odds ratio from contingency table -/
def oddsRatioFromTable (t : AdjustedContingencyTable) : ℝ :=
  (t.n11_tilde * t.n00_tilde) / (t.n10_tilde * t.n01_tilde)

end DSL

end
