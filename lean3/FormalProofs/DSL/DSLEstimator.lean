import FormalProofs.DSL.SamplingTheory
import FormalProofs.DSL.MomentFunctions

/-!
# FormalProofs/DSL/DSLEstimator.lean

## Paper Reference: Section 3 - The DSL Estimator

This file formalizes the core DSL estimator:
- Design-adjusted outcome Ỹ (Equation 4)
- DSL moment function (Equation OA.5)
- DSL estimator for category proportions (Equation 5)
- DSL regression estimator

### Key Formula: Design-Adjusted Outcome (Equation 4)

  Ỹ = Ŷ - (R/π)(Ŷ - Y)

where:
- Ŷ = predicted outcome (from ML/LLM)
- Y = true outcome (when R=1)
- R = sampling indicator
- π = sampling probability

### Key Property

The design-adjusted outcome is unbiased for the true outcome:
  E[Ỹ | X] = E[Y | X]

This holds regardless of prediction accuracy because:
  E[Ỹ - Y | X] = E[Ŷ - Y | X] - E[(R/π)(Ŷ - Y) | X]
               = E[Ŷ - Y | X] - E[Ŷ - Y | X]  (by Assumption 1)
               = 0
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace DSL

/-!
## Design-Adjusted Outcome (Equation 4)
-/

/-- Design-adjusted outcome: Ỹ = Ŷ - (R/π)(Ŷ - Y)

    This is the core of the DSL method. When R=0, we use the predicted value.
    When R=1, we correct for the prediction error using IPW.

    Note: When R=0, Y_true is not available, but we set it to 0 (unused). -/
def designAdjustedOutcome
    (Y_pred : ℝ)      -- Ŷ: predicted outcome
    (Y_true : ℝ)      -- Y: true outcome (only used when R=true)
    (R : SamplingIndicator)  -- R: sampling indicator
    (π : ℝ)           -- π: sampling probability (positive)
    : ℝ :=
  Y_pred - (if R then (1/π) * (Y_pred - Y_true) else 0)

/-- Alternative form: Ỹ = (1 - R/π)Ŷ + (R/π)Y -/
lemma designAdjustedOutcome_alt (Y_pred Y_true : ℝ) (R : SamplingIndicator) (π : ℝ) (hπ : π > 0) :
    designAdjustedOutcome Y_pred Y_true R π =
    (1 - R.toReal / π) * Y_pred + (R.toReal / π) * Y_true := by
  unfold designAdjustedOutcome SamplingIndicator.toReal
  cases R with
  | false => simp
  | true =>
      simp
      ring

/-- When R=0 (not sampled), Ỹ = Ŷ -/
lemma designAdjustedOutcome_unsampled (Y_pred Y_true : ℝ) (π : ℝ) :
    designAdjustedOutcome Y_pred Y_true false π = Y_pred := by
  unfold designAdjustedOutcome
  simp

/-- When R=1 (sampled), Ỹ = Ŷ - (1/π)(Ŷ - Y) = (1 - 1/π)Ŷ + (1/π)Y -/
lemma designAdjustedOutcome_sampled (Y_pred Y_true : ℝ) (π : ℝ) :
    designAdjustedOutcome Y_pred Y_true true π =
    Y_pred - (1/π) * (Y_pred - Y_true) := by
  unfold designAdjustedOutcome
  simp

/-- When π=1 (all sampled), Ỹ = Y for sampled units -/
lemma designAdjustedOutcome_full_sample (Y_pred Y_true : ℝ) :
    designAdjustedOutcome Y_pred Y_true true 1 = Y_true := by
  unfold designAdjustedOutcome
  simp

/-!
## DSL Moment Function (Equation OA.5)
-/

/-- DSL moment function: applies design adjustment to general moment functions.

    m̃(D; β) = m(D^obs, D̂^mis; β) - (R/π)(m(D^obs, D̂^mis; β) - m(D^obs, D^mis; β))

    This generalizes the design-adjusted outcome to any moment-based estimator. -/
def DSLMoment {Obs Mis : Type*} {d : ℕ}
    (m : MomentFunction (Obs × Mis) d)  -- Original moment function
    (d_obs : Obs)
    (d_mis_pred : Mis)    -- D̂^mis: predicted missing
    (d_mis_true : Mis)    -- D^mis: true missing (when R=1)
    (R : SamplingIndicator)
    (π : ℝ)
    (β : Fin d → ℝ)
    : Fin d → ℝ :=
  let m_pred := m (d_obs, d_mis_pred) β
  let m_true := m (d_obs, d_mis_true) β
  fun i => designAdjustedOutcome (m_pred i) (m_true i) R π

/-- DSL moment when not sampled: just use predicted moment -/
lemma DSLMoment_unsampled {Obs Mis : Type*} {d : ℕ}
    (m : MomentFunction (Obs × Mis) d)
    (d_obs : Obs) (d_mis_pred d_mis_true : Mis) (π : ℝ) (β : Fin d → ℝ) :
    DSLMoment m d_obs d_mis_pred d_mis_true false π β =
    m (d_obs, d_mis_pred) β := by
  unfold DSLMoment
  ext i
  simp [designAdjustedOutcome_unsampled]

/-- DSL moment when sampled: correct for prediction error -/
lemma DSLMoment_sampled {Obs Mis : Type*} {d : ℕ}
    (m : MomentFunction (Obs × Mis) d)
    (d_obs : Obs) (d_mis_pred d_mis_true : Mis) (π : ℝ) (β : Fin d → ℝ) :
    DSLMoment m d_obs d_mis_pred d_mis_true true π β = fun i =>
      m (d_obs, d_mis_pred) β i -
      (1/π) * (m (d_obs, d_mis_pred) β i - m (d_obs, d_mis_true) β i) := by
  unfold DSLMoment
  ext i
  simp [designAdjustedOutcome_sampled]

/-!
## DSL for Category Proportions (Equation 5)
-/

/-- DSL estimator for a category proportion.

    (1/N)∑Ỹ = (1/N)∑Ŷ - ((1/n)∑_{R=1}Ŷ - (1/n)∑_{R=1}Y)

    This is the simplest application: estimating the proportion of documents
    in a category (e.g., proportion of attacking ads). -/
def DSLProportion
    (Y_pred : List ℝ)      -- All N predicted indicators
    (Y_true : List ℝ)      -- True indicators (for R=1 units only)
    (R : List SamplingIndicator)  -- Sampling indicators
    (π : ℝ)                -- Uniform sampling probability n/N
    : ℝ :=
  let N := Y_pred.length
  let mean_pred := Y_pred.sum / N
  -- Get predictions for labeled units
  let labeled_pred := (Y_pred.zip R).filterMap fun ⟨y, r⟩ => if r then some y else none
  -- Mean of predictions for labeled
  let mean_pred_labeled := labeled_pred.sum / labeled_pred.length
  -- Mean of true values for labeled
  let mean_true_labeled := Y_true.sum / Y_true.length
  -- DSL correction: subtract the bias in labeled sample
  mean_pred - (mean_pred_labeled - mean_true_labeled)

/-- Alternative DSL proportion formula using individual Ỹ values -/
def DSLProportionAlt
    (docs : List (ℝ × ℝ × SamplingIndicator × ℝ))  -- (Y_pred, Y_true, R, π)
    : ℝ :=
  let N := docs.length
  let Y_tilde_sum := docs.foldl (fun acc ⟨y_pred, y_true, r, π⟩ =>
    acc + designAdjustedOutcome y_pred y_true r π) 0
  Y_tilde_sum / N

/-!
## DSL Linear Regression (Equation 6)
-/

/-- DSL linear regression estimator.

    β̂_DSL = (X'X)⁻¹X'Ỹ

    where Ỹ is the vector of design-adjusted outcomes.
    This is OLS on the adjusted outcomes. -/
structure DSLLinearRegressionData (d : ℕ) where
  n_obs : ℕ                        -- Number of observations
  X : Fin n_obs → Fin d → ℝ        -- Covariate matrix (n × d)
  Y_tilde : Fin n_obs → ℝ          -- Design-adjusted outcomes

/-- Compute X'X (Gram matrix) -/
def gramMatrix {n d : ℕ} (X : Fin n → Fin d → ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  fun i j => ∑ k : Fin n, X k i * X k j

/-- Compute X'Ỹ -/
def XtY {n d : ℕ} (X : Fin n → Fin d → ℝ) (Y : Fin n → ℝ) : Fin d → ℝ :=
  fun i => ∑ k : Fin n, X k i * Y k

/-- DSL linear regression: β̂ = (X'X)⁻¹X'Ỹ -/
def DSLLinearRegression {d : ℕ} (data : DSLLinearRegressionData d)
    (XtX_inv : Matrix (Fin d) (Fin d) ℝ) : Fin d → ℝ :=
  fun i => ∑ j : Fin d, XtX_inv i j * XtY data.X data.Y_tilde j

/-!
## Sample Moment Functions
-/

/-- Sample average of DSL moments -/
def sampleDSLMoment {Obs Mis : Type*} {d : ℕ}
    (m : MomentFunction (Obs × Mis) d)
    (data : List (Obs × Mis × Mis × SamplingIndicator × ℝ))  -- (d_obs, d_mis_pred, d_mis_true, R, π)
    (β : Fin d → ℝ)
    : Fin d → ℝ :=
  let N := data.length
  fun i =>
    (data.foldl (fun acc ⟨d_obs, d_mis_pred, d_mis_true, R, π⟩ =>
      acc + DSLMoment m d_obs d_mis_pred d_mis_true R π β i) 0) / N

/-!
## Unbiasedness Properties
-/

/-- Key unbiasedness result: E[Ỹ - Y | X] = 0 under Assumption 1.

    This theorem captures the core insight of DSL: the design-adjusted
    outcome is unbiased for the true outcome, regardless of prediction error.

    The proof relies on:
    1. E[R/π | D^obs, Q] = 1 (Horvitz-Thompson)
    2. R ⊥ D^mis | D^obs, Q (conditional independence from Assumption 1) -/
theorem DSL_unbiased
    {Obs Mis Con : Type*} {d : ℕ}
    (dbs : DesignBasedSampling Obs Mis Con)
    (Y_pred Y_true : ℝ)
    (d_obs : Obs) (q : Con)
    (π_eq : dbs.prob d_obs q = dbs.prob d_obs q)  -- π is known
    -- Expectation operator (weighted by sampling distribution)
    (E_cond : (SamplingIndicator → ℝ) → ℝ)
    -- E[R] = π
    (hE_R : E_cond (fun R => R.toReal) = dbs.prob d_obs q)
    -- E[1] = 1
    (hE_1 : E_cond (fun _ => 1) = 1)
    -- Linearity of expectation
    (hE_linear : ∀ f g : SamplingIndicator → ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g)
    : E_cond (fun R => designAdjustedOutcome Y_pred Y_true R (dbs.prob d_obs q) - Y_true) = 0 := by
  -- Ỹ - Y = Ŷ - Y - (R/π)(Ŷ - Y) = (1 - R/π)(Ŷ - Y)
  -- E[(1 - R/π)(Ŷ - Y)] = (Ŷ - Y) · E[1 - R/π] = (Ŷ - Y) · (1 - E[R]/π)
  --                     = (Ŷ - Y) · (1 - π/π) = 0
  have h1 : E_cond (fun R => designAdjustedOutcome Y_pred Y_true R (dbs.prob d_obs q) - Y_true) =
            E_cond (fun R => (1 - R.toReal / dbs.prob d_obs q) * (Y_pred - Y_true)) := by
    congr 1
    ext R
    unfold designAdjustedOutcome SamplingIndicator.toReal
    cases R with
    | false => simp
    | true =>
        simp
        ring
  rw [h1]
  -- Use linearity
  have h2 : E_cond (fun R => (1 - R.toReal / dbs.prob d_obs q) * (Y_pred - Y_true)) = 0 := by
    calc E_cond (fun R => (1 - R.toReal / dbs.prob d_obs q) * (Y_pred - Y_true))
        = E_cond (fun R => (Y_pred - Y_true) * 1 + (-(Y_pred - Y_true) / dbs.prob d_obs q) * R.toReal) := by
            congr 1; ext R; ring
      _ = (Y_pred - Y_true) * E_cond (fun _ => 1) +
          (-(Y_pred - Y_true) / dbs.prob d_obs q) * E_cond (fun R => R.toReal) := by
            rw [hE_linear]
      _ = (Y_pred - Y_true) * 1 + (-(Y_pred - Y_true) / dbs.prob d_obs q) * dbs.prob d_obs q := by
            rw [hE_1, hE_R]
      _ = (Y_pred - Y_true) * (1 - dbs.prob d_obs q / dbs.prob d_obs q) := by ring
      _ = (Y_pred - Y_true) * (1 - 1) := by
            rw [div_self (ne_of_gt (dbs.prob_pos d_obs q))]
      _ = 0 := by ring
  rw [h2]

/-- DSL moment is unbiased for each coordinate under Assumption 1. -/
theorem DSLMoment_unbiased
    {Obs Mis Con : Type*} {d : ℕ}
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (d_obs : Obs) (d_mis_pred d_mis_true : Mis) (q : Con)
    (β : Fin d → ℝ) (i : Fin d)
    (E_cond : (SamplingIndicator → ℝ) → ℝ)
    (hE_R : E_cond (fun R => R.toReal) = dbs.prob d_obs q)
    (hE_1 : E_cond (fun _ => 1) = 1)
    (hE_linear : ∀ f g : SamplingIndicator → ℝ, ∀ a b : ℝ,
      E_cond (fun R => a * f R + b * g R) = a * E_cond f + b * E_cond g) :
    E_cond (fun R =>
      DSLMoment m d_obs d_mis_pred d_mis_true R (dbs.prob d_obs q) β i
        - m (d_obs, d_mis_true) β i) = 0 := by
  -- Reduce to scalar DSL_unbiased on coordinate i.
  have h := DSL_unbiased (d := d) (dbs := dbs)
      (Y_pred := m (d_obs, d_mis_pred) β i)
      (Y_true := m (d_obs, d_mis_true) β i)
      (d_obs := d_obs) (q := q) (π_eq := rfl)
      (E_cond := E_cond) (hE_R := hE_R) (hE_1 := hE_1) (hE_linear := hE_linear)
  simpa [DSLMoment] using h

/-!
## DSL Bundle for Complete Data
-/

/-- Complete DSL estimation setup -/
structure DSLSetup (Obs Mis Con : Type*) (d : ℕ) where
  /-- Design-based sampling assumption -/
  sampling : DesignBasedSampling Obs Mis Con
  /-- Moment function for the analysis -/
  moment : MomentFunction (Obs × Mis) d

/-- Apply DSL to a single document -/
def DSLSetup.applyDocument {Obs Mis Con : Type*} {d : ℕ}
    (setup : DSLSetup Obs Mis Con d)
    (doc : AnnotatedDocument Obs Mis Con)
    (β : Fin d → ℝ) : Fin d → ℝ :=
  match doc.d_mis_true with
  | some d_mis_true =>
      DSLMoment setup.moment doc.d_obs doc.d_mis_pred d_mis_true
        doc.sampled (setup.sampling.prob doc.d_obs doc.content) β
  | none =>
      -- When not sampled, d_mis_true is not available
      -- Use d_mis_pred for both (the R/π correction will be 0)
      setup.moment (doc.d_obs, doc.d_mis_pred) β

lemma DSLSetup.applyDocument_oracle {Obs Mis Con : Type*} {d : ℕ}
    (setup : DSLSetup Obs Mis Con d)
    (oracle : OracleAccess Obs Mis Con)
    (doc : AnnotatedDocument Obs Mis Con)
    (β : Fin d → ℝ)
    (h_sampled : doc.sampled = true) :
    setup.applyDocument doc β =
      DSLMoment setup.moment doc.d_obs doc.d_mis_pred (oracle.oracle doc.content)
        doc.sampled (setup.sampling.prob doc.d_obs doc.content) β := by
  have h_oracle := oracle.agrees_on_sampled doc h_sampled
  simp [DSLSetup.applyDocument, h_oracle]

end DSL

end
