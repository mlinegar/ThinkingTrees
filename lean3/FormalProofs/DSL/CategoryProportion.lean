/-
# FormalProofs/DSL/CategoryProportion.lean

## Paper Reference: Section 3.1, Equation 5

This file formalizes DSL for estimating category proportions:
- DSL proportion estimator
- Standard errors
- Comparison with naive approach

### Main Formula (Equation 5)

  μ̂_DSL = (1/N)∑Ŷᵢ - ((1/n)∑_{Rᵢ=1}Ŷᵢ - (1/n)∑_{Rᵢ=1}Yᵢ)

Equivalently:
  μ̂_DSL = (1/N)∑Ỹᵢ

where Ỹᵢ = Ŷᵢ - (Rᵢ/πᵢ)(Ŷᵢ - Yᵢ)

### Example Application

Estimating the proportion of political ads that are "attacking":
- N = 100,000 political ads
- LLM predicts "attacking" status for all
- n = 1,000 ads expert-coded
- DSL gives valid estimate and CI for true proportion
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
## Category Proportion Data
-/

/-- Data for estimating a category proportion -/
structure ProportionData (n : ℕ) where
  /-- Predicted category indicators Ŷ ∈ {0, 1} -/
  Y_pred : Fin n → ℝ
  /-- True category indicators Y ∈ {0, 1} (for sampled units) -/
  Y_true : Fin n → ℝ
  /-- Sampling indicators R -/
  R : Fin n → SamplingIndicator
  /-- Sampling probabilities π -/
  π : Fin n → ℝ
  /-- Positivity condition -/
  h_π_pos : ∀ i, π i > 0
  /-- Binary predictions -/
  h_pred_binary : ∀ i, Y_pred i = 0 ∨ Y_pred i = 1
  /-- Binary true values (when sampled) -/
  h_true_binary : ∀ i, R i = true → (Y_true i = 0 ∨ Y_true i = 1)

/-!
## DSL Proportion Estimator (Equation 5)
-/

/-- Design-adjusted category indicators -/
def Y_tilde_proportion {n : ℕ} (data : ProportionData n) : Fin n → ℝ :=
  fun i => designAdjustedOutcome (data.Y_pred i) (data.Y_true i) (data.R i) (data.π i)

/-- DSL proportion estimator: μ̂ = (1/N)∑Ỹᵢ -/
def DSLProportionEstimator {n : ℕ} (data : ProportionData n) : ℝ :=
  (∑ i : Fin n, Y_tilde_proportion data i) / n

/-- Alternative form (Equation 5):
    μ̂ = mean(Ŷ) - (mean(Ŷ | R=1) - mean(Y | R=1)) -/
def DSLProportionEstimatorAlt {n : ℕ} (data : ProportionData n) : ℝ :=
  let N : ℝ := n
  let mean_pred := (∑ i : Fin n, data.Y_pred i) / N
  -- Count sampled units
  let n_sampled := (Finset.univ.filter (fun i => data.R i = true)).card
  -- Mean of predictions for sampled
  let mean_pred_sampled := (∑ i ∈ Finset.univ.filter (fun i => data.R i = true),
    data.Y_pred i) / n_sampled
  -- Mean of true values for sampled
  let mean_true_sampled := (∑ i ∈ Finset.univ.filter (fun i => data.R i = true),
    data.Y_true i) / n_sampled
  -- DSL correction
  mean_pred - (mean_pred_sampled - mean_true_sampled)

/-- The two forms are equivalent when π matches the realized sampling fraction. -/
theorem proportion_estimators_equivalent {n : ℕ}
    (data : ProportionData n)
    (h_pi : ∀ i,
      data.π i =
        ((Finset.univ.filter (fun j => data.R j = true)).card : ℝ) / n)
    (h_sampled_pos :
      (Finset.univ.filter (fun j => data.R j = true)).card > 0)
    (h_n_pos : n > 0)
    : DSLProportionEstimator data = DSLProportionEstimatorAlt data := by
  classical
  let sampled : Finset (Fin n) := Finset.univ.filter (fun j => data.R j = true)
  have h_sampled_pos' : (sampled.card : ℝ) ≠ 0 := by
    exact_mod_cast (ne_of_gt h_sampled_pos)
  have h_n_pos' : (n : ℝ) ≠ 0 := by
    exact_mod_cast (ne_of_gt h_n_pos)
  have h_pi_const : ∀ i, data.π i = (sampled.card : ℝ) / n := by
    intro i
    simpa [sampled] using h_pi i
  have h_pi_inv : ∀ i, (1 / data.π i) = (n : ℝ) / sampled.card := by
    intro i
    calc
      (1 / data.π i) = 1 / ((sampled.card : ℝ) / n) := by
        simp [h_pi_const i]
      _ = (n : ℝ) / sampled.card := by
        field_simp [h_sampled_pos', h_n_pos']
  have h_sum_if :
      (∑ i : Fin n,
          (if data.R i = true then (1 / data.π i) * (data.Y_pred i - data.Y_true i) else 0)) =
        Finset.sum sampled
          (fun i => (1 / data.π i) * (data.Y_pred i - data.Y_true i)) := by
    classical
    -- Replace `if` with a filtered sum.
    let f : Fin n → ℝ :=
      fun i => (1 / data.π i) * (data.Y_pred i - data.Y_true i)
    have h_if_sampled :
        Finset.sum sampled (fun i => if data.R i = true then f i else 0) =
          Finset.sum sampled f := by
      refine Finset.sum_congr rfl ?_
      intro i hi
      have hR : data.R i = true := (Finset.mem_filter.mp hi).2
      simp [f, hR]
    have h_subset : sampled ⊆ (Finset.univ : Finset (Fin n)) := by
      intro i _; exact Finset.mem_univ i
    have h_zero :
        ∀ i ∈ (Finset.univ : Finset (Fin n)), i ∉ sampled →
          (if data.R i = true then f i else 0) = 0 := by
      intro i _ hi
      have h_not : data.R i ≠ true := by
        intro hR
        exact hi (by simp [sampled, hR])
      cases hRi : data.R i <;> simp [hRi] at h_not ⊢
    have h_sum_subset :
        Finset.sum sampled (fun i => if data.R i = true then f i else 0) =
          Finset.sum (Finset.univ : Finset (Fin n)) (fun i => if data.R i = true then f i else 0) :=
      Finset.sum_subset h_subset h_zero
    calc
      (∑ i : Fin n, if data.R i = true then f i else 0) =
          Finset.sum sampled (fun i => if data.R i = true then f i else 0) := by
            simpa using h_sum_subset.symm
      _ = Finset.sum sampled f := h_if_sampled
  have h_sum_scaled :
      Finset.sum sampled (fun i => (1 / data.π i) * (data.Y_pred i - data.Y_true i)) =
        ((n : ℝ) / sampled.card) *
          (Finset.sum sampled (fun i => data.Y_pred i) -
            Finset.sum sampled (fun i => data.Y_true i)) := by
    calc
      Finset.sum sampled (fun i => (1 / data.π i) * (data.Y_pred i - data.Y_true i)) =
          Finset.sum sampled (fun i => ((n : ℝ) / sampled.card) * (data.Y_pred i - data.Y_true i)) := by
            refine Finset.sum_congr rfl ?_
            intro i _hi
            simp [h_pi_inv i]
      _ = ((n : ℝ) / sampled.card) *
            (Finset.sum sampled (fun i => data.Y_pred i) -
              Finset.sum sampled (fun i => data.Y_true i)) := by
            have h_mul :
                Finset.sum sampled
                    (fun i => ((n : ℝ) / sampled.card) * (data.Y_pred i - data.Y_true i)) =
                  ((n : ℝ) / sampled.card) *
                    Finset.sum sampled (fun i => data.Y_pred i - data.Y_true i) := by
                  simpa using
                    (Finset.mul_sum sampled (fun i => data.Y_pred i - data.Y_true i)
                      ((n : ℝ) / sampled.card)).symm
            -- Rewrite the inner sum of differences.
            simp [h_mul, Finset.sum_sub_distrib]
  -- Combine the pieces.
  have h_dsl :
      DSLProportionEstimator data =
        (∑ i : Fin n, data.Y_pred i) / n -
          (Finset.sum sampled (fun i => data.Y_pred i) -
            Finset.sum sampled (fun i => data.Y_true i)) / sampled.card := by
    unfold DSLProportionEstimator Y_tilde_proportion
    calc
      (∑ i : Fin n,
          (data.Y_pred i -
            (if data.R i = true then (1 / data.π i) * (data.Y_pred i - data.Y_true i) else 0))) / n =
          ((∑ i : Fin n, data.Y_pred i) -
            ∑ i : Fin n,
              (if data.R i = true then (1 / data.π i) * (data.Y_pred i - data.Y_true i) else 0)) / n := by
            simp [Finset.sum_sub_distrib]
      _ = (∑ i : Fin n, data.Y_pred i) / n -
          (∑ i : Fin n,
            (if data.R i = true then (1 / data.π i) * (data.Y_pred i - data.Y_true i) else 0)) / n := by
            simp [sub_div]
      _ = (∑ i : Fin n, data.Y_pred i) / n -
          (Finset.sum sampled
            (fun i => (1 / data.π i) * (data.Y_pred i - data.Y_true i))) / n := by
            rw [h_sum_if]
      _ = (∑ i : Fin n, data.Y_pred i) / n -
          (((n : ℝ) / sampled.card) *
            (Finset.sum sampled (fun i => data.Y_pred i) -
              Finset.sum sampled (fun i => data.Y_true i))) / n := by
            rw [h_sum_scaled]
      _ = (∑ i : Fin n, data.Y_pred i) / n -
          (Finset.sum sampled (fun i => data.Y_pred i) -
            Finset.sum sampled (fun i => data.Y_true i)) / sampled.card := by
            have h_cancel :
                ((n : ℝ) / sampled.card) / n = (1 : ℝ) / sampled.card := by
              field_simp [h_sampled_pos', h_n_pos']
            set S : ℝ :=
              Finset.sum sampled (fun i => data.Y_pred i) -
                Finset.sum sampled (fun i => data.Y_true i)
            have h_factor : (((n : ℝ) / sampled.card) * S) / n = S / sampled.card := by
              calc
                (((n : ℝ) / sampled.card) * S) / n =
                    S * (((n : ℝ) / sampled.card) / n) := by ring
                _ = S * ((1 : ℝ) / sampled.card) := by simp [h_cancel]
                _ = S / sampled.card := by
                      simp [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc]
            simp [S, h_factor]
  have h_alt :
      DSLProportionEstimatorAlt data =
        (∑ i : Fin n, data.Y_pred i) / n -
          (Finset.sum sampled (fun i => data.Y_pred i) -
            Finset.sum sampled (fun i => data.Y_true i)) / sampled.card := by
    unfold DSLProportionEstimatorAlt
    -- Simplify to a pure algebraic identity.
    simp [sampled, sub_eq_add_neg, add_comm, add_left_comm, add_assoc, sub_div,
      Finset.sum_sub_distrib]
    ring
  exact h_dsl.trans h_alt.symm

/-!
## Standard Error
-/

/-- Variance of DSL proportion estimator.

    Var(μ̂) = (1/N²)[∑Var(Ỹᵢ) + 2∑_{i<j}Cov(Ỹᵢ, Ỹⱼ)]

    Under simple random sampling, this simplifies. -/
def proportionVariance {n : ℕ}
    (data : ProportionData n)
    (σ_Y_sq : ℝ)        -- Var(Y)
    (σ_pred_sq : ℝ)     -- Var(Ŷ - Y)
    (π : ℝ)             -- Uniform sampling probability
    : ℝ :=
  -- Simplified formula under SRS
  σ_Y_sq / n + (1/π - 1) * σ_pred_sq / n

/-- Standard error for proportion -/
def proportionSE {n : ℕ}
    (data : ProportionData n)
    (μ_hat : ℝ)
    (π : ℝ)
    (n_sampled : ℕ)
    (σ_pred_sq_est : ℝ)
    : ℝ :=
  -- Estimated variance components
  let σ_Y_sq := μ_hat * (1 - μ_hat)  -- Bernoulli variance
  -- Caller-provided estimate of prediction-error variance from sampled labels.
  let σ_pred_sq := σ_pred_sq_est
  Real.sqrt (proportionVariance data σ_Y_sq σ_pred_sq π)

/-- 95% confidence interval for proportion -/
def proportionCI95 {n : ℕ}
    (μ_hat : ℝ)
    (SE : ℝ) : ℝ × ℝ :=
  (μ_hat - 1.96 * SE, μ_hat + 1.96 * SE)

/-!
## Comparison with Naive Approach
-/

/-- Naive proportion estimator: just use predicted labels.
    μ̂_naive = (1/N)∑Ŷᵢ -/
def naiveProportionEstimator {n : ℕ} (data : ProportionData n) : ℝ :=
  (∑ i : Fin n, data.Y_pred i) / n

/-- Bias of naive estimator.
    Bias = E[μ̂_naive] - μ = FP_rate - FN_rate -/
def naiveBias
    (FP_rate : ℝ)  -- P(Ŷ = 1 | Y = 0)
    (FN_rate : ℝ)  -- P(Ŷ = 0 | Y = 1)
    : ℝ :=
  FP_rate - FN_rate

/-- Example: Naive estimator is biased with asymmetric errors -/
theorem naive_proportion_biased
    (FP_rate FN_rate : ℝ)
    (h_asymmetric : FP_rate ≠ FN_rate)
    : naiveBias FP_rate FN_rate ≠ 0 := by
  unfold naiveBias
  intro h
  apply h_asymmetric
  linarith

/-!
## Multi-Category Proportions
-/

/-- Data for estimating proportions across K categories -/
structure MultiCategoryData (n K : ℕ) where
  /-- Predicted category (one-hot encoded) -/
  Y_pred : Fin n → Fin K → ℝ
  /-- True category (one-hot) -/
  Y_true : Fin n → Fin K → ℝ
  /-- Sampling indicators -/
  R : Fin n → SamplingIndicator
  /-- Sampling probabilities -/
  π : Fin n → ℝ
  h_π_pos : ∀ i, π i > 0

/-- DSL estimator for each category proportion -/
def DSLMultiCategoryEstimator {n K : ℕ}
    (data : MultiCategoryData n K) : Fin K → ℝ :=
  fun k =>
    (∑ i : Fin n, designAdjustedOutcome
      (data.Y_pred i k) (data.Y_true i k) (data.R i) (data.π i)) / n

/-- Proportions sum to 1 (approximately, due to design adjustment) -/
theorem multi_category_sum_one {n K : ℕ}
    (data : MultiCategoryData n K)
    (h_pred_sum : ∀ i, (∑ k : Fin K, data.Y_pred i k) = 1)
    (h_true_sum : ∀ i, data.R i = true → (∑ k : Fin K, data.Y_true i k) = 1)
    (h_n_pos : n > 0)
    : (∑ k : Fin K, DSLMultiCategoryEstimator data k) = 1 := by
  have h_each : ∀ i : Fin n,
      (∑ k : Fin K, designAdjustedOutcome
        (data.Y_pred i k) (data.Y_true i k) (data.R i) (data.π i)) = 1 := by
    intro i
    cases hR : data.R i with
    | false =>
        simp [designAdjustedOutcome, hR, h_pred_sum i]
    | true =>
        have h_true := h_true_sum i hR
        have h_mul :
            (∑ k : Fin K,
              (1 / data.π i) * (data.Y_pred i k - data.Y_true i k)) =
              (1 / data.π i) *
                (∑ k : Fin K, (data.Y_pred i k - data.Y_true i k)) := by
          symm
          simpa using
            (Finset.mul_sum (s := Finset.univ)
              (f := fun k : Fin K => data.Y_pred i k - data.Y_true i k)
              (a := (1 / data.π i)))
        have h_diff :
            (∑ k : Fin K, (data.Y_pred i k - data.Y_true i k)) =
              (∑ k : Fin K, data.Y_pred i k) - (∑ k : Fin K, data.Y_true i k) := by
          simp [Finset.sum_sub_distrib]
        have h_scaled :
            (∑ k : Fin K,
              (data.π i)⁻¹ * (data.Y_pred i k - data.Y_true i k)) =
              (data.π i)⁻¹ *
                ((∑ k : Fin K, data.Y_pred i k) - (∑ k : Fin K, data.Y_true i k)) := by
          have h_mul' :
              (∑ k : Fin K,
                (data.π i)⁻¹ * (data.Y_pred i k - data.Y_true i k)) =
                (data.π i)⁻¹ *
                  (∑ k : Fin K, (data.Y_pred i k - data.Y_true i k)) := by
            simpa [one_div] using h_mul
          calc
            (∑ k : Fin K,
              (data.π i)⁻¹ * (data.Y_pred i k - data.Y_true i k)) =
                (data.π i)⁻¹ *
                  (∑ k : Fin K, (data.Y_pred i k - data.Y_true i k)) := h_mul'
            _ = (data.π i)⁻¹ *
                  ((∑ k : Fin K, data.Y_pred i k) - (∑ k : Fin K, data.Y_true i k)) := by
                simp [h_diff]
        calc
          ∑ k : Fin K,
              designAdjustedOutcome
                (data.Y_pred i k) (data.Y_true i k) true (data.π i)
              =
              ∑ k : Fin K,
                (data.Y_pred i k -
                  (1 / data.π i) * (data.Y_pred i k - data.Y_true i k)) := by
                    simp [designAdjustedOutcome]
          _ =
              (∑ k : Fin K, data.Y_pred i k) -
                (1 / data.π i) * (∑ k : Fin K, (data.Y_pred i k - data.Y_true i k)) := by
                    calc
                      (∑ k : Fin K,
                        (data.Y_pred i k -
                          (1 / data.π i) * (data.Y_pred i k - data.Y_true i k))) =
                          (∑ k : Fin K, data.Y_pred i k) -
                            (∑ k : Fin K,
                              (1 / data.π i) * (data.Y_pred i k - data.Y_true i k)) := by
                          simp [Finset.sum_sub_distrib]
                      _ = (∑ k : Fin K, data.Y_pred i k) -
                            (1 / data.π i) *
                              (∑ k : Fin K, (data.Y_pred i k - data.Y_true i k)) := by
                          -- Rewrite using the scaled-sum identity.
                          simpa [one_div] using h_scaled
          _ = 1 := by
                simp [h_diff, h_pred_sum i, h_true]
  have h_n_pos' : (0 : ℝ) < n := Nat.cast_pos.mpr h_n_pos
  calc
    ∑ k : Fin K, DSLMultiCategoryEstimator data k
        = (∑ k : Fin K,
            ∑ i : Fin n,
              designAdjustedOutcome
                (data.Y_pred i k) (data.Y_true i k) (data.R i) (data.π i)) / n := by
          simp [DSLMultiCategoryEstimator, Finset.sum_div]
    _ = (∑ i : Fin n,
            ∑ k : Fin K,
              designAdjustedOutcome
                (data.Y_pred i k) (data.Y_true i k) (data.R i) (data.π i)) / n := by
          rw [Finset.sum_comm]
    _ = (∑ i : Fin n, 1) / n := by
          simp [h_each]
    _ = (n : ℝ) / n := by
          simp
    _ = 1 := by
          field_simp [h_n_pos'.ne']

/-!
## Example: Political Ad Classification
-/

/-- Example data structure for political ad study -/
structure PoliticalAdData (n : ℕ) where
  /-- LLM prediction: 1 = attacking, 0 = not attacking -/
  pred_attacking : Fin n → ℝ
  /-- Expert label (when available) -/
  true_attacking : Fin n → ℝ
  /-- Whether expert-coded -/
  sampled : Fin n → SamplingIndicator
  /-- Sampling probability (e.g., 0.01 for 1% sample) -/
  sample_prob : ℝ
  h_prob_pos : sample_prob > 0
  /-- Predicted labels are binary -/
  h_pred_binary : ∀ i, pred_attacking i = 0 ∨ pred_attacking i = 1
  /-- True labels are binary when sampled -/
  h_true_binary : ∀ i, sampled i = true → (true_attacking i = 0 ∨ true_attacking i = 1)

/-- Convert to ProportionData -/
def PoliticalAdData.toProportionData {n : ℕ} (data : PoliticalAdData n) :
    ProportionData n where
  Y_pred := data.pred_attacking
  Y_true := data.true_attacking
  R := data.sampled
  π := fun _ => data.sample_prob
  h_π_pos := fun _ => data.h_prob_pos
  h_pred_binary := fun i => data.h_pred_binary i
  h_true_binary := fun i h => data.h_true_binary i h

/-- Estimate proportion of attacking ads -/
def estimateAttackingProportion {n : ℕ} (data : PoliticalAdData n) : ℝ :=
  DSLProportionEstimator data.toProportionData

/-!
## Properties
-/

/-- Moment function for proportions (d = 1). -/
def proportionMomentFn {Obs : Type*} : MomentFunction (Obs × ℝ) 1 :=
  fun D β => fun _ => proportionMoment D.2 (β ⟨0, by decide⟩)

/-- DSL proportion estimator is consistent under M-estimation conditions. -/
theorem DSL_proportion_unbiased_from_assumptions
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Con : Type*}
    (E : ((Obs × ℝ × ℝ × SamplingIndicator × ℝ) → Fin 1 → ℝ) → Fin 1 → ℝ)
    (h_consistent : MEstimatorConsistencyAssumption μ 1 E)
    (dbs : DesignBasedSampling Obs ℝ Con)
    (β_star : Fin 1 → ℝ)  -- True proportion parameter
    (reg : RegularityConditions (Obs × ℝ × ℝ × SamplingIndicator × ℝ) 1)
    (h_unbiased : MomentUnbiased (DSLMomentFromData (proportionMomentFn (Obs := Obs)))
      E β_star)
    (data_seq : ℕ → Ω → List (Obs × ℝ × ℝ × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin 1 → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData (proportionMomentFn (Obs := Obs)))
      data_seq β_hat_seq)
    : ConvergesInProbability μ β_hat_seq (fun _ => β_star) := by
  exact DSL_consistent_from_assumptions μ E h_consistent dbs (proportionMomentFn (Obs := Obs))
    β_star reg h_unbiased data_seq β_hat_seq h_est

/-- DSL proportion estimator is consistent under M-estimation conditions. -/
theorem DSL_proportion_unbiased
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Con : Type*}
    (axioms : MEstimationAxioms Ω (Obs × ℝ × ℝ × SamplingIndicator × ℝ) μ 1)
    (dbs : DesignBasedSampling Obs ℝ Con)
    (β_star : Fin 1 → ℝ)  -- True proportion parameter
    (reg : RegularityConditions (Obs × ℝ × ℝ × SamplingIndicator × ℝ) 1)
    (h_unbiased : MomentUnbiased (DSLMomentFromData (proportionMomentFn (Obs := Obs)))
      axioms.E β_star)
    (data_seq : ℕ → Ω → List (Obs × ℝ × ℝ × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin 1 → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData (proportionMomentFn (Obs := Obs)))
      data_seq β_hat_seq)
    : ConvergesInProbability μ β_hat_seq (fun _ => β_star) := by
  exact DSL_proportion_unbiased_from_assumptions μ axioms.E
    (mEstimatorConsistency_of_axioms μ 1 axioms)
    dbs β_star reg h_unbiased data_seq β_hat_seq h_est

/-- DSL proportion CI has valid coverage.
    P(μ ∈ CI) → 0.95 as N → ∞ -/
theorem DSL_proportion_valid_coverage_from_assumptions
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Con : Type*}
    (E : ((Obs × ℝ × ℝ × SamplingIndicator × ℝ) → Fin 1 → ℝ) → Fin 1 → ℝ)
    (h_normal : MEstimatorAsymptoticNormalAssumption μ 1 E)
    (coverage_axioms : CoverageFromAsymptoticNormal μ 1)
    (dbs : DesignBasedSampling Obs ℝ Con)
    (β_star : Fin 1 → ℝ)
    (V : Matrix (Fin 1) (Fin 1) ℝ)
    (reg : RegularityConditions (Obs × ℝ × ℝ × SamplingIndicator × ℝ) 1)
    (h_unbiased : MomentUnbiased (DSLMomentFromData (proportionMomentFn (Obs := Obs)))
      E β_star)
    (CI_seq : ℕ → Ω → Fin 1 → ℝ × ℝ)
    (α : ℝ) (h_α : 0 < α ∧ α < 1)
    (centered_scaled_seq : ℕ → Ω → Fin 1 → ℝ)
    : AsymptoticCoverage μ CI_seq β_star α := by
  exact DSL_valid_coverage_from_assumptions μ E h_normal coverage_axioms dbs
    (proportionMomentFn (Obs := Obs)) β_star V reg h_unbiased CI_seq α h_α centered_scaled_seq

/-- DSL proportion CI has valid coverage.
    P(μ ∈ CI) → 0.95 as N → ∞ -/
theorem DSL_proportion_valid_coverage
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Con : Type*}
    (axioms : MEstimationAxioms Ω (Obs × ℝ × ℝ × SamplingIndicator × ℝ) μ 1)
    (coverage_axioms : CoverageAxioms μ 1)
    (dbs : DesignBasedSampling Obs ℝ Con)
    (β_star : Fin 1 → ℝ)
    (V : Matrix (Fin 1) (Fin 1) ℝ)
    (reg : RegularityConditions (Obs × ℝ × ℝ × SamplingIndicator × ℝ) 1)
    (h_unbiased : MomentUnbiased (DSLMomentFromData (proportionMomentFn (Obs := Obs)))
      axioms.E β_star)
    (CI_seq : ℕ → Ω → Fin 1 → ℝ × ℝ)
    (α : ℝ) (h_α : 0 < α ∧ α < 1)
    (centered_scaled_seq : ℕ → Ω → Fin 1 → ℝ)
    : AsymptoticCoverage μ CI_seq β_star α := by
  exact DSL_proportion_valid_coverage_from_assumptions μ axioms.E
    (mEstimatorAsymptoticNormal_of_axioms μ 1 axioms)
    coverage_axioms dbs β_star V reg h_unbiased CI_seq α h_α centered_scaled_seq

end DSL

end
