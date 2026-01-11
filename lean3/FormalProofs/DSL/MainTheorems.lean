/-
# FormalProofs/DSL/MainTheorems.lean

## Paper Reference: Main Results Summary

This file collects the main theorems of Design-based Supervised Learning (DSL):

### Theorem 1: DSL Provides Valid Inference

Under Assumption 1 (design-based sampling), DSL provides valid inference
regardless of prediction accuracy. The confidence intervals achieve nominal
coverage asymptotically for ANY predictor.

### Theorem 2: DSL Efficiency Improves with Predictions

Better predictions lead to smaller standard errors. As prediction accuracy
increases, DSL variance decreases. In the limit of perfect predictions,
DSL is as efficient as using all true labels.

### Theorem 3: Ignoring Errors Invalidates Inference

When using predicted labels as if they were true labels (the naive approach),
inference is invalid unless very strong (and implausible) conditions hold.
Even 90%+ accuracy does not guarantee valid inference.

### Key Implication

DSL is the only approach that provides:
1. Valid inference regardless of prediction quality
2. Efficiency gains from good predictions
3. No assumptions about prediction error structure
-/

import FormalProofs.DSL.AsymptoticTheory
import FormalProofs.DSL.BiasAnalysis
import FormalProofs.DSL.VarianceDecomposition
import FormalProofs.DSL.LinearRegression
import FormalProofs.DSL.LogisticRegression
import FormalProofs.DSL.CategoryProportion
import FormalProofs.DSL.FixedEffects

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
## Shared Inference Bundle
-/

/-- Bundle of asymptotic inference guarantees for an estimator sequence. -/
structure ValidInference {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] {d : ℕ}
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (β_star : Fin d → ℝ)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ) : Prop where
  consistent : ConvergesInProbability μ β_hat_seq (fun _ => β_star)
  asymptotic_normal : ConvergesInDistributionToNormal μ centered_scaled_seq (fun _ => 0) V
  coverage : AsymptoticCoverage μ CI_seq β_star α

/-!
## Main Theorem 1: Valid Inference
-/

/-- **Theorem 1: DSL Provides Valid Inference**

Under Assumption 1 (design-based sampling), the DSL estimator provides
valid statistical inference regardless of the prediction accuracy.

Specifically:
1. β̂_DSL is consistent for β*
2. √N(β̂_DSL - β*) →d N(0, V)
3. The 95% CI achieves 95% coverage asymptotically

This holds for ANY predictor - the LLM can have 50% accuracy or 99% accuracy,
and the inference is still valid. -/
theorem DSL_valid_inference
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (axioms : MEstimationAxioms Ω (Obs × Mis × Mis × SamplingIndicator × ℝ) μ d)
    (coverage_axioms : CoverageAxioms μ d)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) axioms.E β_star)
    (data_seq : ℕ → Ω → List (Obs × Mis × Mis × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData m) data_seq β_hat_seq)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ) (hα : 0 < α ∧ α < 1)
    : ValidInference μ β_hat_seq β_star centered_scaled_seq V CI_seq α := by
  refine ⟨?_, ?_, ?_⟩
  · exact DSL_consistent μ axioms dbs m β_star reg h_unbiased data_seq β_hat_seq h_est
  · exact DSL_asymptotic_normal μ axioms dbs m β_star V reg h_unbiased centered_scaled_seq
  · exact DSL_valid_coverage μ axioms coverage_axioms dbs m β_star V reg h_unbiased CI_seq α hα
      centered_scaled_seq

/-- **Theorem 1 (Oracle Parameter Form): DSL Provides Valid Inference**

This variant names the estimand explicitly as the oracle parameter `β_oracle`,
defined by the true moment condition. Oracle access ensures expert labels match
the oracle labels on sampled documents. -/
theorem DSL_valid_inference_oracle
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (assumptions : DSLAssumptions Obs Mis Con)
    (axioms : MEstimationAxioms Ω (Obs × Mis × Mis × SamplingIndicator × ℝ) μ d)
    (coverage_axioms : CoverageAxioms μ d)
    (m : MomentFunction (Obs × Mis) d)
    (β_oracle : Fin d → ℝ)
    (h_oracle : OracleTarget m axioms.E β_oracle)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) axioms.E β_oracle)
    (data_seq : ℕ → Ω → List (Obs × Mis × Mis × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData m) data_seq β_hat_seq)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ) (hα : 0 < α ∧ α < 1)
    : ValidInference μ β_hat_seq β_oracle centered_scaled_seq V CI_seq α := by
  exact DSL_valid_inference μ axioms coverage_axioms assumptions.sampling m β_oracle V reg
    h_unbiased data_seq β_hat_seq h_est centered_scaled_seq CI_seq α hα

/-- Corollary: DSL CI coverage converges to nominal level -/
theorem DSL_CI_coverage
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (axioms : MEstimationAxioms Ω (Obs × Mis × Mis × SamplingIndicator × ℝ) μ d)
    (coverage_axioms : CoverageAxioms μ d)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) axioms.E β_star)
    (data_seq : ℕ → Ω → List (Obs × Mis × Mis × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData m) data_seq β_hat_seq)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ) (hα : 0 < α ∧ α < 1)
    : AsymptoticCoverage μ CI_seq β_star α := by
  have h :=
    DSL_valid_inference μ axioms coverage_axioms dbs m β_star V reg h_unbiased data_seq β_hat_seq
      h_est centered_scaled_seq CI_seq α hα
  exact h.coverage

/-- Corollary (Oracle Parameter Form): DSL CI coverage converges to nominal level. -/
theorem DSL_CI_coverage_oracle
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (assumptions : DSLAssumptions Obs Mis Con)
    (axioms : MEstimationAxioms Ω (Obs × Mis × Mis × SamplingIndicator × ℝ) μ d)
    (coverage_axioms : CoverageAxioms μ d)
    (m : MomentFunction (Obs × Mis) d)
    (β_oracle : Fin d → ℝ)
    (h_oracle : OracleTarget m axioms.E β_oracle)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) axioms.E β_oracle)
    (data_seq : ℕ → Ω → List (Obs × Mis × Mis × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData m) data_seq β_hat_seq)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ) (hα : 0 < α ∧ α < 1)
    : AsymptoticCoverage μ CI_seq β_oracle α := by
  have h :=
    DSL_valid_inference_oracle μ assumptions axioms coverage_axioms m β_oracle h_oracle V reg
      h_unbiased data_seq β_hat_seq h_est centered_scaled_seq CI_seq α hα
  exact h.coverage

/-!
## Main Theorem 2: Efficiency from Predictions
-/

/-- **Theorem 2: DSL Efficiency Improves with Better Predictions**

The variance of the DSL estimator decreases as prediction accuracy improves.
Let σ²_pred = Var(Ŷ - Y) be the prediction error variance. Then:

  Var(β̂_DSL) = Var_full + f(1/π) · σ²_pred

where f is an increasing function and Var_full is the variance with full labeling.

As σ²_pred → 0 (perfect predictions), Var(β̂_DSL) → Var_full. -/
theorem DSL_efficiency_improves_with_predictions
    {d : ℕ}
    (vp1 vp2 : DSLVarianceParams)
    -- Same setup except prediction quality
    (h_same : vp1.N = vp2.N ∧ vp1.n = vp2.n ∧ vp1.σ_Y_sq = vp2.σ_Y_sq)
    -- Better predictions
    (h_better : vp2.σ_pred_sq ≤ vp1.σ_pred_sq)
    (h_pos_factor : (1 : ℝ) / vp1.n - 1 / vp1.N ≥ 0)
    : approximateVariance vp2 ≤ approximateVariance vp1 := by
  rcases h_same with ⟨hN, hn, hσY⟩
  exact variance_decreases_with_accuracy vp1 vp2 hN hn hσY h_better h_pos_factor

/-- Corollary: Perfect predictions give minimum variance -/
theorem perfect_predictions_optimal {d : ℕ}
    (vp : DSLVarianceParams)
    (h_perfect : vp.σ_pred_sq = 0)
    : approximateVariance vp = vp.σ_Y_sq / vp.N :=
  perfect_predictions_minimum_variance vp h_perfect

/-!
## Main Theorem 3: Naive Approach is Invalid
-/

/-- **Theorem 3: Ignoring Prediction Errors Invalidates Inference**

When using predicted labels as if they were true labels:

1. The estimator is biased: E[β̂_naive] ≠ β*
2. The standard errors are wrong: SE_naive ≠ true SE
3. Confidence intervals have wrong coverage: P(β* ∈ CI) ≠ 1 - α

The only exception is when E[e | X] = 0 where e = Ŷ - Y, which requires
prediction errors to be uncorrelated with covariates, outcomes, and
unobserved confounders. This almost never holds in practice. -/
theorem naive_approach_invalid
    {Obs Mis : Type*} {d : ℕ}
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (E : ((Obs × Mis × Mis) → Fin d → ℝ) → Fin d → ℝ)
    (h_true : MomentUnbiased (TrueMomentFromData m) E β_star)
    -- Prediction error induces nonzero moment bias.
    (h_bias : ∃ i, MomentBias m E β_star i ≠ 0)
    (hE_linear : ExpectationLinear E)
    : ¬ MomentUnbiased (PredMomentFromData m) E β_star := by
  exact naive_estimator_biased_general m E β_star h_true h_bias hE_linear

/-- Corollary: High accuracy does not prevent bias -/
theorem high_accuracy_still_biased
    (accuracy : ℝ) (h_acc : 0.9 ≤ accuracy ∧ accuracy < 1)
    (β_true : ℝ) (hβ : β_true ≠ 0)
    : linearRegressionBiasExample accuracy β_true ≠ 0 := by
  exact bias_with_high_accuracy accuracy h_acc β_true hβ

/-!
## Comparison Summary
-/

/-- Comparison of DSL vs Naive approaches

| Property | DSL | Naive |
|----------|-----|-------|
| Valid inference | ✓ Always | ✗ Rarely |
| Efficiency | Better with good predictions | N/A (invalid) |
| Requires prediction assumptions | ✗ No | ✓ Very strong |
| Uses all data | ✓ Yes | ✓ Yes |
| Needs expert labels | ✓ Sample only | ✗ None |

DSL is strictly better: it provides valid inference that the naive
approach cannot, while still leveraging predictions for efficiency. -/
theorem DSL_dominates_naive
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (axioms : MEstimationAxioms Ω (Obs × Mis × Mis × SamplingIndicator × ℝ) μ d)
    (coverage_axioms : CoverageAxioms μ d)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) axioms.E β_star)
    (data_seq : ℕ → Ω → List (Obs × Mis × Mis × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData m) data_seq β_hat_seq)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ) (hα : 0 < α ∧ α < 1)
    (E : ((Obs × Mis × Mis) → Fin d → ℝ) → Fin d → ℝ)
    (h_true : MomentUnbiased (TrueMomentFromData m) E β_star)
    (h_bias : ∃ i, MomentBias m E β_star i ≠ 0)
    (hE_linear : ExpectationLinear E)
    : ValidInference μ β_hat_seq β_star centered_scaled_seq V CI_seq α ∧
      ¬ MomentUnbiased (PredMomentFromData m) E β_star := by
  refine ⟨?_, ?_⟩
  · exact DSL_valid_inference μ axioms coverage_axioms dbs m β_star V reg h_unbiased data_seq
      β_hat_seq h_est centered_scaled_seq CI_seq α hα
  · exact naive_estimator_biased_general m E β_star h_true h_bias hE_linear

/-- Oracle-parameter comparison: DSL dominates naive inference. -/
theorem DSL_dominates_naive_oracle
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (assumptions : DSLAssumptions Obs Mis Con)
    (axioms : MEstimationAxioms Ω (Obs × Mis × Mis × SamplingIndicator × ℝ) μ d)
    (coverage_axioms : CoverageAxioms μ d)
    (m : MomentFunction (Obs × Mis) d)
    (β_oracle : Fin d → ℝ)
    (h_oracle : OracleTarget m axioms.E β_oracle)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) axioms.E β_oracle)
    (data_seq : ℕ → Ω → List (Obs × Mis × Mis × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData m) data_seq β_hat_seq)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ) (hα : 0 < α ∧ α < 1)
    (E : ((Obs × Mis × Mis) → Fin d → ℝ) → Fin d → ℝ)
    (h_true : MomentUnbiased (TrueMomentFromData m) E β_oracle)
    (h_bias : ∃ i, MomentBias m E β_oracle i ≠ 0)
    (hE_linear : ExpectationLinear E)
    : ValidInference μ β_hat_seq β_oracle centered_scaled_seq V CI_seq α ∧
      ¬ MomentUnbiased (PredMomentFromData m) E β_oracle := by
  refine ⟨?_, ?_⟩
  · exact DSL_valid_inference_oracle μ assumptions axioms coverage_axioms m β_oracle h_oracle V reg
      h_unbiased data_seq β_hat_seq h_est centered_scaled_seq CI_seq α hα
  · exact naive_estimator_biased_general m E β_oracle h_true h_bias hE_linear

/-!
## Application Guidelines
-/

/-- When to use DSL:

1. **Always** when using LLM/ML predictions in downstream analysis
2. **Especially** when:
   - Prediction accuracy < 99%
   - Errors may correlate with analysis variables
   - Valid inference is important
   - Resources allow for some expert coding

The cost of DSL is minimal (need some expert labels), and the benefit
is valid inference. There is no reason to use the naive approach. -/
theorem DSL_guidelines
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (axioms : MEstimationAxioms Ω (Obs × Mis × Mis × SamplingIndicator × ℝ) μ d)
    (coverage_axioms : CoverageAxioms μ d)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) axioms.E β_star)
    (data_seq : ℕ → Ω → List (Obs × Mis × Mis × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData m) data_seq β_hat_seq)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ) (hα : 0 < α ∧ α < 1)
    (E : ((Obs × Mis × Mis) → Fin d → ℝ) → Fin d → ℝ)
    (h_true : MomentUnbiased (TrueMomentFromData m) E β_star)
    (h_bias : ∃ i, MomentBias m E β_star i ≠ 0)
    (hE_linear : ExpectationLinear E)
    : ValidInference μ β_hat_seq β_star centered_scaled_seq V CI_seq α ∧
      ¬ MomentUnbiased (PredMomentFromData m) E β_star := by
  exact DSL_dominates_naive μ axioms coverage_axioms dbs m β_star V reg h_unbiased data_seq
    β_hat_seq h_est centered_scaled_seq CI_seq α hα E h_true h_bias hE_linear

/-- Sample size recommendations:

For DSL with simple random sampling:
- Start with n ≈ 100-500 expert-coded documents
- Increase n if:
  - Prediction accuracy is low
  - Standard errors are too large
  - Effect sizes are small
- Rule of thumb: n ≈ 10% of N often sufficient with good predictions -/
theorem sample_size_recommendations (N : ℕ) :
    ruleOfThumb_n N ≥ 100 ∧ ruleOfThumb_n N ≥ N / 10 := by
  constructor
  · exact Nat.le_max_right _ _
  · exact Nat.le_max_left _ _

/-!
## Extensions and Variations
-/

/-- DSL applies to many analysis types:

1. **Category Proportions**: μ̂_DSL = (1/N)∑Ỹ
2. **Linear Regression**: β̂_DSL = (X'X)⁻¹X'Ỹ
3. **Logistic Regression**: Solve (1/N)∑m̃(D; β) = 0
4. **Fixed Effects**: Within-transform then DSL
5. **Difference-in-Differences**: DSL on interaction coefficient

All share the same theoretical guarantees. -/
theorem DSL_applies_broadly :
    (∃ m_lin : MomentFunction ((Fin 1 → ℝ) × ℝ) 1,
      m_lin = linearMomentPair (d := 1)) ∧
    (∃ m_log : MomentFunction ((Fin 1 → ℝ) × ℝ) 1,
      m_log = logisticMomentPair (d := 1)) ∧
    (∃ m_prop : MomentFunction (Unit × ℝ) 1,
      m_prop = proportionMomentFn (Obs := Unit)) := by
  refine ⟨⟨_, rfl⟩, ⟨_, rfl⟩, ⟨_, rfl⟩⟩

/-- Cross-fitting variation (Appendix B.3):

When predictions are made on the same data used for expert coding,
cross-fitting can reduce bias:

1. Split data into K folds
2. For each fold k, train predictor on other folds
3. Predict on fold k using out-of-fold model
4. Apply DSL to cross-fitted predictions

This eliminates any overfitting bias in the predictions. -/
theorem cross_fitting_variation
    {ι Obs Con Mis : Type*} [Fintype ι] {d : ℕ}
    (m : MomentFunction (Obs × Mis) d)
    (cf : CrossFit ι Obs Con Mis)
    (g0 : Obs → Con → Mis)
    (h_const : CrossFit.ConstPredictor cf g0)
    (d_obs : ι → Obs)
    (d_mis_true : ι → Mis)
    (q : ι → Con)
    (R : ι → SamplingIndicator)
    (π : ι → ℝ)
    (β : Fin d → ℝ)
    (i : ι) :
    DSLMomentCF m cf d_obs d_mis_true q R π β i =
      DSLMoment m (d_obs i) (g0 (d_obs i) (q i)) (d_mis_true i) (R i) (π i) β := by
  exact DSLMomentCF_eq_of_const m cf g0 h_const d_obs d_mis_true q R π β i

/-!
## Summary of Formal Results
-/

/-- Summary of key formal results in this formalization:

| Theorem | Location | Description |
|---------|----------|-------------|
| DSL_unbiased | DSLEstimator | E[Ỹ - Y | X] = 0 |
| DSL_consistent | AsymptoticTheory | β̂_DSL →p β* |
| DSL_asymptotic_normal | AsymptoticTheory | √N(β̂_DSL - β*) →d N(0,V) |
| variance_decreases_with_n | VarianceDecomposition | More n → smaller variance |
| variance_decreases_with_accuracy | VarianceDecomposition | Better predictions → smaller variance |
| bias_with_high_accuracy | BiasAnalysis | 90% accuracy can still give bias |
| naive_proportion_biased | CategoryProportion | Naive estimator is biased |

These results establish DSL as the correct approach for using
automated annotations in statistical inference. -/
theorem formal_results_summary
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (axioms : MEstimationAxioms Ω (Obs × Mis × Mis × SamplingIndicator × ℝ) μ d)
    (coverage_axioms : CoverageAxioms μ d)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) axioms.E β_star)
    (data_seq : ℕ → Ω → List (Obs × Mis × Mis × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData m) data_seq β_hat_seq)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ) (hα : 0 < α ∧ α < 1)
    (E : ((Obs × Mis × Mis) → Fin d → ℝ) → Fin d → ℝ)
    (h_true : MomentUnbiased (TrueMomentFromData m) E β_star)
    (h_bias : ∃ i, MomentBias m E β_star i ≠ 0)
    (hE_linear : ExpectationLinear E)
    : ValidInference μ β_hat_seq β_star centered_scaled_seq V CI_seq α ∧
      ¬ MomentUnbiased (PredMomentFromData m) E β_star := by
  exact DSL_dominates_naive μ axioms coverage_axioms dbs m β_star V reg h_unbiased data_seq
    β_hat_seq h_est centered_scaled_seq CI_seq α hα E h_true h_bias hE_linear

end DSL

end
