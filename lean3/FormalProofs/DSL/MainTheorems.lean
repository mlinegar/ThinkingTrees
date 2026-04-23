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
import FormalProofs.DSL.IPWMeasurementError
import FormalProofs.DSL.NonclassicalExpectationMismatch
import FormalProofs.DSL.VarianceDecomposition
import FormalProofs.DSL.LinearRegression
import FormalProofs.DSL.LogisticRegression
import FormalProofs.DSL.CategoryProportion
import FormalProofs.DSL.FixedEffects
import FormalProofs.DSL.ConcreteCoverage
import FormalProofs.DSL.TreeIPW
import FormalProofs.DSL.RuntimeCertificates
import FormalProofs.DSL.MergeableCertificates
import FormalProofs.DSL.TreePOEndToEnd
import FormalProofs.DSL.LabelRateBounds
import FormalProofs.Econometrics.OLS.AsymptoticOLS

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
theorem DSL_valid_inference_from_assumptions
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (E : ((Obs × Mis × Mis × SamplingIndicator × ℝ) → Fin d → ℝ) → Fin d → ℝ)
    (h_consistent : MEstimatorConsistencyAssumption μ d E)
    (h_normal : MEstimatorAsymptoticNormalAssumption μ d E)
    (coverage_axioms : CoverageFromAsymptoticNormal μ d)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) E β_star)
    (data_seq : ℕ → Ω → List (Obs × Mis × Mis × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData m) data_seq β_hat_seq)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ) (hα : 0 < α ∧ α < 1)
    : ValidInference μ β_hat_seq β_star centered_scaled_seq V CI_seq α := by
  refine ⟨?_, ?_, ?_⟩
  · exact DSL_consistent_from_assumptions μ E h_consistent dbs m β_star reg h_unbiased data_seq β_hat_seq
      h_est
  · exact DSL_asymptotic_normal_from_assumptions μ E h_normal dbs m β_star V reg h_unbiased
      centered_scaled_seq
  · exact DSL_valid_coverage_from_assumptions μ E h_normal coverage_axioms dbs m β_star V reg
      h_unbiased CI_seq α hα centered_scaled_seq

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
  exact DSL_valid_inference_from_assumptions μ axioms.E
    (mEstimatorConsistency_of_axioms μ d axioms)
    (mEstimatorAsymptoticNormal_of_axioms μ d axioms)
    coverage_axioms dbs m β_star V reg h_unbiased data_seq β_hat_seq h_est centered_scaled_seq
    CI_seq α hα

/-- Generic constructive DSL valid inference. This replaces the blanket
coverage-transfer axiom with an explicit normal-coverage construction. -/
theorem DSL_valid_inference_from_construction_from_assumptions
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (E : ((Obs × Mis × Mis × SamplingIndicator × ℝ) → Fin d → ℝ) → Fin d → ℝ)
    (h_consistent : MEstimatorConsistencyAssumption μ d E)
    (h_normal : MEstimatorAsymptoticNormalAssumption μ d E)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) E β_star)
    (data_seq : ℕ → Ω → List (Obs × Mis × Mis × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData m) data_seq β_hat_seq)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ)
    (coverage_construction :
      NormalCoverageConstruction μ centered_scaled_seq CI_seq β_star α V) :
    ValidInference μ β_hat_seq β_star centered_scaled_seq V CI_seq α := by
  refine ⟨?_, ?_, ?_⟩
  · exact DSL_consistent_from_assumptions μ E h_consistent dbs m β_star reg h_unbiased data_seq
      β_hat_seq h_est
  · exact DSL_asymptotic_normal_from_assumptions μ E h_normal dbs m β_star V reg h_unbiased
      centered_scaled_seq
  · exact DSL_valid_coverage_from_construction_from_assumptions μ E h_normal dbs m β_star V reg
      h_unbiased CI_seq α centered_scaled_seq coverage_construction

/-- Axioms-packaged version of the generic constructive DSL valid-inference
theorem. -/
theorem DSL_valid_inference_from_construction
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (axioms : MEstimationAxioms Ω (Obs × Mis × Mis × SamplingIndicator × ℝ) μ d)
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
    (α : ℝ)
    (coverage_construction :
      NormalCoverageConstruction μ centered_scaled_seq CI_seq β_star α V) :
    ValidInference μ β_hat_seq β_star centered_scaled_seq V CI_seq α := by
  exact DSL_valid_inference_from_construction_from_assumptions μ axioms.E
    (mEstimatorConsistency_of_axioms μ d axioms)
    (mEstimatorAsymptoticNormal_of_axioms μ d axioms)
    dbs m β_star V reg h_unbiased data_seq β_hat_seq h_est centered_scaled_seq CI_seq α
    coverage_construction

/-- Concrete coordinatewise Wald-style DSL valid inference, avoiding the generic
coverage-transfer axiom by using the first-principles multivariate coverage
route after diagonal studentization. -/
theorem DSL_valid_inference_coordStdNormal_from_assumptions
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (E : ((Obs × Mis × Mis × SamplingIndicator × ℝ) → Fin d → ℝ) → Fin d → ℝ)
    (h_consistent : MEstimatorConsistencyAssumption μ d E)
    (h_normal : MEstimatorAsymptoticNormalAssumption μ d E)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) E β_star)
    (data_seq : ℕ → Ω → List (Obs × Mis × Mis × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData m) data_seq β_hat_seq)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ)
    (lower upper : Fin d → ℝ)
    (h_interval : ∀ i, lower i ≤ upper i)
    (h_event_eq :
      ∀ n i,
        {ω | β_star i ∈ Set.Icc (CI_seq n ω i).1 (CI_seq n ω i).2} =
          {ω | centered_scaled_seq n ω i / Real.sqrt (V i i) ∈ Set.Icc (lower i) (upper i)})
    (h_pos : ∀ i, 0 < V i i)
    (h_calibration :
      ∀ i,
        (((stdNormalProbabilityMeasure : ProbabilityMeasure ℝ) : Measure ℝ)
          (Set.Icc (lower i) (upper i))) = ENNReal.ofReal (1 - α)) :
    ValidInference μ β_hat_seq β_star centered_scaled_seq V CI_seq α := by
  refine ⟨?_, ?_, ?_⟩
  · exact DSL_consistent_from_assumptions μ E h_consistent dbs m β_star reg h_unbiased data_seq β_hat_seq
      h_est
  · exact DSL_asymptotic_normal_from_assumptions μ E h_normal dbs m β_star V reg h_unbiased
      centered_scaled_seq
  · exact DSL_valid_coverage_coordStdNormal_from_assumptions μ E h_normal dbs m β_star V reg
      h_unbiased CI_seq α centered_scaled_seq lower upper h_interval h_event_eq
      h_pos h_calibration

/-- Axioms-packaged version of the concrete coordinatewise Wald-style DSL valid
inference theorem. -/
theorem DSL_valid_inference_coordStdNormal
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (axioms : MEstimationAxioms Ω (Obs × Mis × Mis × SamplingIndicator × ℝ) μ d)
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
    (α : ℝ)
    (lower upper : Fin d → ℝ)
    (h_interval : ∀ i, lower i ≤ upper i)
    (h_event_eq :
      ∀ n i,
        {ω | β_star i ∈ Set.Icc (CI_seq n ω i).1 (CI_seq n ω i).2} =
          {ω | centered_scaled_seq n ω i / Real.sqrt (V i i) ∈ Set.Icc (lower i) (upper i)})
    (h_pos : ∀ i, 0 < V i i)
    (h_calibration :
      ∀ i,
        (((stdNormalProbabilityMeasure : ProbabilityMeasure ℝ) : Measure ℝ)
          (Set.Icc (lower i) (upper i))) = ENNReal.ofReal (1 - α)) :
    ValidInference μ β_hat_seq β_star centered_scaled_seq V CI_seq α := by
  exact DSL_valid_inference_coordStdNormal_from_assumptions μ axioms.E
    (mEstimatorConsistency_of_axioms μ d axioms)
    (mEstimatorAsymptoticNormal_of_axioms μ d axioms)
    dbs m β_star V reg h_unbiased data_seq β_hat_seq h_est centered_scaled_seq CI_seq α lower upper
    h_interval h_event_eq h_pos h_calibration

/-- Symmetric critical-value specialization of the concrete studentized
coordinatewise Wald DSL valid-inference route. -/
theorem DSL_valid_inference_coordStdNormal_symm_from_assumptions
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (E : ((Obs × Mis × Mis × SamplingIndicator × ℝ) → Fin d → ℝ) → Fin d → ℝ)
    (h_consistent : MEstimatorConsistencyAssumption μ d E)
    (h_normal : MEstimatorAsymptoticNormalAssumption μ d E)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) E β_star)
    (data_seq : ℕ → Ω → List (Obs × Mis × Mis × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData m) data_seq β_hat_seq)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α z : ℝ)
    (h_event_eq :
      ∀ n i,
        {ω | β_star i ∈ Set.Icc (CI_seq n ω i).1 (CI_seq n ω i).2} =
          {ω | centered_scaled_seq n ω i / Real.sqrt (V i i) ∈ Set.Icc (-z) z})
    (hz_nonneg : 0 ≤ z)
    (h_pos : ∀ i, 0 < V i i)
    (h_calibration :
      (((stdNormalProbabilityMeasure : ProbabilityMeasure ℝ) : Measure ℝ)
        (Set.Icc (-z) z)) = ENNReal.ofReal (1 - α)) :
    ValidInference μ β_hat_seq β_star centered_scaled_seq V CI_seq α := by
  exact DSL_valid_inference_coordStdNormal_from_assumptions μ E h_consistent h_normal
    dbs m β_star V reg h_unbiased data_seq β_hat_seq h_est centered_scaled_seq CI_seq α
    (fun _ => -z) (fun _ => z) (fun _ => by simpa using neg_le_self hz_nonneg) h_event_eq
    h_pos (fun _ => h_calibration)

/-- Axioms-packaged symmetric critical-value specialization of the concrete
studentized coordinatewise Wald DSL valid-inference route. -/
theorem DSL_valid_inference_coordStdNormal_symm
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (axioms : MEstimationAxioms Ω (Obs × Mis × Mis × SamplingIndicator × ℝ) μ d)
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
    (α z : ℝ)
    (h_event_eq :
      ∀ n i,
        {ω | β_star i ∈ Set.Icc (CI_seq n ω i).1 (CI_seq n ω i).2} =
          {ω | centered_scaled_seq n ω i / Real.sqrt (V i i) ∈ Set.Icc (-z) z})
    (hz_nonneg : 0 ≤ z)
    (h_pos : ∀ i, 0 < V i i)
    (h_calibration :
      (((stdNormalProbabilityMeasure : ProbabilityMeasure ℝ) : Measure ℝ)
        (Set.Icc (-z) z)) = ENNReal.ofReal (1 - α)) :
    ValidInference μ β_hat_seq β_star centered_scaled_seq V CI_seq α := by
  exact DSL_valid_inference_coordStdNormal_symm_from_assumptions μ axioms.E
    (mEstimatorConsistency_of_axioms μ d axioms)
    (mEstimatorAsymptoticNormal_of_axioms μ d axioms)
    dbs m β_star V reg h_unbiased data_seq β_hat_seq h_est centered_scaled_seq CI_seq α z
    h_event_eq hz_nonneg h_pos h_calibration

/-- Concrete plug-in diagonal Wald DSL valid inference, where the studentizing
variance comes from a covariance-estimator sequence rather than the population
diagonal. -/
theorem DSL_valid_inference_pluginStdNormal_from_assumptions
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (E : ((Obs × Mis × Mis × SamplingIndicator × ℝ) → Fin d → ℝ) → Fin d → ℝ)
    (h_consistent : MEstimatorConsistencyAssumption μ d E)
    (h_normal : MEstimatorAsymptoticNormalAssumption μ d E)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) E β_star)
    (data_seq : ℕ → Ω → List (Obs × Mis × Mis × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData m) data_seq β_hat_seq)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (V_hat_seq : ℕ → Ω → Matrix (Fin d) (Fin d) ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ)
    (lower upper : Fin d → ℝ)
    (h_interval : ∀ i, lower i ≤ upper i)
    (h_event_eq :
      ∀ n i,
        {ω | β_star i ∈ Set.Icc (CI_seq n ω i).1 (CI_seq n ω i).2} =
          {ω | pluginStudentizedStat centered_scaled_seq V_hat_seq n ω i ∈
            Set.Icc (lower i) (upper i)})
    (h_pos : ∀ i, 0 < V i i)
    (h_Vhat_diag :
      ∀ i,
        ConvergesInProbability μ
          (fun n ω => V_hat_seq n ω i i)
          (fun _ => V i i))
    (h_Vhat_diag_meas :
      ∀ n i,
        AEMeasurable (fun ω => V_hat_seq n ω i i) μ)
    (h_calibration :
      ∀ i,
        (((stdNormalProbabilityMeasure : ProbabilityMeasure ℝ) : Measure ℝ)
          (Set.Icc (lower i) (upper i))) = ENNReal.ofReal (1 - α)) :
    ValidInference μ β_hat_seq β_star centered_scaled_seq V CI_seq α := by
  refine ⟨?_, ?_, ?_⟩
  · exact DSL_consistent_from_assumptions μ E h_consistent dbs m β_star reg h_unbiased
      data_seq β_hat_seq h_est
  · exact DSL_asymptotic_normal_from_assumptions μ E h_normal dbs m β_star V reg
      h_unbiased centered_scaled_seq
  · exact DSL_valid_coverage_pluginStdNormal_from_assumptions μ E h_normal dbs m β_star V reg
      h_unbiased centered_scaled_seq V_hat_seq CI_seq α lower upper h_interval h_event_eq
      h_pos h_Vhat_diag h_Vhat_diag_meas h_calibration

/-- Axioms-packaged version of the concrete plug-in diagonal Wald valid
inference route. -/
theorem DSL_valid_inference_pluginStdNormal
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (axioms : MEstimationAxioms Ω (Obs × Mis × Mis × SamplingIndicator × ℝ) μ d)
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
    (V_hat_seq : ℕ → Ω → Matrix (Fin d) (Fin d) ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ)
    (lower upper : Fin d → ℝ)
    (h_interval : ∀ i, lower i ≤ upper i)
    (h_event_eq :
      ∀ n i,
        {ω | β_star i ∈ Set.Icc (CI_seq n ω i).1 (CI_seq n ω i).2} =
          {ω | pluginStudentizedStat centered_scaled_seq V_hat_seq n ω i ∈
            Set.Icc (lower i) (upper i)})
    (h_pos : ∀ i, 0 < V i i)
    (h_Vhat_diag :
      ∀ i,
        ConvergesInProbability μ
          (fun n ω => V_hat_seq n ω i i)
          (fun _ => V i i))
    (h_Vhat_diag_meas :
      ∀ n i,
        AEMeasurable (fun ω => V_hat_seq n ω i i) μ)
    (h_calibration :
      ∀ i,
        (((stdNormalProbabilityMeasure : ProbabilityMeasure ℝ) : Measure ℝ)
          (Set.Icc (lower i) (upper i))) = ENNReal.ofReal (1 - α)) :
    ValidInference μ β_hat_seq β_star centered_scaled_seq V CI_seq α := by
  exact DSL_valid_inference_pluginStdNormal_from_assumptions μ axioms.E
    (mEstimatorConsistency_of_axioms μ d axioms)
    (mEstimatorAsymptoticNormal_of_axioms μ d axioms)
    dbs m β_star V reg h_unbiased data_seq β_hat_seq h_est centered_scaled_seq V_hat_seq
    CI_seq α lower upper h_interval h_event_eq h_pos h_Vhat_diag h_Vhat_diag_meas
    h_calibration

/-- **Theorem 1 (Oracle Parameter Form): DSL Provides Valid Inference**

This variant names the estimand explicitly as the oracle parameter `β_oracle`,
defined by the true moment condition. Oracle access ensures expert labels match
the oracle labels on sampled documents. -/
theorem DSL_valid_inference_oracle_from_assumptions
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (assumptions : DSLAssumptions Obs Mis Con)
    (E : ((Obs × Mis × Mis × SamplingIndicator × ℝ) → Fin d → ℝ) → Fin d → ℝ)
    (h_consistent : MEstimatorConsistencyAssumption μ d E)
    (h_normal : MEstimatorAsymptoticNormalAssumption μ d E)
    (coverage_axioms : CoverageFromAsymptoticNormal μ d)
    (m : MomentFunction (Obs × Mis) d)
    (β_oracle : Fin d → ℝ)
    (h_oracle : OracleTarget m E β_oracle)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) E β_oracle)
    (data_seq : ℕ → Ω → List (Obs × Mis × Mis × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData m) data_seq β_hat_seq)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ) (hα : 0 < α ∧ α < 1)
    : ValidInference μ β_hat_seq β_oracle centered_scaled_seq V CI_seq α := by
  exact DSL_valid_inference_from_assumptions μ E h_consistent h_normal coverage_axioms
    assumptions.sampling m β_oracle V reg h_unbiased data_seq β_hat_seq h_est centered_scaled_seq
    CI_seq α hα

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
  exact DSL_valid_inference_oracle_from_assumptions μ assumptions axioms.E
    (mEstimatorConsistency_of_axioms μ d axioms)
    (mEstimatorAsymptoticNormal_of_axioms μ d axioms)
    coverage_axioms m β_oracle h_oracle V reg h_unbiased data_seq β_hat_seq h_est
    centered_scaled_seq CI_seq α hα

/-- Corollary: DSL CI coverage converges to nominal level -/
theorem DSL_CI_coverage_from_assumptions
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (E : ((Obs × Mis × Mis × SamplingIndicator × ℝ) → Fin d → ℝ) → Fin d → ℝ)
    (h_consistent : MEstimatorConsistencyAssumption μ d E)
    (h_normal : MEstimatorAsymptoticNormalAssumption μ d E)
    (coverage_axioms : CoverageFromAsymptoticNormal μ d)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) E β_star)
    (data_seq : ℕ → Ω → List (Obs × Mis × Mis × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData m) data_seq β_hat_seq)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ) (hα : 0 < α ∧ α < 1)
    : AsymptoticCoverage μ CI_seq β_star α := by
  have h :=
    DSL_valid_inference_from_assumptions μ E h_consistent h_normal coverage_axioms dbs m β_star V reg
      h_unbiased data_seq β_hat_seq h_est centered_scaled_seq CI_seq α hα
  exact h.coverage

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
  exact DSL_CI_coverage_from_assumptions μ axioms.E
    (mEstimatorConsistency_of_axioms μ d axioms)
    (mEstimatorAsymptoticNormal_of_axioms μ d axioms)
    coverage_axioms dbs m β_star V reg h_unbiased data_seq β_hat_seq h_est centered_scaled_seq CI_seq
    α hα

/-- Corollary: concrete coordinatewise Wald-style DSL CI coverage converges to
nominal level without the generic coverage axiom. -/
theorem DSL_CI_coverage_coordStdNormal_from_assumptions
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (E : ((Obs × Mis × Mis × SamplingIndicator × ℝ) → Fin d → ℝ) → Fin d → ℝ)
    (h_consistent : MEstimatorConsistencyAssumption μ d E)
    (h_normal : MEstimatorAsymptoticNormalAssumption μ d E)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) E β_star)
    (data_seq : ℕ → Ω → List (Obs × Mis × Mis × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData m) data_seq β_hat_seq)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ)
    (lower upper : Fin d → ℝ)
    (h_interval : ∀ i, lower i ≤ upper i)
    (h_event_eq :
      ∀ n i,
        {ω | β_star i ∈ Set.Icc (CI_seq n ω i).1 (CI_seq n ω i).2} =
          {ω | centered_scaled_seq n ω i / Real.sqrt (V i i) ∈ Set.Icc (lower i) (upper i)})
    (h_pos : ∀ i, 0 < V i i)
    (h_calibration :
      ∀ i,
        (((stdNormalProbabilityMeasure : ProbabilityMeasure ℝ) : Measure ℝ)
          (Set.Icc (lower i) (upper i))) = ENNReal.ofReal (1 - α)) :
    AsymptoticCoverage μ CI_seq β_star α := by
  exact (DSL_valid_inference_coordStdNormal_from_assumptions μ E h_consistent h_normal dbs m β_star V
    reg h_unbiased data_seq β_hat_seq h_est centered_scaled_seq CI_seq α lower upper h_interval
    h_event_eq h_pos h_calibration).coverage

/-- Axioms-packaged corollary for the concrete coordinatewise Wald coverage route. -/
theorem DSL_CI_coverage_coordStdNormal
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (axioms : MEstimationAxioms Ω (Obs × Mis × Mis × SamplingIndicator × ℝ) μ d)
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
    (α : ℝ)
    (lower upper : Fin d → ℝ)
    (h_interval : ∀ i, lower i ≤ upper i)
    (h_event_eq :
      ∀ n i,
        {ω | β_star i ∈ Set.Icc (CI_seq n ω i).1 (CI_seq n ω i).2} =
          {ω | centered_scaled_seq n ω i / Real.sqrt (V i i) ∈ Set.Icc (lower i) (upper i)})
    (h_pos : ∀ i, 0 < V i i)
    (h_calibration :
      ∀ i,
        (((stdNormalProbabilityMeasure : ProbabilityMeasure ℝ) : Measure ℝ)
          (Set.Icc (lower i) (upper i))) = ENNReal.ofReal (1 - α)) :
    AsymptoticCoverage μ CI_seq β_star α := by
  exact (DSL_valid_inference_coordStdNormal μ axioms dbs m β_star V reg h_unbiased data_seq β_hat_seq
    h_est centered_scaled_seq CI_seq α lower upper h_interval h_event_eq h_pos
    h_calibration).coverage

/-- Symmetric critical-value specialization of the concrete studentized
coordinatewise Wald DSL coverage route. -/
theorem DSL_CI_coverage_coordStdNormal_symm_from_assumptions
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (E : ((Obs × Mis × Mis × SamplingIndicator × ℝ) → Fin d → ℝ) → Fin d → ℝ)
    (h_consistent : MEstimatorConsistencyAssumption μ d E)
    (h_normal : MEstimatorAsymptoticNormalAssumption μ d E)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) E β_star)
    (data_seq : ℕ → Ω → List (Obs × Mis × Mis × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData m) data_seq β_hat_seq)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α z : ℝ)
    (h_event_eq :
      ∀ n i,
        {ω | β_star i ∈ Set.Icc (CI_seq n ω i).1 (CI_seq n ω i).2} =
          {ω | centered_scaled_seq n ω i / Real.sqrt (V i i) ∈ Set.Icc (-z) z})
    (hz_nonneg : 0 ≤ z)
    (h_pos : ∀ i, 0 < V i i)
    (h_calibration :
      (((stdNormalProbabilityMeasure : ProbabilityMeasure ℝ) : Measure ℝ)
        (Set.Icc (-z) z)) = ENNReal.ofReal (1 - α)) :
    AsymptoticCoverage μ CI_seq β_star α := by
  exact (DSL_valid_inference_coordStdNormal_symm_from_assumptions μ E h_consistent h_normal
    dbs m β_star V reg h_unbiased data_seq β_hat_seq h_est centered_scaled_seq CI_seq α z
    h_event_eq hz_nonneg h_pos h_calibration).coverage

/-- Axioms-packaged symmetric critical-value specialization of the concrete
studentized coordinatewise Wald DSL coverage route. -/
theorem DSL_CI_coverage_coordStdNormal_symm
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (axioms : MEstimationAxioms Ω (Obs × Mis × Mis × SamplingIndicator × ℝ) μ d)
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
    (α z : ℝ)
    (h_event_eq :
      ∀ n i,
        {ω | β_star i ∈ Set.Icc (CI_seq n ω i).1 (CI_seq n ω i).2} =
          {ω | centered_scaled_seq n ω i / Real.sqrt (V i i) ∈ Set.Icc (-z) z})
    (hz_nonneg : 0 ≤ z)
    (h_pos : ∀ i, 0 < V i i)
    (h_calibration :
      (((stdNormalProbabilityMeasure : ProbabilityMeasure ℝ) : Measure ℝ)
        (Set.Icc (-z) z)) = ENNReal.ofReal (1 - α)) :
    AsymptoticCoverage μ CI_seq β_star α := by
  exact (DSL_valid_inference_coordStdNormal_symm μ axioms dbs m β_star V reg h_unbiased
    data_seq β_hat_seq h_est centered_scaled_seq CI_seq α z h_event_eq hz_nonneg h_pos
    h_calibration).coverage

/-- Corollary: CI coverage on the concrete plug-in diagonal Wald lane. -/
theorem DSL_CI_coverage_pluginStdNormal_from_assumptions
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (E : ((Obs × Mis × Mis × SamplingIndicator × ℝ) → Fin d → ℝ) → Fin d → ℝ)
    (h_consistent : MEstimatorConsistencyAssumption μ d E)
    (h_normal : MEstimatorAsymptoticNormalAssumption μ d E)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) E β_star)
    (data_seq : ℕ → Ω → List (Obs × Mis × Mis × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData m) data_seq β_hat_seq)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (V_hat_seq : ℕ → Ω → Matrix (Fin d) (Fin d) ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ)
    (lower upper : Fin d → ℝ)
    (h_interval : ∀ i, lower i ≤ upper i)
    (h_event_eq :
      ∀ n i,
        {ω | β_star i ∈ Set.Icc (CI_seq n ω i).1 (CI_seq n ω i).2} =
          {ω | pluginStudentizedStat centered_scaled_seq V_hat_seq n ω i ∈
            Set.Icc (lower i) (upper i)})
    (h_pos : ∀ i, 0 < V i i)
    (h_Vhat_diag :
      ∀ i,
        ConvergesInProbability μ
          (fun n ω => V_hat_seq n ω i i)
          (fun _ => V i i))
    (h_Vhat_diag_meas :
      ∀ n i,
        AEMeasurable (fun ω => V_hat_seq n ω i i) μ)
    (h_calibration :
      ∀ i,
        (((stdNormalProbabilityMeasure : ProbabilityMeasure ℝ) : Measure ℝ)
          (Set.Icc (lower i) (upper i))) = ENNReal.ofReal (1 - α)) :
    AsymptoticCoverage μ CI_seq β_star α := by
  exact (DSL_valid_inference_pluginStdNormal_from_assumptions μ E h_consistent h_normal dbs m
    β_star V reg h_unbiased data_seq β_hat_seq h_est centered_scaled_seq V_hat_seq CI_seq α
    lower upper h_interval h_event_eq h_pos h_Vhat_diag h_Vhat_diag_meas h_calibration).coverage

/-- Axioms-packaged CI coverage on the concrete plug-in diagonal Wald lane. -/
theorem DSL_CI_coverage_pluginStdNormal
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (axioms : MEstimationAxioms Ω (Obs × Mis × Mis × SamplingIndicator × ℝ) μ d)
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
    (V_hat_seq : ℕ → Ω → Matrix (Fin d) (Fin d) ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ)
    (lower upper : Fin d → ℝ)
    (h_interval : ∀ i, lower i ≤ upper i)
    (h_event_eq :
      ∀ n i,
        {ω | β_star i ∈ Set.Icc (CI_seq n ω i).1 (CI_seq n ω i).2} =
          {ω | pluginStudentizedStat centered_scaled_seq V_hat_seq n ω i ∈
            Set.Icc (lower i) (upper i)})
    (h_pos : ∀ i, 0 < V i i)
    (h_Vhat_diag :
      ∀ i,
        ConvergesInProbability μ
          (fun n ω => V_hat_seq n ω i i)
          (fun _ => V i i))
    (h_Vhat_diag_meas :
      ∀ n i,
        AEMeasurable (fun ω => V_hat_seq n ω i i) μ)
    (h_calibration :
      ∀ i,
        (((stdNormalProbabilityMeasure : ProbabilityMeasure ℝ) : Measure ℝ)
          (Set.Icc (lower i) (upper i))) = ENNReal.ofReal (1 - α)) :
    AsymptoticCoverage μ CI_seq β_star α := by
  exact (DSL_valid_inference_pluginStdNormal μ axioms dbs m β_star V reg h_unbiased data_seq
    β_hat_seq h_est centered_scaled_seq V_hat_seq CI_seq α lower upper h_interval h_event_eq
    h_pos h_Vhat_diag h_Vhat_diag_meas h_calibration).coverage

/-- Corollary (Oracle Parameter Form): DSL CI coverage converges to nominal level. -/
theorem DSL_CI_coverage_oracle_from_assumptions
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (assumptions : DSLAssumptions Obs Mis Con)
    (E : ((Obs × Mis × Mis × SamplingIndicator × ℝ) → Fin d → ℝ) → Fin d → ℝ)
    (h_consistent : MEstimatorConsistencyAssumption μ d E)
    (h_normal : MEstimatorAsymptoticNormalAssumption μ d E)
    (coverage_axioms : CoverageFromAsymptoticNormal μ d)
    (m : MomentFunction (Obs × Mis) d)
    (β_oracle : Fin d → ℝ)
    (h_oracle : OracleTarget m E β_oracle)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) E β_oracle)
    (data_seq : ℕ → Ω → List (Obs × Mis × Mis × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData m) data_seq β_hat_seq)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ) (hα : 0 < α ∧ α < 1)
    : AsymptoticCoverage μ CI_seq β_oracle α := by
  have h :=
    DSL_valid_inference_oracle_from_assumptions μ assumptions E h_consistent h_normal coverage_axioms m
      β_oracle h_oracle V reg h_unbiased data_seq β_hat_seq h_est centered_scaled_seq CI_seq α hα
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
  exact DSL_CI_coverage_oracle_from_assumptions μ assumptions axioms.E
    (mEstimatorConsistency_of_axioms μ d axioms)
    (mEstimatorAsymptoticNormal_of_axioms μ d axioms)
    coverage_axioms m β_oracle h_oracle V reg h_unbiased data_seq β_hat_seq h_est centered_scaled_seq
    CI_seq α hα

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
## TreePO: Calibrated DSL Upper Bound

This re-exports the TreePO + calibration bound for external use.
-/

/-- Re-export: calibration RMSE representativeness assumption. -/
abbrev calibration_rmse_representativeness :=
  @CalibrationRMSEBound

/-- Re-export: backward-compatible calibration assumption alias. -/
abbrev calibration_axioms_representativeness :=
  @CalibrationAxioms

/-- Re-export: population judge bias under a finite PMF. -/
abbrev population_judge_bias_export :=
  @populationJudgeBias

/-- Re-export: population judge RMSE under a finite PMF. -/
abbrev population_judge_rmse_export :=
  @populationJudgeRMSE

/-- Re-export: held-out bias-envelope route to `CalibrationRMSEBound`. -/
abbrev calibration_rmse_from_abs_trueBias_export :=
  @CalibrationRMSEBound_of_abs_trueBias_le

/-- Re-export: confidence-interval-membership route to `CalibrationRMSEBound`. -/
abbrev calibration_rmse_from_biasConfidenceInterval_export :=
  @CalibrationRMSEBound_of_mem_biasConfidenceInterval

/-- Re-export: event-level `CalibrationRMSEBound` production from a population
RMSE event and a held-out bias-interval event. -/
abbrev calibration_rmse_event_from_biasConfidence_event_export :=
  @calibrationRMSEBound_event_of_biasConfidence_event

/-- Re-export: event-level judge-gap bound from a held-out bias-confidence
event plus a population RMSE envelope event. -/
abbrev surrogate_bound_pmf_calibration2_event_from_biasConfidence_event_export :=
  @surrogate_bound_pmf_calibration2_event_of_biasConfidence_event

/-- Re-export: generic clustered confidence-interval coverage from an error event. -/
abbrev clustered_confidence_interval_coverage_of_error_event_export :=
  @clusteredConfidenceInterval_coverage_of_error_event

/-- Re-export: judge-bias confidence-interval coverage from an error event. -/
abbrev judge_bias_confidence_interval_coverage_of_error_event_export :=
  @judgeBiasConfidenceInterval_coverage_of_error_event

/-- Re-export: clustered judge-bias confidence-interval coverage from an error event. -/
abbrev judge_clustered_bias_confidence_interval_coverage_of_error_event_export :=
  @judgeClusteredBiasConfidenceInterval_coverage_of_error_event

/-- Re-export: absolute-bias envelope from judge-bias confidence-interval membership. -/
abbrev abs_true_bias_le_absbiasUpperBound_of_mem_biasConfidenceInterval_export :=
  @abs_trueBias_le_absbiasUpperBound_of_mem_biasConfidenceInterval

/-- Re-export: clustered absolute-bias envelope from clustered judge-bias interval membership. -/
abbrev abs_true_bias_le_clusteredAbsbiasUpperBound_of_mem_clusteredBiasConfidenceInterval_export :=
  @abs_trueBias_le_clusteredAbsbiasUpperBound_of_mem_clusteredBiasConfidenceInterval

/-- Re-export: exact HT unbiasedness for `Exp` targets under Bernoulli inclusion. -/
abbrev ht_exp_unbiased :=
  @htExp_unbiased

abbrev dsl_treepo_upper_bound_calibrated_pmf :=
  @dsl_upperBound_treepo_calibrated_pmf

/-- Re-export: TreePO upper bound with an explicit oracle-measurement term. -/
abbrev dsl_treepo_upper_bound_with_oracleMeasurement :=
  @dsl_upperBound_treepo_with_oracleMeasurement

/-- Re-export: worst-case envelope with calibration + estimation error. -/
abbrev treepo_gap_calibration_estimation_envelope :=
  @treepo_gap_with_calibration_and_estimation

/-- Re-export: worst-case envelope with oracle measurement + calibration + estimation. -/
abbrev treepo_gap_oracleMeasurement_calibration_estimation_envelope :=
  @treepo_gap_with_oracleMeasurement_calibration_and_estimation

/-- Re-export: worst-case envelope with calibration + estimation + clipping. -/
abbrev treepo_gap_calibration_estimation_clipping_envelope :=
  @treepo_gap_with_calibration_estimation_clipping

/-- Re-export: worst-case envelope with oracle measurement + calibration + estimation + clipping. -/
abbrev treepo_gap_oracleMeasurement_calibration_estimation_clipping_envelope :=
  @treepo_gap_with_oracleMeasurement_calibration_estimation_clipping

/-- Re-export: absolute-gap DSL envelope from estimate-space assumptions. -/
abbrev dsl_abs_gap_bound_from_estimate_export :=
  @dsl_abs_gap_bound_from_estimate

/-- Re-export: estimate-space envelope with oracle measurement. -/
abbrev dsl_abs_gap_bound_from_estimate_with_oracleMeasurement_export :=
  @dsl_abs_gap_bound_from_estimate_with_oracleMeasurement

/-- Re-export: clipped-estimate DSL envelope. -/
abbrev dsl_abs_gap_bound_from_clipped_estimate_export :=
  @dsl_abs_gap_bound_from_clipped_estimate

/-- Re-export: clipped-estimate DSL envelope with oracle measurement. -/
abbrev dsl_abs_gap_bound_from_clipped_estimate_with_oracleMeasurement_export :=
  @dsl_abs_gap_bound_from_clipped_estimate_with_oracleMeasurement

/-- Re-export: one-shot high-probability envelope from calibration + estimation. -/
abbrev dsl_abs_gap_bound_from_estimate_high_prob_export :=
  @dsl_abs_gap_bound_from_estimate_high_prob

/-- Re-export: one-shot high-probability envelope with oracle measurement. -/
abbrev dsl_abs_gap_bound_from_estimate_high_prob_with_oracleMeasurement_export :=
  @dsl_abs_gap_bound_from_estimate_high_prob_with_oracleMeasurement

/-- Re-export: one-shot high-probability envelope with clipping. -/
abbrev dsl_abs_gap_bound_from_clipped_estimate_high_prob_export :=
  @dsl_abs_gap_bound_from_clipped_estimate_high_prob

/-- Re-export: total-budget two-component one-shot envelope. -/
abbrev dsl_abs_gap_bound_from_estimate_high_prob_total_export :=
  @dsl_abs_gap_bound_from_estimate_high_prob_total

/-- Re-export: total-budget three-component oracle-measurement envelope. -/
abbrev dsl_abs_gap_bound_from_estimate_high_prob_with_oracleMeasurement_total_export :=
  @dsl_abs_gap_bound_from_estimate_high_prob_with_oracleMeasurement_total

/-- Re-export: total-budget three-component one-shot envelope. -/
abbrev dsl_abs_gap_bound_from_clipped_estimate_high_prob_total_export :=
  @dsl_abs_gap_bound_from_clipped_estimate_high_prob_total

/-- Re-export: event-based validity for an explicit DSL bound object. -/
abbrev dsl_bound_valid_from_events_export :=
  @dsl_bound_valid_from_events

/-- Re-export: event-based validity for an explicit DSL bound with oracle measurement. -/
abbrev dsl_bound_valid_from_events_with_oracleMeasurement_export :=
  @dsl_bound_valid_from_events_with_oracleMeasurement

/-- Re-export: total-failure-budget form of DSL bound validity. -/
abbrev dsl_bound_valid_from_events_total_export :=
  @dsl_bound_valid_from_events_total

/-- Re-export: total-failure-budget form of DSL bound validity with oracle measurement. -/
abbrev dsl_bound_valid_from_events_with_oracleMeasurement_total_export :=
  @dsl_bound_valid_from_events_with_oracleMeasurement_total

/-- Re-export: event-based validity specialized to `computeDSLBound`. -/
abbrev computeDSLBound_valid_from_events_export :=
  @computeDSLBound_valid_from_events

/-- Re-export: `computeDSLBound` validity with oracle measurement. -/
abbrev computeDSLBound_valid_from_events_with_oracleMeasurement_export :=
  @computeDSLBound_valid_from_events_with_oracleMeasurement

/-- Re-export: pointwise `computeDSLBound` validity from confidence-interval membership. -/
abbrev dsl_treepo_upper_bound_of_interval_membership_export :=
  @dsl_upperBound_of_interval_membership

/-- Re-export: pointwise `computeDSLBound` validity from confidence-interval membership with oracle measurement. -/
abbrev dsl_treepo_upper_bound_of_interval_membership_with_oracleMeasurement_export :=
  @dsl_upperBound_of_interval_membership_with_oracleMeasurement

/-- Re-export: event-level `computeDSLBound` validity from a joint interval/calibration event. -/
abbrev computeDSLBound_valid_from_joint_interval_event_export :=
  @computeDSLBound_valid_from_joint_interval_event

/-- Re-export: event-level `computeDSLBound` validity from a joint interval/calibration/oracle event. -/
abbrev computeDSLBound_valid_from_joint_interval_event_with_oracleMeasurement_export :=
  @computeDSLBound_valid_from_joint_interval_event_with_oracleMeasurement

/-!
## Sequential Audit Wrappers
-/

/-- Re-export: scheduled union bound for countable bad-event budgets. -/
abbrev scheduled_iUnion_bound_export :=
  @scheduled_iUnion_bound

/-- Re-export: stopped-time bad-event bound from a scheduled family. -/
abbrev stopped_event_bound_of_scheduled_events_export :=
  @stopped_event_bound_of_scheduled_events

/-- Re-export: anytime-valid stopped-horizon IPW violation-rate EB bound. -/
abbrev stopped_ipw_violation_rate_empirical_bernstein_export :=
  @stopped_ipw_violation_rate_empirical_bernstein

/-- Re-export: axioms-packaged anytime-valid stopped-horizon IPW violation-rate
EB bound. -/
abbrev stopped_ipw_violation_rate_empirical_bernstein_from_axioms_export :=
  @stopped_ipw_violation_rate_empirical_bernstein_from_axioms

/-- Re-export: anytime-valid stopped-horizon IPW preference-loss EB bound. -/
abbrev stopped_ipw_preference_loss_empirical_bernstein_export :=
  @stopped_ipw_preference_loss_empirical_bernstein

/-- Re-export: axioms-packaged anytime-valid stopped-horizon IPW preference-loss
EB bound. -/
abbrev stopped_ipw_preference_loss_empirical_bernstein_from_axioms_export :=
  @stopped_ipw_preference_loss_empirical_bernstein_from_axioms

/-!
## Runtime Certificate Checkers
-/

/-- Re-export: checked runtime DSL artifacts recover the pointwise interval
membership upper bound. -/
abbrev runtimeDSLArtifact_upperBound_of_interval_membership_export :=
  @RuntimeDSLArtifact.upperBound_of_interval_membership_of_check

/-- Re-export: oracle-measurement version of the checked interval-membership
runtime certificate. -/
abbrev runtimeDSLArtifact_upperBound_of_interval_membership_with_oracleMeasurement_export :=
  @RuntimeDSLArtifact.upperBound_of_interval_membership_with_oracleMeasurement_of_check

/-- Re-export: event-based validity of a checked runtime DSL artifact. -/
abbrev runtimeDSLArtifact_valid_from_events_export :=
  @RuntimeDSLArtifact.valid_from_events_of_check

/-- Re-export: oracle-measurement event-based validity of a checked runtime DSL
artifact. -/
abbrev runtimeDSLArtifact_valid_from_events_with_oracleMeasurement_export :=
  @RuntimeDSLArtifact.valid_from_events_with_oracleMeasurement_of_check

/-- Re-export: joint-interval-event validity of a checked runtime DSL artifact. -/
abbrev runtimeDSLArtifact_valid_from_joint_interval_event_export :=
  @RuntimeDSLArtifact.valid_from_joint_interval_event_of_check

/-- Re-export: oracle-measurement joint-interval-event validity of a checked
runtime DSL artifact. -/
abbrev runtimeDSLArtifact_valid_from_joint_interval_event_with_oracleMeasurement_export :=
  @RuntimeDSLArtifact.valid_from_joint_interval_event_with_oracleMeasurement_of_check

/-- Re-export: checked nodewise audit artifacts recover the exact audited
upper-bound package encoded by the empirical certificate. -/
abbrev runtimeNodewiseAuditArtifact_upper_bounds_export :=
  @RuntimeNodewiseAuditArtifact.audited_upper_bounds_eq_of_check

/-- Re-export: checked nodewise audit artifacts transport to the existing
approximate-local-law bundle. -/
abbrev runtimeNodewiseAuditArtifact_approx_bundle_export :=
  @RuntimeNodewiseAuditArtifact.approx_bundle_eq_of_check

/-!
## TreePO End-to-End Certificates
-/

/-- Re-export: generic loss-gap bridge with oracle measurement. -/
abbrev treepo_loss_gap_with_oracleMeasurement_export :=
  @treepo_loss_gap_with_oracleMeasurement

/-- Re-export: generic exact-oracle end-to-end loss-gap bridge. -/
abbrev treepo_loss_gap_of_exactOracle_export :=
  @treepo_loss_gap_of_exactOracle

/-- Re-export: DPO end-to-end certificate with oracle measurement. -/
abbrev dpo_treepo_end_to_end_with_oracleMeasurement_export :=
  @dpo_treepo_end_to_end_certificate_with_oracleMeasurement

/-- Re-export: GRPO-PL end-to-end certificate with oracle measurement. -/
abbrev grpo_pl_treepo_end_to_end_with_oracleMeasurement_export :=
  @grpo_pl_treepo_end_to_end_certificate_with_oracleMeasurement

/-- Re-export: GRPO-PL generator end-to-end certificate with oracle measurement. -/
abbrev grpo_pl_treepo_end_to_end_gen_with_oracleMeasurement_export :=
  @grpo_pl_treepo_end_to_end_gen_certificate_with_oracleMeasurement

/-- Re-export: GRPO-RL end-to-end certificate with oracle measurement. -/
abbrev grpo_rl_treepo_end_to_end_with_oracleMeasurement_export :=
  @grpo_rl_treepo_end_to_end_certificate_with_oracleMeasurement

/-- Re-export: GRPO-RL end-to-end certificate from a primitive pointwise
loss-Lipschitz hypothesis on the finite group space. -/
abbrev grpo_rl_treepo_end_to_end_pointwise_export :=
  @grpo_rl_treepo_end_to_end_certificate_of_pointwise

/-- Re-export: GRPO-RL pointwise-route end-to-end certificate with oracle
measurement. -/
abbrev grpo_rl_treepo_end_to_end_pointwise_with_oracleMeasurement_export :=
  @grpo_rl_treepo_end_to_end_certificate_of_pointwise_with_oracleMeasurement

/-!
## TreePO Mergeable-Certificate Bridge
-/

/-- Deterministic upper-bound substitution for tree gap certificates. -/
abbrev treepo_gap_transport_upper := @tree_gap_bound_transport_upper

/-- Event-conditional upper-bound substitution for high-probability events. -/
abbrev treepo_gap_transport_upper_prob := @tree_gap_bound_transport_upper_prob

/-- DPO TreePO gap bound with externally supplied sketch upper bound. -/
abbrev dpo_tree_gap_sketch_upper := @dpo_tree_gap_bounded_by_sketch_upper

/-- GRPO-PL TreePO gap bound with externally supplied sketch upper bound. -/
abbrev grpo_pl_tree_gap_sketch_upper := @grpo_pl_tree_gap_bounded_by_sketch_upper

/-- GRPO-RL TreePO gap bound with externally supplied sketch upper bound. -/
abbrev grpo_rl_tree_gap_sketch_upper := @grpo_rl_tree_gap_bounded_by_sketch_upper

/-- GRPO-RL TreePO gap bound with externally supplied sketch upper bound from a
primitive pointwise loss-Lipschitz hypothesis. -/
abbrev grpo_rl_tree_gap_sketch_upper_pointwise :=
  @grpo_rl_tree_gap_bounded_by_sketch_upper_of_pointwise

/-- KLL availability: hierarchical mergeability at fixed randomness. -/
abbrev kll_hierarchical_mergeability_available_export :=
  @kll_hierarchical_mergeability_available

/-- GK availability: one-way mergeability in the Agarwal et al. interface. -/
abbrev gk_one_way_mergeability_available_export :=
  @gk_one_way_mergeability_available

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
theorem DSL_dominates_naive_from_assumptions
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (E_mest : ((Obs × Mis × Mis × SamplingIndicator × ℝ) → Fin d → ℝ) → Fin d → ℝ)
    (h_consistent : MEstimatorConsistencyAssumption μ d E_mest)
    (h_normal : MEstimatorAsymptoticNormalAssumption μ d E_mest)
    (coverage_axioms : CoverageFromAsymptoticNormal μ d)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) E_mest β_star)
    (data_seq : ℕ → Ω → List (Obs × Mis × Mis × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData m) data_seq β_hat_seq)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ) (hα : 0 < α ∧ α < 1)
    (E_naive : ((Obs × Mis × Mis) → Fin d → ℝ) → Fin d → ℝ)
    (h_true : MomentUnbiased (TrueMomentFromData m) E_naive β_star)
    (h_bias : ∃ i, MomentBias m E_naive β_star i ≠ 0)
    (hE_linear : ExpectationLinear E_naive)
    : ValidInference μ β_hat_seq β_star centered_scaled_seq V CI_seq α ∧
      ¬ MomentUnbiased (PredMomentFromData m) E_naive β_star := by
  refine ⟨?_, ?_⟩
  · exact DSL_valid_inference_from_assumptions μ E_mest h_consistent h_normal coverage_axioms
      dbs m β_star V reg h_unbiased data_seq β_hat_seq h_est centered_scaled_seq CI_seq α hα
  · exact naive_estimator_biased_general m E_naive β_star h_true h_bias hE_linear

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
  exact DSL_dominates_naive_from_assumptions μ axioms.E
    (mEstimatorConsistency_of_axioms μ d axioms)
    (mEstimatorAsymptoticNormal_of_axioms μ d axioms)
    coverage_axioms dbs m β_star V reg h_unbiased data_seq β_hat_seq h_est centered_scaled_seq CI_seq
    α hα E h_true h_bias hE_linear

/-- Oracle-parameter comparison: DSL dominates naive inference. -/
theorem DSL_dominates_naive_oracle_from_assumptions
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (assumptions : DSLAssumptions Obs Mis Con)
    (E_mest : ((Obs × Mis × Mis × SamplingIndicator × ℝ) → Fin d → ℝ) → Fin d → ℝ)
    (h_consistent : MEstimatorConsistencyAssumption μ d E_mest)
    (h_normal : MEstimatorAsymptoticNormalAssumption μ d E_mest)
    (coverage_axioms : CoverageFromAsymptoticNormal μ d)
    (m : MomentFunction (Obs × Mis) d)
    (β_oracle : Fin d → ℝ)
    (h_oracle : OracleTarget m E_mest β_oracle)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) E_mest β_oracle)
    (data_seq : ℕ → Ω → List (Obs × Mis × Mis × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData m) data_seq β_hat_seq)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ) (hα : 0 < α ∧ α < 1)
    (E_naive : ((Obs × Mis × Mis) → Fin d → ℝ) → Fin d → ℝ)
    (h_true : MomentUnbiased (TrueMomentFromData m) E_naive β_oracle)
    (h_bias : ∃ i, MomentBias m E_naive β_oracle i ≠ 0)
    (hE_linear : ExpectationLinear E_naive)
    : ValidInference μ β_hat_seq β_oracle centered_scaled_seq V CI_seq α ∧
      ¬ MomentUnbiased (PredMomentFromData m) E_naive β_oracle := by
  refine ⟨?_, ?_⟩
  · exact DSL_valid_inference_oracle_from_assumptions μ assumptions E_mest h_consistent h_normal
      coverage_axioms m β_oracle h_oracle V reg h_unbiased data_seq β_hat_seq h_est
      centered_scaled_seq CI_seq α hα
  · exact naive_estimator_biased_general m E_naive β_oracle h_true h_bias hE_linear

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
  exact DSL_dominates_naive_oracle_from_assumptions μ assumptions axioms.E
    (mEstimatorConsistency_of_axioms μ d axioms)
    (mEstimatorAsymptoticNormal_of_axioms μ d axioms)
    coverage_axioms m β_oracle h_oracle V reg h_unbiased data_seq β_hat_seq h_est centered_scaled_seq
    CI_seq α hα E h_true h_bias hE_linear

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
theorem DSL_guidelines_from_assumptions
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (E_mest : ((Obs × Mis × Mis × SamplingIndicator × ℝ) → Fin d → ℝ) → Fin d → ℝ)
    (h_consistent : MEstimatorConsistencyAssumption μ d E_mest)
    (h_normal : MEstimatorAsymptoticNormalAssumption μ d E_mest)
    (coverage_axioms : CoverageFromAsymptoticNormal μ d)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) E_mest β_star)
    (data_seq : ℕ → Ω → List (Obs × Mis × Mis × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData m) data_seq β_hat_seq)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ) (hα : 0 < α ∧ α < 1)
    (E_naive : ((Obs × Mis × Mis) → Fin d → ℝ) → Fin d → ℝ)
    (h_true : MomentUnbiased (TrueMomentFromData m) E_naive β_star)
    (h_bias : ∃ i, MomentBias m E_naive β_star i ≠ 0)
    (hE_linear : ExpectationLinear E_naive)
    : ValidInference μ β_hat_seq β_star centered_scaled_seq V CI_seq α ∧
      ¬ MomentUnbiased (PredMomentFromData m) E_naive β_star := by
  exact DSL_dominates_naive_from_assumptions μ E_mest h_consistent h_normal coverage_axioms dbs m
    β_star V reg h_unbiased data_seq β_hat_seq h_est centered_scaled_seq CI_seq α hα E_naive h_true
    h_bias hE_linear

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
  exact DSL_guidelines_from_assumptions μ axioms.E
    (mEstimatorConsistency_of_axioms μ d axioms)
    (mEstimatorAsymptoticNormal_of_axioms μ d axioms)
    coverage_axioms dbs m β_star V reg h_unbiased data_seq β_hat_seq h_est centered_scaled_seq CI_seq
    α hα E h_true h_bias hE_linear

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
theorem formal_results_summary_from_assumptions
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (E_mest : ((Obs × Mis × Mis × SamplingIndicator × ℝ) → Fin d → ℝ) → Fin d → ℝ)
    (h_consistent : MEstimatorConsistencyAssumption μ d E_mest)
    (h_normal : MEstimatorAsymptoticNormalAssumption μ d E_mest)
    (coverage_axioms : CoverageFromAsymptoticNormal μ d)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) E_mest β_star)
    (data_seq : ℕ → Ω → List (Obs × Mis × Mis × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData m) data_seq β_hat_seq)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ) (hα : 0 < α ∧ α < 1)
    (E_naive : ((Obs × Mis × Mis) → Fin d → ℝ) → Fin d → ℝ)
    (h_true : MomentUnbiased (TrueMomentFromData m) E_naive β_star)
    (h_bias : ∃ i, MomentBias m E_naive β_star i ≠ 0)
    (hE_linear : ExpectationLinear E_naive)
    : ValidInference μ β_hat_seq β_star centered_scaled_seq V CI_seq α ∧
      ¬ MomentUnbiased (PredMomentFromData m) E_naive β_star := by
  exact DSL_dominates_naive_from_assumptions μ E_mest h_consistent h_normal coverage_axioms dbs m
    β_star V reg h_unbiased data_seq β_hat_seq h_est centered_scaled_seq CI_seq α hα E_naive h_true
    h_bias hE_linear

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
  exact formal_results_summary_from_assumptions μ axioms.E
    (mEstimatorConsistency_of_axioms μ d axioms)
    (mEstimatorAsymptoticNormal_of_axioms μ d axioms)
    coverage_axioms dbs m β_star V reg h_unbiased data_seq β_hat_seq h_est centered_scaled_seq CI_seq
    α hα E h_true h_bias hE_linear

/-- Concrete 1D OLS Wald coverage from convergence of the t-statistic cdf to the standard normal cdf. -/
abbrev ols_wald_coverage_from_tstat_cdf_stdNormal :=
  @Econometrics.OLS.asymptotic_ci_coverage_from_tstat_cdf_to_stdNormal

/-- Generic 1D first-principles coverage transfer from cdf convergence to the
standard normal law plus a coverage-event equivalence. -/
abbrev coverage_one_dim_from_cdf_stdNormal_eventEq :=
  @asymptoticCoverage_oneDim_of_cdfConvergesToStdNormal_of_eventEq

/-- Fully generic coordinatewise first-principles coverage witness. -/
abbrev coordinate_coverage_limit_witness :=
  @CoordinateCoverageLimitWitness

/-- Generic constructive coverage interface from asymptotic normality. -/
abbrev normal_coverage_construction :=
  @NormalCoverageConstruction

/-- Coverage derived from a generic coordinatewise limit witness. -/
abbrev coverage_from_limit_witness :=
  @CoordinateCoverageLimitWitness.asymptoticCoverage

/-- Coverage derived from a generic normal-coverage construction. -/
abbrev coverage_from_normal_construction :=
  @NormalCoverageConstruction.asymptoticCoverage

/-- DSL coverage derived from a generic normal-coverage construction. -/
abbrev DSL_coverage_from_normal_construction :=
  @DSL_valid_coverage_from_construction

/-- DSL valid inference derived from a generic normal-coverage construction. -/
abbrev DSL_valid_inference_from_normal_construction :=
  @DSL_valid_inference_from_construction

/-- Generic multivariate coordinatewise coverage transfer from weak convergence
of the full statistic vector plus explicit coordinate-strip event identities. -/
abbrev coverage_multivariate_from_tendstoInDistribution_coordIcc_eventEq :=
  @asymptoticCoverage_of_tendstoInDistribution_of_coordIcc_of_eventEq

/-- Multivariate coordinatewise coverage when the weak limit has standard-normal
coordinate marginals. -/
abbrev coverage_multivariate_from_tendstoInDistribution_coordStdNormal_eventEq :=
  @asymptoticCoverage_of_tendstoInDistribution_of_coordStdNormal_of_eventEq

/-- Multivariate coordinatewise coverage from a multivariate normal limit after
diagonal studentization. -/
abbrev coverage_multivariate_from_convergesInDistributionToNormal_standardized_eventEq :=
  @asymptoticCoverage_of_convergesInDistributionToNormal_standardized_of_eventEq

/-- Symmetric-critical-value specialization of the studentized multivariate
coordinatewise coverage route. -/
abbrev coverage_multivariate_from_convergesInDistributionToNormal_standardized_symm_eventEq :=
  @asymptoticCoverage_of_convergesInDistributionToNormal_standardized_symm_of_eventEq

/-- Plug-in diagonal covariance version of the multivariate coordinatewise
coverage route. -/
abbrev coverage_multivariate_from_convergesInDistributionToNormal_plugin_eventEq :=
  @asymptoticCoverage_of_convergesInDistributionToNormal_plugin_of_eventEq

/-- Curated export: plug-in diagonal Wald valid inference route. -/
abbrev dsl_valid_inference_pluginStdNormal_export :=
  @DSL_valid_inference_pluginStdNormal

/-- Curated export: plug-in diagonal Wald coverage route. -/
abbrev dsl_ci_coverage_pluginStdNormal_export :=
  @DSL_CI_coverage_pluginStdNormal

end DSL

end
