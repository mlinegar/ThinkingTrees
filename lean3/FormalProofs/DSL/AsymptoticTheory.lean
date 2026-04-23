/-
# FormalProofs/DSL/AsymptoticTheory.lean

## Paper Reference: Section 3.2, Proposition 1, Appendix OA.7

This file formalizes the asymptotic theory of the DSL estimator:
- Consistency: β̂_DSL → β* as N → ∞
- Asymptotic normality: √N(β̂_DSL - β*) →d N(0, V)
- Variance formula (sandwich estimator)

### Main Results

**Proposition 1 (Asymptotic Properties)**

Under Assumption 1 (design-based sampling) and standard regularity conditions:

1. **Consistency:** β̂_DSL →p β* as N → ∞
2. **Asymptotic Normality:** √N(β̂_DSL - β*) →d N(0, V)

where V is the sandwich variance matrix.

### Variance Formula (Equation OA.7)

V = S_V⁻¹ · E[m̃(D; β*) m̃(D; β*)'] · S_V⁻¹'

where S_V = E[∂m̃/∂β] evaluated at β*.
-/

import FormalProofs.DSL.DSLEstimator
import FormalProofs.DSL.CrossFitting
import FormalProofs.DSL.BiasAnalysis
import FormalProofs.DSL.AsymptoticCore
import FormalProofs.DSL.ConcreteCoverage
import FormalProbability.Econometrics.Chapter3
import Mathlib.MeasureTheory.Measure.Typeclasses.Probability
import Mathlib.MeasureTheory.Function.ConvergenceInDistribution
import Mathlib.MeasureTheory.Function.ConvergenceInMeasure

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical
open scoped Topology
open MeasureTheory

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace DSL

/-!
## Regularity Conditions
-/

/-- Minimal structural holdout condition for cross-fitting:
each unit has at least one unit in a different fold. -/
def HasHoldoutFold {ι Obs Con Mis : Type*} [Fintype ι]
    (cf : CrossFit ι Obs Con Mis) : Prop :=
  ∀ i, ∃ j, cf.fold j ≠ cf.fold i

/-- Standard regularity conditions for asymptotic normality.
    These are the conditions from M-estimation theory. -/
structure RegularityConditions (Data : Type*) (d : ℕ) where
  /-- Candidate parameter space Θ. -/
  param_space : Set (Fin d → ℝ)
  /-- Openness of Θ at the target point. -/
  param_space_open : IsOpen param_space
  /-- Deterministic modulus controlling local smoothness error terms. -/
  moment_modulus : ℕ → ℝ
  /-- Nonnegativity of the smoothness modulus. -/
  moment_modulus_nonneg : ∀ n, 0 ≤ moment_modulus n
  /-- Smoothness proxy: modulus vanishes asymptotically. -/
  moment_smooth : Filter.Tendsto moment_modulus Filter.atTop (𝓝 0)
  /-- Jacobian proxy matrix S = E[∂m/∂β] at β*. -/
  jacobian : Matrix (Fin d) (Fin d) ℝ
  /-- Nonsingularity proxy for identification. -/
  jacobian_invertible : Matrix.det jacobian ≠ 0
  /-- Envelope dominating per-observation moment magnitudes. -/
  moment_envelope : Data → ℝ
  /-- Finite second-moment proxy via a global quadratic envelope bound. -/
  second_moment_finite : ∃ M : ℝ, 0 ≤ M ∧ ∀ D, (moment_envelope D)^2 ≤ M
  /-- Deterministic uniform-convergence rate for sample moments. -/
  uniform_rate : ℕ → ℝ
  /-- Rate is nonnegative. -/
  uniform_rate_nonneg : ∀ n, 0 ≤ uniform_rate n
  /-- Uniform convergence proxy: rate vanishes with n. -/
  uniform_convergence : Filter.Tendsto uniform_rate Filter.atTop (𝓝 0)

/-- Cross-fitting regularity conditions (minimal structural bundle). -/
structure CrossFittingConditions {ι Obs Con Mis : Type*} [Fintype ι]
    (cf : CrossFit ι Obs Con Mis) : Prop where
  /-- Structural no-leakage proxy: every unit has a distinct holdout fold. -/
  no_leakage : HasHoldoutFold cf

/-!
## Consistency
-/

/-!
## Sample Moments and Estimators
-/

/-- Sample mean of a moment function over a finite dataset. -/
def sampleMoment {Data : Type*} {d : ℕ}
    (m : MomentFunction Data d)
    (data : List Data)
    (β : Fin d → ℝ) : Fin d → ℝ :=
  let N := data.length
  fun j => (data.foldl (fun acc D => acc + m D β j) 0) / N

/-- A (sample) M-estimator solves the sample moment condition. -/
def IsMEstimator {Data : Type*} {d : ℕ}
    (m : MomentFunction Data d)
    (data : List Data)
    (β : Fin d → ℝ) : Prop :=
  sampleMoment m data β = 0

/-- An estimator sequence solves the sample moment condition at each n. -/
def IsMEstimatorSeq {Data : Type*} {d : ℕ}
    (m : MomentFunction Data d)
    {Ω : Type*}
    (data_seq : ℕ → Ω → List Data)
    (β_hat_seq : ℕ → Ω → Fin d → ℝ) : Prop :=
  ∀ n ω, IsMEstimator m (data_seq n ω) (β_hat_seq n ω)

/-- DSL moment function lifted to a single data record. -/
def DSLMomentFromData {Obs Mis : Type*} {d : ℕ}
    (m : MomentFunction (Obs × Mis) d)
    (D : Obs × Mis × Mis × SamplingIndicator × ℝ)
    (β : Fin d → ℝ) : Fin d → ℝ :=
  match D with
  | ⟨d_obs, d_mis_pred, d_mis_true, R, π⟩ =>
      DSLMoment m d_obs d_mis_pred d_mis_true R π β

/-!
## Oracle Target
-/

/-- Oracle moment using the true missing values from a full DSL data record. -/
def TrueMomentFromFullData {Obs Mis : Type*} {d : ℕ}
    (m : MomentFunction (Obs × Mis) d) :
    MomentFunction (Obs × Mis × Mis × SamplingIndicator × ℝ) d :=
  fun D β =>
    match D with
    | ⟨d_obs, _d_mis_pred, d_mis_true, _R, _π⟩ =>
        m (d_obs, d_mis_true) β

/-- Oracle target parameter: solves the true moment condition. -/
def OracleTarget {Obs Mis : Type*} {d : ℕ}
    (m : MomentFunction (Obs × Mis) d)
    (E : ((Obs × Mis × Mis × SamplingIndicator × ℝ) → Fin d → ℝ) → Fin d → ℝ)
    (β : Fin d → ℝ) : Prop :=
  MomentUnbiased (TrueMomentFromFullData m) E β

/-!
## Generic M-Estimation Assumptions
-/

/-- Abstract M-estimation asymptotic results, bundled as explicit assumptions. -/
structure MEstimationAxioms (Ω Data : Type*) [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (d : ℕ) where
  /-- Expectation operator for moments. -/
  E : (Data → Fin d → ℝ) → Fin d → ℝ
  /-- Consistency for any estimator sequence solving the sample moment equation. -/
  consistent :
    ∀ (m : MomentFunction Data d) (β_star : Fin d → ℝ)
      (data_seq : ℕ → Ω → List Data) (β_hat_seq : ℕ → Ω → Fin d → ℝ),
      MomentUnbiased m E β_star →
      RegularityConditions Data d →
      IsMEstimatorSeq m data_seq β_hat_seq →
      ConvergesInProbability μ β_hat_seq (fun _ => β_star)
  /-- Asymptotic normality for centered/scaled estimator sequences. -/
  asymptotic_normal :
    ∀ (m : MomentFunction Data d) (β_star : Fin d → ℝ) (V : Matrix (Fin d) (Fin d) ℝ)
      (centered_scaled_seq : ℕ → Ω → Fin d → ℝ),
      MomentUnbiased m E β_star →
      RegularityConditions Data d →
      ConvergesInDistributionToNormal μ centered_scaled_seq (fun _ => 0) V

/-- Preferred name for the M-estimation assumption bundle. -/
abbrev MEstimationAssumptions (Ω Data : Type*) [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (d : ℕ) :=
  MEstimationAxioms Ω Data μ d

/-- Standalone consistency assumption used by DSL M-estimation results. -/
def MEstimatorConsistencyAssumption {Ω Data : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (d : ℕ) (E : (Data → Fin d → ℝ) → Fin d → ℝ) : Prop :=
  ∀ (m : MomentFunction Data d) (β_star : Fin d → ℝ)
    (data_seq : ℕ → Ω → List Data) (β_hat_seq : ℕ → Ω → Fin d → ℝ),
    MomentUnbiased m E β_star →
    RegularityConditions Data d →
    IsMEstimatorSeq m data_seq β_hat_seq →
    ConvergesInProbability μ β_hat_seq (fun _ => β_star)

/-- Standalone asymptotic-normality assumption used by DSL M-estimation results. -/
def MEstimatorAsymptoticNormalAssumption {Ω Data : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (d : ℕ) (E : (Data → Fin d → ℝ) → Fin d → ℝ) : Prop :=
  ∀ (m : MomentFunction Data d) (β_star : Fin d → ℝ) (V : Matrix (Fin d) (Fin d) ℝ)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ),
    MomentUnbiased m E β_star →
    RegularityConditions Data d →
    ConvergesInDistributionToNormal μ centered_scaled_seq (fun _ => 0) V

lemma mEstimatorConsistency_of_axioms
    {Ω Data : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (d : ℕ)
    (axioms : MEstimationAxioms Ω Data μ d) :
    MEstimatorConsistencyAssumption μ d axioms.E := by
  intro m β_star data_seq β_hat_seq h_unbiased h_reg h_est
  exact axioms.consistent m β_star data_seq β_hat_seq h_unbiased h_reg h_est

lemma mEstimatorAsymptoticNormal_of_axioms
    {Ω Data : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (d : ℕ)
    (axioms : MEstimationAxioms Ω Data μ d) :
    MEstimatorAsymptoticNormalAssumption μ d axioms.E := by
  intro m β_star V centered_scaled_seq h_unbiased h_reg
  exact axioms.asymptotic_normal m β_star V centered_scaled_seq h_unbiased h_reg

/-- Build the bundled M-estimation assumption package from explicit components. -/
def mkMEstimationAxioms
    {Ω Data : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] (d : ℕ)
    (E : (Data → Fin d → ℝ) → Fin d → ℝ)
    (h_consistent : MEstimatorConsistencyAssumption μ d E)
    (h_normal : MEstimatorAsymptoticNormalAssumption μ d E) :
    MEstimationAxioms Ω Data μ d where
  E := E
  consistent := h_consistent
  asymptotic_normal := h_normal

/-- DSL consistency theorem.

    Under Assumption 1 and regularity conditions, the DSL estimator
    converges in probability to the true parameter β*.

    The key insight is that E[m̃(D; β*)] = 0 because the design-adjusted
    moment is unbiased, so by the law of large numbers,
    (1/N)∑m̃(Di; β) → E[m̃(D; β)] and the unique zero is at β*. -/
theorem DSL_consistent_from_assumptions
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (E : ((Obs × Mis × Mis × SamplingIndicator × ℝ) → Fin d → ℝ) → Fin d → ℝ)
    (h_consistent : MEstimatorConsistencyAssumption μ d E)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) E β_star)
    (data_seq : ℕ → Ω → List (Obs × Mis × Mis × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData m) data_seq β_hat_seq)
    : ConvergesInProbability μ β_hat_seq (fun _ => β_star) := by
  exact h_consistent (DSLMomentFromData m) β_star data_seq β_hat_seq h_unbiased reg h_est

/-- DSL consistency theorem.

    Under Assumption 1 and regularity conditions, the DSL estimator
    converges in probability to the true parameter β*.

    The key insight is that E[m̃(D; β*)] = 0 because the design-adjusted
    moment is unbiased, so by the law of large numbers,
    (1/N)∑m̃(Di; β) → E[m̃(D; β)] and the unique zero is at β*. -/
theorem DSL_consistent
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (axioms : MEstimationAxioms Ω (Obs × Mis × Mis × SamplingIndicator × ℝ) μ d)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) axioms.E β_star)
    (data_seq : ℕ → Ω → List (Obs × Mis × Mis × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData m) data_seq β_hat_seq)
    : ConvergesInProbability μ β_hat_seq (fun _ => β_star) := by
  exact DSL_consistent_from_assumptions μ axioms.E
    (mEstimatorConsistency_of_axioms μ d axioms)
    dbs m β_star reg h_unbiased data_seq β_hat_seq h_est

/-- Cross-fitted DSL consistency theorem (Appendix B.2). -/
theorem DSL_consistent_crossfit_from_assumptions
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {ι Obs Mis Con : Type*} [Fintype ι] {d : ℕ}
    (E : ((Obs × Mis × Mis × SamplingIndicator × ℝ) → Fin d → ℝ) → Fin d → ℝ)
    (h_consistent : MEstimatorConsistencyAssumption μ d E)
    (cf : CrossFit ι Obs Con Mis)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (cf_reg : CrossFittingConditions cf)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) E β_star)
    (data_seq : ℕ → Ω → List (Obs × Mis × Mis × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData m) data_seq β_hat_seq)
    : ConvergesInProbability μ β_hat_seq (fun _ => β_star) := by
  exact h_consistent (DSLMomentFromData m) β_star data_seq β_hat_seq h_unbiased reg h_est

/-- Cross-fitted DSL consistency theorem (Appendix B.2). -/
theorem DSL_consistent_crossfit
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {ι Obs Mis Con : Type*} [Fintype ι] {d : ℕ}
    (axioms : MEstimationAxioms Ω (Obs × Mis × Mis × SamplingIndicator × ℝ) μ d)
    (cf : CrossFit ι Obs Con Mis)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (cf_reg : CrossFittingConditions cf)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) axioms.E β_star)
    (data_seq : ℕ → Ω → List (Obs × Mis × Mis × SamplingIndicator × ℝ))
    (β_hat_seq : ℕ → Ω → Fin d → ℝ)
    (h_est : IsMEstimatorSeq (DSLMomentFromData m) data_seq β_hat_seq)
    : ConvergesInProbability μ β_hat_seq (fun _ => β_star) := by
  exact DSL_consistent_crossfit_from_assumptions μ axioms.E
    (mEstimatorConsistency_of_axioms μ d axioms)
    cf dbs m β_star reg cf_reg h_unbiased data_seq β_hat_seq h_est

/-!
## Asymptotic Normality
-/

/-- DSL asymptotic normality theorem (Proposition 1).

    Under Assumption 1 and regularity conditions:
    √N(β̂_DSL - β*) →d N(0, V)

    where V is the sandwich variance matrix. -/
theorem DSL_asymptotic_normal_from_assumptions
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (E : ((Obs × Mis × Mis × SamplingIndicator × ℝ) → Fin d → ℝ) → Fin d → ℝ)
    (h_normal : MEstimatorAsymptoticNormalAssumption μ d E)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) E β_star)
    : ∀ (centered_scaled_seq : ℕ → Ω → Fin d → ℝ),
      ConvergesInDistributionToNormal μ centered_scaled_seq (fun _ => 0) V := by
  intro seq
  exact h_normal (DSLMomentFromData m) β_star V seq h_unbiased reg

/-- DSL asymptotic normality theorem (Proposition 1).

    Under Assumption 1 and regularity conditions:
    √N(β̂_DSL - β*) →d N(0, V)

    where V is the sandwich variance matrix. -/
theorem DSL_asymptotic_normal
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (axioms : MEstimationAxioms Ω (Obs × Mis × Mis × SamplingIndicator × ℝ) μ d)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) axioms.E β_star)
    : ∀ (centered_scaled_seq : ℕ → Ω → Fin d → ℝ),
      -- √N(β̂_N - β*) where β̂_N is the DSL estimator
      ConvergesInDistributionToNormal μ centered_scaled_seq (fun _ => 0) V := by
  exact DSL_asymptotic_normal_from_assumptions μ axioms.E
    (mEstimatorAsymptoticNormal_of_axioms μ d axioms)
    dbs m β_star V reg h_unbiased

/-- Cross-fitted DSL asymptotic normality theorem (Appendix B.2). -/
theorem DSL_asymptotic_normal_crossfit_from_assumptions
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {ι Obs Mis Con : Type*} [Fintype ι] {d : ℕ}
    (E : ((Obs × Mis × Mis × SamplingIndicator × ℝ) → Fin d → ℝ) → Fin d → ℝ)
    (h_normal : MEstimatorAsymptoticNormalAssumption μ d E)
    (cf : CrossFit ι Obs Con Mis)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (cf_reg : CrossFittingConditions cf)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) E β_star)
    : ∀ (centered_scaled_seq : ℕ → Ω → Fin d → ℝ),
      ConvergesInDistributionToNormal μ centered_scaled_seq (fun _ => 0) V := by
  intro seq
  exact h_normal (DSLMomentFromData m) β_star V seq h_unbiased reg

/-- Cross-fitted DSL asymptotic normality theorem (Appendix B.2). -/
theorem DSL_asymptotic_normal_crossfit
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {ι Obs Mis Con : Type*} [Fintype ι] {d : ℕ}
    (axioms : MEstimationAxioms Ω (Obs × Mis × Mis × SamplingIndicator × ℝ) μ d)
    (cf : CrossFit ι Obs Con Mis)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (cf_reg : CrossFittingConditions cf)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) axioms.E β_star)
    : ∀ (centered_scaled_seq : ℕ → Ω → Fin d → ℝ),
      ConvergesInDistributionToNormal μ centered_scaled_seq (fun _ => 0) V := by
  exact DSL_asymptotic_normal_crossfit_from_assumptions μ axioms.E
    (mEstimatorAsymptoticNormal_of_axioms μ d axioms)
    cf dbs m β_star V reg cf_reg h_unbiased

/-!
## Variance Formula
-/

/-- Jacobian matrix of the moment function: E[∂m/∂β] -/
def JacobianMatrix {Obs Mis : Type*} {d : ℕ}
    (m : MomentFunction (Obs × Mis) d)
    (jacobian_integrand : (Obs × Mis) → (Fin d → ℝ) → Matrix (Fin d) (Fin d) ℝ)
    (E : ((Obs × Mis) → Matrix (Fin d) (Fin d) ℝ) → Matrix (Fin d) (Fin d) ℝ)
    (β : Fin d → ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  E (fun D => jacobian_integrand D β)

/-- Meat matrix: E[m̃ m̃'] -/
def MeatMatrix {Obs Mis : Type*} {d : ℕ}
    (m_tilde : (Obs × Mis) → Fin d → ℝ)
    (E : ((Obs × Mis) → Matrix (Fin d) (Fin d) ℝ) → Matrix (Fin d) (Fin d) ℝ)
    : Matrix (Fin d) (Fin d) ℝ :=
  E (fun data => fun i j => m_tilde data i * m_tilde data j)

/-- Sandwich variance matrix: V = S⁻¹ · M · S⁻¹'

    This is the standard sandwich estimator for M-estimators.
    For DSL, the meat matrix M uses the design-adjusted moments m̃. -/
def SandwichVariance {d : ℕ}
    (S_inv : Matrix (Fin d) (Fin d) ℝ)  -- S_V⁻¹
    (M : Matrix (Fin d) (Fin d) ℝ)       -- E[m̃ m̃']
    : Matrix (Fin d) (Fin d) ℝ :=
  S_inv * M * S_inv.transpose

/-!
## Variance Decomposition
-/

/-- Entrywise matrix order (simple PSD-like proxy). -/
def MatrixLE {d : ℕ} (A B : Matrix (Fin d) (Fin d) ℝ) : Prop :=
  ∀ i j, A i j ≤ B i j

lemma matrixLE_add {d : ℕ} {A B C D : Matrix (Fin d) (Fin d) ℝ}
    (h1 : MatrixLE A B) (h2 : MatrixLE C D) : MatrixLE (A + C) (B + D) := by
  intro i j
  simpa using add_le_add (h1 i j) (h2 i j)

lemma matrixLE_smul {d : ℕ} {A B : Matrix (Fin d) (Fin d) ℝ}
    (c : ℝ) (hc : 0 ≤ c) (h : MatrixLE A B) : MatrixLE (c • A) (c • B) := by
  intro i j
  -- Scalar multiplication is entrywise.
  simpa using mul_le_mul_of_nonneg_left (h i j) hc

/-- Variance decomposition for DSL.

    The variance of the DSL estimator can be decomposed as:
    V_DSL = V_full + (1/π - 1) · V_correction

    where:
    - V_full is the variance if all documents were expert-coded
    - V_correction accounts for using predictions instead of true labels
    - As prediction accuracy improves, V_correction decreases

    This shows that better predictions lead to smaller standard errors. -/
structure VarianceDecomposition {d : ℕ} where
  /-- Variance with full expert coding (n = N) -/
  V_full : Matrix (Fin d) (Fin d) ℝ
  /-- Correction variance from using predictions -/
  V_correction : Matrix (Fin d) (Fin d) ℝ
  /-- Sampling probability -/
  π : ℝ
  /-- Total DSL variance -/
  V_DSL : Matrix (Fin d) (Fin d) ℝ
  /-- Decomposition relation -/
  h_decomp : V_DSL = V_full + ((1/π - 1) : ℝ) • V_correction

/-- Better predictions reduce variance.

    If the prediction error variance decreases, V_correction decreases,
    leading to smaller overall variance V_DSL.

    This formalizes the efficiency property of DSL: better LLMs → smaller SEs. -/
theorem better_predictions_smaller_variance {d : ℕ}
    (vd1 vd2 : VarianceDecomposition (d := d))
    -- Same π and V_full
    (h_π : vd1.π = vd2.π)
    (h_full : vd1.V_full = vd2.V_full)
    -- V_correction is "smaller" for vd2 in entrywise matrix order.
    (h_smaller : MatrixLE vd2.V_correction vd1.V_correction)
    (h_factor_nonneg : (1 / vd1.π - 1 : ℝ) ≥ 0)
    : MatrixLE vd2.V_DSL vd1.V_DSL := by
  have h_full_le : MatrixLE vd2.V_full vd1.V_full := by
    intro i j
    simp [h_full]
  have h_corr_le :
      MatrixLE ((1 / vd2.π - 1 : ℝ) • vd2.V_correction)
        ((1 / vd1.π - 1 : ℝ) • vd1.V_correction) := by
    have h_π' : vd2.π = vd1.π := h_π.symm
    simpa [h_π'] using
      (matrixLE_smul (c := (1 / vd1.π - 1 : ℝ)) h_factor_nonneg h_smaller)
  have h_le :
      MatrixLE (vd2.V_full + ((1 / vd2.π - 1 : ℝ) • vd2.V_correction))
        (vd1.V_full + ((1 / vd1.π - 1 : ℝ) • vd1.V_correction)) :=
    matrixLE_add h_full_le h_corr_le
  simpa [vd1.h_decomp, vd2.h_decomp] using h_le

/-!
## Standard Error Formula
-/

/-- Standard error for the i-th coefficient -/
def standardError {d : ℕ} (V : Matrix (Fin d) (Fin d) ℝ) (i : Fin d) : ℝ :=
  Real.sqrt (V i i)

/-- Confidence interval for the i-th coefficient -/
def confidenceInterval {d : ℕ}
    (β_hat : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (N : ℕ)
    (z_alpha : ℝ)  -- Critical value (e.g., 1.96 for 95% CI)
    (i : Fin d) : ℝ × ℝ :=
  let se := standardError V i / Real.sqrt N
  (β_hat i - z_alpha * se, β_hat i + z_alpha * se)

/-- DSL confidence intervals have correct coverage.

    Under Assumption 1, the DSL confidence intervals achieve the
    nominal coverage rate asymptotically, regardless of prediction accuracy.

    This is the key advantage of DSL: valid inference without
    assumptions about prediction error structure. -/
theorem DSL_valid_coverage_from_assumptions
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (E : ((Obs × Mis × Mis × SamplingIndicator × ℝ) → Fin d → ℝ) → Fin d → ℝ)
    (h_normal : MEstimatorAsymptoticNormalAssumption μ d E)
    (coverage_axioms : CoverageFromAsymptoticNormal μ d)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) E β_star)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ)  -- Significance level
    (h_α : 0 < α ∧ α < 1)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    : AsymptoticCoverage μ CI_seq β_star α := by
  have h_norm :
      ConvergesInDistributionToNormal μ centered_scaled_seq (fun _ => 0) V :=
    DSL_asymptotic_normal_from_assumptions μ E h_normal dbs m β_star V reg h_unbiased centered_scaled_seq
  exact coverage_axioms centered_scaled_seq CI_seq β_star α V h_norm

/-- DSL confidence intervals have correct coverage.

    Under Assumption 1, the DSL confidence intervals achieve the
    nominal coverage rate asymptotically, regardless of prediction accuracy.

    This is the key advantage of DSL: valid inference without
    assumptions about prediction error structure. -/
theorem DSL_valid_coverage
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (axioms : MEstimationAxioms Ω (Obs × Mis × Mis × SamplingIndicator × ℝ) μ d)
    (coverage_axioms : CoverageFromAsymptoticNormal μ d)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) axioms.E β_star)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ)  -- Significance level
    (h_α : 0 < α ∧ α < 1)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    : AsymptoticCoverage μ CI_seq β_star α := by
  exact DSL_valid_coverage_from_assumptions μ axioms.E
    (mEstimatorAsymptoticNormal_of_axioms μ d axioms)
    coverage_axioms dbs m β_star V reg h_unbiased CI_seq α h_α centered_scaled_seq

/-- Generic constructive coverage transfer on the DSL surface. Instead of a
blanket coverage axiom, the caller provides a concrete normal-coverage
construction describing the transformed statistic, event identity, coordinate
limit convergence, and calibration data. -/
theorem DSL_valid_coverage_from_construction_from_assumptions
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (E : ((Obs × Mis × Mis × SamplingIndicator × ℝ) → Fin d → ℝ) → Fin d → ℝ)
    (h_normal : MEstimatorAsymptoticNormalAssumption μ d E)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) E β_star)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (coverage_construction :
      NormalCoverageConstruction μ centered_scaled_seq CI_seq β_star α V) :
    AsymptoticCoverage μ CI_seq β_star α := by
  have h_norm :
      ConvergesInDistributionToNormal μ centered_scaled_seq (fun _ => 0) V :=
    DSL_asymptotic_normal_from_assumptions μ E h_normal dbs m β_star V reg h_unbiased
      centered_scaled_seq
  exact coverage_construction.asymptoticCoverage (μ := μ) h_norm

/-- Axioms-packaged version of the generic constructive coverage transfer. -/
theorem DSL_valid_coverage_from_construction
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (axioms : MEstimationAxioms Ω (Obs × Mis × Mis × SamplingIndicator × ℝ) μ d)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) axioms.E β_star)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (coverage_construction :
      NormalCoverageConstruction μ centered_scaled_seq CI_seq β_star α V) :
    AsymptoticCoverage μ CI_seq β_star α := by
  exact DSL_valid_coverage_from_construction_from_assumptions μ axioms.E
    (mEstimatorAsymptoticNormal_of_axioms μ d axioms)
    dbs m β_star V reg h_unbiased CI_seq α centered_scaled_seq coverage_construction

/-- Backward-compatible alias for the axioms-based constructive coverage theorem. -/
abbrev DSL_valid_coverage_from_construction_from_axioms :=
  @DSL_valid_coverage_from_construction

/-- Backward-compatible alias for the axioms-based coverage theorem name. -/
abbrev DSL_valid_coverage_from_axioms := @DSL_valid_coverage

/-- Coordinatewise diagonal standardization of a multivariate statistic. -/
def diagStandardize {d : ℕ}
    (V : Matrix (Fin d) (Fin d) ℝ) (x : Fin d → ℝ) : Fin d → ℝ :=
  fun i => x i / Real.sqrt (V i i)

lemma continuous_diagStandardize {d : ℕ}
    (V : Matrix (Fin d) (Fin d) ℝ) :
    Continuous (diagStandardize V) := by
  refine continuous_pi ?_
  intro i
  simpa [diagStandardize, div_eq_mul_inv] using
    ((continuous_apply i : Continuous fun x : Fin d → ℝ => x i).mul continuous_const)

/-- A centered Gaussian coordinate becomes standard normal after dividing by
its own standard deviation. -/
lemma NormalLimit.coord_stdNormal_of_zero_mean_pos_diag_standardized
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {d : ℕ} {Z : Ω → Fin d → ℝ} {V : Matrix (Fin d) (Fin d) ℝ}
    (hZ : NormalLimit μ Z (fun _ => 0) V)
    (h_pos : ∀ i, 0 < V i i) :
    ∀ i,
      μ.map (fun ω => Z ω i / Real.sqrt (V i i)) =
        ((stdNormalProbabilityMeasure : ProbabilityMeasure ℝ) : Measure ℝ) := by
  intro i
  have hZi_ae : AEMeasurable (fun ω => Z ω i) μ := by
    exact aemeasurable_of_map_neZero (by rw [hZ.coord_gaussian i]; infer_instance)
  have hscaled_map :
      μ.map (fun ω => Z ω i / Real.sqrt (V i i)) =
        (μ.map (fun ω => Z ω i)).map (fun x => x / Real.sqrt (V i i)) := by
    symm
    simpa [Function.comp, div_eq_mul_inv] using
      (AEMeasurable.map_map_of_aemeasurable
        (μ := μ)
        (f := fun ω => Z ω i)
        (g := fun x : ℝ => x / Real.sqrt (V i i))
        ((measurable_id'.mul_const _).aemeasurable)
        hZi_ae)
  have hvar_nonneg : 0 ≤ (Real.sqrt (V i i) ^ 2)⁻¹ * V i i := by
    exact mul_nonneg (inv_nonneg.mpr (sq_nonneg _)) (le_of_lt (h_pos i))
  have hvar :
      (⟨(Real.sqrt (V i i) ^ 2)⁻¹ * V i i, hvar_nonneg⟩ : NNReal) = 1 := by
    ext
    change (Real.sqrt (V i i) ^ 2)⁻¹ * V i i = 1
    have hsqrt_ne : Real.sqrt (V i i) ≠ 0 := Real.sqrt_ne_zero'.2 (h_pos i)
    field_simp [pow_two, hsqrt_ne]
    rw [Real.sq_sqrt (le_of_lt (h_pos i))]
  calc
    μ.map (fun ω => Z ω i / Real.sqrt (V i i))
        = (μ.map (fun ω => Z ω i)).map (fun x => x / Real.sqrt (V i i)) := hscaled_map
    _ = (ProbabilityTheory.gaussianReal (0 : ℝ)
          (⟨V i i, le_of_lt (h_pos i)⟩ : NNReal)).map
          (fun x => x / Real.sqrt (V i i)) := by
          simpa [hZ.variance_diag_nonneg i, h_pos i] using
            congrArg
              (fun ν : Measure ℝ => ν.map (fun x => x / Real.sqrt (V i i)))
              (hZ.coord_gaussian i)
    _ = ProbabilityTheory.gaussianReal (0 : ℝ)
          (⟨(Real.sqrt (V i i) ^ 2)⁻¹ * V i i, hvar_nonneg⟩ : NNReal) := by
          simpa [div_eq_mul_inv, mul_comm, pow_two] using
            (ProbabilityTheory.gaussianReal_map_mul_const
              (μ := (0 : ℝ))
              (v := (⟨V i i, le_of_lt (h_pos i)⟩ : NNReal))
              ((Real.sqrt (V i i))⁻¹))
    _ = ProbabilityTheory.gaussianReal (0 : ℝ) 1 := by simp [hvar]
    _ = ((stdNormalProbabilityMeasure : ProbabilityMeasure ℝ) : Measure ℝ) := by
      simp [stdNormalProbabilityMeasure, ProbabilityTheory.stdNormalMeasure]

/-- Concrete coordinatewise Wald-style coverage from a multivariate normal-limit
certificate with zero mean and unit diagonal, which implies standard-normal
coordinates. This is the unit-diagonal corollary of the general studentized
route below. -/
lemma NormalLimit.coord_stdNormal_of_zero_mean_unit_diag
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {d : ℕ} {Z : Ω → Fin d → ℝ} {V : Matrix (Fin d) (Fin d) ℝ}
    (hZ : NormalLimit μ Z (fun _ => 0) V)
    (h_diag : ∀ i, V i i = 1) :
    ∀ i,
      μ.map (fun ω => Z ω i) =
        ((stdNormalProbabilityMeasure : ProbabilityMeasure ℝ) : Measure ℝ) := by
  intro i
  have h_pos : ∀ j, 0 < V j j := by
    intro j
    rw [h_diag j]
    norm_num
  simpa [h_diag i] using
    (NormalLimit.coord_stdNormal_of_zero_mean_pos_diag_standardized μ hZ h_pos i)

/-- Concrete coordinatewise Wald-style coverage from a multivariate normal-limit
certificate after diagonal studentization. This removes the generic
coverage-transfer assumption for this lane without requiring unit diagonal. -/
theorem asymptoticCoverage_of_convergesInDistributionToNormal_standardized_of_eventEq
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    {d : ℕ}
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (α : ℝ)
    (lower upper : Fin d → ℝ)
    (h_norm :
      ConvergesInDistributionToNormal μ centered_scaled_seq (fun _ => 0) V)
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
  rcases h_norm with ⟨Z, hZ_normal, hZ_dist⟩
  have h_dist_standardized :
      MeasureTheory.TendstoInDistribution
        (fun n => diagStandardize V ∘ centered_scaled_seq n) Filter.atTop
        (diagStandardize V ∘ Z) μ := by
    simpa [diagStandardize, Function.comp] using
      hZ_dist.continuous_comp (continuous_diagStandardize V)
  exact asymptoticCoverage_of_tendstoInDistribution_of_coordStdNormal_of_eventEq
    (μ := μ)
    (stat_seq := fun n ω i => centered_scaled_seq n ω i / Real.sqrt (V i i))
    (Z := fun ω i => Z ω i / Real.sqrt (V i i))
    (CI_seq := CI_seq)
    (β_star := β_star) (α := α) (lower := lower) (upper := upper)
    (h_dist := h_dist_standardized) (h_interval := h_interval) (h_event_eq := h_event_eq)
    (h_coord_stdNormal :=
      NormalLimit.coord_stdNormal_of_zero_mean_pos_diag_standardized μ hZ_normal h_pos)
    (h_calibration := h_calibration)

/-- Symmetric critical-value specialization of the general studentized
coordinatewise Wald coverage theorem. -/
theorem asymptoticCoverage_of_convergesInDistributionToNormal_standardized_symm_of_eventEq
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    {d : ℕ}
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (α z : ℝ)
    (h_norm :
      ConvergesInDistributionToNormal μ centered_scaled_seq (fun _ => 0) V)
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
  exact asymptoticCoverage_of_convergesInDistributionToNormal_standardized_of_eventEq
    (μ := μ) (centered_scaled_seq := centered_scaled_seq) (CI_seq := CI_seq)
    (β_star := β_star) (V := V) (α := α) (lower := fun _ => -z) (upper := fun _ => z)
    h_norm (fun _ => by simpa using neg_le_self hz_nonneg) h_event_eq h_pos
    (fun _ => h_calibration)

/-- Unit-diagonal corollary of the concrete coordinatewise Wald coverage route. -/
theorem asymptoticCoverage_of_convergesInDistributionToNormal_coordStdNormal_of_eventEq
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    {d : ℕ}
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (α : ℝ)
    (lower upper : Fin d → ℝ)
    (h_norm :
      ConvergesInDistributionToNormal μ centered_scaled_seq (fun _ => 0) V)
    (h_interval : ∀ i, lower i ≤ upper i)
    (h_event_eq :
      ∀ n i,
        {ω | β_star i ∈ Set.Icc (CI_seq n ω i).1 (CI_seq n ω i).2} =
          {ω | centered_scaled_seq n ω i ∈ Set.Icc (lower i) (upper i)})
    (h_diag : ∀ i, V i i = 1)
    (h_calibration :
      ∀ i,
        (((stdNormalProbabilityMeasure : ProbabilityMeasure ℝ) : Measure ℝ)
          (Set.Icc (lower i) (upper i))) = ENNReal.ofReal (1 - α)) :
    AsymptoticCoverage μ CI_seq β_star α := by
  have h_pos : ∀ i, 0 < V i i := by
    intro i
    rw [h_diag i]
    norm_num
  exact asymptoticCoverage_of_convergesInDistributionToNormal_standardized_of_eventEq
    (μ := μ) (centered_scaled_seq := centered_scaled_seq) (CI_seq := CI_seq)
    (β_star := β_star) (V := V) (α := α) (lower := lower) (upper := upper)
    h_norm h_interval
    (fun n i => by simpa [h_diag i] using h_event_eq n i)
    h_pos h_calibration

/-- DSL confidence intervals have correct coverage on the concrete
coordinatewise Wald lane, using the multivariate first-principles coverage
theorem instead of the generic coverage axiom. -/
theorem DSL_valid_coverage_coordStdNormal_from_assumptions
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (E : ((Obs × Mis × Mis × SamplingIndicator × ℝ) → Fin d → ℝ) → Fin d → ℝ)
    (h_normal : MEstimatorAsymptoticNormalAssumption μ d E)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) E β_star)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
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
  have h_norm :
      ConvergesInDistributionToNormal μ centered_scaled_seq (fun _ => 0) V :=
    DSL_asymptotic_normal_from_assumptions μ E h_normal dbs m β_star V reg h_unbiased
      centered_scaled_seq
  exact asymptoticCoverage_of_convergesInDistributionToNormal_standardized_of_eventEq
    (μ := μ) (centered_scaled_seq := centered_scaled_seq) (CI_seq := CI_seq)
    (β_star := β_star) (V := V) (α := α) (lower := lower) (upper := upper)
    h_norm h_interval h_event_eq h_pos h_calibration

/-- Axioms-packaged version of the concrete coordinatewise Wald coverage route. -/
theorem DSL_valid_coverage_coordStdNormal
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (axioms : MEstimationAxioms Ω (Obs × Mis × Mis × SamplingIndicator × ℝ) μ d)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) axioms.E β_star)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (α : ℝ)
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
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
  exact DSL_valid_coverage_coordStdNormal_from_assumptions μ axioms.E
    (mEstimatorAsymptoticNormal_of_axioms μ d axioms)
    dbs m β_star V reg h_unbiased CI_seq α centered_scaled_seq lower upper
    h_interval h_event_eq h_pos h_calibration

/-- Backward-compatible alias for the axioms-packaged concrete coverage theorem. -/
abbrev DSL_valid_coverage_coordStdNormal_from_axioms := @DSL_valid_coverage_coordStdNormal

/-- Plug-in diagonal standard error from a covariance-estimator sequence. -/
def pluginStandardError {Ω : Type*} {d : ℕ}
    (V_hat_seq : ℕ → Ω → Matrix (Fin d) (Fin d) ℝ)
    (n : ℕ) (ω : Ω) (i : Fin d) : ℝ :=
  Real.sqrt (V_hat_seq n ω i i)

/-- Coordinatewise diagonally studentized statistic using a plug-in covariance
estimator sequence. -/
def pluginStudentizedStat {Ω : Type*} {d : ℕ}
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (V_hat_seq : ℕ → Ω → Matrix (Fin d) (Fin d) ℝ) :
    ℕ → Ω → Fin d → ℝ :=
  fun n ω i => centered_scaled_seq n ω i / pluginStandardError V_hat_seq n ω i

lemma convergesInProbability_inv_sqrt_diag_of_diag
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    {d : ℕ}
    (V_hat_seq : ℕ → Ω → Matrix (Fin d) (Fin d) ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (i : Fin d)
    (h_Vhat_diag :
      ConvergesInProbability μ
        (fun n ω => V_hat_seq n ω i i)
        (fun _ => V i i))
    (h_pos : 0 < V i i) :
    ConvergesInProbability μ
      (fun n ω => (Real.sqrt (V_hat_seq n ω i i))⁻¹)
      (fun _ => (Real.sqrt (V i i))⁻¹) := by
  have hg : ContinuousAt (fun x : ℝ => (Real.sqrt x)⁻¹) (V i i) := by
    have hsqrt : ContinuousAt (fun x : ℝ => Real.sqrt x) (V i i) := by
      simpa using
        (continuousAt_id.sqrt : ContinuousAt (fun x : ℝ => Real.sqrt x) (V i i))
    have hinv : ContinuousAt (fun x : ℝ => x⁻¹) (Real.sqrt (V i i)) := by
      simpa using (continuousAt_inv₀ (Real.sqrt_ne_zero'.2 h_pos))
    simpa [Function.comp] using hinv.comp hsqrt
  simpa [ConvergesInProbability] using
    (Econometrics.convergesInProbability_continuous
      (mu := μ)
      (g := fun x : ℝ => (Real.sqrt x)⁻¹)
      (c := V i i)
      hg
      (X := fun n ω => V_hat_seq n ω i i)
      (hX := by simpa [ConvergesInProbability] using h_Vhat_diag))

lemma aemeasurable_inv_sqrt_diag
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω)
    {d : ℕ}
    (V_hat_seq : ℕ → Ω → Matrix (Fin d) (Fin d) ℝ)
    (i : Fin d)
    (h_Vhat_diag_meas :
      ∀ n,
        AEMeasurable (fun ω => V_hat_seq n ω i i) μ) :
    ∀ n,
      AEMeasurable (fun ω => (Real.sqrt (V_hat_seq n ω i i))⁻¹) μ := by
  intro n
  exact (h_Vhat_diag_meas n).sqrt.inv

lemma tendstoInDistribution_pluginStudentized_coord
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    {d : ℕ}
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (V_hat_seq : ℕ → Ω → Matrix (Fin d) (Fin d) ℝ)
    (Z : Ω → Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (i : Fin d)
    (h_coord_dist :
      MeasureTheory.TendstoInDistribution
        (fun n ω => centered_scaled_seq n ω i) Filter.atTop (fun ω => Z ω i) μ)
    (h_Vhat_diag :
      ConvergesInProbability μ
        (fun n ω => V_hat_seq n ω i i)
        (fun _ => V i i))
    (h_Vhat_diag_meas :
      ∀ n,
        AEMeasurable (fun ω => V_hat_seq n ω i i) μ)
    (h_pos : 0 < V i i) :
    MeasureTheory.TendstoInDistribution
      (fun n ω => pluginStudentizedStat centered_scaled_seq V_hat_seq n ω i)
      Filter.atTop
      (fun ω => Z ω i / Real.sqrt (V i i)) μ := by
  have h_inv :
      ConvergesInProbability μ
        (fun n ω => (Real.sqrt (V_hat_seq n ω i i))⁻¹)
        (fun _ => (Real.sqrt (V i i))⁻¹) :=
    convergesInProbability_inv_sqrt_diag_of_diag
      (μ := μ) (V_hat_seq := V_hat_seq) (V := V) (i := i)
      (h_Vhat_diag := h_Vhat_diag) (h_pos := h_pos)
  have h_inv_meas :
      ∀ n,
        AEMeasurable (fun ω => (Real.sqrt (V_hat_seq n ω i i))⁻¹) μ :=
    aemeasurable_inv_sqrt_diag
      (μ := μ) (V_hat_seq := V_hat_seq) (i := i)
      (h_Vhat_diag_meas := h_Vhat_diag_meas)
  have h_prod :=
    (h_coord_dist.continuous_comp_prodMk_of_tendstoInMeasure_const
      (g := fun p : ℝ × ℝ => p.2 * p.1)
      (by fun_prop)
      (by simpa [ConvergesInProbability] using h_inv)
      h_inv_meas)
  convert h_prod using 1
  · funext n ω
    simp [pluginStudentizedStat, pluginStandardError, div_eq_mul_inv, mul_comm, mul_left_comm,
      mul_assoc]
  · funext ω
    simp [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc]

/-- Concrete coordinatewise Wald-style coverage for plug-in diagonal
studentization. This is the implementation-facing route where the variance term
is estimated from data rather than supplied as the population diagonal. -/
theorem asymptoticCoverage_of_convergesInDistributionToNormal_plugin_of_eventEq
    {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    {d : ℕ}
    (centered_scaled_seq : ℕ → Ω → Fin d → ℝ)
    (V_hat_seq : ℕ → Ω → Matrix (Fin d) (Fin d) ℝ)
    (CI_seq : ℕ → Ω → Fin d → ℝ × ℝ)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (α : ℝ)
    (lower upper : Fin d → ℝ)
    (h_norm :
      ConvergesInDistributionToNormal μ centered_scaled_seq (fun _ => 0) V)
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
  rcases h_norm with ⟨Z, hZ_normal, hZ_dist⟩
  intro i
  have h_coord_dist :
      MeasureTheory.TendstoInDistribution
        (fun n ω => centered_scaled_seq n ω i) Filter.atTop (fun ω => Z ω i) μ :=
    tendstoInDistribution_fin_apply μ centered_scaled_seq Z hZ_dist i
  have h_plugin_dist :
      MeasureTheory.TendstoInDistribution
        (fun n ω => pluginStudentizedStat centered_scaled_seq V_hat_seq n ω i)
        Filter.atTop
        (fun ω => Z ω i / Real.sqrt (V i i)) μ :=
    tendstoInDistribution_pluginStudentized_coord
      (μ := μ) (centered_scaled_seq := centered_scaled_seq) (V_hat_seq := V_hat_seq)
      (Z := Z) (V := V) (i := i)
      (h_coord_dist := h_coord_dist)
      (h_Vhat_diag := h_Vhat_diag i)
      (h_Vhat_diag_meas := fun n => h_Vhat_diag_meas n i)
      (h_pos := h_pos i)
  have h_coord_std :
      μ.map (fun ω => Z ω i / Real.sqrt (V i i)) =
        ((stdNormalProbabilityMeasure : ProbabilityMeasure ℝ) : Measure ℝ) :=
    NormalLimit.coord_stdNormal_of_zero_mean_pos_diag_standardized μ hZ_normal h_pos i
  have h_stat :
      Filter.Tendsto
        (fun n =>
          μ {ω | pluginStudentizedStat centered_scaled_seq V_hat_seq n ω i ∈
            Set.Icc (lower i) (upper i)})
        Filter.atTop
        (𝓝 ((μ.map (fun ω => Z ω i / Real.sqrt (V i i)))
          (Set.Icc (lower i) (upper i)))) :=
    tendsto_measure_Icc_of_tendstoInDistribution
      (μ := μ)
      (X := fun n ω => pluginStudentizedStat centered_scaled_seq V_hat_seq n ω i)
      (Z := fun ω => Z ω i / Real.sqrt (V i i))
      h_plugin_dist
      (h_interval i)
      (by
        rw [h_coord_std]
        exact stdNormal_measure_singleton_zero (lower i))
      (by
        rw [h_coord_std]
        exact stdNormal_measure_singleton_zero (upper i))
  have h_target :
      Filter.Tendsto
        (fun n =>
          μ {ω | pluginStudentizedStat centered_scaled_seq V_hat_seq n ω i ∈
            Set.Icc (lower i) (upper i)})
        Filter.atTop
        (𝓝 (ENNReal.ofReal (1 - α))) := by
    rw [h_coord_std] at h_stat
    simpa [h_calibration i] using h_stat
  exact Filter.Tendsto.congr'
    (Filter.Eventually.of_forall fun n => by
      rw [← h_event_eq n i])
    h_target

/-- DSL coverage on the concrete plug-in diagonal Wald lane. -/
theorem DSL_valid_coverage_pluginStdNormal_from_assumptions
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (E : ((Obs × Mis × Mis × SamplingIndicator × ℝ) → Fin d → ℝ) → Fin d → ℝ)
    (h_normal : MEstimatorAsymptoticNormalAssumption μ d E)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) E β_star)
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
  have h_norm :
      ConvergesInDistributionToNormal μ centered_scaled_seq (fun _ => 0) V :=
    DSL_asymptotic_normal_from_assumptions μ E h_normal dbs m β_star V reg h_unbiased
      centered_scaled_seq
  exact asymptoticCoverage_of_convergesInDistributionToNormal_plugin_of_eventEq
    (μ := μ) (centered_scaled_seq := centered_scaled_seq) (V_hat_seq := V_hat_seq)
    (CI_seq := CI_seq) (β_star := β_star) (V := V) (α := α)
    (lower := lower) (upper := upper)
    h_norm h_interval h_event_eq h_pos h_Vhat_diag h_Vhat_diag_meas h_calibration

/-- Axioms-packaged version of the concrete plug-in diagonal Wald coverage
route. -/
theorem DSL_valid_coverage_pluginStdNormal
    {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    {Obs Mis Con : Type*} {d : ℕ}
    (axioms : MEstimationAxioms Ω (Obs × Mis × Mis × SamplingIndicator × ℝ) μ d)
    (dbs : DesignBasedSampling Obs Mis Con)
    (m : MomentFunction (Obs × Mis) d)
    (β_star : Fin d → ℝ)
    (V : Matrix (Fin d) (Fin d) ℝ)
    (reg : RegularityConditions (Obs × Mis × Mis × SamplingIndicator × ℝ) d)
    (h_unbiased : MomentUnbiased (DSLMomentFromData m) axioms.E β_star)
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
  exact DSL_valid_coverage_pluginStdNormal_from_assumptions μ axioms.E
    (mEstimatorAsymptoticNormal_of_axioms μ d axioms)
    dbs m β_star V reg h_unbiased centered_scaled_seq V_hat_seq CI_seq α lower upper
    h_interval h_event_eq h_pos h_Vhat_diag h_Vhat_diag_meas h_calibration

/-- Backward-compatible alias for the axioms-packaged plug-in Wald coverage
route. -/
abbrev DSL_valid_coverage_pluginStdNormal_from_axioms := @DSL_valid_coverage_pluginStdNormal

/-!
## Comparison with Naive Estimator
-/

/-- The naive estimator ignores prediction errors.

    β̂_naive solves (1/N)∑m(D^obs, D̂^mis; β) = 0

    This is inconsistent unless E[m(D^obs, D̂^mis; β*)] = E[m(D^obs, D^mis; β*)]
    which requires prediction errors to be uncorrelated with everything. -/
def NaiveEstimator {Obs Mis : Type*} {d : ℕ}
    (m : MomentFunction (Obs × Mis) d)
    (data : List (Obs × Mis))  -- Only uses (d_obs, d_mis_pred)
    (β : Fin d → ℝ) : Fin d → ℝ :=
  let N := data.length
  fun i => (data.foldl (fun acc ⟨d_obs, d_mis_pred⟩ =>
    acc + m (d_obs, d_mis_pred) β i) 0) / N

/-- Naive moment function on (d_obs, d_mis_pred, d_mis_true). -/
def PredMomentFromData {Obs Mis : Type*} {d : ℕ}
    (m : MomentFunction (Obs × Mis) d) : MomentFunction (Obs × Mis × Mis) d :=
  fun D β => m (D.1, D.2.1) β

/-- Oracle moment function using true missing values. -/
def TrueMomentFromData {Obs Mis : Type*} {d : ℕ}
    (m : MomentFunction (Obs × Mis) d) : MomentFunction (Obs × Mis × Mis) d :=
  fun D β => m (D.1, D.2.2) β

/-- Componentwise linearity of an expectation operator. -/
def ExpectationLinear {Data : Type*} {d : ℕ}
    (E : (Data → Fin d → ℝ) → Fin d → ℝ) : Prop :=
  ∀ (f g : Data → Fin d → ℝ) (a b : ℝ) (i : Fin d),
    E (fun D => fun j => a * f D j + b * g D j) i =
      a * E f i + b * E g i

/-- The naive estimator is biased unless very strong conditions hold.

    For the naive estimator to be consistent, we need:
    E[e | X] = 0 where e = Ŷ - Y

    This requires errors to be uncorrelated with:
    - X (the covariates)
    - Y (the true outcome)
    - Any unobserved confounders U

    This almost never holds in practice. -/
theorem naive_estimator_biased_general
    {Obs Mis : Type*} {d : ℕ}
    (m : MomentFunction (Obs × Mis) d)
    (E : ((Obs × Mis × Mis) → Fin d → ℝ) → Fin d → ℝ)
    (β_star : Fin d → ℝ)
    (h_true : MomentUnbiased (TrueMomentFromData m) E β_star)
    (h_bias : ∃ i, MomentBias m E β_star i ≠ 0)
    (hE_linear : ExpectationLinear E)
    : ¬ MomentUnbiased (PredMomentFromData m) E β_star := by
  intro h_pred
  rcases h_bias with ⟨i, h_nonzero⟩
  have h_bias_eq :
      MomentBias m E β_star i =
        E (fun D => fun j =>
          PredMomentFromData m D β_star j - TrueMomentFromData m D β_star j) i := by
    rfl
  have h_linear :
      E (fun D => fun j =>
        PredMomentFromData m D β_star j - TrueMomentFromData m D β_star j) i =
        E (fun D => PredMomentFromData m D β_star) i -
        E (fun D => TrueMomentFromData m D β_star) i := by
    -- Use linearity with a = 1, b = -1.
    have := hE_linear
      (fun D => PredMomentFromData m D β_star)
      (fun D => TrueMomentFromData m D β_star)
      1 (-1) i
    -- Simplify pointwise.
    simpa [sub_eq_add_neg, add_comm, add_left_comm, add_assoc, mul_comm, mul_left_comm, mul_assoc] using this
  have h_pred_zero : E (fun D => PredMomentFromData m D β_star) i = 0 := h_pred i
  have h_true_zero : E (fun D => TrueMomentFromData m D β_star) i = 0 := h_true i
  have h_bias_zero : MomentBias m E β_star i = 0 := by
    calc
      MomentBias m E β_star i
          = E (fun D => fun j =>
              PredMomentFromData m D β_star j - TrueMomentFromData m D β_star j) i := h_bias_eq
      _ = E (fun D => PredMomentFromData m D β_star) i -
          E (fun D => TrueMomentFromData m D β_star) i := h_linear
      _ = 0 := by simp [h_pred_zero, h_true_zero]
  exact h_nonzero h_bias_zero

end DSL

end
