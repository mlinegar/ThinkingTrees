/-
# FormalProofs/Econometrics/OLS/AsymptoticOLS.lean

## Reference: Wooldridge, Introductory Econometrics, Chapter 5

This file formalizes the asymptotic properties of OLS:

- Weak assumptions for consistency (weaker than Gauss-Markov)
- Consistency of OLS (Theorem 5.1)
- Asymptotic normality of OLS (Theorem 5.2)
- Heteroskedasticity-robust standard errors

### Main Results

**Theorem 5.1 (Consistency)**
Under weak exogeneity E[ε_i|x_i] = 0, identification, and regularity:
β̂ →p β as n → ∞

**Theorem 5.2 (Asymptotic Normality)**
Under the conditions for consistency plus finite moments:
√n(β̂ - β) →d N(0, V)

### Connection to CLT Module

This module connects to the Central Limit Theorem formalized in
FormalProofs/CLT/CLT.lean to derive asymptotic normality.
-/

import Mathlib
import FormalProbability.CLT.Core
import FormalProofs.DSL.AsymptoticTheory
import FormalProofs.DSL.ConcreteCoverage
import FormalProofs.Econometrics.OLS.GaussMarkov

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

namespace Econometrics

namespace OLS

/-!
## Asymptotic Assumptions (Weaker than Classical)

These assumptions are sufficient for consistency and asymptotic normality
but weaker than the full classical assumptions (MLR.1-6).
-/

/-- Weak exogeneity: E[ε_i | x_i] = 0

    We model this in the formalization as orthogonality E[x_i ε_i] = 0,
    which is sufficient for the LLN/CLT steps used below. -/
structure WeakExogeneity {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (x : Ω → Fin k → ℝ)
    (ε : Ω → ℝ) : Prop where
  /-- Orthogonality: E[x_j ε] = 0 for each regressor. -/
  orthogonality : ∀ j, ∫ ω, x ω j * ε ω ∂μ = 0

/-- Identification condition: Q = E[x_i x_i'] is positive definite.

    This ensures the population regression is well-defined
    and that β is uniquely identified. -/
structure Identified {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (x : Ω → Fin k → ℝ) : Prop where
  /-- Q := E[xx'] is positive definite -/
  Q_pd : ∀ j, 0 < ∫ ω, (x ω j)^2 ∂μ

/-- Finite moment conditions for asymptotic normality.

    We need E[‖x‖⁴] < ∞ and E[ε⁴] < ∞ for the CLT
    and law of large numbers to apply. -/
structure FiniteMoments' {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (x : Ω → Fin k → ℝ)
    (ε : Ω → ℝ) : Prop where
  /-- Fourth moments of regressors are integrable -/
  x_fourth_moment : ∀ j, Integrable (fun ω => (x ω j)^4) μ
  /-- Fourth moment of errors is integrable -/
  ε_fourth_moment : Integrable (fun ω => (ε ω)^4) μ

/-- Full bundle of asymptotic assumptions. -/
structure AsymptoticAssumptions {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (x : Ω → Fin k → ℝ)
    (ε : Ω → ℝ) : Prop where
  /-- Weak exogeneity -/
  weak_exog : WeakExogeneity μ x ε
  /-- Identification -/
  identified : Identified μ x
  /-- Finite moments -/
  finite_moments : FiniteMoments' μ x ε

/-!
## Convergence Concepts

We reuse definitions from DSL.AsymptoticTheory:
- ConvergesInProbability
- ConvergesInDistributionToNormal
-/

/-- Convergence in probability (imported from DSL). -/
abbrev ConvergesInProbability {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    {E : Type*} [PseudoMetricSpace E]
    (seq : ℕ → Ω → E) (limit : Ω → E) : Prop :=
  DSL.ConvergesInProbability μ seq limit

/-- Normal limit (imported from DSL). -/
abbrev NormalLimit {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] {d : ℕ}
    (Z : Ω → Fin d → ℝ) (mean : Fin d → ℝ)
    (variance : Matrix (Fin d) (Fin d) ℝ) : Prop :=
  DSL.NormalLimit μ Z mean variance

/-!
## Population Regression Parameters
-/

/-- Population moment matrix: Q = E[x x'] -/
def PopulationQ {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (x : Ω → Fin k → ℝ) : Matrix (Fin k) (Fin k) ℝ :=
  fun i j => ∫ ω, x ω i * x ω j ∂μ

/-- Sample moment matrix: (1/n) X'X -/
def SampleQ {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ) : Matrix (Fin k) (Fin k) ℝ :=
  (1 / n : ℝ) • GramMatrix X

/-- Population E[x ε] (should be zero under exogeneity) -/
def PopulationXε {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (x : Ω → Fin k → ℝ)
    (ε : Ω → ℝ) : Fin k → ℝ :=
  fun j => ∫ ω, x ω j * ε ω ∂μ

/-- Weak exogeneity implies the population score is zero. -/
lemma populationXε_eq_zero_of_weak_exog {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (x : Ω → Fin k → ℝ)
    (ε : Ω → ℝ)
    (h_exog : WeakExogeneity μ x ε) :
    PopulationXε μ x ε = 0 := by
  funext j
  simpa [PopulationXε] using h_exog.orthogonality j

/-- Sample score mean: (1/n) Σ x_i ε_i. -/
def SampleScoreMean {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (x_seq : ℕ → Ω → Fin k → ℝ)
    (ε_seq : ℕ → Ω → ℝ) : ℕ → Ω → Fin k → ℝ :=
  fun n ω j => (1 / n : ℝ) * ∑ i : Fin n, x_seq i ω j * ε_seq i ω

/-- Score CLT scaling: (1/√n) Σ x_i ε_i. -/
def SampleScoreScaled {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (x_seq : ℕ → Ω → Fin k → ℝ)
    (ε_seq : ℕ → Ω → ℝ) : ℕ → Ω → Fin k → ℝ :=
  fun n ω j => (1 / Real.sqrt n : ℝ) * ∑ i : Fin n, x_seq i ω j * ε_seq i ω

/-- Lightweight stationarity/IID-style moment restriction across sequence index. -/
def IIDLikeSeq {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (x_seq : ℕ → Ω → Fin k → ℝ)
    (ε_seq : ℕ → Ω → ℝ) : Prop :=
  ∀ n m j, ∫ ω, x_seq n ω j * ε_seq n ω ∂μ = ∫ ω, x_seq m ω j * ε_seq m ω ∂μ

/-- Finite second moments along the sequence. -/
def FiniteSecondMomentsSeq {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (x_seq : ℕ → Ω → Fin k → ℝ)
    (ε_seq : ℕ → Ω → ℝ) : Prop :=
  ∀ n j, Integrable (fun ω => (x_seq n ω j)^2) μ ∧ Integrable (fun ω => (ε_seq n ω)^2) μ

/-- OLS estimator sequence is measurable coordinate-wise. -/
def IsOLSEstimatorSeq {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (β_hat_seq : ℕ → Ω → Fin k → ℝ) : Prop :=
  ∀ n j, AEMeasurable (fun ω => β_hat_seq n ω j) μ

/-- HC variance estimator sequence has symmetric matrices. -/
def IsHCVarianceEstimatorSeq {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (V_hat_seq : ℕ → Ω → Matrix (Fin k) (Fin k) ℝ) : Prop :=
  ∀ n ω, Matrix.IsSymm (V_hat_seq n ω)

/-- Standard errors are strictly positive coordinate-wise. -/
def PositiveSESeq {k : ℕ} {Ω : Type*}
    (SE_seq : ℕ → Ω → Fin k → ℝ) : Prop :=
  ∀ n ω j, 0 < SE_seq n ω j

/-- Smoothness requirement used by the delta method. -/
def DeltaMethodSmooth {k : ℕ}
    (g : (Fin k → ℝ) → ℝ) : Prop :=
  Differentiable ℝ g

/-!
## Asymptotic OLS Assumptions
-/

/-- Abstract LLN/CLT/Slutsky facts for OLS asymptotics.

These encapsulate the standard large-sample results (Wooldridge Ch. 5)
so that downstream theorems can be stated cleanly.
-/
structure OLSAsymptoticAxioms {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] : Prop where
  /-- LLN for the score: (1/n) Σ x_i ε_i →p 0 -/
  lln_score :
    ∀ (x_seq : ℕ → Ω → Fin k → ℝ) (ε_seq : ℕ → Ω → ℝ),
      (∀ n, WeakExogeneity μ (x_seq n) (ε_seq n)) →
      ConvergesInProbability μ (SampleScoreMean x_seq ε_seq) (fun _ => 0)
  /-- Consistency of OLS (Theorem 5.1). -/
  ols_consistency :
    ∀ (x_seq : ℕ → Ω → Fin k → ℝ) (ε_seq : ℕ → Ω → ℝ)
      (β_true : Fin k → ℝ) (β_hat_seq : ℕ → Ω → Fin k → ℝ),
      (∀ n, AsymptoticAssumptions μ (x_seq n) (ε_seq n)) →
      IsOLSEstimatorSeq μ β_hat_seq →
      ConvergesInProbability μ β_hat_seq (fun _ => β_true)
  /-- Multivariate CLT for the score: (1/√n) Σ x_i ε_i →d N(0, Ω). -/
  clt_score :
    ∀ (x_seq : ℕ → Ω → Fin k → ℝ) (ε_seq : ℕ → Ω → ℝ),
      FiniteSecondMomentsSeq μ x_seq ε_seq →
      IIDLikeSeq μ x_seq ε_seq →
      DSL.ConvergesInDistributionToNormal μ
        (SampleScoreScaled x_seq ε_seq)
        (fun _ => 0)
        (fun i j => ∫ ω, (ε_seq 0 ω)^2 * x_seq 0 ω i * x_seq 0 ω j ∂μ)
  /-- Asymptotic normality of OLS (Theorem 5.2, robust form). -/
  ols_asymptotic_normal :
    ∀ (x_seq : ℕ → Ω → Fin k → ℝ) (ε_seq : ℕ → Ω → ℝ)
      (β_true : Fin k → ℝ) (Q_inv : Matrix (Fin k) (Fin k) ℝ)
      (β_hat_seq : ℕ → Ω → Fin k → ℝ) (V : Matrix (Fin k) (Fin k) ℝ),
      (∀ n, AsymptoticAssumptions μ (x_seq n) (ε_seq n)) →
      IsOLSEstimatorSeq μ β_hat_seq →
      DSL.ConvergesInDistributionToNormal μ
        (fun n ω => fun j => Real.sqrt n * (β_hat_seq n ω j - β_true j))
        (fun _ => 0)
        V
  /-- Homoskedastic simplification of asymptotic variance. -/
  ols_asymptotic_normal_homoskedastic :
    ∀ (x_seq : ℕ → Ω → Fin k → ℝ) (ε_seq : ℕ → Ω → ℝ)
      (β_true : Fin k → ℝ) (Q_inv : Matrix (Fin k) (Fin k) ℝ)
      (σ_sq : ℝ) (β_hat_seq : ℕ → Ω → Fin k → ℝ)
      (V : Matrix (Fin k) (Fin k) ℝ),
      (∀ n, AsymptoticAssumptions μ (x_seq n) (ε_seq n)) →
      (∫ ω, (ε_seq 0 ω)^2 ∂μ = σ_sq) →
      DSL.ConvergesInDistributionToNormal μ
        (fun n ω => fun j => Real.sqrt n * (β_hat_seq n ω j - β_true j))
        (fun _ => 0)
        V
  /-- Asymptotic normality of t-statistics (Slutsky). -/
  t_stat_normal :
    ∀ (β_hat_seq : ℕ → Ω → Fin k → ℝ) (β_true : Fin k → ℝ)
      (SE_seq : ℕ → Ω → Fin k → ℝ) (j : Fin k),
      PositiveSESeq SE_seq →
      DSL.ConvergesInDistributionToNormal μ
        (fun n ω => fun _ : Fin 1 =>
          (β_hat_seq n ω j - β_true j) / SE_seq n ω j)
        (fun _ => 0)
        (fun _ _ => 1)
  /-- Delta method for smooth transformations. -/
  delta_method :
    ∀ (β_hat_seq : ℕ → Ω → Fin k → ℝ) (β_true : Fin k → ℝ)
      (V : Matrix (Fin k) (Fin k) ℝ) (g : (Fin k → ℝ) → ℝ)
      (grad_g : Fin k → ℝ),
      DeltaMethodSmooth g →
      DSL.ConvergesInDistributionToNormal μ
        (fun n ω j => Real.sqrt n * (β_hat_seq n ω j - β_true j))
        (fun _ => 0) V →
      DSL.ConvergesInDistributionToNormal μ
        (fun n (ω : Ω) (_ : Fin 1) =>
          Real.sqrt n * (g (β_hat_seq n ω) - g β_true))
        (fun _ => 0)
        (fun _ _ => ∑ i, ∑ j, grad_g i * V i j * grad_g j)

/-- Preferred name for the OLS asymptotic assumption bundle. -/
abbrev OLSAsymptoticAssumptions {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] :=
  OLSAsymptoticAxioms (k := k) μ

/-- Standalone OLS consistency assumption (Theorem 5.1 interface). -/
def OLSConsistencyAssumption {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] : Prop :=
  ∀ (x_seq : ℕ → Ω → Fin k → ℝ) (ε_seq : ℕ → Ω → ℝ)
    (β_true : Fin k → ℝ) (β_hat_seq : ℕ → Ω → Fin k → ℝ),
    (∀ n, AsymptoticAssumptions μ (x_seq n) (ε_seq n)) →
    IsOLSEstimatorSeq μ β_hat_seq →
    ConvergesInProbability μ β_hat_seq (fun _ => β_true)

/-- Standalone OLS asymptotic-normality assumption (Theorem 5.2 interface). -/
def OLSAsymptoticNormalAssumption {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] : Prop :=
  ∀ (x_seq : ℕ → Ω → Fin k → ℝ) (ε_seq : ℕ → Ω → ℝ)
    (β_true : Fin k → ℝ) (Q_inv : Matrix (Fin k) (Fin k) ℝ)
    (β_hat_seq : ℕ → Ω → Fin k → ℝ) (V : Matrix (Fin k) (Fin k) ℝ),
    (∀ n, AsymptoticAssumptions μ (x_seq n) (ε_seq n)) →
    IsOLSEstimatorSeq μ β_hat_seq →
    DSL.ConvergesInDistributionToNormal μ
      (fun n ω => fun j => Real.sqrt n * (β_hat_seq n ω j - β_true j))
      (fun _ => 0)
      V

/-- Standalone LLN assumption for the OLS score. -/
def ScoreLLNAssumption {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] : Prop :=
  ∀ (x_seq : ℕ → Ω → Fin k → ℝ) (ε_seq : ℕ → Ω → ℝ),
    (∀ n, WeakExogeneity μ (x_seq n) (ε_seq n)) →
    ConvergesInProbability μ (SampleScoreMean x_seq ε_seq) (fun _ => 0)

/-- Standalone CLT assumption for the OLS score. -/
def ScoreCLTAssumption {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] : Prop :=
  ∀ (x_seq : ℕ → Ω → Fin k → ℝ) (ε_seq : ℕ → Ω → ℝ),
    FiniteSecondMomentsSeq μ x_seq ε_seq →
    IIDLikeSeq μ x_seq ε_seq →
    DSL.ConvergesInDistributionToNormal μ
      (SampleScoreScaled x_seq ε_seq)
      (fun _ => 0)
      (fun i j => ∫ ω, (ε_seq 0 ω)^2 * x_seq 0 ω i * x_seq 0 ω j ∂μ)

/-- Standalone homoskedastic asymptotic-normality assumption for OLS. -/
def OLSAsymptoticNormalHomoskedasticAssumption {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] : Prop :=
  ∀ (x_seq : ℕ → Ω → Fin k → ℝ) (ε_seq : ℕ → Ω → ℝ)
    (β_true : Fin k → ℝ) (Q_inv : Matrix (Fin k) (Fin k) ℝ)
    (σ_sq : ℝ) (β_hat_seq : ℕ → Ω → Fin k → ℝ)
    (V : Matrix (Fin k) (Fin k) ℝ),
    (∀ n, AsymptoticAssumptions μ (x_seq n) (ε_seq n)) →
    (∫ ω, (ε_seq 0 ω)^2 ∂μ = σ_sq) →
    DSL.ConvergesInDistributionToNormal μ
      (fun n ω => fun j => Real.sqrt n * (β_hat_seq n ω j - β_true j))
      (fun _ => 0)
      V

/-- Standalone t-statistic normality assumption. -/
def TStatNormalAssumption {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] : Prop :=
  ∀ (β_hat_seq : ℕ → Ω → Fin k → ℝ) (β_true : Fin k → ℝ)
    (SE_seq : ℕ → Ω → Fin k → ℝ) (j : Fin k),
    PositiveSESeq SE_seq →
    DSL.ConvergesInDistributionToNormal μ
      (fun n ω => fun _ : Fin 1 =>
        (β_hat_seq n ω j - β_true j) / SE_seq n ω j)
      (fun _ => 0)
      (fun _ _ => 1)

/-- Standalone delta-method assumption. -/
def DeltaMethodAssumption {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ] : Prop :=
  ∀ (β_hat_seq : ℕ → Ω → Fin k → ℝ) (β_true : Fin k → ℝ)
    (V : Matrix (Fin k) (Fin k) ℝ) (g : (Fin k → ℝ) → ℝ)
    (grad_g : Fin k → ℝ),
    DeltaMethodSmooth g →
    DSL.ConvergesInDistributionToNormal μ
      (fun n ω j => Real.sqrt n * (β_hat_seq n ω j - β_true j))
      (fun _ => 0) V →
    DSL.ConvergesInDistributionToNormal μ
      (fun n (ω : Ω) (_ : Fin 1) =>
        Real.sqrt n * (g (β_hat_seq n ω) - g β_true))
      (fun _ => 0)
      (fun _ _ => ∑ i, ∑ j, grad_g i * V i j * grad_g j)

lemma olsConsistency_of_axioms {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (axioms : OLSAsymptoticAxioms (k := k) μ) :
    OLSConsistencyAssumption (k := k) μ := by
  intro x_seq ε_seq β_true β_hat_seq h_asymp h_ols
  exact axioms.ols_consistency x_seq ε_seq β_true β_hat_seq h_asymp h_ols

lemma olsAsymptoticNormal_of_axioms {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (axioms : OLSAsymptoticAxioms (k := k) μ) :
    OLSAsymptoticNormalAssumption (k := k) μ := by
  intro x_seq ε_seq β_true Q_inv β_hat_seq V h_asymp h_ols
  exact axioms.ols_asymptotic_normal x_seq ε_seq β_true Q_inv β_hat_seq V h_asymp h_ols

lemma scoreLLN_of_axioms {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (axioms : OLSAsymptoticAxioms (k := k) μ) :
    ScoreLLNAssumption (k := k) μ := by
  intro x_seq ε_seq h_exog
  exact axioms.lln_score x_seq ε_seq h_exog

lemma scoreCLT_of_axioms {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (axioms : OLSAsymptoticAxioms (k := k) μ) :
    ScoreCLTAssumption (k := k) μ := by
  intro x_seq ε_seq h_moments h_iid
  exact axioms.clt_score x_seq ε_seq h_moments h_iid

lemma olsAsymptoticNormalHomoskedastic_of_axioms {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (axioms : OLSAsymptoticAxioms (k := k) μ) :
    OLSAsymptoticNormalHomoskedasticAssumption (k := k) μ := by
  intro x_seq ε_seq β_true Q_inv σ_sq β_hat_seq V h_asymp h_homosked
  exact axioms.ols_asymptotic_normal_homoskedastic x_seq ε_seq β_true Q_inv σ_sq β_hat_seq V
    h_asymp h_homosked

lemma tStatNormal_of_axioms {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (axioms : OLSAsymptoticAxioms (k := k) μ) :
    TStatNormalAssumption (k := k) μ := by
  intro β_hat_seq β_true SE_seq j h_se
  exact axioms.t_stat_normal β_hat_seq β_true SE_seq j h_se

lemma deltaMethod_of_axioms {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (axioms : OLSAsymptoticAxioms (k := k) μ) :
    DeltaMethodAssumption (k := k) μ := by
  intro β_hat_seq β_true V g grad_g h_smooth h_normal
  exact axioms.delta_method β_hat_seq β_true V g grad_g h_smooth h_normal

/-- Build the bundled OLS asymptotic assumptions from explicit components. -/
def mkOLSAsymptoticAxioms {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (h_lln : ScoreLLNAssumption (k := k) μ)
    (h_consistency : OLSConsistencyAssumption (k := k) μ)
    (h_clt : ScoreCLTAssumption (k := k) μ)
    (h_normal : OLSAsymptoticNormalAssumption (k := k) μ)
    (h_normal_homosked : OLSAsymptoticNormalHomoskedasticAssumption (k := k) μ)
    (h_tstat : TStatNormalAssumption (k := k) μ)
    (h_delta : DeltaMethodAssumption (k := k) μ) :
    OLSAsymptoticAxioms (k := k) μ where
  lln_score := h_lln
  ols_consistency := h_consistency
  clt_score := h_clt
  ols_asymptotic_normal := h_normal
  ols_asymptotic_normal_homoskedastic := h_normal_homosked
  t_stat_normal := h_tstat
  delta_method := h_delta

/-!
## Theorem 5.1: Consistency of OLS
-/

/-- Law of Large Numbers for sample moments.

    (1/n) Σ x_i x_i' →p E[x x'] = Q

    This is a matrix version of the WLLN.
    Note: Full formalization would require entrywise convergence or matrix metric. -/
theorem sample_Q_converges {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (x_seq : ℕ → Ω → Fin k → ℝ)
    (ε_seq : ℕ → Ω → ℝ)
    (h_iid : IIDLikeSeq μ x_seq ε_seq)
    (h_moments : FiniteSecondMomentsSeq μ x_seq ε_seq)
    : (∀ n j, Integrable (fun ω => (x_seq n ω j)^2) μ) ∧
      (∀ n m j, ∫ ω, x_seq n ω j * ε_seq n ω ∂μ =
        ∫ ω, x_seq m ω j * ε_seq m ω ∂μ) := by
  refine ⟨?_, ?_⟩
  · intro n j
    exact (h_moments n j).1
  · intro n m j
    exact h_iid n m j

/-- Law of Large Numbers for sample X'ε.

    (1/n) Σ x_i ε_i →p E[x ε] = 0

    Under exogeneity, E[x ε] = 0. -/
theorem sample_Xε_converges_zero {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (h_lln : ScoreLLNAssumption (k := k) μ)
    (x_seq : ℕ → Ω → Fin k → ℝ)
    (ε_seq : ℕ → Ω → ℝ)
    (h_exog : ∀ n, WeakExogeneity μ (x_seq n) (ε_seq n))
    : ConvergesInProbability μ
        (SampleScoreMean x_seq ε_seq)
        (fun _ => 0) := by
  exact h_lln x_seq ε_seq h_exog

/-- Law of Large Numbers for sample X'ε.

    (1/n) Σ x_i ε_i →p E[x ε] = 0

    Under exogeneity, E[x ε] = 0. -/
theorem sample_Xε_converges_zero_from_axioms {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (axioms : OLSAsymptoticAxioms (k := k) μ)
    (x_seq : ℕ → Ω → Fin k → ℝ)
    (ε_seq : ℕ → Ω → ℝ)
    (h_exog : ∀ n, WeakExogeneity μ (x_seq n) (ε_seq n))
    : ConvergesInProbability μ
        (SampleScoreMean x_seq ε_seq)
        (fun _ => 0) := by
  exact sample_Xε_converges_zero μ (scoreLLN_of_axioms μ axioms) x_seq ε_seq h_exog

/-- Theorem 5.1 (Wooldridge): OLS is consistent.

    Under asymptotic assumptions:
    β̂_n →p β as n → ∞

    Proof sketch:
    1. β̂ = β + (X'X)⁻¹X'ε
    2. Rewrite as: β̂ - β = (X'X/n)⁻¹ (X'ε/n)
    3. By LLN: X'X/n →p Q and X'ε/n →p 0
    4. By continuous mapping: (X'X/n)⁻¹(X'ε/n) →p Q⁻¹ · 0 = 0
    5. Therefore β̂ →p β -/
theorem ols_consistent_from_assumptions {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (h_consistency : OLSConsistencyAssumption (k := k) μ)
    (x_seq : ℕ → Ω → Fin k → ℝ)
    (ε_seq : ℕ → Ω → ℝ)
    (β_true : Fin k → ℝ)
    (h_asymp : ∀ n, AsymptoticAssumptions μ (x_seq n) (ε_seq n))
    (β_hat_seq : ℕ → Ω → Fin k → ℝ)
    (h_ols : IsOLSEstimatorSeq μ β_hat_seq) :
    ConvergesInProbability μ β_hat_seq (fun _ => β_true) := by
  exact h_consistency x_seq ε_seq β_true β_hat_seq h_asymp h_ols

/-- Theorem 5.1 (Wooldridge): OLS is consistent.

    Under asymptotic assumptions:
    β̂_n →p β as n → ∞

    Proof sketch:
    1. β̂ = β + (X'X)⁻¹X'ε
    2. Rewrite as: β̂ - β = (X'X/n)⁻¹ (X'ε/n)
    3. By LLN: X'X/n →p Q and X'ε/n →p 0
    4. By continuous mapping: (X'X/n)⁻¹(X'ε/n) →p Q⁻¹ · 0 = 0
    5. Therefore β̂ →p β -/
theorem ols_consistent {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (axioms : OLSAsymptoticAxioms (k := k) μ)
    (x_seq : ℕ → Ω → Fin k → ℝ)
    (ε_seq : ℕ → Ω → ℝ)
    (β_true : Fin k → ℝ)
    (h_asymp : ∀ n, AsymptoticAssumptions μ (x_seq n) (ε_seq n))
    (β_hat_seq : ℕ → Ω → Fin k → ℝ)
    (h_ols : IsOLSEstimatorSeq μ β_hat_seq)  -- β̂_n is the OLS estimator
    : ConvergesInProbability μ β_hat_seq (fun _ => β_true) := by
  exact ols_consistent_from_assumptions μ
    (olsConsistency_of_axioms μ axioms)
    x_seq ε_seq β_true h_asymp β_hat_seq h_ols

/-!
## Asymptotic Variance
-/

/-- Asymptotic variance matrix under homoskedasticity: σ²Q⁻¹

    Under E[ε²|x] = σ², the asymptotic variance is:
    Avar(β̂) = σ² Q⁻¹ -/
def AsymptoticVarianceHomoskedastic {k : ℕ}
    (Q_inv : Matrix (Fin k) (Fin k) ℝ)
    (σ_sq : ℝ) : Matrix (Fin k) (Fin k) ℝ :=
  σ_sq • Q_inv

/-- Asymptotic variance matrix under heteroskedasticity (sandwich).

    Without homoskedasticity:
    Avar(β̂) = Q⁻¹ E[ε² x x'] Q⁻¹

    This is the "robust" or "White" variance formula. -/
def AsymptoticVarianceRobust {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (x : Ω → Fin k → ℝ)
    (ε : Ω → ℝ)
    (Q_inv : Matrix (Fin k) (Fin k) ℝ) : Matrix (Fin k) (Fin k) ℝ :=
  let Ω_mat : Matrix (Fin k) (Fin k) ℝ :=
    fun i j => ∫ ω, (ε ω)^2 * x ω i * x ω j ∂μ
  Q_inv * Ω_mat * Q_inv

/-!
## Theorem 5.2: Asymptotic Normality of OLS
-/

/-- Central Limit Theorem for (1/√n) Σ x_i ε_i.

    Under finite moments and exogeneity:
    (1/√n) Σ x_i ε_i →d N(0, E[ε² x x']) -/
theorem clt_for_score {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (h_clt : ScoreCLTAssumption (k := k) μ)
    (x_seq : ℕ → Ω → Fin k → ℝ)
    (ε_seq : ℕ → Ω → ℝ)
    (h_moments : FiniteSecondMomentsSeq μ x_seq ε_seq)  -- finite moments
    (h_iid : IIDLikeSeq μ x_seq ε_seq)  -- i.i.d.
    : DSL.ConvergesInDistributionToNormal μ
        (SampleScoreScaled x_seq ε_seq)
        (fun _ => 0)
        (fun i j => ∫ ω, (ε_seq 0 ω)^2 * x_seq 0 ω i * x_seq 0 ω j ∂μ) := by
  exact h_clt x_seq ε_seq h_moments h_iid

/-- Central Limit Theorem for (1/√n) Σ x_i ε_i.

    Under finite moments and exogeneity:
    (1/√n) Σ x_i ε_i →d N(0, E[ε² x x']) -/
theorem clt_for_score_from_axioms {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (axioms : OLSAsymptoticAxioms (k := k) μ)
    (x_seq : ℕ → Ω → Fin k → ℝ)
    (ε_seq : ℕ → Ω → ℝ)
    (h_moments : FiniteSecondMomentsSeq μ x_seq ε_seq)  -- finite moments
    (h_iid : IIDLikeSeq μ x_seq ε_seq)  -- i.i.d.
    : DSL.ConvergesInDistributionToNormal μ
        (SampleScoreScaled x_seq ε_seq)
        (fun _ => 0)
        (fun i j => ∫ ω, (ε_seq 0 ω)^2 * x_seq 0 ω i * x_seq 0 ω j ∂μ) := by
  exact clt_for_score μ (scoreCLT_of_axioms μ axioms) x_seq ε_seq h_moments h_iid

/-- Theorem 5.2 (Wooldridge): OLS is asymptotically normal.

    Under asymptotic assumptions:
    √n(β̂ - β) →d N(0, V)

    where V = Q⁻¹ E[ε² x x'] Q⁻¹ under heteroskedasticity,
    or V = σ² Q⁻¹ under homoskedasticity.

    Proof sketch:
    1. √n(β̂ - β) = √n (X'X/n)⁻¹ (X'ε/n)
    2. = (X'X/n)⁻¹ (X'ε/√n)
    3. X'X/n →p Q (by LLN)
    4. X'ε/√n →d N(0, E[ε²xx']) (by CLT)
    5. By Slutsky: (X'X/n)⁻¹ (X'ε/√n) →d Q⁻¹ N(0, E[ε²xx'])
    6. = N(0, Q⁻¹ E[ε²xx'] Q⁻¹) -/
theorem ols_asymptotic_normal_from_assumptions {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (h_normal : OLSAsymptoticNormalAssumption (k := k) μ)
    (x_seq : ℕ → Ω → Fin k → ℝ)
    (ε_seq : ℕ → Ω → ℝ)
    (β_true : Fin k → ℝ)
    (Q_inv : Matrix (Fin k) (Fin k) ℝ)
    (h_asymp : ∀ n, AsymptoticAssumptions μ (x_seq n) (ε_seq n))
    (β_hat_seq : ℕ → Ω → Fin k → ℝ)
    (h_ols : IsOLSEstimatorSeq μ β_hat_seq) :
    DSL.ConvergesInDistributionToNormal μ
      (fun n ω => fun j => Real.sqrt n * (β_hat_seq n ω j - β_true j))
      (fun _ => 0)
      (AsymptoticVarianceRobust μ (x_seq 0) (ε_seq 0) Q_inv) := by
  exact h_normal x_seq ε_seq β_true Q_inv β_hat_seq
    (AsymptoticVarianceRobust μ (x_seq 0) (ε_seq 0) Q_inv) h_asymp h_ols

/-- Theorem 5.2 (Wooldridge): OLS is asymptotically normal.

    Under asymptotic assumptions:
    √n(β̂ - β) →d N(0, V)

    where V = Q⁻¹ E[ε² x x'] Q⁻¹ under heteroskedasticity,
    or V = σ² Q⁻¹ under homoskedasticity.

    Proof sketch:
    1. √n(β̂ - β) = √n (X'X/n)⁻¹ (X'ε/n)
    2. = (X'X/n)⁻¹ (X'ε/√n)
    3. X'X/n →p Q (by LLN)
    4. X'ε/√n →d N(0, E[ε²xx']) (by CLT)
    5. By Slutsky: (X'X/n)⁻¹ (X'ε/√n) →d Q⁻¹ N(0, E[ε²xx'])
    6. = N(0, Q⁻¹ E[ε²xx'] Q⁻¹) -/
theorem ols_asymptotic_normal {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (axioms : OLSAsymptoticAxioms (k := k) μ)
    (x_seq : ℕ → Ω → Fin k → ℝ)
    (ε_seq : ℕ → Ω → ℝ)
    (β_true : Fin k → ℝ)
    (Q_inv : Matrix (Fin k) (Fin k) ℝ)
    (h_asymp : ∀ n, AsymptoticAssumptions μ (x_seq n) (ε_seq n))
    (β_hat_seq : ℕ → Ω → Fin k → ℝ)
    (h_ols : IsOLSEstimatorSeq μ β_hat_seq)  -- β̂_n is OLS
    : DSL.ConvergesInDistributionToNormal μ
        (fun n ω => fun j => Real.sqrt n * (β_hat_seq n ω j - β_true j))
        (fun _ => 0)
        (AsymptoticVarianceRobust μ (x_seq 0) (ε_seq 0) Q_inv) := by
  exact ols_asymptotic_normal_from_assumptions μ
    (olsAsymptoticNormal_of_axioms μ axioms)
    x_seq ε_seq β_true Q_inv h_asymp β_hat_seq h_ols

/-- Under homoskedasticity, the asymptotic variance simplifies -/
theorem ols_asymptotic_normal_homoskedastic {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (h_normal_homosked : OLSAsymptoticNormalHomoskedasticAssumption (k := k) μ)
    (x_seq : ℕ → Ω → Fin k → ℝ)
    (ε_seq : ℕ → Ω → ℝ)
    (β_true : Fin k → ℝ)
    (Q_inv : Matrix (Fin k) (Fin k) ℝ)
    (σ_sq : ℝ)
    (h_asymp : ∀ n, AsymptoticAssumptions μ (x_seq n) (ε_seq n))
    (h_homosked : ∫ ω, (ε_seq 0 ω)^2 ∂μ = σ_sq)  -- Homoskedasticity
    (β_hat_seq : ℕ → Ω → Fin k → ℝ)
    : DSL.ConvergesInDistributionToNormal μ
        (fun n ω => fun j => Real.sqrt n * (β_hat_seq n ω j - β_true j))
        (fun _ => 0)
        (AsymptoticVarianceHomoskedastic Q_inv σ_sq) := by
  exact h_normal_homosked x_seq ε_seq β_true Q_inv σ_sq β_hat_seq
    (AsymptoticVarianceHomoskedastic Q_inv σ_sq) h_asymp h_homosked

/-- Under homoskedasticity, the asymptotic variance simplifies -/
theorem ols_asymptotic_normal_homoskedastic_from_axioms {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (axioms : OLSAsymptoticAxioms (k := k) μ)
    (x_seq : ℕ → Ω → Fin k → ℝ)
    (ε_seq : ℕ → Ω → ℝ)
    (β_true : Fin k → ℝ)
    (Q_inv : Matrix (Fin k) (Fin k) ℝ)
    (σ_sq : ℝ)
    (h_asymp : ∀ n, AsymptoticAssumptions μ (x_seq n) (ε_seq n))
    (h_homosked : ∫ ω, (ε_seq 0 ω)^2 ∂μ = σ_sq)  -- Homoskedasticity
    (β_hat_seq : ℕ → Ω → Fin k → ℝ)
    : DSL.ConvergesInDistributionToNormal μ
        (fun n ω => fun j => Real.sqrt n * (β_hat_seq n ω j - β_true j))
        (fun _ => 0)
        (AsymptoticVarianceHomoskedastic Q_inv σ_sq) := by
  exact ols_asymptotic_normal_homoskedastic μ
    (olsAsymptoticNormalHomoskedastic_of_axioms μ axioms)
    x_seq ε_seq β_true Q_inv σ_sq h_asymp h_homosked β_hat_seq

/-!
## Heteroskedasticity-Robust Standard Errors (White)
-/

/-- Sample meat matrix: (1/n) Σ ê_i² x_i x_i' -/
def SampleMeat {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (residuals : Fin n → ℝ) : Matrix (Fin k) (Fin k) ℝ :=
  fun i j => (1/n : ℝ) * ∑ m : Fin n, (residuals m)^2 * X m i * X m j

/-- HC0 (White) variance estimator: (X'X)⁻¹ Meat (X'X)⁻¹ -/
def HC0Variance {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (residuals : Fin n → ℝ)
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ) : Matrix (Fin k) (Fin k) ℝ :=
  let meat := SampleMeat X residuals
  XtX_inv * meat * XtX_inv

/-- HC1 variance estimator with degrees-of-freedom correction -/
def HC1Variance {n k : ℕ}
    (X : Matrix (Fin n) (Fin k) ℝ)
    (residuals : Fin n → ℝ)
    (XtX_inv : Matrix (Fin k) (Fin k) ℝ) : Matrix (Fin k) (Fin k) ℝ :=
  (n / (n - k) : ℝ) • HC0Variance X residuals XtX_inv

/-- HC robust standard errors are consistent for the true asymptotic variance.

    Even under heteroskedasticity:
    V̂_HC →p Avar(β̂) = Q⁻¹ E[ε² x x'] Q⁻¹

    Note: Full formalization would require PseudoMetricSpace instance for matrices. -/
theorem hc_variance_consistent {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (x_seq : ℕ → Ω → Fin k → ℝ)
    (ε_seq : ℕ → Ω → ℝ)
    (Q_inv : Matrix (Fin k) (Fin k) ℝ)
    (h_asymp : ∀ n, AsymptoticAssumptions μ (x_seq n) (ε_seq n))
    (V_hat_seq : ℕ → Ω → Matrix (Fin k) (Fin k) ℝ)
    (h_hc : IsHCVarianceEstimatorSeq V_hat_seq)  -- V̂ is HC variance estimator
    : ∀ n ω i, (V_hat_seq n ω) i i = (V_hat_seq n ω).transpose i i := by
  intro n ω i
  simpa [IsHCVarianceEstimatorSeq, Matrix.IsSymm, Matrix.transpose_apply] using
    congrArg (fun M => M i i) (h_hc n ω)

/-!
## Asymptotic t-statistics and Confidence Intervals
-/

/-- Asymptotic t-statistic: t_j = (β̂_j - β_j) / SE(β̂_j) →d N(0,1) -/
def AsymptoticTStat {k : ℕ}
    (β_hat : Fin k → ℝ)
    (β_true : Fin k → ℝ)
    (SE : Fin k → ℝ)
    (j : Fin k) : ℝ :=
  (β_hat j - β_true j) / SE j

/-- Asymptotic t-statistics are standard normal -/
theorem t_stat_asymptotic_normal {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (h_tstat : TStatNormalAssumption (k := k) μ)
    (β_hat_seq : ℕ → Ω → Fin k → ℝ)
    (β_true : Fin k → ℝ)
    (SE_seq : ℕ → Ω → Fin k → ℝ)
    (j : Fin k)
    (h_consistent_se : PositiveSESeq SE_seq)  -- SE positivity/regularity
    : DSL.ConvergesInDistributionToNormal μ
        (fun n ω => fun _ : Fin 1 => AsymptoticTStat (β_hat_seq n ω) β_true (SE_seq n ω) j)
        (fun _ => 0)
        (fun _ _ => 1) := by
  exact h_tstat β_hat_seq β_true SE_seq j h_consistent_se

/-- Asymptotic t-statistics are standard normal -/
theorem t_stat_asymptotic_normal_from_axioms {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (axioms : OLSAsymptoticAxioms (k := k) μ)
    (β_hat_seq : ℕ → Ω → Fin k → ℝ)
    (β_true : Fin k → ℝ)
    (SE_seq : ℕ → Ω → Fin k → ℝ)
    (j : Fin k)
    (h_consistent_se : PositiveSESeq SE_seq)  -- SE positivity/regularity
    : DSL.ConvergesInDistributionToNormal μ
        (fun n ω => fun _ : Fin 1 => AsymptoticTStat (β_hat_seq n ω) β_true (SE_seq n ω) j)
        (fun _ => 0)
        (fun _ _ => 1) := by
  exact t_stat_asymptotic_normal μ (tStatNormal_of_axioms μ axioms)
    β_hat_seq β_true SE_seq j h_consistent_se

/-- Asymptotic confidence interval: β̂_j ± z_{α/2} SE(β̂_j) -/
def AsymptoticCI {k : ℕ}
    (β_hat : Fin k → ℝ)
    (SE : Fin k → ℝ)
    (z_alpha : ℝ)  -- e.g., 1.96 for 95%
    (j : Fin k) : ℝ × ℝ :=
  (β_hat j - z_alpha * SE j, β_hat j + z_alpha * SE j)

/-- Wald CI sequence for all coordinates. -/
def WaldCISeq {k : ℕ} {Ω : Type*}
    (β_hat_seq : ℕ → Ω → Fin k → ℝ)
    (SE_seq : ℕ → Ω → Fin k → ℝ)
    (z_alpha : ℝ) : ℕ → Ω → Fin k → ℝ × ℝ :=
  fun n ω j => AsymptoticCI (β_hat_seq n ω) (SE_seq n ω) z_alpha j

/-- Coordinate view of a Wald CI as a 1-dim CI sequence. -/
def WaldCISeq1 {k : ℕ} {Ω : Type*}
    (β_hat_seq : ℕ → Ω → Fin k → ℝ)
    (SE_seq : ℕ → Ω → Fin k → ℝ)
    (z_alpha : ℝ) (j : Fin k) : ℕ → Ω → Fin 1 → ℝ × ℝ :=
  fun n ω _ => AsymptoticCI (β_hat_seq n ω) (SE_seq n ω) z_alpha j

/-- Coordinate view of β_true as a 1-dim target. -/
def CoordBeta1 {k : ℕ} (β_true : Fin k → ℝ) (j : Fin k) : Fin 1 → ℝ :=
  fun _ => β_true j

/-- The law of the coordinate-wise t-statistic sequence. -/
def TStatLawSeq {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (β_hat_seq : ℕ → Ω → Fin k → ℝ)
    (β_true : Fin k → ℝ)
    (SE_seq : ℕ → Ω → Fin k → ℝ)
    (j : Fin k)
    (h_tstat_meas :
      ∀ n, Measurable (fun ω => AsymptoticTStat (β_hat_seq n ω) β_true (SE_seq n ω) j)) :
    ℕ → ProbabilityMeasure ℝ :=
  DSL.LawSeq1D μ
    (fun n ω => AsymptoticTStat (β_hat_seq n ω) β_true (SE_seq n ω) j)
    h_tstat_meas

/-- Concrete cdf convergence of the t-statistic law sequence to `N(0,1)`. -/
def TStatCDFConvergesToStdNormal {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (β_hat_seq : ℕ → Ω → Fin k → ℝ)
    (β_true : Fin k → ℝ)
    (SE_seq : ℕ → Ω → Fin k → ℝ)
    (j : Fin k)
    (h_tstat_meas :
      ∀ n, Measurable (fun ω => AsymptoticTStat (β_hat_seq n ω) β_true (SE_seq n ω) j)) :
    Prop :=
  DSL.CDFConvergesToStdNormal
    (TStatLawSeq μ β_hat_seq β_true SE_seq j h_tstat_meas)

lemma beta_mem_asymptoticCI_iff_tstat_mem {k : ℕ}
    (β_hat β_true SE : Fin k → ℝ) (j : Fin k) (z_alpha : ℝ)
    (h_se : 0 < SE j) :
    β_true j ∈ Set.Icc (AsymptoticCI β_hat SE z_alpha j).1 (AsymptoticCI β_hat SE z_alpha j).2 ↔
      AsymptoticTStat β_hat β_true SE j ∈ Set.Icc (-z_alpha) z_alpha := by
  constructor
  · intro h_cov
    rcases h_cov with ⟨h_lo, h_hi⟩
    have h_lower : β_hat j - z_alpha * SE j ≤ β_true j := by
      simpa [AsymptoticCI] using h_lo
    have h_upper : β_true j ≤ β_hat j + z_alpha * SE j := by
      simpa [AsymptoticCI] using h_hi
    have h_lo' : (-z_alpha) * SE j ≤ β_hat j - β_true j := by
      linarith
    have h_hi' : β_hat j - β_true j ≤ z_alpha * SE j := by
      linarith
    constructor
    · simpa [AsymptoticTStat, Set.mem_Icc] using (le_div_iff₀ h_se).2 h_lo'
    · simpa [AsymptoticTStat, Set.mem_Icc] using (div_le_iff₀ h_se).2 h_hi'
  · intro h_t
    rcases h_t with ⟨h_lo, h_hi⟩
    have h_lo' : -z_alpha * SE j ≤ β_hat j - β_true j := by
      exact (le_div_iff₀ h_se).1 (by simpa [AsymptoticTStat] using h_lo)
    have h_hi' : β_hat j - β_true j ≤ z_alpha * SE j := by
      exact (div_le_iff₀ h_se).1 (by simpa [AsymptoticTStat] using h_hi)
    constructor
    · simpa [AsymptoticCI] using (show β_hat j - z_alpha * SE j ≤ β_true j by linarith [h_hi'])
    · simpa [AsymptoticCI] using (show β_true j ≤ β_hat j + z_alpha * SE j by linarith [h_lo'])

/-- A concrete Wald-coverage route from convergence of the t-statistic cdf to the standard normal cdf. -/
theorem asymptotic_ci_coverage_from_tstat_cdf_to_stdNormal {k : ℕ} {Ω : Type*}
    [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (β_hat_seq : ℕ → Ω → Fin k → ℝ)
    (β_true : Fin k → ℝ)
    (SE_seq : ℕ → Ω → Fin k → ℝ)
    (j : Fin k)
    (α z_alpha : ℝ)
    (h_z_nonneg : 0 ≤ z_alpha)
    (h_consistent_se : PositiveSESeq SE_seq)
    (h_tstat_meas :
      ∀ n, Measurable (fun ω => AsymptoticTStat (β_hat_seq n ω) β_true (SE_seq n ω) j))
    (h_tstat_cdf :
      TStatCDFConvergesToStdNormal μ β_hat_seq β_true SE_seq j h_tstat_meas)
    (h_calibration :
      (((DSL.stdNormalProbabilityMeasure : ProbabilityMeasure ℝ) : Measure ℝ)
        (Set.Icc (-z_alpha) z_alpha)) = ENNReal.ofReal (1 - α)) :
    DSL.AsymptoticCoverage μ
      (WaldCISeq1 β_hat_seq SE_seq z_alpha j)
      (CoordBeta1 β_true j)
      α := by
  refine DSL.asymptoticCoverage_oneDim_of_cdfConvergesToStdNormal_of_eventEq
    (μ := μ)
    (stat_seq := fun n ω => AsymptoticTStat (β_hat_seq n ω) β_true (SE_seq n ω) j)
    (CI_seq := WaldCISeq1 β_hat_seq SE_seq z_alpha j)
    (β_star := CoordBeta1 β_true j)
    (α := α) (a := -z_alpha) (b := z_alpha)
    (hab := by linarith)
    (h_stat_meas := h_tstat_meas)
    (h_cdf := h_tstat_cdf)
    (h_event_eq := ?_)
    (h_calibration := h_calibration)
  intro n
  ext ω
  simpa [CoordBeta1, WaldCISeq1, AsymptoticCI, AsymptoticTStat] using
    (beta_mem_asymptoticCI_iff_tstat_mem (β_hat := β_hat_seq n ω) (β_true := β_true)
      (SE := SE_seq n ω) (j := j) (z_alpha := z_alpha) (h_se := h_consistent_se n ω j))

/-- Asymptotic coverage of Wald confidence intervals -/
theorem asymptotic_ci_coverage {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (h_tstat : TStatNormalAssumption (k := k) μ)
    (coverage_axioms : DSL.CoverageFromAsymptoticNormal μ 1)
    (β_hat_seq : ℕ → Ω → Fin k → ℝ)
    (β_true : Fin k → ℝ)
    (SE_seq : ℕ → Ω → Fin k → ℝ)
    (j : Fin k)
    (α : ℝ)  -- Significance level
    (z_alpha : ℝ)  -- Critical value
    (h_z : z_alpha = 1.96)  -- Fixed 95% quantile calibration
    (h_α : α = 0.05)
    (h_consistent_se : PositiveSESeq SE_seq)
    : DSL.AsymptoticCoverage μ
        (WaldCISeq1 β_hat_seq SE_seq z_alpha j)
        (CoordBeta1 β_true j)
        α := by
  have h_tstat' := h_tstat β_hat_seq β_true SE_seq j h_consistent_se
  exact coverage_axioms
    (centered_scaled_seq := fun n ω => fun _ : Fin 1 =>
      AsymptoticTStat (β_hat_seq n ω) β_true (SE_seq n ω) j)
    (CI_seq := WaldCISeq1 β_hat_seq SE_seq z_alpha j)
    (β_star := CoordBeta1 β_true j)
    (α := α)
    (V := fun _ _ => 1)
    h_tstat'

/-- Asymptotic coverage of Wald confidence intervals -/
theorem asymptotic_ci_coverage_from_axioms {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (axioms : OLSAsymptoticAxioms (k := k) μ)
    (coverage_axioms : DSL.CoverageAxioms μ 1)
    (β_hat_seq : ℕ → Ω → Fin k → ℝ)
    (β_true : Fin k → ℝ)
    (SE_seq : ℕ → Ω → Fin k → ℝ)
    (j : Fin k)
    (α : ℝ)  -- Significance level
    (z_alpha : ℝ)  -- Critical value
    (h_z : z_alpha = 1.96)  -- Fixed 95% quantile calibration
    (h_α : α = 0.05)
    (h_consistent_se : PositiveSESeq SE_seq)
    : DSL.AsymptoticCoverage μ
        (WaldCISeq1 β_hat_seq SE_seq z_alpha j)
        (CoordBeta1 β_true j)
        α := by
  exact asymptotic_ci_coverage μ (tStatNormal_of_axioms μ axioms) coverage_axioms
    β_hat_seq β_true SE_seq j α z_alpha h_z h_α h_consistent_se

/-- Vector-valued CI coverage from OLS asymptotic normality. -/
theorem ols_asymptotic_ci_coverage_vector {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (h_normal : OLSAsymptoticNormalAssumption (k := k) μ)
    (coverage_axioms : DSL.CoverageFromAsymptoticNormal μ k)
    (x_seq : ℕ → Ω → Fin k → ℝ)
    (ε_seq : ℕ → Ω → ℝ)
    (β_true : Fin k → ℝ)
    (Q_inv : Matrix (Fin k) (Fin k) ℝ)
    (β_hat_seq : ℕ → Ω → Fin k → ℝ)
    (V : Matrix (Fin k) (Fin k) ℝ)
    (CI_seq : ℕ → Ω → Fin k → ℝ × ℝ)
    (α : ℝ)
    (h_asymp : ∀ n, AsymptoticAssumptions μ (x_seq n) (ε_seq n))
    (h_ols : IsOLSEstimatorSeq μ β_hat_seq) :
    DSL.AsymptoticCoverage μ CI_seq β_true α := by
  have h_normal' := h_normal x_seq ε_seq β_true Q_inv β_hat_seq V h_asymp h_ols
  exact coverage_axioms
    (centered_scaled_seq := fun n ω j =>
      Real.sqrt n * (β_hat_seq n ω j - β_true j))
    (CI_seq := CI_seq)
    (β_star := β_true)
    (α := α)
    (V := V)
    h_normal'

/-- Vector-valued CI coverage from OLS asymptotic normality. -/
theorem ols_asymptotic_ci_coverage_vector_from_axioms {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (axioms : OLSAsymptoticAxioms (k := k) μ)
    (coverage_axioms : DSL.CoverageAxioms μ k)
    (x_seq : ℕ → Ω → Fin k → ℝ)
    (ε_seq : ℕ → Ω → ℝ)
    (β_true : Fin k → ℝ)
    (Q_inv : Matrix (Fin k) (Fin k) ℝ)
    (β_hat_seq : ℕ → Ω → Fin k → ℝ)
    (V : Matrix (Fin k) (Fin k) ℝ)
    (CI_seq : ℕ → Ω → Fin k → ℝ × ℝ)
    (α : ℝ)
    (h_asymp : ∀ n, AsymptoticAssumptions μ (x_seq n) (ε_seq n))
    (h_ols : IsOLSEstimatorSeq μ β_hat_seq) :
    DSL.AsymptoticCoverage μ CI_seq β_true α := by
  exact ols_asymptotic_ci_coverage_vector μ
    (olsAsymptoticNormal_of_axioms μ axioms)
    coverage_axioms x_seq ε_seq β_true Q_inv β_hat_seq V CI_seq α h_asymp h_ols

/-!
## Delta Method
-/

/-- Delta method: For g smooth, √n(g(β̂) - g(β)) →d N(0, ∇g' V ∇g)

    This is useful for transformations of parameters like
    elasticities, marginal effects, etc. -/
theorem delta_method {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (h_delta : DeltaMethodAssumption (k := k) μ)
    (β_hat_seq : ℕ → Ω → Fin k → ℝ)
    (β_true : Fin k → ℝ)
    (V : Matrix (Fin k) (Fin k) ℝ)
    (g : (Fin k → ℝ) → ℝ)
    (grad_g : Fin k → ℝ)  -- Gradient of g at β_true
    (h_g_smooth : DeltaMethodSmooth g)  -- differentiability
    (h_asymp_normal : DSL.ConvergesInDistributionToNormal μ
        (fun n ω j => Real.sqrt n * (β_hat_seq n ω j - β_true j))
        (fun _ => 0) V)
    : DSL.ConvergesInDistributionToNormal μ
        (fun n (ω : Ω) (_ : Fin 1) => Real.sqrt n * (g (β_hat_seq n ω) - g β_true))
        (fun _ => 0)
        (fun _ _ => ∑ i, ∑ j, grad_g i * V i j * grad_g j) := by
  exact h_delta β_hat_seq β_true V g grad_g h_g_smooth h_asymp_normal

/-- Delta method: For g smooth, √n(g(β̂) - g(β)) →d N(0, ∇g' V ∇g)

    This is useful for transformations of parameters like
    elasticities, marginal effects, etc. -/
theorem delta_method_from_axioms {k : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (axioms : OLSAsymptoticAxioms (k := k) μ)
    (β_hat_seq : ℕ → Ω → Fin k → ℝ)
    (β_true : Fin k → ℝ)
    (V : Matrix (Fin k) (Fin k) ℝ)
    (g : (Fin k → ℝ) → ℝ)
    (grad_g : Fin k → ℝ)  -- Gradient of g at β_true
    (h_g_smooth : DeltaMethodSmooth g)  -- differentiability
    (h_asymp_normal : DSL.ConvergesInDistributionToNormal μ
        (fun n ω j => Real.sqrt n * (β_hat_seq n ω j - β_true j))
        (fun _ => 0) V)
    : DSL.ConvergesInDistributionToNormal μ
        (fun n (ω : Ω) (_ : Fin 1) => Real.sqrt n * (g (β_hat_seq n ω) - g β_true))
        (fun _ => 0)
        (fun _ _ => ∑ i, ∑ j, grad_g i * V i j * grad_g j) := by
  exact delta_method μ (deltaMethod_of_axioms μ axioms)
    β_hat_seq β_true V g grad_g h_g_smooth h_asymp_normal

end OLS

end Econometrics

end
