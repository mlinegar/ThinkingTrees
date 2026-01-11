import FormalProofs.DSL.MomentFunctions
import FormalProofs.Probability.Core

/-!
# FormalProofs/DSL/MEstimationCore.lean

Probability and M-estimation scaffolding for DSL:
- population moments as expectations under a data-generating process
- empirical moments from an i.i.d. sequence
- identification and consistency assumption bundles
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical
open scoped Topology

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace DSL

open MeasureTheory
open Filter
open ProbabilityTheory

/-!
## Expectation Operator Induced by a Data Sequence
-/

/-- Expectation operator induced by a data-generating process `X`. -/
def expectationFromSeq {Omega Data : Type*} [MeasurableSpace Omega]
    (mu : Measure Omega) [IsProbabilityMeasure mu]
    (X : ℕ → Omega → Data) {d : ℕ} :
    (Data → Fin d → ℝ) → Fin d → ℝ :=
  fun f => expectationVec mu (fun omega => f (X 0 omega))

/-!
## Population and Empirical Moments
-/

/-- Population moment for a fixed parameter value. -/
def populationMoment {Omega Data : Type*} [MeasurableSpace Omega]
    (mu : Measure Omega) [IsProbabilityMeasure mu]
    {d : ℕ} (X : ℕ → Omega → Data)
    (m : MomentFunction Data d) (beta : Fin d → ℝ) : Fin d → ℝ :=
  expectationVec mu (fun omega => m (X 0 omega) beta)

/-- Empirical moment as a sample mean over the first `n` draws. -/
def empiricalMoment {Omega Data : Type*} [MeasurableSpace Omega]
    {d : ℕ} (X : ℕ → Omega → Data)
    (m : MomentFunction Data d) (beta : Fin d → ℝ) (n : ℕ) :
    Omega → Fin d → ℝ :=
  sampleMeanVec (fun i omega => m (X i omega) beta) n

/-- A random estimator sequence solves the empirical moment condition. -/
def IsMEstimatorSeqRV {Omega Data : Type*} [MeasurableSpace Omega]
    {d : ℕ} (X : ℕ → Omega → Data)
    (m : MomentFunction Data d) (beta_hat : ℕ → Omega → Fin d → ℝ) : Prop :=
  ∀ n omega, empiricalMoment X m (beta_hat n omega) n omega = 0

/-!
## Identification
-/

/-- Identification: the population moment has a unique zero. -/
def Identified {d : ℕ}
    (pop : (Fin d → ℝ) → Fin d → ℝ) (beta_star : Fin d → ℝ) : Prop :=
  pop beta_star = 0 ∧ ∀ beta, pop beta = 0 → beta = beta_star

/-- Unbiasedness of the population moment at `beta_star`. -/
def MomentUnbiasedMeasure {Omega Data : Type*} [MeasurableSpace Omega]
    (mu : Measure Omega) [IsProbabilityMeasure mu] {d : ℕ}
    (X : ℕ → Omega → Data)
    (m : MomentFunction Data d) (beta_star : Fin d → ℝ) : Prop :=
  populationMoment mu X m beta_star = 0

/-!
## M-Estimation Assumptions (Consistency Core)
-/

structure MEstimationAssumptions {Omega Data : Type*} [MeasurableSpace Omega]
    (mu : Measure Omega) [IsProbabilityMeasure mu]
    (d : ℕ) (X : ℕ → Omega → Data)
    (m : MomentFunction Data d) (beta_star : Fin d → ℝ) : Prop where
  iid : IID mu X
  integrable : ∀ j, Integrable (fun omega => m (X 0 omega) beta_star j) mu
  unbiased : MomentUnbiasedMeasure mu X m beta_star
  identified : Identified (populationMoment mu X m) beta_star
  lln :
    ConvergesInProbability mu
      (fun n omega => empiricalMoment X m beta_star n omega)
      (fun _ => populationMoment mu X m beta_star)

end DSL

end
