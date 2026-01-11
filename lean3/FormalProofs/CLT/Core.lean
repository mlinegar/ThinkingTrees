import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.MeasureTheory.Function.ConvergenceInMeasure
import Mathlib.MeasureTheory.Function.ConvergenceInDistribution
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Probability.IdentDistrib
import Mathlib.Probability.Independence.Basic
import Mathlib.Topology.Metrizable.Basic

/-!
# FormalProofs/Probability/Core.lean

Core probability utilities for the DSL formalization:
- expectation (scalar and finite-dimensional)
- i.i.d. predicate
- sample means
- convergence in probability (as `TendstoInMeasure`)
- convergence in distribution (as `TendstoInDistribution`)
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

namespace ProbabilityTheory

open MeasureTheory
open Filter

/-!
## Expectation
-/

/-- Scalar expectation under a probability measure. -/
def expectation {Omega : Type*} [MeasurableSpace Omega]
    (mu : Measure Omega) [IsProbabilityMeasure mu]
    (f : Omega → ℝ) : ℝ :=
  ∫ omega, f omega ∂mu

/-- Coordinatewise expectation for finite-dimensional vectors. -/
def expectationVec {Omega : Type*} [MeasurableSpace Omega]
    (mu : Measure Omega) [IsProbabilityMeasure mu] {d : ℕ}
    (f : Omega → Fin d → ℝ) : Fin d → ℝ :=
  fun j => ∫ omega, f omega j ∂mu

lemma expectationVec_add {Omega : Type*} [MeasurableSpace Omega]
    (mu : Measure Omega) [IsProbabilityMeasure mu] {d : ℕ}
    (f g : Omega → Fin d → ℝ)
    (hf : ∀ j, Integrable (fun omega => f omega j) mu)
    (hg : ∀ j, Integrable (fun omega => g omega j) mu) :
    expectationVec mu (fun omega j => f omega j + g omega j) =
      fun j => expectationVec mu f j + expectationVec mu g j := by
  ext j
  simp [expectationVec, integral_add, hf j, hg j]

lemma expectationVec_smul {Omega : Type*} [MeasurableSpace Omega]
    (mu : Measure Omega) [IsProbabilityMeasure mu] {d : ℕ}
    (c : ℝ) (f : Omega → Fin d → ℝ)
    (hf : ∀ j, Integrable (fun omega => f omega j) mu) :
    expectationVec mu (fun omega j => c * f omega j) =
      fun j => c * expectationVec mu f j := by
  ext j
  simp [expectationVec, integral_const_mul, hf j]

/-!
## IID
-/

/-- IID predicate for a sequence of random variables. -/
def IID {Omega : Type*} [MeasurableSpace Omega]
    {Alpha : Type*} [MeasurableSpace Alpha] (mu : Measure Omega) (X : ℕ → Omega → Alpha) : Prop :=
  iIndepFun X mu ∧ ∀ i, IdentDistrib (X i) (X 0) mu mu

/-!
## Sample Means
-/

/-- Sample mean of a real-valued sequence. -/
def sampleMean {Omega : Type*} [MeasurableSpace Omega]
    (X : ℕ → Omega → ℝ) (n : ℕ) : Omega → ℝ :=
  fun omega => (n : ℝ)⁻¹ * Finset.sum (Finset.range n) (fun i => X i omega)

/-- Sample mean of a finite-dimensional sequence (coordinatewise). -/
def sampleMeanVec {Omega : Type*} [MeasurableSpace Omega] {d : ℕ}
    (X : ℕ → Omega → Fin d → ℝ) (n : ℕ) : Omega → Fin d → ℝ :=
  fun omega j => (n : ℝ)⁻¹ * Finset.sum (Finset.range n) (fun i => X i omega j)

/-!
## Convergence
-/

/-- Convergence in probability, expressed as `TendstoInMeasure`. -/
def ConvergesInProbability {Omega : Type*} [MeasurableSpace Omega]
    (mu : Measure Omega) [IsProbabilityMeasure mu]
    {E : Type*} [PseudoMetricSpace E]
    (seq : ℕ → Omega → E) (limit : Omega → E) : Prop :=
  MeasureTheory.TendstoInMeasure mu seq Filter.atTop limit

/-- Convergence in distribution, as `TendstoInDistribution`. -/
def ConvergesInDistribution {Omega : Type*} [MeasurableSpace Omega]
    (mu : Measure Omega) [IsProbabilityMeasure mu]
    {E : Type*} [MeasurableSpace E] [TopologicalSpace E] [OpensMeasurableSpace E]
    (seq : ℕ → Omega → E) (limit : Omega → E) : Prop :=
  MeasureTheory.TendstoInDistribution seq Filter.atTop limit mu

end ProbabilityTheory

end
