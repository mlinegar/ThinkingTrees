import Mathlib
import FormalProbability.CLT.Core

/-!
# FormalProofs/Econometrics/Core.lean

Core econometrics definitions for IPW prerequisites.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Classical
open MeasureTheory

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace Econometrics

/-!
## Basic Types
-/

/-- Binary treatment indicator. -/
abbrev Treatment := Bool

/-- Convert treatment indicator to real-valued 0/1. -/
def Treatment.toReal (d : Treatment) : ℝ :=
  if d then 1 else 0

lemma Treatment.toReal_true : Treatment.toReal true = (1 : ℝ) := by
  simp [Treatment.toReal]

lemma Treatment.toReal_false : Treatment.toReal false = (0 : ℝ) := by
  simp [Treatment.toReal]

lemma Treatment.toReal_sq (d : Treatment) : (Treatment.toReal d) ^ 2 = Treatment.toReal d := by
  cases d <;> simp [Treatment.toReal]

lemma Treatment.toReal_mul_complement (d : Treatment) :
    Treatment.toReal d * (1 - Treatment.toReal d) = 0 := by
  cases d <;> simp [Treatment.toReal]

lemma Treatment.toReal_add_complement (d : Treatment) :
    Treatment.toReal d + (1 - Treatment.toReal d) = 1 := by
  cases d <;> simp [Treatment.toReal]

/-- Potential outcomes for a unit. -/
structure PotentialOutcomes (Ω : Type*) where
  y0 : Ω → ℝ
  y1 : Ω → ℝ

/-- Treatment indicator as a real-valued random variable. -/
def treatmentIndicator {Ω : Type*} (D : Ω → Treatment) : Ω → ℝ :=
  fun ω => Treatment.toReal (D ω)

/-- Individual treatment effect (ITE): tau = Y1 - Y0. -/
def treatmentEffect {Ω : Type*} (po : PotentialOutcomes Ω) : Ω → ℝ :=
  fun ω => po.y1 ω - po.y0 ω

/-- Observed outcome under treatment assignment D. -/
def observedOutcome {Ω : Type*} (D : Ω → Treatment) (po : PotentialOutcomes Ω) : Ω → ℝ :=
  fun ω =>
    let d := Treatment.toReal (D ω)
    d * po.y1 ω + (1 - d) * po.y0 ω

lemma observedOutcome_eq_by_cases {Ω : Type*} (D : Ω → Treatment) (po : PotentialOutcomes Ω) :
    observedOutcome D po = fun ω => if D ω then po.y1 ω else po.y0 ω := by
  funext ω
  cases h : D ω <;> simp [observedOutcome, Treatment.toReal, h]

/-!
## Average Treatment Effect (ATE)
-/

/-- Average treatment effect: E[Y1 - Y0]. -/
def ATE {Ω : Type*} [MeasurableSpace Ω]
    (mu : Measure Ω) [IsProbabilityMeasure mu] (po : PotentialOutcomes Ω) : ℝ :=
  ProbabilityTheory.expectation mu (treatmentEffect po)

end Econometrics
