import Mathlib
import FormalProofs.CLT.Core

/-!
# FormalProofs/ML/Core.lean

Core supervised learning definitions:
- hypotheses and losses
- population risk under a data distribution
- labeled datasets
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

namespace ML

/-!
## Basic Objects
-/

/-- Hypothesis (predictor). -/
def Hypothesis (X Y : Type*) := X → Y

/-- Pointwise loss: compares prediction and label. -/
def Loss (Y : Type*) := Y → Y → ℝ

/-- Labeled example. -/
abbrev LabeledExample (X Y : Type*) := X × Y

/-- Finite dataset of size n. -/
abbrev Dataset (X Y : Type*) (n : ℕ) := Fin n → LabeledExample X Y

/-!
## Risk (Population Loss)
-/

/-- Loss incurred by hypothesis h on a labeled example. -/
def lossOn {X Y : Type*} (loss : Loss Y) (h : Hypothesis X Y) (z : LabeledExample X Y) : ℝ :=
  loss (h z.1) z.2

/-- Population risk under a probability measure on examples. -/
def expectedRisk {X Y : Type*} [MeasurableSpace X] [MeasurableSpace Y]
    (mu : Measure (X × Y)) [IsProbabilityMeasure mu]
    (loss : Loss Y) (h : Hypothesis X Y) : ℝ :=
  ProbabilityTheory.expectation mu (lossOn loss h)

end ML
