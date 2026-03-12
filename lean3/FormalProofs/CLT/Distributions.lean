import Mathlib.Probability.ProbabilityMassFunction.Binomial
import Mathlib.Probability.Distributions.Geometric
import Mathlib.Probability.Distributions.Poisson
import Mathlib.Probability.Distributions.Exponential
import Mathlib.Probability.Distributions.Uniform
import Mathlib.Probability.Distributions.Gaussian.Real

import FormalProofs.CLT.Normal

/-!
# FormalProofs/CLT/Distributions.lean

Convenience wrappers for standard undergrad distributions.
-/

set_option linter.mathlibStandardSet false

open scoped Classical
open scoped Topology
open scoped NNReal ENNReal

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace ProbabilityTheory

open MeasureTheory

/-!
## Binomial law
-/

/-- Binomial PMF as a probability mass function on `Fin (n + 1)`. -/
abbrev binomialPMF (p : ℝ≥0) (hp : p ≤ 1) (n : ℕ) : PMF (Fin (n + 1)) :=
  PMF.binomial p hp n

/-- Binomial measure derived from the PMF. -/
def binomialMeasure (p : ℝ≥0) (hp : p ≤ 1) (n : ℕ) : Measure (Fin (n + 1)) :=
  (binomialPMF p hp n).toMeasure

instance binomialMeasure_isProbabilityMeasure (p : ℝ≥0) (hp : p ≤ 1) (n : ℕ) :
    IsProbabilityMeasure (binomialMeasure p hp n) :=
  PMF.toMeasure.isProbabilityMeasure (binomialPMF p hp n)

/-!
## Geometric and Poisson laws
-/

abbrev geometricLaw (p : ℝ) (hp_pos : 0 < p) (hp_le_one : p ≤ 1) : Measure ℕ :=
  geometricMeasure (p := p) hp_pos hp_le_one

lemma geometricLaw_isProbabilityMeasure (p : ℝ) (hp_pos : 0 < p) (hp_le_one : p ≤ 1) :
    IsProbabilityMeasure (geometricLaw p hp_pos hp_le_one) := by
  simpa [geometricLaw] using isProbabilityMeasure_geometricMeasure (p := p) hp_pos hp_le_one

abbrev poissonLaw (r : ℝ≥0) : Measure ℕ := poissonMeasure r

lemma poissonLaw_isProbabilityMeasure (r : ℝ≥0) :
    IsProbabilityMeasure (poissonLaw r) := by
  infer_instance

/-!
## Uniform and exponential laws
-/

/-- Uniform measure on a measurable set, defined via conditional probability. -/
def uniformMeasure {Alpha : Type*} [MeasurableSpace Alpha]
    (mu : Measure Alpha) (s : Set Alpha) : Measure Alpha :=
  cond mu s

lemma uniformMeasure_isProbabilityMeasure {Alpha : Type*} [MeasurableSpace Alpha]
    (mu : Measure Alpha) {s : Set Alpha} (hs_pos : mu s ≠ 0) (hs_ne_top : mu s ≠ ∞) :
    IsProbabilityMeasure (uniformMeasure mu s) := by
  simpa [uniformMeasure] using cond_isProbabilityMeasure_of_finite (μ := mu) (s := s) hs_pos hs_ne_top

abbrev exponentialLaw (r : ℝ) : Measure ℝ := expMeasure r

lemma exponentialLaw_isProbabilityMeasure {r : ℝ} (hr : 0 < r) :
    IsProbabilityMeasure (exponentialLaw r) := by
  simpa [exponentialLaw] using isProbabilityMeasure_expMeasure (r := r) hr

/-!
## Gaussian law
-/

abbrev gaussianLaw (m : ℝ) (v : NNReal) : Measure ℝ := gaussianReal m v

lemma gaussianLaw_isProbabilityMeasure (m : ℝ) (v : NNReal) :
    IsProbabilityMeasure (gaussianLaw m v) := by
  simpa [gaussianLaw] using (inferInstance : IsProbabilityMeasure (gaussianReal m v))

abbrev stdNormalLaw : Measure ℝ := stdNormalMeasure

end ProbabilityTheory
