import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.MeasureTheory.Measure.CharacteristicFunction

/-!
# FormalProofs/CLT/Normal.lean

Standard normal distribution on `ℝ`, expressed via mathlib's `gaussianReal`.
This avoids re-deriving measure-theoretic facts and directly reuses the
characteristic function formula already available in mathlib.
-/

set_option linter.mathlibStandardSet false

open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace ProbabilityTheory

open MeasureTheory

/-!
## Standard Normal Measure
-/

def stdNormalMeasure : Measure ℝ :=
  gaussianReal (0 : ℝ) (1 : NNReal)

instance : IsProbabilityMeasure stdNormalMeasure := by
  simpa [stdNormalMeasure] using
    (inferInstance : IsProbabilityMeasure (gaussianReal (0 : ℝ) (1 : NNReal)))

/-!
## Characteristic Function
-/

lemma charFun_stdNormal (t : ℝ) :
    charFun stdNormalMeasure t = Complex.exp (-((t : ℂ) ^ 2) / 2) := by
  -- `gaussianReal` uses variance `v : ℝ≥0`; here `v = 1`.
  have h := charFun_gaussianReal (μ := (0 : ℝ)) (v := (1 : NNReal)) t
  -- Simplify the mean/variance parameters.
  have h' :
      charFun stdNormalMeasure t = Complex.exp (-( (t : ℂ) ^ 2 / 2)) := by
    simpa [stdNormalMeasure, mul_comm, mul_left_comm, mul_assoc] using h
  have h_exp : (-( (t : ℂ) ^ 2 / 2)) = (-((t : ℂ) ^ 2) / 2) := by
    ring
  simpa [h_exp] using h'

end ProbabilityTheory
