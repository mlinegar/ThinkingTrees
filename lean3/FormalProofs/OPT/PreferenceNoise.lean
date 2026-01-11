import Mathlib
import FormalProofs.OPT.OracleMeasurable

/-!
# FormalProofs/OPT/PreferenceNoise.lean

Abstract preference noise models for pairwise comparisons.

The model supplies a probability that action `a` is preferred to action `b`
given a document `x`. We keep this intentionally abstract so different
preference models (BTL, Thurstone, etc.) can be instantiated later.
-/

set_option linter.mathlibStandardSet false

open scoped Classical
open scoped NNReal

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace OPT

/-!
## Preference Noise Model
-/

/-- Pairwise preference noise model. -/
structure PreferenceNoiseModel (Strings A : Type*) where
  /-- P(a ≻ b | x) as a nonnegative real. -/
  prefProb : Strings → A → A → ℝ≥0
  /-- Probability is bounded by 1. -/
  prob_le_one : ∀ x a b, prefProb x a b ≤ 1
  /-- Symmetry: P(a ≻ b) + P(b ≻ a) = 1. -/
  symmetry : ∀ x a b, prefProb x a b + prefProb x b a = 1

/-- Bernoulli label distribution for a preference comparison. -/
def PreferenceNoiseModel.labelPMF {Strings A : Type*}
    (model : PreferenceNoiseModel Strings A) (x : Strings) (a b : A) : PMF Bool :=
  PMF.bernoulli (model.prefProb x a b) (model.prob_le_one x a b)

/-!
## Oracle-Indexed Noise
-/

/-- Noise model depends on `x` only through the oracle value. -/
def OracleIndexedNoise {Strings Y A : Type*} [PseudoMetricSpace Y]
    (fstar : Strings → Y) (model : PreferenceNoiseModel Strings A) : Prop :=
  ∀ x x' a b, dist (fstar x) (fstar x') = 0 → model.prefProb x a b = model.prefProb x' a b

end OPT
