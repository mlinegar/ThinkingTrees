import Mathlib
import FormalProofs.OPT.PreferenceNoise

/-!
# FormalProofs/OPT/SamplingModel.lean

Sampling model for preference data:
- document distribution
- pair generator
- preference noise model

This provides a compact generative description of preference datasets.
-/

set_option linter.mathlibStandardSet false

open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace OPT

/-!
## Preference Examples and Datasets
-/

/-- A preference example: document and a winner/loser pair. -/
structure PreferenceExample (Strings A : Type*) where
  x : Strings
  winner : A
  loser : A

/-- Finite preference dataset of size n. -/
abbrev PreferenceDataset (Strings A : Type*) (n : ℕ) := Fin n → PreferenceExample Strings A

/-- An observed preference tuple with a boolean label. -/
abbrev PreferenceObservation (Strings A : Type*) := Strings × A × A × Bool

/-!
## Pair Generators
-/

/-- Pair generator: draws (a_w, a_l) conditioned on document x. -/
def PairGenerator (Strings A : Type*) := Strings → PMF (A × A)

/-- Oracle-indexed pair generator. -/
def OracleIndexedPairGen {Strings Y A : Type*} [PseudoMetricSpace Y]
    (fstar : Strings → Y) (gen : PairGenerator Strings A) : Prop :=
  ∀ x x', dist (fstar x) (fstar x') = 0 → gen x = gen x'

/-!
## Preference Sampling Model
-/

/-- Full generative model for preference data. -/
structure PreferenceSamplingModel (Strings A : Type*) where
  docDist : PMF Strings
  pairGen : PairGenerator Strings A
  noise : PreferenceNoiseModel Strings A

/-- Single-observation distribution implied by the sampling model. -/
def observationPMF {Strings A : Type*}
    (model : PreferenceSamplingModel Strings A) : PMF (PreferenceObservation Strings A) :=
  model.docDist.bind (fun x =>
    (model.pairGen x).bind (fun p =>
      (model.noise.labelPMF x p.1 p.2).map (fun b => (x, p.1, p.2, b))))

end OPT
