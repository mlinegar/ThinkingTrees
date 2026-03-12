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

/-!
## Tree Preference Sampling Model (TreePO)

This extends the pairwise sampling model to the tree setting by inserting
an explicit **node sampler** between document sampling and action sampling.

This mirrors the TreePO objective:
  E_{x~μ} E_{u~q(·|x)} E_{g~gen(node)} [loss(span(u), g)]
-/

/-- Node sampler: draws a node conditioned on document x. -/
def NodeSampler (Strings Node : Type*) := Strings → PMF Node

/-- Group generator at a node (k-wise preference candidates). -/
def NodeGroupGenerator (Node A : Type*) (k : ℕ) := Node → PMF (Fin k → A)

/-- Oracle-indexed node sampler: depends on x only through f*(x). -/
def OracleIndexedNodeSampler {Strings Y Node : Type*} [PseudoMetricSpace Y]
    (fstar : Strings → Y) (q : NodeSampler Strings Node) : Prop :=
  ∀ x x', dist (fstar x) (fstar x') = 0 → q x = q x'

/-- Tree-based preference sampling model (document → node → group). -/
structure TreePreferenceSamplingModel (Strings Node A : Type*) (k : ℕ) where
  docDist : PMF Strings
  nodeSampler : NodeSampler Strings Node
  nodeSpan : Node → Strings
  groupGen : NodeGroupGenerator Node A k

/-- Expected preference loss under a tree sampling model. -/
noncomputable def ExpectedTreePreferenceLoss {Strings Node A : Type*} {k : ℕ}
    (model : TreePreferenceSamplingModel Strings Node A k)
    (loss : Strings → (Fin k → A) → ℝ) : ℝ :=
  ∑' x, (model.docDist x).toReal *
    ∑' u, (model.nodeSampler x u).toReal *
      ∑' g, (model.groupGen u g).toReal * loss (model.nodeSpan u) g

end OPT
