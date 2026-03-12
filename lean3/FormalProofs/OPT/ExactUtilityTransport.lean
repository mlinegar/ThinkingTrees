import FormalProofs.OPT.CoreDefinitions
import FormalProofs.OPT.PreferenceLearning
import FormalProofs.OPT.SketchSummaryOperators

/-!
# FormalProofs/OPT/ExactUtilityTransport.lean

This file packages the exact-utility transport layer that sits underneath the
new exact utility-transport simulations.

There are two complementary results here:

1. **Oracle-indexed objective transport**:
   any objective that factors through an exact latent/feature oracle inherits
   the generic zero-distortion transport theorem already proved in
   `PreferenceLearning.lean`.
2. **Exact mergeable-state utility preservation**:
   if a latent state is represented by an exact mergeable fold
   (`encode` at leaves, `merge` at internal nodes), then **any utility on that
   exact state** is preserved by the tree exactly.

This is the formal bridge we need for the exact Markov / nonseparable /
boundary-topic simulation lanes:

- the objective family can vary (supervised state loss, reward, pairwise loss,
  group/listwise loss, PPO-style reward optimization);
- the core latent state can still be treated as the theorem-bearing object.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators
open scoped Real
open scoped Nat
open scoped Classical
open scoped Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

section OracleIndexedObjectives

variable {Strings : Type*} [Monoid Strings]
variable {Feature : Type*} [Encodable Feature]
variable {α : Type*}

/-- An oracle-indexed objective obtained by composing a feature/state oracle
with an objective on that oracle value. -/
def featureIndexedObjective
    (feature : Strings → Feature) (objective : Feature → α → ℝ) :
    Strings → α → ℝ :=
  fun x a => objective (feature x) a

/-- A direct supervised-state loss: compare a predicted feature/state `a` to the
true feature/state of document `x`. -/
def supervisedStateLoss
    (feature : Strings → Feature) (loss : Feature → Feature → ℝ) :
    Strings → Feature → ℝ :=
  fun x a => loss a (feature x)

/-- Constant generator used to view deterministic utility evaluation as a generic
expected loss. -/
def trivialExampleGenerator (Strings : Type*) : Strings → PMF PUnit :=
  fun _ => PMF.pure PUnit.unit

/-- Objectives that depend on a document only through an encoded feature/state
are oracle-measurable for the encoded-feature oracle. -/
lemma oracleMeasurableLossGeneric_of_featureIndexedObjective
    (feature : Strings → Feature) (objective : Feature → α → ℝ) :
    OracleMeasurableLossGeneric
      (featureIndexedObjective (Strings := Strings) feature objective)
      (encodedOracle (Strings := Strings) feature) := by
  intro x x' a hdist
  have hreal :
      encodedOracle (Strings := Strings) feature x =
        encodedOracle (Strings := Strings) feature x' := by
    exact dist_eq_zero.mp hdist
  have hcast :
      ((Encodable.encode (feature x) : ℕ) : ℝ) =
        ((Encodable.encode (feature x') : ℕ) : ℝ) := by
    simpa [encodedOracle] using hreal
  have hcode : Encodable.encode (feature x) = Encodable.encode (feature x') := by
    exact_mod_cast hcast
  have hfeature : feature x = feature x' := Encodable.encode_injective hcode
  simp [featureIndexedObjective, hfeature]

/-- The constant unit generator is oracle-indexed for every oracle. -/
lemma oracleIndexedGenGeneric_trivialExampleGenerator
    (fstar : Strings → ℝ) :
    OracleIndexedGenGeneric
      (trivialExampleGenerator Strings) fstar := by
  intro x x' hdist
  simp [trivialExampleGenerator]

/-- Generic oracle-indexed feature/state objective transport.

This is the exact objective-level theorem used by the new utility-transport
suite: once the objective factors through a feature/state oracle, the generic
zero-distortion theorem applies immediately. -/
theorem featureIndexedObjective_eq_of_zero_dist
    (feature : Strings → Feature)
    (objective : Feature → α → ℝ)
    (gen : Strings → PMF α)
    (μ_X μ_Z : PMF Strings)
    (h_zero :
      ∀ z x, z ∈ μ_Z.support → x ∈ μ_X.support →
        dist ((encodedOracle (Strings := Strings) feature) z)
          ((encodedOracle (Strings := Strings) feature) x) = 0)
    (h_gen :
      OracleIndexedGenGeneric gen
        (encodedOracle (Strings := Strings) feature)) :
    ExpectedLossGeneric
      (featureIndexedObjective (Strings := Strings) feature objective) μ_X gen =
    ExpectedLossGeneric
      (featureIndexedObjective (Strings := Strings) feature objective) μ_Z gen := by
  exact expected_loss_eq_of_zero_dist_generic
    (fstar := encodedOracle (Strings := Strings) feature)
    (loss := featureIndexedObjective (Strings := Strings) feature objective)
    (gen := gen)
    (μ_X := μ_X)
    (μ_Z := μ_Z)
    h_zero
    (oracleMeasurableLossGeneric_of_featureIndexedObjective
      (Strings := Strings) (feature := feature) (objective := objective))
    h_gen

/-- Direct supervised-state learning is a special case of the generic
oracle-indexed objective transport theorem. -/
theorem supervisedStateExpectedLoss_eq_of_zero_dist
    (feature : Strings → Feature)
    (loss : Feature → Feature → ℝ)
    (gen : Strings → PMF Feature)
    (μ_X μ_Z : PMF Strings)
    (h_zero :
      ∀ z x, z ∈ μ_Z.support → x ∈ μ_X.support →
        dist ((encodedOracle (Strings := Strings) feature) z)
          ((encodedOracle (Strings := Strings) feature) x) = 0)
    (h_gen :
      OracleIndexedGenGeneric gen
        (encodedOracle (Strings := Strings) feature)) :
    ExpectedLossGeneric
      (supervisedStateLoss (Strings := Strings) feature loss) μ_X gen =
    ExpectedLossGeneric
      (supervisedStateLoss (Strings := Strings) feature loss) μ_Z gen := by
  simpa [supervisedStateLoss, featureIndexedObjective] using
    (featureIndexedObjective_eq_of_zero_dist
      (Strings := Strings)
      (feature := feature)
      (objective := fun y a => loss a y)
      (gen := gen)
      (μ_X := μ_X)
      (μ_Z := μ_Z)
      h_zero h_gen)

end OracleIndexedObjectives

section ExactUtilities

variable {Action State β : Type*}

/-- Utility obtained by normalizing an error function. -/
def normalizedErrorUtility
    (err : Action → State → ℝ) (scale : ℝ) :
    Action → State → ℝ :=
  fun a y => 1 - err a y / scale

/-- Utility regret against an exact/reference action. -/
def utilityRegret
    (utility : Action → State → ℝ)
    (aStar a : Action) (y : State) : ℝ :=
  utility aStar y - utility a y

/-- For normalized exact-state utilities, regret against an exact action is just
the normalized error gap. -/
theorem normalizedErrorUtility_regret_eq_error_gap
    (err : Action → State → ℝ) (scale : ℝ)
    {aStar a : Action} {y : State}
    (h_exact : err aStar y = 0) :
    utilityRegret (normalizedErrorUtility err scale) aStar a y =
      err a y / scale := by
  unfold utilityRegret normalizedErrorUtility
  rw [h_exact]
  ring

/-- Zero regret coincides with zero error for normalized exact-state utilities. -/
theorem normalizedErrorUtility_zero_regret_iff_zero_error
    (err : Action → State → ℝ) (scale : ℝ)
    {aStar a : Action} {y : State}
    (hscale : 0 < scale)
    (h_exact : err aStar y = 0) :
    utilityRegret (normalizedErrorUtility err scale) aStar a y = 0
      ↔ err a y = 0 := by
  rw [normalizedErrorUtility_regret_eq_error_gap
    (err := err) (scale := scale) (aStar := aStar) (a := a) (y := y) h_exact]
  constructor
  · intro hzero
    have hdiv : err a y / scale = 0 := hzero
    have hs : scale ≠ 0 := by linarith
    exact (div_eq_zero_iff).mp hdiv |>.resolve_right hs
  · intro herr
    simp [herr]

end ExactUtilities

section ExactMergeableState

variable {Strings : Type*} [Monoid Strings]
variable {Sketch β : Type*}

/-- Bottom-up exact fold of a mergeable latent state over a tree. -/
def mergeFold
    (encode : Strings → Sketch)
    (merge : Sketch → Sketch → Sketch) :
    BinTree Strings → Sketch
  | BinTree.leaf b => encode b
  | BinTree.node TL TR => merge (mergeFold encode merge TL) (mergeFold encode merge TR)

/-- If a latent feature/state is exactly mergeable, tree folding recovers the
same state as computing the feature on the full span directly. -/
theorem mergeFold_eq_feature
    (encode : Strings → Sketch)
    (merge : Sketch → Sketch → Sketch)
    (feature : Strings → Sketch)
    (h_encode : ∀ x, encode x = feature x)
    (h_merge : ∀ x y, merge (feature x) (feature y) = feature (x * y))
    (T : BinTree Strings) :
    mergeFold encode merge T = feature (S T) := by
  induction T with
  | leaf b =>
      simpa [mergeFold] using h_encode b
  | node TL TR ihL ihR =>
      simp [mergeFold, S, ihL, ihR, h_merge]

/-- Any downstream utility on an exact mergeable latent state is preserved by
tree reduction exactly. This is the exact-control theorem used by the
simulation suite. -/
theorem mergeableStateUtility_exact_on_tree
    (encode : Strings → Sketch)
    (merge : Sketch → Sketch → Sketch)
    (feature : Strings → Sketch)
    (h_encode : ∀ x, encode x = feature x)
    (h_merge : ∀ x y, merge (feature x) (feature y) = feature (x * y))
    (u : Sketch → β)
    (T : BinTree Strings) :
    u (mergeFold encode merge T) = u (feature (S T)) := by
  rw [mergeFold_eq_feature
    (encode := encode) (merge := merge) (feature := feature) h_encode h_merge T]

end ExactMergeableState

end FormalProofs.OPT
