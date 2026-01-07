/-
FormalProofs/LocalLaws.lean

Local Laws for oracle preservation:
- L1: Leaf idempotence
- L2: Internal node idempotence
- L3: Global idempotence on range
- Egu: Tree expectation
- InRange: Support membership
-/

import FormalProofs.CoreDefinitions

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

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]

/-!
## Tree Expectation
-/

/-- Expected value of f under the hierarchical reduction of tree T -/
def Egu (g : Summarizer Strings) (T : BinTree Strings) (f : Strings → ℝ) : ℝ :=
  ∑' z, (reduce g T z).toReal * f z

/-!
## Local Laws
-/

/-- L1: Expected distortion is 0 at each leaf (leaf idempotence) -/
def L1 (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y) : Prop :=
  ∀ b, b ∈ leaves T → Eg g (fun z => D fstar z b) b = 0

/-- L2: Expected distortion is 0 at each internal node -/
def L2 (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y) : Prop :=
  ∀ p, p ∈ internal_nodes T →
    let (T_L, T_R) := p
    Egu g (BinTree.node T_L T_R) (fun z => D fstar z (S (BinTree.node T_L T_R))) = 0

/-- InRange: z is in the support of g(x) for some x -/
def InRange (g : Summarizer Strings) (z : Strings) : Prop := ∃ x, z ∈ (g x).support

/-- L3: Expected distortion is 0 for any oracle value in the range -/
def L3 (g : Summarizer Strings) (fstar : Strings → Y) : Prop :=
  ∀ Z, InRange g Z → Eg g (fun z => D fstar z Z) Z = 0

end
