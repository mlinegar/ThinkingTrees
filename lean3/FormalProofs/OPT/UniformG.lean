import FormalProofs.OPT.CoreDefinitions

/-!
# FormalProofs/OPT/UniformG.lean

Shared interface for theorem-facing C-TreePO / Thinking Trees statements.

The theorem-facing contract has one carrier space. Raw leaves are embedded
into that carrier, merge inputs are constructed inside that carrier, and the
learned map itself is a single endomorphism `g : Carrier → Carrier` applied at
every tree node. Downstream oracles/readouts then have type `Carrier → Y`.
-/

set_option linter.mathlibStandardSet false
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

/-- One shared compositional summary operator.

`Carrier` is the single theorem-facing space: it contains encoded leaf
material and all intermediate summary states. `leafInput` embeds raw leaves
into `Carrier`; `mergeInput` forms a carrier element from two carrier states.
The field `g` is the one learned summarizer, with type `Carrier → Carrier`,
used at both leaves and internal nodes. -/
structure UniformG (Leaf Carrier : Type*) where
  leafInput : Leaf → Carrier
  mergeInput : Carrier → Carrier → Carrier
  g : Carrier → Carrier

namespace UniformG

/-- Leaf state induced by one shared `g`. -/
def leaf {Leaf Carrier : Type*} (G : UniformG Leaf Carrier) : Leaf → Carrier :=
  fun x => G.g (G.leafInput x)

/-- Merge state induced by the same shared `g`. -/
def merge {Leaf Carrier : Type*} (G : UniformG Leaf Carrier) :
    Carrier → Carrier → Carrier :=
  fun s t => G.g (G.mergeInput s t)

/-- Bottom-up tree evaluation induced by one shared `g`. -/
def treeEval {Leaf Carrier : Type*} (G : UniformG Leaf Carrier) :
    BinTree Leaf → Carrier
  | BinTree.leaf x => leaf G x
  | BinTree.node TL TR => merge G (treeEval G TL) (treeEval G TR)

/-- Special case where raw leaves already live in the carrier space. -/
def onCarrier {Carrier : Type*}
    (combine : Carrier → Carrier → Carrier)
    (g : Carrier → Carrier) :
    UniformG Carrier Carrier where
  leafInput := id
  mergeInput := combine
  g := g

@[simp] theorem leaf_onCarrier {Carrier : Type*}
    (combine : Carrier → Carrier → Carrier)
    (g : Carrier → Carrier) (x : Carrier) :
    leaf (onCarrier combine g) x = g x :=
  rfl

@[simp] theorem merge_onCarrier {Carrier : Type*}
    (combine : Carrier → Carrier → Carrier)
    (g : Carrier → Carrier) (s t : Carrier) :
    merge (onCarrier combine g) s t = g (combine s t) :=
  rfl

@[simp] theorem treeEval_leaf {Leaf Carrier : Type*}
    (G : UniformG Leaf Carrier) (x : Leaf) :
    treeEval G (BinTree.leaf x) = leaf G x :=
  rfl

@[simp] theorem treeEval_node {Leaf Carrier : Type*}
    (G : UniformG Leaf Carrier) (TL TR : BinTree Leaf) :
    treeEval G (BinTree.node TL TR) =
      merge G (treeEval G TL) (treeEval G TR) :=
  rfl

end UniformG

end FormalProofs.OPT
