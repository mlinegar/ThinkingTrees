import FormalProofs.OPT.SlicedContextualSufficiency

/-!
# FormalProofs/OPT/UnifiedGEstimator.lean

This file packages the theorem-facing version of the package-level state-model
contract:

* every learned state model uses one shared `UniformG`;
* problem adapters supply the oracle/query family to preserve; and
* estimator choices are different ways to realize that same shared-`g`
  contract, not different formal contracts.

The file intentionally does **not** formalize SGD, architecture search, or a
specific neural-operator implementation. It records the certification routes
that a runtime estimator may discharge: exact-state decoding, finite contextual
responses, sliced contextual responses, and composed two-sided readout error.
-/

set_option linter.mathlibStandardSet false

open scoped Nat
open scoped Classical

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

variable {X Ctx Carrier State Y Slice SliceVal Estimator : Type*}

/-! ## Problem and estimator records -/

/-- A problem adapter is just the contextual query family whose fibers the
learned state must preserve. Markov, HLL, LDA, and other domains can all be
presented through this one field. -/
structure UnifiedGProblem (X Ctx Y : Type*) where
  query : Ctx → X → Y

namespace UnifiedGProblem

/-- The full contextual-response signature associated with a problem adapter. -/
def responseSignature (P : UnifiedGProblem X Ctx Y) (x : X) : Ctx → Y :=
  ResponseSignature P.query x

/-- Two-sided compositional problems are the common C-TreePO case:
`query (left,right) x = fstar (left * x * right)`. -/
def twoSided [Monoid X] (fstar : X → Y) : UnifiedGProblem X (X × X) Y where
  query := TwoSidedContextQuery fstar

@[simp] theorem twoSided_query [Monoid X] (fstar : X → Y) (ctx : X × X) (x : X) :
    (twoSided (X := X) fstar).query ctx x = fstar (ctx.1 * x * ctx.2) :=
  rfl

end UnifiedGProblem

/-- A unified-`g` estimator is a realized shared endomorphism on one carrier
space. The estimator may be exact, learned, neural-operator-based, sliced, or
external; the theorem-facing contract is always the same `UniformG`. -/
structure UnifiedGEstimator (X Carrier : Type*) where
  G : UniformG X Carrier

namespace UnifiedGEstimator

/-- Leaf state induced by the shared `g`. -/
def leafState (E : UnifiedGEstimator X Carrier) : X → Carrier :=
  UniformG.leaf E.G

/-- Merge state induced by the same shared `g`. -/
def mergeState (E : UnifiedGEstimator X Carrier) : Carrier → Carrier → Carrier :=
  UniformG.merge E.G

/-- Bottom-up tree evaluation induced by the same shared `g`. -/
def treeEval (E : UnifiedGEstimator X Carrier) : BinTree X → Carrier :=
  UniformG.treeEval E.G

@[simp] theorem leafState_eq (E : UnifiedGEstimator X Carrier) (x : X) :
    E.leafState x = E.G.g (E.G.leafInput x) :=
  rfl

@[simp] theorem mergeState_eq
    (E : UnifiedGEstimator X Carrier) (s t : Carrier) :
    E.mergeState s t = E.G.g (E.G.mergeInput s t) :=
  rfl

@[simp] theorem treeEval_leaf (E : UnifiedGEstimator X Carrier) (x : X) :
    E.treeEval (BinTree.leaf x) = E.leafState x :=
  rfl

@[simp] theorem treeEval_node
    (E : UnifiedGEstimator X Carrier) (TL TR : BinTree X) :
    E.treeEval (BinTree.node TL TR) =
      E.mergeState (E.treeEval TL) (E.treeEval TR) :=
  rfl

end UnifiedGEstimator

/-- A family of possible estimator realizations. This is the formal counterpart
of a package-level `estimator = ...` knob: different estimators may realize
different `UniformG`s, but they are certified through one shared interface. -/
structure UnifiedGEstimatorFamily (Estimator X Carrier : Type*) where
  realize : Estimator → UnifiedGEstimator X Carrier

/-! ## Problem-level sufficiency predicates -/

/-- A realized unified-`g` estimator is sufficient for a problem when its leaf
states refine the problem's contextual-response fibers. -/
def UnifiedGQuerySufficient
    (E : UnifiedGEstimator X Carrier)
    (P : UnifiedGProblem X Ctx Y) : Prop :=
  QuerySufficient E.leafState P.query

/-- Approximate version of `UnifiedGQuerySufficient`. -/
def UnifiedGQuerySufficientWithin
    [PseudoMetricSpace Y]
    (ε : ℝ)
    (E : UnifiedGEstimator X Carrier)
    (P : UnifiedGProblem X Ctx Y) : Prop :=
  QuerySufficientWithin ε E.leafState P.query

/-- Finite-context version, matching empirical contextual-response supervision. -/
def UnifiedGQuerySufficientOn
    [DecidableEq Ctx]
    (contexts : Finset Ctx)
    (E : UnifiedGEstimator X Carrier)
    (P : UnifiedGProblem X Ctx Y) : Prop :=
  QuerySufficientOn contexts E.leafState P.query

/-! ## Certification route 1: readout or exact-state decoding -/

/-- If a contextual readout from leaf states realizes every problem response,
then the unified-`g` estimator is sufficient for that problem. -/
theorem unifiedG_contextReadoutRealizes_implies_querySufficient
    {E : UnifiedGEstimator X Carrier}
    {P : UnifiedGProblem X Ctx Y}
    {readout : Carrier → Ctx → Y}
    (hReadout : ContextReadoutRealizes E.leafState P.query readout) :
    UnifiedGQuerySufficient E P := by
  intro x y hxy c
  calc
    P.query c x = readout (E.leafState x) c := (hReadout x c).symm
    _ = readout (E.leafState y) c := by rw [hxy]
    _ = P.query c y := hReadout y c

/-- If a learned state decodes to an exact theorem-domain state, and that exact
state has a query readout, the learned unified-`g` leaf state is sufficient.

This is the generic Markov/HLL route: Markov may decode to
`(count, first, last)`, while HLL may decode to register state. -/
theorem unifiedG_exactStateDecoder_implies_querySufficient
    {E : UnifiedGEstimator X Carrier}
    {P : UnifiedGProblem X Ctx Y}
    {exactState : X → State}
    {stateReadout : State → Ctx → Y}
    {decode : Carrier → State}
    (hStateReadout : ∀ x c, stateReadout (exactState x) c = P.query c x)
    (hDecode : ∀ x, decode (E.leafState x) = exactState x) :
    UnifiedGQuerySufficient E P := by
  apply unifiedG_contextReadoutRealizes_implies_querySufficient
    (E := E)
    (P := P)
    (readout := fun r c => stateReadout (decode r) c)
  intro x c
  change stateReadout (decode (E.leafState x)) c = P.query c x
  rw [hDecode x]
  exact hStateReadout x c

/-! ## Certification route 2: finite contextual responses -/

/-- Finite contextual-response preservation plus a finite-context cover
certifies the full problem-level sufficiency condition. -/
theorem unifiedG_finiteContext_zeroLoss_implies_querySufficient
    [DecidableEq Ctx]
    {contexts : Finset Ctx}
    {E : UnifiedGEstimator X Carrier}
    {P : UnifiedGProblem X Ctx Y}
    (hCover : FiniteContextCovers contexts P.query)
    (hZero : UnifiedGQuerySufficientOn contexts E P) :
    UnifiedGQuerySufficient E P :=
  finiteContext_zeroLoss_implies_querySufficient
    (contexts := contexts)
    (rep := E.leafState)
    (query := P.query)
    hCover
    hZero

/-- Approximate finite contextual-response preservation plus an approximate
finite-context cover certifies approximate problem-level sufficiency. -/
theorem unifiedG_finiteContext_within_implies_querySufficientWithin
    [PseudoMetricSpace Y]
    [DecidableEq Ctx]
    {contexts : Finset Ctx}
    {ε δ : ℝ}
    {E : UnifiedGEstimator X Carrier}
    {P : UnifiedGProblem X Ctx Y}
    (hCover : FiniteContextCoversWithin contexts ε δ P.query)
    (hWithin : QuerySufficientWithinOn contexts ε E.leafState P.query) :
    UnifiedGQuerySufficientWithin δ E P :=
  finiteContext_within_implies_querySufficientWithin
    (contexts := contexts)
    (rep := E.leafState)
    (query := P.query)
    hCover
    hWithin

/-! ## Certification route 3: sliced contextual responses -/

/-- Selected response-signature slices are just slices of the problem adapter's
query family. -/
def UnifiedGSlicedResponse
    (P : UnifiedGProblem X Ctx Y)
    (slice : Slice → (Ctx → Y) → SliceVal)
    (x : X) : Slice → SliceVal :=
  SlicedResponseSignature P.query slice x

/-- Exact finite-slice preservation by a unified-`g` estimator. -/
def UnifiedGSlicedQuerySufficientOn
    (selected : Finset Slice)
    (E : UnifiedGEstimator X Carrier)
    (P : UnifiedGProblem X Ctx Y)
    (slice : Slice → (Ctx → Y) → SliceVal) : Prop :=
  SlicedQuerySufficientOn selected E.leafState P.query slice

/-- Finite selected-slice preservation plus a slice-cover condition certifies
ordinary problem-level sufficiency. -/
theorem unifiedG_finiteSliced_zeroLoss_implies_querySufficient
    {selected : Finset Slice}
    {E : UnifiedGEstimator X Carrier}
    {P : UnifiedGProblem X Ctx Y}
    {slice : Slice → (Ctx → Y) → SliceVal}
    (hCover : FiniteSlicesCoverResponseFibers selected P.query slice)
    (hZero : UnifiedGSlicedQuerySufficientOn selected E P slice) :
    UnifiedGQuerySufficient E P :=
  finiteSliced_zeroLoss_implies_querySufficient
    (selected := selected)
    (rep := E.leafState)
    (query := P.query)
    (slice := slice)
    hCover
    hZero

/-- Approximate finite-slice preservation plus an approximate slice-cover
condition certifies approximate problem-level sufficiency. -/
theorem unifiedG_finiteSlicedWithin_implies_querySufficientWithin
    [PseudoMetricSpace SliceVal]
    [PseudoMetricSpace Y]
    {selected : Finset Slice}
    {δ ε : ℝ}
    {E : UnifiedGEstimator X Carrier}
    {P : UnifiedGProblem X Ctx Y}
    {slice : Slice → (Ctx → Y) → SliceVal}
    (hCover : FiniteSlicesCoverResponseFibersWithin selected δ ε P.query slice)
    (hWithin : SlicedQuerySufficientWithinOn selected δ E.leafState P.query slice) :
    UnifiedGQuerySufficientWithin ε E P :=
  finiteSlicedWithin_implies_querySufficientWithin
    (selected := selected)
    (rep := E.leafState)
    (query := P.query)
    (slice := slice)
    hCover
    hWithin

/-! ## Certification route 4: two-sided composed readouts -/

/-- Exact composed two-sided readout behavior certifies two-sided contextual
sufficiency for the leaf state induced by a realized unified `g`. -/
theorem unifiedG_composedTwoSidedReadoutExact_implies_querySufficient
    [Monoid X]
    {E : UnifiedGEstimator X Carrier}
    {readout : Carrier → Y}
    {fstar : X → Y}
    (hExact :
      ∀ left x right,
        readout
          (E.mergeState
            (E.mergeState (E.leafState left) (E.leafState x))
            (E.leafState right)) =
          fstar (left * x * right)) :
    UnifiedGQuerySufficient E (UnifiedGProblem.twoSided fstar) :=
  uniformComposedTwoSidedReadoutExact_implies_twoSidedContextSufficient
    (G := E.G)
    (readout := readout)
    (fstar := fstar)
    hExact

/-- Approximate composed two-sided readout behavior certifies approximate
two-sided contextual sufficiency, with the existing `2ε` slack. -/
theorem unifiedG_composedTwoSidedReadoutWithin_implies_querySufficientWithin
    [Monoid X]
    [PseudoMetricSpace Y]
    {ε : ℝ}
    {E : UnifiedGEstimator X Carrier}
    {readout : Carrier → Y}
    {fstar : X → Y}
    (hApprox :
      ∀ left x right,
        dist
          (readout
            (E.mergeState
              (E.mergeState (E.leafState left) (E.leafState x))
              (E.leafState right)))
          (fstar (left * x * right)) ≤ ε) :
    UnifiedGQuerySufficientWithin
      (2 * ε)
      E
      (UnifiedGProblem.twoSided fstar) :=
  uniformComposedTwoSidedReadoutWithinEps_implies_twoSidedContextSufficientWithin
    (G := E.G)
    (readout := readout)
    (fstar := fstar)
    hApprox

/-! ## Estimator-family wrappers -/

/-- A realized member of an estimator family is certified by exact-state
decoding through the same generic route. -/
theorem unifiedG_estimatorFamily_exactStateDecoder_implies_querySufficient
    {F : UnifiedGEstimatorFamily Estimator X Carrier}
    {η : Estimator}
    {P : UnifiedGProblem X Ctx Y}
    {exactState : X → State}
    {stateReadout : State → Ctx → Y}
    {decode : Carrier → State}
    (hStateReadout : ∀ x c, stateReadout (exactState x) c = P.query c x)
    (hDecode : ∀ x, decode ((F.realize η).leafState x) = exactState x) :
    UnifiedGQuerySufficient (F.realize η) P :=
  unifiedG_exactStateDecoder_implies_querySufficient
    (E := F.realize η)
    (P := P)
    hStateReadout
    hDecode

/-- A realized member of an estimator family is certified by approximate
two-sided composed-readout error through the same generic route. -/
theorem unifiedG_estimatorFamily_composedTwoSidedWithin_implies_querySufficientWithin
    [Monoid X]
    [PseudoMetricSpace Y]
    {ε : ℝ}
    {F : UnifiedGEstimatorFamily Estimator X Carrier}
    {η : Estimator}
    {readout : Carrier → Y}
    {fstar : X → Y}
    (hApprox :
      ∀ left x right,
        dist
          (readout
            ((F.realize η).mergeState
              ((F.realize η).mergeState
                ((F.realize η).leafState left)
                ((F.realize η).leafState x))
              ((F.realize η).leafState right)))
          (fstar (left * x * right)) ≤ ε) :
    UnifiedGQuerySufficientWithin
      (2 * ε)
      (F.realize η)
      (UnifiedGProblem.twoSided fstar) :=
  unifiedG_composedTwoSidedReadoutWithin_implies_querySufficientWithin
    (E := F.realize η)
    (readout := readout)
    (fstar := fstar)
    hApprox

end FormalProofs.OPT

