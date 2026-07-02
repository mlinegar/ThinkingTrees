import Mathlib.Data.Finset.Basic
import Mathlib.Topology.MetricSpace.Basic
import FormalProofs.OPT.ContextualQuerySufficiency

/-!
# FormalProofs/OPT/SlicedContextualSufficiency.lean

This file formalizes the deterministic core behind the SSS/NASSS-style
"learn many low-dimensional slices" objective.

The probabilistic part of SSS -- drawing random slice directions -- is not
formalized here. Once a training run has selected a finite set of slice
functions, Lean treats those functions as deterministic probes of the full
contextual response signature

`ResponseSignature query x = fun c => query c x`.

The bridge says that if representation collisions preserve all selected slice
values, and those selected slices cover the full response-signature fibers,
then the representation is contextually sufficient in the existing
`QuerySufficient` sense. The approximate version uses pseudometric slack and
feeds directly into `QuerySufficientWithin`.
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

variable {X Ctx Rep Y Slice SliceVal : Type*}

/-- Apply a family of slice functions to the full contextual response signature.

For SSS/NASSS intuition, `slice s` is a deterministic Lean stand-in for a
selected direction such as `phi_s^T R_K(x)`. -/
def SlicedResponseSignature
    (query : Ctx → X → Y)
    (slice : Slice → (Ctx → Y) → SliceVal)
    (x : X) : Slice → SliceVal :=
  fun s => slice s (ResponseSignature query x)

/-- Exact all-slice sufficiency: representation collisions preserve every
selected slice value. -/
def SlicedQuerySufficient
    (rep : X → Rep)
    (query : Ctx → X → Y)
    (slice : Slice → (Ctx → Y) → SliceVal) : Prop :=
  ∀ ⦃x y : X⦄,
    rep x = rep y →
      ∀ s : Slice,
        SlicedResponseSignature query slice x s =
          SlicedResponseSignature query slice y s

/-- Slice-cover condition: equality of all slice values implies equality of the
full contextual response signature. -/
def SlicesCoverResponseFibers
    (query : Ctx → X → Y)
    (slice : Slice → (Ctx → Y) → SliceVal) : Prop :=
  ∀ ⦃x y : X⦄,
    (∀ s : Slice,
      SlicedResponseSignature query slice x s =
        SlicedResponseSignature query slice y s) →
      ResponseSignature query x = ResponseSignature query y

/-- If the selected slices cover response-signature fibers, exact sliced
sufficiency implies ordinary contextual query sufficiency. -/
theorem slicedQuerySufficient_implies_querySufficient
    {rep : X → Rep}
    {query : Ctx → X → Y}
    {slice : Slice → (Ctx → Y) → SliceVal}
    (hCover : SlicesCoverResponseFibers query slice)
    (hSliced : SlicedQuerySufficient rep query slice) :
    QuerySufficient rep query := by
  intro x y hxy c
  have hSig : ResponseSignature query x = ResponseSignature query y :=
    hCover (fun s => hSliced hxy s)
  exact congrFun hSig c

section FiniteSlices

/-- Finite selected-slice sufficiency: representation collisions preserve slice
values on the empirical finite slice set. -/
def SlicedQuerySufficientOn
    (selected : Finset Slice)
    (rep : X → Rep)
    (query : Ctx → X → Y)
    (slice : Slice → (Ctx → Y) → SliceVal) : Prop :=
  ∀ ⦃x y : X⦄,
    rep x = rep y →
      ∀ s ∈ selected,
        SlicedResponseSignature query slice x s =
          SlicedResponseSignature query slice y s

/-- Finite slice-cover condition: equality on the selected slice set implies
equality of the full contextual response signature. -/
def FiniteSlicesCoverResponseFibers
    (selected : Finset Slice)
    (query : Ctx → X → Y)
    (slice : Slice → (Ctx → Y) → SliceVal) : Prop :=
  ∀ ⦃x y : X⦄,
    (∀ s ∈ selected,
      SlicedResponseSignature query slice x s =
        SlicedResponseSignature query slice y s) →
      ResponseSignature query x = ResponseSignature query y

/-- If a finite selected slice set covers true response-signature fibers, zero
sliced collision loss on that set implies full contextual sufficiency. -/
theorem finiteSliced_zeroLoss_implies_querySufficient
    {selected : Finset Slice}
    {rep : X → Rep}
    {query : Ctx → X → Y}
    {slice : Slice → (Ctx → Y) → SliceVal}
    (hCover : FiniteSlicesCoverResponseFibers selected query slice)
    (hZero : SlicedQuerySufficientOn selected rep query slice) :
    QuerySufficient rep query := by
  intro x y hxy c
  have hSig : ResponseSignature query x = ResponseSignature query y :=
    hCover (fun s hs => hZero hxy s hs)
  exact congrFun hSig c

end FiniteSlices

/-! ## Concrete slice-cover witnesses -/

/-- Coordinate slice: evaluate a response signature at one context. -/
def CoordinateSlice (c : Ctx) (signature : Ctx → Y) : Y :=
  signature c

/-- All coordinate slices cover the full response-signature fiber. -/
theorem coordinateSlices_cover_responseFibers
    (query : Ctx → X → Y) :
    SlicesCoverResponseFibers
      query
      (fun c : Ctx => CoordinateSlice (Ctx := Ctx) (Y := Y) c) := by
  intro x y hxy
  funext c
  exact hxy c

/-- The full finite set of coordinate slices covers the full response-signature
fiber. This is the finite-context analogue of observing every coordinate of
`R_K(x)`. -/
theorem finiteCoordinateSlices_univ_cover_responseFibers
    [Fintype Ctx]
    [DecidableEq Ctx]
    (query : Ctx → X → Y) :
    FiniteSlicesCoverResponseFibers
      (Finset.univ : Finset Ctx)
      query
      (fun c : Ctx => CoordinateSlice (Ctx := Ctx) (Y := Y) c) := by
  intro x y hxy
  funext c
  exact hxy c (Finset.mem_univ c)

/-- If a family of slice values has a left inverse back to the full signature,
then equality of all slice values covers response-signature equality. This is a
deterministic stand-in for left-invertible slice matrices on finite real-valued
signatures. -/
theorem leftInvertibleSlices_cover_responseFibers
    {slice : Slice → (Ctx → Y) → SliceVal}
    {recover : (Slice → SliceVal) → Ctx → Y}
    (hLeft : ∀ signature : Ctx → Y,
      recover (fun s => slice s signature) = signature)
    (query : Ctx → X → Y) :
    SlicesCoverResponseFibers query slice := by
  intro x y hxy
  have hSlices :
      (fun s => slice s (ResponseSignature query x)) =
        (fun s => slice s (ResponseSignature query y)) := by
    funext s
    exact hxy s
  calc
    ResponseSignature query x
        = recover (fun s => slice s (ResponseSignature query x)) := by
          exact (hLeft (ResponseSignature query x)).symm
    _ = recover (fun s => slice s (ResponseSignature query y)) := by
          rw [hSlices]
    _ = ResponseSignature query y := hLeft (ResponseSignature query y)

/-! ## Approximate sliced sufficiency -/

section ApproximateSlices

variable [PseudoMetricSpace SliceVal] [PseudoMetricSpace Y]

/-- Approximate all-slice sufficiency: representation collisions keep every
slice value within slack `δ`. -/
def SlicedQuerySufficientWithin
    (δ : ℝ)
    (rep : X → Rep)
    (query : Ctx → X → Y)
    (slice : Slice → (Ctx → Y) → SliceVal) : Prop :=
  ∀ ⦃x y : X⦄,
    rep x = rep y →
      ∀ s : Slice,
        dist (SlicedResponseSignature query slice x s)
             (SlicedResponseSignature query slice y s) ≤ δ

/-- Approximate slice-cover condition: if all slice values are within `δ`, then
all contextual query responses are within `ε`. -/
def SlicesCoverResponseFibersWithin
    (δ ε : ℝ)
    (query : Ctx → X → Y)
    (slice : Slice → (Ctx → Y) → SliceVal) : Prop :=
  ∀ ⦃x y : X⦄,
    (∀ s : Slice,
      dist (SlicedResponseSignature query slice x s)
           (SlicedResponseSignature query slice y s) ≤ δ) →
      ∀ c : Ctx, dist (query c x) (query c y) ≤ ε

/-- Approximate all-slice bridge into ordinary approximate contextual
sufficiency. -/
theorem slicedWithin_implies_querySufficientWithin
    {δ ε : ℝ}
    {rep : X → Rep}
    {query : Ctx → X → Y}
    {slice : Slice → (Ctx → Y) → SliceVal}
    (hCover : SlicesCoverResponseFibersWithin δ ε query slice)
    (hSliced : SlicedQuerySufficientWithin δ rep query slice) :
    QuerySufficientWithin ε rep query := by
  intro x y hxy c
  exact hCover (fun s => hSliced hxy s) c

/-- If all coordinate slices are within `ε`, then every contextual response is
within `ε`. -/
theorem coordinateSlices_cover_responseFibersWithin
    (ε : ℝ)
    (query : Ctx → X → Y) :
    SlicesCoverResponseFibersWithin
      ε ε
      query
      (fun c : Ctx => CoordinateSlice (Ctx := Ctx) (Y := Y) c) := by
  intro x y hxy c
  exact hxy c

section FiniteApproximateSlices

/-- Approximate finite selected-slice sufficiency: representation collisions
keep every selected slice value within slack `δ`. -/
def SlicedQuerySufficientWithinOn
    (selected : Finset Slice)
    (δ : ℝ)
    (rep : X → Rep)
    (query : Ctx → X → Y)
    (slice : Slice → (Ctx → Y) → SliceVal) : Prop :=
  ∀ ⦃x y : X⦄,
    rep x = rep y →
      ∀ s ∈ selected,
        dist (SlicedResponseSignature query slice x s)
             (SlicedResponseSignature query slice y s) ≤ δ

/-- Approximate finite slice-cover condition: closeness on the selected slice
set implies contextual-response closeness. -/
def FiniteSlicesCoverResponseFibersWithin
    (selected : Finset Slice)
    (δ ε : ℝ)
    (query : Ctx → X → Y)
    (slice : Slice → (Ctx → Y) → SliceVal) : Prop :=
  ∀ ⦃x y : X⦄,
    (∀ s ∈ selected,
      dist (SlicedResponseSignature query slice x s)
           (SlicedResponseSignature query slice y s) ≤ δ) →
      ∀ c : Ctx, dist (query c x) (query c y) ≤ ε

/-- If the selected finite slices cover contextual responses within slack, then
finite sliced collision control implies approximate contextual sufficiency. -/
theorem finiteSlicedWithin_implies_querySufficientWithin
    {selected : Finset Slice}
    {δ ε : ℝ}
    {rep : X → Rep}
    {query : Ctx → X → Y}
    {slice : Slice → (Ctx → Y) → SliceVal}
    (hCover : FiniteSlicesCoverResponseFibersWithin selected δ ε query slice)
    (hSliced : SlicedQuerySufficientWithinOn selected δ rep query slice) :
    QuerySufficientWithin ε rep query := by
  intro x y hxy c
  exact hCover (fun s hs => hSliced hxy s hs) c

/-- The full finite set of coordinate slices covers all contextual responses
with the same slack. -/
theorem finiteCoordinateSlices_univ_cover_responseFibersWithin
    [Fintype Ctx]
    [DecidableEq Ctx]
    (ε : ℝ)
    (query : Ctx → X → Y) :
    FiniteSlicesCoverResponseFibersWithin
      (Finset.univ : Finset Ctx)
      ε ε
      query
      (fun c : Ctx => CoordinateSlice (Ctx := Ctx) (Y := Y) c) := by
  intro x y hxy c
  exact hxy c (Finset.mem_univ c)

end FiniteApproximateSlices

end ApproximateSlices

end FormalProofs.OPT
