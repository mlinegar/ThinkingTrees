import Mathlib.MeasureTheory.Measure.Typeclasses.Probability
import FormalProofs.OPT.SlicedContextualSufficiency

/-!
# FormalProofs/OPT/RandomSlicedContextualSufficiency.lean

Event-level probability wrappers for SSS/NASSS-style random slice selection.

`SlicedContextualSufficiency.lean` proves the deterministic bridge after a
finite slice set has been selected. This file adds the bounded probabilistic
surface used in paper prose: if a random seed lands in the event where the
selected finite slices cover response fibers and the learned representation has
zero/within sliced collision loss, then the seed's representation is
contextually sufficient. If the good-seed event fails with probability at most
`η`, the contextual-sufficiency failure event also has probability at most
`η`.

This does not prove analytic random-direction coverage, random-matrix
conditioning, estimator consistency, or PAC generalization. Those are supplied,
if needed, as assumptions on the good-seed event.
-/

set_option linter.mathlibStandardSet false

open scoped Classical ENNReal
open MeasureTheory Set

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

variable {Ω X Ctx Rep Y Slice SliceVal : Type*}

/-! ## Exact random finite-slice events -/

/-- Good-seed event for exact finite sliced contextual sufficiency.

At seed `ω`, both the selected finite slice set and the learned representation
may change. The slice family itself may also be seed-dependent, covering random
directions as deterministic functions once `ω` is fixed. -/
def RandomFiniteSlicedGoodEvent
    (selected : Ω → Finset Slice)
    (rep : Ω → X → Rep)
    (query : Ctx → X → Y)
    (slice : Ω → Slice → (Ctx → Y) → SliceVal) : Set Ω :=
  {ω |
    FiniteSlicesCoverResponseFibers (selected ω) query (slice ω) ∧
      SlicedQuerySufficientOn (selected ω) (rep ω) query (slice ω)}

/-- Seedwise bridge: every exact good seed yields contextual sufficiency for
that seed's learned representation. -/
theorem randomFiniteSlicedGoodEvent_implies_querySufficient
    {selected : Ω → Finset Slice}
    {rep : Ω → X → Rep}
    {query : Ctx → X → Y}
    {slice : Ω → Slice → (Ctx → Y) → SliceVal}
    {ω : Ω}
    (hGood : ω ∈ RandomFiniteSlicedGoodEvent selected rep query slice) :
    QuerySufficient (rep ω) query := by
  exact finiteSliced_zeroLoss_implies_querySufficient hGood.1 hGood.2

/-- High-probability transport: if the exact random finite-slice good event
fails with probability at most `η`, then contextual-sufficiency failure also
has probability at most `η`. -/
theorem randomFiniteSliced_contextualSufficiency_failure_prob_le
    [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    {η : ENNReal}
    {selected : Ω → Finset Slice}
    {rep : Ω → X → Rep}
    {query : Ctx → X → Y}
    {slice : Ω → Slice → (Ctx → Y) → SliceVal}
    (hGood :
      μ (RandomFiniteSlicedGoodEvent selected rep query slice)ᶜ ≤ η) :
    μ {ω | ¬ QuerySufficient (rep ω) query} ≤ η := by
  have hSubset :
      {ω | ¬ QuerySufficient (rep ω) query} ⊆
        (RandomFiniteSlicedGoodEvent selected rep query slice)ᶜ := by
    intro ω hBad hωGood
    exact hBad (randomFiniteSlicedGoodEvent_implies_querySufficient hωGood)
  exact le_trans (measure_mono hSubset) hGood

/-! ## Approximate random finite-slice events -/

section Approximate

variable [PseudoMetricSpace SliceVal] [PseudoMetricSpace Y]

/-- Good-seed event for approximate finite sliced contextual sufficiency.

The event packages both assumptions needed by the deterministic approximate
slice bridge: selected slices cover full contextual responses within `ε`
whenever their slice distances are within `δ`, and representation collisions
keep the selected slice distances within `δ`. -/
def RandomFiniteSlicedWithinGoodEvent
    (selected : Ω → Finset Slice)
    (δ ε : ℝ)
    (rep : Ω → X → Rep)
    (query : Ctx → X → Y)
    (slice : Ω → Slice → (Ctx → Y) → SliceVal) : Set Ω :=
  {ω |
    FiniteSlicesCoverResponseFibersWithin (selected ω) δ ε query (slice ω) ∧
      SlicedQuerySufficientWithinOn (selected ω) δ (rep ω) query (slice ω)}

/-- Seedwise approximate bridge from a random finite-slice good event to
approximate contextual sufficiency. -/
theorem randomFiniteSlicedWithinGoodEvent_implies_querySufficientWithin
    {selected : Ω → Finset Slice}
    {δ ε : ℝ}
    {rep : Ω → X → Rep}
    {query : Ctx → X → Y}
    {slice : Ω → Slice → (Ctx → Y) → SliceVal}
    {ω : Ω}
    (hGood :
      ω ∈ RandomFiniteSlicedWithinGoodEvent selected δ ε rep query slice) :
    QuerySufficientWithin ε (rep ω) query := by
  exact finiteSlicedWithin_implies_querySufficientWithin hGood.1 hGood.2

/-- High-probability transport for approximate random finite slices. If the
approximate good event fails with probability at most `η`, the failure of
`ε`-contextual sufficiency also has probability at most `η`. -/
theorem randomFiniteSlicedWithin_contextualSufficiency_failure_prob_le
    [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    {η : ENNReal}
    {selected : Ω → Finset Slice}
    {δ ε : ℝ}
    {rep : Ω → X → Rep}
    {query : Ctx → X → Y}
    {slice : Ω → Slice → (Ctx → Y) → SliceVal}
    (hGood :
      μ (RandomFiniteSlicedWithinGoodEvent selected δ ε rep query slice)ᶜ ≤ η) :
    μ {ω | ¬ QuerySufficientWithin ε (rep ω) query} ≤ η := by
  have hSubset :
      {ω | ¬ QuerySufficientWithin ε (rep ω) query} ⊆
        (RandomFiniteSlicedWithinGoodEvent selected δ ε rep query slice)ᶜ := by
    intro ω hBad hωGood
    exact hBad (randomFiniteSlicedWithinGoodEvent_implies_querySufficientWithin hωGood)
  exact le_trans (measure_mono hSubset) hGood

end Approximate

end FormalProofs.OPT
