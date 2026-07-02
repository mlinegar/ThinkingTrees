import FormalProofs.OPT.PreferenceScope
import FormalProbability.ML.MergeableSummaries.Agarwal2013Full

/-!
# FormalProofs/OPT/AgarwalNesting.lean

C-TreePO-facing nesting surface for Agarwal et al. 2013 mergeable summaries.

The important shape is state-level and relational: summaries merge states first,
preserve a validity relation against the represented stream, and only then read
out the downstream query at the root.  This is broader than the existing exact
`MergeablePreferenceShape`, which requires a canonical state function and
state equality after each merge.
-/

set_option linter.mathlibStandardSet false
set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

open ML.MergeableSummary
open MeasureTheory

namespace FormalProofs
namespace OPT

/-!
## Relational State-Level Preference Shape

The generic relational shape lives in `PreferenceScope.lean`, where it can be
exported next to the existing exact `MergeablePreferenceShape`.  This file adds
the Agarwal-specific instantiation from `StateLevelMergeableSummary`.
-/

/-!
## C-TreePO-to-Agarwal Adapter

To compare a C-TreePO construction to Agarwal et al.'s notation, the needed
transformations are explicit:

* choose the stream representation `Stream α` of represented data;
* fix the paper error parameter `ε`;
* expose the leaf/state builder and binary state merge;
* choose the validity relation meaning "`s` is an `S(D, ε)` summary";
* provide the state size functional and size profile `k(n, ε)`;
* expose the root readout and target preference/oracle;
* prove build validity, merge closure, readout correctness, and the size
  profile.

The following bundle names exactly that adapter.  It is intentionally
state-level: scalar child query answers are not part of the merge interface.
-/

/-- Explicit adapter from a C-TreePO state/readout construction to Agarwal's
fixed-`ε` state-level mergeable-summary notation. -/
structure CTreePOToAgarwalTransform (α State Pref : Type*) where
  ε : ℝ
  build : Stream α → State
  merge : State → State → State
  valid : Stream α → State → Prop
  readout : State → Pref
  pref : Stream α → Pref
  size : State → Nat
  profile : Agarwal2013.SizeProfile
  build_valid : ∀ xs : Stream α, valid xs (build xs)
  merge_valid : MergeClosed valid merge
  readout_valid : ∀ xs s, valid xs s → readout s = pref xs
  size_valid : Agarwal2013Full.ValidStateSizeProfile valid size profile ε

/-- Epsilon-error adapter from a C-TreePO state/readout construction to
Agarwal's fixed-`ε` notation.  This is the metric version used when validity
certifies task error at most `ε`, rather than exact readout equality. -/
structure CTreePOToAgarwalEpsilonTransform
    (α State Pref : Type*) [PseudoMetricSpace Pref] where
  ε : ℝ
  build : Stream α → State
  merge : State → State → State
  valid : Stream α → State → Prop
  readout : State → Pref
  pref : Stream α → Pref
  size : State → Nat
  profile : Agarwal2013.SizeProfile
  build_valid : ∀ xs : Stream α, valid xs (build xs)
  merge_valid : MergeClosed valid merge
  readout_valid : ∀ xs s, valid xs s → dist (readout s) (pref xs) ≤ ε
  size_valid : Agarwal2013Full.ValidStateSizeProfile valid size profile ε

namespace CTreePOToAgarwalTransform

variable {α State Pref : Type*}

/-- Forget the size profile and view the adapter as Agarwal's state-level
summary interface. -/
def toStateLevelMergeableSummary
    (H : CTreePOToAgarwalTransform α State Pref) :
    StateLevelMergeableSummary α State Pref where
  build := H.build
  merge := H.merge
  query := H.readout
  valid := H.valid
  build_valid := H.build_valid
  merge_valid := H.merge_valid

/-- View the adapter as C-TreePO's relational mergeable preference shape. -/
def toRelationalShape
    (H : CTreePOToAgarwalTransform α State Pref) :
    RelationalMergeablePreferenceShape
      H.build H.merge H.valid H.readout H.pref where
  build_valid := H.build_valid
  merge_valid := H.merge_valid
  readout_valid := H.readout_valid

/-- Leaf transformation: a C-TreePO leaf builder produces a valid
`S(D, ε)` state with the paper size-profile bound. -/
def buildValidSizedState
    (H : CTreePOToAgarwalTransform α State Pref)
    (xs : Stream α) :
    Agarwal2013Full.ValidSizedState
      H.valid H.size H.profile H.ε xs :=
  Agarwal2013Full.buildValidSizedState
    (hbuild := H.build_valid)
    (hsize := H.size_valid)
    xs

/-- Merge transformation: two valid `S(Dᵢ, ε)` states merge to a valid
`S(D₁ ++ D₂, ε)` state with the `k(|D₁|+|D₂|, ε)` size bound. -/
def mergeValidSizedState
    (H : CTreePOToAgarwalTransform α State Pref)
    {xs ys : Stream α}
    (sx :
      Agarwal2013Full.ValidSizedState
        H.valid H.size H.profile H.ε xs)
    (sy :
      Agarwal2013Full.ValidSizedState
        H.valid H.size H.profile H.ε ys) :
    Agarwal2013Full.ValidSizedState
      H.valid H.size H.profile H.ε (xs ++ ys) :=
  Agarwal2013Full.mergeValidSizedState
    (hmerge := H.merge_valid)
    (hsize := H.size_valid)
    sx sy

/-- Tree transformation: evaluating the C-TreePO state merge over any binary
merge tree gives a valid Agarwal `S(D, ε)` state for the represented union. -/
def mergeTree_validSizedState
    (H : CTreePOToAgarwalTransform α State Pref)
    (t : MergeTree α) :
    Agarwal2013Full.ValidSizedState
      H.valid H.size H.profile H.ε (MergeTree.data t) :=
  Agarwal2013Full.mergeTree_validSizedState
    (hbuild := H.build_valid)
    (hmerge := H.merge_valid)
    (hsize := H.size_valid)
    t

/-- Root transformation: merge states first, then read out the target
preference/oracle at the root. -/
theorem readout_of_mergeTree
    (H : CTreePOToAgarwalTransform α State Pref)
    (t : MergeTree α) :
    H.readout (MergeTree.eval H.build H.merge t) =
      H.pref (MergeTree.data t) :=
  H.toRelationalShape.readout_of_mergeTree t

/-- The merged root state satisfies the paper size-profile bound
`k(|D|, ε)`. -/
theorem mergeTree_size_bound
    (H : CTreePOToAgarwalTransform α State Pref)
    (t : MergeTree α) :
    (H.size (MergeTree.eval H.build H.merge t) : ℝ) ≤
      H.profile H.ε (MergeTree.data t).length :=
  (H.mergeTree_validSizedState t).size_bound

end CTreePOToAgarwalTransform

namespace CTreePOToAgarwalEpsilonTransform

variable {α State Pref : Type*} [PseudoMetricSpace Pref]

/-- Forget the size profile and view the epsilon adapter as Agarwal's
state-level summary interface. -/
def toStateLevelMergeableSummary
    (H : CTreePOToAgarwalEpsilonTransform α State Pref) :
    StateLevelMergeableSummary α State Pref where
  build := H.build
  merge := H.merge
  query := H.readout
  valid := H.valid
  build_valid := H.build_valid
  merge_valid := H.merge_valid

/-- View the adapter as C-TreePO's epsilon relational mergeable preference
shape. -/
def toEpsilonRelationalShape
    (H : CTreePOToAgarwalEpsilonTransform α State Pref) :
    EpsilonRelationalMergeablePreferenceShape
      H.build H.merge H.valid H.readout H.pref H.ε where
  build_valid := H.build_valid
  merge_valid := H.merge_valid
  readout_valid := H.readout_valid

/-- Tree transformation: evaluating the C-TreePO state merge over any binary
merge tree gives a valid Agarwal `S(D, ε)` state for the represented union. -/
def mergeTree_validSizedState
    (H : CTreePOToAgarwalEpsilonTransform α State Pref)
    (t : MergeTree α) :
    Agarwal2013Full.ValidSizedState
      H.valid H.size H.profile H.ε (MergeTree.data t) :=
  Agarwal2013Full.mergeTree_validSizedState
    (hbuild := H.build_valid)
    (hmerge := H.merge_valid)
    (hsize := H.size_valid)
    t

/-- Root transformation in the epsilon setting: merge states first, then read
out a score within the target task error `ε`. -/
theorem readout_error_of_mergeTree
    (H : CTreePOToAgarwalEpsilonTransform α State Pref)
    (t : MergeTree α) :
    dist (H.readout (MergeTree.eval H.build H.merge t))
      (H.pref (MergeTree.data t)) ≤ H.ε :=
  H.toEpsilonRelationalShape.readout_error_of_mergeTree t

/-- The merged root state satisfies the paper size-profile bound
`k(|D|, ε)`. -/
theorem mergeTree_size_bound
    (H : CTreePOToAgarwalEpsilonTransform α State Pref)
    (t : MergeTree α) :
    (H.size (MergeTree.eval H.build H.merge t) : ℝ) ≤
      H.profile H.ε (MergeTree.data t).length :=
  (H.mergeTree_validSizedState t).size_bound

end CTreePOToAgarwalEpsilonTransform

/-!
## Agarwal Summary Instantiation
-/

/-- Every state-level mergeable summary with query correctness instantiates the
relational C-TreePO preference shape. -/
theorem stateLevelMergeableSummary_relationalShape
    {α State Pref : Type*}
    (A : StateLevelMergeableSummary α State Pref)
    (oracle : Stream α → Pref)
    (h_query : StateLevelMergeableSummary.QueryCorrect A.valid oracle A.query) :
    RelationalMergeablePreferenceShape A.build A.merge A.valid A.query oracle where
  build_valid := A.build_valid
  merge_valid := A.merge_valid
  readout_valid := h_query

/-- Root-readout theorem specialized to `StateLevelMergeableSummary`. -/
theorem stateLevelMergeableSummary_readout_of_mergeTree
    {α State Pref : Type*}
    (A : StateLevelMergeableSummary α State Pref)
    (oracle : Stream α → Pref)
    (h_query : StateLevelMergeableSummary.QueryCorrect A.valid oracle A.query)
    (t : MergeTree α) :
    A.query (MergeTree.eval A.build A.merge t) = oracle (MergeTree.data t) :=
  (stateLevelMergeableSummary_relationalShape A oracle h_query).readout_of_mergeTree t

/-- Every state-level mergeable summary with epsilon query correctness
instantiates the epsilon relational C-TreePO preference shape. -/
theorem stateLevelMergeableSummary_epsilonRelationalShape
    {α State Pref : Type*} [PseudoMetricSpace Pref]
    (A : StateLevelMergeableSummary α State Pref)
    (oracle : Stream α → Pref) (ε : ℝ)
    (h_query : Agarwal2013Full.EpsilonQueryCorrect A.valid oracle A.query ε) :
    EpsilonRelationalMergeablePreferenceShape A.build A.merge A.valid A.query oracle ε where
  build_valid := A.build_valid
  merge_valid := A.merge_valid
  readout_valid := h_query

/-- Root epsilon-readout theorem specialized to `StateLevelMergeableSummary`. -/
theorem stateLevelMergeableSummary_readout_error_of_mergeTree
    {α State Pref : Type*} [PseudoMetricSpace Pref]
    (A : StateLevelMergeableSummary α State Pref)
    (oracle : Stream α → Pref) (ε : ℝ)
    (h_query : Agarwal2013Full.EpsilonQueryCorrect A.valid oracle A.query ε)
    (t : MergeTree α) :
    dist (A.query (MergeTree.eval A.build A.merge t))
      (oracle (MergeTree.data t)) ≤ ε :=
  (stateLevelMergeableSummary_epsilonRelationalShape A oracle ε h_query)
    |>.readout_error_of_mergeTree t

/-- Canonical/equality-valued state summaries recover the existing exact
`MergeablePreferenceShape`. -/
theorem stateLevelMergeableSummary_to_mergeablePreferenceShape_of_canonical
    {α State Pref : Type*}
    (A : StateLevelMergeableSummary α State Pref)
    (oracle : Stream α → Pref)
    (h_query : StateLevelMergeableSummary.QueryCorrect A.valid oracle A.query)
    (state : Stream α → State)
    (hbuild : ∀ xs : Stream α, A.build xs = state xs)
    (hmerge : ∀ xs ys : Stream α, A.merge (state xs) (state ys) = state (xs ++ ys)) :
    MergeablePreferenceShape state A.build A.merge oracle :=
  (stateLevelMergeableSummary_relationalShape A oracle h_query)
    |>.to_mergeablePreferenceShape_of_canonical state hbuild hmerge

/-!
## Randomized Agarwal Summary Instantiation

The randomized paper definition is probability over root validity for each
merge tree.  C-TreePO inherits that probability unchanged because readout
correctness is deterministic conditional on validity.
-/

/-- Any randomized Agarwal-style summary whose root state is valid with
probability at least `p` for every merge tree instantiates the randomized
relational C-TreePO preference shape. -/
theorem randomizedMergeableSummary_relationalShape
    {Ω α State Pref : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (build : Ω → Stream α → State)
    (merge : State → State → State)
    (valid : Stream α → State → Prop)
    (readout : State → Pref)
    (oracle : Stream α → Pref)
    (p : ℝ)
    (h_success :
      ∀ t : MergeTree α,
        Agarwal2013Full.RandomizedTreeSuccess μ build valid merge t p)
    (h_query : ∀ xs s, valid xs s → readout s = oracle xs) :
    RandomizedRelationalMergeablePreferenceShape
      μ build merge valid readout oracle p where
  tree_success := h_success
  readout_valid := h_query

/-- Root-readout probability specialized to randomized Agarwal-style
state-level summaries. -/
theorem randomizedMergeableSummary_readout_success_of_mergeTree
    {Ω α State Pref : Type*} [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (build : Ω → Stream α → State)
    (merge : State → State → State)
    (valid : Stream α → State → Prop)
    (readout : State → Pref)
    (oracle : Stream α → Pref)
    (p : ℝ)
    (h_success :
      ∀ t : MergeTree α,
        Agarwal2013Full.RandomizedTreeSuccess μ build valid merge t p)
    (h_query : ∀ xs s, valid xs s → readout s = oracle xs)
    (t : MergeTree α) :
    RandomizedTreeReadoutSuccess μ build merge readout oracle t p :=
  (randomizedMergeableSummary_relationalShape
    μ build merge valid readout oracle p h_success h_query)
    |>.readout_success_of_mergeTree t

/-- Any randomized Agarwal-style summary whose root state is valid with
probability at least `p` for every merge tree instantiates the randomized
epsilon relational C-TreePO preference shape. -/
theorem randomizedMergeableSummary_epsilonRelationalShape
    {Ω α State Pref : Type*} [MeasurableSpace Ω] [PseudoMetricSpace Pref]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (build : Ω → Stream α → State)
    (merge : State → State → State)
    (valid : Stream α → State → Prop)
    (readout : State → Pref)
    (oracle : Stream α → Pref)
    (ε p : ℝ)
    (h_success :
      ∀ t : MergeTree α,
        Agarwal2013Full.RandomizedTreeSuccess μ build valid merge t p)
    (h_query : Agarwal2013Full.EpsilonQueryCorrect valid oracle readout ε) :
    RandomizedEpsilonRelationalMergeablePreferenceShape
      μ build merge valid readout oracle ε p where
  tree_success := h_success
  readout_valid := h_query

/-- Root epsilon-readout probability specialized to randomized Agarwal-style
state-level summaries. -/
theorem randomizedMergeableSummary_epsilon_readout_success_of_mergeTree
    {Ω α State Pref : Type*} [MeasurableSpace Ω] [PseudoMetricSpace Pref]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (build : Ω → Stream α → State)
    (merge : State → State → State)
    (valid : Stream α → State → Prop)
    (readout : State → Pref)
    (oracle : Stream α → Pref)
    (ε p : ℝ)
    (h_success :
      ∀ t : MergeTree α,
        Agarwal2013Full.RandomizedTreeSuccess μ build valid merge t p)
    (h_query : Agarwal2013Full.EpsilonQueryCorrect valid oracle readout ε)
    (t : MergeTree α) :
    RandomizedTreeEpsilonReadoutSuccess μ build merge readout oracle t ε p :=
  (randomizedMergeableSummary_epsilonRelationalShape
    μ build merge valid readout oracle ε p h_success h_query)
    |>.readout_success_of_mergeTree t

end OPT
end FormalProofs

end
