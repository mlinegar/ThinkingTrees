import FormalProofs.OPT.GlobalAssumptions
import FormalProofs.OPT.SketchSummaryOperators
import FormalProbability.ML.MergeableSummaries.Literature

/-!
# FormalProofs/OPT/MergeableReduction.lean

Bridge lemmas from OPS assumptions to the mergeable-summary interfaces in
`FormalProbability.ML.MergeableSummaries`.

There are two distinct routes:

* the strict oracle-output route (`A3_global`) where oracle values themselves
  carry a merge operator; and
* the classical state-level route where bounded sketch states merge first and
  the query/readout is applied only at the root.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real NNReal Nat Classical Pointwise
open scoped MeasureTheory
open MeasureTheory

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

open ML.MergeableSummary

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]
variable {Sketch : Type*}

/-- Concatenate a chunk stream using the ambient monoid operation. -/
def streamConcat : List Strings → Strings
| [] => 1
| x :: xs => x * streamConcat xs

lemma streamConcat_append (xs ys : List Strings) :
    streamConcat (xs ++ ys) = streamConcat xs * streamConcat ys := by
  induction xs with
  | nil =>
      simp [streamConcat]
  | cons x xs ih =>
      simp [streamConcat, ih, mul_assoc]

/-- Deterministic OPS build adapter on chunk streams. -/
def opsBuildDet (g : Strings → Strings) : Stream Strings → Strings :=
  fun xs => g (streamConcat xs)

/-- OPS validity relation: summary preserves oracle value of chunk concatenation. -/
def opsValidDet (g : Strings → Strings) (fstar : Strings → Y) :
    Stream Strings → Strings → Prop :=
  fun xs s => D fstar s (streamConcat xs) = 0

/-- OPS merge adapter in the classical sketch form. -/
def opsMergeDet (g : Strings → Strings) : Strings → Strings → Strings :=
  fun s t => g (s * t)

/-- Valid-sketch packaging of the deterministic OPS adapter. -/
def opsValidSketchDet (g : Strings → Strings) (fstar : Strings → Y)
    (hA1 : A1_global g fstar) : ValidSketch Strings Strings where
  build := opsBuildDet g
  valid := opsValidDet g fstar
  build_valid := by
    intro xs
    simpa [opsBuildDet, opsValidDet] using hA1 (streamConcat xs)

/-- A1/A2/A3 imply closure of the OPS adapter under summary merge.

This is the strict oracle-output route: `A3_global` supplies a merge on oracle
values, which is stronger than the usual state-level mergeable-sketch condition. -/
theorem ops_mergeClosed_of_global (g : Strings → Strings) (fstar : Strings → Y)
    (hA1 : A1_global g fstar) (hA2 : A2_global g fstar) (hA3 : A3_global g fstar) :
    MergeClosed (opsValidDet g fstar) (opsMergeDet g) := by
  intro xs ys sx sy hsx hsy
  unfold opsValidDet at hsx hsy ⊢
  unfold opsMergeDet
  have h_concat :
      D fstar (sx * sy) (streamConcat xs * streamConcat ys) = 0 :=
    concat_congruence g fstar hA1 hA2 hA3
      sx (streamConcat xs) sy (streamConcat ys) hsx hsy
  have hA1_merge : D fstar (g (sx * sy)) (sx * sy) = 0 := hA1 (sx * sy)
  have h_target : D fstar (sx * sy) (streamConcat (xs ++ ys)) = 0 := by
    simpa [streamConcat_append] using h_concat
  apply le_antisymm _ dist_nonneg
  calc
    D fstar (g (sx * sy)) (streamConcat (xs ++ ys))
        ≤ D fstar (g (sx * sy)) (sx * sy) + D fstar (sx * sy) (streamConcat (xs ++ ys)) :=
          D_triangle fstar (g (sx * sy)) (sx * sy) (streamConcat (xs ++ ys))
    _ = 0 + 0 := by rw [hA1_merge, h_target]
    _ = 0 := by ring

/-- Deterministic OPS with A1/A2/A3 is hierarchically mergeable over merge trees
in the strict oracle-output sense. -/
theorem ops_hierarchical_mergeable_of_global (g : Strings → Strings) (fstar : Strings → Y)
    (hA1 : A1_global g fstar) (hA2 : A2_global g fstar) (hA3 : A3_global g fstar) :
    HierarchicalMergeable (opsBuildDet g) (opsValidDet g fstar) (opsMergeDet g) := by
  simpa [opsValidSketchDet] using
    (hierarchical_of_full
      (V := opsValidSketchDet g fstar hA1)
      (merge := opsMergeDet g)
      (ops_mergeClosed_of_global g fstar hA1 hA2 hA3))

/-- Strict oracle-homomorphism reduction: when oracle values themselves have the
`A3_global` merge, deterministic OPS reduces to the oracle-level mergeable
summary interface.  This is a special case, not the full classical
state-level sketch condition. -/
theorem ops_reduction_to_classical_mergeable (g : Strings → Strings) (fstar : Strings → Y)
    (hA1 : A1_global g fstar) (hA2 : A2_global g fstar) (hA3 : A3_global g fstar) :
    IsMergeableSummary g fstar :=
  prop3_mergeable_classical g fstar hA1 hA2 hA3

/-- Build a sketch state for a stream by encoding the concatenated stream. -/
def sketchBuildStream (op : SketchOperator Strings Sketch) : Stream Strings → Sketch :=
  fun xs => op.encode (streamConcat xs)

/-- State-level validity for a sketch: the decoded state preserves the oracle
value of the concatenated stream. -/
def sketchStateValid (op : SketchOperator Strings Sketch) (fstar : Strings → Y) :
    Stream Strings → Sketch → Prop :=
  fun xs s => D fstar (op.decode s) (streamConcat xs) = 0

/-- Query/readout associated with decoded sketch states. -/
def sketchStateReadout (op : SketchOperator Strings Sketch) (fstar : Strings → Y) :
    Sketch → Y :=
  fun s => fstar (op.decode s)

/-- A sketch operator whose leaf decode preserves the oracle gives a valid
state-level sketch system over streams. -/
def sketchValidSketchOfLeaf
    (op : SketchOperator Strings Sketch) (fstar : Strings → Y)
    (h_leaf : SketchLeafPreserving op fstar) : ValidSketch Strings Sketch where
  build := sketchBuildStream op
  valid := sketchStateValid op fstar
  build_valid := by
    intro xs
    simpa [sketchBuildStream, sketchStateValid, SketchLeafPreserving,
      PointwisePreserving, summaryFromSketch, D] using h_leaf (streamConcat xs)

/-- Sketch merge compatibility is exactly merge closure for the state-level
validity relation.  No oracle-value merge `Y → Y → Y` is required. -/
theorem sketch_state_mergeClosed_of_compatible
    (op : SketchOperator Strings Sketch) (fstar : Strings → Y)
    (h_merge : SketchMergeCompatible op fstar) :
    MergeClosed (sketchStateValid op fstar) op.merge := by
  intro xs ys sx sy hsx hsy
  unfold sketchStateValid at hsx hsy ⊢
  have h :=
    h_merge sx sy (streamConcat xs) (streamConcat ys) hsx hsy
  simpa [streamConcat_append] using h

/-- Classical state-level reduction: a sketch operator with leaf preservation and
merge compatibility is hierarchically mergeable as a bounded state system.  The
readout is applied to the final state, so scalar oracle values need not compose. -/
theorem sketch_state_level_reduction_to_classical_mergeable
    (op : SketchOperator Strings Sketch) (fstar : Strings → Y)
    (h_leaf : SketchLeafPreserving op fstar)
    (h_merge : SketchMergeCompatible op fstar) :
    HierarchicalMergeable
      (sketchBuildStream op) (sketchStateValid op fstar) op.merge := by
  exact hierarchical_of_full
    (V := sketchValidSketchOfLeaf op fstar h_leaf)
    (merge := op.merge)
    (sketch_state_mergeClosed_of_compatible op fstar h_merge)

/-- Under commutative oracle semantics, OPS merge is commutative at oracle level. -/
theorem ops_merge_commutative_oracle (g : Strings → Strings) (fstar : Strings → Y)
    (hA1 : A1_global g fstar) (h_comm : OracleCommutative fstar) :
    ∀ a b, D fstar (opsMergeDet g a b) (opsMergeDet g b a) = 0 := by
  intro a b
  simpa [opsMergeDet, summaryMerge] using mergeable_commutative g fstar hA1 h_comm a b

/-!
## Literature-Facing Formalization Aliases

These aliases make the imported `FormalProbability.ML.MergeableSummaries`
interfaces visible from the C-TreePO Lean surface.  They are intentionally thin:
the proofs live in the reusable FormalProbability dependency, while this file
records exactly which theorem names back the C-TreePO mergeable-sketch story.
-/

section LiteratureFormalizationAliases

/-- Agarwal et al. state-level mergeable summaries: merge states first, then
query/read out at the root. -/
theorem ctreepo_agarwal2013_state_level_hierarchical_readout {α S Q : Type*}
    (A : StateLevelMergeableSummary α S Q)
    (oracle : Stream α → Q)
    (h_query : StateLevelMergeableSummary.QueryCorrect A.valid oracle A.query)
    (t : MergeTree α) :
    A.query (MergeTree.eval A.build A.merge t) = oracle (MergeTree.data t) :=
  StateLevelMergeableSummary.agarwal2013_state_level_hierarchical_readout
    A oracle h_query t

/-- Agarwal et al. full mergeability implies one-way mergeability by building a
summary for the absorbed raw stream. -/
theorem ctreepo_agarwal2013_full_implies_one_way_with_build {α S : Type*}
    (V : ValidSketch α S) (merge : S → S → S)
    (hfull : MergeClosed V.valid merge) :
    OneWayMergeable V (mergeIntoWithBuild V merge) :=
  Agarwal2013.full_implies_one_way_with_build V merge hfull

/-- Agarwal et al. Theorem 2: incrementally maintainable summaries are one-way
mergeable. -/
theorem ctreepo_agarwal2013_incrementally_maintainable_one_way
    {α S : Type*} {V : ValidSketch α S}
    (I : Agarwal2013.IncrementallyMaintainable V) :
    OneWayMergeable V I.mergeInto :=
  I.oneWayMergeable

/-- Agarwal et al. linear-sketch example: additive linear sketches are fully
mergeable. -/
theorem ctreepo_agarwal2013_linearSketch_fullMergeable {α β : Type*}
    [AddCommMonoid β] (phi : α → β) :
    FullMergeable (linearMergeableSketch (α := α) (β := β) phi) :=
  Agarwal2013.linearSketch_fullMergeable phi

/-- Agarwal et al. linear-sketch example made concrete: Count-Min-style
additive counter tables are state-level mergeable. -/
theorem ctreepo_agarwal2013_countMin_state_level_mergeable
    {α Q : Type*} {d w : Nat}
    (bucket : α → Fin d → Fin w)
    (query : CountMin.Table d w → Q) :
    HierarchicalMergeable
      (CountMin.Table.build bucket)
      (CountMin.Table.valid bucket)
      (@CountMin.Table.merge d w) :=
  Agarwal2013.countMin_state_level_mergeable bucket query

/-- Agarwal et al. linear-sketch example made concrete: Count-Min-style
additive table merge is not idempotent on nonempty tables. -/
theorem ctreepo_agarwal2013_countMin_merge_not_idempotent_of_pos {d w : Nat}
    (hd : 0 < d) (hw : 0 < w) :
    ¬ MergeIdempotent (@CountMin.Table.merge d w) :=
  Agarwal2013.countMin_merge_not_idempotent_of_pos hd hw

/-- Agarwal et al. Theorem 1 surface: MG algorithm bundles are hierarchically
mergeable. -/
theorem ctreepo_agarwal2013_misraGries_hierarchical
    {α : Type*} [DecidableEq α]
    (mg : HeavyHitters.MGAlgorithm α) :
    HierarchicalMergeable mg.build mg.valid mg.merge :=
  Agarwal2013.misraGries_hierarchical mg

/-- C-TreePO-facing placement theorem: MG algorithm bundles are generic sized
mergeable query sketches, with query readout after state merging. -/
theorem ctreepo_agarwal2013_misraGries_subset_sizedMergeableQuerySketch
    {α : Type*} [DecidableEq α]
    (mg : HeavyHitters.MGAlgorithm α) :
    MergeClosed
      (Agarwal2013.misraGries_toSizedMergeableQuerySketch mg).valid
      (Agarwal2013.misraGries_toSizedMergeableQuerySketch mg).merge ∧
    (Agarwal2013.misraGries_toSizedMergeableQuerySketch mg).query = mg.query :=
  Agarwal2013.misraGries_subset_sizedMergeableQuerySketch mg

/-- Agarwal et al. executable MG core: the concrete counter table build never
exceeds its configured capacity. -/
theorem ctreepo_agarwal2013_executableMisraGries_boundedBy
    {α : Type*} [DecidableEq α]
    (k : Nat) (xs : Stream α) :
    HeavyHitters.boundedBy k (HeavyHitters.MisraGries.build k xs) :=
  Agarwal2013.executableMisraGries_boundedBy k xs

/-- Agarwal et al. executable MG bookkeeping: one update increases total stored
counter mass by at most one. -/
theorem ctreepo_agarwal2013_executableMisraGries_update_totalCounterMass_le_succ
    {α : Type*} [DecidableEq α]
    (k : Nat) (x : α) (s : HeavyHitters.MGSummary α) :
    HeavyHitters.totalCounterMass
        (HeavyHitters.MisraGries.update k x s) ≤
      HeavyHitters.totalCounterMass s + 1 :=
  Agarwal2013.executableMisraGries_update_totalCounterMass_le_succ k x s

/-- Agarwal et al. executable MG bookkeeping: total stored counter mass is at
most the processed stream length. -/
theorem ctreepo_agarwal2013_executableMisraGries_totalCounterMass_le_length
    {α : Type*} [DecidableEq α]
    (k : Nat) (xs : Stream α) :
    HeavyHitters.totalCounterMass
        (HeavyHitters.MisraGries.build k xs) ≤ xs.length :=
  Agarwal2013.executableMisraGries_totalCounterMass_le_length k xs

/-- Agarwal et al. executable MG invariant: stored counters remain strictly
positive. -/
theorem ctreepo_agarwal2013_executableMisraGries_positiveCounts
    {α : Type*} [DecidableEq α]
    (k : Nat) (xs : Stream α) :
    HeavyHitters.positiveCounts (HeavyHitters.MisraGries.build k xs) :=
  Agarwal2013.executableMisraGries_positiveCounts k xs

/-- Agarwal et al. executable MG frequency-error induction core: the traced
potential is bounded by processed length. -/
theorem ctreepo_agarwal2013_executableMisraGries_tracedPotential_le_length
    {α : Type*} [DecidableEq α]
    (k : Nat) (xs : Stream α) :
    HeavyHitters.MisraGries.tracedPotential k
        (HeavyHitters.MisraGries.tracedBuild k xs) ≤ xs.length :=
  Agarwal2013.executableMisraGries_tracedPotential_le_length k xs

/-- Agarwal et al. executable MG frequency-error induction core: global
decrement/prune steps are charged to blocks of `k+1` processed items. -/
theorem ctreepo_agarwal2013_executableMisraGries_debt_mul_succ_le_length
    {α : Type*} [DecidableEq α]
    (k : Nat) (xs : Stream α) :
    (k + 1) *
        (HeavyHitters.MisraGries.tracedBuild k xs).debt ≤ xs.length :=
  Agarwal2013.executableMisraGries_debt_mul_succ_le_length k xs

/-- C-TreePO-facing alias for Agarwal et al. Lemma 2.1: executable MG
estimates are lower bounds and the undercount is controlled by decrement debt. -/
theorem ctreepo_agarwal2013_executableMisraGries_lemma21_frequency_envelope
    {α : Type*} [DecidableEq α]
    (k : Nat) (xs : Stream α) (a : α) :
    HeavyHitters.estimateCount
        (HeavyHitters.MisraGries.build k xs) a ≤ frequency xs a ∧
    frequency xs a ≤
        HeavyHitters.estimateCount
          (HeavyHitters.MisraGries.build k xs) a +
          (HeavyHitters.MisraGries.tracedBuild k xs).debt ∧
    (k + 1) *
        (HeavyHitters.MisraGries.tracedBuild k xs).debt ≤ xs.length :=
  Agarwal2013.executableMisraGries_lemma21_frequency_envelope k xs a

/-- C-TreePO-facing alias for executable MG's real-valued Lemma 2.1 error
bound. -/
theorem ctreepo_agarwal2013_executableMisraGries_lemma21_real_error_bound
    {α : Type*} [DecidableEq α]
    (k : Nat) (xs : Stream α) (a : α) :
    |((HeavyHitters.estimateCount
          (HeavyHitters.MisraGries.build k xs) a : ℝ) -
        (frequency xs a : ℝ))| ≤
      (xs.length : ℝ) / (((k + 1 : Nat) : ℝ)) :=
  Agarwal2013.executableMisraGries_lemma21_real_error_bound k xs a

/-- C-TreePO-facing alias for executable MG's frequency-error interface with
`ε = 1/(k+1)`. -/
theorem ctreepo_agarwal2013_executableMisraGries_frequencyError_inv_succ
    {α : Type*} [DecidableEq α] (k : Nat) :
    FrequencyErrorGuaranteeFn
      (HeavyHitters.MisraGries.build (α := α) k)
      HeavyHitters.MisraGries.query
      (1 / (((k + 1 : Nat) : ℝ))) :=
  Agarwal2013.executableMisraGries_frequencyError_inv_succ k

/-- C-TreePO-facing alias for the MG combine/prune proof core used by
`MERGEABLEMINERROR` and `MERGEABLEMINSPACE`. -/
theorem ctreepo_agarwal2013_misraGries_lemma21Envelope_mergeOfPruneCertificate
    {α : Type*} [DecidableEq α]
    {k n₁ n₂ mass₁ mass₂ massAfter threshold : Nat}
    {estimate₁ estimate₂ true₁ true₂ after : α → Nat}
    (h₁ : HeavyHitters.MisraGries.Lemma21Envelope
      k n₁ mass₁ estimate₁ true₁)
    (h₂ : HeavyHitters.MisraGries.Lemma21Envelope
      k n₂ mass₂ estimate₂ true₂)
    (C : HeavyHitters.MisraGries.PruneCertificate
      k (mass₁ + mass₂) massAfter threshold
      (fun a => estimate₁ a + estimate₂ a) after) :
    HeavyHitters.MisraGries.Lemma21Envelope
      k (n₁ + n₂) massAfter after
      (fun a => true₁ a + true₂ a) :=
  Agarwal2013.misraGries_lemma21Envelope_mergeOfPruneCertificate h₁ h₂ C

/-- Agarwal et al. Corollary 1: SpaceSaving is hierarchically mergeable once an
MG/SpaceSaving isomorphism witness is available. -/
theorem ctreepo_agarwal2013_spaceSaving_hierarchical_of_isomorphism
    {α : Type*} [DecidableEq α]
    (mg : HeavyHitters.MGAlgorithm α)
    (ss : HeavyHitters.SpaceSavingAlgorithm α)
    (hiso : HeavyHitters.IsomorphicMGSpaceSaving mg ss) :
    HierarchicalMergeable ss.build ss.valid ss.merge :=
  Agarwal2013.spaceSaving_hierarchical_of_isomorphism mg ss hiso

/-- C-TreePO-facing placement theorem: SpaceSaving bundles are generic sized
mergeable query sketches once packaged with their validity witnesses. -/
theorem ctreepo_agarwal2013_spaceSaving_subset_sizedMergeableQuerySketch
    {α : Type*} [DecidableEq α]
    (ss : HeavyHitters.SpaceSavingAlgorithm α) :
    MergeClosed
      (Agarwal2013.spaceSaving_toSizedMergeableQuerySketch ss).valid
      (Agarwal2013.spaceSaving_toSizedMergeableQuerySketch ss).merge ∧
    (Agarwal2013.spaceSaving_toSizedMergeableQuerySketch ss).query = ss.query :=
  Agarwal2013.spaceSaving_subset_sizedMergeableQuerySketch ss

/-- Agarwal et al. executable SpaceSaving bookkeeping: the concrete table
never exceeds capacity. -/
theorem ctreepo_agarwal2013_executableSpaceSaving_boundedBy
    {α : Type*} [DecidableEq α]
    (k : Nat) (xs : Stream α) :
    HeavyHitters.boundedBy k (HeavyHitters.SpaceSaving.build k xs) :=
  Agarwal2013.executableSpaceSaving_boundedBy k xs

/-- Agarwal et al. executable SpaceSaving bookkeeping: one update increases
stored mass by at most one. -/
theorem ctreepo_agarwal2013_executableSpaceSaving_update_totalCounterMass_le_succ
    {α : Type*} [DecidableEq α]
    (k : Nat) (x : α) (s : HeavyHitters.SpaceSavingSummary α) :
    HeavyHitters.totalCounterMass
        (HeavyHitters.SpaceSaving.update k x s) ≤
      HeavyHitters.totalCounterMass s + 1 :=
  Agarwal2013.executableSpaceSaving_update_totalCounterMass_le_succ k x s

/-- Agarwal et al. executable SpaceSaving bookkeeping: total stored mass is at
most the processed stream length. -/
theorem ctreepo_agarwal2013_executableSpaceSaving_totalCounterMass_le_length
    {α : Type*} [DecidableEq α]
    (k : Nat) (xs : Stream α) :
    HeavyHitters.totalCounterMass
        (HeavyHitters.SpaceSaving.build k xs) ≤ xs.length :=
  Agarwal2013.executableSpaceSaving_totalCounterMass_le_length k xs

/-- Agarwal et al. Corollary 2: GK-style quantile summaries are one-way
mergeable when packaged as a GK algorithm bundle. -/
theorem ctreepo_agarwal2013_gk_corollary2_oneWay {α : Type*}
    [Preorder α] [DecidableRel ((· ≤ ·) : α → α → Prop)]
    (A : GK.Algorithm α) :
    GK.corollary2_statement A :=
  Agarwal2013.gk_corollary2_oneWay A

/-- C-TreePO-facing placement theorem: GK quantile summaries are one-way
sized query sketches, not full state-state mergeable sketches. -/
theorem ctreepo_agarwal2013_gk_subset_oneWaySizedQuerySketch {α : Type*}
    [Preorder α] [DecidableRel ((· ≤ ·) : α → α → Prop)]
    (A : GK.Algorithm α) :
    OneWayMergeable
      (Agarwal2013.gk_toOneWaySizedQuerySketch A).toValidSketch
      (Agarwal2013.gk_toOneWaySizedQuerySketch A).mergeInto ∧
    (Agarwal2013.gk_toOneWaySizedQuerySketch A).query = A.queryRank :=
  Agarwal2013.gk_subset_oneWaySizedQuerySketch A

/-- Agarwal et al. Theorem 3.2 CFF triangle step for one-way quantile merging. -/
theorem ctreepo_agarwal2013_cumulativeError_oneWay_merge_bound
    {α : Type*} {ε₁ ε₂ : ℝ} {n₁ n₂ : Nat}
    {F₁ F₂ Fhat₂ Fout : α → ℝ}
    (hmain :
      Agarwal2013.CumulativeErrorBound
        (ε₁ * (((n₁ + n₂ : Nat) : ℝ)))
        Fout (Agarwal2013.pointwiseAddFn F₁ Fhat₂))
    (hbatch :
      Agarwal2013.CumulativeErrorBound
        (ε₂ * (n₂ : ℝ)) Fhat₂ F₂) :
    Agarwal2013.CumulativeErrorBound
      (ε₁ * (((n₁ + n₂ : Nat) : ℝ)) + ε₂ * (n₂ : ℝ))
      Fout (Agarwal2013.pointwiseAddFn F₁ F₂) :=
  LiteratureChronology.agarwal2013_12c_cumulativeError_oneWay_merge_bound
    hmain hbatch

/-- Agarwal et al. Theorem 3.2 budget arithmetic for the common half-ε
rescaling. -/
theorem ctreepo_agarwal2013_cumulativeError_oneWay_twoMerge_half_budget
    {ε ε₁ ε₂ : ℝ} {n₁ n₂ : Nat}
    (hε : 0 ≤ ε)
    (h₁ : ε₁ ≤ ε / 2)
    (h₂ : ε₂ ≤ ε / 2) :
    ε₁ * (((n₁ + n₂ : Nat) : ℝ)) + ε₂ * (n₂ : ℝ) ≤
      ε * (((n₁ + n₂ : Nat) : ℝ)) :=
  LiteratureChronology.agarwal2013_12d_cumulativeError_oneWay_twoMerge_half_budget
    hε h₁ h₂

/-- Agarwal et al. Theorem 3.2 CFF/rank one-way merge bound under half-ε
rescaling. -/
theorem ctreepo_agarwal2013_cumulativeError_oneWay_merge_half_epsilon
    {α : Type*} {ε ε₁ ε₂ : ℝ} {n₁ n₂ : Nat}
    {F₁ F₂ Fhat₂ Fout : α → ℝ}
    (hε : 0 ≤ ε)
    (h₁ : ε₁ ≤ ε / 2)
    (h₂ : ε₂ ≤ ε / 2)
    (hmain :
      Agarwal2013.CumulativeErrorBound
        (ε₁ * (((n₁ + n₂ : Nat) : ℝ)))
        Fout (Agarwal2013.pointwiseAddFn F₁ Fhat₂))
    (hbatch :
      Agarwal2013.CumulativeErrorBound
        (ε₂ * (n₂ : ℝ)) Fhat₂ F₂) :
    Agarwal2013.CumulativeErrorBound
      (ε * (((n₁ + n₂ : Nat) : ℝ)))
      Fout (Agarwal2013.pointwiseAddFn F₁ F₂) :=
  LiteratureChronology.agarwal2013_12e_cumulativeError_oneWay_merge_half_epsilon
    hε h₁ h₂ hmain hbatch

/-- Agarwal et al. Theorem 5 citation schema: randomized fully mergeable
quantile summaries. -/
def ctreepo_agarwal2013_randomizedQuantileFullyMergeable
    (Ω α : Type*) [Preorder α] [DecidableRel ((· ≤ ·) : α → α → Prop)]
    [MeasurableSpace Ω]
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    (ε δ : ℝ) : Prop :=
  Agarwal2013.theorem5_randomizedQuantileFullyMergeable Ω α μ ε δ

/-- Agarwal et al. Theorem 6 citation schema: mergeable ε-approximations for
range spaces. -/
def ctreepo_agarwal2013_rangeSpaceEpsilonApproximation
    (d : Nat) (ε : ℝ) : Prop :=
  Agarwal2013.theorem6_rangeSpaceEpsilonApproximation d ε

/-- Agarwal et al. finite range-space trace object backed by mathlib's
`Finset.Shatters`/`Finset.vcDim` API. -/
abbrev ctreepo_agarwal2013_FiniteRangeSpace
    (α ρ : Type*) [DecidableEq α] :=
  Agarwal2013.FiniteRangeSpace α ρ

/-- Agarwal et al. mathlib-backed VC bound: any finite set shattered by a
finite range trace has size at most the trace VC dimension. -/
theorem ctreepo_agarwal2013_finiteRangeSpace_shattered_card_le_vcDim
    {α ρ : Type*} [DecidableEq α]
    (F : Agarwal2013.FiniteRangeSpace α ρ) {s : Finset α}
    (hs : F.Shatters s) :
    s.card ≤ F.vcDim :=
  LiteratureChronology.agarwal2013_16a1_finiteRangeSpace_shattered_card_le_vcDim
    F hs

/-- Agarwal et al. mathlib-backed Sauer-Shelah trace growth bound for finite
range spaces. -/
theorem ctreepo_agarwal2013_finiteRangeSpace_trace_card_le_sauerShelah
    {α ρ : Type*} [DecidableEq α] [Fintype α]
    (F : Agarwal2013.FiniteRangeSpace α ρ) :
    F.trace.card ≤
      ∑ k ∈ Finset.Iic F.vcDim, (Fintype.card α).choose k :=
  LiteratureChronology.agarwal2013_16a2_finiteRangeSpace_trace_card_le_sauerShelah
    F

/-- Agarwal et al. finite ε-approximation scaffold: uniform per-trace tails
compose by a finite union bound over the trace family. -/
theorem ctreepo_agarwal2013_finiteRangeSpace_traceFailureEvent_le_card_mul
    {Ω α ρ : Type*} [MeasurableSpace Ω] [DecidableEq α]
    {μ : Measure Ω} [IsFiniteMeasure μ]
    (F : Agarwal2013.FiniteRangeSpace α ρ)
    (bad : Ω → Finset α → Prop) (δ : ℝ)
    (hbad : ∀ T : F.trace, μ.real {ω : Ω | bad ω T.1} ≤ δ) :
    μ.real (F.traceFailureEvent bad) ≤ (F.trace.card : ℝ) * δ :=
  LiteratureChronology.agarwal2013_16a4_finiteRangeSpace_traceFailureEvent_le_card_mul
    F bad δ hbad

/-- Agarwal et al. finite ε-approximation scaffold: Sauer-Shelah converts the
finite union bound into a VC-growth-controlled uniform failure bound. -/
theorem ctreepo_agarwal2013_finiteRangeSpace_traceFailureEvent_le_sauerShelah_mul
    {Ω α ρ : Type*} [MeasurableSpace Ω] [DecidableEq α] [Fintype α]
    {μ : Measure Ω} [IsFiniteMeasure μ]
    (F : Agarwal2013.FiniteRangeSpace α ρ)
    (bad : Ω → Finset α → Prop) (δ : ℝ)
    (hδ : 0 ≤ δ)
    (hbad : ∀ T : F.trace, μ.real {ω : Ω | bad ω T.1} ≤ δ) :
    μ.real (F.traceFailureEvent bad) ≤
      ((∑ k ∈ Finset.Iic F.vcDim,
          (Fintype.card α).choose k : Nat) : ℝ) * δ :=
  LiteratureChronology.agarwal2013_16a5_finiteRangeSpace_traceFailureEvent_le_sauerShelah_mul
    F bad δ hδ hbad

/-- Agarwal et al. Section 4 finite-buffer range count used by the
low-discrepancy merge step. -/
def ctreepo_agarwal2013_finiteRangeSpace_rangeCountOn
    {α ρ : Type*} [DecidableEq α]
    (F : Agarwal2013.FiniteRangeSpace α ρ)
    (s : Finset α) (R : ρ) : Nat :=
  LiteratureChronology.agarwal2013_16a6_finiteRangeSpace_rangeCountOn F s R

/-- Agarwal et al. Section 4 Lemma 4.1: a low-discrepancy coloring merge is
unbiased for every range query. -/
theorem ctreepo_agarwal2013_finiteRangeSpace_coloredRangeEstimate_unbiased
    {α ρ : Type*} [DecidableEq α]
    (F : Agarwal2013.FiniteRangeSpace α ρ)
    (s : Finset α) (χ : α → Bool) (R : ρ) :
    Agarwal2013.twoPointMean (fun keepPositive =>
      F.coloredRangeEstimate s χ keepPositive R) =
        (F.rangeCountOn s R : ℝ) :=
  LiteratureChronology.agarwal2013_16a7_finiteRangeSpace_coloredRangeEstimate_unbiased
    F s χ R

/-- Agarwal et al. Section 4 Lemma 4.1: a low-discrepancy coloring merge has
absolute range error at most the coloring discrepancy budget. -/
theorem ctreepo_agarwal2013_finiteRangeSpace_lowDiscrepancy_coloredRangeError_abs_le
    {α ρ : Type*} [DecidableEq α]
    (F : Agarwal2013.FiniteRangeSpace α ρ)
    (s : Finset α) (χ : α → Bool) {Δ : ℝ}
    (hχ : F.LowDiscrepancyColoring s χ Δ)
    (keepPositive : Bool) (R : ρ) :
    |F.coloredRangeError s χ keepPositive R| ≤ Δ :=
  LiteratureChronology.agarwal2013_16a8_finiteRangeSpace_lowDiscrepancy_coloredRangeError_abs_le
    F s χ hχ keepPositive R

/-- Agarwal et al. Section 4 bundled low-discrepancy merge certificate. -/
abbrev ctreepo_agarwal2013_LowDiscrepancyMergeCertificate
    (α ρ : Type*) [DecidableEq α] :=
  LiteratureChronology.agarwal2013_16a9_LowDiscrepancyMergeCertificate
    α ρ

/-- Agarwal et al. Lemma 4.3 setup: level-scaled range-space over-count has
radius `2^level * Δ` from a one-step discrepancy budget `Δ`. -/
theorem ctreepo_agarwal2013_rangeSpaceColoring_level_error_abs_le
    (level : Nat) {Δ baseError : ℝ}
    (hbase : |baseError| ≤ Δ) :
    |Agarwal2013.rangeSpaceColoringLevelError level baseError|
      ≤ (2 ^ level : ℝ) * Δ :=
  LiteratureChronology.agarwal2013_16a10_rangeSpaceColoring_level_error_abs_le
    level hbase

/-- Agarwal et al. Section 4 complete-tree low-discrepancy coloring process. -/
abbrev ctreepo_agarwal2013_RangeSpaceColoringCompleteTreeProcess :=
  LiteratureChronology.agarwal2013_16a11_RangeSpaceColoringCompleteTreeProcess

/-- Agarwal et al. Lemma 4.3 stochastic step: complete-tree low-discrepancy
range-space halving has a two-sided Azuma/Hoeffding tail under explicit
martingale hypotheses. -/
theorem ctreepo_agarwal2013_rangeSpaceColoring_completeTree_hoeffding_tail
    {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω}
    {m : Nat} {Δ : ℝ}
    [StandardBorelSpace Ω] [IsZeroOrProbabilityMeasure μ]
    (P : Agarwal2013.RangeSpaceColoringCompleteTreeProcess Ω mΩ μ m Δ)
    {η : ℝ} (hη : 0 ≤ η) :
    μ.real {ω | η ≤ |P.totalError ω|}
      ≤ 2 * Real.exp
        (-η ^ 2 /
          (2 * ((Finset.sum (Finset.range P.horizon) (fun i : Nat =>
            if i = 0 then (0 : ℝ≥0)
            else
              (‖Agarwal2013.rangeSpaceColoringAzumaUpper Δ i -
                  Agarwal2013.rangeSpaceColoringAzumaLower Δ i‖₊ / 2) ^ 2) :
            ℝ≥0) : ℝ))) :=
  LiteratureChronology.agarwal2013_16a12_rangeSpaceColoring_completeTree_hoeffding_tail
    P hη

/-- Agarwal et al. exact geometric range-count query. -/
def ctreepo_agarwal2013_geometricRangeCount {d : Nat}
    (xs : Stream (Point d)) (R : Set (Point d)) : Nat :=
  LiteratureChronology.agarwal2013_16b_geometricRangeCount xs R

/-- Agarwal et al. exact all-points range-space ε-approximation witness. -/
def ctreepo_agarwal2013_exactRangeSpaceEpsilonApproximationSpec
    (d : Nat) (ε : ℝ) (hε : 0 ≤ ε) :
    Agarwal2013.StateLevelEpsilonApproximationSpec
      (Point d) (Set (Point d)) (Stream (Point d)) :=
  LiteratureChronology.agarwal2013_16c_exactRangeSpaceEpsilonApproximationSpec
    d ε hε

/-- Agarwal et al. exact all-points range-space witness as a generic
mergeable query sketch. -/
def ctreepo_agarwal2013_exactRangeSpaceSizedMergeableQuerySketch
    (d : Nat) (ε : ℝ) (hε : 0 ≤ ε) :
    SizedMergeableQuerySketch
      (Point d) (Stream (Point d)) (RangeCountQuery d) :=
  LiteratureChronology.agarwal2013_16d_exactRangeSpaceSizedMergeableQuerySketch
    d ε hε

/-- Agarwal et al. Theorem 7 citation schema: mergeable ε-kernels in a common
reference frame. -/
def ctreepo_agarwal2013_epsilonKernelCommonReferenceFrame
    (d : Nat) (ε : ℝ) : Prop :=
  Agarwal2013.theorem7_epsilonKernelCommonReferenceFrame d ε

/-- Agarwal et al. finite-dimensional projection used by ε-kernel width
queries. -/
def ctreepo_agarwal2013_pointDot {d : Nat}
    (x direction : Point d) : ℝ :=
  LiteratureChronology.agarwal2013_17a_pointDot x direction

/-- Agarwal et al. streamwise translation used by ε-kernel width queries. -/
def ctreepo_agarwal2013_translateStream {d : Nat}
    (offset : Point d) (xs : Stream (Point d)) : Stream (Point d) :=
  LiteratureChronology.agarwal2013_17a2_translateStream offset xs

/-- Agarwal et al. projection algebra: translation adds the offset projection
inside the dot product. -/
theorem ctreepo_agarwal2013_pointDot_translatePoint {d : Nat}
    (offset x direction : Point d) :
    Agarwal2013.pointDot (Agarwal2013.translatePoint offset x) direction =
      Agarwal2013.pointDot x direction +
        Agarwal2013.pointDot offset direction :=
  LiteratureChronology.agarwal2013_17a3_pointDot_translatePoint
    offset x direction

/-- Agarwal et al. typed target for same-weight interval ε-approximations. -/
abbrev ctreepo_agarwal2013_SameWeightIntervalApproximationSpec :=
  Agarwal2013.SameWeightIntervalApproximationSpec

/-- Agarwal et al. exact interval counts add over concatenation. -/
theorem ctreepo_agarwal2013_intervalCount_append
    (xs ys : Stream ℝ) (I : Agarwal2013.Interval1D) :
    Agarwal2013.intervalCount (xs ++ ys) I =
      Agarwal2013.intervalCount xs I +
        Agarwal2013.intervalCount ys I :=
  LiteratureChronology.agarwal2013_13c_intervalCount_append xs ys I

/-- Agarwal et al. state-level ε-approximation interface. -/
abbrev ctreepo_agarwal2013_StateLevelEpsilonApproximationSpec :=
  Agarwal2013.StateLevelEpsilonApproximationSpec

/-- Agarwal et al. state-level ε-approximations as generic mergeable query
sketches. -/
def ctreepo_agarwal2013_stateLevelEpsilonApproximation_toSizedMergeableQuerySketch
    {α ρ S : Type*}
    (A : Agarwal2013.StateLevelEpsilonApproximationSpec α ρ S) :
    SizedMergeableQuerySketch α S (ρ → Nat) :=
  LiteratureChronology.agarwal2013_13d1_stateLevelEpsilonApproximation_toSizedMergeableQuerySketch A

/-- Agarwal et al. state-level ε-approximation error after a merge tree. -/
theorem ctreepo_agarwal2013_stateLevelEpsilonApproximation_tree_error
    {α ρ S : Type*}
    (A : Agarwal2013.StateLevelEpsilonApproximationSpec α ρ S)
    (t : MergeTree α) (R : ρ) :
    |((A.query (MergeTree.eval A.build A.merge t) R : ℝ) -
        (A.rangeCount (MergeTree.data t) R : ℝ))|
      ≤ A.ε * ((MergeTree.data t).length : ℝ) :=
  LiteratureChronology.agarwal2013_13e_stateLevelEpsilonApproximation_tree_error
    A t R

/-- Agarwal et al. exact all-points ε-approximation witness for the
state-level error/merge interface. -/
def ctreepo_agarwal2013_exactStateLevelEpsilonApproximationSpec
    {α ρ : Type*}
    (rangeCount : Stream α → ρ → Nat)
    (ε : ℝ) (hε : 0 ≤ ε) :
    Agarwal2013.StateLevelEpsilonApproximationSpec α ρ (Stream α) :=
  LiteratureChronology.agarwal2013_13e1_exactStateLevelEpsilonApproximationSpec
    rangeCount ε hε

/-- Agarwal et al. exact all-points ε-approximation error after a merge tree. -/
theorem ctreepo_agarwal2013_exactStateLevelEpsilonApproximation_tree_error
    {α ρ : Type*}
    (rangeCount : Stream α → ρ → Nat)
    (ε : ℝ) (hε : 0 ≤ ε)
    (t : MergeTree α) (R : ρ) :
    |(((Agarwal2013.exactStateLevelEpsilonApproximationSpec rangeCount ε hε).query
        (MergeTree.eval
          (Agarwal2013.exactStateLevelEpsilonApproximationSpec
            rangeCount ε hε).build
          (Agarwal2013.exactStateLevelEpsilonApproximationSpec
            rangeCount ε hε).merge t) R : ℝ) -
        (rangeCount (MergeTree.data t) R : ℝ))|
      ≤ ε * ((MergeTree.data t).length : ℝ) :=
  LiteratureChronology.agarwal2013_13e2_exactStateLevelEpsilonApproximation_tree_error
    rangeCount ε hε t R

/-- Agarwal et al. same-weight interval validity on equal-length sibling trees. -/
theorem ctreepo_agarwal2013_sameWeightInterval_valid_on_equalLengthTree
    {S : Type*}
    (A : Agarwal2013.SameWeightIntervalApproximationSpec S)
    (t : MergeTree ℝ)
    (ht : Agarwal2013.EqualLengthSiblingTree t) :
    A.valid (MergeTree.data t) (MergeTree.eval A.build A.merge t) :=
  LiteratureChronology.agarwal2013_13f_sameWeightInterval_valid_on_equalLengthTree
    A t ht

/-- Agarwal et al. same-weight interval error after reducing an equal-length
sibling merge tree. -/
theorem ctreepo_agarwal2013_sameWeightInterval_tree_error_on_equalLength
    {S : Type*}
    (A : Agarwal2013.SameWeightIntervalApproximationSpec S)
    (t : MergeTree ℝ)
    (ht : Agarwal2013.EqualLengthSiblingTree t)
    (I : Agarwal2013.Interval1D) :
    |((A.query (MergeTree.eval A.build A.merge t) I : ℝ) -
        (Agarwal2013.intervalCount (MergeTree.data t) I : ℝ))|
      ≤ A.ε * ((MergeTree.data t).length : ℝ) :=
  LiteratureChronology.agarwal2013_13g_sameWeightInterval_tree_error_on_equalLength
    A t ht I

/-- Agarwal et al. Lemma 3 core: alternating parity halves partition length. -/
theorem ctreepo_agarwal2013_paritySplit_length_sum {α : Type*}
    (xs : List α) :
    (Agarwal2013.paritySplit xs).1.length +
        (Agarwal2013.paritySplit xs).2.length = xs.length :=
  LiteratureChronology.agarwal2013_13h_paritySplit_length_sum xs

/-- Agarwal et al. Lemma 3 core: same-weight halving is unbiased for an
interval count under the uniform even/odd choice. -/
theorem ctreepo_agarwal2013_sameWeightHalving_unbiased_interval_count
    (xs : Stream ℝ) (I : Agarwal2013.Interval1D) :
    Agarwal2013.twoPointMean
        (fun keepEven =>
          Agarwal2013.sameWeightHalvingIntervalEstimate keepEven xs I) =
      (Agarwal2013.intervalCount xs I : ℝ) :=
  LiteratureChronology.agarwal2013_13i_sameWeightHalving_unbiased_interval_count
    xs I

/-- Agarwal et al. Lemma 3 core: one-step same-weight halving has zero mean
over-count error. -/
theorem ctreepo_agarwal2013_sameWeightHalving_interval_error_mean_zero
    (xs : Stream ℝ) (I : Agarwal2013.Interval1D) :
    Agarwal2013.twoPointMean
        (fun keepEven =>
          Agarwal2013.sameWeightHalvingIntervalError keepEven xs I) = 0 :=
  LiteratureChronology.agarwal2013_13j_sameWeightHalving_interval_error_mean_zero
    xs I

/-- Agarwal et al. Lemma 3 core: either same-weight halving parity choice has
absolute interval over-count at most one. -/
theorem ctreepo_agarwal2013_sameWeightHalving_interval_error_abs_le_one
    (keepEven : Bool) (xs : Stream ℝ) (I : Agarwal2013.Interval1D) :
    |Agarwal2013.sameWeightHalvingIntervalError keepEven xs I| ≤ 1 :=
  LiteratureChronology.agarwal2013_13k_sameWeightHalving_interval_error_abs_le_one
    keepEven xs I

/-- Agarwal et al. Lemma 4 setup: level-scaled same-weight halving over-count
has radius `2^level`. -/
theorem ctreepo_agarwal2013_sameWeightHalving_level_error_abs_le
    (level : Nat) (keepEven : Bool)
    (xs : Stream ℝ) (I : Agarwal2013.Interval1D) :
    |Agarwal2013.sameWeightHalvingLevelError level keepEven xs I|
      ≤ (2 ^ level : ℝ) :=
  LiteratureChronology.agarwal2013_13l_sameWeightHalving_level_error_abs_le
    level keepEven xs I

/-- Agarwal et al. Lemma 4 arithmetic: the complete-tree Hoeffding denominator
is at most `2^(2m+1)`. -/
theorem ctreepo_agarwal2013_sameWeightHalving_hoeffdingDenominator_le
    (m : Nat) :
    Agarwal2013.sameWeightHalvingHoeffdingDenominator m ≤
      2 ^ (2 * m + 1) :=
  LiteratureChronology.agarwal2013_13m_sameWeightHalving_hoeffdingDenominator_le m

/-- Agarwal et al. Lemma 4 final scaling: a total over-count bound at root scale
is the same as an `εn` bound for `n = kε * 2^m`. -/
theorem ctreepo_agarwal2013_sameWeightHalving_total_error_to_epsilon_n
    (kε m : Nat) (ε totalError : ℝ)
    (h : |totalError| ≤ ε * (kε : ℝ) * (2 ^ m : ℝ)) :
    |totalError| ≤
      ε * (Agarwal2013.sameWeightHalvingRepresentedLength kε m : ℝ) :=
  LiteratureChronology.agarwal2013_13n_sameWeightHalving_total_error_to_epsilon_n
    kε m ε totalError h

/-- Agarwal et al. Lemma 4 scaling: a Hoeffding root-radius bound
`|M| ≤ h 2^m` becomes an `εn` bound when `h ≤ ε kε`. -/
theorem ctreepo_agarwal2013_sameWeightHalving_root_error_to_epsilon_n_of_scale
    (kε m : Nat) (ε h totalError : ℝ)
    (hscale : h ≤ ε * (kε : ℝ))
    (herror : |totalError| ≤ h * (2 ^ m : ℝ)) :
    |totalError| ≤
      ε * (Agarwal2013.sameWeightHalvingRepresentedLength kε m : ℝ) :=
  LiteratureChronology.agarwal2013_13o_sameWeightHalving_root_error_to_epsilon_n_of_scale
    kε m ε h totalError hscale herror

/-- Agarwal et al. Lemma 4 stochastic step: complete-tree same-weight halving
has the two-sided Azuma/Hoeffding tail bound under explicit martingale
hypotheses. -/
theorem ctreepo_agarwal2013_sameWeightHalving_completeTree_hoeffding_tail
    {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω} {m : Nat}
    [StandardBorelSpace Ω] [IsZeroOrProbabilityMeasure μ]
    (P : Agarwal2013.SameWeightHalvingCompleteTreeProcess Ω mΩ μ m)
    {η : ℝ} (hη : 0 ≤ η) :
    μ.real {ω | η ≤ |P.totalError ω|}
      ≤ 2 * Real.exp
        (-η ^ 2 /
          (2 * ((Finset.sum (Finset.range P.horizon) (fun i : Nat =>
            if i = 0 then (0 : ℝ≥0)
            else
              (‖Agarwal2013.sameWeightHalvingAzumaUpper i -
                  Agarwal2013.sameWeightHalvingAzumaLower i‖₊ / 2) ^ 2) :
            ℝ≥0) : ℝ))) :=
  LiteratureChronology.agarwal2013_13p_sameWeightHalving_completeTree_hoeffding_tail
    P hη

/-- Agarwal et al. Lemma 4 stochastic `εn` tail: the Hoeffding root-radius
tail controls the final interval-error threshold when `h ≤ ε kε`. -/
theorem ctreepo_agarwal2013_sameWeightHalving_completeTree_epsilon_n_tail
    {Ω : Type*} {mΩ : MeasurableSpace Ω} {μ : Measure Ω} {m : Nat}
    [StandardBorelSpace Ω] [IsZeroOrProbabilityMeasure μ]
    (P : Agarwal2013.SameWeightHalvingCompleteTreeProcess Ω mΩ μ m)
    (kε : Nat) (ε h : ℝ)
    (hscale : h ≤ ε * (kε : ℝ))
    (hroot_nonneg : 0 ≤ h * (2 ^ m : ℝ)) :
    μ.real
        {ω |
          ε * (Agarwal2013.sameWeightHalvingRepresentedLength kε m : ℝ) <
            |P.totalError ω|}
      ≤ 2 * Real.exp
        (-(h * (2 ^ m : ℝ)) ^ 2 /
          (2 * ((Finset.sum (Finset.range P.horizon) (fun i : Nat =>
            if i = 0 then (0 : ℝ≥0)
            else
              (‖Agarwal2013.sameWeightHalvingAzumaUpper i -
                  Agarwal2013.sameWeightHalvingAzumaLower i‖₊ / 2) ^ 2) :
            ℝ≥0) : ℝ))) :=
  LiteratureChronology.agarwal2013_13q_sameWeightHalving_completeTree_epsilon_n_tail
    P kε ε h hscale hroot_nonneg

/-- Agarwal et al. Theorem 3.7/logarithmic-layer algebra: finite layer errors
sum to the total represented-mass error bound. -/
theorem ctreepo_agarwal2013_finiteLayer_interval_error_sum_bound
    {ι ρ : Type*} [Fintype ι]
    (ε : ℝ) (mass : ι → Nat)
    (estimate trueCount : ι → ρ → ℝ)
    (h :
      ∀ i R,
        |estimate i R - trueCount i R| ≤ ε * (mass i : ℝ)) :
    ∀ R,
      |(∑ i, estimate i R) - (∑ i, trueCount i R)|
        ≤ ε * (∑ i, (mass i : ℝ)) :=
  LiteratureChronology.agarwal2013_13r_finiteLayer_interval_error_sum_bound
    ε mass estimate trueCount h

/-- Agarwal et al. typed ε-kernel state/readout interface. -/
abbrev ctreepo_agarwal2013_EpsilonKernelSpec :=
  Agarwal2013.EpsilonKernelSpec

/-- Agarwal et al. ε-kernel summaries with merge-closed state are
hierarchically mergeable. -/
theorem ctreepo_agarwal2013_epsilonKernel_hierarchical
    {d : Nat} {S : Type*}
    (K : Agarwal2013.EpsilonKernelSpec d S) :
    HierarchicalMergeable K.build K.valid K.merge :=
  Agarwal2013.epsilonKernel_hierarchical K

/-- Agarwal et al. ε-kernel specs instantiate the generic sized mergeable
query-sketch interface. -/
def ctreepo_agarwal2013_epsilonKernel_toSizedMergeableQuerySketch
    {d : Nat} {S : Type*}
    (K : Agarwal2013.EpsilonKernelSpec d S) :
    SizedMergeableQuerySketch (Point d) S (WidthQuery d) :=
  LiteratureChronology.agarwal2013_17d_epsilonKernel_toSizedMergeableQuerySketch K

/-- Agarwal et al. ε-kernel width error after a merge tree. -/
theorem ctreepo_agarwal2013_epsilonKernel_tree_widthError
    {d : Nat} {S : Type*}
    (K : Agarwal2013.EpsilonKernelSpec d S)
    (t : MergeTree (Point d))
    (direction : Point d) :
    |K.queryWidth (MergeTree.eval K.build K.merge t) direction -
        K.trueWidth (MergeTree.data t) direction|
      ≤ K.ε * max 1 (|K.trueWidth (MergeTree.data t) direction|) :=
  LiteratureChronology.agarwal2013_17e_epsilonKernel_tree_widthError K t direction

/-- Agarwal et al. max-projection state is mergeable by optional `max` in a
common reference frame. -/
theorem ctreepo_agarwal2013_maxProjectionState_append {d : Nat}
    (xs ys : Stream (Point d)) (direction : Point d) :
    Agarwal2013.maxProjectionState (xs ++ ys) direction =
      Agarwal2013.mergeMaxProjectionState
        (Agarwal2013.maxProjectionState xs direction)
        (Agarwal2013.maxProjectionState ys direction) :=
  LiteratureChronology.agarwal2013_17f_maxProjectionState_append xs ys direction

/-- Agarwal et al. min-projection state is mergeable by optional `min` in a
common reference frame. -/
theorem ctreepo_agarwal2013_minProjectionState_append {d : Nat}
    (xs ys : Stream (Point d)) (direction : Point d) :
    Agarwal2013.minProjectionState (xs ++ ys) direction =
      Agarwal2013.mergeMinProjectionState
        (Agarwal2013.minProjectionState xs direction)
        (Agarwal2013.minProjectionState ys direction) :=
  LiteratureChronology.agarwal2013_17g_minProjectionState_append xs ys direction

/-- Agarwal et al. directional width of a merged stream is determined by the
merged max/min projection state. -/
theorem ctreepo_agarwal2013_directionalWidth_append {d : Nat}
    (xs ys : Stream (Point d)) (direction : Point d) :
    Agarwal2013.directionalWidth (xs ++ ys) direction =
      Agarwal2013.projectionWidth
        (Agarwal2013.mergeMaxProjectionState
          (Agarwal2013.maxProjectionState xs direction)
          (Agarwal2013.maxProjectionState ys direction))
        (Agarwal2013.mergeMinProjectionState
          (Agarwal2013.minProjectionState xs direction)
          (Agarwal2013.minProjectionState ys direction)) :=
  LiteratureChronology.agarwal2013_17h_directionalWidth_append xs ys direction

/-- Agarwal et al. common-reference-frame fact: translating all points leaves
directional width unchanged. -/
theorem ctreepo_agarwal2013_directionalWidth_translateStream {d : Nat}
    (offset : Point d) (xs : Stream (Point d)) (direction : Point d) :
    Agarwal2013.directionalWidth
        (Agarwal2013.translateStream offset xs) direction =
      Agarwal2013.directionalWidth xs direction :=
  LiteratureChronology.agarwal2013_17h1_directionalWidth_translateStream
    offset xs direction

/-- Agarwal et al. common-reference-frame fact: nonnegative scaling of all
points scales directional width. -/
theorem ctreepo_agarwal2013_directionalWidth_scaleStream_of_nonneg {d : Nat}
    {c : ℝ} (hc : 0 ≤ c)
    (xs : Stream (Point d)) (direction : Point d) :
    Agarwal2013.directionalWidth
        (Agarwal2013.scaleStream c xs) direction =
      c * Agarwal2013.directionalWidth xs direction :=
  LiteratureChronology.agarwal2013_17h2_directionalWidth_scaleStream_of_nonneg
    hc xs direction

/-- Agarwal et al. exact all-points ε-kernel witness for any nonnegative ε. -/
def ctreepo_agarwal2013_exactEpsilonKernelSpec
    (d : Nat) (ε : ℝ) (hε : 0 ≤ ε) :
    Agarwal2013.EpsilonKernelSpec d (Stream (Point d)) :=
  LiteratureChronology.agarwal2013_17i_exactEpsilonKernelSpec d ε hε

/-- Agarwal et al. exact all-points ε-kernel width error after a merge tree. -/
theorem ctreepo_agarwal2013_exactEpsilonKernel_tree_widthError
    (d : Nat) (ε : ℝ) (hε : 0 ≤ ε)
    (t : MergeTree (Point d)) (direction : Point d) :
    |(Agarwal2013.exactEpsilonKernelSpec d ε hε).queryWidth
        (MergeTree.eval
          (Agarwal2013.exactEpsilonKernelSpec d ε hε).build
          (Agarwal2013.exactEpsilonKernelSpec d ε hε).merge t) direction -
        (Agarwal2013.exactEpsilonKernelSpec d ε hε).trueWidth
          (MergeTree.data t) direction|
      ≤ ε * max 1
          (|(Agarwal2013.exactEpsilonKernelSpec d ε hε).trueWidth
            (MergeTree.data t) direction|) :=
  LiteratureChronology.agarwal2013_17j_exactEpsilonKernel_tree_widthError
    d ε hε t direction

/-- Agarwal et al. hybrid quantile deterministic invariant: promotion traces
only move retained items upward in the hierarchy. -/
theorem ctreepo_agarwal2013_hybridTrace_level_monotone
    (trace : List Agarwal2013.HybridPromotion)
    (h : Agarwal2013.HybridTraceOnlyMovesUp trace) :
    ∀ p ∈ trace, p.beforeLevel ≤ p.afterLevel :=
  Agarwal2013.hybridTrace_level_monotone trace h

/-- Agarwal et al. hybrid random-buffer layer: finite per-level tail budgets
union-bound to the probability that any buffer/level fails. -/
theorem ctreepo_agarwal2013_hybridRandomBuffer_failure_bound
    {Ω Level : Type*} [MeasurableSpace Ω] [Fintype Level]
    {μ : Measure Ω} [IsFiniteMeasure μ]
    (levelError : Ω → Level → ℝ) (threshold δ : Level → ℝ)
    (hTail : ∀ ℓ,
      μ.real {ω | threshold ℓ ≤ |levelError ω ℓ|} ≤ δ ℓ) :
    μ.real (Agarwal2013.hybridRandomBufferFailureEvent levelError threshold) ≤
      ∑ ℓ : Level, δ ℓ :=
  LiteratureChronology.agarwal2013_18a_hybridRandomBuffer_failure_bound
    levelError threshold δ hTail

/-- Agarwal et al. hybrid random-buffer layer with a uniform per-level failure
budget. -/
theorem ctreepo_agarwal2013_hybridRandomBuffer_failure_bound_uniform
    {Ω Level : Type*} [MeasurableSpace Ω] [Fintype Level]
    {μ : Measure Ω} [IsFiniteMeasure μ]
    (levelError : Ω → Level → ℝ) (threshold : Level → ℝ) (δ : ℝ)
    (hTail : ∀ ℓ,
      μ.real {ω | threshold ℓ ≤ |levelError ω ℓ|} ≤ δ) :
    μ.real (Agarwal2013.hybridRandomBufferFailureEvent levelError threshold) ≤
      (Fintype.card Level : ℝ) * δ :=
  LiteratureChronology.agarwal2013_18b_hybridRandomBuffer_failure_bound_uniform
    levelError threshold δ hTail

/-- Agarwal et al. hybrid-summary algebra: random-sample approximation error
plus mergeable-summary approximation error gives the sum of the two budgets. -/
theorem ctreepo_agarwal2013_epsilonApproximation_error_add
    {ρ : Type*} (εs εh n : ℝ)
    (sample summary trueCount : ρ → ℝ)
    (hsample : ∀ R, |sample R - trueCount R| ≤ εs * n)
    (hsummary : ∀ R, |summary R - sample R| ≤ εh * n) :
    ∀ R, |summary R - trueCount R| ≤ (εs + εh) * n :=
  LiteratureChronology.agarwal2013_18c_epsilonApproximation_error_add
    εs εh n sample summary trueCount hsample hsummary

/-- GK executable witness: the stream-fold state records exactly the number of
processed items. -/
theorem ctreepo_gk2001_executable_build_n {α : Type*} [LinearOrder α]
    (p : GK.Executable.Params) (xs : Stream α) :
    (GK.Executable.build p xs).n = xs.length :=
  LiteratureChronology.gk2001_01_executable_build_n p xs

/-- GK executable witness: the sum of tuple gaps equals the number of processed
items. -/
theorem ctreepo_gk2001_executable_build_gapMassValid {α : Type*} [LinearOrder α]
    (p : GK.Executable.Params) (xs : Stream α) :
    GK.Executable.gapMassValid (GK.Executable.build p xs) :=
  LiteratureChronology.gk2001_02_executable_build_gapMassValid p xs

/-- GK executable witness: one transition preserves positive tuple gaps. -/
theorem ctreepo_gk2001_executable_positiveGaps_step
    {α : Type*} [LinearOrder α]
    (p : GK.Executable.Params) (x : α) (st : GK.Executable.State α)
    (h : GK.positiveGaps st.summary) :
    GK.positiveGaps (GK.Executable.step p x st).summary :=
  LiteratureChronology.gk2001_03_executable_positiveGaps_step p x st h

/-- KLL theorem-bundle surface: the mergeable variant supplies its randomized
space bound and state-level hierarchical mergeability. -/
theorem ctreepo_kll2016_theorem4_mergeable_variant_of_algorithm
    {Ω α : Type*}
    [Preorder α] [DecidableRel ((· ≤ ·) : α → α → Prop)]
    [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ]
    (A : KLL.Algorithm Ω α μ) :
    KLL.theorem4_statement A :=
  LiteratureChronology.kll2016_01_theorem4_mergeable_variant_of_algorithm A

/-- C-TreePO-facing placement theorem: KLL-style mergeable randomized
quantile bundles are randomized sized mergeable query sketches. -/
theorem ctreepo_kll2016_subset_randomizedSizedMergeableQuerySketch
    {Ω α : Type*}
    [Preorder α] [DecidableRel ((· ≤ ·) : α → α → Prop)]
    [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ]
    (A : KLL.Algorithm Ω α μ) :
    MergeClosed
      (A.toRandomizedSizedMergeableQuerySketch).valid
      (A.toRandomizedSizedMergeableQuerySketch).merge ∧
    (A.toRandomizedSizedMergeableQuerySketch).query = A.queryRank :=
  LiteratureChronology.kll2016_01a_subset_randomizedSizedMergeableQuerySketch A

/-- KLL theorem-bundle surface: the optimal-space statement is exposed as a
separate typed theorem target. -/
theorem ctreepo_kll2016_theorem5_optimal_variant_of_algorithm
    {Ω α : Type*}
    [Preorder α] [DecidableRel ((· ≤ ·) : α → α → Prop)]
    [MeasurableSpace Ω]
    {μ : Measure Ω} [IsProbabilityMeasure μ]
    (A : KLL.Algorithm Ω α μ) :
    KLL.theorem5_statement A :=
  LiteratureChronology.kll2016_02_theorem5_optimal_variant_of_algorithm A

/-- KLL executable witness: one transition increases represented weighted mass
by one item. -/
theorem ctreepo_kll2016_executable_weightedCount_step {α : Type*}
    (p : KLL.Executable.Params) (x : α) (st : KLL.State α) :
    KLL.weightedCount (KLL.Executable.step p x st) =
      KLL.weightedCount st + 1 :=
  LiteratureChronology.kll2016_03_executable_weightedCount_step p x st

/-- KLL executable witness: the stream-fold state represents exactly the input
stream length as weighted mass. -/
theorem ctreepo_kll2016_executable_build_massValid {α : Type*}
    (p : KLL.Executable.Params) (xs : Stream α) :
    KLL.Executable.massValid xs (KLL.Executable.build p xs) :=
  LiteratureChronology.kll2016_04_executable_build_massValid p xs

/-- Gibbons ordered/free-monoid homomorphism: schedule invariance follows from
preserving list concatenation; commutativity is not part of the hypothesis. -/
theorem ctreepo_gibbons1996_ordered_schedule_invariance {α S : Type*}
    (h : Stream α → S) (combine : S → S → S)
    (h_hom : OrderedListHomomorphism h combine)
    (t₁ t₂ : MergeTree α)
    (h_data : MergeTree.data t₁ = MergeTree.data t₂) :
    MergeTree.eval h combine t₁ = MergeTree.eval h combine t₂ :=
  gibbons1996_ordered_schedule_invariance h combine h_hom t₁ t₂ h_data

/-- Gibbons Third Homomorphism Theorem: a list function that is both leftwards
and rightwards is a list homomorphism. -/
theorem ctreepo_gibbons1996_third_homomorphism {α S : Type*}
    (h : Stream α → S)
    (stepL : α → S → S) (stepR : S → α → S)
    (h_left : Gibbons1996.Leftwards h stepL)
    (h_right : Gibbons1996.Rightwards h stepR) :
    ∃ combine : S → S → S, Gibbons1996.Homomorphic h combine :=
  Gibbons1996.theorem_4_1_third_homomorphism h stepL stepR h_left h_right

/-- Gibbons Lemma 4.3: homomorphism is equivalent to concatenation being
well-defined on equivalence classes induced by the list function. -/
theorem ctreepo_gibbons1996_kernel_congruence_characterization {α S : Type*}
    (h : Stream α → S) :
    (∃ combine : S → S → S, Gibbons1996.Homomorphic h combine) ↔
      Gibbons1996.ConcatKernelCongruent h :=
  Gibbons1996.lemma_4_3_homomorphic_iff_kernel_congruent h

/-- Gibbons Section 5 runtime vocabulary: the reference linear growth function
satisfies the formal linear-time predicate. -/
theorem ctreepo_gibbons1996_linearGrowth_linearTime :
    Gibbons1996.LinearTime Gibbons1996.linearGrowth :=
  LiteratureChronology.gibbons1996_00q_linearGrowth_linearTime

/-- Gibbons Section 5 runtime vocabulary: the reference quadratic growth
function satisfies the formal quadratic-time predicate. -/
theorem ctreepo_gibbons1996_quadraticGrowth_quadraticTime :
    Gibbons1996.QuadraticTime Gibbons1996.quadraticGrowth :=
  LiteratureChronology.gibbons1996_00r_quadraticGrowth_quadraticTime

/-- Gibbons Section 5 runtime vocabulary: the reference `n log n` growth
function satisfies the formal `n log n` predicate. -/
theorem ctreepo_gibbons1996_nLogNGrowth_nLogNTime :
    Gibbons1996.NLogNTime Gibbons1996.nLogNGrowth :=
  LiteratureChronology.gibbons1996_00s_nLogNGrowth_nLogNTime

/-- Gibbons Section 5 runtime vocabulary: reference cost-model package for the
quadratic/linear/`n log n` sorting claims. -/
def ctreepo_gibbons1996_referenceSection5RuntimeClaims
    {α : Type*} (r : α → α → Prop) [DecidableRel r] :
    Gibbons1996.Section5RuntimeClaims r :=
  LiteratureChronology.gibbons1996_00t_referenceSection5RuntimeClaims r

/-- Feldman et al. MUD aggregation: associative/commutative state merging gives
unordered distributed aggregation as a state-level mergeable summary. -/
theorem ctreepo_feldman2008_mud_state_level_mergeable {α S Q : Type*}
    (A : MUDAggregator α S Q) :
    HierarchicalMergeable A.build A.valid A.merge :=
  MUDAggregator.feldman2008_mud_state_level_mergeable A

/-- Feldman et al. MUD aggregation: the induced readout is invariant under
permutation of input records. -/
theorem ctreepo_feldman2008_mud_readout_permutation_invariant {α S Q : Type*}
    (A : MUDAggregator α S Q) :
    PermutationInvariant (fun xs : Stream α => A.readout (A.build xs)) :=
  MUDAggregator.feldman2008_mud_readout_permutation_invariant A

/-- Feldman et al. MUD computation-tree semantics: every item-level tree
evaluates to the canonical folded state on its represented leaves. -/
theorem ctreepo_feldman2008_item_tree_state_eq_build {α S Q : Type*}
    (A : MUDAggregator α S Q) (t : Feldman2008.ComputationTree α) :
    Feldman2008.ComputationTree.evalState A t =
      A.build (Feldman2008.ComputationTree.data t) :=
  Feldman2008.ComputationTree.evalState_eq_build_data A t

/-- Feldman et al. MUD computation-tree semantics: item-level tree readouts are
invariant across trees that represent permuted inputs. -/
theorem ctreepo_feldman2008_item_tree_readout_permutation_invariant {α S Q : Type*}
    (A : MUDAggregator α S Q) {t₁ t₂ : Feldman2008.ComputationTree α}
    (hperm : (Feldman2008.ComputationTree.data t₁).Perm
      (Feldman2008.ComputationTree.data t₂)) :
    Feldman2008.ComputationTree.evalReadout A t₁ =
      Feldman2008.ComputationTree.evalReadout A t₂ :=
  Feldman2008.ComputationTree.evalReadout_eq_of_data_perm A hperm

/-- Feldman et al. Lemma 1: two streaming prefixes that reach the same state
remain indistinguishable after appending the same suffix. -/
theorem ctreepo_feldman2008_streaming_state_congruence_append {α S Q : Type*}
    (A : Feldman2008.StreamingAlgorithm α S Q)
    {q : S} {xPrefix xPrefix' xSuffix : Stream α}
    (hstate : A.runFrom q xPrefix = A.runFrom q xPrefix') :
    A.runFrom q (xPrefix ++ xSuffix) =
      A.runFrom q (xPrefix' ++ xSuffix) :=
  Feldman2008.StreamingAlgorithm.lemma1_streaming_state_congruence_append
    A hstate

/-- Feldman et al. Lemma 2, semantic representative-state merge existence. -/
theorem ctreepo_feldman2008_representative_merge_exists {α S Q : Type*}
    (A : Feldman2008.StreamingAlgorithm α S Q) {qA qB : S} {nA nB : Nat}
    (hA : A.ReachableAtLength qA nA)
    (hB : A.ReachableAtLength qB nB) :
    ∃ xA xB qC,
      xA.length = nA ∧
      xB.length = nB ∧
      A.run xA = qA ∧
      A.run xB = qB ∧
      qC = A.runFrom qA xB ∧
      A.run (xA ++ xB) = qC ∧
      A.ReachableAtLength qC (nA + nB) :=
  Feldman2008.StreamingAlgorithm.lemma2_representative_merge_exists A hA hB

/-- Feldman et al. easy inclusion: polylogarithmic MUD computability implies
polylogarithmic streaming computability. -/
theorem ctreepo_feldman2008_mud_polylog_subset_streaming {α Q : Type*}
    {f : Stream α → Q} :
    Feldman2008.PolylogMUDComputable f →
      Feldman2008.PolylogStreamingComputable f :=
  Feldman2008.mud_polylog_subset_streaming

/-- Feldman et al. asymptotic accounting: a squared polylogarithmic space
bound is still polylogarithmic. -/
theorem ctreepo_feldman2008_polylog_square {r : Nat → ℝ}
    (hr : Feldman2008.PolylogRate r) :
    Feldman2008.PolylogRate (Feldman2008.squareRate r) :=
  hr.square

/-- Feldman et al. Theorem 1, typed citation schema for the hard direction
from symmetric deterministic streaming to MUD. -/
def ctreepo_feldman2008_theorem1_deterministic_streaming_to_mud_statement : Prop :=
  Feldman2008.theorem1_deterministic_streaming_to_mud_statement

/-- Feldman et al. Theorem 1, mechanized semantic construction from symmetric
streaming to the general paper-MUD model. -/
theorem ctreepo_feldman2008_theorem1_deterministic_streaming_to_mud_semantic :
    Feldman2008.theorem1_deterministic_streaming_to_mud_statement :=
  Feldman2008.theorem1_deterministic_streaming_to_mud_semantic

/-- Feldman et al. Theorem 1 concrete representative construction: the
representative MUD induced by a symmetric streaming algorithm computes the same
function on every computation tree. -/
theorem ctreepo_feldman2008_representativeMUDFromStreaming_computesOnAllTrees
    {α Q : Type*} (f : Stream α → Q)
    (hsym : Feldman2008.SymmetricFunction f)
    (A : Feldman2008.CostedStreamingAlgorithm α Q)
    (hcomp : A.Computes f) :
    (Feldman2008.representativeMUDFromStreaming A).ComputesOnAllTrees f :=
  Feldman2008.representativeMUDFromStreaming_computesOnAllTrees f hsym A hcomp

/-- Feldman et al. Theorem 1 corollary: polylogarithmic symmetric streaming
computations have polylogarithmic general paper-MUD simulations. -/
theorem ctreepo_feldman2008_polylog_streaming_subset_general_mud {α Q : Type*}
    {f : Stream α → Q} (hsym : Feldman2008.SymmetricFunction f) :
    Feldman2008.PolylogStreamingComputable f →
      Feldman2008.PolylogGeneralMUDComputable f :=
  Feldman2008.polylog_streaming_subset_general_mud hsym

/-- Feldman et al. Theorem 2, mechanized simultaneous-communication protocol
construction from a symmetric streaming computation. -/
theorem ctreepo_feldman2008_theorem2_streaming_to_scm_semantic :
    Feldman2008.theorem2_streaming_to_scm_statement :=
  Feldman2008.theorem2_streaming_to_scm_semantic

/-- Feldman et al. Theorem 2 corollary: symmetric polylogarithmic streaming
algorithms induce polylogarithmic SCM protocols. -/
theorem ctreepo_feldman2008_polylog_streaming_subset_scm {α Q : Type*}
    {f : Stream α → Q} (hsym : Feldman2008.SymmetricFunction f) :
    Feldman2008.PolylogStreamingComputable f →
      Feldman2008.PolylogSCMComputable f :=
  Feldman2008.polylog_streaming_subset_scm hsym

/-- SCM lower-bound transport: super-polylogarithmic SCM lower bounds rule out
polylogarithmic streaming algorithms for symmetric functions. -/
theorem ctreepo_feldman2008_not_polylog_streaming_of_scm_lower_bound
    {α Q : Type*} {f : Stream α → Q} {lower : Nat → ℝ}
    (hsym : Feldman2008.SymmetricFunction f)
    (hlower : Feldman2008.SCMCommunicationLowerBound f lower)
    (hsuper : Feldman2008.SuperPolylogRate lower) :
    ¬ Feldman2008.PolylogStreamingComputable f :=
  Feldman2008.not_polylog_streaming_of_scm_lower_bound hsym hlower hsuper

/-- Feldman et al. Set Parity is symmetric. -/
theorem ctreepo_feldman2008_setParity_symmetric :
    Feldman2008.SymmetricFunction Feldman2008.setParity :=
  Feldman2008.setParity_symmetric

/-- Feldman et al. finite Set Parity construction: split-stream parity computes
Boolean-vector equality. -/
theorem ctreepo_feldman2008_finSetParity_two_vectors_eq {n : Nat}
    (x y : Fin n → Bool) :
    Feldman2008.finSetParity
      (Feldman2008.finSetParityRecords x ++ Feldman2008.finSetParityRecords y) =
        Feldman2008.boolVectorEquality x y :=
  Feldman2008.finSetParity_two_vectors_eq x y

/-- Feldman et al. finite Set Parity is symmetric. -/
theorem ctreepo_feldman2008_finSetParity_symmetric {n : Nat} :
    Feldman2008.SymmetricFunction (@Feldman2008.finSetParity n) :=
  Feldman2008.finSetParity_symmetric

/-- Feldman et al. finite equality lower-bound obligation used by the Set
Parity reduction. -/
def ctreepo_feldman2008_boolVectorEquality_scm_sqrt_lower_bound_statement : Prop :=
  Feldman2008.boolVectorEquality_scm_sqrt_lower_bound_statement

/-- Feldman et al. deterministic finite-message equality lower-bound core:
Alice's messages must distinguish all Boolean vectors. -/
theorem ctreepo_feldman2008_boolVectorEquality_messageA_card_lower {n : Nat}
    (P : Feldman2008.FiniteTwoPartyProtocol
      (Fin n → Bool) (Fin n → Bool) Bool)
    (hP : P.Computes (@Feldman2008.boolVectorEquality n)) :
    2 ^ n ≤ @Fintype.card P.MessageA P.fintypeMessageA :=
  Feldman2008.boolVectorEquality_messageA_card_lower P hP

/-- Feldman et al. deterministic bit-accounted equality lower-bound core:
Alice must send at least `n` bits. -/
theorem ctreepo_feldman2008_boolVectorEquality_bitsA_lower {n : Nat}
    (P : Feldman2008.BitAccountedTwoPartyProtocol
      (Fin n → Bool) (Fin n → Bool) Bool)
    (hP : P.Computes (@Feldman2008.boolVectorEquality n)) :
    n ≤ P.bitsA :=
  Feldman2008.boolVectorEquality_bitsA_lower P hP

/-- Feldman et al. deterministic bit-accounted equality protocol families have
linear communication lower bound. -/
theorem ctreepo_feldman2008_bitAccountedEquality_linear_bigO_lower
    (F : Feldman2008.BitAccountedEqualityProtocolFamily) :
    BigO Feldman2008.linearRate
      (Feldman2008.BitAccountedEqualityProtocolFamily.communicationRate F) :=
  F.linear_bigO_lower

/-- Feldman et al. finite Set Parity inherits two-party equality lower bounds
through the explicit split-stream reduction. -/
theorem ctreepo_feldman2008_finSetParity_scm_lower_bound_of_equality {n : Nat}
    {lower : Nat → ℝ}
    (heq : Feldman2008.TwoPartyCommunicationLowerBound
      (@Feldman2008.boolVectorEquality n) lower) :
    Feldman2008.SCMCommunicationLowerBound (@Feldman2008.finSetParity n) lower :=
  Feldman2008.finSetParity_scm_lower_bound_of_equality heq

/-- Feldman et al. finite Set Parity SCM lower-bound obligation. -/
def ctreepo_feldman2008_finiteSetParity_scm_sqrt_lower_bound_statement : Prop :=
  Feldman2008.finiteSetParity_scm_sqrt_lower_bound_statement

/-- Feldman et al. finite Set Parity square-root lower-bound statement follows
from the finite equality lower-bound statement by the mechanized reduction. -/
theorem ctreepo_feldman2008_finiteSetParity_scm_sqrt_lower_bound_of_equality
    (heq : Feldman2008.boolVectorEquality_scm_sqrt_lower_bound_statement) :
    Feldman2008.finiteSetParity_scm_sqrt_lower_bound_statement :=
  Feldman2008.finiteSetParity_scm_sqrt_lower_bound_of_equality heq

/-- Feldman et al. deterministic bit-accounted finite Set Parity lower-bound
core: finite Set Parity SCM protocols inherit the equality bit lower bound. -/
theorem ctreepo_feldman2008_finSetParity_bitAccounted_bitsA_lower {n : Nat}
    (P : Feldman2008.BitAccountedTwoPartyProtocol
      (Stream (Feldman2008.FinSetParityRecord n))
      (Stream (Feldman2008.FinSetParityRecord n)) Bool)
    (hP : Feldman2008.BitAccountedSCMComputes P (@Feldman2008.finSetParity n)) :
    n ≤ P.bitsA :=
  Feldman2008.finSetParity_bitAccounted_bitsA_lower P hP

/-- Feldman et al. deterministic bit-accounted finite Set Parity SCM families
have linear communication lower bound. -/
theorem ctreepo_feldman2008_bitAccountedFinSetParity_linear_bigO_lower
    (F : Feldman2008.BitAccountedFinSetParitySCMFamily) :
    BigO Feldman2008.linearRate
      (Feldman2008.BitAccountedFinSetParitySCMFamily.communicationRate F) :=
  F.linear_bigO_lower

/-- Feldman et al. private-coin finite Set Parity reduction preserves
bounded-error correctness seed-counts. -/
theorem ctreepo_feldman2008_privateCoinFinSetParity_success_preserved {n : Nat}
    (P : Feldman2008.PrivateCoinBitAccountedTwoPartyProtocol
      (Stream (Feldman2008.FinSetParityRecord n))
      (Stream (Feldman2008.FinSetParityRecord n)) Bool)
    (x y : Fin n → Bool) :
    (Feldman2008.privateCoinEqualityProtocolFromFinSetParity P).successCount
      (@Feldman2008.boolVectorEquality n) x y =
    P.successCount (Feldman2008.scmSplitFunction (@Feldman2008.finSetParity n))
      (Feldman2008.finSetParityRecords x) (Feldman2008.finSetParityRecords y) :=
  Feldman2008.privateCoinEqualityProtocolFromFinSetParity_successCount P x y

/-- Feldman et al. private-coin finite Set Parity lower-bound obligation reduces
to the corresponding private-coin equality lower-bound obligation. -/
theorem ctreepo_feldman2008_privateCoinFinSetParity_scm_sqrt_lower_bound_of_equality
    {successNumerator successDenominator : Nat}
    (heq : Feldman2008.privateCoinEquality_scm_sqrt_lower_bound_statement
      successNumerator successDenominator) :
    Feldman2008.privateCoinFinSetParity_scm_sqrt_lower_bound_statement
      successNumerator successDenominator :=
  Feldman2008.privateCoinFinSetParity_scm_sqrt_lower_bound_of_equality heq

/-- Feldman et al. Set Parity SCM lower-bound obligation. -/
def ctreepo_feldman2008_setParity_scm_sqrt_lower_bound_statement : Prop :=
  Feldman2008.setParity_scm_sqrt_lower_bound_statement

/-- Feldman et al. public-randomness bookkeeping: if every public seed computes
the target exactly, every seed is successful on every input. -/
theorem ctreepo_feldman2008_publicRandom_successSet_eq_univ_of_computesSeedwise
    {Ω α Q : Type*}
    (A : Feldman2008.PublicRandomStreamingFamily Ω α Q)
    {f : Stream α → Q}
    (hcomp : A.ComputesSeedwise f) (xs : Stream α) :
    A.SuccessSet f xs = Set.univ :=
  LiteratureChronology.feldman2008_15j_publicRandom_successSet_eq_univ_of_computesSeedwise
    A hcomp xs

/-- Feldman et al. public-randomness probability layer: seedwise exactness gives
success probability one for every input. -/
theorem ctreepo_feldman2008_publicRandom_successProbability_eq_one_of_computesSeedwise
    {Ω α Q : Type*} [MeasurableSpace Ω]
    (A : Feldman2008.PublicRandomStreamingFamily Ω α Q)
    (μ : Measure Ω) [IsProbabilityMeasure μ]
    {f : Stream α → Q}
    (hcomp : A.ComputesSeedwise f) (xs : Stream α) :
    A.SuccessProbability μ f xs = 1 :=
  LiteratureChronology.feldman2008_15j1_publicRandom_successProbability_eq_one_of_computesSeedwise
    A μ hcomp xs

/-- Feldman et al. public-randomness positive theorem for the general MUD
model: seedwise exact public-coin streaming computations convert seedwise to
general MUD computations with polylogarithmic costs. -/
theorem ctreepo_feldman2008_public_randomness_seedwise_general_mud :
    Feldman2008.public_randomness_seedwise_general_mud_statement :=
  LiteratureChronology.feldman2008_15k_public_randomness_seedwise_general_mud

/-- Feldman et al. public-randomness extension, seedwise deterministic theorem
schema. -/
def ctreepo_feldman2008_public_randomness_seedwise_extension_statement : Prop :=
  Feldman2008.public_randomness_seedwise_extension_statement

/-- Feldman et al. Theorem 3, typed citation schema for the private-randomness
separation, stated with the concrete computability predicates
`Feldman2008.RandomStreamingPolylogComputable` (existing
public-randomness streaming success layer) and
`Feldman2008.PrivateRandomMUDPolylogComputable` (per-node private-seed
counting MUD model); earlier forms existentially quantified or merely
parameterized the predicates and could be instantiated vacuously. See the
docstring in `FormalProbability/ML/MergeableSummaries/Feldman2008.lean`. -/
def ctreepo_feldman2008_theorem3_private_randomness_separation_statement : Prop :=
  Feldman2008.theorem3_private_randomness_separation_statement

/-- Feldman et al. Symmetric Index canonical promised streams inhabit the
promise domain. -/
theorem ctreepo_feldman2008_symmetricIndexCanonical_mem_domain {n : Nat}
    (x y : Fin n → Bool) (p q : Fin n) (hpromise : x q = y p) :
    @Feldman2008.symmetricIndexDomain n
      (Feldman2008.symmetricIndexCanonical x y p q) :=
  Feldman2008.symmetricIndexCanonical_mem_domain x y p q hpromise

/-- Feldman et al. Symmetric Index canonical readout correctness: the concrete
readout returns `x_q`. -/
theorem ctreepo_feldman2008_symmetricIndexCanonical_readout_eq {n : Nat}
    (x y : Fin n → Bool) (p q : Fin n) :
    Feldman2008.symmetricIndex (Feldman2008.symmetricIndexCanonical x y p q) =
      x q :=
  Feldman2008.symmetricIndexCanonical_readout_eq x y p q

/-- Feldman et al. Symmetric Index readout is permutation-invariant on the
promise domain. -/
theorem ctreepo_feldman2008_symmetricIndex_promise_symmetric {n : Nat} :
    Feldman2008.PromiseSymmetric
      (@Feldman2008.symmetricIndexDomain n) (@Feldman2008.symmetricIndex n) :=
  Feldman2008.symmetricIndex_promise_symmetric

/-- Feldman et al. Symmetric Index SCM lower-bound obligation. -/
def ctreepo_feldman2008_symmetricIndex_scm_linear_lower_bound_statement : Prop :=
  Feldman2008.symmetricIndex_scm_linear_lower_bound_statement

/-- Feldman et al. Theorem 4, typed citation schema for the promise-problem
separation. -/
def ctreepo_feldman2008_theorem4_promise_separation_statement : Prop :=
  Feldman2008.theorem4_promise_separation_statement

/-- Feldman et al. Theorem 5, typed citation schema for the indeterminate
function separation. -/
def ctreepo_feldman2008_theorem5_indeterminate_separation_statement : Prop :=
  Feldman2008.theorem5_indeterminate_separation_statement

/-- Gray et al. Data Cube taxonomy: scalar-output distributive aggregates are a
special case of algebraic state/readout aggregates. -/
theorem ctreepo_gray1997_distributive_is_algebraic {α Q : Type*}
    (A : DistributiveAggregate α Q) :
    HierarchicalMergeable A.aggregate (fun xs q => q = A.aggregate xs) A.combine :=
  Gray1997.distributive_is_algebraic A

/-- Gray et al. Data Cube taxonomy: classical state-level sketches live in the
algebraic-aggregate class. -/
theorem ctreepo_gray1997_state_level_summary_is_algebraic {α S Q : Type*}
    (A : StateLevelMergeableSummary α S Q) :
    HierarchicalMergeable A.build A.valid A.merge :=
  Gray1997.state_level_summary_is_algebraic A

/-- Gray et al. Data Cube shape: a cube over `N` dimensions has `2^N` masks. -/
theorem ctreepo_gray1997_cubeMask_card {ι : Type*} [DecidableEq ι] [Fintype ι] :
    Fintype.card (Gray1997.CubeMask ι) = 2 ^ Fintype.card ι :=
  Gray1997.cubeMask_card

/-- Gray et al. Data Cube shape: excluding the GROUP BY core leaves `2^N-1`
super-aggregate masks. -/
theorem ctreepo_gray1997_superAggregateMask_card {ι : Type*}
    [DecidableEq ι] [Fintype ι] :
    Fintype.card (Gray1997.SuperAggregateMask ι) =
      2 ^ Fintype.card ι - 1 :=
  Gray1997.superAggregateMask_card

/-- Gray et al. Data Cube shape: a heterogeneous cube address space has
`∏ᵢ(Cᵢ+1)` cells. -/
theorem ctreepo_gray1997_cubeAddressD_card {ι : Type*} [Fintype ι]
    (V : ι → Type*) [∀ i, Fintype (V i)] :
    Fintype.card (Gray1997.CubeAddressD V) =
      ∏ i, (Fintype.card (V i) + 1) :=
  Gray1997.cubeAddressD_card V

/-- Gray et al. Data Cube shape: ROLLUP over `n` ordered dimensions has `n+1`
levels. -/
theorem ctreepo_gray1997_rollupLevel_card (n : Nat) :
    Fintype.card (Fin (n + 1)) = n + 1 :=
  Gray1997.rollupLevel_card n

/-- Gray et al. Data Cube shape: ROLLUP over `n` ordered dimensions adds `n`
super-aggregate levels beyond GROUP BY. -/
theorem ctreepo_gray1997_rollupSuperLevel_card (n : Nat) :
    Fintype.card (Gray1997.RollupSuperLevel n) = n :=
  Gray1997.rollupSuperLevel_card n

/-- Gray et al. Data Cube shape: distinct ROLLUP prefix levels give distinct
masks. -/
theorem ctreepo_gray1997_rollupPrefixMask_injective {n : Nat} :
    Function.Injective (Gray1997.rollupPrefixMask (n := n)) :=
  Gray1997.rollupPrefixMask_injective

/-- Gray et al. Data Cube algorithm count: direct cube has `2^N-1`
super-aggregate updates per tuple beyond the GROUP BY core update. -/
theorem ctreepo_gray1997_directCubeSuperAggregateUpdatesPerTuple
    {ι : Type*} [DecidableEq ι] [Fintype ι] :
    Gray1997.directCubeSuperAggregateUpdatesPerTuple ι =
      2 ^ Fintype.card ι - 1 :=
  Gray1997.directCubeSuperAggregateUpdatesPerTuple_eq_pow_sub_one

/-- Gray et al. Data Cube algorithm count: direct ROLLUP has `n`
super-aggregate updates per tuple beyond the GROUP BY core update. -/
theorem ctreepo_gray1997_directRollupSuperAggregateUpdatesPerTuple (n : Nat) :
    Gray1997.directRollupSuperAggregateUpdatesPerTuple n = n :=
  Gray1997.directRollupSuperAggregateUpdatesPerTuple_eq n

/-- Gray et al. Data Cube algorithm count: ROLLUP touches no more cells per
tuple than CUBE for the same dimension count. -/
theorem ctreepo_gray1997_directRollupUpdates_le_directCubeUpdates (n : Nat) :
    Gray1997.directRollupUpdatesPerTuple n ≤
      Gray1997.directCubeUpdatesPerTuple (Fin n) :=
  Gray1997.directRollupUpdatesPerTuple_le_directCubeUpdatesPerTuple n

/-- Gray et al. Data Cube algorithm count: the same comparison for
super-aggregate cells beyond the GROUP BY core. -/
theorem ctreepo_gray1997_directRollupSuperAggregateUpdates_le_directCubeSuperAggregateUpdates
    (n : Nat) :
    Gray1997.directRollupSuperAggregateUpdatesPerTuple n ≤
      Gray1997.directCubeSuperAggregateUpdatesPerTuple (Fin n) :=
  Gray1997.directRollupSuperAggregateUpdatesPerTuple_le_directCubeSuperAggregateUpdatesPerTuple n

/-- Gray et al. Data Cube taxonomy: average is algebraic via fixed `(sum,count)`
state, but scalar average alone is not a distributive output homomorphism. -/
theorem ctreepo_gray1997_average_not_distributive_scalar :
    ¬ ∃ combine : Rat → Rat → Rat,
      OrderedListHomomorphism Gray1997.averageRat combine :=
  Gray1997.averageRat_not_distributive_scalar

/-- Gray et al. Data Cube taxonomy: average is algebraic at the oracle level. -/
theorem ctreepo_gray1997_average_is_algebraic_oracle :
    Gray1997.IsAlgebraicOracle Gray1997.averageRat :=
  Gray1997.averageRat_is_algebraic_oracle

/-- Gray et al. Data Cube taxonomy: scalar average is not distributive at the
oracle level. -/
theorem ctreepo_gray1997_average_not_distributive_oracle :
    ¬ Gray1997.IsDistributiveOracle Gray1997.averageRat :=
  Gray1997.averageRat_not_distributive_oracle

/-- Gray et al. Data Cube dynamic-maintenance warning: scalar maximum supports
insertions but no exact deletion law exists on the scalar state alone. -/
theorem ctreepo_gray1997_max_no_scalar_delete_front :
    ¬ ∃ delete : Nat → Nat → Nat,
      ∀ xs x, delete (Gray1997.maxNat (x :: xs)) x = Gray1997.maxNat xs :=
  Gray1997.maxNat_no_scalar_delete_front

/-- Gray et al. Data Cube lower-bound schema: an exact finite mergeable state
must inject every prefix family that can be separated by a future context. -/
theorem ctreepo_gray1997_contextual_state_lower_bound
    {α S Q ι : Type*} [Fintype ι] [Fintype S]
    (build : Stream α → S) (merge : S → S → S) (decode : S → Q)
    (oracle : Stream α → Q) (family : ι → Stream α)
    (h_hom : OrderedListHomomorphism build merge)
    (h_exact : ∀ xs, decode (build xs) = oracle xs)
    (h_sep : Gray1997.ContextuallySeparated family oracle) :
    Fintype.card ι ≤ Fintype.card S :=
  Gray1997.state_card_lower_bound_of_contextual_separation
    build merge decode oracle family h_hom h_exact h_sep

/-- Gray et al. Data Cube holistic example: exact Boolean mode over bounded
prefix families requires state cardinality growing with the bound. -/
theorem ctreepo_gray1997_modeBool_state_card_lower_bound
    (n : Nat) {S : Type*} [Fintype S]
    (build : Stream Bool → S) (merge : S → S → S) (decode : S → Bool)
    (h_hom : OrderedListHomomorphism build merge)
    (h_exact : ∀ xs, decode (build xs) = Gray1997.modeBool xs) :
    n + 1 ≤ Fintype.card S :=
  Gray1997.modeBool_state_card_lower_bound n build merge decode h_hom h_exact

/-- Gray et al. Data Cube holistic example: exact Boolean mode has no finite
state-level ordered-list-homomorphic realization. -/
theorem ctreepo_gray1997_modeBool_no_finite_state_homomorphic_realization :
    ¬ ∃ S : Type, ∃ _ : Fintype S,
      ∃ build : Stream Bool → S, ∃ merge : S → S → S, ∃ decode : S → Bool,
        OrderedListHomomorphism build merge ∧
          ∀ xs, decode (build xs) = Gray1997.modeBool xs :=
  Gray1997.modeBool_no_finite_state_homomorphic_realization

/-- Gray et al. Data Cube holistic median/rank-style example: exact Boolean
median/majority over bounded prefix families requires state cardinality growing
with the bound. -/
theorem ctreepo_gray1997_medianMajorityBool_state_card_lower_bound
    (n : Nat) {S : Type*} [Fintype S]
    (build : Stream Bool → S) (merge : S → S → S) (decode : S → Bool)
    (h_hom : OrderedListHomomorphism build merge)
    (h_exact : ∀ xs, decode (build xs) = Gray1997.medianMajorityBool xs) :
    n + 1 ≤ Fintype.card S :=
  Gray1997.medianMajorityBool_state_card_lower_bound n build merge decode h_hom h_exact

/-- Gray et al. Data Cube holistic median/rank-style example: exact Boolean
median/majority has no finite state-level ordered-list-homomorphic
realization. -/
theorem ctreepo_gray1997_medianMajorityBool_no_finite_state_homomorphic_realization :
    ¬ ∃ S : Type, ∃ _ : Fintype S,
      ∃ build : Stream Bool → S, ∃ merge : S → S → S, ∃ decode : S → Bool,
        OrderedListHomomorphism build merge ∧
          ∀ xs, decode (build xs) = Gray1997.medianMajorityBool xs :=
  Gray1997.medianMajorityBool_no_finite_state_homomorphic_realization

/-- Flajolet et al. HyperLogLog: register merge is the associative max algebra. -/
theorem ctreepo_flajolet2007_hll_merge_associative {m : Nat} :
    MergeAssociative (@HLLRegisters.merge m) :=
  flajolet2007_hll_merge_associative

/-- Flajolet et al. HyperLogLog: register merge is commutative. -/
theorem ctreepo_flajolet2007_hll_merge_commutative {m : Nat} :
    MergeCommutative (@HLLRegisters.merge m) :=
  flajolet2007_hll_merge_commutative

/-- Flajolet et al. HyperLogLog: register merge is idempotent. -/
theorem ctreepo_flajolet2007_hll_merge_idempotent {m : Nat} :
    MergeIdempotent (@HLLRegisters.merge m) :=
  flajolet2007_hll_merge_idempotent

/-- Flajolet et al. HyperLogLog: max-register states are classical
state-level mergeable summaries for any supplied readout. -/
theorem ctreepo_flajolet2007_hll_state_level_mergeable {α Q : Type*} {m : Nat}
    (bucket : α → Fin m) (rank : α → Nat)
    (query : HLLRegisters m → Q) :
    HierarchicalMergeable
      (HLLRegisters.build bucket rank)
      (HLLRegisters.valid bucket rank)
      HLLRegisters.merge :=
  flajolet2007_hll_state_level_mergeable bucket rank query

/-- Flajolet et al. HyperLogLog RSE expression at `p = 14`, used to justify
the paper text's "under 1%" phrasing for this precision. -/
theorem ctreepo_flajolet2007_hll_rse_p14_formula :
    hllRelativeStandardError 14 =
      ((104 : ℝ) / 100) / Real.sqrt (16384 : ℝ) :=
  flajolet2007_hll_rse_p14_formula

/-- Flajolet et al. HyperLogLog: at `p = 14`, the asymptotic RSE formula is
exactly `13/1600`. -/
theorem ctreepo_flajolet2007_hll_rse_p14_exact :
    hllRelativeStandardError 14 = (13 : ℝ) / 1600 :=
  flajolet2007_hll_rse_p14_exact

/-- Flajolet et al. HyperLogLog: at `p = 14`, the asymptotic RSE formula is
below one percent. -/
theorem ctreepo_flajolet2007_hll_rse_p14_under_one_percent :
    hllRelativeStandardError 14 < (1 : ℝ) / 100 :=
  flajolet2007_hll_rse_p14_under_one_percent

/-- Flajolet et al. HyperLogLog: `rho` is a positive, one-indexed first-one
position in the hash suffix bits. -/
theorem ctreepo_flajolet2007_hll_rho_positive (bits : List Bool) :
    0 < Flajolet2007.rho bits :=
  Flajolet2007.rho_positive_statement bits

/-- Flajolet et al. HyperLogLog: a prefix bit list denotes a bucket below
`2^p`, where `p` is the prefix length. -/
theorem ctreepo_flajolet2007_bitsToNat_lt_two_pow_length
    (bits : List Bool) :
    Flajolet2007.bitsToNat bits < 2 ^ bits.length :=
  Flajolet2007.prefix_bits_bucket_bound bits

/-- Flajolet et al. HyperLogLog: full hash words have positive extracted
ranks. -/
theorem ctreepo_flajolet2007_hashWord_rank_positive {p : Nat}
    (w : Flajolet2007.HashWord p) :
    0 < w.rank :=
  Flajolet2007.hashWord_rank_positive w

/-- Flajolet et al. HyperLogLog: the hash-observation build is a
register-state homomorphism over append. -/
theorem ctreepo_flajolet2007_hll_buildFromHashes_append {m : Nat}
    (xs ys : Stream (Flajolet2007.HashObservation m)) :
    HLLRegisters.buildFromHashes (xs ++ ys) =
      HLLRegisters.merge
        (HLLRegisters.buildFromHashes xs)
        (HLLRegisters.buildFromHashes ys) :=
  Flajolet2007.hll_buildFromHashes_append xs ys

/-- Flajolet et al. HyperLogLog: hash-observation states are classical
state-level mergeable summaries for any supplied readout. -/
theorem ctreepo_flajolet2007_hll_hash_state_level_mergeable {Q : Type*} {m : Nat}
    (query : HLLRegisters m → Q) :
    HierarchicalMergeable
      (@HLLRegisters.buildFromHashes m)
      (@HLLRegisters.validFromHashes m)
      (@HLLRegisters.merge m) :=
  Flajolet2007.hll_stateLevelSummaryFromHashes_hierarchical query

/-- Flajolet et al. HyperLogLog: HLL with an explicit ideal hash-word source
is state-level mergeable. -/
theorem ctreepo_flajolet2007_idealHash_state_level_mergeable
    {α Q : Type*} {p : Nat}
    (H : Flajolet2007.IdealHashFamily α p)
    (query : HLLRegisters (2 ^ p) → Q) :
    HierarchicalMergeable H.build H.valid HLLRegisters.merge :=
  Flajolet2007.idealHash_stateLevelSummary_hierarchical H query

/-- Flajolet et al. ideal-hash probability-law layer: fixing a seed in a random
ideal hash law recovers the deterministic HLL append homomorphism. -/
theorem ctreepo_flajolet2007_randomIdealHash_seedFamily_build_append
    {Ω α : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {p : Nat}
    (H : Flajolet2007.RandomIdealHashFamily Ω α μ p) (ω : Ω)
    (xs ys : Stream α) :
    (H.seedFamily ω).build (xs ++ ys) =
      HLLRegisters.merge ((H.seedFamily ω).build xs) ((H.seedFamily ω).build ys) :=
  Flajolet2007.RandomIdealHashFamily.seedFamily_build_append H ω xs ys

/-- Flajolet et al. ideal-hash probability-law layer: the seeded family remains
state-level mergeable because the merge proof is deterministic per seed. -/
theorem ctreepo_flajolet2007_randomIdealHash_seedFamily_hierarchical
    {Ω α Q : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {p : Nat}
    (H : Flajolet2007.RandomIdealHashFamily Ω α μ p) (ω : Ω)
    (query : HLLRegisters (2 ^ p) → Q) :
    HierarchicalMergeable (H.seedFamily ω).build (H.seedFamily ω).valid HLLRegisters.merge :=
  Flajolet2007.RandomIdealHashFamily.seedFamily_hierarchical H ω query

/-- Flajolet et al. HyperLogLog: the all-zero state has harmonic indicator one
when the register count is nonzero. -/
theorem ctreepo_flajolet2007_hll_indicatorZ_empty {m : Nat} (hm : m ≠ 0) :
    HLLRegisters.indicatorZ (HLLRegisters.empty m) = 1 :=
  Flajolet2007.hll_indicatorZ_empty hm

/-- Flajolet et al. HyperLogLog: raw estimator/readout on the all-zero state. -/
theorem ctreepo_flajolet2007_hll_rawEstimator_empty {m : Nat} (hm : m ≠ 0) :
    HLLRegisters.rawEstimator (HLLRegisters.empty m) =
      HLLRegisters.alpha m * (m : ℝ) ^ (2 : Nat) :=
  Flajolet2007.hll_rawEstimator_empty hm

/-- Flajolet et al. HyperLogLog: the precision-parameter RSE formula factors
through the register count `m = 2^p`. -/
theorem ctreepo_flajolet2007_hll_relativeStandardError_registerCount (p : Nat) :
    hllRelativeStandardError p =
      Flajolet2007.relativeStandardErrorOfRegisterCount
        (hllRegisterCount p) :=
  Flajolet2007.hll_relativeStandardError_registerCount p

/-- Flajolet et al. HyperLogLog: typed theorem-schema bundle for the stochastic
estimator claims over an ideal hash model. -/
def ctreepo_flajolet2007_hll_stochasticEstimatorClaims
    (mean stddev cardinality : Nat → ℝ) (m : Nat) : Prop :=
  Flajolet2007.hll_stochasticEstimatorClaims mean stddev cardinality m

/-- Flajolet et al. HyperLogLog: Big-O relaxation of the stochastic estimator
schema. -/
def ctreepo_flajolet2007_hll_stochasticEstimatorBigOClaims
    (mean stddev cardinality : Nat → ℝ) (m : Nat) : Prop :=
  Flajolet2007.StochasticEstimatorBigOClaims mean stddev cardinality m

/-- Flajolet et al. poissonization layer: a supplied poissonized sequence is
the integer-sampled Poisson mixture of the fixed-cardinality sequence. -/
def ctreepo_flajolet2007_PoissonizedBySeries
    (fixed poisson : Nat → ℝ) : Prop :=
  Flajolet2007.PoissonizedBySeries fixed poisson

/-- Flajolet et al. poissonization/depoissonization package: Poisson mixture
identity, poissonized asymptotics, and depoissonization transfer. -/
abbrev ctreepo_flajolet2007_PoissonizationDepoissonizationAnalysis :=
  Flajolet2007.PoissonizationDepoissonizationAnalysis

/-- Flajolet et al. analytic pipeline: poissonized asymptotics plus a
depoissonization transfer imply the fixed-cardinality asymptotic. -/
theorem ctreepo_flajolet2007_fixedCardinality_asymptotic_of_poisson_depoissonization
    {fixed poisson asymptoticFormula : Nat → ℝ}
    (hpoisson :
      Flajolet2007.PoissonIndicatorExpectationAsymptotic poisson asymptoticFormula)
    (htransfer :
      Flajolet2007.DepoissonizationTransfer fixed poisson asymptoticFormula) :
    Asymptotics.IsEquivalent Filter.atTop fixed asymptoticFormula :=
  Flajolet2007.fixedCardinality_asymptotic_of_poisson_depoissonization
    hpoisson htransfer

/-- Flajolet et al. analytic pipeline: the full poissonization package entails
the fixed-cardinality asymptotic. -/
theorem ctreepo_flajolet2007_fixedCardinality_asymptotic_of_poissonization_analysis
    {fixed poisson asymptoticFormula : Nat → ℝ}
    (A : Flajolet2007.PoissonizationDepoissonizationAnalysis
      fixed poisson asymptoticFormula) :
    Asymptotics.IsEquivalent Filter.atTop fixed asymptoticFormula :=
  Flajolet2007.fixedCardinality_asymptotic_of_poissonization_analysis A

/-- Flajolet et al. analytic pipeline: RSE asymptotic equivalence entails the
weaker Big-O RSE statement. -/
theorem ctreepo_flajolet2007_relativeStandardErrorBigO_of_asymptotic
    {stddev cardinality : Nat → ℝ} {m : Nat}
    (h : Flajolet2007.RelativeStandardErrorAsymptotic stddev cardinality m) :
    Flajolet2007.RelativeStandardErrorBigO stddev cardinality m :=
  Flajolet2007.relativeStandardErrorBigO_of_asymptotic h

end LiteratureFormalizationAliases
