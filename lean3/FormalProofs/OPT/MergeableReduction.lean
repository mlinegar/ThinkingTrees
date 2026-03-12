import FormalProofs.OPT.GlobalAssumptions
import FormalProbability.ML.MergeableSummaries

/-!
# FormalProofs/OPT/MergeableReduction.lean

Bridge lemmas from OPS global assumptions to the mergeable-summary interfaces
in `FormalProbability.ML.MergeableSummaries`.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

open ML.MergeableSummary

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]

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

/-- A1/A2/A3 imply closure of the OPS adapter under summary merge. -/
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

/-- Deterministic OPS with A1/A2/A3 is hierarchically mergeable over merge trees. -/
theorem ops_hierarchical_mergeable_of_global (g : Strings → Strings) (fstar : Strings → Y)
    (hA1 : A1_global g fstar) (hA2 : A2_global g fstar) (hA3 : A3_global g fstar) :
    HierarchicalMergeable (opsBuildDet g) (opsValidDet g fstar) (opsMergeDet g) := by
  simpa [opsValidSketchDet] using
    (hierarchical_of_full
      (V := opsValidSketchDet g fstar hA1)
      (merge := opsMergeDet g)
      (ops_mergeClosed_of_global g fstar hA1 hA2 hA3))

/-- OPS Proposition 3 as a direct reduction to the classical mergeable-summary notion. -/
theorem ops_reduction_to_classical_mergeable (g : Strings → Strings) (fstar : Strings → Y)
    (hA1 : A1_global g fstar) (hA2 : A2_global g fstar) (hA3 : A3_global g fstar) :
    IsMergeableSummary g fstar :=
  prop3_mergeable_classical g fstar hA1 hA2 hA3

/-- Under commutative oracle semantics, OPS merge is commutative at oracle level. -/
theorem ops_merge_commutative_oracle (g : Strings → Strings) (fstar : Strings → Y)
    (hA1 : A1_global g fstar) (h_comm : OracleCommutative fstar) :
    ∀ a b, D fstar (opsMergeDet g a b) (opsMergeDet g b a) = 0 := by
  intro a b
  simpa [opsMergeDet, summaryMerge] using mergeable_commutative g fstar hA1 h_comm a b

