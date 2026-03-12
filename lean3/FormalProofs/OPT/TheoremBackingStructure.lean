import FormalProofs.OPT.TheoremBackingAssumptions
import FormalProofs.OPT.MergeableReduction
import FormalProofs.OPT.ExpectationTheory

/-!
# FormalProofs/OPT/TheoremBackingStructure.lean

Structural consequences of the theorem-backing interfaces.

This file packages three key points.

1. The broadest exact interface admits an "on-support" characterization:
   exact theorem-backedness means every realized leaf summary, internal-node
   reduction, and in-range resummary is oracle-exact on support.
2. For deterministic theorem-domain summaries, exact theorem-backedness on
   **all trees** plus `A3` collapses to the global `A1/A2/A3` regime.
3. Sketch / codec assumptions are explicit special cases of the broadest direct
   interfaces, and under `A3` they induce the classical mergeable-summary view.
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

open ML.MergeableSummary

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]
variable {Sketch : Type*}

/-- Support-level characterization of exact theorem-backedness. -/
def SupportExactTheoremBacked
    (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y) : Prop :=
  (∀ b, b ∈ leaves T → ∀ z ∈ (g b).support, D fstar z b = 0) ∧
  (∀ p, p ∈ internal_nodes T →
    ∀ z ∈ (reduce g (BinTree.node p.1 p.2)).support,
      D fstar z (S (BinTree.node p.1 p.2)) = 0) ∧
  (∀ Z, InRange g Z → ∀ z ∈ (g Z).support, D fstar z Z = 0)

section SupportCharacterization

variable {Y : Type*} [BoundedPseudoMetricSpace Y]

/-- Exact theorem-backedness is equivalent to support-level zero distortion on
all realized leaves, internal merges, and in-range resummaries. -/
theorem exactTheoremBacked_nonempty_iff_supportExactTheoremBacked
    {g : Summarizer Strings} {T : BinTree Strings} {fstar : Strings → Y} :
    Nonempty (ExactTheoremBacked g T fstar) ↔ SupportExactTheoremBacked g T fstar := by
  constructor
  · intro hExact
    rcases hExact with ⟨hExact⟩
    refine ⟨?_, ?_, ?_⟩
    · exact (L1_iff_dist_zero_on_support_typeclass g T fstar).1 hExact.localLaws.law1
    · exact (L2_iff_dist_zero_on_support_typeclass g T fstar).1 hExact.localLaws.law2
    · exact (L3_iff_dist_zero_on_support_typeclass g fstar).1 hExact.localLaws.law3
  · intro hSupport
    rcases hSupport with ⟨hLeaf, hMerge, hIdemp⟩
    refine ⟨ExactTheoremBacked.ofLocalLaws ?_⟩
    refine ⟨?_, ?_, ?_⟩
    · exact (L1_iff_dist_zero_on_support_typeclass g T fstar).2 hLeaf
    · exact (L2_iff_dist_zero_on_support_typeclass g T fstar).2 hMerge
    · exact (L3_iff_dist_zero_on_support_typeclass g fstar).2 hIdemp

end SupportCharacterization

/-- Direct summary exact assumptions are just the broadest exact interface under
a route-specific name. -/
def directSummaryExactAssumptionsEquivExactTheoremBacked
    {g : Summarizer Strings} {T : BinTree Strings} {fstar : Strings → Y} :
    DirectSummaryExactAssumptions g T fstar ≃ ExactTheoremBacked g T fstar where
  toFun := fun h => ⟨h.localLaws⟩
  invFun := fun h => ⟨h.localLaws⟩
  left_inv := by intro h; cases h; rfl
  right_inv := by intro h; cases h; rfl

/-- Proposition-level form of `directSummaryExactAssumptionsEquivExactTheoremBacked`. -/
theorem directSummaryExactAssumptions_nonempty_iff_exactTheoremBacked
    {g : Summarizer Strings} {T : BinTree Strings} {fstar : Strings → Y} :
    Nonempty (DirectSummaryExactAssumptions g T fstar) ↔
      Nonempty (ExactTheoremBacked g T fstar) := by
  constructor
  · intro h
    rcases h with ⟨h⟩
    exact ⟨directSummaryExactAssumptionsEquivExactTheoremBacked.toFun h⟩
  · intro h
    rcases h with ⟨h⟩
    exact ⟨directSummaryExactAssumptionsEquivExactTheoremBacked.invFun h⟩

/-- Direct summary approximate assumptions are just the broadest approximate
interface under a route-specific name. -/
def directSummaryApproxAssumptionsEquivApproxTheoremBacked
    {g : Summarizer Strings} {T : BinTree Strings} {fstar : Strings → Y} :
    DirectSummaryApproxAssumptions g T fstar ≃ ApproxTheoremBacked g T fstar where
  toFun := fun h => ⟨h.approxLocalLaws⟩
  invFun := fun h => ⟨h.approxLocalLaws⟩
  left_inv := by intro h; cases h; rfl
  right_inv := by intro h; cases h; rfl

/-- Proposition-level form of `directSummaryApproxAssumptionsEquivApproxTheoremBacked`. -/
theorem directSummaryApproxAssumptions_nonempty_iff_approxTheoremBacked
    {g : Summarizer Strings} {T : BinTree Strings} {fstar : Strings → Y} :
    Nonempty (DirectSummaryApproxAssumptions g T fstar) ↔
      Nonempty (ApproxTheoremBacked g T fstar) := by
  constructor
  · intro h
    rcases h with ⟨h⟩
    exact ⟨directSummaryApproxAssumptionsEquivApproxTheoremBacked.toFun h⟩
  · intro h
    rcases h with ⟨h⟩
    exact ⟨directSummaryApproxAssumptionsEquivApproxTheoremBacked.invFun h⟩

namespace SketchCodecExactAssumptions

/-- Sketch / codec exact assumptions are a special case of the direct-summary
exact interface for the induced deterministic summarizer. -/
def toDirectSummaryExactAssumptions
    {op : SketchOperator Strings Sketch} {fstar : Strings → Y}
    (assumptions : SketchCodecExactAssumptions op fstar)
    (T : BinTree Strings) :
    DirectSummaryExactAssumptions (sketchSummarizer op) T fstar where
  localLaws := (assumptions.toExactTheoremBacked T).localLaws

/-- The exact sketch / codec route yields exact theorem-backedness uniformly over
all trees for the induced deterministic summarizer. -/
theorem exact_on_all_trees
    {op : SketchOperator Strings Sketch} {fstar : Strings → Y}
    (assumptions : SketchCodecExactAssumptions op fstar) :
    ∀ T : BinTree Strings, ExactTheoremBacked (sketchSummarizer op) T fstar := by
  intro T
  exact assumptions.toExactTheoremBacked T

end SketchCodecExactAssumptions

namespace SketchCodecApproxAssumptions

/-- Sketch / codec approximate assumptions are a special case of the direct-summary
approximate interface for the induced deterministic summarizer. -/
def toDirectSummaryApproxAssumptions
    {op : SketchOperator Strings Sketch} {T : BinTree Strings} {fstar : Strings → Y}
    (assumptions : SketchCodecApproxAssumptions op T fstar) :
    DirectSummaryApproxAssumptions (sketchSummarizer op) T fstar where
  approxLocalLaws := (assumptions.toApproxTheoremBacked).approxLocalLaws

end SketchCodecApproxAssumptions

/-- Global exact route factors through the direct-summary exact interface. -/
def directSummaryExactAssumptions_of_globalPreservation
    {g_det : Strings → Strings} {fstar : Strings → Y}
    [GlobalPreservation g_det fstar]
    (T : BinTree Strings) :
    DirectSummaryExactAssumptions (deterministicSummarizer g_det) T fstar where
  localLaws := (exactTheoremBacked_of_globalPreservation (g_det := g_det) (fstar := fstar) T).localLaws

/-- Exact theorem-backedness for a deterministic summary operator, uniformly over
all trees. -/
def ExactTheoremBackedAllTrees (g_det : Strings → Strings) (fstar : Strings → Y) : Prop :=
  ∀ T : BinTree Strings, Nonempty (ExactTheoremBacked (deterministicSummarizer g_det) T fstar)

/-- For deterministic theorem-domain summaries, exact theorem-backedness on all
trees collapses to `A1 ∧ A2` once `A3` is supplied. -/
theorem exactTheoremBackedAllTrees_iff_A1_A2_of_A3
    (g_det : Strings → Strings) (fstar : Strings → Y)
    (hA3 : A3_global g_det fstar) :
    ExactTheoremBackedAllTrees g_det fstar ↔
      (A1_global g_det fstar ∧ A2_global g_det fstar) := by
  constructor
  · intro hExact
    have hL1_all : ∀ T : BinTree Strings, L1 (deterministicSummarizer g_det) T fstar := by
      intro T
      exact (Classical.choice (hExact T)).localLaws.law1
    have hA1 : A1_global g_det fstar :=
      (A1_iff_L1_for_all_trees g_det fstar).2 hL1_all
    have hL2_all : ∀ T : BinTree Strings, L2 (deterministicSummarizer g_det) T fstar := by
      intro T
      exact (Classical.choice (hExact T)).localLaws.law2
    have hA2 : A2_global g_det fstar :=
      (A2_iff_L2_on_all_trees_of_A1_A3 g_det fstar hA1 hA3).2 hL2_all
    exact ⟨hA1, hA2⟩
  · intro hGlobal
    rcases hGlobal with ⟨hA1, hA2⟩
    intro T
    refine ⟨ExactTheoremBacked.ofLocalLaws ?_⟩
    refine ⟨?_, ?_, ?_⟩
    · exact A1_implies_L1 g_det fstar hA1 T
    · exact A1_A2_A3_implies_L2 g_det fstar hA1 hA2 hA3 T
    · exact A1_implies_L3 g_det fstar hA1

/-- Equivalent packaged form: exact theorem-backedness on all trees is exactly
the global `A1/A2/A3` regime once `A3` is included on both sides. -/
theorem exactTheoremBackedAllTrees_iff_globalAssumptions
    (g_det : Strings → Strings) (fstar : Strings → Y) :
    (ExactTheoremBackedAllTrees g_det fstar ∧ A3_global g_det fstar) ↔
      (A1_global g_det fstar ∧ A2_global g_det fstar ∧ A3_global g_det fstar) := by
  constructor
  · intro h
    rcases h with ⟨hExact, hA3⟩
    have hA12 :
        A1_global g_det fstar ∧ A2_global g_det fstar :=
      (exactTheoremBackedAllTrees_iff_A1_A2_of_A3 g_det fstar hA3).1 hExact
    exact ⟨hA12.1, hA12.2, hA3⟩
  · intro h
    rcases h with ⟨hA1, hA2, hA3⟩
    exact ⟨
      (exactTheoremBackedAllTrees_iff_A1_A2_of_A3 g_det fstar hA3).2 ⟨hA1, hA2⟩,
      hA3
    ⟩

/-- Under `A3`, exact theorem-backed deterministic summaries are classical
mergeable summaries. -/
theorem exactTheoremBackedAllTrees_implies_classical_mergeable
    (g_det : Strings → Strings) (fstar : Strings → Y)
    (hA3 : A3_global g_det fstar)
    (hExact : ExactTheoremBackedAllTrees g_det fstar) :
    IsMergeableSummary g_det fstar := by
  have hA12 :
      A1_global g_det fstar ∧ A2_global g_det fstar :=
    (exactTheoremBackedAllTrees_iff_A1_A2_of_A3 g_det fstar hA3).1 hExact
  exact ops_reduction_to_classical_mergeable g_det fstar hA12.1 hA12.2 hA3

/-- Exact sketch / codec assumptions are a special case of the broad exact
theorem-backed regime for deterministic summaries. -/
theorem sketchCodecExactAssumptions_imply_global_A1_A2
    {op : SketchOperator Strings Sketch} {fstar : Strings → Y}
    (assumptions : SketchCodecExactAssumptions op fstar)
    (hA3 : A3_global (summaryFromSketch op) fstar) :
    A1_global (summaryFromSketch op) fstar ∧ A2_global (summaryFromSketch op) fstar := by
  have hExact :
      ExactTheoremBackedAllTrees (summaryFromSketch op) fstar := by
    intro T
    exact ⟨by simpa [sketchSummarizer] using assumptions.exact_on_all_trees T⟩
  exact
    (exactTheoremBackedAllTrees_iff_A1_A2_of_A3
      (g_det := summaryFromSketch op) (fstar := fstar) hA3).1 hExact

/-- Under `A3`, exact sketch / codec assumptions induce the classical mergeable
summary interface on the induced deterministic summarizer. -/
theorem sketchCodecExactAssumptions_imply_classical_mergeable
    {op : SketchOperator Strings Sketch} {fstar : Strings → Y}
    (assumptions : SketchCodecExactAssumptions op fstar)
    (hA3 : A3_global (summaryFromSketch op) fstar) :
    IsMergeableSummary (summaryFromSketch op) fstar := by
  have hExact :
      ExactTheoremBackedAllTrees (summaryFromSketch op) fstar := by
    intro T
    exact ⟨by simpa [sketchSummarizer] using assumptions.exact_on_all_trees T⟩
  exact exactTheoremBackedAllTrees_implies_classical_mergeable
    (g_det := summaryFromSketch op) (fstar := fstar) hA3 hExact

end FormalProofs.OPT
