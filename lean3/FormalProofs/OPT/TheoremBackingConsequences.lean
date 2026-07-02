import FormalProofs.OPT.HLLIdempotence
import FormalProofs.OPT.SketchRecovery
import FormalProofs.OPT.GlobalAssumptions
import FormalProofs.OPT.MergeableReduction
import FormalProofs.OPT.ExpectationTheory
import FormalProofs.OPT.PreferenceLearning

/-!
# FormalProofs/OPT/TheoremBackingConsequences.lean

Consolidated 2026-07-02: this file now also contains the verbatim content of
`TheoremBackingAssumptions.lean` and `TheoremBackingStructure.lean` (which sat
directly below it in the import chain); the two upper-cluster files
(`TheoremBackingMeasurementError`, `TheoremBackingApproxMeasurementError`) live
in `TheoremBacking.lean`, which imports this file. NOTE: this file must never
import `FormalProofs.OPT.TheoremBacking` (cycle).
-/

/-! ## From TheoremBackingAssumptions.lean (consolidated 2026-07-02) -/
section
/-!
# FormalProofs/OPT/TheoremBackingAssumptions.lean

Reusable exact and approximate assumption bundles for theorem-backed operators.

This file makes explicit a distinction that is implicit in the rest of the OPT
development:

- the **broadest exact sufficient interface** is simply a `LocalLawsBundle`;
- the **broadest approximate sufficient interface** is simply an
  `ApproxLocalLawsBundle`;
- direct theorem-domain summaries (e.g. text summaries) use those interfaces
  directly;
- sketch / codec operators discharge them through `SketchLeafPreserving`,
  `SketchMergeCompatible`, `SketchSummaryCompatible`, or their approximate
  budgeted analogues;
- the stronger global route `A1/A2/A3` also compiles down to the same exact
  local-law bundle.
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

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]
variable {Sketch : Type*}

/-- The broadest exact sufficient interface currently formalized: an operator is
theorem-backed once its induced summarizer carries a `LocalLawsBundle`. -/
structure ExactTheoremBacked (g : Summarizer Strings) (T : BinTree Strings)
    (fstar : Strings → Y) where
  localLaws : LocalLawsBundle g T fstar

namespace ExactTheoremBacked

/-- Direct exact theorem-backedness from a `LocalLawsBundle`. -/
def ofLocalLaws
    {g : Summarizer Strings} {T : BinTree Strings} {fstar : Strings → Y}
    (laws : LocalLawsBundle g T fstar) :
    ExactTheoremBacked g T fstar where
  localLaws := laws

end ExactTheoremBacked

/-- The broadest approximate sufficient interface currently formalized: an
operator is approximately theorem-backed once its induced summarizer carries an
`ApproxLocalLawsBundle`. -/
structure ApproxTheoremBacked (g : Summarizer Strings) (T : BinTree Strings)
    (fstar : Strings → Y) where
  approxLocalLaws : ApproxLocalLawsBundle g T fstar

namespace ApproxTheoremBacked

/-- Direct approximate theorem-backedness from an `ApproxLocalLawsBundle`. -/
def ofApproxLocalLaws
    {g : Summarizer Strings} {T : BinTree Strings} {fstar : Strings → Y}
    (laws : ApproxLocalLawsBundle g T fstar) :
    ApproxTheoremBacked g T fstar where
  approxLocalLaws := laws

end ApproxTheoremBacked

/-- Exact theorem-domain route: for a direct summary operator, giving the local
laws themselves is enough. -/
structure DirectSummaryExactAssumptions (g : Summarizer Strings) (T : BinTree Strings)
    (fstar : Strings → Y) where
  localLaws : LocalLawsBundle g T fstar

namespace DirectSummaryExactAssumptions

/-- Direct summary exact assumptions compile immediately to exact theorem-backedness. -/
def toExactTheoremBacked
    {g : Summarizer Strings} {T : BinTree Strings} {fstar : Strings → Y}
    (assumptions : DirectSummaryExactAssumptions g T fstar) :
    ExactTheoremBacked g T fstar where
  localLaws := assumptions.localLaws

end DirectSummaryExactAssumptions

/-- Approximate theorem-domain route: for a direct summary operator, an
approximate local-law bundle is enough. -/
structure DirectSummaryApproxAssumptions (g : Summarizer Strings) (T : BinTree Strings)
    (fstar : Strings → Y) where
  approxLocalLaws : ApproxLocalLawsBundle g T fstar

namespace DirectSummaryApproxAssumptions

/-- Direct summary approximate assumptions compile immediately to approximate theorem-backedness. -/
def toApproxTheoremBacked
    {g : Summarizer Strings} {T : BinTree Strings} {fstar : Strings → Y}
    (assumptions : DirectSummaryApproxAssumptions g T fstar) :
    ApproxTheoremBacked g T fstar where
  approxLocalLaws := assumptions.approxLocalLaws

end DirectSummaryApproxAssumptions

/-- Exact sketch / codec route: the standard sufficient assumptions for a
supplied encode/merge/decode operator. -/
structure SketchCodecExactAssumptions (op : SketchOperator Strings Sketch)
    (fstar : Strings → Y) where
  leaf : SketchLeafPreserving op fstar
  merge : SketchMergeCompatible op fstar
  compat : SketchSummaryCompatible op

namespace SketchCodecExactAssumptions

/-- Exact sketch / codec assumptions induce exact theorem-backedness for the
induced deterministic summarizer. -/
def toExactTheoremBacked
    {op : SketchOperator Strings Sketch} {fstar : Strings → Y}
    (assumptions : SketchCodecExactAssumptions op fstar)
    (T : BinTree Strings) :
    ExactTheoremBacked (sketchSummarizer op) T fstar where
  localLaws :=
    local_laws_of_sketch
      (op := op) (fstar := fstar) (T := T)
      assumptions.leaf assumptions.merge assumptions.compat

end SketchCodecExactAssumptions

/-- Approximate sketch / codec route: leaf and merge nodewise budgets plus an
idempotence budget on the induced summarizer. -/
structure SketchCodecApproxAssumptions (op : SketchOperator Strings Sketch)
    (T : BinTree Strings) (fstar : Strings → Y) where
  epsLeaf : Strings → ℝ
  epsMerge : BinTree Strings × BinTree Strings → ℝ
  epsIdemp : ℝ
  leaf : SketchLeafApproxPreserving op fstar epsLeaf
  merge : SketchMergeApproxCompatible op fstar epsMerge
  idemp : L3ε (sketchSummarizer op) T fstar epsIdemp

namespace SketchCodecApproxAssumptions

/-- Approximate sketch / codec assumptions induce approximate theorem-backedness
for the induced deterministic summarizer. -/
def toApproxTheoremBacked
    {op : SketchOperator Strings Sketch} {T : BinTree Strings} {fstar : Strings → Y}
    (assumptions : SketchCodecApproxAssumptions op T fstar) :
    ApproxTheoremBacked (sketchSummarizer op) T fstar where
  approxLocalLaws :=
    approx_bundle_of_sketch
      (op := op) (fstar := fstar) (T := T)
      (ε_leaf := assumptions.epsLeaf)
      (ε_merge := assumptions.epsMerge)
      (ε_idemp := assumptions.epsIdemp)
      assumptions.leaf assumptions.merge assumptions.idemp

end SketchCodecApproxAssumptions

/-- Stronger exact route: a global `A1/A2/A3` witness gives theorem-backedness
for any tree automatically. -/
def exactTheoremBacked_of_globalPreservation
    {g_det : Strings → Strings} {fstar : Strings → Y}
    [inst : GlobalPreservation g_det fstar]
    (T : BinTree Strings) :
    ExactTheoremBacked (deterministicSummarizer g_det) T fstar where
  localLaws := by
    simpa [deterministicSummarizer] using
      (GlobalPreservation.toLocalLawsBundle
        (g_det := g_det) (fstar := fstar) (T := T))

end FormalProofs.OPT

end -- closes the source file's dangling `noncomputable section`
end -- closes the consolidation wrapper section

/-! ## From TheoremBackingStructure.lean (consolidated 2026-07-02) -/
section
/-!
# FormalProofs/OPT/TheoremBackingStructure.lean

Structural consequences of the theorem-backing interfaces.

This file packages three key points.

1. The broadest exact interface admits an "on-support" characterization:
   exact theorem-backedness means every realized leaf summary, internal-node
   reduction, and in-range resummary is oracle-exact on support.
2. For deterministic theorem-domain summaries, exact theorem-backedness on
   **all trees** plus strict oracle-output `A3` collapses to the global
   `A1/A2/A3` regime.
3. Sketch / codec assumptions are explicit special cases of the broadest direct
   interfaces; with strict `A3` they induce the oracle-level mergeable view,
   while the state-level sketch route is handled in `MergeableReduction`.
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

/-- Under strict oracle-output `A3`, exact theorem-backed deterministic
summaries are oracle-level mergeable. -/
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

/-- Under strict oracle-output `A3`, exact sketch / codec assumptions induce
the oracle-level mergeable interface on the induced deterministic summarizer. -/
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

end -- closes the source file's dangling `noncomputable section`
end -- closes the consolidation wrapper section

/-! ## Original TheoremBackingConsequences content -/
/-!
# FormalProofs/OPT/TheoremBackingConsequences.lean

Consequences of exact theorem-backedness for multi-round reduction and downstream
objective equivalence.

This file makes explicit a chain that was already latent in the development:

1. `ExactTheoremBacked` packages `L1/L2/L3`.
2. `L1/L2/L3` imply exact multi-round zero distortion on `ZR`.
3. Zero distortion implies equality of any oracle-measurable expected loss,
   including DPO, GRPO, GRPO-RL, and compositional preference programs.

The boundary is important: these consequences apply to objectives indexed by the
same oracle `fstar`. Utilities that depend on a richer exact latent state are
handled separately by `ExactUtilityTransport.lean`.
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

open Set

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [BoundedMetricSpace Y]

/-- If the expected oracle distortion under a PMF is zero, then distortion is zero
at every support point of that PMF. -/
lemma dist_zero_on_support_of_Exp_zero
    (p : PMF Strings) (fstar : Strings → Y) (x : Strings)
    (h_exp_zero : Exp p (fun z => D fstar z x) = 0) :
    ∀ z ∈ p.support, dist (fstar z) (fstar x) = 0 := by
  let M : ℝ := BoundedMetricSpace.diameterBound (α := Y)
  have hM : 0 ≤ M := BoundedMetricSpace.diameterBound_nonneg (α := Y)
  have hbound : ∀ z, D fstar z x ≤ M := by
    intro z
    simpa [M, D] using (BoundedMetricSpace.dist_le (fstar z) (fstar x))
  have h_summable : Summable (fun z => (p z).toReal * D fstar z x) :=
    summable_D_of_bounded p fstar x M hM hbound
  have h_term_zero : ∀ z, (p z).toReal * D fstar z x = 0 :=
    tsum_eq_zero_of_nonneg
      (fun z => (p z).toReal * D fstar z x)
      (fun z => mul_nonneg ENNReal.toReal_nonneg dist_nonneg)
      h_summable
      (by simpa [Exp] using h_exp_zero)
  intro z hz
  have hz_ne0 : p z ≠ 0 := by
    simpa [PMF.mem_support_iff] using hz
  have hz_toReal_pos : 0 < (p z).toReal :=
    ENNReal.toReal_pos hz_ne0 (PMF.apply_ne_top p z)
  have hz_mul : (p z).toReal * D fstar z x = 0 := h_term_zero z
  rcases mul_eq_zero.mp hz_mul with hz_toReal | hz_dist
  · exfalso
    exact (ne_of_gt hz_toReal_pos) hz_toReal
  · simpa [D] using hz_dist

/-- Exact theorem-backedness implies supportwise zero distortion for the multi-round
reduction distribution `ZR`. -/
theorem zero_distortion_on_ZR_support_of_exactTheoremBacked
    {g : Summarizer Strings} {T : BinTree Strings} {x : Strings} {R : ℕ}
    {fstar : Strings → Y}
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1) :
    ∀ z ∈ (ZR g x R T).support, dist (fstar z) (fstar x) = 0 := by
  have h_exp_zero : Exp (ZR g x R T) (fun z => D fstar z x) = 0 :=
    multi_round_typeclass g T x R fstar hp
      hExact.localLaws.law1 hExact.localLaws.law2 hExact.localLaws.law3 hR
  exact dist_zero_on_support_of_Exp_zero (p := ZR g x R T) (fstar := fstar) (x := x) h_exp_zero

/-- Exact theorem-backedness implies the multi-round zero-distortion theorem. -/
theorem multi_round_of_exactTheoremBacked
    {g : Summarizer Strings} {T : BinTree Strings} {x : Strings} {R : ℕ}
    {fstar : Strings → Y}
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1) :
    Exp (ZR g x R T) (fun z => D fstar z x) = 0 := by
  exact multi_round_typeclass g T x R fstar hp
    hExact.localLaws.law1 hExact.localLaws.law2 hExact.localLaws.law3 hR

/-- Exact theorem-backedness is enough for any oracle-measurable expected-loss
objective on `PMF.pure x` versus the multi-round reduction `ZR`. -/
theorem expected_loss_eq_via_ZR_of_exactTheoremBacked
    {α : Type*}
    (fstar : Strings → Y)
    (loss : Strings → α → ℝ)
    (gen : Strings → PMF α)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (h_meas : OracleMeasurableLossGeneric loss fstar)
    (h_gen : OracleIndexedGenGeneric gen fstar) :
    ExpectedLossGeneric loss (PMF.pure x) gen =
    ExpectedLossGeneric loss (ZR g x R T) gen := by
  exact expected_loss_eq_of_zero_dist_generic fstar loss gen (PMF.pure x) (ZR g x R T)
    (by
      intro z x' hz hx'
      simp only [PMF.support_pure, mem_singleton_iff] at hx'
      subst hx'
      exact zero_distortion_on_ZR_support_of_exactTheoremBacked hp hExact hR z hz)
    h_meas
    h_gen

/-- Exact theorem-backedness is enough for any compositional preference loss on
`PMF.pure x` versus `ZR`. -/
theorem expected_pref_loss_eq_via_ZR_of_exactTheoremBacked
    {α : Type*}
    (fstar : Strings → Y)
    (loss : PrefLoss Strings α)
    (gen : PrefGen Strings α)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (h_meas : OracleMeasurablePrefLoss loss fstar)
    (h_gen : OracleIndexedGenComb gen fstar) :
    ExpectedPrefLoss loss (PMF.pure x) gen =
    ExpectedPrefLoss loss (ZR g x R T) gen := by
  exact expected_pref_loss_eq_of_zero_dist (fstar := fstar)
    (loss := loss) (gen := gen) (μ_X := PMF.pure x) (μ_Z := ZR g x R T)
    (by
      intro z x' hz hx'
      simp only [PMF.support_pure, mem_singleton_iff] at hx'
      subst hx'
      exact zero_distortion_on_ZR_support_of_exactTheoremBacked hp hExact hR z hz)
    h_meas
    h_gen

/-- Exact theorem-backedness is enough for nested preference programs built from
oracle-indexed preference samplers. -/
theorem expected_pref_loss_prog_eq_via_ZR_of_exactTheoremBacked
    {α : Type*}
    (fstar : Strings → Y)
    (loss : PrefLoss Strings α)
    (prog : PrefProgram Strings α)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (h_meas : OracleMeasurablePrefLoss loss fstar)
    (h_prog : OracleIndexedProgram fstar prog) :
    ExpectedPrefLossProg loss (PMF.pure x) prog =
    ExpectedPrefLossProg loss (ZR g x R T) prog := by
  exact expected_pref_loss_prog_eq_of_zero_dist (fstar := fstar)
    (loss := loss) (μ_X := PMF.pure x) (μ_Z := ZR g x R T) (prog := prog)
    (by
      intro z x' hz hx'
      simp only [PMF.support_pure, mem_singleton_iff] at hx'
      subst hx'
      exact zero_distortion_on_ZR_support_of_exactTheoremBacked hp hExact hR z hz)
    h_meas
    h_prog

section DPO

variable {A : Type*}

/-- DPO expected-loss equivalence follows directly from exact theorem-backedness. -/
theorem dpo_equivalence_via_ZR_of_exactTheoremBacked
    (fstar : Strings → Y)
    (pol pol_ref : Policy Strings A) (β : ℝ)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (h_meas_pol : DPO.OracleMeasurable pol fstar)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_gen : OracleIndexedPairGen gen fstar) :
    ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen =
    ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen := by
  exact expected_loss_eq_of_zero_dist fstar pol pol_ref gen (PMF.pure x) (ZR g x R T) β
    (by
      intro z x' hz hx'
      simp only [PMF.support_pure, mem_singleton_iff] at hx'
      subst hx'
      exact zero_distortion_on_ZR_support_of_exactTheoremBacked hp hExact hR z hz)
    h_meas_pol
    h_meas_ref
    h_gen

/-- DPO exact argmin preservation over `PMF.pure x` versus `ZR` follows from exact
theorem-backedness. -/
theorem dpo_exact_metric_via_ZR_of_exactTheoremBacked
    (fstar : Strings → Y)
    (pol_ref : Policy Strings A)
    (gen : PairGenerator Strings A)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (β : ℝ)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (h_meas_ref : DPO.OracleMeasurable pol_ref fstar)
    (h_gen : OracleIndexedPairGen gen fstar) :
    SameOracleMeasurableArgmin
      (fun pol => ExpectedDPOLoss pol pol_ref β (PMF.pure x) gen)
      (fun pol => ExpectedDPOLoss pol pol_ref β (ZR g x R T) gen)
      fstar := by
  exact dpo_exact_metric fstar pol_ref gen (PMF.pure x) (ZR g x R T) β
    (by
      intro z x' hz hx'
      simp only [PMF.support_pure, mem_singleton_iff] at hx'
      subst hx'
      exact zero_distortion_on_ZR_support_of_exactTheoremBacked hp hExact hR z hz)
    h_meas_ref
    h_gen

end DPO

section GRPO

variable {A : Type*}
variable {k : ℕ}

/-- GRPO-Plackett-Luce expected-loss equivalence follows directly from exact
theorem-backedness. -/
theorem grpo_equivalence_via_ZR_of_exactTheoremBacked
    (fstar : Strings → Y)
    (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (h_pol : GRPOOracleMeasurable (Y := Y) pol fstar)
    (h_ranker : OracleIndexedRanker (Y := Y) ranker fstar)
    (h_gen : OracleIndexedGroupGen (Y := Y) gen fstar) :
    ExpectedGRPOLoss pol ranker (PMF.pure x) gen =
    ExpectedGRPOLoss pol ranker (ZR g x R T) gen := by
  exact grpo_equivalence (Y := Y) fstar pol ranker gen (PMF.pure x) (ZR g x R T)
    (by
      intro z x' hz hx'
      simp only [PMF.support_pure, mem_singleton_iff] at hx'
      subst hx'
      exact zero_distortion_on_ZR_support_of_exactTheoremBacked hp hExact hR z hz)
    h_pol
    h_ranker
    h_gen

/-- GRPO-RL expected-loss equivalence follows directly from exact theorem-backedness. -/
theorem grpo_rl_equivalence_via_ZR_of_exactTheoremBacked
    (fstar : Strings → Y)
    (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ) (eps beta : ℝ)
    (gen : GroupGenerator Strings A k)
    (g : Summarizer Strings) (x : Strings) (R : ℕ) (T : BinTree Strings)
    (hp : S T = x)
    (hExact : ExactTheoremBacked g T fstar)
    (hR : R ≥ 1)
    (h_meas : OracleMeasurableGRPORLLoss k pol pol_old pol_ref reward eps beta fstar)
    (h_gen : OracleIndexedGroupGen (Y := Y) gen fstar) :
    ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (PMF.pure x) gen =
    ExpectedGRPORLLoss pol pol_old pol_ref reward eps beta (ZR g x R T) gen := by
  exact grpo_rl_equivalence (Y := Y) (k := k) fstar pol pol_old pol_ref reward eps beta gen
    (PMF.pure x) (ZR g x R T)
    (by
      intro z x' hz hx'
      simp only [PMF.support_pure, mem_singleton_iff] at hx'
      subst hx'
      exact zero_distortion_on_ZR_support_of_exactTheoremBacked hp hExact hR z hz)
    h_meas
    h_gen

end GRPO

end FormalProofs.OPT
