import FormalProofs.OPT.HLLIdempotence
import FormalProofs.OPT.SketchRecovery
import FormalProofs.OPT.GlobalAssumptions

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
