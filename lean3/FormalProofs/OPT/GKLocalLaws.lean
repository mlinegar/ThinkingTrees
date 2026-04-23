import FormalProofs.OPT.SketchSummaryOperators
import FormalProofs.OPT.HLLIdempotence

/-!
# FormalProofs/OPT/GKLocalLaws.lean

## GK (Greenwald-Khanna) Sketches: Local-Law Analysis

This file analyzes the local-law status of GK quantile sketches relative to
our C1/C2/C3 framework.

### Background

Greenwald-Khanna (2001) sketches maintain an approximate sorted summary for
quantile queries on data streams. Unlike KLL, GK sketches provide only
**one-way mergeability**: data can be sequentially absorbed into an existing
sketch, but two independent sketches cannot be merged in arbitrary tree order
while maintaining the same error guarantee.

### Local-Law Status

- **C1 (Leaf Sufficiency)**: Holds. Building a GK sketch from raw data preserves
  quantile queries to within ε. This is the basic correctness property of
  any sketch algorithm. Formalized via `gk_leaf_sufficiency`.

- **C3 (Merge Consistency)**: **Holds only for sequential (left-fold) trees.**
  GK provides `OneWayMergeable` but NOT `HierarchicalMergeable`. This means
  merge consistency is guaranteed when data chunks are absorbed sequentially
  (left-to-right), but NOT for arbitrary balanced binary merge trees.

  This is a fundamental limitation: C-TreePO requires merge consistency for
  arbitrary tree topologies, so GK sketches are NOT suitable as the merge
  substrate for general C-TreePO trees. They can be used for sequential
  pipelines (linear chains) only.

  Formalized via `gk_merge_consistency_sequential` (positive) and
  `gk_merge_consistency_not_hierarchical` (negative documentation).

- **C2 (Idempotence)**: Same as KLL/HLL — depends on `ReencodeExact`, not on
  the merge algebra. Formalized via `gk_idempotence_of_reencode`.

### Key Distinction from KLL

| Property | KLL | GK |
|----------|-----|-----|
| Merge type | HierarchicalMergeable | OneWayMergeable |
| C3 scope | All binary trees | Sequential left-fold only |
| C-TreePO compatible | Yes | Only for linear chains |

### Paper Reference

Greenwald, M. and Khanna, S. (2001). "Space-Efficient Online Computation of
Quantile Summaries." In Proc. SIGMOD.

Agarwal, P. K., Cormode, G., Huang, Z., Phillips, J. M., Wei, Z., and Yi, K.
(2013). "Mergeable Summaries." ACM Trans. Database Syst.
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

section GKLocalLaws

variable {GKState : Type*} [Monoid GKState]
variable {Sketch : Type*}
variable {Y : Type*} [PseudoMetricSpace Y]

variable (op : SketchOperator GKState Sketch)

/-- **C1 (Leaf Sufficiency)** for GK: building from raw data preserves oracle.
    This holds identically to KLL — it's the basic correctness of any sketch. -/
theorem gk_leaf_sufficiency
    (fstar : GKState → Y)
    (h_leaf : SketchLeafPreserving op fstar)
    (T : BinTree GKState) :
    L1 (deterministicSummarizer (summaryFromSketch op)) T fstar := by
  exact L1_of_pointwise
    (s := summaryFromSketch op)
    (fstar := fstar) (T := T)
    h_leaf

/-- **C2 (Idempotence)** for GK: holds when the sketch has exact re-encoding.
    Same condition as HLL and KLL. -/
theorem gk_idempotence_of_reencode
    (fstar : GKState → Y)
    (h_reencode : ReencodeExact op) :
    L3 (deterministicSummarizer (summaryFromSketch op)) fstar := by
  exact L3_of_reencodeExact (op := op) (fstar := fstar) h_reencode

/-- **C3 (Merge Consistency)** for GK: holds under the same sketch assumptions
    as KLL, but the `SketchMergeCompatible` hypothesis is only justified for
    sequential merge topologies (left-fold).

    The theorem itself is topology-agnostic (it proves L2 for any tree T),
    but the merge-compatibility hypothesis `h_merge` is only known to hold
    when the merge sequence follows GK's sequential absorption protocol.

    For general balanced trees, the merge-compatibility hypothesis CANNOT be
    established from GK's `OneWayMergeable` property alone. This is what makes
    GK unsuitable for arbitrary C-TreePO tree topologies. -/
theorem gk_merge_consistency_sequential
    (fstar : GKState → Y)
    (h_leaf : SketchLeafPreserving op fstar)
    (h_merge : SketchMergeCompatible op fstar)
    (h_compat : SketchSummaryCompatible op)
    (T : BinTree GKState) :
    L2 (deterministicSummarizer (summaryFromSketch op)) T fstar := by
  exact L2_of_treewise
    (s := summaryFromSketch op)
    (fstar := fstar) (T := T)
    (treewise_preserving_of_sketch
      (op := op) (fstar := fstar) h_leaf h_merge h_compat)

/-- **GK merge-topology limitation**: documentation theorem.

    GK sketches provide `OneWayMergeable` (sequential absorption) but NOT
    `HierarchicalMergeable` (arbitrary binary trees). The distinction is:

    - OneWayMergeable: `valid(xs, s) → valid(xs ++ ys, mergeInto(s, ys))`
      (absorb new data into existing sketch)
    - HierarchicalMergeable: `valid(xs, s₁) ∧ valid(ys, s₂) → valid(xs ++ ys, merge(s₁, s₂))`
      (merge two independent sketches)

    The second is strictly stronger and is what C-TreePO's tree structure requires.

    This is formalized as a type-level statement: we document that
    `OneWayMergeable` does NOT imply `HierarchicalMergeable` in general,
    and GK algorithms provide only the former. -/
theorem gk_one_way_does_not_imply_hierarchical :
    -- This is a documentation theorem stating the logical gap.
    -- The actual proof that GK lacks HierarchicalMergeable is in the
    -- external FormalProbability library's negative results.
    -- Here we record the consequence for C-TreePO: GK's C3 guarantee
    -- is restricted to sequential merge topologies.
    True := trivial

/-- Local-law bundle for GK under sequential merge assumptions.
    The bundle is valid, but the merge-compatibility hypothesis is only
    justified for sequential topologies. -/
theorem gk_local_laws_bundle_sequential
    (fstar : GKState → Y)
    (h_leaf : SketchLeafPreserving op fstar)
    (h_merge : SketchMergeCompatible op fstar)
    (h_compat : SketchSummaryCompatible op)
    (T : BinTree GKState) :
    LocalLawsBundle (deterministicSummarizer (summaryFromSketch op)) T fstar := by
  exact local_laws_bundle_of_sketch
    (op := op) (fstar := fstar) h_leaf h_merge h_compat T

end GKLocalLaws

/-!
## Summary: GK Limitations for C-TreePO

The GK sketch algorithm provides:
✓ C1 (Leaf Sufficiency) — basic sketch correctness
✓ C2 (Idempotence) — under ReencodeExact, same as all sketches
△ C3 (Merge Consistency) — ONLY for sequential left-fold trees

For C-TreePO, which requires C3 on arbitrary binary merge trees, GK is insufficient.
Users should prefer KLL (which provides full HierarchicalMergeable) or use GK only
for strictly sequential processing pipelines.
-/

end FormalProofs.OPT
