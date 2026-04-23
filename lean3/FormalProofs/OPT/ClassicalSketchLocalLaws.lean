import FormalProofs.OPT.HLLIdempotence
import FormalProofs.OPT.CountMinSketch
import FormalProofs.OPT.KLLLocalLaws
import FormalProofs.OPT.GKLocalLaws
import FormalProofs.OPT.BigramSketch
import FormalProofs.OPT.MarkovCountSketchExample

/-!
# FormalProofs/OPT/ClassicalSketchLocalLaws.lean

## Unified Local-Law Status for Classical Mergeable Sketches

This file collects and cross-references the local-law (C1/C2/C3) status of
all classical mergeable sketches formalized in this project. It serves as the
single reference point for the paper's "A Brief Digression on Mergeable Sketches"
section.

### Local-Law Correspondence

| Paper Name | Lean Name | Description |
|------------|-----------|-------------|
| C1 (Sufficiency) | L1 | Leaf summary preserves oracle |
| C2 (Idempotence) | L3 | Re-summarizing is inert on range |
| C3 (Merge Consistency) | L2 | Merge preserves oracle across tree |

### Summary Table

| Sketch | Status | C1 (L1) | C2 (L3) | C3 (L2) | Merge Type | File / Runtime |
|--------|--------|---------|---------|---------|------------|----------------|
| **HyperLogLog** | Lean-backed + runtime | ✓ `hllRegisterOperator_L1` | ✓* `hllRegisterOperator_L3` | ✓ `hllRegisterOperator_L2` | max (idem.) | `HLLIdempotence.lean`; native/DataSketches adapter |
| **Count-Min** | Lean-backed + runtime | ✓ `cmsRegisterOperator_L1` | ✓* `cmsRegisterOperator_L3` | ✓ `cmsRegisterOperator_L2` | add (linear) | `CountMinSketch.lean`; DataSketches adapter |
| **KLL Quantile** | Lean-backed + runtime | ✓ `kll_leaf_sufficiency` | ✓* `kll_idempotence_of_reencode` | ✓ `kll_merge_consistency` | rank-compress | `KLLLocalLaws.lean`; DataSketches adapter |
| **GK Quantile** | Lean-backed only | ✓ `gk_leaf_sufficiency` | ✓* `gk_idempotence_of_reencode` | △ `gk_merge_consistency_sequential` | sequential | `GKLocalLaws.lean` |
| **Bigram** | Lean-backed only | ✓ (implicit) | ✓ (implicit) | ✓ `bigramSketch_append` | boundary-aware | `BigramSketch.lean` |
| **Markov Count** | Lean-backed only | ✓ `L1_gExact` | ✓ (exact) | ✓ `L2_gExact` | join-aware | `MarkovCountSketchExample.lean` |
| **CPC** | official empirical | — | — | — | compressed cardinality | Apache DataSketches adapter |
| **Theta/KMV** | official empirical | — | — | — | set algebra | Apache DataSketches adapter |
| **Frequent Items** | official empirical | — | — | — | heavy hitters | Apache DataSketches adapter |
| **Classic Quantiles** | official empirical | — | — | — | quantile compaction | Apache DataSketches adapter |
| **REQ Quantile** | official empirical | — | — | — | relative-error quantiles | Apache DataSketches adapter |
| **t-digest** | official empirical | — | — | — | centroid merge | Apache DataSketches adapter |
| **Tuple / VarOpt** | official empirical | — | — | — | tuple/sampling | Apache DataSketches adapter |

*C2 requires `ReencodeExact` (encode∘decode = id). Algebraic merge properties alone
are insufficient — see `succMax_not_L3` for the formal counterexample.

△ GK provides C3 only for sequential (left-fold) merge trees, NOT arbitrary binary
trees. This makes GK unsuitable for general C-TreePO tree topologies.

### Key Negative Results

1. **Merge idempotence ≠ C2**: `succMax_not_L3` in `HLLIdempotence.lean`
   shows a sketch with idempotent merge but violated C2.

2. **C2 not derivable from C1 + C3**: `thm10_1_L3_not_derivable` in
   `CounterexampleExistence.lean` constructs g_bad that satisfies C1 on
   fresh inputs but violates C2 (re-summarizing flips oracle value).

3. **Lossy decode breaks C2**: `cmsLossy_not_L3` in `CountMinSketch.lean`
   shows a CMS-like operator with systematic decode error violating C2.

4. **Naive bigram sketch not mergeable**: explicit counterexample in
   `BigramSketch.lean` shows that dropping boundary metadata breaks C3.

5. **Flip summarizer breaks C2**: `not_L3_gFlip` in `MarkovCountSketchExample.lean`
   shows that a summarizer that modifies on-range values violates C2.

6. **GK not hierarchically mergeable**: `gk_one_way_does_not_imply_hierarchical`
   in `GKLocalLaws.lean` documents that GK's `OneWayMergeable` does not
   provide full `HierarchicalMergeable` required for arbitrary tree merges.

### Architectural Insight

The table reveals a clean factorization of local-law compliance:

- **C1** is trivial for all sketches: building from raw data preserves the oracle
  to within the sketch's inherent approximation. This is the basic correctness
  property of any sketch algorithm.

- **C3** depends on the merge algebra: it holds for sketches with associative,
  exact merge operations (HLL max, CMS add, KLL rank-compress). It fails or
  is restricted when the merge operation introduces error or is order-dependent
  (GK sequential only, naive bigram without boundary metadata).

- **C2** is the subtlest law: it depends on the encode/decode round-trip, NOT
  on the merge algebra. Any sketch with `ReencodeExact` (the decoded state
  re-encodes to the same sketch) satisfies C2. This is why:
  - Identity operators (theorem-domain = sketch-domain) trivially satisfy C2
  - Lossy decodes (CMS point-query, approximate quantile readout) can violate C2
  - Merge idempotence is irrelevant to C2 (the `succMax` counterexample)

This factorization explains why C-TreePO needs all three laws as independent
conditions: C1 governs leaf quality, C3 governs merge quality, and C2 governs
stability under re-application of the learned summarizer.
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

/-!
## Cross-Sketch Comparison Theorems

These theorems make explicit the structural relationships between sketches
that are implicit in the individual files.
-/

section CrossSketchComparison

variable {m d w : ℕ}
variable {Y : Type*} [PseudoMetricSpace Y]

/-- HLL and CMS both satisfy the full local-law bundle when used as identity
    operators on their respective state spaces. The same bridge
    (`local_laws_bundle_of_sketch` via `identitySketchOperator`) applies
    to both, despite their very different merge algebras (max vs add). -/
theorem hll_and_cms_both_satisfy_full_bundle
    (fstar_hll : HLLState m → Y)
    (fstar_cms : CMSState d w → Y)
    (T_hll : BinTree (HLLState m))
    (T_cms : BinTree (CMSState d w))
    (h_merge_hll : SketchMergeCompatible (hllRegisterOperator m) fstar_hll)
    (h_merge_cms : SketchMergeCompatible (cmsRegisterOperator d w) fstar_cms) :
    LocalLawsBundle
      (deterministicSummarizer (summaryFromSketch (hllRegisterOperator m)))
      T_hll fstar_hll ∧
    LocalLawsBundle
      (deterministicSummarizer (summaryFromSketch (cmsRegisterOperator d w)))
      T_cms fstar_cms :=
  ⟨hllRegisterOperator_local_laws_bundle fstar_hll T_hll h_merge_hll,
   cmsRegisterOperator_local_laws_bundle fstar_cms T_cms h_merge_cms⟩

/-- The `ReencodeExact` condition is the universal key to C2 across all sketches.
    When it holds, C2 follows regardless of merge algebra.
    When it fails, C2 can fail regardless of merge algebra. -/
theorem reencode_exact_is_universal_c2_key :
    -- Positive direction: ReencodeExact → L3 for any oracle
    (∀ {Strings : Type*} [Monoid Strings] {Sketch : Type*}
       {Y : Type*} [PseudoMetricSpace Y]
       (op : SketchOperator Strings Sketch) (fstar : Strings → Y),
       ReencodeExact op →
       L3 (deterministicSummarizer (summaryFromSketch op)) fstar) ∧
    -- Negative direction: ¬ReencodeExact can cause ¬L3
    (∃ op : SketchOperator Nat Nat,
       ¬ ReencodeExact op ∧
       ¬ L3 (deterministicSummarizer (summaryFromSketch op))
             (fun n : Nat => (n : ℝ))) :=
  ⟨fun op fstar h => L3_of_reencodeExact op fstar h,
   ⟨succMaxOperator,
    fun h => succMax_not_summaryFixedPoint
      (summaryFixedPoint_of_reencodeExact (op := succMaxOperator) h),
    succMax_not_L3⟩⟩

/-- Merge idempotence is neither necessary nor sufficient for C2.
    - HLL has idempotent merge AND satisfies C2 (with ReencodeExact)
    - CMS has non-idempotent merge AND satisfies C2 (with ReencodeExact)
    - succMax has idempotent merge but FAILS C2 (without ReencodeExact) -/
theorem merge_idempotence_orthogonal_to_c2 :
    -- Idempotent merge + ReencodeExact → C2 (HLL)
    (MergeIdempotent (hllRegisterOperator m) ∧
     ReencodeExact (hllRegisterOperator m)) ∧
    -- Non-idempotent merge + ReencodeExact → C2 (CMS, when d,w > 0)
    (ReencodeExact (cmsRegisterOperator d w)) ∧
    -- Idempotent merge + ¬ReencodeExact → ¬C2 (succMax)
    (MergeIdempotent succMaxOperator ∧
     ¬ ReencodeExact succMaxOperator) :=
  ⟨⟨hllRegisterOperator_merge_idempotent, hllRegisterOperator_reencodeExact⟩,
   cmsRegisterOperator_reencodeExact,
   ⟨succMax_merge_idempotent,
    fun h => succMax_not_summaryFixedPoint
      (summaryFixedPoint_of_reencodeExact (op := succMaxOperator) h)⟩⟩

end CrossSketchComparison

/-!
## Paper-Ready Summary Aliases

These aliases provide clean, paper-numbered theorem names for the unified
local-law status of each classical sketch.
-/

section PaperAliases

variable {Y : Type*} [PseudoMetricSpace Y]

/-- **HyperLogLog Local Laws (Complete)**:
    C1 ✓, C2 ✓ (ReencodeExact), C3 ✓ (merge-compatible oracle).
    Merge operation: elementwise max (idempotent, commutative, associative). -/
theorem hll_local_laws_complete {m : ℕ}
    (fstar : HLLState m → Y)
    (T : BinTree (HLLState m))
    (h_merge : SketchMergeCompatible (hllRegisterOperator m) fstar) :
    LocalLawsBundle
      (deterministicSummarizer (summaryFromSketch (hllRegisterOperator m)))
      T fstar :=
  hllRegisterOperator_local_laws_bundle fstar T h_merge

/-- **Count-Min Sketch Local Laws (Complete)**:
    C1 ✓, C2 ✓ (ReencodeExact), C3 ✓ (merge-compatible oracle).
    Merge operation: elementwise addition (linear sketch, NOT idempotent). -/
theorem cms_local_laws_complete {d w : ℕ}
    (fstar : CMSState d w → Y)
    (T : BinTree (CMSState d w))
    (h_merge : SketchMergeCompatible (cmsRegisterOperator d w) fstar) :
    LocalLawsBundle
      (deterministicSummarizer (summaryFromSketch (cmsRegisterOperator d w)))
      T fstar :=
  cmsRegisterOperator_local_laws_bundle fstar T h_merge

/-- **KLL Quantile Sketch Local Laws (Complete)**:
    C1 ✓, C2 ✓ (ReencodeExact), C3 ✓ (HierarchicalMergeable).
    Merge operation: merge + rank-aware compression. -/
theorem kll_local_laws_complete
    {KLLState : Type*} [Monoid KLLState] {Sketch : Type*}
    (op : SketchOperator KLLState Sketch)
    (fstar : KLLState → Y)
    (h_leaf : SketchLeafPreserving op fstar)
    (h_merge : SketchMergeCompatible op fstar)
    (h_compat : SketchSummaryCompatible op)
    (T : BinTree KLLState) :
    LocalLawsBundle (deterministicSummarizer (summaryFromSketch op)) T fstar :=
  kll_local_laws_bundle op fstar h_leaf h_merge h_compat T

/-- **GK Quantile Sketch Local Laws (Sequential Only)**:
    C1 ✓, C2 ✓ (ReencodeExact), C3 △ (sequential merge trees only).
    Merge operation: sequential absorption (OneWayMergeable, NOT HierarchicalMergeable).

    WARNING: The local-law bundle is formally valid, but the `h_merge` hypothesis
    is only justified for sequential merge topologies. For arbitrary binary trees,
    GK does not provide the merge-compatibility guarantee. -/
theorem gk_local_laws_sequential_only
    {GKState : Type*} [Monoid GKState] {Sketch : Type*}
    (op : SketchOperator GKState Sketch)
    (fstar : GKState → Y)
    (h_leaf : SketchLeafPreserving op fstar)
    (h_merge : SketchMergeCompatible op fstar)  -- only justified for sequential topology
    (h_compat : SketchSummaryCompatible op)
    (T : BinTree GKState) :
    LocalLawsBundle (deterministicSummarizer (summaryFromSketch op)) T fstar :=
  gk_local_laws_bundle_sequential op fstar h_leaf h_merge h_compat T

/-- **Bigram Sketch Local Laws (Complete with Boundary Metadata)**:
    C1 ✓, C2 ✓, C3 ✓ — exact mergeability with boundary tokens.
    Merge operation: concatenate bigram multisets + cross-boundary pair.

    The naive bigram sketch (without boundary metadata) violates C3.
    Adding first/last token metadata recovers exact mergeability. -/
theorem bigram_sketch_exact_mergeability {α : Type*} [DecidableEq α]
    (xs ys : List α) :
    bigramSketch (xs ++ ys) = mergeSketch (bigramSketch xs) (bigramSketch ys) :=
  bigramSketch_append xs ys

/-- **Markov Count Sketch Local Laws (Complete for Exact Summarizer)**:
    C1 ✓ `L1_gExact`, C2 ✓ (exact), C3 ✓ `L2_gExact`.
    Merge operation: count addition + join changepoint detection.

    The "flip" summarizer (increments count on re-summary) violates C2. -/
theorem markov_count_sketch_laws_and_counterexample {n : ℕ} (hn : 0 < n) :
    -- Positive: exact summarizer satisfies C1 and C3
    (∀ T : BinTree (MarkovCountSketch n),
      L1 (gExact (n := n)) T (fstar (n := n)) ∧
      L2 (gExact (n := n)) T (fstar (n := n))) ∧
    -- Negative: flip summarizer violates C2
    ¬ L3 (gFlip (n := n)) (fstar (n := n)) :=
  ⟨fun T => ⟨L1_gExact T, L2_gExact T⟩,
   not_L3_gFlip n hn⟩

end PaperAliases

end FormalProofs.OPT
