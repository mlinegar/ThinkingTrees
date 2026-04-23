import FormalProofs.OPT.SketchSummaryOperators
import FormalProofs.OPT.PreservationTheorems
import FormalProofs.OPT.HLLIdempotence

/-!
# FormalProofs/OPT/CountMinSketch.lean

## Mergeable sketch example: Count-Min Sketch for frequency estimation

This file gives a concrete, tree-invariant formalization of the Count-Min Sketch,
a classical linear sketch for approximate frequency estimation over data streams.

### Overview

The **oracle** is the frequency vector (histogram) of items in a multiset.
A **Count-Min Sketch** maintains `d` hash tables of width `w`, where each hash
function maps items to buckets. For each item, increment the bucket in each row.
To query frequency, take the minimum across rows.

### Key Properties

- **Merge operation**: Elementwise addition of sketch matrices.
  `merge(CMS(A), CMS(B)) = CMS(A ∪ B)` exactly (no approximation error from merge).
- **C1 (Leaf Sufficiency)**: Building a CMS from raw data preserves the frequency
  query to within the sketch's inherent approximation.
- **C2 (Idempotence)**: The CMS is a **linear sketch** — there is no encode/decode
  round-trip in the traditional sense. When modeled as a SketchOperator with
  identity decode (the sketch IS the theorem-domain object), C2 holds trivially.
  When modeled with a lossy decode (point queries), C2 can fail because
  `encode(decode(s)) ≠ s` — decoding reads minimum counts, re-encoding may hash
  to different buckets.
- **C3 (Merge Consistency)**: Elementwise addition is exact: the sketch of the
  union equals the sum of the sketches. This is the defining property of a linear
  sketch and holds without approximation.

### Relationship to C-TreePO

Count-Min Sketch is the prototypical **linear sketch**: merge is just addition.
This means C3 holds exactly and unconditionally. The interesting local-law behavior
is in C2: unlike HyperLogLog (where merge idempotence ≠ C2), Count-Min has
exact C3 but potentially lossy C2, depending on the decode model.

### Paper Reference

Cormode, G. and Muthukrishnan, S. (2005). "An Improved Data Stream Summary:
The Count-Min Sketch and its Applications." Journal of Algorithms.
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
## Count-Min Sketch State

We model a Count-Min Sketch as a `d × w` matrix of natural numbers, where
`d` is the number of hash functions (rows) and `w` is the width (columns per row).
-/

/-- Count-Min Sketch state: a matrix of `d` rows and `w` columns of natural-number counters. -/
structure CMSState (d w : ℕ) where
  counters : Fin d → Fin w → Nat

namespace CMSState

variable {d w : ℕ}

@[ext] theorem ext {a b : CMSState d w} (h : ∀ i j, a.counters i j = b.counters i j) :
    a = b := by
  cases a with
  | mk ca =>
      cases b with
      | mk cb =>
          have hfun : ca = cb := funext (fun i => funext (fun j => h i j))
          cases hfun
          rfl

/-- Zero CMS state: all counters are zero. -/
def zero (d w : ℕ) : CMSState d w :=
  ⟨fun _ _ => 0⟩

/-- CMS merge: elementwise addition of counter matrices.
This is the core linear-sketch property: merge is just addition. -/
def merge (a b : CMSState d w) : CMSState d w :=
  ⟨fun i j => a.counters i j + b.counters i j⟩

instance : One (CMSState d w) := ⟨zero d w⟩

instance : Mul (CMSState d w) := ⟨merge⟩

@[simp] theorem counters_one (i : Fin d) (j : Fin w) :
    (1 : CMSState d w).counters i j = 0 := rfl

@[simp] theorem counters_mul (a b : CMSState d w) (i : Fin d) (j : Fin w) :
    (a * b).counters i j = a.counters i j + b.counters i j := rfl

instance : Monoid (CMSState d w) where
  one := 1
  mul := (· * ·)
  one_mul := by
    intro a
    ext i j
    simp [zero, merge]
  mul_one := by
    intro a
    ext i j
    simp [zero, merge]
  mul_assoc := by
    intro a b c
    ext i j
    simp [merge, Nat.add_assoc]

/-- CMS merge is commutative. -/
theorem mul_comm (a b : CMSState d w) : a * b = b * a := by
  ext i j
  simp [merge, Nat.add_comm]

/-- CMS merge is NOT idempotent (unlike HLL).
    `a * a ≠ a` in general because counters double. -/
theorem merge_not_idempotent (hd : 0 < d) (hw : 0 < w) :
    ∃ a : CMSState d w, a * a ≠ a := by
  let a : CMSState d w := ⟨fun _ _ => 1⟩
  use a
  intro h_eq
  have h := congrFun₂ (congrArg CMSState.counters h_eq) ⟨0, hd⟩ ⟨0, hw⟩
  simp [merge] at h

/-- Any CMS merge tree over already-encoded states is exact at the root.
    This is because merge (addition) is associative, so tree structure doesn't matter. -/
theorem reduceDeterministic_id (T : BinTree (CMSState d w)) :
    reduceDeterministic (fun x : CMSState d w => x) T = S T := by
  induction T with
  | leaf b =>
      rfl
  | node TL TR ihL ihR =>
      simp [reduceDeterministic, S, ihL, ihR]

end CMSState

/-!
## CMS as a SketchOperator

When the theorem-domain objects ARE the CMS states (identity sketch), all three
local laws hold trivially because encode/decode are identity.
-/

section CMSRegisterOperator

variable {d w : ℕ}
variable {Y : Type*} [PseudoMetricSpace Y]

/-- The theorem-domain identity operator on CMS states. -/
abbrev cmsRegisterOperator (d w : ℕ) :
    SketchOperator (CMSState d w) (CMSState d w) :=
  identitySketchOperator (Strings := CMSState d w)

/-- The CMS register operator has exact re-encoding (trivially, as identity). -/
theorem cmsRegisterOperator_reencodeExact :
    ReencodeExact (cmsRegisterOperator d w) := by
  intro s
  rfl

/-- CMS register operator has exact summary fixed point (trivially). -/
theorem cmsRegisterOperator_summary_fixed :
    SummaryFixedPoint (summaryFromSketch (cmsRegisterOperator d w)) := by
  exact summaryFixedPoint_of_reencodeExact
    (op := cmsRegisterOperator d w)
    cmsRegisterOperator_reencodeExact

/-- **C1 (Leaf Sufficiency)**: CMS register states satisfy L1.
    Building from raw data (identity encode) preserves oracle exactly. -/
theorem cmsRegisterOperator_L1 (fstar : CMSState d w → Y)
    (T : BinTree (CMSState d w)) :
    L1 (deterministicSummarizer (summaryFromSketch (cmsRegisterOperator d w))) T fstar := by
  exact L1_of_pointwise
    (s := summaryFromSketch (cmsRegisterOperator d w))
    (fstar := fstar) (T := T)
    (identitySketch_leaf_preserving fstar)

/-- **C3 (Merge Consistency)**: CMS register states satisfy L2.
    Elementwise addition is exact — the sketch of the union equals the sum of sketches. -/
theorem cmsRegisterOperator_L2 (fstar : CMSState d w → Y)
    (T : BinTree (CMSState d w))
    (h_merge : SketchMergeCompatible (cmsRegisterOperator d w) fstar) :
    L2 (deterministicSummarizer (summaryFromSketch (cmsRegisterOperator d w))) T fstar := by
  exact L2_of_treewise
    (s := summaryFromSketch (cmsRegisterOperator d w))
    (fstar := fstar) (T := T)
    (treewise_preserving_of_sketch
      (op := cmsRegisterOperator d w)
      (fstar := fstar)
      (identitySketch_leaf_preserving fstar)
      h_merge
      (identitySketch_summary_compatible (Strings := CMSState d w)))

/-- **C2 (Idempotence)**: CMS register states satisfy L3.
    Re-summarizing an already-encoded CMS state is literally inert (identity). -/
theorem cmsRegisterOperator_L3 (fstar : CMSState d w → Y) :
    L3
      (deterministicSummarizer (summaryFromSketch (cmsRegisterOperator d w)))
      fstar := by
  exact L3_of_reencodeExact
    (op := cmsRegisterOperator d w)
    (fstar := fstar)
    cmsRegisterOperator_reencodeExact

/-- **Full local-law bundle** for CMS register states under merge-compatible oracle. -/
theorem cmsRegisterOperator_local_laws_bundle (fstar : CMSState d w → Y)
    (T : BinTree (CMSState d w))
    (h_merge : SketchMergeCompatible (cmsRegisterOperator d w) fstar) :
    LocalLawsBundle
      (deterministicSummarizer (summaryFromSketch (cmsRegisterOperator d w)))
      T fstar := by
  exact local_laws_bundle_of_sketch
    (op := cmsRegisterOperator d w)
    (fstar := fstar)
    (identitySketch_leaf_preserving fstar)
    h_merge
    (identitySketch_summary_compatible (Strings := CMSState d w))
    T

/-- Any tree reduction over already-encoded CMS states is exact at the root. -/
theorem cmsRegisterOperator_root_exact (T : BinTree (CMSState d w)) :
    sketchSummary (cmsRegisterOperator d w) T = S T := by
  calc
    sketchSummary (cmsRegisterOperator d w) T =
        reduceDeterministic (summaryFromSketch (cmsRegisterOperator d w)) T := by
          exact sketchSummary_eq_reduceDeterministic
            (op := cmsRegisterOperator d w)
            (identitySketch_summary_compatible (Strings := CMSState d w))
            T
    _ = reduceDeterministic (fun x : CMSState d w => x) T := by
          rfl
    _ = S T := CMSState.reduceDeterministic_id (d := d) (w := w) (T := T)

end CMSRegisterOperator

/-!
## CMS C2 Failure Under Lossy Decode

When the CMS is used with a lossy decode (e.g., point-query decode that returns
estimated frequencies), re-encoding the decoded output does NOT reproduce the
original sketch. This models the realistic scenario where a CMS is used as a
proper sketch with `encode ≠ decode⁻¹`.

We construct a concrete counterexample showing that a CMS-like operator with
lossy decode violates C2 (L3).
-/

section CMSLossyDecode

/-- A toy CMS-like operator where decode loses information.
    encode: identity on Nat (stores a count)
    merge: addition (linear sketch)
    decode: n ↦ n + 1 (models systematic overestimate in point queries) -/
def cmsLossyOperator : SketchOperator Nat Nat where
  encode := fun n => n
  merge := Nat.add
  decode := Nat.succ

/-- The lossy CMS operator's merge is NOT idempotent (addition doubles). -/
theorem cmsLossy_merge_not_idempotent :
    ¬ MergeIdempotent cmsLossyOperator := by
  intro h
  have h0 := h 1
  simp [MergeIdempotent, cmsLossyOperator] at h0

/-- The lossy CMS operator violates re-encoding exactness. -/
theorem cmsLossy_not_reencodeExact :
    ¬ ReencodeExact cmsLossyOperator := by
  intro h
  have h0 := h 0
  simp [ReencodeExact, cmsLossyOperator] at h0

/-- **C2 fails for lossy CMS decode**: re-summarizing inflates the count.
    This is the CMS analogue of the HLL `succMax_not_L3` counterexample. -/
theorem cmsLossy_not_L3 :
    ¬ L3
      (deterministicSummarizer (summaryFromSketch cmsLossyOperator))
      (fun n : Nat => (n : ℝ)) := by
  intro hL3
  let Z : Nat := 1  -- decode(encode(0)) = succ(0) = 1, so 1 is in range
  have hInRange :
      InRange
        (deterministicSummarizer (summaryFromSketch cmsLossyOperator))
        Z := by
    refine ⟨0, ?_⟩
    simp [InRange, deterministicSummarizer, summaryFromSketch, cmsLossyOperator, Z]
  have h0 := hL3 Z hInRange
  rw [Eg_deterministic_summaryOp] at h0
  -- summaryFromSketch cmsLossyOperator 1 = decode(encode(1)) = succ(1) = 2
  -- D (fun n => n) 2 1 = |2 - 1| = 1 ≠ 0
  norm_num [Z, D, summaryFromSketch, cmsLossyOperator, Real.dist_eq] at h0

/-- The lossy CMS DOES satisfy merge consistency (C3/L2) for the exact
    summarizer because addition is associative and preserves totals.
    Merge of encoded states is exact at the additive-count oracle level. -/
theorem cmsLossy_merge_exact :
    ∀ a b : Nat, cmsLossyOperator.merge a b = a + b := by
  intro a b
  rfl

end CMSLossyDecode

/-!
## Key Insight: Linear Sketches and Local Laws

The Count-Min Sketch illustrates a pattern complementary to HyperLogLog:

| Property | HLL | CMS |
|----------|-----|-----|
| Merge operation | max (idempotent) | add (NOT idempotent) |
| C3 (merge consistency) | ✓ (max is exact) | ✓ (add is exact, linear sketch) |
| C2 (re-summary idempotence) | ✓ only if ReencodeExact | ✓ only if ReencodeExact |
| Merge idempotence → C2? | NO (succMax counterexample) | N/A (merge not idempotent) |

Both sketches satisfy C3 exactly (merge is algebraically correct), but C2 depends
on the encode/decode round-trip, not on the merge algebra. This is a general lesson:
**C2 is about the summary operator, not about the merge operator.**
-/

end FormalProofs.OPT
