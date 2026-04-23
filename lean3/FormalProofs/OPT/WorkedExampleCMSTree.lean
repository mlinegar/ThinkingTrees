import FormalProofs.OPT.CountMinSketch

/-!
# FormalProofs/OPT/WorkedExampleCMSTree.lean

## Worked Example: Count-Min Sketch Merge Failure in a Tree

This file shows a concrete 2-leaf tree where the Count-Min Sketch merge
operation is EXACT at the sketch level (C3 holds), but a LOSSY DECODE
causes C2 failure that accumulates through the tree.

### The Key Insight

For Count-Min Sketch:
- **C3 (Merge Consistency) always holds exactly.** Elementwise addition of
  counter matrices is the defining property of a linear sketch.
  `CMS(A ∪ B) = CMS(A) + CMS(B)` with zero error.

- **C2 (Idempotence) can fail under lossy decode.** When you decode a CMS
  (read out frequency estimates via min-of-row-counts) and then re-encode
  (hash back into a new CMS), the result differs from the original because:
  - Decoding produces approximate frequencies (with hash-collision overestimates)
  - Re-encoding hashes these approximate values back, landing in different buckets
  - The round-trip `encode(decode(s)) ≠ s`

- **The failure compounds through the tree.** Each merge is exact at the counter
  level, but if you decode and re-encode at intermediate nodes (as a learned
  semantic summarizer would), the error accumulates.

### Why This Matters for C-TreePO

In the semantic setting, the "summarizer" g is a language model. It reads text
(decode) and produces a summary (encode). If the summary of a summary drifts
from the summary itself, that's a C2 violation. The CMS example isolates this
mechanism in a clean algebraic setting.

### Paper Connection

This example supports the paper's claim that C2 (idempotence) is substantive
and independent of C3 (merge consistency). A sketch can have perfect merge
algebra but still fail under re-application of the summary operator.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

open CMSState

/-!
## Part 1: CMS Merge is Exact (C3 Always Holds)

With a tiny 1-row, 2-column CMS (d=1, w=2), we show that merge = elementwise
addition produces the correct combined counter matrix.
-/

-- A tiny CMS: 1 hash function, 2 buckets
abbrev TinyCMS := CMSState 1 2

def bucket0 : Fin 2 := ⟨0, by omega⟩
def bucket1 : Fin 2 := ⟨1, by omega⟩
def row0 : Fin 1 := ⟨0, by omega⟩

-- Leaf A: item hashed to bucket 0 (3 times)
def cmsLeafA : TinyCMS := ⟨fun _ j => if j = bucket0 then 3 else 0⟩

-- Leaf B: item hashed to bucket 1 (2 times), and bucket 0 (1 time)
def cmsLeafB : TinyCMS := ⟨fun _ j => if j = bucket0 then 1 else 2⟩

-- Merge = elementwise addition
def cmsMerged : TinyCMS := cmsLeafA * cmsLeafB

-- The merge correctly adds counters: bucket 0 gets 3+1=4, bucket 1 gets 0+2=2
theorem merge_counter_bucket0 :
    cmsMerged.counters row0 bucket0 = 4 := by
  simp [cmsMerged, cmsLeafA, cmsLeafB, merge, bucket0, row0]

theorem merge_counter_bucket1 :
    cmsMerged.counters row0 bucket1 = 2 := by
  simp [cmsMerged, cmsLeafA, cmsLeafB, merge, bucket0, bucket1, row0]
  decide

-- This is EXACT: CMS(A ∪ B) = CMS(A) + CMS(B). No approximation error from merge.
-- This is C3 in its purest form: the merge algebra is perfectly compositional.

/-!
## Part 2: The Lossy-Decode C2 Failure

Now model a lossy decode: read the minimum count across rows for each "item",
then re-encode by hashing back. Because we have only 1 row, the "min across
rows" is just the count itself, but the re-encoding step introduces error
because decoded frequency estimates get hashed to potentially different buckets.

We use the `cmsLossyOperator` from CountMinSketch.lean (encode=id, merge=add,
decode=succ) as the simplest model of systematic decode error.
-/

-- Recall: cmsLossyOperator has decode = Nat.succ (adds 1 to every value)
-- This models a CMS where decoding systematically overestimates by 1

-- A single value, processed through the lossy operator
-- encode(5) = 5, decode(5) = 6, encode(decode(5)) = encode(6) = 6 ≠ 5
theorem lossy_roundtrip_example :
    cmsLossyOperator.encode (cmsLossyOperator.decode 5) ≠ 5 := by
  simp [cmsLossyOperator]

-- After one round-trip, the value drifts by 1
theorem lossy_one_roundtrip :
    summaryFromSketch cmsLossyOperator 5 = 6 := by
  simp [summaryFromSketch, cmsLossyOperator]

-- After two round-trips, it drifts by 2
theorem lossy_two_roundtrips :
    summaryFromSketch cmsLossyOperator (summaryFromSketch cmsLossyOperator 5) = 7 := by
  simp [summaryFromSketch, cmsLossyOperator]

-- After n round-trips, drift = n. This is unbounded!
-- This is exactly why C2 matters: without idempotence, re-summarization drifts.

/-!
## Part 3: How Drift Compounds in a Tree

Consider a 2-leaf tree under the lossy operator:

```
    Root
   /    \
  A      B
```

- Leaf A: value 3, Leaf B: value 2
- After leaf encoding: encode(3) = 3, encode(2) = 2 ✓ (C1 holds)
- After merge: merge(3, 2) = 3 + 2 = 5 ✓ (C3 holds, merge is exact)
- After root decode: decode(5) = 6 (one overestimate from decode)
- True value of A ∪ B: 3 + 2 = 5
- Reported value: 6
- Error: 1

Now compare with a deeper tree that requires an INTERMEDIATE decode+re-encode:

```
       Root
      /    \
    AB      CD
   /  \    /  \
  A    B  C    D
```

If AB is decoded and re-encoded before merging with CD:
- AB merge: merge(3,2) = 5
- AB decode: decode(5) = 6
- AB re-encode: encode(6) = 6  ← This is where C2 failure bites
- CD merge: merge(1,4) = 5
- CD decode: decode(5) = 6
- CD re-encode: encode(6) = 6
- Root merge: merge(6,6) = 12
- Root decode: decode(12) = 13
- True value: 3 + 2 + 1 + 4 = 10
- Reported value: 13
- Error: 3 (grew from 1!)

Each decode+re-encode step adds 1 to the error. In a tree of depth d,
the error is O(d). This is the tree-compounding effect of C2 failure.
-/

-- Concrete: merge under lossy operator is exact at sketch level
theorem lossy_merge_exact (a b : Nat) :
    cmsLossyOperator.merge a b = a + b := by
  rfl

-- But the full round-trip (encode → merge → decode) adds systematic error
-- For a 2-leaf tree: leaf values 3 and 2
-- True: 3 + 2 = 5
-- Lossy decode of merge: decode(merge(encode(3), encode(2))) = decode(5) = 6
theorem lossy_tree_2leaf :
    cmsLossyOperator.decode
      (cmsLossyOperator.merge
        (cmsLossyOperator.encode 3)
        (cmsLossyOperator.encode 2)) = 6 := by
  simp [cmsLossyOperator]

-- The error for 2 leaves: |6 - 5| = 1
theorem lossy_tree_2leaf_error :
    cmsLossyOperator.decode
      (cmsLossyOperator.merge
        (cmsLossyOperator.encode 3)
        (cmsLossyOperator.encode 2)) - (3 + 2) = 1 := by
  simp [cmsLossyOperator]

-- If we re-encode after decoding (C2 violation), error compounds
-- decode(merge(re-encode(decode(merge(3,2))), re-encode(decode(merge(1,4)))))
-- = decode(merge(encode(6), encode(6)))
-- = decode(merge(6, 6))
-- = decode(12)
-- = 13
-- True: 3 + 2 + 1 + 4 = 10
-- Error: 3
theorem lossy_tree_4leaf_with_reencode :
    let ab_sketch := cmsLossyOperator.merge
      (cmsLossyOperator.encode 3) (cmsLossyOperator.encode 2)
    let cd_sketch := cmsLossyOperator.merge
      (cmsLossyOperator.encode 1) (cmsLossyOperator.encode 4)
    let ab_reencoded := cmsLossyOperator.encode (cmsLossyOperator.decode ab_sketch)
    let cd_reencoded := cmsLossyOperator.encode (cmsLossyOperator.decode cd_sketch)
    let root_sketch := cmsLossyOperator.merge ab_reencoded cd_reencoded
    cmsLossyOperator.decode root_sketch = 13 := by
  simp [cmsLossyOperator]

-- Without re-encoding (if we could keep sketch states), error is only 1
-- decode(merge(merge(3,2), merge(1,4))) = decode(merge(5, 5)) = decode(10) = 11
-- Error: 1 (only the final decode)
theorem lossy_tree_4leaf_without_reencode :
    let ab_sketch := cmsLossyOperator.merge
      (cmsLossyOperator.encode 3) (cmsLossyOperator.encode 2)
    let cd_sketch := cmsLossyOperator.merge
      (cmsLossyOperator.encode 1) (cmsLossyOperator.encode 4)
    let root_sketch := cmsLossyOperator.merge ab_sketch cd_sketch
    cmsLossyOperator.decode root_sketch = 11 := by
  simp [cmsLossyOperator]

-- Error comparison:
-- Without re-encoding: |11 - 10| = 1  (decode error only)
-- With re-encoding:    |13 - 10| = 3  (decode + C2 compound error)
-- The difference (2) is exactly the number of internal decode+re-encode steps

/-!
## Part 4: The Zero-Merge / One-Merge / Deep-Tree Progression

### Zero merges (leaf only, no tree):
- You process each leaf independently and get leaf-level answers.
- You only need C1 (leaf summaries are correct).
- But you CANNOT combine the answers correctly for non-separable queries.

### One merge (2-leaf tree):
- You need C3 (merge consistency) to combine correctly.
- For CMS, merge is exact (addition), so C3 holds perfectly.
- The only error comes from the final decode, which is not a tree issue.

### Deep tree with intermediate re-encoding:
- Each intermediate decode+re-encode step is a C2 check.
- If C2 fails (lossy decode), error compounds with tree depth.
- This is the structural reason why C2 matters for trees.

### Deep tree WITHOUT intermediate re-encoding:
- If you can keep sketch states through the tree (no intermediate decoding),
  merge is exact at every level.
- Error comes only from the final decode at the root.
- This is the ideal case: the tree gives you the same answer as processing
  everything at once, plus one decode step.

### The C-TreePO analogue:
- In the semantic setting, "decode" = reading the summary as text.
- "Re-encode" = re-summarizing the summary.
- C2 (idempotence) ensures that re-summarizing doesn't drift.
- Without C2, each level of the tree compounds the drift.
- This is why the tree structure FORCES you to care about C2.
-/

-- Clean comparison: exact operator (identity) vs lossy operator
-- Both have exact merge. The difference is ONLY in the decode.

-- Exact operator: encode=id, merge=add, decode=id
-- decode(merge(merge(3,2), merge(1,4))) = 10 ✓
theorem exact_tree_correct :
    let op := identitySketchOperator (Strings := Nat)
    op.decode (op.merge (op.merge (op.encode 3) (op.encode 2))
                        (op.merge (op.encode 1) (op.encode 4))) = 10 := by
  simp [identitySketchOperator]

-- Re-encoding under exact operator is harmless (C2 holds)
-- decode(merge(encode(decode(merge(3,2))), encode(decode(merge(1,4)))))
-- = merge(merge(3,2), merge(1,4)) = 10 ✓
theorem exact_tree_with_reencode_still_correct :
    let op := identitySketchOperator (Strings := Nat)
    let ab := op.merge (op.encode 3) (op.encode 2)
    let cd := op.merge (op.encode 1) (op.encode 4)
    let ab_re := op.encode (op.decode ab)
    let cd_re := op.encode (op.decode cd)
    op.decode (op.merge ab_re cd_re) = 10 := by
  simp [identitySketchOperator]

/-!
## Summary

| Scenario | C1 needed | C3 needed | C2 needed | CMS exact | CMS lossy |
|----------|-----------|-----------|-----------|-----------|-----------|
| Leaf only (no merge) | ✓ | — | — | ✓ correct | ✓ correct |
| One merge (2-leaf) | ✓ | ✓ | — | ✓ correct | 1 error |
| Deep tree, no re-encode | ✓ | ✓ | — | ✓ correct | 1 error |
| Deep tree, with re-encode | ✓ | ✓ | ✓ | ✓ correct | O(depth) error |

The tree structure FORCES you to care about all three laws:
- C1 for leaves (always needed)
- C3 for merges (needed the moment you have any merge)
- C2 for re-application (needed when summaries pass through intermediate nodes)
-/

end FormalProofs.OPT
