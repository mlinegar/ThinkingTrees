import FormalProofs.OPT.MarkovCountSketchExample

/-!
# FormalProofs/OPT/WorkedExampleMarkovTree.lean

## Worked Example: Markov Changepoint Counting on a 4-Leaf Tree

This file walks through a concrete 4-leaf binary tree, step by step, showing
exactly how each local law (C1, C2, C3) applies at specific nodes, how the
merge operation works, and how the tree recovers the correct changepoint count.

### Setup

We use a 2-regime Markov chain (regime 0 = "calm", regime 1 = "spike").
Each leaf is a contiguous segment of the chain. The oracle counts total
changepoints in the full sequence.

### The Document

Consider a 4-segment document with the following regime sequence:

```
Leaf A: [calm]        → sketch (count=0, first=0, last=0)
Leaf B: [spike]       → sketch (count=0, first=1, last=1)
Leaf C: [spike]       → sketch (count=0, first=1, last=1)
Leaf D: [calm]        → sketch (count=0, first=0, last=0)
```

Full sequence: calm → spike → spike → calm
Changepoints: calm→spike (1), spike→calm (1) = **2 total**

### The Tree

```
         Root (should get count=2)
        /    \
      AB      CD
     /  \    /  \
    A    B  C    D
```

### Why This Example Matters

- With **no merges** (just leaves), each leaf reports 0 changepoints. Summing
  these gives 0, which is WRONG. The changepoints live at the BOUNDARIES between
  leaves, not within them. This is the non-additively-separable target.

- With **one merge** (AB or CD), the boundary changepoint is detected because
  the sketch carries `first` and `last` regime metadata. This is exactly what
  C3 (merge consistency) checks.

- The **tree structure** ensures that ALL boundary changepoints are detected,
  regardless of where the partition falls. This is the compositional guarantee
  that makes the sketch mergeable.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

set_option maxHeartbeats 0
set_option maxRecDepth 4000
set_option relaxedAutoImplicit false
set_option autoImplicit false

noncomputable section

namespace FormalProofs.OPT

open MarkovCountSketch

/-!
## Step 1: Define the Concrete Leaves

Each leaf is a single-token segment. The sketch for a single token has
count=0 (no internal changepoints), and first=last=regime.
-/

-- Two regimes: 0 = calm, 1 = spike
abbrev calm : Fin 2 := ⟨0, by omega⟩
abbrev spike : Fin 2 := ⟨1, by omega⟩

-- Leaf sketches: single-token segments have 0 internal changepoints
def leafA : MarkovCountSketch 2 := nonempty 0 calm calm    -- [calm]
def leafB : MarkovCountSketch 2 := nonempty 0 spike spike  -- [spike]
def leafC : MarkovCountSketch 2 := nonempty 0 spike spike  -- [spike]
def leafD : MarkovCountSketch 2 := nonempty 0 calm calm    -- [calm]

/-!
## Step 2: C1 (Leaf Sufficiency) — What Happens at Each Leaf

C1 says: `g(raw_leaf) ≈ raw_leaf` at the oracle level.

For the exact summarizer (`gExact = PMF.pure`), g is the identity, so C1
holds trivially: the sketch IS the raw data.

The oracle is `fstar(s) = s.count`. Each leaf has count=0, which is correct:
a single-token segment has no internal changepoints.
-/

-- Each leaf sketch correctly reports 0 internal changepoints
theorem leafA_oracle : fstar leafA = 0 := by simp [fstar, leafA, MarkovCountSketch.count]
theorem leafB_oracle : fstar leafB = 0 := by simp [fstar, leafB, MarkovCountSketch.count]
theorem leafC_oracle : fstar leafC = 0 := by simp [fstar, leafC, MarkovCountSketch.count]
theorem leafD_oracle : fstar leafD = 0 := by simp [fstar, leafD, MarkovCountSketch.count]

/-!
## Step 3: The Naive (Wrong) Approach — Summing Leaf Counts

If we just sum the leaf oracle values: 0 + 0 + 0 + 0 = 0.
But the true answer is 2 changepoints. The error is 100%.

This is the additively-separable failure: changepoints live at BOUNDARIES
between leaves, not within them. No leaf-only method can detect them.
-/

theorem naive_sum_is_wrong :
    fstar leafA + fstar leafB + fstar leafC + fstar leafD = 0 := by
  simp [fstar, leafA, leafB, leafC, leafD, MarkovCountSketch.count]

-- The true answer for the full sequence calm→spike→spike→calm is 2
-- (We'll prove this after building the tree)

/-!
## Step 4: C3 (Merge Consistency) — The First Merge

Now merge leaf A with leaf B. The merge operation:
1. Takes the internal counts: 0 + 0 = 0
2. Checks the BOUNDARY: leafA.last = calm, leafB.first = spike
3. Since calm ≠ spike, adds 1 for the boundary changepoint
4. Result: (count=0+0+1, first=calm, last=spike) = (count=1, first=0, last=1)

This is C3 in action: the merge operation detects the changepoint that neither
leaf could see individually. The boundary metadata (first, last) is the
"interaction-bearing state" that makes the sketch mergeable.
-/

def mergeAB : MarkovCountSketch 2 := leafA * leafB

-- The merge correctly detects the calm→spike changepoint
theorem mergeAB_value :
    mergeAB = nonempty 1 calm spike := by
  simp [mergeAB, leafA, leafB, mul, join, calm, spike]

theorem mergeAB_oracle : fstar mergeAB = 1 := by
  simp [fstar, mergeAB_value, MarkovCountSketch.count]

-- Similarly for C and D:
def mergeCD : MarkovCountSketch 2 := leafC * leafD

-- The merge detects the spike→calm changepoint
theorem mergeCD_value :
    mergeCD = nonempty 1 spike calm := by
  simp [mergeCD, leafC, leafD, mul, join, calm, spike]

theorem mergeCD_oracle : fstar mergeCD = 1 := by
  simp [fstar, mergeCD_value, MarkovCountSketch.count]

/-!
## Step 5: The Root Merge — Composing Two Subtrees

Now merge AB with CD. The merge operation:
1. Internal counts: 1 + 1 = 2
2. Boundary: mergeAB.last = spike, mergeCD.first = spike
3. Since spike = spike, adds 0 (no changepoint at this boundary)
4. Result: (count=2+0, first=calm, last=calm) = (count=2, first=0, last=0)

The root correctly reports 2 changepoints!
-/

def rootMerge : MarkovCountSketch 2 := mergeAB * mergeCD

theorem rootMerge_value :
    rootMerge = nonempty 2 calm calm := by
  simp [rootMerge, mergeAB_value, mergeCD_value, mul, join, calm, spike]

theorem rootMerge_oracle : fstar rootMerge = 2 := by
  simp [fstar, rootMerge_value, MarkovCountSketch.count]

/-!
## Step 6: Tree-Level Verification

Build the actual `BinTree` and verify that hierarchical reduction gives the
same answer.
-/

def exampleTree : BinTree (MarkovCountSketch 2) :=
  BinTree.node
    (BinTree.node (BinTree.leaf leafA) (BinTree.leaf leafB))
    (BinTree.node (BinTree.leaf leafC) (BinTree.leaf leafD))

-- The monoid concatenation S(T) computes the correct answer
theorem tree_S_value : S exampleTree = nonempty 2 calm calm := by
  simp [S, exampleTree, leafA, leafB, leafC, leafD, mul, join, calm, spike]

theorem tree_oracle_correct : fstar (S exampleTree) = 2 := by
  simp [fstar, tree_S_value, MarkovCountSketch.count]

-- Under the exact summarizer, the tree reduction also gives 2
theorem tree_reduction_correct :
    Egu (gExact (n := 2)) (root exampleTree)
      (fun z => D (fstar (n := 2)) z (S exampleTree)) = 0 := by
  exact exactSketch_root_distortion_zero exampleTree

/-!
## Step 7: What Happens When C3 Fails — The "Naive Merge"

Suppose instead of the correct merge, we used a "naive" merge that just
adds internal counts WITHOUT checking the boundary:

  naiveMerge(s₁, s₂) = (s₁.count + s₂.count, s₁.first, s₂.last)

This drops the `join` term. Let's see what happens.
-/

def naiveMerge (a b : MarkovCountSketch 2) : MarkovCountSketch 2 :=
  match a, b with
  | empty, b => b
  | a, empty => a
  | nonempty c₁ f₁ _, nonempty c₂ _ l₂ =>
      nonempty (c₁ + c₂) f₁ l₂  -- NO join term!

-- Naive merge of AB misses the changepoint
theorem naive_mergeAB_wrong :
    naiveMerge leafA leafB = nonempty 0 calm spike := by
  simp [naiveMerge, leafA, leafB]

theorem naive_mergeAB_oracle : fstar (naiveMerge leafA leafB) = 0 := by
  simp [fstar, naive_mergeAB_wrong, MarkovCountSketch.count]

-- Naive merge gives 0 at the root instead of 2
theorem naive_root_wrong :
    fstar (naiveMerge (naiveMerge leafA leafB) (naiveMerge leafC leafD)) = 0 := by
  simp [naiveMerge, leafA, leafB, leafC, leafD, fstar, MarkovCountSketch.count]

-- The error: expected 2, got 0
theorem naive_merge_error :
    |fstar (naiveMerge (naiveMerge leafA leafB) (naiveMerge leafC leafD)) -
     fstar (S exampleTree)| = 2 := by
  simp [naive_root_wrong, tree_oracle_correct, fstar, tree_S_value,
        MarkovCountSketch.count]

/-!
## Step 8: The Zero-Merge vs One-Merge Distinction

This is the critical pedagogical point:

- **Zero merges** (leaf-only processing): You only need C1.
  Each leaf sketch correctly represents its own segment.
  But you CANNOT recover the global answer because changepoints
  live at boundaries.

- **One merge**: You need C3. The merge operation must detect
  boundary interactions. With even a single merge, the boundary
  metadata becomes load-bearing.

- **Multiple merges** (the full tree): C3 composes inductively.
  Each merge level detects its boundary changepoints, and the
  tree structure ensures all boundaries are covered.

- **Re-application** (multi-round): You need C2. If you re-summarize
  the root, the count must not change. The `gFlip` counterexample
  (which increments count) shows what goes wrong without C2.
-/

-- Zero merges: each leaf is correct individually
theorem zero_merges_each_leaf_correct :
    fstar leafA = 0 ∧ fstar leafB = 0 ∧ fstar leafC = 0 ∧ fstar leafD = 0 :=
  ⟨leafA_oracle, leafB_oracle, leafC_oracle, leafD_oracle⟩

-- But summing them gives the wrong global answer
theorem zero_merges_global_wrong :
    fstar leafA + fstar leafB + fstar leafC + fstar leafD ≠
    fstar (S exampleTree) := by
  simp [leafA_oracle, leafB_oracle, leafC_oracle, leafD_oracle, tree_oracle_correct]

-- One merge already recovers boundary information
theorem one_merge_detects_boundary :
    fstar mergeAB = 1 ∧ fstar mergeCD = 1 :=
  ⟨mergeAB_oracle, mergeCD_oracle⟩

-- Full tree (two merge levels) recovers the exact answer
theorem full_tree_exact :
    fstar rootMerge = fstar (S exampleTree) := by
  simp [rootMerge_oracle, tree_oracle_correct]

/-!
## Step 9: Why the Tree Structure Matters

Consider a DIFFERENT tree topology over the same 4 leaves:

```
         Root
        /    \
      ABC     D
     /   \
    AB    C
   /  \
  A    B
```

The answer must be the same (2 changepoints) regardless of tree shape.
This is the content of Hierarchical Merge Invariance.
-/

def leftSkewedTree : BinTree (MarkovCountSketch 2) :=
  BinTree.node
    (BinTree.node
      (BinTree.node (BinTree.leaf leafA) (BinTree.leaf leafB))
      (BinTree.leaf leafC))
    (BinTree.leaf leafD)

-- The left-skewed tree gives the same answer
theorem left_skewed_same_oracle :
    fstar (S leftSkewedTree) = 2 := by
  simp [fstar, S, leftSkewedTree, leafA, leafB, leafC, leafD,
    MarkovCountSketch.count, mul, join, calm, spike]

-- Both topologies give the same oracle value
theorem tree_topology_invariance :
    fstar (S exampleTree) = fstar (S leftSkewedTree) := by
  simp [tree_oracle_correct, left_skewed_same_oracle]

/-!
## Summary: The Markov Sketch Pedagogy

1. **The oracle** (changepoint count) is NOT additively separable:
   changepoints live at boundaries between leaves, not within them.

2. **C1** ensures each leaf correctly represents its own segment.
   This is necessary but not sufficient for the global answer.

3. **C3** ensures that each merge correctly detects boundary changepoints.
   The sketch's `(count, first, last)` state is the minimal mergeable
   information: `count` accumulates, `first`/`last` enable boundary detection.

4. **The tree structure** composes merges inductively. Each merge level
   handles its own boundaries. The root gets the correct global answer
   regardless of tree topology.

5. **Without boundary metadata** (the naive merge), every merge misses its
   boundary changepoint. The error is exactly the number of boundaries
   where regimes change — which is the quantity we're trying to measure.

6. **C2** ensures stability under re-application. If we re-summarize the
   root (e.g., in a multi-round protocol), the count must not drift.
   The `gFlip` summarizer violates this by incrementing count each time.
-/

end FormalProofs.OPT
