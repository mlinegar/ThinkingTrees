import FormalProofs.OPT.CoreDefinitions

/-!
# FormalProofs/LocalLaws.lean

## Paper Reference: Section 3 (Consistency Conditions)

This file formalizes the three local consistency conditions from Section 3 of the paper:

### Correspondence with Paper Notation

| Paper Name | Paper Equation | Lean Name | Description |
|------------|----------------|-----------|-------------|
| **C1** (Sufficiency) | `g(b) ~ b` | **L1** | Leaf preserves oracle |
| **C2** (Idempotence) | `g(s) ~ s` for s ∈ range(g) | **L3** | Re-summary is inert |
| **C3** (Merge) | `u·v ~ g(u·v) ~ g(g(u)·g(v))` | **L2** | Merge preserves oracle |

### Key Insight: L2 captures BOTH parts of paper's C3 (Merge Consistency)

Paper's C3 (Merge Consistency) has two parts:
- **Part 1 (Raw → Joint):** `u·v ~ g(u·v)` — Jointly summarizing raw strings preserves oracle
- **Part 2 (Joint → Disjoint):** `g(u·v) ~ g(g(u)·g(v))` — Disjoint summarization equals joint

Lean's L2 directly checks the **composition** of both parts:
```
E[D(reduce g (node T_L T_R), S(node T_L T_R))] = 0
```

For a 2-leaf tree with leaves b_L, b_R:
- `reduce g (node (leaf b_L) (leaf b_R))` = distribution over `g(g(b_L) * g(b_R))`
- `S (node (leaf b_L) (leaf b_R))` = `b_L * b_R`

So L2 checks: `E[D(g(g(b_L)*g(b_R)), b_L*b_R)] = 0`

By transitivity of ~, if C3 Part 1 AND Part 2 hold, then this combined condition holds.
The two-part decomposition in the paper is useful for **auditing** (can test each part
separately when context constraints differ), while L2's combined form is cleaner for proofs.

### Key Definitions

- `Egu`: Tree expectation under hierarchical reduction
- `InRange`: Support membership predicate
- `LocalLawsBundle`: Bundle of all three laws for cleaner theorem signatures

### Code Correspondence

In the Python implementation (`src/core/`):
- `g` is implemented by `Summarizer.__call__(text, rubric)`
- The monoid operation `*` (string concatenation) is implemented by `format_merge_input(s_L, s_R)`
- Both leaf and merge use the same `g` function; the only difference is input formatting

This means:
- L1 check: `g(raw_leaf_text, rubric)`
- L2 check: `g(format_merge_input(s_L, s_R), rubric)`
- L3 check: `g(existing_summary, rubric)`

When `unified_mode=True` in `BatchedStrategy`, the code directly implements the theory's
single `g : Strings → Strings` with format_merge_input serving as the ⊕ operator.
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

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]

/-!
## Tree Expectation
-/

/-- Expected value of f under the hierarchical reduction of tree T -/
def Egu (g : Summarizer Strings) (T : BinTree Strings) (f : Strings → ℝ) : ℝ :=
  ∑' z, (reduce g T z).toReal * f z

/-!
## Local Laws
-/

/-- **L1: Leaf Sufficiency** (Paper: Condition C1)

**Paper Reference:** Section 3, Equation (C1)

Expected distortion is 0 at each leaf: `E[D(g(b), b)] = 0` for all leaves b.

This corresponds to paper's C1: `g(b) ~ b` for all realized leaves.
In expectation form: the summary of any leaf must preserve its oracle value. -/
def L1 (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y) : Prop :=
  ∀ b, b ∈ leaves T → Eg g (fun z => D fstar z b) b = 0

/-- **L2: Merge Consistency** (Paper: Condition C3)

**Paper Reference:** Section 3, Equation (C3)

Expected distortion is 0 at each internal node merge.

Paper's C3 has two parts:
- Part 1: `u·v ~ g(u·v)` (raw concatenation equals its summary)
- Part 2: `g(u·v) ~ g(g(u)·g(v))` (joint summary equals disjoint summary)

L2 captures the **composition** of both parts via transitivity:
`E[D(reduce g (node T_L T_R), S (node T_L T_R))] = 0`

where `reduce g (node T_L T_R)` hierarchically reduces both subtrees and then
summarizes their concatenation: `g(reduce(T_L) * reduce(T_R))`.

**⚠ Packaging caveat (see `MergeTriangle.lean`):** each L2 instance constrains
the *full recursive reduction of the subtree* at that node, not a single merge
call. At the root, L2 therefore already asserts the conclusion of `one_pass` —
the legacy preservation theorems are read-offs, not compositions. The genuinely
one-call local laws are `LeafSufficiency` / `MergeSufficiency` /
`ContextCompatible` in `FormalProofs/OPT/MergeTriangle.lean`; the bridge
`L2_of_local` derives this L2 from them, and `one_pass_of_local` proves
preservation by honest induction. New work should state hypotheses in those
terms and use L2 only as a derived quantity. -/
def L2 (g : Summarizer Strings) (T : BinTree Strings) (fstar : Strings → Y) : Prop :=
  ∀ p, p ∈ internal_nodes T →
    let (T_L, T_R) := p
    Egu g (BinTree.node T_L T_R) (fun z => D fstar z (S (BinTree.node T_L T_R))) = 0

/-- InRange: z is in the support of g(x) for some x.
This defines when a string is "on the range" of the summarizer. -/
def InRange (g : Summarizer Strings) (z : Strings) : Prop := ∃ x, z ∈ (g x).support

/-- **L3: On-Range Idempotence** (Paper: Condition C2)

**Paper Reference:** Section 3, Equation (C2)

Expected distortion is 0 for any string in the range of g: `E[D(g(Z), Z)] = 0`
for all Z ∈ range(g).

This corresponds to paper's C2: `g(s) ~ s` for all summaries s ∈ range(g).
In expectation form: re-summarizing an existing summary must preserve its oracle value.

**Why L3 is substantive:** Theorem 10.1 (`thm10_1_L3_not_derivable`) in
CounterexampleExistence.lean constructs a summarizer satisfying L1 on fresh inputs
but violating L3. This proves L3 cannot be derived from L1 and L2 alone. -/
def L3 (g : Summarizer Strings) (fstar : Strings → Y) : Prop :=
  ∀ Z, InRange g Z → Eg g (fun z => D fstar z Z) Z = 0

/-!
## Paper Condition Aliases

These aliases use the paper's naming convention (C1, C2, C3) for readers
following along with the paper.

**IMPORTANT:** The mapping is:
- C1 (Sufficiency) = L1
- C2 (Idempotence) = L3
- C3 (Merge Consistency) = L2
-/

/-- **Condition C1** (Paper notation alias for L1: Leaf Sufficiency) -/
abbrev C1 := @L1

/-- **Condition C2** (Paper notation alias for L3: Idempotence/On-Range Stability) -/
abbrev C2 := @L3

/-- **Condition C3** (Paper notation alias for L2: Merge Consistency).

**⚠** The paper's C3 is a one-call two-link law (`u·v ~ g(u·v) ~ g(g(u)·g(v))`);
this legacy alias points at the strictly stronger subtree-level L2. The faithful
one-call formalization is `MergeSufficiency` + `MergeTriangle` in
`FormalProofs/OPT/MergeTriangle.lean`. -/
abbrev C3 := @L2

/-!
## Local Laws Bundle

A structure bundling all three local laws (L1, L2, L3) for a summarizer on a tree.
This eliminates repeated parameter patterns in theorems that require all three laws.
-/

/-- Bundle of local preservation laws for a summarizer on a tree.

All three local laws (L1, L2, L3) are required together in theorems about
multi-round preservation and DPO gap bounds. This structure bundles them
to simplify theorem signatures. -/
structure LocalLawsBundle (g : Summarizer Strings) (T : BinTree Strings)
    (fstar : Strings → Y) where
  /-- L1: Leaf idempotence -/
  law1 : L1 g T fstar
  /-- L2: Internal node idempotence -/
  law2 : L2 g T fstar
  /-- L3: Global idempotence on range -/
  law3 : L3 g fstar

namespace LocalLawsBundle

variable {g : Summarizer Strings} {T : BinTree Strings} {fstar : Strings → Y}

/-- Construct a LocalLawsBundle from individual proofs -/
def mk' (h1 : L1 g T fstar) (h2 : L2 g T fstar) (h3 : L3 g fstar) :
    LocalLawsBundle g T fstar :=
  ⟨h1, h2, h3⟩

/-- Extract the L1 law from a bundle -/
lemma get_L1 (laws : LocalLawsBundle g T fstar) : L1 g T fstar := laws.law1

/-- Extract the L2 law from a bundle -/
lemma get_L2 (laws : LocalLawsBundle g T fstar) : L2 g T fstar := laws.law2

/-- Extract the L3 law from a bundle -/
lemma get_L3 (laws : LocalLawsBundle g T fstar) : L3 g fstar := laws.law3

end LocalLawsBundle

end
