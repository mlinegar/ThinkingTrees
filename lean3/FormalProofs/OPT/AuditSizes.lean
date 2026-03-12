/-
FormalProofs/OPT/AuditSizes.lean

Count-based audit bounds and sample-size scaling with tree size.

This file is intentionally "combinatorial": it rewrites the existing union bounds
from `AuditBounds.lean` in terms of tree counts (`numLeaves`, `numInternalNodes`)
and the *average* violation rates (`pLeafAvg`, `pMergeAvg`).

It also contains a small sample-complexity lemma showing how Hoeffding margins
scale when the total error budget is split across `N` leaves (or merges).
-/

import FormalProofs.OPT.AuditBounds
import FormalProofs.OPT.AuditCore
import FormalProofs.OPT.TreeProperties

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

section AuditSizes

variable {Strings : Type*} [Monoid Strings]
variable {Y : Type*} [PseudoMetricSpace Y]

/-!
## Turning totals into (count × average)

`AuditBounds.lean` defines:
- `totalLeafViolation` / `totalMergeViolation` as explicit sums over leaves/internal nodes
- `pLeafAvg` / `pMergeAvg` as the corresponding averages

The lemmas below connect them to the *structural* counts (`numLeaves`, `numInternalNodes`).
-/

/-- Total leaf violation equals `(numLeaves) × (average leaf violation)`. -/
lemma totalLeafViolation_eq_numLeaves_mul_pLeafAvg (g : Summarizer Strings) (fstar : Strings → Y)
    (T : BinTree Strings) :
    totalLeafViolation g fstar T = (numLeaves T : ℝ) * pLeafAvg g fstar T := by
  have hpos : 0 < numLeaves T := numLeaves_pos T
  have hnum_ne : numLeaves T ≠ 0 := Nat.ne_of_gt hpos
  have hlen_ne : (leaves T).length ≠ 0 := by
    have : 0 < (leaves T).length := by
      simpa [leaves_length_eq T] using hpos
    exact Nat.ne_of_gt this
  have hN_ne : (numLeaves T : ℝ) ≠ 0 := by
    exact_mod_cast (ne_of_gt hpos)
  -- Unfold and simplify the `if` in `pLeafAvg` using non-emptiness.
  unfold totalLeafViolation pLeafAvg
  simp [hlen_ne, leaves_length_eq, hnum_ne]

/-- Total merge violation equals `(numInternalNodes) × (average merge violation)`. -/
lemma totalMergeViolation_eq_numInternal_mul_pMergeAvg (g : Summarizer Strings) (fstar : Strings → Y)
    (T : BinTree Strings) :
    totalMergeViolation g fstar T = (numInternalNodes T : ℝ) * pMergeAvg g fstar T := by
  by_cases hlen : (internal_nodes T).length = 0
  · -- No internal nodes: both sides are 0.
    have hnodes : internal_nodes T = [] := List.length_eq_zero_iff.mp hlen
    have hnum : numInternalNodes T = 0 := by
      -- `internal_nodes_length_eq` gives `length = numInternalNodes`.
      simpa [internal_nodes_length_eq T] using hlen
    unfold totalMergeViolation pMergeAvg
    simp [hnodes, hlen, hnum]
  · -- Nonempty internal nodes: same algebra as leaves.
    have hlen_pos : 0 < (internal_nodes T).length := Nat.pos_of_ne_zero hlen
    have hlen_ne : (internal_nodes T).length ≠ 0 := Nat.ne_of_gt hlen_pos
    have hnum_ne_nat : numInternalNodes T ≠ 0 := by
      intro h0
      apply hlen
      simp [internal_nodes_length_eq T, h0]
    have hM_ne : (numInternalNodes T : ℝ) ≠ 0 := by
      exact_mod_cast hnum_ne_nat
    unfold totalMergeViolation pMergeAvg
    simp [hlen_ne, internal_nodes_length_eq, hnum_ne_nat]

/-- Internal nodes = leaves - 1, cast to `ℝ`. -/
lemma numInternalNodes_real_eq_numLeaves_real_sub_one {α : Type*} (T : BinTree α) :
    (numInternalNodes T : ℝ) = (numLeaves T : ℝ) - 1 := by
  have hle : 1 ≤ numLeaves T := Nat.succ_le_of_lt (numLeaves_pos T)
  calc
    (numInternalNodes T : ℝ)
        = (numLeaves T - 1 : ℕ) := by
            exact_mod_cast (internal_eq_leaves_minus_one T)
    _ = (numLeaves T : ℝ) - 1 := by
          simpa using (Nat.cast_sub hle)

/-- Union bound (R=1) rewritten using counts and average violation rates. -/
theorem union_bound_one_round_counts (g : Summarizer Strings) (fstar : Strings → Y)
    (T : BinTree Strings) (x : Strings) (hp : S T = x)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1) :
    Exp (reduce g T) (fun z => D fstar z x) ≤
      (numLeaves T : ℝ) * pLeafAvg g fstar T +
      (numInternalNodes T : ℝ) * pMergeAvg g fstar T := by
  have h := union_bound_one_round g fstar T x hp hbound hbound_global
  calc
    Exp (reduce g T) (fun z => D fstar z x)
        ≤ totalLeafViolation g fstar T + totalMergeViolation g fstar T := h
    _ = (numLeaves T : ℝ) * pLeafAvg g fstar T +
        (numInternalNodes T : ℝ) * pMergeAvg g fstar T := by
          simp [totalLeafViolation_eq_numLeaves_mul_pLeafAvg,
            totalMergeViolation_eq_numInternal_mul_pMergeAvg]

/-- Union bound (R rounds) rewritten using counts and average violation rates. -/
theorem union_bound_multi_round_counts (g : Summarizer Strings) (fstar : Strings → Y)
    (T : BinTree Strings) (x : Strings) (hp : S T = x)
    (R : ℕ) (hR : R ≥ 1)
    (hbound : ∀ z, D fstar z x ≤ 1)
    (hbound_global : ∀ w z, D fstar w z ≤ 1)
    (h_mono : ∀ p, pIdemp g fstar (p.bind g) ≤ pIdemp g fstar p) :
    Exp (ZR g x R T) (fun z => D fstar z x) ≤
      (numLeaves T : ℝ) * pLeafAvg g fstar T +
      (numInternalNodes T : ℝ) * pMergeAvg g fstar T +
      (R - 1) * pIdemp g fstar (reduce g T) := by
  have h := union_bound_multi_round_bounded g fstar T x hp R hR hbound hbound_global h_mono
  calc
    Exp (ZR g x R T) (fun z => D fstar z x) ≤
        totalLeafViolation g fstar T + totalMergeViolation g fstar T +
          (R - 1) * pIdemp g fstar (reduce g T) := h
    _ = (numLeaves T : ℝ) * pLeafAvg g fstar T +
        (numInternalNodes T : ℝ) * pMergeAvg g fstar T +
          (R - 1) * pIdemp g fstar (reduce g T) := by
          simp [totalLeafViolation_eq_numLeaves_mul_pLeafAvg,
            totalMergeViolation_eq_numInternal_mul_pMergeAvg]

/-!
## Sample complexity scaling with the number of leaves

`AuditCore.sample_complexity` inverts Hoeffding's margin:
if `n ≥ sample_complexity eps δ` then `confidence_margin δ n ≤ eps`.

When using a union bound with `N` leaves, to make `N * margin ≤ ε` it suffices
to require `margin ≤ ε / N`.
-/

/-- If `n` is chosen for margin `ε/N`, then `N * margin ≤ ε`. -/
lemma leaf_scaled_margin (T : BinTree Strings) (ε δ : ℝ)
    (hε : 0 < ε) (hδ : 0 < δ) (hδ' : δ < 2)
    (n : ℕ) (hn : n ≥ sample_complexity (ε / (numLeaves T : ℝ)) δ) :
    (numLeaves T : ℝ) * confidence_margin δ n ≤ ε := by
  have hN_pos : 0 < (numLeaves T : ℝ) := by
    exact_mod_cast (numLeaves_pos T)
  have hN_ne : (numLeaves T : ℝ) ≠ 0 := ne_of_gt hN_pos
  have hε' : 0 < ε / (numLeaves T : ℝ) := div_pos hε hN_pos
  have hcm : confidence_margin δ n ≤ ε / (numLeaves T : ℝ) :=
    sample_complexity_gives_margin (ε / (numLeaves T : ℝ)) δ hε' hδ hδ' n hn
  have hmul :
      (numLeaves T : ℝ) * confidence_margin δ n ≤
        (numLeaves T : ℝ) * (ε / (numLeaves T : ℝ)) :=
    mul_le_mul_of_nonneg_left hcm (le_of_lt hN_pos)
  calc
    (numLeaves T : ℝ) * confidence_margin δ n
        ≤ (numLeaves T : ℝ) * (ε / (numLeaves T : ℝ)) := hmul
    _ = ε := by
        field_simp [hN_ne]

end AuditSizes

/-!
## Corpus-level size relationships (expectations)

If documents are sampled from a PMF `μ` and each document induces a tree (a chunking),
then we can talk about *expected* tree sizes across the corpus.

We prove the structural identity
`E[numInternalNodes] = E[numLeaves] - 1`
under a mild summability hypothesis (needed only to legally split `tsum`).
-/

section CorpusExpectations

variable {Doc : Type*} {α : Type*}

/-- Expected number of leaves under a document distribution. -/
def expectedNumLeaves (μ : PMF Doc) (treeOf : Doc → BinTree α) : ℝ :=
  Exp μ (fun d => (numLeaves (treeOf d) : ℝ))

/-- Expected number of internal nodes under a document distribution. -/
def expectedNumInternalNodes (μ : PMF Doc) (treeOf : Doc → BinTree α) : ℝ :=
  Exp μ (fun d => (numInternalNodes (treeOf d) : ℝ))

/-- Expected internal nodes equal expected leaves minus one.

The only non-structural assumption is a summability hypothesis enabling `tsum` linearity. -/
lemma expectedNumInternalNodes_eq_expectedNumLeaves_minus_one
    (μ : PMF Doc) (treeOf : Doc → BinTree α)
    (h_summable : Summable (fun d => (μ d).toReal * (numLeaves (treeOf d) : ℝ))) :
    expectedNumInternalNodes μ treeOf = expectedNumLeaves μ treeOf - 1 := by
  have hsum_const : Summable (fun d => (μ d).toReal * (-1 : ℝ)) := by
    apply PMF.summable_coe_real_mul_of_bounded μ (fun _ => (-1 : ℝ)) 1 (by norm_num)
    intro _; simp

  unfold expectedNumInternalNodes expectedNumLeaves Exp
  -- Rewrite `numInternalNodes` pointwise as `numLeaves - 1`.
  have hrewrite :
      (fun d => (μ d).toReal * (numInternalNodes (treeOf d) : ℝ)) =
        fun d => (μ d).toReal * ((numLeaves (treeOf d) : ℝ) - 1) := by
    funext d
    simp [numInternalNodes_real_eq_numLeaves_real_sub_one]
  simp [hrewrite]

  -- Expand subtraction inside the `tsum` and split.
  calc
    (∑' d, (μ d).toReal * ((numLeaves (treeOf d) : ℝ) - 1))
        = ∑' d, ((μ d).toReal * (numLeaves (treeOf d) : ℝ) + (μ d).toReal * (-1 : ℝ)) := by
            refine tsum_congr ?_
            intro d
            ring
    _ = (∑' d, (μ d).toReal * (numLeaves (treeOf d) : ℝ)) +
          ∑' d, (μ d).toReal * (-1 : ℝ) := by
          simpa using (Summable.tsum_add h_summable hsum_const)
    _ = (∑' d, (μ d).toReal * (numLeaves (treeOf d) : ℝ)) + (-1) := by
          have hmul_to_neg :
              (∑' d, (μ d).toReal * (-1 : ℝ)) = ∑' d, -(μ d).toReal := by
            refine tsum_congr ?_
            intro d
            ring
          have htsum_neg : (∑' d, -(μ d).toReal) = (-1 : ℝ) := by
            -- `∑ μ = 1` for PMFs.
            simpa [PMF.toReal_tsum_coe μ] using
              (tsum_neg (L := SummationFilter.unconditional Doc) (f := fun d : Doc => (μ d).toReal))
          calc
            (∑' d, (μ d).toReal * (numLeaves (treeOf d) : ℝ)) + (∑' d, (μ d).toReal * (-1 : ℝ))
                = (∑' d, (μ d).toReal * (numLeaves (treeOf d) : ℝ)) + ∑' d, -(μ d).toReal := by
                    rw [hmul_to_neg]
            _ = (∑' d, (μ d).toReal * (numLeaves (treeOf d) : ℝ)) + (-1) := by
                    rw [htsum_neg]
    _ = (∑' d, (μ d).toReal * (numLeaves (treeOf d) : ℝ)) - 1 := by
          ring

end CorpusExpectations
