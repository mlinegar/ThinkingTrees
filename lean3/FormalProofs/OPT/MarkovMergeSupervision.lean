import FormalProofs.OPT.MarkovCountSketchExample
import FormalProofs.OPT.MarkovSufficiency

/-!
# FormalProofs/OPT/MarkovMergeSupervision.lean

This file packages the merge-supervision facts used by the Markov exact-leaf
feasibility study.

The runtime comparison separates three ideas:

- pure algebraic `C3/L2`;
- direct parent count supervision; and
- direct parent full-sketch supervision.

The Lean point is straightforward:

- exact parent full-sketch supervision recovers the same exact Markov sketch
  route as `gExact`, hence `L2/C3` and zero root distortion follow;
- count-only parent supervision is not sufficient in general; and
- positive node weights change optimization geometry, not the exact zero-loss
  optimum of nodewise exact-sketch supervision.
-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

noncomputable section

namespace FormalProofs.OPT

open MarkovCountSketch

/-- Exact parent full-sketch supervision on the Markov sketch recovers
`L2/C3` on every tree. In this concrete worked example the exact route is
`gExact`, so the result is immediate. -/
theorem markov_exact_parent_fullSketch_implies_L2
    (T : BinTree (MarkovCountSketch n)) :
    L2 (gExact (n := n)) T (fstar (n := n)) :=
  L2_gExact (n := n) T

/-- Exact leaves plus exact internal parent sketches imply zero root
distortion by the same `one_pass` specialization used in the Markov worked
example. -/
theorem markov_exact_leaf_and_parent_fullSketch_zero_root_distortion
    (T : BinTree (MarkovCountSketch n)) :
    Egu (gExact (n := n)) (root T) (fun z => D (fstar (n := n)) z (S T)) = 0 :=
  exactSketch_root_distortion_zero (n := n) T

/-- Count-only parent supervision is not sufficient for the Markov
changepoint task in general: the endpoints still matter. -/
theorem markov_parent_countOnly_not_sufficient
    (hn : 1 < n) :
    ¬ MarkovCountQuerySufficient
      (n := n)
      (Summary := ℕ)
      markovCountOnlySummary :=
  markov_countOnly_not_query_sufficient (n := n) hn

/-- Positive node weights do not change the exact zero-loss optimum of a
nodewise nonnegative exact-sketch objective. -/
theorem positive_weighted_nodewise_zero_iff
    {ι : Type*} [Fintype ι]
    (w ℓ : ι → ℝ)
    (hw : ∀ i, 0 < w i)
    (hℓ : ∀ i, 0 ≤ ℓ i) :
    (∑ i, w i * ℓ i) = 0 ↔ ∀ i, ℓ i = 0 := by
  constructor
  · intro hsum
    intro i
    have hnonneg : ∀ j ∈ (Finset.univ : Finset ι), 0 ≤ w j * ℓ j := by
      intro j _hj
      exact mul_nonneg (le_of_lt (hw j)) (hℓ j)
    have hzero :
        ∀ j : ι, w j * ℓ j = 0 := by
      simpa using (Finset.sum_eq_zero_iff_of_nonneg hnonneg).1 hsum
    exact (mul_eq_zero.mp (hzero i)).resolve_left (ne_of_gt (hw i))
  · intro hzero
    simp [hzero]

end FormalProofs.OPT
