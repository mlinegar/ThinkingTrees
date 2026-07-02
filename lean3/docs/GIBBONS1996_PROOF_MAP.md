# Gibbons 1996 Proof Map

Source: Jeremy Gibbons, "The Third Homomorphism Theorem", JFP 6(4), 1996.

Checked Lean module:

- `FormalProbability/ML/MergeableSummaries/Gibbons1996.lean`

Chronology re-export:

- `FormalProbability/ML/MergeableSummaries/LiteratureChronology.lean`

## Claim Inventory

| Paper item | Lean names | Formalization status |
|---|---|---|
| Abstract/main claim: left-to-right plus right-to-left implies homomorphic, hence arbitrary parenthesization | `Gibbons1996.theorem_4_1_third_homomorphism`, `Gibbons1996.theorem_4_1_parenthesization_invariance` | Fully formalized. |
| Introduction examples: `id`, `map`, `concat`, `head`, `length`, `sum`, `min`, `all` | `Gibbons1996.id_homomorphic`, `map_homomorphic`, `concat_homomorphic`, `head?_homomorphic`, `length_homomorphic`, `nat_sum_homomorphic`, `min_with_top_homomorphic`, `bool_all_homomorphic`; chronology aliases `gibbons1996_intro_*` | Fully formalized. `head` is totalized as `head?`; `min` is totalized with an `OrderTop` unit. |
| Introduction non-example: longest sorted prefix is not homomorphic | `Gibbons1996.longestSortedPrefixNat`, `longestSortedPrefixNat_kernel_counterexample`, `longestSortedPrefixNat_not_homomorphic`, `gibbons1996_intro_lsp_not_homomorphic` | Fully formalized for a concrete natural-number instance. |
| Section 2: list homomorphism over concatenation | `Gibbons1996.Homomorphic`, existing `OrderedListHomomorphism` | Fully formalized. |
| Section 2: homomorphic operator is associative on the range | `Gibbons1996.homomorphic_associative_on_range`, `orderedListHomomorphism_associative_on_image`, `gibbons1996_03_associative_on_homomorphic_image` | Fully formalized. |
| Section 2: `h []` is the unit on the range | `Gibbons1996.homomorphic_empty_left_unit_on_range`, `Gibbons1996.homomorphic_empty_right_unit_on_range` | Fully formalized. |
| Section 2: leftwards function | `Gibbons1996.Leftwards` | Fully formalized as `h (a :: xs) = step a (h xs)`. |
| Section 2: rightwards function | `Gibbons1996.Rightwards` | Fully formalized as `h (xs ++ [a]) = step (h xs) a`. |
| Equation (1), foldr over append | `Gibbons1996.foldr_append_acc` | Fully formalized. |
| Equation (2), foldl over append | `Gibbons1996.foldl_append_acc` | Fully formalized. |
| Section 2 three-element expansion examples | `Gibbons1996.foldr_three`, `Gibbons1996.foldl_three` | Fully formalized. |
| Foldr presentation is leftwards | `Gibbons1996.foldr_is_leftwards` | Fully formalized. |
| Foldl presentation is rightwards | `Gibbons1996.foldl_is_rightwards` | Fully formalized. |
| Uniqueness of `foldr`/`foldl` for fixed empty value | `Gibbons1996.leftwards_eq_foldr_of_empty`, `Gibbons1996.rightwards_eq_foldl_of_empty` | Fully formalized. |
| Definition 3.1, reduction | `Gibbons1996.reduction`, `gibbons1996_00a_reduction_definition` | Fully formalized. |
| Definition 3.2, map | `Gibbons1996.listMap`, `gibbons1996_00b_map_definition` | Fully formalized. |
| `map f = hom (++) ([.] ∘ f) []` | `Gibbons1996.listMap_eq_hom_append_singleton` | Fully formalized. |
| Reductions are homomorphic in monoid notation | `Gibbons1996.reduction_homomorphic_monoid` | Fully formalized. |
| Theorem 3.3, every homomorphism factors as reduction after map | `Gibbons1996.theorem_3_3_first_homomorphism_factorization`, `gibbons1996_00c_first_homomorphism_factorization` | Fully formalized. |
| Theorem 3.3 converse, reduction after map is a homomorphism | `Gibbons1996.theorem_3_3_first_homomorphism_converse_monoid`, `gibbons1996_00d_first_homomorphism_converse_monoid` | Fully formalized in the monoid form used by Lean. |
| Theorem 3.3 uniqueness in monoid notation | `Gibbons1996.theorem_3_3_first_homomorphism_unique_monoid`, `gibbons1996_00d_unique_monoid` | Fully formalized. |
| Theorem 3.4, homomorphisms are leftwards | `Gibbons1996.theorem_3_4_second_homomorphism_leftwards`, `gibbons1996_00e_second_homomorphism_leftwards` | Fully formalized. |
| Theorem 3.4, homomorphisms are rightwards | `Gibbons1996.theorem_3_4_second_homomorphism_rightwards`, `gibbons1996_00f_second_homomorphism_rightwards` | Fully formalized. |
| Theorem 3.4 fold equalities in monoid notation | `Gibbons1996.hom_monoid_eq_foldr`, `Gibbons1996.hom_monoid_eq_foldl`, `Gibbons1996.theorem_3_4_hom_eq_foldr_monoid`, `Gibbons1996.theorem_3_4_hom_eq_foldl_monoid`, `gibbons1996_00f_foldr_monoid`, `gibbons1996_00f_foldl_monoid` | Fully formalized. |
| Lemma 4.2, representative function on the range | `Gibbons1996.RangeSection`, `Gibbons1996.lemma_4_2_classical_range_section`, `gibbons1996_00g_range_section` | Fully formalized as the classical range-section property. The paper's computability/enumerability strengthening is documented but not needed for the algebraic theorem. |
| Lemma 4.3 only-if direction | `Gibbons1996.lemma_4_3_only_if` | Fully formalized. |
| Lemma 4.3 if direction with section | `Gibbons1996.lemma_4_3_if_with_section` | Fully formalized with explicit constructed operator. |
| Lemma 4.3 iff form | `Gibbons1996.lemma_4_3_homomorphic_iff_kernel_congruent`, `gibbons1996_00h_homomorphic_iff_kernel_congruent` | Fully formalized. |
| Corollary after Lemma 4.3, injective functions are homomorphic | `Gibbons1996.injective_function_is_homomorphic` | Fully formalized. |
| Theorem 4.1 with explicit section | `Gibbons1996.theorem_4_1_third_homomorphism_with_section` | Fully formalized with `t ⊙ u = h (g t ++ g u)`. |
| Theorem 4.1 existence form | `Gibbons1996.theorem_4_1_third_homomorphism`, `gibbons1996_00i_third_homomorphism`, `ctreepo_gibbons1996_third_homomorphism` | Fully formalized. |
| Abstract parenthesization consequence | `Gibbons1996.theorem_4_1_parenthesization_invariance` | Fully formalized using merge trees. |
| Section 5, sorting derivation core step | `Gibbons1996.section_5_sort_homomorphic_of_foldr_foldl`, `gibbons1996_00j_sort_homomorphic_of_foldr_foldl` | Formalized as a theorem schema: any sorting-like function with both insert-fold presentations is homomorphic. |
| Section 5, `sort = foldr ins []` | `Gibbons1996.insertionSort_eq_foldr_orderedInsert` | Fully formalized with mathlib `List.insertionSort` and `List.orderedInsert`. |
| Section 5, equation (4), `sort = foldl ins' []` | `Gibbons1996.backwardsInsertionSort`, `Gibbons1996.backwardsInsertionSort_eq_insertionSort`, `Gibbons1996.insertionSort_eq_foldl_orderedInsert`, `gibbons1996_00k_backwards_insertionSort_eq` | Fully formalized with mathlib sortedness/permutation lemmas. |
| Section 5, homomorphic sorting consequence | `Gibbons1996.insertionSort_homomorphic_exists`, `gibbons1996_00l_insertionSort_homomorphic_exists` | Fully formalized. |
| Section 5, proof-chosen inefficient operator `u ⊙ v = sort (u ++ v)` | `Gibbons1996.inefficientSortMerge`, `Gibbons1996.insertionSort_inefficientMerge_homomorphic`, `gibbons1996_00m_inefficient_merge_homomorphic` | Fully formalized. |
| Section 5, sorted-first-argument simplification and `u ⊙ [] = u` | `Gibbons1996.inefficientSortMerge_eq_foldl_of_pairwise`, `Gibbons1996.inefficientSortMerge_nil_right_of_pairwise` | Fully formalized. |
| Lemma 5.1, preserving the smallest head through insertion folds | `Gibbons1996.lemma_5_1_foldl_orderedInsertLT_cons`, `gibbons1996_00n_lemma_5_1` | Fully formalized for mathlib `List.orderedInsert` under a strict linear-order hypothesis. |
| Section 5, standard merge equations | `Gibbons1996.standardMerge`, `standardMerge_nil_left`, `standardMerge_nil_right`, `standardMerge_cons_cons` | Fully formalized via mathlib `List.merge`. |
| Section 5, merge preserves sortedness | `Gibbons1996.standardMerge_pairwise`, `gibbons1996_00n_standard_merge_pairwise` | Fully formalized. |
| Section 5, mergesort agrees with insertion sort extensionally | `Gibbons1996.mergeSort_eq_insertionSort`, `gibbons1996_00o_mergeSort_eq_insertionSort` | Fully formalized via mathlib `List.mergeSort_eq_insertionSort`. |
| Section 5 runtime claims: insertion sort quadratic, balanced mergesort `O(n log n)`, merge linear | `Gibbons1996.RuntimeCost`, `linearGrowth`, `quadraticGrowth`, `nLogNGrowth`, `SizedCostModel`, `LinearTime`, `QuadraticTime`, `NLogNTime`, `linearGrowth_linearTime`, `quadraticGrowth_quadraticTime`, `nLogNGrowth_nLogNTime`, `insertionSortReferenceCostModel`, `standardMergeReferenceCostModel`, `mergeSortReferenceCostModel`, `Section5RuntimeClaims`, `referenceSection5RuntimeClaims`, `section_5_runtime_claims_extract`, `gibbons1996_00p_runtime_claims_extract`, `gibbons1996_00q_linearGrowth_linearTime`, `gibbons1996_00r_quadraticGrowth_quadraticTime`, `gibbons1996_00s_nLogNGrowth_nLogNTime`, `gibbons1996_00t_referenceSection5RuntimeClaims`, `ctreepo_gibbons1996_referenceSection5RuntimeClaims` | Formalized as mathlib `IsBigO` obligations over explicit length-indexed cost models. The reference Section 5 cost package is now inhabited for quadratic insertion sort, linear merge, and `n log n` merge sort. Low-level operation-count semantics for Lean's executable sorting implementations remain a separate systems-cost layer. |

## Proof Spine

1. Define leftwards and rightwards computation as local equations.
2. Prove the fold append laws corresponding to paper equations (1) and (2).
3. Prove the first homomorphism theorem:
   - factorization: homomorphism equals reduction after singleton map;
   - converse: reduction after map is homomorphic in the monoid setting.
4. Prove specialization: every homomorphism is both leftwards and rightwards.
5. Prove a range-section version of Lemma 4.2.
6. Prove Lemma 4.3: homomorphism iff `h` respects concatenation of equivalence classes induced by `h`.
7. Prove the Third Homomorphism Theorem by showing leftwards plus rightwards implies Lemma 4.3's congruence condition.
8. Derive the parenthesization/tree-invariance corollary used by C-TreePO.
9. Record the sorting application as a schema over any `sort`, `ins`, and `ins'` satisfying the two fold presentations.
10. Instantiate the sorting schema with mathlib insertion sort and connect the final step to mathlib merge/merge-sort correctness.

## Scope Notes

The Lean module mechanizes the algebraic theorem and its formal consequences.
It deliberately separates:

- the algebraic range-section property used in the proof, which is mechanized;
- the paper's computable/enumerable construction of that section, which is a
  computability refinement not needed downstream;
- the runtime complexity claims in Section 5, which are represented as formal
  Big-O cost-model obligations. The extensional sorting functions, merge
  equations, sortedness preservation, homomorphic sorting operator, and
  merge-sort equivalence are checked in Lean; reference length-indexed cost
  models inhabit the quadratic/linear/`n log n` Section 5 package. A true
  low-level operation-count semantics for the executable implementations is the
  remaining runtime layer.
