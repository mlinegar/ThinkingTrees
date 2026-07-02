# Mergeable-Sketch Full Formalization Roadmap

This document is the working inventory for formalizing the five local
literature sources in chronological order.  It is more detailed than
`MERGEABLE_SKETCH_LITERATURE_PROOF_MAP.md`: that file records the checked Lean
surface; this file records every important formalizable claim we should either
mechanize, expose as a typed theorem schema, or deliberately leave as an
external analytic/complexity theorem.

Local source bundle:

- `docs/literature/mergeable_sketches/gibbons_third_homomorphism_theorem.pdf`
- `docs/literature/mergeable_sketches/gray_data_cube.pdf`
- `docs/literature/mergeable_sketches/feldman_mud_unordered_distributed_data.pdf`
- `docs/literature/mergeable_sketches/flajolet_hyperloglog.pdf`
- `docs/literature/mergeable_sketches/agarwal_mergeable_summaries.pdf`

Status legend:

- `Mechanized`: already checked in Lean.
- `Next`: should be implemented in the next chronological pass.
- `Planned`: useful, but not blocking the next paper.
- `Citation schema`: formal statement/interface can be typed in Lean, but the
  full external proof is too large for this pass.
- `Out of scope`: not a useful Lean target for the C-TreePO mergeable-sketch
  bridge.

## Shared Lean Spine

| Formal object | Status | Lean surface | Purpose |
|---|---|---|---|
| Standalone literature import surface | Mechanized | `FormalProbability.ML.MergeableSummaries.Literature` | Imports every separately formalized mergeable-sketch module before C-TreePO re-exports selected names. |
| Stream as ordered input | Mechanized | `Stream alpha := List alpha` | Common input representation. |
| Binary merge tree | Mechanized | `MergeTree`, `MergeTree.data`, `MergeTree.eval` | Parenthesized distributed reductions. |
| State validity | Mechanized | `ValidSketch`, `MergeClosed`, `HierarchicalMergeable` | Classical state-level mergeability. |
| Query/readout correctness | Mechanized | `StateLevelMergeableSummary.QueryCorrect`, `query_tree_eq_oracle` | Readout after state merge. |
| Ordered homomorphism | Mechanized | `OrderedListHomomorphism` | Ordered text/list composition. |
| Optional laws | Mechanized | `MergeAssociative`, `MergeCommutative`, `MergeIdempotent` | Keep associativity, commutativity, idempotence separate. |
| Size and asymptotic vocabulary | Partial | `BigO`, `SizedMergeableQuerySketch`, `RandomizedSizedMergeableQuerySketch` | Needed for Agarwal and analytic estimator claims. |
| Randomized sketch correctness | Partial | `RandomizedSizedMergeableQuerySketch`, `RandomSampleEpsilonApproximationSpec`, `KLL.Algorithm` | Probability/tail-bound theorem targets are typed; executable state accounting is mechanized for KLL. |
| Finite-state lower bounds | Mechanized for mode | `Gray1997.ContextuallySeparated`, `modeBool_no_finite_state_homomorphic_realization` | Generic information-retention lower-bound schema. |

## 1. Gibbons 1996: The Third Homomorphism Theorem

Target module:

- `FormalProbability/ML/MergeableSummaries/Gibbons1996.lean`

Dedicated map:

- `lean3/docs/GIBBONS1996_PROOF_MAP.md`

### Claim Inventory

| Paper claim | Status | Lean target or existing name | Notes |
|---|---|---|---|
| List homomorphism over concatenation | Mechanized | `Gibbons1996.Homomorphic`, `OrderedListHomomorphism` | Core ordered/free-monoid interface. |
| Homomorphic combine is associative on the image | Mechanized | `homomorphic_associative_on_range`, `orderedListHomomorphism_associative_on_image` | This is the schedule-invariance law C-TreePO needs. |
| Empty list gives left/right unit on the image | Mechanized | `homomorphic_empty_left_unit_on_range`, `homomorphic_empty_right_unit_on_range` | Image-level unit, not global monoid law. |
| Introduction examples: identity, map, concat, head, length, sum, min, all | Mechanized | `id_homomorphic`, `map_homomorphic`, `concat_homomorphic`, `head?_homomorphic`, `length_homomorphic`, `nat_sum_homomorphic`, `min_with_top_homomorphic`, `bool_all_homomorphic` | Totalized where necessary. |
| Longest sorted prefix is not homomorphic | Mechanized | `longestSortedPrefixNat_not_homomorphic` | Concrete natural-number counterexample. |
| Foldr/foldl append equations | Mechanized | `foldr_append_acc`, `foldl_append_acc` | Paper equations (1) and (2). |
| Leftwards and rightwards definitions | Mechanized | `Leftwards`, `Rightwards` | Local directional fold laws. |
| Foldr functions are leftwards; foldl functions are rightwards | Mechanized | `foldr_is_leftwards`, `foldl_is_rightwards` | Basic direction. |
| Uniqueness of foldr/foldl from empty value | Mechanized | `leftwards_eq_foldr_of_empty`, `rightwards_eq_foldl_of_empty` | Useful proof spine. |
| First homomorphism theorem, factorization | Mechanized | `theorem_3_3_first_homomorphism_factorization` | `h = reduce combine after map singleton`. |
| First homomorphism theorem, converse | Mechanized | `theorem_3_3_first_homomorphism_converse_monoid` | Monoid form. |
| Second homomorphism theorem | Mechanized | `theorem_3_4_second_homomorphism_leftwards`, `theorem_3_4_second_homomorphism_rightwards` | Homomorphisms are both directional. |
| Lemma 4.2, range representative function | Mechanized | `RangeSection`, `lemma_4_2_classical_range_section` | Classical section; computability refinement is separate. |
| Lemma 4.3, kernel congruence iff homomorphic | Mechanized | `ConcatKernelCongruent`, `lemma_4_3_homomorphic_iff_kernel_congruent` | The algebraic heart of the third theorem. |
| Injective functions are homomorphic | Mechanized | `injective_function_is_homomorphic` | Corollary of Lemma 4.3. |
| Third Homomorphism Theorem | Mechanized | `theorem_4_1_third_homomorphism` | Leftwards plus rightwards implies homomorphic. |
| Parenthesization/tree invariance | Mechanized | `theorem_4_1_parenthesization_invariance`, `gibbons1996_ordered_schedule_invariance` | C-TreePO-facing consequence. |
| Sorting application: foldr/foldl sorting implies homomorphic sort | Mechanized | `section_5_sort_homomorphic_of_foldr_foldl` | Theorem schema. |
| Insertion-sort foldr and foldl presentations | Mechanized | `insertionSort_eq_foldr_orderedInsert`, `insertionSort_eq_foldl_orderedInsert` | Uses mathlib list sorting. |
| Inefficient merge `sort (u ++ v)` is homomorphic | Mechanized | `insertionSort_inefficientMerge_homomorphic` | Proof-chosen merge. |
| Standard merge equations and sortedness preservation | Mechanized | `standardMerge_*`, `standardMerge_pairwise` | Uses mathlib `List.merge`. |
| Merge sort equals insertion sort extensionally | Mechanized | `mergeSort_eq_insertionSort` | Bridges to standard mergesort. |
| Runtime statements: insertion sort quadratic, merge linear, balanced merge sort `n log n` | Mechanized reference package | `Section5RuntimeClaims`, `section_5_runtime_claims_extract`, `linearGrowth_linearTime`, `quadraticGrowth_quadraticTime`, `nLogNGrowth_nLogNTime`, `insertionSortReferenceCostModel`, `standardMergeReferenceCostModel`, `mergeSortReferenceCostModel`, `referenceSection5RuntimeClaims`, `gibbons1996_00t_referenceSection5RuntimeClaims`, `ctreepo_gibbons1996_referenceSection5RuntimeClaims` | Big-O obligations and reference growth/cost-model inhabitants exist. Low-level operation-count semantics for Lean's executable implementations remains separate. |
| Computable/enumerable construction of the range section | Planned | `ComputableRangeSection` or citation schema | Not needed for C-TreePO; useful only for a fully constructive Gibbons pass. |

### Remaining Work

Gibbons is effectively complete for the mergeable-sketch bridge.  Section 5 now
has a checked reference Big-O cost package for insertion sort, linear merge, and
`n log n` merge sort.  The remaining runtime work would be a lower-level
operation-count semantics for the concrete executable definitions.  The
computability refinement of Lemma 4.2 can stay out of scope unless we decide to
formalize constructive enumeration.

## 2. Gray et al. 1997: Data Cube

Target module:

- `FormalProbability/ML/MergeableSummaries/Gray1997.lean`

Dedicated map:

- `lean3/docs/GRAY1997_PROOF_MAP.md`

### Claim Inventory

| Paper claim | Status | Lean target or existing name | Notes |
|---|---|---|---|
| `ALL` marker and cube addresses | Mechanized | `AllValue`, `CubeAddress`, `maskAddress`, `allValueSemantics`, `addressSemantics` | `none` represents `ALL`. |
| GROUP BY core and grand total | Mechanized | `groupByMask`, `totalMask`, `maskAddress_groupBy`, `maskAddress_total` | Basic address semantics. |
| Cube masks are the powerset of dimensions | Mechanized | `CubeMask`, `concreteDimensionSet`, `cubeMaskSetEquiv` | Formal powerset equivalence. |
| Cube mask count `2^N` | Mechanized | `cubeMask_card` | Uses `Fintype.card_fun`. |
| Super-aggregate masks `2^N - 1` beyond GROUP BY | Mechanized | `SuperAggregateMask`, `superAggregateMask_card` | Excludes the core GROUP BY mask. |
| Homogeneous address count `(C+1)^N` | Mechanized | `cubeAddress_card` | Extra `+1` is `ALL`. |
| Heterogeneous address count `prod_i (C_i+1)` | Mechanized | `CubeAddressD`, `cubeAddressD_card` | Uses `Fintype.card_pi`. |
| Direct cube update surface `2^N` per tuple | Mechanized | `directCubeUpdatesPerTuple_eq_pow`, `directCubeUpdateCalls`, `directCubeUpdateCalls_eq` | Exact per-tuple and total tuple-count vocabulary. |
| ROLLUP prefix chain | Mechanized | `rollupPrefixMask`, `rollupPrefixMask_injective`, `rollupLevel_card`, `directRollupUpdatesPerTuple_eq_succ` | Ordered dimensions. |
| ROLLUP super-aggregate levels `n` beyond GROUP BY | Mechanized | `RollupSuperLevel`, `rollupSuperLevel_card`, `directRollupSuperAggregateUpdatesPerTuple_eq`, `directRollupUpdatesPerTuple_le_directCubeUpdatesPerTuple`, `directRollupSuperAggregateUpdatesPerTuple_le_directCubeSuperAggregateUpdatesPerTuple`, `gray1997_06k_directRollupUpdates_le_directCubeUpdates`, `gray1997_06l_directRollupSuperAggregateUpdates_le_directCubeSuperAggregateUpdates` | Direct ROLLUP count and checked comparison with direct CUBE update counts. |
| GROUP/ROLLUP/CUBE operator algebra | Mechanized | `AggregationOperator.compose`, `cube_of_rollup`, `rollup_of_groupBy`, `compose_assoc`, `compose_comm` | Shape algebra. |
| User-defined aggregate callbacks | Mechanized | `UserDefinedAggregate`, `run`, `handle` | `start`/`next`/`finish` scratchpad. |
| COUNT, SUM, MIN, MAX are distributive | Mechanized | `countDistributiveAggregate`, `natSumDistributiveAggregate`, `minDistributiveAggregateWithTop`, `maxDistributiveAggregateWithBot` | Totalized empty cases for min/max. |
| Distributive implies algebraic | Mechanized | `distributive_is_algebraic` | Output-as-state. |
| AVG is algebraic via `(sum,count)` | Mechanized | `AverageState`, `averageAlgebraicAggregateNat` | Fixed-size state. |
| Scalar average is not distributive | Mechanized | `averageRat_not_distributive_scalar`, `averageRat_not_distributive_oracle` | Concrete kernel counterexample. |
| Dynamic insert/delete maintenance interface | Mechanized | `DynamicAggregate`, `DynamicAggregate.update_front_correct` | Update is delete plus insert. |
| COUNT and SUM support dynamic maintenance | Mechanized | `countDynamicAggregate`, `natSumDynamicAggregate` | Exact insert/delete laws. |
| Scalar MAX supports insert but not delete | Mechanized | `maxNat_insert_correct`, `maxNat_no_scalar_delete_front` | Formalizes Section 6 warning. |
| Holistic predicate | Mechanized | `HasFixedStateRealization`, `HolisticAggregate` | Predicate-level taxonomy. |
| Contextual finite-state lower-bound schema | Mechanized | `ContextuallySeparated`, `state_card_lower_bound_of_contextual_separation` | Reusable lower-bound tool. |
| Boolean mode no finite exact state | Mechanized | `modeBool_state_card_lower_bound`, `modeBool_no_finite_state_homomorphic_realization` | Concrete mode-style holistic example. |
| SQL syntax and catalog semantics | Out of scope | none | Parser/catalog layer, not mergeability. |
| Physical cube computation plans | Out of scope | none | Systems/cost-model layer. |
| Full communication lower bounds for median/rank | Partial | `ContextuallySeparated`, `medianMajorityBool_state_card_lower_bound`, `medianMajorityBool_no_finite_state_homomorphic_realization` | Boolean median/majority finite-state lower bound is done; broader SQL median/rank communication lower bounds remain future work. |

### Remaining Work

Gray is complete for the algebraic mergeable-sketch layer.  The Boolean
median/majority finite-state lower bound now instantiates the holistic/rank
warning, and direct ROLLUP-vs-CUBE update-count comparisons are checked.
Optional follow-up: add broader SQL median/rank communication lower bounds and
physical query-plan cost models.

## 3. Feldman et al. 2006/2008: MUD Aggregation

Current core Lean surface:

- `MUDAggregator` in `FormalProbability/ML/MergeableSummaries.lean`
- `Feldman2008.lean` for item-level MUD trees, streaming interfaces,
  polylog cost vocabulary, representative-state streaming-to-MUD, SCM
  protocols, MUD-to-streaming inclusion, concrete Set Parity/Symmetric Index
  problem surfaces, and remaining lower-bound schemas
- chronology aliases in `FormalProbability/ML/MergeableSummaries/LiteratureChronology.lean`

Target module:

- `FormalProbability/ML/MergeableSummaries/Feldman2008.lean`

Dedicated map:

- `lean3/docs/FELDMAN2008_PROOF_MAP.md`

### Claim Inventory

| Paper claim | Status | Lean target or existing name | Notes |
|---|---|---|---|
| MUD algorithm is a triple `(Phi, op, eta)` | Mechanized | `MUDAggregator.mapItem`, `merge`, `readout`; `Feldman2008.CostedMUDAlgorithm` | Costed wrapper carries communication/space/time fields. |
| General paper-MUD model without algebraic state laws | Mechanized | `Feldman2008.GeneralMUDAlgorithm` | Correct target for Theorem 1. |
| Local map produces a state for one item | Mechanized | `MUDAggregator.mapItem` | Existing. |
| Arbitrary computation trees should give same answer | Mechanized | `Feldman2008.ComputationTree.evalState_eq_build_data`, `feldman2008_07_item_tree_state_eq_build` | Explicit paper tree semantics now exists. |
| Associativity and commutativity suffice for permutation invariance | Mechanized | `merge_assoc`, `merge_comm`, `build_perm` | This is the C-TreePO-relevant theorem. |
| MUD build is homomorphic over list append | Mechanized | `MUDAggregator.build_append`, `feldman2006_01_mud_build_append` | Ordered list representation of state folding. |
| Canonical MUD states are merge-closed | Mechanized | `MUDAggregator.mergeClosed`, `feldman2006_02_mud_merge_closed` | State-level merge closure. |
| MUD is a state-level mergeable summary | Mechanized | `MUDAggregator.toStateLevelSummary`, `feldman2006_05_mud_state_level_mergeable` | Already bridges to Agarwal-style summaries. |
| MUD readout is permutation invariant | Mechanized | `MUDAggregator.readout_perm`, `feldman2006_04_mud_readout_permutation_invariant` | Existing. |
| Symmetric function definition | Mechanized | `Feldman2008.SymmetricFunction`, `feldman2008_06_symmetric_function_iff` | `f xs = f ys` when `xs.Perm ys`. |
| Class `MUD` for symmetric functions with polylog state/communication | Mechanized | `Feldman2008.CostedMUDAlgorithm`, `PolylogMUDComputable` | Uses asymptotic obligations rather than bit-level machines. |
| Streaming class `SS` | Mechanized | `Feldman2008.StreamingAlgorithm`, `CostedStreamingAlgorithm`, `PolylogStreamingComputable` | Used by Theorem 1 statement. |
| Lemma 1, streaming-state congruence under continuation | Mechanized | `Feldman2008.StreamingAlgorithm.lemma1_streaming_state_congruence_append`, `feldman2008_10_streaming_state_congruence_append` | If two prefixes produce same state, appending the same suffix preserves output equality. |
| Readout-context equivalence and frontier replacement algebra | Mechanized | `ReadoutContextEq`, `readoutContextEq_append_of_symmetric`, `readoutContextEq_pair_replacement_of_run_eq` | Uses mathlib list permutations to formalize the paper's block-swapping argument. |
| Lemma 2, merging streaming states by representative search | Mechanized semantically | `lemma2_representative_merge_exists`, `representativeMerge_spec`, `feldman2008_10b_representative_merge_exists` | Machine-space implementation remains future work. |
| Theorem 1, deterministic symmetric streaming equals MUD up to `O(g(n)^2)` state | Mechanized semantically | `deterministic_streaming_to_representative_mud`, `PolylogRate.square`, `polylog_streaming_subset_general_mud`, `theorem1_deterministic_streaming_to_mud_semantic`, `feldman2008_13b_theorem1_deterministic_streaming_to_mud_semantic`, `feldman2008_13c_polylog_streaming_subset_general_mud` | Proves the general paper-MUD construction and mathlib Big-O polylog-square closure; Savitch machine-space accounting remains external. |
| Streaming can simulate MUD | Mechanized | `Feldman2008.streamingFromMUD_run_eq_build`, `Feldman2008.mud_polylog_subset_streaming`, `feldman2008_12_mud_polylog_subset_streaming` | Easier direction; traverse input and fold MUD states. |
| Theorem 2, SS computed in simultaneous communication model | Mechanized | `SCMProtocol`, `scmFromStreaming`, `scmFromStreaming_computes`, `theorem2_streaming_to_scm_semantic`, `polylog_streaming_subset_scm` | Protocol construction is formalized. |
| SCM lower-bound transport | Mechanized | `SCMCommunicationLowerBound`, `SCMCommunicationLowerBoundOnPromise`, `SuperPolylogRate`, `not_polylog_scm_of_lower_bound`, `not_polylog_scm_on_promise_of_lower_bound`, `not_polylog_streaming_of_scm_lower_bound` | Converts super-polylogarithmic SCM lower bounds into no-polylog protocol/streaming consequences. |
| Randomized classes `rSS`, `rMUD` | Partial | `PublicRandomStreamingFamily`, `PublicRandomMUDFamily`, `PublicRandomGeneralMUDFamily`, `PublicRandomStreamingFamily.SuccessSet`, `PublicRandomStreamingFamily.SuccessProbability`, `PublicRandomStreamingFamily.ComputesWithSuccessAtLeast`, `PublicRandomGeneralMUDFamily.SuccessProbabilityOnTree`, `publicRandomRepresentativeMUDFromStreaming`, `public_randomness_seedwise_general_mud` | Public seedwise interfaces, success-set bookkeeping, exact seedwise success-probability wrappers, and the seedwise streaming-to-general-MUD construction are formalized; nontrivial bounded-error/randomized lower-bound probability proofs remain external. |
| Deterministic finite-message equality lower bound | Mechanized | `FiniteTwoPartyProtocol`, `BitAccountedTwoPartyProtocol`, `boolVectorEquality_sendA_injective_finite`, `boolVectorEquality_messageA_card_lower`, `boolVectorEquality_bitsA_lower`, `BitAccountedEqualityProtocolFamily.linear_bigO_lower` | Equality forces injective messages, `2^n` messages, and `Omega(n)` bit-accounted deterministic communication. |
| Finite Set Parity target and equality reduction for Theorem 3 | Mechanized | `boolVectorEquality`, `finSetParity`, `finSetParityRecords`, `finSetParity_two_vectors_eq`, `finSetParity_symmetric`, `equalityProtocolFromFinSetParity_computes`, `finSetParity_scm_lower_bound_of_equality`, `finiteSetParity_scm_sqrt_lower_bound_of_equality`, `equalityBitProtocolFromFinSetParity_computes`, `finSetParity_bitAccounted_bitsA_lower`, `BitAccountedFinSetParitySCMFamily.linear_bigO_lower` | The split-stream reduction from Boolean-vector equality to finite Set Parity is checked, including deterministic bit-accounted `Omega(n)` finite Set Parity families; the randomized equality lower bound remains external. |
| Private-coin Set Parity reduction for Theorem 3 | Mechanized | `PrivateCoinBitAccountedTwoPartyProtocol`, `PrivateCoinBitAccountedSCMComputesWithSuccess`, `privateCoinEqualityProtocolFromFinSetParity_successCount`, `privateCoinEqualityProtocolFromFinSetParity_computesWithSuccess`, `PrivateCoinBitAccountedFinSetParitySCMFamily.toEqualityFamily`, `privateCoinFinSetParity_scm_sqrt_lower_bound_of_equality` | Finite private seed-count model and bounded-error success preservation are mechanized; randomized equality lower-bound proof remains external. |
| Nat-index Set Parity target for Theorem 3 | Mechanized surface | `setParity`, `setParity_symmetric`, `setParity_scm_sqrt_lower_bound_statement` | Total unbounded-index function and symmetry are mechanized; deterministic and private-coin finite Set Parity reductions are mechanized. |
| Theorem 3, randomized separation | Citation schema | `Feldman2008.theorem3_private_randomness_separation_statement` | External private-randomness model and Set Parity lower-bound proof. |
| Promise classes `pSS`, `pMUD` | Mechanized | `PromiseSymmetric`, `PolylogStreamingComputableOnPromise`, `PolylogMUDComputableOnPromise` | Definitions are formalized. |
| Symmetric Index promise problem for Theorem 4 | Mechanized surface | `SymmetricIndexRecord`, `symmetricIndexCanonical`, `symmetricIndexDomain`, `symmetricIndexCanonical_mem_domain`, `symmetricIndexCanonical_readout_eq`, `symmetricIndex`, `symmetricIndex_promise_symmetric`, `symmetricIndex_scm_linear_lower_bound_statement` | Concrete promised records, domain membership, canonical readout correctness, totalized readout, and promise symmetry are checked; the promise linear SCM lower-bound proof remains external. |
| Theorem 4, promise separation `pMUD` proper subset `pSS` | Citation schema | `Feldman2008.theorem4_promise_separation_statement` | External Symmetric Index lower-bound proof. |
| Incomplete-output classes `iSS`, `iMUD` | Mechanized | `IndeterminateSymmetric`, `PolylogStreamingComputableIndeterminate`, `PolylogMUDComputableIndeterminate` | Definitions are formalized. |
| Theorem 5, incomplete-output separation | Citation schema | `Feldman2008.theorem5_indeterminate_separation_statement` | Follows from Theorem 4 in paper. |
| Polylog communication/space terminology | Mechanized | `PolylogRate`, `AtLeastLogRate`, `squareRate`, `PolylogRate.square`, `sqrtRate`, `linearRate` | Uses mathlib Big-O with `Nat -> Real` cost functions and closure under squaring. |

### Remaining Work for Feldman

1. Add a finite-state machine/reachability layer that accounts for the
   deterministic `O(g(n)^2)` Savitch implementation of `representativeMerge`.
2. Finish the nonempty/no-identity general-MUD-to-streaming inclusion, or
   restrict that direction to identity-equipped algebraic MUD, which is already
   mechanized.
3. Extend the probability-measure layer over public/private seeds from exact
   seedwise success to nontrivial bounded-error success and randomized
   communication lower-bound arguments.
4. Prove the randomized Boolean-vector equality and Symmetric Index SCM lower
   bounds locally.  Deterministic finite-message equality, finite Set Parity,
   and the private-coin Set Parity reduction are now mechanized; the randomized
   equality lower-bound proof and the Symmetric Index promise lower-bound proof
   remain external.

## 4. Flajolet et al. 2007: HyperLogLog

Current core Lean surface:

- `HLLRegisters` in `FormalProbability/ML/MergeableSummaries.lean`
- `Flajolet2007` in
  `FormalProbability/ML/MergeableSummaries/Flajolet2007.lean`
- chronology aliases in `LiteratureChronology.lean`

Dedicated target module:

- `FormalProbability/ML/MergeableSummaries/Flajolet2007.lean` (implemented)

Dedicated map:

- `lean3/docs/FLAJOLET2007_PROOF_MAP.md`

### Claim Inventory

| Paper claim | Status | Lean target or existing name | Notes |
|---|---|---|---|
| Hash values as bucket/suffix observations | Mechanized | `Flajolet2007.HashObservation` | Abstracts the random hash source into register bucket plus suffix bits. |
| `rho(s)` is position of first `1` | Mechanized | `Flajolet2007.rho`, `Flajolet2007.rho_positive_statement` | One-indexed first-one position with sentinel `length+1` for all-zero suffixes. |
| Register index from prefix bits | Mechanized | `Flajolet2007.bitsToNat`, `Flajolet2007.bitsToNat_lt_two_pow_length`, `Flajolet2007.bucketOfPrefix`, `Flajolet2007.HashWord.bucket` | Fixed-width prefix bits are parsed into `Fin (2^p)`. |
| HLL register state | Mechanized | `HLLRegisters m` | Vector of `m` natural registers. |
| Empty register state | Mechanized | `HLLRegisters.empty` | All zero registers. |
| Singleton item update | Mechanized | `HLLRegisters.singleton`, `HLLRegisters.update`, `regs_update_bucket`, `regs_update_of_ne` | Uses supplied bucket/rank and proves the one-bucket update semantics. |
| Stream build by folding updates | Mechanized | `HLLRegisters.build` | Existing. |
| Merge is pointwise max | Mechanized | `HLLRegisters.merge` | Exact state merge. |
| Merge identity laws | Mechanized | `merge_empty_left`, `merge_empty_right` | Existing. |
| Merge associativity | Mechanized | `merge_associative`, `flajolet2007_01_hll_merge_associative` | Existing. |
| Merge commutativity | Mechanized | `merge_commutative`, `flajolet2007_02_hll_merge_commutative` | Existing. |
| Merge idempotence | Mechanized | `merge_idempotent`, `flajolet2007_03_hll_merge_idempotent` | Existing. |
| Build homomorphism over append | Mechanized | `HLLRegisters.build_append`, `flajolet2007_04_hll_build_append` | Existing. |
| HLL is state-level mergeable for any readout | Mechanized | `flajolet2007_05_hll_state_level_mergeable`, `flajolet2007_10_hll_hash_state_level_mergeable` | Generic bucket/rank and paper hash-observation variants. |
| Precision parameter gives `m = 2^p` registers | Mechanized | `hllRegisterCount`, `flajolet2007_06_hll_register_count_p14` | Existing for p=14. |
| Raw estimator `E = alpha_m m^2 Z` | Mechanized | `HLLRegisters.alpha`, `HLLRegisters.rawEstimator` | Readout over state, not a scalar merge law. |
| Harmonic mean indicator `Z` | Mechanized | `HLLRegisters.indicatorZ`, `HLLRegisters.inversePowerSum`, `flajolet2007_11_hll_indicatorZ_empty` | Register readout denominator and empty-state sanity theorem. |
| Small-range correction by linear counting | Mechanized as readout definition | `HLLRegisters.linearCountingCorrection` | Program-level correction, not core merge law. |
| Large-range correction | Mechanized as readout definition | `HLLRegisters.largeRangeCorrection` | Program-level correction. |
| Ideal hash model for HLL streams | Mechanized deterministic and random-law interfaces | `Flajolet2007.IdealHashFamily`, `IdealHashFamily.build`, `IdealHashFamily.build_append`, `flajolet2007_10b_idealHash_state_level_mergeable`, `Flajolet2007.RandomIdealHashFamily`, `RandomIdealHashFamily.seedFamily_build_append`, `RandomIdealHashFamily.seedFamily_hierarchical`, `flajolet2007_10c_randomIdealHash_seedFamily_build_append`, `flajolet2007_10d_randomIdealHash_seedFamily_hierarchical` | The hash-to-state pipeline and seedwise random ideal-hash algebra are formalized; existence/independence and analytic probability laws of the hash source remain external. |
| Theorem 1(i), asymptotic almost unbiased | Citation schema | `Flajolet2007.AsymptoticallyAlmostUnbiased`, `flajolet2007_14_theorem1_stochasticEstimatorClaims` | Uses mathlib little-o notation; analytic proof remains outside this pass. |
| Theorem 1(ii), standard error constant | Citation schema plus proved Big-O weakening | `Flajolet2007.RelativeStandardErrorAsymptotic`, `Flajolet2007.RelativeStandardErrorBigO`, `Flajolet2007.relativeStandardErrorBigO_of_asymptotic`, `Flajolet2007.StochasticEstimatorClaims.toBigOClaims` | Uses mathlib asymptotic equivalence and Big-O notation; the implication from equivalence to Big-O is checked. |
| `1.04 / sqrt(m)` relative standard error | Mechanized as formula | `hllRelativeStandardError`, `Flajolet2007.relativeStandardErrorOfRegisterCount`, `flajolet2007_13_hll_relativeStandardError_registerCount` | Precision and register-count versions are linked. |
| `p = 14` example RSE is under one percent | Mechanized | `flajolet2007_hll_rse_p14_exact`, `flajolet2007_hll_rse_p14_under_one_percent`, `flajolet2007_07b_hll_rse_p14_exact`, `flajolet2007_07c_hll_rse_p14_under_one_percent` | Proves the corrected prose numerics: `13/1600 < 1/100`. |
| Proposition 1, exact expectation for fixed cardinality | Citation schema | `Flajolet2007.FixedCardinalityIndicatorExpectation` | Heavy finite sum/product expression remains a parameter. |
| Poissonized series identity layer | Mechanized package | `Flajolet2007.poissonWeight`, `Flajolet2007.poissonizedSeries`, `Flajolet2007.PoissonizedBySeries`, `Flajolet2007.PoissonizationDepoissonizationAnalysis` | Formalizes the Poisson-mixture transform and bundles it with the analytic asymptotic/depoissonization obligations. |
| Proposition 2, Poisson expectation asymptotic | Citation schema | `Flajolet2007.PoissonIndicatorExpectationAsymptotic` | Mellin analysis. |
| Lemma 1, Mellin/local integrand asymptotics | Citation schema | `mellin_integrand_asymptotic` | External analytic theorem. |
| Proposition 3, depoissonization | Citation schema plus transfer theorem | `Flajolet2007.DepoissonizationTransfer`, `Flajolet2007.fixedCardinality_asymptotic_of_poisson_depoissonization`, `Flajolet2007.fixedCardinality_asymptotic_of_poissonization_analysis` | The composition from poissonized asymptotic plus transfer to fixed-cardinality asymptotic is checked, including the packaged Poisson-series analysis; analytic hypotheses remain external. |
| Second moment and variance asymptotics | Citation schema | `Flajolet2007.IndicatorSecondMomentAsymptotic`, `Flajolet2007.VarianceAsymptotic` | Section 3. |
| CLT/simulation discussion | Out of scope | none | Empirical validation, not proof-critical. |

### Remaining Lean Work for Flajolet

1. Connect the random ideal-hash law interface to concrete independence and
   distributional existence results, then to the existing Poissonized-series
   package.
2. Mechanize the Mellin-transform estimates and analytic depoissonization proof, or
   import them from a dedicated probability/analytic-combinatorics library.
3. Keep the exact max-register merge algebra as the fully mechanized C-TreePO
   bridge.

## 5. Agarwal et al. 2012/2013: Mergeable Summaries

Current core Lean surface:

- `StateLevelMergeableSummary`
- `SizedMergeableQuerySketch`
- `RandomizedSizedMergeableQuerySketch`
- `FullMergeable`, `OneWayMergeable`, `EqualSizeMergeable`
- `Agarwal2013` in
  `FormalProbability/ML/MergeableSummaries/Agarwal2013.lean`
- `linearMergeableSketch`
- algorithm-existence schemas near the end of
  `FormalProbability/ML/MergeableSummaries.lean`
- `GK.lean` for one-way mergeability of GK-style quantile summaries
- `Complexity.lean` for asymptotic vocabulary

Dedicated target module:

- `FormalProbability/ML/MergeableSummaries/Agarwal2013.lean` (implemented)

Dedicated map:

- `lean3/docs/AGARWAL2013_PROOF_MAP.md` (implemented)

### Claim Inventory

| Paper claim | Status | Lean target or existing name | Notes |
|---|---|---|---|
| Summary method may be one-to-many | Mechanized | `ValidSketch.valid`, `Agarwal2013.SummaryMethod` | Validity relation captures non-unique outputs. |
| Size function `k(n, epsilon)` | Mechanized as vocabulary | `Agarwal2013.SizeProfile`, `Agarwal2013.HasEpsilonSizeRate` | Depends on existing `HasSizeRateFn`. |
| Mergeable summary definition | Mechanized | `StateLevelMergeableSummary`, `MergeClosed`, `HierarchicalMergeable` | Core state-level condition. |
| Full mergeability | Mechanized | `FullMergeable` | Existing. |
| One-way mergeability | Mechanized | `OneWayMergeable`; `GK.lean` | Existing interface. |
| Equal-size/restricted mergeability | Mechanized | `EqualSizeMergeable` | Existing interface. |
| Linear sketches are mergeable | Mechanized | `linearMergeableSketch`, `linearMergeableSketch_fullMergeable` | Generic additive merge surface. |
| Count-Min additive table mergeability | Mechanized | `CountMin.Table.state_level_mergeable`, `Agarwal2013.countMin_state_level_mergeable` | Concrete `d × w` counter tables with pointwise-add merge. |
| Count-Min additive merge is not idempotent | Mechanized | `CountMin.Table.merge_not_idempotent_of_pos`, `Agarwal2013.countMin_merge_not_idempotent_of_pos` | Formal contrast with HLL max-register idempotence. |
| Frequency estimation problem | Mechanized as spec | `Agarwal2013.FrequencyEstimationSpec` | Query-error predicate. |
| Heavy hitters problem | Mechanized as spec | `Agarwal2013.HeavyHittersSpec` | Frequency threshold with false-positive band. |
| Quantile summary problem | Mechanized as spec | `Agarwal2013.QuantileSummarySpec` | Rank error. |
| Epsilon-approximation of range spaces | Mechanized as spec, mathlib-backed finite VC trace layer, finite-trace failure union bound, and exact witness | `Agarwal2013.EpsilonApproximationSpec`, `Agarwal2013.FiniteRangeSpace`, `Agarwal2013.FiniteRangeSpace.shattered_card_le_vcDim`, `Agarwal2013.FiniteRangeSpace.trace_card_le_sauerShelah`, `Agarwal2013.FiniteRangeSpace.measureReal_traceFailureEvent_le_sauerShelah_mul`, `Agarwal2013.exactStateLevelEpsilonApproximationSpec`, `Agarwal2013.geometricRangeCount`, `Agarwal2013.exactRangeSpaceEpsilonApproximationSpec`, `Agarwal2013.exactRangeSpaceSizedMergeableQuerySketch` | Abstract range-count error plus a linear-size exact all-points witness; finite VC/Sauer-Shelah growth and the union-bound upgrade from pointwise trace tails to uniform failure are mechanized; compact discrepancy construction remains external. |
| Epsilon-kernel / width approximation | Mechanized as width geometry, translation/scaling laws, spec, hierarchical lift, and exact witness | `Agarwal2013.pointDot`, `Agarwal2013.pointDot_translatePoint`, `Agarwal2013.pointDot_scalePoint`, `Agarwal2013.maxProjectionState_append`, `Agarwal2013.minProjectionState_append`, `Agarwal2013.directionalWidth_append`, `Agarwal2013.directionalWidth_translateStream`, `Agarwal2013.directionalWidth_scaleStream_of_nonneg`, `Agarwal2013.WidthApproximationGuaranteeFnD`, `Agarwal2013.EpsilonKernelSpec`, `Agarwal2013.epsilonKernel_hierarchical`, `Agarwal2013.exactEpsilonKernelSpec`, `Agarwal2013.exactEpsilonKernel_tree_widthError` | Common-reference-frame projection merge laws, translation/scaling laws, and the tree width-error layer are checked; compact ε-kernel size proof remains external. |
| MG summary definition | Executable bookkeeping mechanized | `HeavyHitters.MisraGries.update`, `HeavyHitters.MisraGries.build`, `HeavyHitters.totalCounterMass`, `HeavyHitters.MisraGries.build_positiveCounts`, `HeavyHitters.MisraGries.tracedBuild_potential_le_length`, `HeavyHitters.MisraGries.tracedBuild_debt_mul_succ_le_length`, `HeavyHitters.MGAlgorithm` | Concrete update/build, capacity, positivity, total counter-mass accounting, and the traced decrement-debt induction are mechanized; full query sandwich and merge correctness remain in the algorithm bundle. |
| MG Lemma 1, counter lower/upper frequency bound | Bundled as invariant plus executable debt induction | `HeavyHitters.MGAlgorithm.frequency_error`, `Agarwal2013.misraGries_frequency_error`, `Agarwal2013.executableMisraGries_debt_mul_succ_le_length` | The standard `k+1` decrement-charge accounting is proved; the final algorithm-specific query sandwich remains bundled. |
| MG merge algorithm | Bundle mechanized | `HeavyHitters.MGAlgorithm.merge` | Merge correctness supplied by `merge_valid`. |
| Theorem 1, MG mergeable with size `O(1/epsilon)` | Capacity/mass/debt executable; mergeability bundled | `HeavyHitters.MisraGries.build_boundedBy`, `HeavyHitters.MisraGries.update_totalCounterMass_le_succ`, `HeavyHitters.MisraGries.build_totalCounterMass_le_length`, `HeavyHitters.MisraGries.tracedBuild_debt_mul_succ_le_length`, `Agarwal2013.executableMisraGries_boundedBy`, `Agarwal2013.executableMisraGries_totalCounterMass_le_length`, `Agarwal2013.executableMisraGries_debt_mul_succ_le_length`, `Agarwal2013.misraGries_theorem1_of_algorithm`, `Agarwal2013.misraGries_hierarchical` | Concrete capacity, mass, positivity, and decrement-debt accounting are proved; full frequency-error/mergeability theorem still uses invariant bundle. |
| SpaceSaving summary definition | Executable bookkeeping plus bundle mechanized | `HeavyHitters.SpaceSaving.update`, `HeavyHitters.SpaceSaving.build_boundedBy`, `HeavyHitters.SpaceSaving.build_totalCounterMass_le_length`, `HeavyHitters.SpaceSavingAlgorithm` | Capacity and stored-mass bookkeeping are mechanized; full SpaceSaving mergeability still follows by isomorphism transport. |
| MG-SpaceSaving isomorphism Lemma 2 | Mechanized as transport witness | `HeavyHitters.IsomorphicMGSpaceSaving` | The isomorphism object drives the corollary. |
| Corollary 1, SpaceSaving mergeable | Mechanized from isomorphism | `Agarwal2013.spaceSaving_hierarchical_of_isomorphism` | Follows from generic isomorphism transport. |
| Definition 1, one-way mergeability | Mechanized | `OneWayMergeable` | Existing. |
| Theorem 2, incrementally maintainable quantile implies one-way mergeable | Mechanized | `Agarwal2013.IncrementallyMaintainable.oneWayMergeable` | Generic proof by folding insertions. |
| Corollary 2, GK one-way mergeable | Mechanized from bundle | `Agarwal2013.gk_corollary2_oneWay` | Dedicated Agarwal alias added. |
| Executable GK state accounting | Mechanized | `GK.Executable.build_n`, `GK.Executable.build_gapMassValid`, `gk2001_01_executable_build_n`, `gk2001_02_executable_build_gapMassValid` | Keeps item count and tuple gap-mass invariants separate from GK's abstract theorem bundle. |
| KLL mergeable and optimal theorem bundles | Mechanized from bundle | `KLL.theorem4_of_algorithm`, `KLL.theorem5_of_algorithm`, `kll2016_01_theorem4_mergeable_variant_of_algorithm`, `kll2016_02_theorem5_optimal_variant_of_algorithm` | Exposes the mergeable variant and optimal-space variant as separate formal targets. |
| Executable KLL state accounting | Mechanized | `KLL.Executable.weightedCount_step`, `KLL.Executable.build_massValid`, `kll2016_03_executable_weightedCount_step`, `kll2016_04_executable_build_massValid` | Proves the executable transition preserves represented weighted stream mass. |
| State-level epsilon-approximation tree error | Mechanized | `StateLevelEpsilonApproximationGuaranteeFn`, `StateLevelEpsilonApproximationSpec`, `StateLevelEpsilonApproximationSpec.toSizedMergeableQuerySketch`, `stateLevelEpsilonApproximation_tree_error`, `exactStateLevelEpsilonApproximation_tree_error` | Generic theorem: once range-query error holds for every valid state, arbitrary merge-tree readout inherits the same ε-error bound; exact all-points state proves the interface is inhabited with zero error for any nonnegative ε. |
| Same-weight merge for one-dimensional epsilon-approximations | Mechanized state/tree layer | `Interval1D`, `intervalCount`, `intervalCount_append`, `EqualLengthSiblingTree`, `SameWeightIntervalApproximationSpec`, `sameWeightInterval_valid_on_equalLengthTree`, `sameWeightInterval_tree_error_on_equalLength` | Interval count additivity and equal-weight tree-error transport are checked. |
| Lemma 3, unbiased interval count after halving | Mechanized deterministic two-choice core | `paritySplit`, `sameWeightHalving_unbiased_interval_count`, `sameWeightHalving_interval_error_mean_zero`, `sameWeightHalving_interval_error_abs_le_one` | Uniform even/odd choice is represented by `twoPointMean`; the scaled estimate is unbiased and each parity choice has absolute over-count at most one. |
| Lemma 4, concentration over complete merge tree | Mechanized under explicit martingale hypotheses | `sameWeightHalving_level_error_abs_le`, `sameWeightHalvingHoeffdingDenominator`, `sameWeightHalving_hoeffdingDenominator_le`, `sameWeightHalving_root_error_to_epsilon_n_of_scale`, `SameWeightHalvingCompleteTreeProcess.hoeffding_tail`, `SameWeightHalvingCompleteTreeProcess.epsilon_n_tail` | The level-radius, denominator, final `h 2^m` to `εn` arithmetic, two-sided Azuma/Hoeffding tail, and final `εn` tail event are checked for an adapted bounded conditional-mean-zero complete-tree error process. |
| Theorem 3, same-weight mergeable 1D epsilon-approximation | Citation schema | `sameWeight_interval_epsilonApproximation` | Depends on Lemmas 3-4. |
| Theorem 4, mergeable 1D epsilon-approximation with log(epsilon n) size | Citation schema | `weighted_interval_epsilonApproximation` | Layered hierarchy. |
| Fact 1, random sample is mergeable epsilon-approximation | Mechanized as typed target plus finite VC union-bound layer | `RandomSampleEpsilonApproximationSpec`, `FiniteRangeSpace.measureReal_traceFailureEvent_le_sauerShelah_mul` | Natural randomized summary theorem; Lean now proves the finite-trace uniform-failure upgrade from per-trace tails. The sampling model and compact construction remain external. |
| Hybrid quantile scheme definitions | Mechanized bookkeeping surface | `HybridPromotion`, `HybridTraceOnlyMovesUp` | Complex full state remains future work; promotion events are explicit. |
| Lemma 5, points only move up in hierarchy | Mechanized bookkeeping invariant | `hybridTrace_level_monotone` | Deterministic invariant. |
| Lemma 6, promotion count bound | Planned | `hybrid_promotion_count_bound` | Arithmetic invariant. |
| Lemma 7, random buffer overcount concentration | Mechanized finite-level event composition | `hybridRandomBufferFailureEvent`, `hybridRandomBuffer_failure_bound`, `hybridRandomBuffer_failure_bound_uniform` | Per-level tails union-bound to a hierarchy-wide random-buffer failure bound; the per-level concentration source remains external. |
| Theorem 5, fully mergeable quantile summary size independent of n | Citation schema | `randomized_quantile_fully_mergeable` | Major theorem; typed statement first. |
| Higher-dimensional ranges Lemmas 8-10 | Citation schema | `range_sameWeight_unbiased`, `range_sameWeight_size_bound` | Needs range-space/VC-dimension layer. |
| Theorem 6, mergeable epsilon-approximations for VC/range spaces | Citation schema plus finite VC layer and exact witness | `theorem6_rangeSpaceEpsilonApproximation`, `FiniteRangeSpace.trace_card_le_sauerShelah`, `exactRangeSpaceEpsilonApproximationSpec`, `exactRangeSpaceSizedMergeableQuerySketch` | The error/merge layer is mechanized with a linear-size witness and the finite VC growth bound is mathlib-backed; the compact discrepancy construction is the external combinatorial geometry layer. |
| Epsilon-kernel definition and affine invariance | Mechanized width-query geometry plus translation/scaling laws | `pointDot`, `pointDot_translatePoint`, `pointDot_scalePoint`, `maxProjectionState_append`, `minProjectionState_append`, `directionalWidth_append`, `directionalWidth_translateStream`, `directionalWidth_scaleStream_of_nonneg`, `EpsilonKernelSpec`, `WidthApproximationGuaranteeFnD`, `StateLevelWidthApproximationGuaranteeFnD` | Projection-state merge laws, directional width, translation invariance, and nonnegative scaling are checked; full compactness/size proof remains a future geometry layer. |
| Epsilon-kernel common reference frame merge | Citation schema plus state-level lift, tree error, and exact witness | `theorem7_epsilonKernelCommonReferenceFrame`, `epsilonKernel_hierarchical`, `EpsilonKernelSpec.toSizedMergeableQuerySketch`, `epsilonKernel_tree_widthError`, `exactEpsilonKernelSpec`, `exactEpsilonKernel_tree_widthError` | The C-TreePO state-level conclusion and tree-level width-error transport are checked, including a linear-size all-points kernel witness. |
| Theorem 7, mergeable epsilon-kernel in common reference frame | Citation schema plus projection algebra and translation/scaling laws | `theorem7_epsilonKernelCommonReferenceFrame`, `directionalWidth_append`, `directionalWidth_translateStream`, `directionalWidth_scaleStream_of_nonneg` | External geometry details are now narrowed to compact ε-kernel construction and size claims. |
| Lower-bound discussion for deterministic quantiles | Citation schema | `deterministic_quantile_mergeable_lower_bound` | The paper discusses nonexistence evidence; formalize only exact stated claims. |

### Remaining Lean Work for Agarwal

1. Prove the final executable Misra-Gries query sandwich and merge-correctness
   theorem on top of the checked capacity/mass/positivity/debt induction.
2. Instantiate a concrete MG/SpaceSaving isomorphism witness beyond the current
   abstract transport theorem and checked SpaceSaving bookkeeping.
3. Use the mechanized complete-tree halving tail as the probabilistic input
   when instantiating concrete valid-state interval/range error invariants.
4. Add the compact ε-kernel geometry layer if the paper needs the full
   Theorem 7 size/affine proof rather than the current exact all-points
   valid-state/tree-error layer.
5. Align `GK.lean` with Theorem 2 / Corollary 2.
6. Replace the current compact interval, sampling, range-space, and ε-kernel
   theorem schemas with probability/geometry proofs when the supporting
   mathlib layer is available; the finite VC union-bound and hybrid buffer
   event-composition layers are already mechanized.

## Chronological Execution Plan

| Step | Work item | Expected result |
|---|---|---|
| 1 | Feldman module and proof map | MUD paper is represented beyond the current algebraic core, with symmetric/streaming/communication classes and theorem schemas. |
| 2 | Flajolet module and proof map | HLL max-register algebra stays mechanized; estimator/readout and analytic theorem statements become explicit. |
| 3 | Agarwal module and proof map | Generic mergeability definitions plus a concrete MG implementation target are centralized. |
| 4 | Misra-Gries implementation | First algorithm-specific Agarwal theorem, with frequency-error and mergeability proof. |
| 5 | SpaceSaving isomorphism | Formal transport theorem matching Agarwal Corollary 1. |
| 6 | Quantile/range/geometry schemas | Typed statements for the large randomized and geometric results, with proof obligations isolated. |
| 7 | C-TreePO bridge update | `MergeableReduction.lean` exposes the new theorem names, while paper docs point to both mechanized facts and citation schemas. |

## Boundary Decisions

- Fully mechanize algebraic state laws, tree invariance, permutation
  invariance, and concrete finite counterexamples.
- Use typed citation schemas for heavy analytic probability, communication
  complexity lower bounds, and computational geometry theorems until those
  libraries exist locally.
- Keep scalar-output composition separate from state-level composition in every
  theorem name and docstring.
- Keep ordered Gibbons/C-TreePO text composition separate from unordered MUD
  aggregation.
- Treat idempotence as optional and sketch-specific: HLL has it; additive
  sketches and most frequency summaries do not.
