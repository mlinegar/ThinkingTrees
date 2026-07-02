# Mergeable-Sketch Literature Proof Map

This is the core-proof inventory for the five mergeable-sketch papers used in
the C-TreePO algebraic story. The checked chronological Lean surface is:

- `FormalProbability/ML/MergeableSummaries/Literature.lean`
- `FormalProbability/ML/MergeableSummaries/LiteratureChronology.lean`
- `FormalProbability/ML/MergeableSummaries/Gibbons1996.lean`
- `FormalProbability/ML/MergeableSummaries/Gray1997.lean`
- `FormalProbability/ML/MergeableSummaries/Feldman2008.lean`
- `FormalProbability/ML/MergeableSummaries/Flajolet2007.lean`
- `FormalProbability/ML/MergeableSummaries/Agarwal2013.lean`

Dedicated paper proof maps:

- `lean3/docs/GIBBONS1996_PROOF_MAP.md`
- `lean3/docs/GRAY1997_PROOF_MAP.md`
- `lean3/docs/FELDMAN2008_PROOF_MAP.md`
- `lean3/docs/FLAJOLET2007_PROOF_MAP.md`
- `lean3/docs/AGARWAL2013_PROOF_MAP.md`

For the forward-looking inventory of all important formalizable statements,
including planned theorem names and citation-schema boundaries for the remaining
papers, see `lean3/docs/MERGEABLE_SKETCH_FULL_FORMALIZATION_ROADMAP.md`.

The reusable definitions and supporting proofs live in:

- `FormalProbability/ML/MergeableSummaries.lean`
- `FormalProbability/ML/MergeableSummaries/Gibbons1996.lean`
- `FormalProbability/ML/MergeableSummaries/Gray1997.lean`
- `FormalProbability/ML/MergeableSummaries/Feldman2008.lean`
- `FormalProbability/ML/MergeableSummaries/Flajolet2007.lean`
- `FormalProbability/ML/MergeableSummaries/CountMin.lean`
- `FormalProbability/ML/MergeableSummaries/Agarwal2013.lean`
- `FormalProbability/ML/MergeableSummaries/GKExecutable.lean`
- `FormalProbability/ML/MergeableSummaries/KLLExecutable.lean`
- `lean3/FormalProofs/OPT/MergeableReduction.lean`
- `lean3/FormalProofs/OPT/ClassicalSketchLocalLaws.lean`

## Core Proof Obligations

| Obligation | Lean surface | Purpose |
|---|---|---|
| Stream model | `Stream α := List α` | Common insertion-order representation for all sketch inputs. |
| Binary merge-tree model | `MergeTree`, `MergeTree.data`, `MergeTree.eval` | Represents shard reductions and parenthesization. |
| Valid state relation | `ValidSketch.valid`, `StateLevelMergeableSummary.valid` | Separates canonical build states from arbitrary states. |
| Merge closure | `MergeClosed` | Classical state-level mergeability condition. |
| Hierarchical mergeability | `HierarchicalMergeable`, `hierarchical_of_full` | Lifts pairwise merge closure to arbitrary binary trees. |
| Final readout correctness | `StateLevelMergeableSummary.QueryCorrect`, `query_tree_eq_oracle` | Formalizes that query/readout occurs after state merging. |
| Ordered schedule invariance | `OrderedListHomomorphism`, `orderedListHomomorphism_eval_eq`, `orderedListHomomorphism_schedule_invariant` | Handles ordered text without assuming commutativity. |
| Optional state laws | `MergeAssociative`, `MergeCommutative`, `MergeIdempotent` | Keeps associativity, commutativity, and idempotence separate. |
| State-vs-output distinction | `sketch_state_level_reduction_to_classical_mergeable`, `scalarDistinctCount_not_child_cardinality_mergeable` | Prevents the scalar-output homomorphism overclaim. |
| Dynamic maintenance | `Gray1997.DynamicAggregate`, `Gray1997.DynamicAggregate.update_front_correct`, `Gray1997.maxNat_no_scalar_delete_front`, `ctreepo_gray1997_max_no_scalar_delete_front` | Separates insert/update maintenance from exact delete maintenance when scalar state has lost history. |
| Finite-state lower bounds | `Gray1997.ContextuallySeparated`, `Gray1997.modeBool_state_card_lower_bound`, `Gray1997.modeBool_no_finite_state_homomorphic_realization`, `Gray1997.medianMajorityBool_state_card_lower_bound`, `Gray1997.medianMajorityBool_no_finite_state_homomorphic_realization`, `ctreepo_gray1997_contextual_state_lower_bound`, `ctreepo_gray1997_modeBool_state_card_lower_bound`, `ctreepo_gray1997_modeBool_no_finite_state_homomorphic_realization`, `ctreepo_gray1997_medianMajorityBool_state_card_lower_bound`, `ctreepo_gray1997_medianMajorityBool_no_finite_state_homomorphic_realization` | States the information-retention obligation for exact state-level mergeability: separated prefix families must inject into state, ruling out finite exact state for Boolean mode and a Boolean median/majority rank-style aggregate. |
| HLL/CMS contrast | `flajolet2007_03_hll_merge_idempotent`, `CountMin.Table.merge_not_idempotent_of_pos`, `ctreepo_agarwal2013_countMin_merge_not_idempotent_of_pos`, `hll_idempotent_cms_not_idempotent_when_nonempty` | Checks that idempotence is HLL-specific, not generic. |
| Interval/range tree-error transport | `Agarwal2013.StateLevelEpsilonApproximationSpec`, `Agarwal2013.stateLevelEpsilonApproximation_tree_error`, `Agarwal2013.exactStateLevelEpsilonApproximationSpec`, `Agarwal2013.exactRangeSpaceEpsilonApproximationSpec`, `Agarwal2013.exactRangeSpaceSizedMergeableQuerySketch`, `Agarwal2013.sameWeightInterval_tree_error_on_equalLength`, `ctreepo_agarwal2013_stateLevelEpsilonApproximation_tree_error`, `ctreepo_agarwal2013_exactStateLevelEpsilonApproximation_tree_error`, `ctreepo_agarwal2013_exactRangeSpaceSizedMergeableQuerySketch`, `ctreepo_agarwal2013_sameWeightInterval_tree_error_on_equalLength` | Shows that valid-state ε-approximation guarantees survive distributed merge-tree evaluation; exact all-points range summaries witness zero error for any nonnegative ε; same-weight interval summaries use equal-length sibling trees. |
| Same-weight halving core | `Agarwal2013.paritySplit`, `Agarwal2013.sameWeightHalving_unbiased_interval_count`, `Agarwal2013.sameWeightHalving_interval_error_abs_le_one`, `Agarwal2013.sameWeightHalving_level_error_abs_le`, `Agarwal2013.sameWeightHalving_hoeffdingDenominator_le`, `Agarwal2013.SameWeightHalvingCompleteTreeProcess.hoeffding_tail`, `Agarwal2013.SameWeightHalvingCompleteTreeProcess.epsilon_n_tail`, `ctreepo_agarwal2013_sameWeightHalving_unbiased_interval_count`, `ctreepo_agarwal2013_sameWeightHalving_root_error_to_epsilon_n_of_scale`, `ctreepo_agarwal2013_sameWeightHalving_completeTree_hoeffding_tail`, `ctreepo_agarwal2013_sameWeightHalving_completeTree_epsilon_n_tail` | Mechanizes Agarwal Lemma 3's unbiased/bounded same-weight interval halving and Lemma 4's complete-tree radius, denominator, root scaling, Azuma/Hoeffding tail, and `εn` tail under explicit martingale hypotheses. |
| Epsilon-kernel tree-width transport | `Agarwal2013.pointDot`, `Agarwal2013.maxProjectionState_append`, `Agarwal2013.minProjectionState_append`, `Agarwal2013.directionalWidth_append`, `Agarwal2013.EpsilonKernelSpec.toSizedMergeableQuerySketch`, `Agarwal2013.epsilonKernel_tree_widthError`, `Agarwal2013.exactEpsilonKernelSpec`, `Agarwal2013.exactEpsilonKernel_tree_widthError`, `ctreepo_agarwal2013_pointDot`, `ctreepo_agarwal2013_maxProjectionState_append`, `ctreepo_agarwal2013_minProjectionState_append`, `ctreepo_agarwal2013_directionalWidth_append`, `ctreepo_agarwal2013_epsilonKernel_toSizedMergeableQuerySketch`, `ctreepo_agarwal2013_epsilonKernel_tree_widthError`, `ctreepo_agarwal2013_exactEpsilonKernel_tree_widthError` | Bridges epsilon-kernel state/readout summaries into the generic sized mergeable sketch interface, proves common-reference-frame projection-state merge laws, and proves width-error transport through merge trees with an exact all-points witness. |
| Executable quantile-sketch witnesses | `gk2001_01_executable_build_n`, `gk2001_02_executable_build_gapMassValid`, `kll2016_03_executable_weightedCount_step`, `kll2016_04_executable_build_massValid`, `ctreepo_gk2001_executable_build_gapMassValid`, `ctreepo_kll2016_executable_build_massValid` | Keeps GK/KLL implementation invariants separate from the external concentration proofs. |

## Chronological Paper Map

| Order | Paper | Checked Lean names | What is fully formalized for C-TreePO |
|---|---|---|---|
| 1 | Gibbons 1996, *The Third Homomorphism Theorem* | `Gibbons1996.theorem_4_1_third_homomorphism`, `Gibbons1996.lemma_4_3_homomorphic_iff_kernel_congruent`, `Gibbons1996.longestSortedPrefixNat_not_homomorphic`, `Gibbons1996.insertionSort_inefficientMerge_homomorphic`, `Gibbons1996.mergeSort_eq_insertionSort`, `Gibbons1996.Section5RuntimeClaims`, `Gibbons1996.linearGrowth_linearTime`, `Gibbons1996.quadraticGrowth_quadraticTime`, `Gibbons1996.nLogNGrowth_nLogNTime`, `Gibbons1996.referenceSection5RuntimeClaims`, `gibbons1996_00i_third_homomorphism`, `gibbons1996_00m_inefficient_merge_homomorphic`, `gibbons1996_00p_runtime_claims_extract`, `gibbons1996_00q_linearGrowth_linearTime`, `gibbons1996_00r_quadraticGrowth_quadraticTime`, `gibbons1996_00s_nLogNGrowth_nLogNTime`, `gibbons1996_00t_referenceSection5RuntimeClaims`, `gibbons1996_01_ordered_tree_evaluation`, `gibbons1996_02_ordered_schedule_invariance`, `gibbons1996_03_associative_on_homomorphic_image` | Full algebraic theorem spine plus the introduction examples/non-example, a mathlib-backed Section 5 sorting/merge-sort surface, formal Big-O runtime predicates, proved reference-growth inhabitants, and an inhabited reference Section 5 cost package. Low-level operation-count semantics for Lean's executable sort implementations remain separate. |
| 2 | Gray et al. 1997, *Data Cube* | `Gray1997.cubeMaskSetEquiv`, `Gray1997.cubeMask_card`, `Gray1997.superAggregateMask_card`, `Gray1997.cubeAddress_card`, `Gray1997.cubeAddressD_card`, `Gray1997.rollupPrefixMask_injective`, `Gray1997.rollupSuperLevel_card`, `Gray1997.directRollupUpdatesPerTuple_le_directCubeUpdatesPerTuple`, `Gray1997.directRollupSuperAggregateUpdatesPerTuple_le_directCubeSuperAggregateUpdatesPerTuple`, `Gray1997.countDistributiveAggregate`, `Gray1997.averageAlgebraicAggregateNat`, `Gray1997.averageRat_not_distributive_oracle`, `Gray1997.DynamicAggregate`, `Gray1997.DynamicAggregate.update_front_correct`, `Gray1997.maxNat_no_scalar_delete_front`, `Gray1997.ContextuallySeparated`, `Gray1997.modeBool_state_card_lower_bound`, `Gray1997.modeBool_no_finite_state_homomorphic_realization`, `Gray1997.medianMajorityBool_state_card_lower_bound`, `Gray1997.medianMajorityBool_no_finite_state_homomorphic_realization`, `gray1997_01_distributive_is_algebraic`, `gray1997_05_cubeMask_card`, `gray1997_05b_superAggregateMask_card`, `gray1997_06a_cubeAddressD_card`, `gray1997_06d1_rollupSuperLevel_card`, `gray1997_06g_rollupPrefixMask_injective`, `gray1997_06k_directRollupUpdates_le_directCubeUpdates`, `gray1997_06l_directRollupSuperAggregateUpdates_le_directCubeSuperAggregateUpdates`, `gray1997_11_average_algebraic`, `gray1997_12b_average_not_distributive_oracle`, `gray1997_15_max_no_scalar_delete_front`, `gray1997_16_state_card_lower_bound_of_contextual_separation`, `gray1997_17_modeBool_state_card_lower_bound`, `gray1997_18_modeBool_no_finite_state_homomorphic_realization`, `gray1997_19_medianMajorityBool_state_card_lower_bound`, `gray1997_20_medianMajorityBool_no_finite_state_homomorphic_realization`, `ctreepo_gray1997_cubeMask_card`, `ctreepo_gray1997_superAggregateMask_card`, `ctreepo_gray1997_cubeAddressD_card`, `ctreepo_gray1997_rollupSuperLevel_card`, `ctreepo_gray1997_rollupPrefixMask_injective`, `ctreepo_gray1997_directRollupUpdates_le_directCubeUpdates`, `ctreepo_gray1997_directRollupSuperAggregateUpdates_le_directCubeSuperAggregateUpdates`, `ctreepo_gray1997_average_not_distributive_oracle`, `ctreepo_gray1997_max_no_scalar_delete_front`, `ctreepo_gray1997_modeBool_state_card_lower_bound`, `ctreepo_gray1997_modeBool_no_finite_state_homomorphic_realization`, `ctreepo_gray1997_medianMajorityBool_state_card_lower_bound`, `ctreepo_gray1997_medianMajorityBool_no_finite_state_homomorphic_realization` | Cube/ALL set semantics, powerset mask view, `2^N` mask count, `2^N-1` super-aggregate masks, `(C+1)^N` and `∏ᵢ(Cᵢ+1)` address counts, `n+1` ROLLUP prefix levels with `n` super-aggregate levels, direct ROLLUP-vs-CUBE update-count comparisons, GROUP/ROLLUP/CUBE algebra, distributive examples, algebraic average state, scalar-output average counterexample, dynamic maintenance interface with update-as-delete-plus-insert, scalar `MAX` delete impossibility, and contextual finite-state lower bounds ruling out exact finite-state Boolean mode and Boolean median/majority. |
| 3 | Feldman et al. 2006/2008, MUD aggregation | `Feldman2008.ComputationTree.evalState_eq_build_data`, `Feldman2008.StreamingAlgorithm.lemma1_streaming_state_congruence_append`, `Feldman2008.StreamingAlgorithm.lemma2_representative_merge_exists`, `Feldman2008.representativeMerge_spec`, `Feldman2008.PolylogRate.square`, `Feldman2008.deterministic_streaming_to_representative_mud`, `Feldman2008.representativeMUDFromStreaming_computesOnAllTrees`, `Feldman2008.polylog_streaming_subset_general_mud`, `Feldman2008.PublicRandomStreamingFamily.SuccessProbability`, `Feldman2008.PublicRandomStreamingFamily.ComputesWithSuccessAtLeast`, `Feldman2008.PublicRandomGeneralMUDFamily.SuccessProbabilityOnTree`, `Feldman2008.publicRandomRepresentativeMUDFromStreaming`, `Feldman2008.public_randomness_seedwise_general_mud`, `Feldman2008.theorem1_deterministic_streaming_to_mud_semantic`, `Feldman2008.SCMProtocol`, `Feldman2008.scmFromStreaming_computes`, `Feldman2008.theorem2_streaming_to_scm_semantic`, `Feldman2008.SCMCommunicationLowerBound`, `Feldman2008.not_polylog_streaming_of_scm_lower_bound`, `Feldman2008.boolVectorEquality`, `Feldman2008.boolVectorEquality_messageA_card_lower`, `Feldman2008.boolVectorEquality_bitsA_lower`, `Feldman2008.BitAccountedEqualityProtocolFamily.linear_bigO_lower`, `Feldman2008.finSetParity_two_vectors_eq`, `Feldman2008.finSetParity_scm_lower_bound_of_equality`, `Feldman2008.finSetParity_bitAccounted_bitsA_lower`, `Feldman2008.BitAccountedFinSetParitySCMFamily.linear_bigO_lower`, `Feldman2008.privateCoinEqualityProtocolFromFinSetParity_successCount`, `Feldman2008.privateCoinEqualityProtocolFromFinSetParity_computesWithSuccess`, `Feldman2008.privateCoinFinSetParity_scm_sqrt_lower_bound_of_equality`, `Feldman2008.finiteSetParity_scm_sqrt_lower_bound_of_equality`, `Feldman2008.setParity_symmetric`, `Feldman2008.setParity_scm_sqrt_lower_bound_statement`, `Feldman2008.symmetricIndexDomain`, `Feldman2008.symmetricIndexCanonical_mem_domain`, `Feldman2008.symmetricIndexCanonical_readout_eq`, `Feldman2008.symmetricIndex_promise_symmetric`, `Feldman2008.symmetricIndex_scm_linear_lower_bound_statement`, `Feldman2008.mud_polylog_subset_streaming`, `Feldman2008.theorem4_promise_separation_statement`, `feldman2006_01_mud_build_append`, `feldman2006_02_mud_merge_closed`, `feldman2006_03_mud_build_permutation_invariant`, `feldman2006_04_mud_readout_permutation_invariant`, `feldman2006_05_mud_state_level_mergeable`, `feldman2008_06a_polylog_square`, `feldman2008_10b_representative_merge_exists`, `feldman2008_13b1_representativeMUDFromStreaming_computesOnAllTrees`, `feldman2008_13b_theorem1_deterministic_streaming_to_mud_semantic`, `feldman2008_13c_polylog_streaming_subset_general_mud`, `feldman2008_14b_theorem2_streaming_to_scm_semantic`, `feldman2008_14d_not_polylog_streaming_of_scm_lower_bound`, `feldman2008_15e3_bitAccountedEquality_linear_bigO_lower`, `feldman2008_15h2_bitAccountedFinSetParity_linear_bigO_lower`, `feldman2008_15j_publicRandom_successSet_eq_univ_of_computesSeedwise`, `feldman2008_15j1_publicRandom_successProbability_eq_one_of_computesSeedwise`, `feldman2008_15j2_publicRandom_computesWithSuccessAtLeast_of_computesSeedwise`, `feldman2008_15k_public_randomness_seedwise_general_mud`, `feldman2008_16a1_symmetricIndexCanonical_readout_eq`, `feldman2008_16b_symmetricIndex_promise_symmetric` | Unordered distributed aggregation from local map, identity, associative merge, commutative merge, and final readout; explicit item-level computation-tree semantics; streaming continuation congruence; semantic representative-state streaming-to-general-MUD construction; public-random seedwise streaming-to-general-MUD construction with success-set and exact success-probability bookkeeping; mathlib Big-O polylog-square closure; SCM protocol construction; SCM lower-bound predicates and no-polylog transport; deterministic finite-message equality and finite Set Parity lower bounds; private-coin Set Parity bounded-error reduction; concrete Symmetric Index promise-domain/readout correctness and symmetry; typed randomized Set Parity and Symmetric Index lower-bound obligations. |
| 4 | Flajolet et al. 2007, HyperLogLog | `Flajolet2007.rho`, `Flajolet2007.bitsToNat`, `Flajolet2007.bitsToNat_lt_two_pow_length`, `Flajolet2007.HashObservation`, `Flajolet2007.HashWord`, `Flajolet2007.HashWord.rank_pos`, `Flajolet2007.IdealHashFamily`, `Flajolet2007.RandomIdealHashFamily`, `Flajolet2007.RandomIdealHashFamily.seedFamily_build_append`, `Flajolet2007.RandomIdealHashFamily.seedFamily_hierarchical`, `HLLRegisters.update`, `HLLRegisters.indicatorZ`, `HLLRegisters.rawEstimator`, `HLLRegisters.linearCountingCorrection`, `HLLRegisters.largeRangeCorrection`, `Flajolet2007.poissonWeight`, `Flajolet2007.PoissonizedBySeries`, `Flajolet2007.PoissonizationDepoissonizationAnalysis`, `Flajolet2007.fixedCardinality_asymptotic_of_poissonization_analysis`, `flajolet2007_01_hll_merge_associative`, `flajolet2007_02_hll_merge_commutative`, `flajolet2007_03_hll_merge_idempotent`, `flajolet2007_04_hll_build_append`, `flajolet2007_05_hll_state_level_mergeable`, `flajolet2007_07b_hll_rse_p14_exact`, `flajolet2007_07c_hll_rse_p14_under_one_percent`, `flajolet2007_08b_bitsToNat_lt_two_pow_length`, `flajolet2007_08c_hashWord_rank_positive`, `flajolet2007_09_hll_buildFromHashes_append`, `flajolet2007_10_hll_hash_state_level_mergeable`, `flajolet2007_10b_idealHash_state_level_mergeable`, `flajolet2007_10c_randomIdealHash_seedFamily_build_append`, `flajolet2007_10d_randomIdealHash_seedFamily_hierarchical`, `flajolet2007_13_hll_relativeStandardError_registerCount`, `flajolet2007_14_theorem1_stochasticEstimatorClaims`, `flajolet2007_15b_PoissonizedBySeries`, `flajolet2007_15c_PoissonizationDepoissonizationAnalysis`, `flajolet2007_16_fixedCardinality_asymptotic_of_poisson_depoissonization`, `flajolet2007_16b_fixedCardinality_asymptotic_of_poissonization_analysis`, `flajolet2007_17_relativeStandardErrorBigO_of_asymptotic` | HLL hash prefix/bucket/suffix abstraction; deterministic and random-law ideal-hash-to-register pipeline; `rho`; one-bucket max update semantics; register-vector state; pointwise max merge laws; state-level merge closure; harmonic indicator and raw estimator readout; correction readouts; Poisson-mixture series package and checked fixed-cardinality consequence from a depoissonization-transfer package; RSE formula including the checked `p=14` value `13/1600 < 1%`; a checked Big-O weakening of the RSE asymptotic; and typed analytic theorem schemas. |
| 5 | Agarwal et al. 2012/2013, *Mergeable Summaries* | `Agarwal2013.SummaryMethod`, `Agarwal2013.HasEpsilonSizeRate`, `Agarwal2013.FrequencyEstimationSpec`, `Agarwal2013.HeavyHittersSpec`, `Agarwal2013.QuantileSummarySpec`, `Agarwal2013.EpsilonApproximationSpec`, `Agarwal2013.StateLevelEpsilonApproximationSpec`, `Agarwal2013.stateLevelEpsilonApproximation_tree_error`, `Agarwal2013.SameWeightIntervalApproximationSpec`, `Agarwal2013.intervalCount_append`, `Agarwal2013.sameWeightInterval_tree_error_on_equalLength`, `Agarwal2013.paritySplit`, `Agarwal2013.sameWeightHalving_unbiased_interval_count`, `Agarwal2013.sameWeightHalving_interval_error_abs_le_one`, `Agarwal2013.sameWeightHalving_level_error_abs_le`, `Agarwal2013.sameWeightHalving_hoeffdingDenominator_le`, `Agarwal2013.sameWeightHalving_root_error_to_epsilon_n_of_scale`, `Agarwal2013.SameWeightHalvingCompleteTreeProcess.hoeffding_tail`, `Agarwal2013.SameWeightHalvingCompleteTreeProcess.epsilon_n_tail`, `Agarwal2013.RandomSampleEpsilonApproximationSpec`, `Agarwal2013.EpsilonKernelSpec`, `Agarwal2013.epsilonKernel_hierarchical`, `Agarwal2013.EpsilonKernelSpec.toSizedMergeableQuerySketch`, `Agarwal2013.epsilonKernel_tree_widthError`, `Agarwal2013.hybridTrace_level_monotone`, `CountMin.Table.state_level_mergeable`, `CountMin.Table.merge_not_idempotent_of_pos`, `HeavyHitters.totalCounterMass`, `HeavyHitters.MisraGries.update`, `HeavyHitters.MisraGries.build`, `HeavyHitters.MisraGries.build_boundedBy`, `HeavyHitters.MisraGries.build_totalCounterMass_le_length`, `agarwal2013_01_state_level_hierarchical`, `agarwal2013_02_state_level_hierarchical_readout`, `agarwal2013_03_isomorphic_merge_closed_transport`, `agarwal2013_04_spacesaving_mergeability_corollary_true`, `agarwal2013_05_full_implies_one_way_with_build`, `agarwal2013_07_incrementally_maintainable_one_way`, `agarwal2013_08_linearSketch_fullMergeable`, `agarwal2013_08b_countMin_state_level_mergeable`, `agarwal2013_08c_countMin_merge_not_idempotent_of_pos`, `agarwal2013_10_misraGries_hierarchical`, `agarwal2013_10b_executableMisraGries_boundedBy`, `agarwal2013_10c_executableMisraGries_update_totalCounterMass_le_succ`, `agarwal2013_10d_executableMisraGries_totalCounterMass_le_length`, `agarwal2013_11_spaceSaving_hierarchical_of_isomorphism`, `agarwal2013_12_gk_corollary2_oneWay`, `agarwal2013_13b_SameWeightIntervalApproximationSpec`, `agarwal2013_13e_stateLevelEpsilonApproximation_tree_error`, `agarwal2013_13g_sameWeightInterval_tree_error_on_equalLength`, `agarwal2013_13i_sameWeightHalving_unbiased_interval_count`, `agarwal2013_13k_sameWeightHalving_interval_error_abs_le_one`, `agarwal2013_13m_sameWeightHalving_hoeffdingDenominator_le`, `agarwal2013_13p_sameWeightHalving_completeTree_hoeffding_tail`, `agarwal2013_13q_sameWeightHalving_completeTree_epsilon_n_tail`, `agarwal2013_15_theorem5_randomizedQuantileFullyMergeable`, `agarwal2013_17b_EpsilonKernelSpec`, `agarwal2013_17c_epsilonKernel_hierarchical`, `agarwal2013_17d_epsilonKernel_toSizedMergeableQuerySketch`, `agarwal2013_17e_epsilonKernel_tree_widthError`, `agarwal2013_18_hybridTrace_level_monotone` | Generic state-level mergeable-summary interface; hierarchical readout theorem; mergeability transport across isomorphic summary systems; full/one-way/equal-size relationships; incrementally maintainable summaries imply one-way mergeability; explicit Count-Min additive state merge with non-idempotence; executable MG update/build with capacity and total counter-mass invariants; linear-sketch, MG, SpaceSaving, and GK theorem surfaces; state-level interval/range error transport through merge trees; same-weight interval tree error on equal-length sibling trees; Lemma 3 same-weight halving unbiasedness/boundedness and Lemma 4 complete-tree radius/denominator/scaling arithmetic plus the Azuma/Hoeffding and `εn` tail bounds under explicit martingale hypotheses; epsilon-kernel sized-sketch adapter and tree width-error theorem; typed randomized quantile/range theorem schemas; and a checked hybrid-promotion monotonicity invariant. |

## C-TreePO Bridge Map

| Paper proposition / claim | Lean bridge |
|---|---|
| Strict oracle-output special case | `ops_mergeClosed_of_global`, `ops_hierarchical_mergeable_of_global`, `ops_reduction_to_classical_mergeable` |
| Classical state-level reduction | `sketch_state_mergeClosed_of_compatible`, `sketch_state_level_reduction_to_classical_mergeable` |
| Literature aliases for paper prose | `ctreepo_agarwal2013_state_level_hierarchical_readout`, `ctreepo_agarwal2013_full_implies_one_way_with_build`, `ctreepo_agarwal2013_incrementally_maintainable_one_way`, `ctreepo_agarwal2013_linearSketch_fullMergeable`, `ctreepo_agarwal2013_countMin_state_level_mergeable`, `ctreepo_agarwal2013_countMin_merge_not_idempotent_of_pos`, `ctreepo_agarwal2013_misraGries_hierarchical`, `ctreepo_agarwal2013_spaceSaving_hierarchical_of_isomorphism`, `ctreepo_agarwal2013_gk_corollary2_oneWay`, `ctreepo_agarwal2013_SameWeightIntervalApproximationSpec`, `ctreepo_agarwal2013_EpsilonKernelSpec`, `ctreepo_agarwal2013_epsilonKernel_hierarchical`, `ctreepo_agarwal2013_hybridTrace_level_monotone`, `ctreepo_gibbons1996_ordered_schedule_invariance`, `ctreepo_feldman2008_mud_state_level_mergeable`, `ctreepo_feldman2008_item_tree_state_eq_build`, `ctreepo_feldman2008_streaming_state_congruence_append`, `ctreepo_feldman2008_representative_merge_exists`, `ctreepo_feldman2008_mud_polylog_subset_streaming`, `ctreepo_feldman2008_theorem1_deterministic_streaming_to_mud_semantic`, `ctreepo_feldman2008_theorem2_streaming_to_scm_semantic`, `ctreepo_feldman2008_public_randomness_seedwise_extension_statement`, `ctreepo_feldman2008_theorem3_private_randomness_separation_statement`, `ctreepo_gray1997_state_level_summary_is_algebraic`, `ctreepo_flajolet2007_hll_state_level_mergeable`, `ctreepo_flajolet2007_bitsToNat_lt_two_pow_length`, `ctreepo_flajolet2007_hashWord_rank_positive`, `ctreepo_flajolet2007_hll_buildFromHashes_append`, `ctreepo_flajolet2007_hll_hash_state_level_mergeable`, `ctreepo_flajolet2007_idealHash_state_level_mergeable`, `ctreepo_flajolet2007_hll_rawEstimator_empty`, `ctreepo_flajolet2007_hll_stochasticEstimatorClaims`, `ctreepo_flajolet2007_hll_stochasticEstimatorBigOClaims`, `ctreepo_flajolet2007_fixedCardinality_asymptotic_of_poisson_depoissonization`, `ctreepo_flajolet2007_relativeStandardErrorBigO_of_asymptotic` |
| Standalone literature entry point | `FormalProbability.ML.MergeableSummaries.Literature`, `ML.MergeableSummary.Literature.moduleMap`, `ML.MergeableSummary.Literature.algebraicSketchSurface`, `ML.MergeableSummary.Literature.majorStatementCoverage`, `ML.MergeableSummary.Literature.externalObligationSurface` | Imports every separately formalized sketch-literature module and records the checked algebraic surface, major statement coverage, and typed external obligations. |
| Scalar distinct-count warning | `scalarDistinctCount_not_child_cardinality_mergeable` |
| Gray dynamic/lower-bound warnings | `ctreepo_gray1997_max_no_scalar_delete_front`, `ctreepo_gray1997_contextual_state_lower_bound`, `ctreepo_gray1997_modeBool_state_card_lower_bound`, `ctreepo_gray1997_modeBool_no_finite_state_homomorphic_realization` |
| Optional idempotence warning | `merge_idempotence_orthogonal_to_c2`, `CountMin.Table.merge_not_idempotent_of_pos`, `ctreepo_agarwal2013_countMin_merge_not_idempotent_of_pos`, `hll_idempotent_cms_not_idempotent_when_nonempty` |

Additional bridge aliases added by the HLL-poissonization/Feldman-randomization
pass:

- `ctreepo_feldman2008_representativeMUDFromStreaming_computesOnAllTrees`
- `ctreepo_feldman2008_publicRandom_successSet_eq_univ_of_computesSeedwise`
- `ctreepo_feldman2008_public_randomness_seedwise_general_mud`
- `ctreepo_flajolet2007_PoissonizedBySeries`
- `ctreepo_flajolet2007_PoissonizationDepoissonizationAnalysis`
- `ctreepo_flajolet2007_fixedCardinality_asymptotic_of_poissonization_analysis`

Additional bridge aliases added by the remainder-gap pass:

- `ctreepo_gibbons1996_referenceSection5RuntimeClaims`
- `ctreepo_gray1997_directRollupUpdates_le_directCubeUpdates`
- `ctreepo_gray1997_directRollupSuperAggregateUpdates_le_directCubeSuperAggregateUpdates`
- `ctreepo_feldman2008_publicRandom_successProbability_eq_one_of_computesSeedwise`
- `ctreepo_flajolet2007_randomIdealHash_seedFamily_build_append`
- `ctreepo_flajolet2007_randomIdealHash_seedFamily_hierarchical`
- `ctreepo_agarwal2013_executableMisraGries_update_totalCounterMass_le_succ`
- `ctreepo_agarwal2013_executableMisraGries_totalCounterMass_le_length`
- `ctreepo_agarwal2013_stateLevelEpsilonApproximation_tree_error`
- `ctreepo_agarwal2013_exactStateLevelEpsilonApproximation_tree_error`
- `ctreepo_agarwal2013_exactRangeSpaceSizedMergeableQuerySketch`
- `ctreepo_agarwal2013_sameWeightInterval_tree_error_on_equalLength`
- `ctreepo_agarwal2013_epsilonKernel_toSizedMergeableQuerySketch`
- `ctreepo_agarwal2013_epsilonKernel_tree_widthError`
- `ctreepo_agarwal2013_maxProjectionState_append`
- `ctreepo_agarwal2013_minProjectionState_append`
- `ctreepo_agarwal2013_directionalWidth_append`
- `ctreepo_agarwal2013_exactEpsilonKernel_tree_widthError`

## Scope Boundaries

The Lean layer fully formalizes the algebraic core needed by C-TreePO. It does
not claim to mechanize every theorem in the external papers:

- Gibbons: the algebraic Third Homomorphism Theorem, Lemma 4.3,
  parenthesization consequences, introduction examples/non-example, and
  extensional sorting/merge-sort derivation are mechanized. The reference
  `linear`, `quadratic`, and `n log n` growth predicates are inhabited, and a
  reference Section 5 runtime package supplies length-indexed quadratic,
  linear, and `n log n` cost models. The computable enumerable-domain
  construction for the range representative is represented by the classical
  range-section property used by the proof. A lower-level operation-count
  semantics for Lean's executable sorting code is not yet supplied.
- Gray: the cube/ALL shape, operator algebra, distributive/algebraic taxonomy,
  average example, dynamic-maintenance interface, scalar max delete failure,
  direct ROLLUP-vs-CUBE update-count comparisons, and contextual finite-state
  lower-bound schema for mode and Boolean median/majority are mechanized. SQL
  parsing, catalog semantics, physical query plans, and full
  communication-complexity lower bounds for every SQL median/rank variant remain
  outside this algebraic layer.
- Feldman et al.: the deterministic MUD algebra, item-level computation trees,
  streaming continuation congruence, semantic representative-state
  streaming-to-general-MUD construction, public-random seedwise
  streaming-to-general-MUD construction, exact seedwise success-probability
  wrappers, mathlib Big-O polylog-square closure, SCM protocol construction,
  SCM lower-bound transport, deterministic
  finite-message equality and finite Set Parity lower bounds, private-coin Set
  Parity bounded-error reduction, concrete Symmetric Index promise symmetry,
  and algebraic MUD-to-streaming inclusion are mechanized. The executable
  Savitch-space refinement, randomized equality lower-bound proof, and
  Symmetric Index promise communication lower-bound proof remain outside this
  layer.
- Flajolet et al.: the max-register merge algebra, hash prefix/bucket/suffix/rank abstraction,
  ideal-hash-to-state pipeline, random ideal-hash seedwise algebra, one-bucket update semantics, harmonic
  indicator, raw estimator, correction readouts, RSE expression, checked
  consequences from asymptotic equivalence to Big-O, the Poisson-mixture series
  package, the checked composition from poissonized asymptotics plus a
  depoissonization transfer to fixed-cardinality asymptotics, and typed
  stochastic theorem schemas are formalized.  The Mellin-transform estimates
  and analytic depoissonization proof remain outside this layer.
- Agarwal et al.: the generic mergeability interface, hierarchical readout,
  one-way/equal-size relationships, theorem 2 incremental-maintenance proof,
  linear-sketch and explicit Count-Min additive-table examples, executable MG
  capacity and total counter-mass accounting, MG/SpaceSaving/GK theorem
  surfaces, state-level interval/range tree-error transport with exact
  all-points range witnesses, equal-weight
  interval tree-error transport, same-weight complete-tree halving
  Azuma/Hoeffding concentration under explicit martingale hypotheses,
  epsilon-kernel projection merge laws and tree width-error transport with an
  exact all-points kernel witness, and hybrid promotion bookkeeping are
  formalized.  The remaining executable
  MG/SpaceSaving details and the randomized geometric concentration proofs that
  supply compact valid-state error invariants and size bounds remain typed
  theorem schemas or bundled invariant assumptions.
