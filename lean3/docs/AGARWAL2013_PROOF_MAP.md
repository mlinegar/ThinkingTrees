# Agarwal et al. 2012/2013 Proof Map

Source: Agarwal, Cormode, Huang, Phillips, Wei, and Yi, "Mergeable summaries"
(PODS 2012 / TODS 2013).

Lean modules:

- `FormalProbability/ML/MergeableSummaries.lean`
- `FormalProbability/ML/MergeableSummaries/Agarwal2013.lean`
- `FormalProbability/ML/MergeableSummaries/Agarwal2013Full.lean`
- `FormalProbability/ML/MergeableSummaries/HeavyHitters.lean`
- `FormalProbability/ML/MergeableSummaries/CountMin.lean`
- `FormalProbability/ML/MergeableSummaries/GK.lean`
- `FormalProbability/ML/MergeableSummaries/GKExecutable.lean`
- `FormalProbability/ML/MergeableSummaries/KLL.lean`
- `FormalProbability/ML/MergeableSummaries/KLLExecutable.lean`
- `FormalProbability/ML/MergeableSummaries/Literature.lean`
- `FormalProofs/OPT/PreferenceScope.lean`
- `FormalProofs/OPT/AgarwalNesting.lean`

Important audit correction: the local PDF `docs/mergeable_summaries.pdf` does
not contain an epsilon-kernel theorem.  Existing epsilon-kernel declarations are
kept as adjacent/background geometry unless backed by a separate source.  They
must not be cited as claims of this PDF.

## PDF Claim Inventory

`Agarwal2013Full.paperClaimInventory` records the status labels used here:
`proved`, `proofBearingBundle`, `theoremTarget`, `proseOnly`, and
`adjacentBackground`.

| PDF item | Lean status |
|---|---|
| Summary method, valid summary, full/one-way/restricted/equal-size mergeability, size profile `k(n, ε)`, randomized success, comparison model | Mechanized definitions/specifications in `Agarwal2013`; Section 1.1 proof layer in `Agarwal2013Full` (`ComparisonModel`, `boundedUniverseComparisonModel`, `UnitStorageConvention`, `ValidStateSizeProfile`, `ValidSizedState`, `mergeValidSizedState`, `mergeTree_validSizedState`, `RandomizedTreeSuccess`, `randomizedTreeSuccess_one_of_seedwise_hierarchical`) |
| Streaming as the `|D₂| = 1` special case | Proved via `oneWayMergeable_singleton_step` and the incremental-maintenance route |
| Frequency estimation and heavy-hitter specifications | Typed specs; Count-Min/MG bookkeeping has proved executable witnesses; MG and SpaceSaving are explicitly placed in `SizedMergeableQuerySketch`; MG Lemma 2.1 and the shared merge/prune algebra are proved; algorithm-specific threshold selection remains a proof-bearing certificate |
| Quantiles/ranks | Typed specs; GK is explicitly placed in `OneWaySizedQuerySketch` and `QuantileSummarySpec`; Theorem 3.2 CFF error algebra, logarithmic-layer error summation, hybrid error composition, and executable invariants are proved from bundles; weighted, randomized, and hybrid compact constructions remain proof-bearing theorem targets |
| Weighted/unweighted ε-approximations and range spaces | Typed specs; exact state-level and finite VC/Sauer-Shelah/union-bound layers proved; compact low-discrepancy construction remains conditional on named discrepancy/coloring content |
| VC dimension and finite traces | Proved with mathlib finite VC/Sauer-Shelah and finite-union probability scaffolding |
| MG update rules, Lemma 2.1, `MERGEABLEMINERROR`, `MERGEABLEMINSPACE`, Theorem 2.2 | Executable MG capacity/mass/debt invariants, Lemma 2.1 error envelope, `ε = 1/(k+1)` frequency-error interface, and the shared combine/prune proof for both merge algorithms are proved; concrete threshold-selection routines remain proof-bearing certificates |
| SpaceSaving update rules, Lemma 2.3, Corollary 2.4 | True min-counter replacement update plus capacity/mass bookkeeping proved; MG/SpaceSaving isomorphism and Corollary 2.4 mergeability transport remain represented by an isomorphism/certificate bundle |
| Definition 3.1, Theorem 3.2, Corollary 3.3, Lemmas 3.4-3.5, Theorems 3.6-3.7 | Quantile/rank specs, one-way/bundle routes, Theorem 3.2 CFF triangle inequality, same-weight halving, complete-tree concentration, and finite-layer error summation mechanized; compact weighted and randomized constructions are theorem targets |
| Fact 1, hybrid-summary construction, Lemmas 3.8-3.10, Theorem 3.11, Corollary 3.12 | Hybrid promotion monotonicity, random-buffer union bounds, and sample-plus-summary error composition proved; final hybrid/random-sample constructions represented as proof-bearing bundles |
| Low-discrepancy range merge, Lemmas 4.1-4.3, Theorem 4.4 | Exact range-space, finite VC layers, low-discrepancy merge algebra, Lemma 4.1 unbiased/error bound, and Lemma 4.3 Azuma/Hoeffding scaffold proved; compact coloring construction and Theorem 4.4 size theorem remain isolated as named construction dependencies |
| Section 5.1 experiment setup | Prose-only; simulated 1024-node sensor-network routing tree, bottom-up summary merging, maximum summary size and root actual error recorded |
| Section 5.2 heavy-hitter experiments | Prose-only; compares `MERGEABLEMINERROR` and `MERGEABLEMINSPACE` against `TRIBUTARYANDDELTA` and `MINMAXLOAD` on Zipf data and a contrived tree; empirical plots/tables are not Lean theorems |
| Section 5.3 quantile experiments | Prose-only; compares the Section 3.2 same-weight/logarithmic quantile implementation against GK, q-digest, and SB-p on Gaussian data; empirical trade-offs are not Lean theorems |
| Section 6 concluding and open-problem claims | Prose-only; not encoded as Lean theorems |
| Epsilon-kernel material | Adjacent/background only for this PDF; not part of the Agarwal 2013 PDF claim set |

## Mechanized Core

The union formalization is kept literal throughout the Lean surface:
represented datasets are streams, merges represent `xs ++ ys`, and arbitrary
merge schedules represent `MergeTree.data`.  The approximation parameter
`ε` is also kept explicit through the paper size profile `k(n, ε)` and the
concentration/sample-size side conditions.  Thus a C-TreePO question such as
"fix `ε`; how many documents/leaves/training examples are needed?" belongs to
the size-profile or concentration layer (`ValidStateSizeProfile`,
`mergeTree_validSizedState`, and existing sample-size rules such as
`docsRequired`/`leavesPerDocRequired`), not to a scalar child-query merge law.

Flagged gap: `Agarwal2013Full.section3CompactConstructionGapFlag` records that
the Section 3 compact weighted/hybrid construction is not yet a fully
executable construction.  The reusable CFF, layer-summation, hybrid-error, and
concentration scaffolds are proved; the compact construction remains a theorem
target/proof-bearing bundle.

Flagged gap: `Agarwal2013Full.section4CompactColoringConstructionGapFlag`
records that Section 4's low-discrepancy coloring consequences are proved from
a coloring certificate, while constructing the cited coloring and closing the
compact Theorem 4.4 size theorem remain deferred.

All remaining compact-construction bundles now also have checked placement
theorems into the generic sketch classes:
`Agarwal2013Full.mgMergingAlgorithmCertificate_subset_sizedMergeableQuerySketch`,
`Agarwal2013Full.spaceSavingCertificate_subset_sizedMergeableQuerySketch`,
`Agarwal2013Full.weightedIntervalConstruction_subset_sizedMergeableQuerySketch`,
`Agarwal2013Full.randomizedQuantileConstruction_subset_randomizedSizedMergeableQuerySketch`,
`Agarwal2013Full.randomSampleEpsilonApproximationFact_subset_randomizedSizedMergeableQuerySketch`,
and `Agarwal2013Full.rangeSpaceConstruction_subset_sizedMergeableQuerySketch`.
Thus the deferred content is constructing/certifying those algorithms, not
showing how they nest in the mergeable-sketch interfaces.

Section 5 is empirical evaluation.  `Agarwal2013Full.section5ExperimentsAreProseOnly`
records that its setup, plots, tables, and comparative conclusions are tracked
in this document rather than encoded as Lean theorem targets.

| Paper concept | Lean declaration | Status |
|---|---|---|
| Summary methods can be one-to-many | `ValidSketch`, `Agarwal2013.SummaryMethod` | Mechanized as validity relation |
| Mergeable summary definition | `StateLevelMergeableSummary`, `MergeClosed`, `HierarchicalMergeable` | Mechanized |
| Readout after state merge | `StateLevelMergeableSummary.QueryCorrect`, `Agarwal2013.state_level_hierarchical_readout` | Proved |
| Merged summary size remains within `k(|D₁| + |D₂|, ε)` | `Agarwal2013Full.mergeValidSizedState`, `Agarwal2013Full.mergeTree_validSizedState` | Proved from merge closure plus valid-state size profile |
| Randomized state-level nesting in C-TreePO | `RandomizedTreeReadoutSuccess`, `RandomizedRelationalMergeablePreferenceShape`, `randomizedTreeReadoutSuccess_of_randomizedTreeSuccess`, `randomizedMergeableSummary_relationalShape`, `randomizedMergeableSummary_readout_success_of_mergeTree` | Proved: if the Agarwal randomized root-validity event has probability at least `p`, then C-TreePO root readout correctness has probability at least `p`; this does not assert scalar child-query merging |
| Full mergeability | `FullMergeable` | Mechanized |
| One-way mergeability | `OneWayMergeable` | Mechanized |
| Equal-size/restricted mergeability | `EqualSizeMergeable`, `RestrictedMergeability` | Mechanized |
| Full implies equal-size | `Agarwal2013.full_implies_equal_size`, `agarwal2013_06_full_implies_equal_size` | Proved |
| Full implies one-way via fresh build | `Agarwal2013.full_implies_one_way_with_build`, `agarwal2013_05_full_implies_one_way_with_build` | Proved |
| Incremental insertion fold | `Agarwal2013.foldInsert`, `Agarwal2013.foldInsert_append` | Proved |
| Streaming as singleton/raw-suffix merge | `Agarwal2013Full.oneWayMergeable_singleton_step`, `Agarwal2013Full.oneWayMergeable_streaming_suffix` | Proved |
| Theorem 2, incrementally maintainable implies one-way mergeable | `Agarwal2013.IncrementallyMaintainable.oneWayMergeable`, `agarwal2013_07_incrementally_maintainable_one_way` | Proved |
| Linear sketches are fully mergeable | `Agarwal2013.linearSketch_fullMergeable`, `agarwal2013_08_linearSketch_fullMergeable` | Proved |
| Count-Min additive tables are state-level mergeable | `CountMin.Table.state_level_mergeable`, `Agarwal2013.countMin_state_level_mergeable`, `agarwal2013_08b_countMin_state_level_mergeable` | Proved |
| Count-Min additive merge is not idempotent | `CountMin.Table.merge_not_idempotent_of_pos`, `Agarwal2013.countMin_merge_not_idempotent_of_pos`, `agarwal2013_08c_countMin_merge_not_idempotent_of_pos` | Proved for nonempty tables |
| Frequency-estimation spec | `Agarwal2013.FrequencyEstimationSpec` | Typed spec |
| Heavy-hitter spec | `Agarwal2013.HeavyHittersSpec`, `Agarwal2013.HeavyHittersGuaranteeFn` | Typed spec |
| Quantile/rank spec | `Agarwal2013.QuantileSummarySpec` | Typed spec |
| Theorem 3.2 one-way quantile CFF algebra | `Agarwal2013.CumulativeErrorBound`, `Agarwal2013.pointwiseAddFn`, `Agarwal2013.cumulativeError_oneWay_merge_bound`, `Agarwal2013.cumulativeError_oneWay_twoMerge_half_budget`, `Agarwal2013.cumulativeError_oneWay_merge_half_epsilon`, `agarwal2013_12b_CumulativeErrorBound`, `agarwal2013_12c_cumulativeError_oneWay_merge_bound`, `agarwal2013_12d_cumulativeError_oneWay_twoMerge_half_budget`, `agarwal2013_12e_cumulativeError_oneWay_merge_half_epsilon` | Proved the displayed Section 3.1 triangle inequality `‖F̂' - (F₁ + F₂)‖∞ ≤ ε₁(n₁+n₂)+ε₂n₂` and the half-ε budget specialization |
| ε-approximation spec | `Agarwal2013.EpsilonApproximationSpec` | Typed spec |
| State-level ε-approximation tree error | `Agarwal2013.StateLevelEpsilonApproximationGuaranteeFn`, `Agarwal2013.StateLevelEpsilonApproximationSpec`, `Agarwal2013.stateLevelEpsilonApproximation_hierarchical`, `Agarwal2013.stateLevelEpsilonApproximation_tree_error`, `agarwal2013_13d_StateLevelEpsilonApproximationSpec`, `agarwal2013_13e_stateLevelEpsilonApproximation_tree_error` | Proved: valid state-level range summaries carry their ε-error guarantee through arbitrary merge trees |
| Exact ε-approximation witness | `Agarwal2013.StateLevelEpsilonApproximationSpec.toSizedMergeableQuerySketch`, `Agarwal2013.exactStateLevelEpsilonApproximationSpec`, `Agarwal2013.exactStateLevelEpsilonApproximation_tree_error`, `agarwal2013_13d1_stateLevelEpsilonApproximation_toSizedMergeableQuerySketch`, `agarwal2013_13e1_exactStateLevelEpsilonApproximationSpec`, `agarwal2013_13e2_exactStateLevelEpsilonApproximation_tree_error` | Proved: exact all-points state has zero range-count error for any nonnegative ε and survives arbitrary merge trees; this is a linear-size witness, not the compact Agarwal construction |
| Finite range-space VC/Sauer-Shelah layer | `Agarwal2013.FiniteRangeSpace`, `Agarwal2013.FiniteRangeSpace.trace`, `Agarwal2013.FiniteRangeSpace.vcDim`, `Agarwal2013.FiniteRangeSpace.shattered_card_le_vcDim`, `Agarwal2013.FiniteRangeSpace.trace_card_le_sauerShelah`, `Agarwal2013.FiniteRangeSpace.traceFailureEvent`, `Agarwal2013.FiniteRangeSpace.measureReal_traceFailureEvent_le_card_mul`, `Agarwal2013.FiniteRangeSpace.measureReal_traceFailureEvent_le_sauerShelah_mul`, `agarwal2013_16a_FiniteRangeSpace`, `agarwal2013_16a1_finiteRangeSpace_shattered_card_le_vcDim`, `agarwal2013_16a2_finiteRangeSpace_trace_card_le_sauerShelah`, `agarwal2013_16a4_finiteRangeSpace_traceFailureEvent_le_card_mul`, `agarwal2013_16a5_finiteRangeSpace_traceFailureEvent_le_sauerShelah_mul` | Proved using mathlib's `Finset.Shatters`, `Finset.vcDim`, Sauer-Shelah, and finite-union measure bounds: finite range traces have VC-dimension bounds, Sauer-Shelah growth control, and a uniform ε-approximation failure scaffold from per-trace tails |
| Section 4 low-discrepancy range-space merge core | `Agarwal2013.FiniteRangeSpace.rangeCountOn`, `Agarwal2013.FiniteRangeSpace.colorDiscrepancy`, `Agarwal2013.FiniteRangeSpace.LowDiscrepancyColoring`, `Agarwal2013.FiniteRangeSpace.coloredRangeEstimate_unbiased`, `Agarwal2013.FiniteRangeSpace.lowDiscrepancy_coloredRangeError_abs_le`, `Agarwal2013.FiniteRangeSpace.LowDiscrepancyMergeCertificate`, `Agarwal2013.rangeSpaceColoringLevelError`, `Agarwal2013.rangeSpaceColoring_level_error_abs_le`, `Agarwal2013.RangeSpaceColoringCompleteTreeProcess`, `Agarwal2013.RangeSpaceColoringCompleteTreeProcess.hoeffding_tail`, `agarwal2013_16a7_finiteRangeSpace_coloredRangeEstimate_unbiased`, `agarwal2013_16a8_finiteRangeSpace_lowDiscrepancy_coloredRangeError_abs_le`, `agarwal2013_16a10_rangeSpaceColoring_level_error_abs_le`, `agarwal2013_16a12_rangeSpaceColoring_completeTree_hoeffding_tail` | Proved conditionally on a supplied low-discrepancy coloring certificate: random retained color class is unbiased, either choice has error bounded by the discrepancy budget, level scaling gives radius `2^level * Δ`, and complete-tree errors satisfy an Azuma/Hoeffding tail under explicit martingale hypotheses |
| Exact range-space ε-approximation witness | `Agarwal2013.geometricRangeCount`, `Agarwal2013.exactRangeSpaceEpsilonApproximationSpec`, `Agarwal2013.exactRangeSpaceSizedMergeableQuerySketch`, `agarwal2013_16b_geometricRangeCount`, `agarwal2013_16c_exactRangeSpaceEpsilonApproximationSpec`, `agarwal2013_16d_exactRangeSpaceSizedMergeableQuerySketch` | Proved exact state-level range-space error and state merge/readout interface; compact `O(1/ε)` range-space construction remains Theorem 6's randomized/discrepancy content |
| Same-weight 1D interval approximation spec | `Agarwal2013.Interval1D`, `Agarwal2013.intervalCount`, `Agarwal2013.intervalCount_append`, `Agarwal2013.intervalCount_le_length`, `Agarwal2013.EqualLengthSiblingTree`, `Agarwal2013.SameWeightIntervalApproximationSpec`, `Agarwal2013.sameWeightInterval_valid_on_equalLengthTree`, `Agarwal2013.sameWeightInterval_tree_error_on_equalLength`, `agarwal2013_13b_SameWeightIntervalApproximationSpec`, `agarwal2013_13c_intervalCount_append`, `agarwal2013_13f_sameWeightInterval_valid_on_equalLengthTree`, `agarwal2013_13g_sameWeightInterval_tree_error_on_equalLength` | Proved interval-count additivity and same-weight tree-error transport for equal-length-sibling merge trees |
| Lemma 3, same-weight halving for intervals | `Agarwal2013.paritySplit`, `Agarwal2013.paritySplit_length_sum`, `Agarwal2013.sameWeightHalving_unbiased_interval_count`, `Agarwal2013.sameWeightHalving_interval_error_mean_zero`, `Agarwal2013.sameWeightHalving_interval_error_abs_le_one`, `agarwal2013_13h_paritySplit_length_sum`, `agarwal2013_13i_sameWeightHalving_unbiased_interval_count`, `agarwal2013_13j_sameWeightHalving_interval_error_mean_zero`, `agarwal2013_13k_sameWeightHalving_interval_error_abs_le_one` | Proved deterministic two-choice halving core: the even/odd scaled interval estimates average to the exact interval count and either choice has absolute over-count at most one |
| Lemma 4, complete-tree halving concentration | `Agarwal2013.sameWeightHalvingLevelError`, `Agarwal2013.sameWeightHalving_level_error_abs_le`, `Agarwal2013.sameWeightHalvingHoeffdingDenominator`, `Agarwal2013.sameWeightHalving_hoeffdingDenominator_le`, `Agarwal2013.sameWeightHalvingRepresentedLength`, `Agarwal2013.sameWeightHalving_root_error_to_epsilon_n_of_scale`, `Agarwal2013.SameWeightHalvingCompleteTreeProcess`, `Agarwal2013.SameWeightHalvingCompleteTreeProcess.hoeffding_tail`, `Agarwal2013.SameWeightHalvingCompleteTreeProcess.epsilon_n_tail`, `agarwal2013_13l_sameWeightHalving_level_error_abs_le`, `agarwal2013_13m_sameWeightHalving_hoeffdingDenominator_le`, `agarwal2013_13o_sameWeightHalving_root_error_to_epsilon_n_of_scale`, `agarwal2013_13p_sameWeightHalving_completeTree_hoeffding_tail`, `agarwal2013_13q_sameWeightHalving_completeTree_epsilon_n_tail` | Proved the level-radius scaling, the closed Hoeffding denominator bound `≤ 2^(2m+1)`, the final root-scale-to-`εn` conversion, and the two-sided Azuma/Hoeffding tail under explicit martingale measurability/adaptedness/bounded-increment/conditional-mean-zero hypotheses |
| Theorem 3.7 finite logarithmic-layer error algebra | `Agarwal2013.finiteLayer_interval_error_sum_bound`, `agarwal2013_13r_finiteLayer_interval_error_sum_bound` | Proved that finite layer errors with bounds `ε * mass_i` sum to `ε` times total represented mass |
| Random-sample ε-approximation spec | `Agarwal2013.RandomSampleEpsilonApproximationSpec`, `Agarwal2013.FiniteRangeSpace.measureReal_traceFailureEvent_le_sauerShelah_mul` | Typed randomized spec plus proved finite-trace/VC union-bound layer for upgrading pointwise trace tails to uniform failure bounds |
| ε-kernel width geometry and tree width error | `Agarwal2013.pointDot`, `Agarwal2013.translatePoint`, `Agarwal2013.translateStream`, `Agarwal2013.pointDot_translatePoint`, `Agarwal2013.scalePoint`, `Agarwal2013.scaleStream`, `Agarwal2013.pointDot_scalePoint`, `Agarwal2013.maxProjection`, `Agarwal2013.minProjection`, `Agarwal2013.directionalWidth`, `Agarwal2013.maxProjectionState_append`, `Agarwal2013.minProjectionState_append`, `Agarwal2013.directionalWidth_append`, `Agarwal2013.directionalWidth_translateStream`, `Agarwal2013.directionalWidth_scaleStream_of_nonneg`, `Agarwal2013.StateLevelWidthApproximationGuaranteeFnD`, `Agarwal2013.EpsilonKernelSpec`, `Agarwal2013.epsilonKernel_hierarchical`, `Agarwal2013.EpsilonKernelSpec.toSizedMergeableQuerySketch`, `Agarwal2013.epsilonKernel_tree_widthError`, `Agarwal2013.exactEpsilonKernelSpec`, `Agarwal2013.exactEpsilonKernel_tree_widthError`, `agarwal2013_17a_pointDot`, `agarwal2013_17a3_pointDot_translatePoint`, `agarwal2013_17b_EpsilonKernelSpec`, `agarwal2013_17c_epsilonKernel_hierarchical`, `agarwal2013_17d_epsilonKernel_toSizedMergeableQuerySketch`, `agarwal2013_17e_epsilonKernel_tree_widthError`, `agarwal2013_17f_maxProjectionState_append`, `agarwal2013_17g_minProjectionState_append`, `agarwal2013_17h_directionalWidth_append`, `agarwal2013_17h1_directionalWidth_translateStream`, `agarwal2013_17h2_directionalWidth_scaleStream_of_nonneg`, `agarwal2013_17i_exactEpsilonKernelSpec`, `agarwal2013_17j_exactEpsilonKernel_tree_widthError`, `Agarwal2013Full.epsilonKernel_audit_correction` | Proved adjacent/background geometry, not a claim of `docs/mergeable_summaries.pdf`; cite only with a separate epsilon-kernel source |
| Hybrid promotion, random buffers, and error composition | `Agarwal2013.HybridPromotion`, `Agarwal2013.HybridTraceOnlyMovesUp`, `Agarwal2013.hybridTrace_level_monotone`, `Agarwal2013.hybridRandomBufferFailureEvent`, `Agarwal2013.hybridRandomBuffer_failure_bound`, `Agarwal2013.hybridRandomBuffer_failure_bound_uniform`, `Agarwal2013.epsilonApproximation_error_add`, `agarwal2013_18_hybridTrace_level_monotone`, `agarwal2013_18a_hybridRandomBuffer_failure_bound`, `agarwal2013_18b_hybridRandomBuffer_failure_bound_uniform`, `agarwal2013_18c_epsilonApproximation_error_add` | Proved deterministic promotion monotonicity, finite-level random-buffer union-bound theorem, and the hybrid algebra that an `εs` sample approximation plus an `εh` summary approximation yields `εs + εh` total error |
| Section 5 experiments | `Agarwal2013Full.section5ExperimentsAreProseOnly` | Prose-only: empirical setup, figures, tables, and qualitative comparisons are documented but not formal theorem targets |
| Executable MG update/build | `HeavyHitters.MisraGries.update`, `HeavyHitters.MisraGries.build` | Mechanized |
| Executable MG capacity invariant | `HeavyHitters.MisraGries.build_boundedBy`, `Agarwal2013.executableMisraGries_boundedBy`, `agarwal2013_10b_executableMisraGries_boundedBy` | Proved |
| Executable MG total counter-mass and debt invariant | `HeavyHitters.totalCounterMass`, `HeavyHitters.MisraGries.update_totalCounterMass_le_succ`, `HeavyHitters.MisraGries.build_totalCounterMass_le_length`, `HeavyHitters.MisraGries.build_positiveCounts`, `HeavyHitters.MisraGries.tracedBuild_potential_le_length`, `HeavyHitters.MisraGries.tracedBuild_debt_mul_succ_le_length`, `Agarwal2013.executableMisraGries_update_totalCounterMass_le_succ`, `Agarwal2013.executableMisraGries_totalCounterMass_le_length`, `Agarwal2013.executableMisraGries_positiveCounts`, `Agarwal2013.executableMisraGries_tracedPotential_le_length`, `Agarwal2013.executableMisraGries_debt_mul_succ_le_length`, `agarwal2013_10c_executableMisraGries_update_totalCounterMass_le_succ`, `agarwal2013_10d_executableMisraGries_totalCounterMass_le_length`, `agarwal2013_10e_executableMisraGries_positiveCounts`, `agarwal2013_10f_executableMisraGries_tracedPotential_le_length`, `agarwal2013_10g_executableMisraGries_debt_mul_succ_le_length` | Proved executable mass accounting and the standard traced-potential induction: one update increases stored mass by at most one, positive counters are preserved, build mass is bounded by stream length, and global decrement steps are charged to blocks of `k+1` processed items |
| MG Lemma 2.1 executable envelope | `HeavyHitters.MisraGries.build_estimate_le_frequency`, `HeavyHitters.MisraGries.frequency_le_build_estimate_add_debt`, `HeavyHitters.MisraGries.lemma21_frequency_envelope`, `HeavyHitters.MisraGries.lemma21_real_error_bound`, `HeavyHitters.MisraGries.lemma21_frequencyErrorGuarantee_inv_succ`, `Agarwal2013.executableMisraGries_lemma21_frequency_envelope`, `Agarwal2013.executableMisraGries_lemma21_real_error_bound`, `Agarwal2013.executableMisraGries_frequencyError_inv_succ` | Proved: executable MG estimates are lower bounds, the undercount is bounded by global decrement debt, debt is charged to `k+1`-item deletion blocks, and the repository's real-valued additive frequency-error interface holds for `ε = 1/(k+1)` |
| MG merge/prune algebra for Theorem 2.2 | `HeavyHitters.MisraGries.Lemma21Envelope`, `HeavyHitters.MisraGries.PruneCertificate`, `HeavyHitters.MisraGries.lemma21Envelope_combine`, `HeavyHitters.MisraGries.lemma21Envelope_prune`, `HeavyHitters.MisraGries.lemma21Envelope_mergeOfPruneCertificate`, `Agarwal2013.misraGries_lemma21Envelope_combine`, `Agarwal2013.misraGries_lemma21Envelope_prune`, `Agarwal2013.misraGries_lemma21Envelope_mergeOfPruneCertificate` | Proved: combining summaries adds estimates/masses without adding error, and any prune whose mass drop pays `(k+1)C` preserves the Lemma 2.1 envelope. This is the common algebra behind `MERGEABLEMINERROR` and `MERGEABLEMINSPACE`; choosing/proving the threshold condition for a concrete implementation is still a certificate field |
| MG theorem bundle and generic-sketch placement | `HeavyHitters.MGAlgorithm`, `HeavyHitters.MGAlgorithm.toSizedMergeableQuerySketch`, `HeavyHitters.MGAlgorithm.toStateLevelMergeableSummary`, `Agarwal2013.misraGries_toSizedMergeableQuerySketch`, `Agarwal2013.misraGries_toStateLevelMergeableSummary`, `Agarwal2013.misraGries_toFrequencyEstimationSpec`, `Agarwal2013.misraGries_theorem1_of_algorithm`, `Agarwal2013.misraGries_hierarchical`, `Agarwal2013.misraGries_subset_sizedMergeableQuerySketch`, `agarwal2013_10a_misraGries_subset_sizedMergeableQuerySketch` | Proved from algorithm bundle; MG is now explicitly a `SizedMergeableQuerySketch` and a state-level summary, with frequency estimation as the query readout |
| SpaceSaving Corollary 1, generic-sketch placement, and executable bookkeeping | `HeavyHitters.SpaceSavingAlgorithm.toSizedMergeableQuerySketch`, `HeavyHitters.SpaceSavingAlgorithm.toStateLevelMergeableSummary`, `Agarwal2013.spaceSaving_toSizedMergeableQuerySketch`, `Agarwal2013.spaceSaving_toStateLevelMergeableSummary`, `Agarwal2013.spaceSaving_toFrequencyEstimationSpec`, `Agarwal2013.spaceSaving_subset_sizedMergeableQuerySketch`, `HeavyHitters.IsomorphicMGSpaceSaving`, `Agarwal2013.spaceSaving_hierarchical_of_isomorphism`, `HeavyHitters.SpaceSaving.extractMinCounter`, `HeavyHitters.SpaceSaving.extractMinCounter_length`, `HeavyHitters.SpaceSaving.extractMinCounter_mass`, `HeavyHitters.SpaceSaving.update`, `HeavyHitters.SpaceSaving.build_boundedBy`, `HeavyHitters.SpaceSaving.build_totalCounterMass_le_length`, `Agarwal2013.executableSpaceSaving_boundedBy`, `Agarwal2013.executableSpaceSaving_update_totalCounterMass_le_succ`, `Agarwal2013.executableSpaceSaving_totalCounterMass_le_length`, `agarwal2013_11aa_spaceSaving_subset_sizedMergeableQuerySketch`, `agarwal2013_11_spaceSaving_hierarchical_of_isomorphism`, `agarwal2013_11a_executableSpaceSaving_boundedBy`, `agarwal2013_11b_executableSpaceSaving_update_totalCounterMass_le_succ`, `agarwal2013_11c_executableSpaceSaving_totalCounterMass_le_length` | SpaceSaving is explicitly a `SizedMergeableQuerySketch` and a state-level summary once packaged with its validity witnesses. True minimum-counter extraction/update is executable and proved to remove exactly one entry while preserving separated counter mass; capacity/mass bookkeeping is proved for that update. Full SpaceSaving mergeability still follows through the MG/SpaceSaving isomorphism certificate |
| GK Corollary 2 and one-way generic-sketch placement | `GK.Algorithm`, `GK.Algorithm.toOneWaySizedQuerySketch`, `GK.Algorithm.subset_oneWaySizedQuerySketch`, `Agarwal2013.gk_corollary2_oneWay`, `Agarwal2013.gk_toOneWaySizedQuerySketch`, `Agarwal2013.gk_toQuantileSummarySpec`, `Agarwal2013.gk_subset_oneWaySizedQuerySketch`, `agarwal2013_12a_gk_subset_oneWaySizedQuerySketch` | Proved from GK bundle; GK is explicitly one-way mergeable and is not claimed as a full state-state mergeable sketch |
| Executable GK item count and gap mass | `GK.Executable.build_n`, `GK.Executable.build_gapMassValid`, `gk2001_01_executable_build_n`, `gk2001_02_executable_build_gapMassValid` | Proved executable invariants |
| KLL mergeable/optimal theorem bundles and randomized-sketch placement | `KLL.Algorithm.toRandomizedSizedMergeableQuerySketch`, `KLL.Algorithm.toSizedMergeableQuerySketchAt`, `KLL.Algorithm.subset_randomizedSizedMergeableQuerySketch`, `KLL.theorem4_of_algorithm`, `KLL.theorem5_of_algorithm`, `kll2016_01_theorem4_mergeable_variant_of_algorithm`, `kll2016_01a_subset_randomizedSizedMergeableQuerySketch`, `kll2016_02_theorem5_optimal_variant_of_algorithm` | Proved from algorithm bundle; the mergeable KLL-style quantile surface is explicitly a randomized sized mergeable query sketch |
| Executable KLL weighted mass | `KLL.Executable.weightedCount_step`, `KLL.Executable.build_massValid`, `kll2016_03_executable_weightedCount_step`, `kll2016_04_executable_build_massValid` | Proved executable invariants |

## Theorem Schemas

The geometry and probability-heavy results are represented as precise theorem
schemas rather than axioms:

| Paper theorem | Lean schema |
|---|---|
| Theorem 3, same-weight 1D interval ε-approximations | `Agarwal2013.theorem3_sameWeightIntervalEpsilonApproximation` |
| Theorem 4, weighted 1D interval ε-approximations | `Agarwal2013.theorem4_weightedIntervalEpsilonApproximation` |
| Theorem 5, randomized fully mergeable quantiles | `Agarwal2013.theorem5_randomizedQuantileFullyMergeable` |
| Theorem 6, range-space ε-approximations | `Agarwal2013.theorem6_rangeSpaceEpsilonApproximation` |
| Epsilon-kernel common-reference-frame theorem | Adjacent/background only for this PDF; declaration remains available as geometry support but is not an Agarwal PDF theorem |

## Section 5 Experiments

Section 5 is intentionally not formalized as theorem statements.

| PDF subsection | Claim classification |
|---|---|
| 5.1 Experiment setup | Prose-only: simulated 1024-node sensor network on the unit square, BFS routing tree, bottom-up merge to the root, measuring maximum summary size and root actual error |
| 5.2 Heavy hitters | Prose-only: empirical comparison of `MERGEABLEMINERROR` and `MERGEABLEMINSPACE` against `TRIBUTARYANDDELTA` and `MINMAXLOAD`, including Zipf data plots and a contrived-tree table |
| 5.3 Quantiles | Prose-only: empirical comparison of the simpler Section 3.2 randomized/logarithmic quantile implementation against GK, q-digest, and SB-p, using Gaussian data and actual-error/summary-size trade-off curves |

These are citation and discussion material for C-TreePO.  They can support
paper prose, but they should not be cited as machine-checked Lean theorem
backing.

## C-TreePO Bridge Names

`FormalProofs/OPT/MergeableReduction.lean` re-exports the Agarwal surface under
C-TreePO-facing names:

- `ctreepo_agarwal2013_state_level_hierarchical_readout`
- `ctreepo_agarwal2013_full_implies_one_way_with_build`
- `ctreepo_agarwal2013_incrementally_maintainable_one_way`
- `ctreepo_agarwal2013_linearSketch_fullMergeable`
- `ctreepo_agarwal2013_countMin_state_level_mergeable`
- `ctreepo_agarwal2013_countMin_merge_not_idempotent_of_pos`
- `ctreepo_agarwal2013_misraGries_hierarchical`
- `ctreepo_agarwal2013_misraGries_subset_sizedMergeableQuerySketch`
- `ctreepo_agarwal2013_executableMisraGries_update_totalCounterMass_le_succ`
- `ctreepo_agarwal2013_executableMisraGries_totalCounterMass_le_length`
- `ctreepo_agarwal2013_executableMisraGries_positiveCounts`
- `ctreepo_agarwal2013_executableMisraGries_tracedPotential_le_length`
- `ctreepo_agarwal2013_executableMisraGries_debt_mul_succ_le_length`
- `ctreepo_agarwal2013_executableMisraGries_lemma21_frequency_envelope`
- `ctreepo_agarwal2013_executableMisraGries_lemma21_real_error_bound`
- `ctreepo_agarwal2013_executableMisraGries_frequencyError_inv_succ`
- `ctreepo_agarwal2013_misraGries_lemma21Envelope_mergeOfPruneCertificate`
- `ctreepo_agarwal2013_spaceSaving_hierarchical_of_isomorphism`
- `ctreepo_agarwal2013_spaceSaving_subset_sizedMergeableQuerySketch`
- `ctreepo_agarwal2013_executableSpaceSaving_boundedBy`
- `ctreepo_agarwal2013_executableSpaceSaving_update_totalCounterMass_le_succ`
- `ctreepo_agarwal2013_executableSpaceSaving_totalCounterMass_le_length`
- `ctreepo_agarwal2013_gk_corollary2_oneWay`
- `ctreepo_agarwal2013_gk_subset_oneWaySizedQuerySketch`
- `ctreepo_agarwal2013_cumulativeError_oneWay_merge_bound`
- `ctreepo_agarwal2013_cumulativeError_oneWay_twoMerge_half_budget`
- `ctreepo_agarwal2013_cumulativeError_oneWay_merge_half_epsilon`
- `ctreepo_agarwal2013_SameWeightIntervalApproximationSpec`
- `ctreepo_agarwal2013_StateLevelEpsilonApproximationSpec`
- `ctreepo_agarwal2013_stateLevelEpsilonApproximation_toSizedMergeableQuerySketch`
- `ctreepo_agarwal2013_stateLevelEpsilonApproximation_tree_error`
- `ctreepo_agarwal2013_exactStateLevelEpsilonApproximationSpec`
- `ctreepo_agarwal2013_exactStateLevelEpsilonApproximation_tree_error`
- `ctreepo_agarwal2013_geometricRangeCount`
- `ctreepo_agarwal2013_finiteRangeSpace_traceFailureEvent_le_card_mul`
- `ctreepo_agarwal2013_finiteRangeSpace_traceFailureEvent_le_sauerShelah_mul`
- `ctreepo_agarwal2013_finiteRangeSpace_rangeCountOn`
- `ctreepo_agarwal2013_finiteRangeSpace_coloredRangeEstimate_unbiased`
- `ctreepo_agarwal2013_finiteRangeSpace_lowDiscrepancy_coloredRangeError_abs_le`
- `ctreepo_agarwal2013_LowDiscrepancyMergeCertificate`
- `ctreepo_agarwal2013_rangeSpaceColoring_level_error_abs_le`
- `ctreepo_agarwal2013_RangeSpaceColoringCompleteTreeProcess`
- `ctreepo_agarwal2013_rangeSpaceColoring_completeTree_hoeffding_tail`
- `ctreepo_agarwal2013_exactRangeSpaceEpsilonApproximationSpec`
- `ctreepo_agarwal2013_exactRangeSpaceSizedMergeableQuerySketch`
- `ctreepo_agarwal2013_sameWeightInterval_tree_error_on_equalLength`
- `ctreepo_agarwal2013_sameWeightHalving_unbiased_interval_count`
- `ctreepo_agarwal2013_sameWeightHalving_interval_error_mean_zero`
- `ctreepo_agarwal2013_sameWeightHalving_interval_error_abs_le_one`
- `ctreepo_agarwal2013_sameWeightHalving_level_error_abs_le`
- `ctreepo_agarwal2013_sameWeightHalving_hoeffdingDenominator_le`
- `ctreepo_agarwal2013_sameWeightHalving_root_error_to_epsilon_n_of_scale`
- `ctreepo_agarwal2013_sameWeightHalving_completeTree_hoeffding_tail`
- `ctreepo_agarwal2013_sameWeightHalving_completeTree_epsilon_n_tail`
- `ctreepo_agarwal2013_finiteLayer_interval_error_sum_bound`
- `ctreepo_agarwal2013_pointDot`
- `ctreepo_agarwal2013_EpsilonKernelSpec`
- `ctreepo_agarwal2013_epsilonKernel_hierarchical`
- `ctreepo_agarwal2013_epsilonKernel_toSizedMergeableQuerySketch`
- `ctreepo_agarwal2013_epsilonKernel_tree_widthError`
- `ctreepo_agarwal2013_maxProjectionState_append`
- `ctreepo_agarwal2013_minProjectionState_append`
- `ctreepo_agarwal2013_directionalWidth_append`
- `ctreepo_agarwal2013_exactEpsilonKernelSpec`
- `ctreepo_agarwal2013_exactEpsilonKernel_tree_widthError`
- `ctreepo_agarwal2013_hybridTrace_level_monotone`
- `ctreepo_agarwal2013_hybridRandomBuffer_failure_bound`
- `ctreepo_agarwal2013_hybridRandomBuffer_failure_bound_uniform`
- `ctreepo_agarwal2013_epsilonApproximation_error_add`
- `ctreepo_gk2001_executable_build_n`
- `ctreepo_gk2001_executable_build_gapMassValid`
- `ctreepo_kll2016_theorem4_mergeable_variant_of_algorithm`
- `ctreepo_kll2016_subset_randomizedSizedMergeableQuerySketch`
- `ctreepo_kll2016_theorem5_optimal_variant_of_algorithm`
- `ctreepo_kll2016_executable_weightedCount_step`
- `ctreepo_kll2016_executable_build_massValid`
- `ctreepo_agarwal2013_randomizedQuantileFullyMergeable`
- `ctreepo_agarwal2013_rangeSpaceEpsilonApproximation`

`FormalProofs/OPT/PreferenceScope.lean` exposes the generic relational C-TreePO
nesting vocabulary used for the paper comparison:

- `RelationalMergeablePreferenceShape`
- `RelationalMergeablePreferenceShape.hierarchical`
- `RelationalMergeablePreferenceShape.readout_of_mergeTree`
- `RelationalMergeablePreferenceShape.to_mergeablePreferenceShape_of_canonical`
- `RandomizedTreeReadoutSuccess`
- `randomizedTreeReadoutSuccess_of_randomizedTreeSuccess`
- `RandomizedRelationalMergeablePreferenceShape`
- `RandomizedRelationalMergeablePreferenceShape.readout_success_of_mergeTree`
- `ScalarQueryMergeLaw`

`FormalProofs/OPT/AgarwalNesting.lean` instantiates that vocabulary for the
Agarwal state-level summary interface:

- `stateLevelMergeableSummary_relationalShape`
- `stateLevelMergeableSummary_readout_of_mergeTree`
- `stateLevelMergeableSummary_to_mergeablePreferenceShape_of_canonical`
- `randomizedMergeableSummary_relationalShape`
- `randomizedMergeableSummary_readout_success_of_mergeTree`

`FormalProofs/OPT/MainTheorems.lean` re-exports the C-TreePO-facing aliases
`randomized_tree_readout_success`,
`randomized_tree_readout_success_of_randomized_tree_success`,
`randomized_relational_mergeable_preference_shape`,
`randomized_relational_mergeable_preference_readout_success_of_tree`,
`agarwal_randomized_relational_shape`, and
`agarwal_randomized_readout_success_of_merge_tree`.

Classical full mergeable summaries nest in C-TreePO through relational
state-level C-Trees. Randomized mergeable summaries nest in the same
formulation in probability: root readout is correct on the event that the
merged root state is valid.

## C-TreePO Nesting Conditions

The nesting theorem is deliberately state-level.  To instantiate it in prose or
in Lean, identify the following objects.

| Role | Required object | Lean shape |
|---|---|---|
| Represented union data | Streams/datasets combine as `xs ++ ys`; a tree represents `MergeTree.data t` | `Stream α`, `MergeTree.data` |
| State builder | Leaf summarizer/state encoder for raw data | `build : Stream α → State` or randomized `build : Ω → Stream α → State` |
| State merge | Binary merge of child states, representing union/concatenation | `merge : State → State → State` |
| Validity relation | State `s` is a valid summary of represented stream `xs` | `valid : Stream α → State → Prop` |
| Merge closure | Valid child states merge to a valid parent for `xs ++ ys` | `MergeClosed valid merge` or probabilistic `RandomizedTreeSuccess` |
| Root readout | C-TreePO readout from the final state | `readout : State → Pref` |
| Target preference/oracle | The value C-TreePO wants to preserve | `pref : Stream α → Pref` |
| Readout correctness | Every valid state reads out the target value | `∀ xs s, valid xs s → readout s = pref xs` |

The explicit Lean adapter for the deterministic conversion is
`CTreePOToAgarwalTransform`.  It additionally records the fixed paper error
parameter `ε`, the state size function, and the size profile `k(n, ε)`:

- `toStateLevelMergeableSummary`: forgets size metadata and produces
  Agarwal's state-level summary interface.
- `toRelationalShape`: produces the C-TreePO relational nesting shape.
- `buildValidSizedState`: leaf builder gives a valid sized `S(D, ε)` state.
- `mergeValidSizedState`: valid child states merge to a valid sized
  `S(D₁ ++ D₂, ε)` state.
- `mergeTree_validSizedState`: arbitrary merge trees produce valid sized root
  states for `MergeTree.data t`.
- `readout_of_mergeTree`: read out after state merge to recover the target.
- `mergeTree_size_bound`: the root state satisfies the `k(|D|, ε)` profile.

For a deterministic classical sketch, `g` must serialize these state operations:
on raw leaves, `g(x)` is the encoded/build state; on two child summaries,
`g(s_L ++ s_R)` is the serialization of `merge s_L s_R`.  The readout `f` is
the query/readout applied to the root state.  Then C1 is the leaf build/readout
case, and C3 is the merge-closure/readout case for the raw union.  C2 is the
extra C-TreePO condition needed when an already produced state can be passed
through `g` again; canonical classical states generally satisfy it trivially.

For a randomized mergeable summary, a single seed `ω : Ω` represents all random
choices used in the tree.  Agarwal's hypothesis supplies
`RandomizedTreeSuccess μ build valid merge t p`, the probability that the root
state is valid for the union represented by `t`.  The C-TreePO theorem
`randomizedTreeReadoutSuccess_of_randomizedTreeSuccess` proves that the root
readout is correct with the same probability `p`, because the validity event is
contained in the readout-correctness event.

Fixing `ε` does not change the nesting shape.  It specializes the validity
relation and size/error profile: valid states must satisfy the paper profile
`k(|D|, ε)` and the readout error bound at that same `ε`.  Questions such as
"how much training data or audit data is needed?" live in the concentration or
learning layer that certifies C1/C3/C2 or randomized success at probability
`p`; they are not part of Agarwal mergeability itself and do not create a law
for merging scalar child answers.

## Neural-Operator Calibrated Route

The neural-operator layer is an ambient realization class for the state-level
interface above.  It does not make an arbitrary checkpoint mergeable.  The exact
class covered by the theorem is

```text
NeuralOperatorClass C ∩ MergeableSketchSummaryClass(fhat)
```

or an approximate neighborhood of that intersection supplied by
`NeuralOperatorTheoremBridgeAssumptions` /
`ApproxNeuralOperatorPreferenceBridge`.  In this route:

- `g` builds leaf states and merges child states;
- `fhat` is the learned/calibrated root readout;
- `fstar` is the true oracle;
- C1/C3/C2 are projection losses for `g` relative to `fhat`;
- optional state-imitation losses are valid only in simulations or sketches
  where exact target states are known.

The Lean bridge is
`trueOracle_delta_R_ZR_le_of_calibrated_neuralOperatorBridge`: if the ideal
operator is exact theorem-backed for `fhat`, the realized neural operator
approximates it with a local-law budget, and
`UniformOracleApproximation fstar fhat εf` holds, then the true-oracle
distortion for `fstar` is bounded by the local-law budget plus `2 * εf`.
The companion aliases in `MainTheorems.lean` are:

- `neural_operator_true_oracle_delta_r_bound_calibrated`
- `neural_operator_true_oracle_delta_r_transfer_moduli_bound_calibrated`
- `neural_operator_true_oracle_utility_bound_calibrated`
- `neural_operator_fd_true_oracle_delta_r_bound_calibrated`
- `neural_operator_fd_true_oracle_delta_r_transfer_moduli_bound_calibrated`
- `neural_operator_fd_true_oracle_utility_bound_calibrated`

For randomized Agarwal summaries, the probability statement remains
state-level: root validity with probability `p` gives root readout correctness
with probability `p` for `fhat`; the calibration bridge then converts the
readout value into a true-oracle error statement for `fstar` with the same
additive `2 * εf` slack.  This still does not imply that scalar query answers
at child nodes merge.

## Scope Boundary

The algebraic state/readout story is fully mechanized.  MG and SpaceSaving are
now explicitly packaged as `SizedMergeableQuerySketch` examples, while GK is
explicitly packaged as a `OneWaySizedQuerySketch` example rather than as full
state-state mergeability.  The MG, SpaceSaving, and GK files still package some
algorithm-specific invariants as bundles and prove the Agarwal conclusions from
those bundles.  MG now has a
concrete executable update/build core, a proved capacity invariant, proved
total counter-mass accounting, positivity preservation, the traced
potential/debt induction that charges global decrement steps to blocks of
`k+1` processed items, the full Lemma 2.1 executable frequency envelope, and
the shared combine/prune algebra used by `MERGEABLEMINERROR` and
`MERGEABLEMINSPACE`.  What remains for those two merge algorithms is the
concrete threshold-selection implementation/certificate, not the envelope
algebra.  SpaceSaving now uses a true minimum-counter replacement update for
its executable bookkeeping, while its full mergeability is still obtained
through the MG isomorphism bundle.  The interval and finite-VC range-space surfaces now include checked
state-level/tree-error transport once a valid-state error invariant is
supplied; they also include exact all-points witnesses for generic
ε-approximation and range-space error, mathlib-backed VC/Sauer-Shelah trace
growth, and the finite-trace uniform failure union bound.  The adjacent
epsilon-kernel geometry includes analogous tree-width-error transport plus
translation/scaling laws for directional width, but it is not part of the local
PDF claim inventory.  These witnesses are
linear-size baselines, so they close the algebraic/error layer without claiming
Agarwal's compact construction.  The same-weight interval layer also includes
the Theorem 3.2 CFF triangle inequality for one-way quantile merging, the
deterministic halving core of Lemma 3, the complete-tree scaling arithmetic used
by Lemma 4, the Azuma/Hoeffding tail theorem for a complete halving tree encoded
as an explicit martingale-difference process, the finite-layer summation
argument behind the logarithmic uneven-weight scheme, and the hybrid
sample-plus-summary error composition.  The Section 4 range-space layer now
has the analogous low-discrepancy merge algebra: a supplied coloring certificate
gives Lemma 4.1 unbiasedness and discrepancy-bounded error, level scaling gives
the Lemma 4.3 radius, and a complete-tree martingale process gives the
Azuma/Hoeffding tail.  The external content left for Section 4 is constructing
the coloring with the cited discrepancy rate and closing the compact size
theorem.
The remaining external content is the nontrivial compact-construction layer:
the range-space discrepancy construction, the algorithm-specific MG threshold
selection routines for min-error/min-space pruning, and the full executable
MG/SpaceSaving isomorphism proof beyond the checked bookkeeping/isomorphism
certificate surface.  The epsilon-kernel layer is intentionally outside the PDF claim
inventory and should be treated as adjacent background until separately cited.
