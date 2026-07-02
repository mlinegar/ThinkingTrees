# Feldman 2006/2008 MUD Proof Map

This document maps Feldman et al., "On the Complexity of Processing Massive,
Unordered, Distributed Data" to the Lean surface in:

- `FormalProbability/ML/MergeableSummaries/Feldman2008.lean`
- `FormalProbability/ML/MergeableSummaries/LiteratureChronology.lean`
- `FormalProbability/ML/MergeableSummaries.lean`

The formalization is split between checked algebra, checked semantic
streaming-to-MUD constructions, and citation-backed lower-bound schemas.  The
MUD algebraic layer, representative-state semantic construction, and
simultaneous-communication protocol construction are mechanized.  The Big-O
polylog closure facts and SCM lower-bound transport lemmas are also checked.
The remaining external pieces are machine-level Savitch space accounting and
the problem-specific communication lower-bound proofs.

## Mechanized Core

| Paper concept | Lean name | Status |
|---|---|---|
| Symmetric function | `Feldman2008.SymmetricFunction`, `feldman2008_06_symmetric_function_iff` | Mechanized definition |
| Polylogarithmic rate vocabulary | `Feldman2008.polylogRate`, `Feldman2008.PolylogRate`, `Feldman2008.AtLeastLogRate` | Mechanized using mathlib `BigO` |
| Squared-space rate and polylog closure | `Feldman2008.squareRate`, `Feldman2008.PolylogRate.square`, `feldman2008_06a_polylog_square` | Mechanized with mathlib `IsBigO.pow` |
| MUD local map / merge / readout | `MUDAggregator.mapItem`, `MUDAggregator.merge`, `MUDAggregator.readout` | Existing mechanized core |
| MUD append homomorphism | `MUDAggregator.build_append`, `feldman2006_01_mud_build_append` | Mechanized |
| Canonical-state merge closure | `MUDAggregator.mergeClosed`, `feldman2006_02_mud_merge_closed` | Mechanized |
| Permutation-invariant state fold | `MUDAggregator.build_perm`, `feldman2006_03_mud_build_permutation_invariant` | Mechanized |
| Permutation-invariant readout | `MUDAggregator.readout_perm`, `feldman2006_04_mud_readout_permutation_invariant` | Mechanized |
| State-level mergeable summary bridge | `MUDAggregator.toStateLevelSummary`, `feldman2006_05_mud_state_level_mergeable` | Mechanized |
| General paper-MUD model without algebraic state laws | `Feldman2008.GeneralMUDAlgorithm` | Mechanized interface |
| Algebraic MUD as paper-MUD special case | `Feldman2008.GeneralMUDAlgorithm.ofAggregator`, `Feldman2008.GeneralMUDAlgorithm.ofAggregator_evalState` | Mechanized |

## Item-Level Computation Trees

The paper requires correctness for all binary computation trees over local
messages.  The new module adds an item-level tree separate from the existing
ordered chunk tree.

| Paper concept | Lean name | Status |
|---|---|---|
| Item-level computation tree | `Feldman2008.ComputationTree` | Mechanized |
| Leaf data represented by a tree | `Feldman2008.ComputationTree.data` | Mechanized |
| Leaf count | `Feldman2008.ComputationTree.leafCount` | Mechanized |
| State evaluation of a tree | `Feldman2008.ComputationTree.evalState` | Mechanized |
| Readout evaluation of a tree | `Feldman2008.ComputationTree.evalReadout` | Mechanized |
| Tree state equals canonical fold | `Feldman2008.ComputationTree.evalState_eq_build_data`, `feldman2008_07_item_tree_state_eq_build` | Mechanized |
| Tree readout equals canonical fold readout | `Feldman2008.ComputationTree.evalReadout_eq_build_readout`, `feldman2008_08_item_tree_readout_eq_build` | Mechanized |
| Readout invariance across permuted trees | `Feldman2008.ComputationTree.evalReadout_eq_of_data_perm`, `feldman2008_09_item_tree_readout_permutation_invariant` | Mechanized |

## Costed MUD and Streaming Classes

| Paper concept | Lean name | Status |
|---|---|---|
| Costed MUD algorithm | `Feldman2008.CostedMUDAlgorithm` | Mechanized interface |
| MUD exact computation | `Feldman2008.CostedMUDAlgorithm.Computes` | Mechanized predicate |
| MUD all-tree computation | `Feldman2008.CostedMUDAlgorithm.ComputesOnAllTrees` | Mechanized predicate |
| Canonical fold correctness implies all-tree correctness | `Feldman2008.CostedMUDAlgorithm.computesOnAllTrees_of_computes` | Mechanized |
| MUD-computed functions are symmetric | `Feldman2008.CostedMUDAlgorithm.computed_function_symmetric` | Mechanized |
| Polylog MUD computability | `Feldman2008.PolylogMUDComputable` | Mechanized predicate |
| Streaming algorithm | `Feldman2008.StreamingAlgorithm` | Mechanized interface |
| Streaming state after a suffix | `Feldman2008.StreamingAlgorithm.runFrom` | Mechanized |
| Streaming run from initial state | `Feldman2008.StreamingAlgorithm.run` | Mechanized |
| Streaming append identity | `Feldman2008.StreamingAlgorithm.runFrom_append` | Mechanized |
| Lemma 1, continuation congruence | `Feldman2008.StreamingAlgorithm.lemma1_streaming_state_congruence_append`, `feldman2008_10_streaming_state_congruence_append` | Mechanized |
| Symmetric computed output is permutation-invariant | `Feldman2008.StreamingAlgorithm.readout_perm_of_computes_symmetric` | Mechanized |
| Costed streaming algorithm | `Feldman2008.CostedStreamingAlgorithm` | Mechanized interface |
| Polylog streaming computability | `Feldman2008.PolylogStreamingComputable` | Mechanized predicate |
| Polylog general paper-MUD computability | `Feldman2008.PolylogGeneralMUDComputable` | Mechanized predicate |

## Representative-State Simulation

This section mechanizes the semantic content of Feldman Theorem 1.  It does not
claim to mechanize the low-level Savitch machine-space implementation.

| Paper concept | Lean name | Status |
|---|---|---|
| Reachable state at exact length | `Feldman2008.StreamingAlgorithm.ReachableAtLength` | Mechanized |
| Reachable state from another state | `Feldman2008.StreamingAlgorithm.ReachableFromAtLength` | Mechanized |
| Readout-context equivalence | `Feldman2008.StreamingAlgorithm.ReadoutContextEq` | Mechanized |
| Context equivalence from equal state | `Feldman2008.StreamingAlgorithm.readoutContextEq_of_run_eq` | Mechanized |
| Block replacement after a prefix using symmetry | `Feldman2008.StreamingAlgorithm.readoutContextEq_append_right_of_symmetric` | Mechanized with mathlib list permutations |
| Concatenating context-equivalent blocks | `Feldman2008.StreamingAlgorithm.readoutContextEq_append_of_symmetric` | Mechanized |
| Pair replacement frontier algebra | `Feldman2008.StreamingAlgorithm.readoutContextEq_pair_replacement_of_run_eq` | Mechanized |
| Lemma 2, semantic representative merge existence | `Feldman2008.StreamingAlgorithm.lemma2_representative_merge_exists`, `feldman2008_10b_representative_merge_exists` | Mechanized |
| Noncomputable representative merge | `Feldman2008.representativeMerge` | Mechanized definition |
| Representative merge spec | `Feldman2008.representativeMerge_spec`, `representativeMerge_fst`, `representativeMerge_reachable` | Mechanized |
| General MUD from streaming | `Feldman2008.representativeMUDFromStreaming` | Mechanized |
| Representative MUD has polylog costs when source streaming costs are polylog | `Feldman2008.representativeMUDFromStreaming_hasPolylogCosts` | Mechanized |
| Guess/context invariant over trees | `Feldman2008.representativeMUD_evalState_guess_contextEq` | Mechanized |
| Representative MUD computes on every binary tree | `Feldman2008.representativeMUDFromStreaming_computesOnAllTrees`, `feldman2008_13b1_representativeMUDFromStreaming_computesOnAllTrees` | Mechanized |
| Theorem 1, semantic streaming-to-general-MUD | `Feldman2008.deterministic_streaming_to_representative_mud`, `Feldman2008.theorem1_deterministic_streaming_to_mud_semantic`, `feldman2008_13b_theorem1_deterministic_streaming_to_mud_semantic` | Mechanized |
| Polylog symmetric streaming subset of general paper-MUD | `Feldman2008.polylog_streaming_subset_general_mud`, `feldman2008_13c_polylog_streaming_subset_general_mud` | Mechanized modulo machine-level Savitch implementation |

## MUD-to-Streaming Inclusion

The easy direction in the paper is fully mechanized.

| Paper claim | Lean name | Status |
|---|---|---|
| Sequential streaming algorithm induced by MUD | `Feldman2008.streamingFromMUD` | Mechanized |
| Induced stream state appends canonical MUD state | `Feldman2008.streamingFromMUD_runFrom_eq_merge_build` | Mechanized |
| Induced stream state equals MUD build from empty | `Feldman2008.streamingFromMUD_run_eq_build`, `feldman2008_11_streaming_from_mud_run_eq_build` | Mechanized |
| Costed MUD to costed streaming conversion | `Feldman2008.CostedMUDAlgorithm.toStreaming` | Mechanized |
| Computation is preserved by conversion | `Feldman2008.CostedMUDAlgorithm.toStreaming_computes` | Mechanized |
| Polylog MUD subset streaming | `Feldman2008.mud_polylog_subset_streaming`, `feldman2008_12_mud_polylog_subset_streaming` | Mechanized |

## Promise, Indeterminate, and Randomized Surfaces

| Paper concept | Lean name | Status |
|---|---|---|
| Promise-domain symmetry | `Feldman2008.PromiseSymmetric` | Mechanized predicate |
| MUD computation on a promise | `Feldman2008.CostedMUDAlgorithm.ComputesOnPromise` | Mechanized predicate |
| Streaming computation on a promise | `Feldman2008.CostedStreamingAlgorithm.ComputesOnPromise` | Mechanized predicate |
| Polylog promise MUD/streaming computability | `Feldman2008.PolylogMUDComputableOnPromise`, `Feldman2008.PolylogStreamingComputableOnPromise` | Mechanized predicates |
| Indeterminate symmetry | `Feldman2008.IndeterminateSymmetric` | Mechanized predicate |
| MUD/streaming indeterminate computation | `Feldman2008.CostedMUDAlgorithm.ComputesIndeterminate`, `Feldman2008.CostedStreamingAlgorithm.ComputesIndeterminate` | Mechanized predicates |
| Polylog indeterminate MUD/streaming computability | `Feldman2008.PolylogMUDComputableIndeterminate`, `Feldman2008.PolylogStreamingComputableIndeterminate` | Mechanized predicates |
| Public-randomness streaming family | `Feldman2008.PublicRandomStreamingFamily` | Mechanized interface |
| Seedwise deterministic algorithm extracted from a public-random streaming family | `Feldman2008.PublicRandomStreamingFamily.seedAlgorithm` | Mechanized |
| Seedwise correctness for public-random streaming | `Feldman2008.PublicRandomStreamingFamily.ComputesSeedwise` | Mechanized predicate |
| Seed success set bookkeeping | `Feldman2008.PublicRandomStreamingFamily.SuccessSet`, `Feldman2008.PublicRandomStreamingFamily.successSet_eq_univ_of_computesSeedwise`, `feldman2008_15j_publicRandom_successSet_eq_univ_of_computesSeedwise` | Mechanized |
| Public-random success probability bookkeeping | `Feldman2008.PublicRandomStreamingFamily.SuccessProbability`, `Feldman2008.PublicRandomStreamingFamily.ComputesWithSuccessAtLeast`, `Feldman2008.PublicRandomStreamingFamily.successProbability_eq_one_of_computesSeedwise`, `Feldman2008.PublicRandomStreamingFamily.computesWithSuccessAtLeast_of_computesSeedwise`, `feldman2008_15j1_publicRandom_successProbability_eq_one_of_computesSeedwise`, `feldman2008_15j2_publicRandom_computesWithSuccessAtLeast_of_computesSeedwise` | Mechanized for the exact seedwise case using mathlib measure vocabulary |
| Public-randomness MUD family | `Feldman2008.PublicRandomMUDFamily` | Mechanized interface |
| Seedwise symmetry for MUD families | `Feldman2008.PublicRandomMUDFamily.seedwiseSymmetric_of_mud` | Mechanized |
| Public-randomness general paper-MUD family | `Feldman2008.PublicRandomGeneralMUDFamily` | Mechanized interface |
| Public-random streaming-to-general-MUD conversion | `Feldman2008.publicRandomRepresentativeMUDFromStreaming`, `Feldman2008.publicRandomRepresentativeMUD_computesSeedwise`, `Feldman2008.publicRandomRepresentativeMUD_hasPolylogCosts` | Mechanized seedwise semantic construction |
| Public-random general-MUD success probability over computation trees | `Feldman2008.PublicRandomGeneralMUDFamily.SuccessSetOnTree`, `SuccessProbabilityOnTree`, `ComputesWithSuccessAtLeast`, `successSetOnTree_eq_univ_of_computesSeedwise`, `successProbabilityOnTree_eq_one_of_computesSeedwise`, `computesWithSuccessAtLeast_of_computesSeedwise` | Mechanized for all binary computation trees in the exact seedwise case |

## Simultaneous Communication

| Paper concept | Lean name | Status |
|---|---|---|
| Two-party SCM protocol | `Feldman2008.SCMProtocol` | Mechanized interface |
| SCM correctness over split streams | `Feldman2008.SCMProtocol.Computes` | Mechanized predicate |
| SCM protocol induced by streaming | `Feldman2008.scmFromStreaming` | Mechanized |
| SCM correctness from symmetric streaming | `Feldman2008.scmFromStreaming_computes` | Mechanized |
| Theorem 2, streaming to SCM protocol | `Feldman2008.streaming_to_scm_protocol`, `Feldman2008.theorem2_streaming_to_scm_semantic`, `feldman2008_14b_theorem2_streaming_to_scm_semantic` | Mechanized |
| Polylog SCM computability | `Feldman2008.PolylogSCMComputable` | Mechanized predicate |
| SCM lower-bound language | `Feldman2008.SCMCommunicationLowerBound`, `Feldman2008.SCMCommunicationLowerBoundOnPromise`, `Feldman2008.SuperPolylogRate` | Mechanized predicates |
| SCM lower bound rules out polylog SCM | `Feldman2008.not_polylog_scm_of_lower_bound`, `Feldman2008.not_polylog_scm_on_promise_of_lower_bound` | Mechanized |
| SCM lower bound rules out polylog streaming for symmetric functions | `Feldman2008.not_polylog_streaming_of_scm_lower_bound`, `feldman2008_14d_not_polylog_streaming_of_scm_lower_bound` | Mechanized via Theorem 2 |
| Finite Boolean-vector equality target | `Feldman2008.boolVectorEquality`, `Feldman2008.boolVectorEquality_scm_sqrt_lower_bound_statement`, `feldman2008_15e_boolVectorEquality_scm_sqrt_lower_bound_statement` | Mechanized target function; randomized lower-bound theorem remains imported as a citation obligation |
| Deterministic finite-message equality lower bound | `Feldman2008.FiniteTwoPartyProtocol`, `Feldman2008.BitAccountedTwoPartyProtocol`, `Feldman2008.boolVectorEquality_sendA_injective_finite`, `Feldman2008.boolVectorEquality_messageA_card_lower`, `Feldman2008.boolVectorEquality_bitsA_lower`, `Feldman2008.BitAccountedEqualityProtocolFamily.linear_bigO_lower`, `feldman2008_15e1_boolVectorEquality_messageA_card_lower`, `feldman2008_15e2_boolVectorEquality_bitsA_lower`, `feldman2008_15e3_bitAccountedEquality_linear_bigO_lower` | Mechanized finite deterministic lower bound: equality forces injective messages, `2^n` messages, and `Omega(n)` bit-accounted communication |
| Finite Set Parity target and equality reduction | `Feldman2008.finSetParity`, `Feldman2008.finSetParityRecords`, `Feldman2008.finSetParity_two_vectors_eq`, `Feldman2008.finSetParity_symmetric`, `Feldman2008.equalityProtocolFromFinSetParity_computes`, `Feldman2008.finSetParity_scm_lower_bound_of_equality`, `Feldman2008.finiteSetParity_scm_sqrt_lower_bound_of_equality`, `Feldman2008.equalityBitProtocolFromFinSetParity_computes`, `Feldman2008.finSetParity_bitAccounted_bitsA_lower`, `Feldman2008.BitAccountedFinSetParitySCMFamily.linear_bigO_lower`, `feldman2008_15c_finSetParity_two_vectors_eq`, `feldman2008_15d_finSetParity_symmetric`, `feldman2008_15h_finiteSetParity_scm_sqrt_lower_bound_of_equality`, `feldman2008_15h1_finSetParity_bitAccounted_bitsA_lower`, `feldman2008_15h2_bitAccountedFinSetParity_linear_bigO_lower` | Mechanized split-stream reduction from equality to finite Set Parity, including deterministic bit-accounted `Omega(n)` family lower bound |
| Private-coin Set Parity reduction | `Feldman2008.PrivateCoinBitAccountedTwoPartyProtocol`, `Feldman2008.PrivateCoinBitAccountedSCMComputesWithSuccess`, `Feldman2008.privateCoinEqualityProtocolFromFinSetParity_successCount`, `Feldman2008.privateCoinEqualityProtocolFromFinSetParity_computesWithSuccess`, `Feldman2008.PrivateCoinBitAccountedFinSetParitySCMFamily.toEqualityFamily`, `Feldman2008.privateCoinEquality_scm_sqrt_lower_bound_statement`, `Feldman2008.privateCoinFinSetParity_scm_sqrt_lower_bound_of_equality`, `feldman2008_15h3_privateCoinFinSetParity_success_preserved`, `feldman2008_15h4_privateCoinFinSetParity_scm_sqrt_lower_bound_of_equality` | Mechanized private-coin finite seed-count model and proof that the Set Parity reduction preserves bounded-error success and communication |
| Nat-index Set Parity target function | `Feldman2008.setParity`, `Feldman2008.setParity_symmetric`, `feldman2008_15b_setParity_symmetric` | Function and symmetry mechanized |
| Set Parity SCM lower-bound obligation | `Feldman2008.setParity_scm_sqrt_lower_bound_statement`, `Feldman2008.finiteSetParity_scm_sqrt_lower_bound_statement`, `feldman2008_15c_setParity_scm_sqrt_lower_bound_statement`, `feldman2008_15g_finiteSetParity_scm_sqrt_lower_bound_statement` | Typed citation statements |
| Symmetric Index promise problem | `Feldman2008.SymmetricIndexRecord`, `Feldman2008.symmetricIndexCanonical`, `Feldman2008.symmetricIndexDomain`, `Feldman2008.symmetricIndexCanonical_mem_domain`, `Feldman2008.symmetricIndexCanonical_readout_eq`, `Feldman2008.symmetricIndex`, `Feldman2008.symmetricIndex_promise_symmetric`, `feldman2008_16a_symmetricIndexCanonical_mem_domain`, `feldman2008_16a1_symmetricIndexCanonical_readout_eq`, `feldman2008_16b_symmetricIndex_promise_symmetric` | Concrete promised problem surface, canonical readout correctness, and promise symmetry mechanized |
| Symmetric Index SCM lower-bound obligation | `Feldman2008.symmetricIndex_scm_linear_lower_bound_statement`, `feldman2008_16b_symmetricIndex_scm_linear_lower_bound_statement`, `feldman2008_16c_symmetricIndex_scm_linear_lower_bound_statement` | Typed citation statement |

## Citation Schemas

These statements are typed and searchable, but their external proofs are not
mechanized in this repository.

| Paper theorem | Lean name | Boundary |
|---|---|---|
| Corollary, `MUD = SS` for total deterministic symmetric functions | `Feldman2008.mud_eq_streaming_statement` | Needs the general-MUD-to-streaming direction for the nonempty/no-identity paper model |
| Public-randomness seedwise extension | `Feldman2008.public_randomness_seedwise_extension_statement`, `feldman2008_15_public_randomness_seedwise_extension_statement` | Seedwise deterministic reduction |
| Public-randomness seedwise streaming-to-general-MUD theorem | `Feldman2008.public_randomness_seedwise_general_mud_statement`, `Feldman2008.public_randomness_seedwise_general_mud`, `feldman2008_15k_public_randomness_seedwise_general_mud` | Mechanized seedwise construction over `PublicRandomGeneralMUDFamily`; exact seedwise success-probability wrappers are checked, while nontrivial bounded-error and independence/concentration arguments remain external |
| Theorem 3, private-randomness separation | `Feldman2008.theorem3_private_randomness_separation_statement`, `Feldman2008.setParity_scm_sqrt_lower_bound_statement`, `Feldman2008.privateCoinFinSetParity_scm_sqrt_lower_bound_statement`, `feldman2008_16_theorem3_private_randomness_separation_statement` | Private-coin seed-count model and Set Parity reduction are mechanized; the randomized equality lower-bound proof itself remains external |
| Theorem 4, promise-problem separation | `Feldman2008.theorem4_promise_separation_statement`, `Feldman2008.symmetricIndex_scm_linear_lower_bound_statement`, `feldman2008_17_theorem4_promise_separation_statement` | Problem-specific Symmetric Index promise lower-bound proof remains external |
| Theorem 5, indeterminate-function separation | `Feldman2008.theorem5_indeterminate_separation_statement`, `feldman2008_18_theorem5_indeterminate_separation_statement` | Reduction from promise separation |

## C-TreePO Bridge Names

`FormalProofs/OPT/MergeableReduction.lean` re-exports the Feldman surface under
C-TreePO-facing names.  In addition to the deterministic MUD, streaming, SCM,
and concrete Set Parity/Symmetric Index aliases listed in the main literature
map, the current bridge exposes the randomized citation surfaces as:

- `ctreepo_feldman2008_public_randomness_seedwise_extension_statement`
- `ctreepo_feldman2008_representativeMUDFromStreaming_computesOnAllTrees`
- `ctreepo_feldman2008_publicRandom_successSet_eq_univ_of_computesSeedwise`
- `ctreepo_feldman2008_publicRandom_successProbability_eq_one_of_computesSeedwise`
- `ctreepo_feldman2008_public_randomness_seedwise_general_mud`
- `ctreepo_feldman2008_theorem3_private_randomness_separation_statement`

## Remaining Work

- Add a finite-state machine/reachability layer that accounts for the
  deterministic `O(g(n)^2)` Savitch implementation of `representativeMerge`.
- Finish the nonempty/no-identity general-MUD-to-streaming inclusion, or
  restrict the paper-MUD model to an explicit identity state and use the already
  mechanized algebraic direction.
- Extend the public/private probability layer beyond exact seedwise success to
  nontrivial bounded-error success, independence assumptions, and randomized SCM
  lower-bound probability arguments.
- Prove the randomized Boolean-vector equality and Symmetric Index SCM lower
  bounds locally.  The deterministic finite-message equality lower bound, the
  finite Set Parity lower bound, and the private-coin Set Parity reduction are
  mechanized; the randomized equality lower-bound proof and the Symmetric Index
  promise lower bound remain the hard external pieces.
