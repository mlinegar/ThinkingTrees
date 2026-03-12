# Paper-to-Lean Map

This file is the repository-side theorem map for the paper. The paper appendix
intentionally stays human-readable; low-level Lean identifiers live here.

For the curated Lean entry point, start from:

- `lean3/FormalProofs/OPT/MainTheorems.lean`

For proof walkthroughs in ordinary mathematical language, see:

- `lean3/docs/CORE_PROOFS.md`

## Main paper results

| Paper result | Paper label | Lean theorem / export | File |
|---|---|---|---|
| Inductive preservation | `thm:one-pass` | `one_pass` | `FormalProofs/OPT/PreservationTheorems.lean` |
| Schedule invariance | `cor:schedule` | `schedule_invariance` | `FormalProofs/OPT/PreservationTheorems.lean` |
| Fold-of-folds invariance | `cor:folds` | `fold_of_folds` | `FormalProofs/OPT/PreservationTheorems.lean` |
| Multi-round preservation | `thm:multi-round` | `multi_round_proper` | `FormalProofs/OPT/ExpectationTheory.lean` |
| DPO equivalence | `thm:dpo-equiv` | `dpo_equivalence` | `FormalProofs/OPT/PreferenceBounds.lean` |
| GRPO-PL equivalence | `thm:grpo-pl` | `grpo_equivalence` | `FormalProofs/OPT/PreferenceLearning.lean` |
| GRPO-RL equivalence | `thm:grpo-rl` | `grpo_rl_equivalence` | `FormalProofs/OPT/PreferenceLearning.lean` |
| Unified preference gap | `thm:unified-gap` | `unified_preference_gap_bounded` | `FormalProofs/OPT/PreferenceBounds.lean` |
| Expected-tree optimizer transfer | `thm:expected-tree-opt` | `oracleMeasurableParamArgmin_subset_pointwiseEpsilonArgmin_of_expectedTree_loss_perturbation` | `FormalProofs/OPT/OptimizationPerturbation.lean` |
| L3 is independent | `thm:l3-necessary` | `thm10_1_L3_not_derivable` | `FormalProofs/OPT/CounterexampleExistence.lean` |

## Adaptive-tree preference results

These are the Lean results behind the adaptive-tree and oracle-uncertainty
discussion in the paper.

| Paper discussion | Lean theorem / export | File |
|---|---|---|
| Tree-level DPO optimizer transfer | `dpo_expected_tree_argmin_subset_true_pointwiseEpsilonArgmin_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement` | `FormalProofs/OPT/OptimizationPerturbation.lean` |
| Tree-level GRPO-PL optimizer transfer | `grpo_pl_expected_tree_argmin_subset_true_pointwiseEpsilonArgmin_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement` | `FormalProofs/OPT/OptimizationPerturbation.lean` |
| Tree-level GRPO-RL optimizer transfer | `grpo_rl_expected_tree_argmin_subset_true_pointwiseEpsilonArgmin_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement` | `FormalProofs/OPT/OptimizationPerturbation.lean` |
| Generic high-probability expected-tree transfer | `oracleMeasurableParamArgmin_failure_prob_le_of_good_event_expectedTree_pointwiseTransfer` | `FormalProofs/OPT/OptimizationPerturbation.lean` |
| High-probability tree-level DPO transfer | `dpo_expected_tree_true_pointwiseEpsilonArgmin_failure_prob_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement` | `FormalProofs/OPT/OptimizationPerturbation.lean` |
| High-probability tree-level GRPO-PL transfer | `grpo_pl_expected_tree_true_pointwiseEpsilonArgmin_failure_prob_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement` | `FormalProofs/OPT/OptimizationPerturbation.lean` |
| High-probability tree-level GRPO-RL transfer | `grpo_rl_expected_tree_true_pointwiseEpsilonArgmin_failure_prob_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement` | `FormalProofs/OPT/OptimizationPerturbation.lean` |

## Appendix-level bridges

| Paper appendix topic | Lean entry points | File(s) |
|---|---|---|
| Score transport / Blackwell-style bridge | `blackwell_transport'`, `condexp_oracle_factored'` | `FormalProofs/OPT/ScoreTransport.lean` |
| DPO gap bounds | `dpo_gap_bounded` | `FormalProofs/OPT/PreferenceBounds.lean` |
| GRPO bounds | `grpo_pl_gap_bounded`, `grpo_rl_gap_bounded` | `FormalProofs/OPT/PreferenceBounds.lean` |
| Training-path composition | `training_path_gap_bound_abstract` and successors | `FormalProofs/OPT/TrainingPipeline.lean` |
| Mergeable-summary bridge | `ops_mergeClosed_of_global`, `ops_hierarchical_mergeable_of_global`, `ops_reduction_to_classical_mergeable` | `FormalProofs/OPT/MergeableReduction.lean` |
| Tree gap transport for sketches | `tree_gap_bound_transport_upper`, `tree_gap_bound_transport_upper_prob` | `FormalProofs/DSL/MergeableCertificates.lean`, `FormalProofs/OPT/MergeableCertificates.lean` |

## How to use this map

1. Read the human proof in the paper appendix.
2. Use this file to jump to the corresponding Lean theorem.
3. Use `lean3/docs/CORE_PROOFS.md` for a step-by-step bridge from the paper proof to the Lean proof structure.
