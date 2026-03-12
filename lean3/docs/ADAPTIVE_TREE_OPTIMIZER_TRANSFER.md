# Adaptive-Tree Optimizer Transfer

This note collects the Lean results behind the paper's adaptive-tree optimizer
story.

## 1. Generic expected-tree perturbation

The core object is the expected tree objective

- `ExpectedAdaptiveTreeObjective`
  in `FormalProofs/OPT/OptimizationPerturbation.lean`

The generic perturbation step is:

- `oracleMeasurableParamArgmin_subset_pointwiseEpsilonArgmin_of_expectedTree_loss_perturbation`

Interpretation:

- if the expected absolute gap between the true objective and the tree objective
  is small for each parameter,
- then exact minimizers of the expected tree objective are pointwise
  near-optimal for the true objective.

There is also a uniform-slack version:

- `oracleMeasurableParamArgmin_subset_epsilonArgmin_of_expectedTree_loss_perturbation`

## 2. Method-specific corollaries

The generic theorem is specialized to the three preference-learning families:

- DPO:
  `dpo_expected_tree_argmin_subset_true_pointwiseEpsilonArgmin_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement`
- GRPO-PL:
  `grpo_pl_expected_tree_argmin_subset_true_pointwiseEpsilonArgmin_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement`
- GRPO-RL:
  `grpo_rl_expected_tree_argmin_subset_true_pointwiseEpsilonArgmin_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement`

All three live in:

- `FormalProofs/OPT/OptimizationPerturbation.lean`

Interpretation:

- approximate local laws create the transport budget,
- oracle uncertainty creates the measurement budget,
- the optimizer slack is the sum of those two expected terms.

## 3. High-probability wrappers

If the adaptive-tree certificate is only known on a good event, use:

- `oracleMeasurableParamArgmin_failure_prob_le_of_good_event_expectedTree_pointwiseTransfer`

Method-specific high-probability corollaries:

- DPO:
  `dpo_expected_tree_true_pointwiseEpsilonArgmin_failure_prob_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement`
- GRPO-PL:
  `grpo_pl_expected_tree_true_pointwiseEpsilonArgmin_failure_prob_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement`
- GRPO-RL:
  `grpo_rl_expected_tree_true_pointwiseEpsilonArgmin_failure_prob_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement`

Interpretation:

- certification uncertainty only appears through the failure probability of the
  good event,
- conditional on the good event, the same optimizer-transfer statement holds.

## 4. Related files

- Expected gap bounds for stochastic adaptive trees:
  `FormalProofs/OPT/AdaptiveChunkingBridge.lean`
- Training-path composition:
  `FormalProofs/OPT/TrainingPipeline.lean`
- Curated exports:
  `FormalProofs/OPT/MainTheorems.lean`
- Theorem index:
  `FormalProofs/OPT/README.lean`
