import FormalProofs.OPT.ExpectationTheory
import FormalProofs.OPT.PreferenceLearning
import FormalProofs.OPT.PreferenceBounds
import FormalProofs.OPT.TrainingPipeline
import FormalProofs.OPT.MergeableReduction
import FormalProofs.OPT.SketchFlipMergeBridge
import FormalProofs.OPT.SketchSummaryOperators
import FormalProofs.OPT.SketchRecovery
import FormalProofs.OPT.SketchRecoveryInstances
import FormalProofs.OPT.ApproximateLocalLaws
import FormalProofs.OPT.RegularizedObjective
import FormalProofs.OPT.OptimizationPerturbation
import FormalProofs.OPT.AdaptiveChunkingBridge
import FormalProofs.OPT.RUMSufficientConditions
import FormalProofs.OPT.GlobalAssumptions
import FormalProofs.DSL.TreeIPW
import FormalProofs.OPT.OracleUtility
import FormalProofs.OPT.ExactUtilityTransport
import FormalProofs.OPT.ExactUtilityTransportInstances
import FormalProofs.OPT.AdversarialChunkingExample
import FormalProofs.OPT.MarkovPathDGP
import FormalProofs.OPT.SerflingAudit
import FormalProofs.OPT.NamespaceCompat

/-!
# Main Theorems: Local-to-Global Oracle Preference Learning

This file collects and documents the main results of the formalization. These theorems
establish that **local consistency conditions imply global training equivalence** for
a broad class of preference learning methods including DPO, GRPO, and PPO-style RL.

## The Core Insight

The central contribution is NOT the distillation/gap-composition results (which follow
from triangle inequality). Rather, it is the **local-to-global mechanism**:

1. **Local Laws (L1, L2, L3)**: Testable conditions on a summarizer `g`
2. **Zero Distortion**: L1 + L2 + L3 imply `E[dist(f*(Z), f*(x))] = 0` after R rounds
3. **Oracle-Measurability**: When loss/generator depend on x only through f*(x)
4. **Training Equivalence**: Optimal policies on summarized data = optimal on originals

This means we can **test locally** (audit the summarizer) and **conclude globally**
(training on summaries is as good as training on full documents).

## Theorem Hierarchy

```
                    Local Laws (L1, L2, L3)
                           │
                           ▼
              multi_round_proper (ExpectationTheory)
              E[D(Z^R, x)] = 0 for all R ≥ 1
                           │
                           ▼
         ┌─────────────────┴─────────────────┐
         ▼                                   ▼
  preference_learning_equivalence     grpo_rl_equivalence
  (PreferenceLearning.lean)           (PreferenceLearning.lean)
         │                                   │
         ▼                                   ▼
  dpo_gap_zero_of_local_laws         DPO/GRPO training sound
  (DPO.lean)                         on summarized data
```

## Coverage of Modern Methods

The formalization captures:

| Method | File | Key Theorem |
|--------|------|-------------|
| DPO | DPO.lean | `dpo_equivalence` |
| Plackett-Luce GRPO | PreferenceLearning.lean | `grpo_equivalence` |
| GRPO-RL (DeepSeek-R1) | PreferenceLearning.lean | `grpo_rl_equivalence` |
| General pairwise | PreferenceLearning.lean | `preference_learning_equivalence` |
| General group-wise | PreferenceLearning.lean | `expected_group_loss_eq_of_zero_dist` |

-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

noncomputable section

namespace MainTheorems

/-!
## Theorem 1: Multi-Round Preservation

**Statement**: If local laws L1, L2, L3 hold for summarizer g on tree T, then
after R rounds of summarization, the expected oracle distortion is exactly zero.

**Significance**: This is the foundational result. It shows that local testable
conditions (which an auditor can check on individual summarizer calls) guarantee
global preservation of oracle information through arbitrary reduction depth.

**Paper Reference**: Theorem 5.1 (Multi-Round Preservation)
-/

/-- Multi-round preservation: local laws imply zero expected distortion.

This is the core theorem enabling local-to-global inference. When a summarizer
satisfies L1 (leaf sufficiency), L2 (merge consistency), and L3 (idempotence),
the expected distortion after R rounds of tree reduction is exactly 0.

Mathematical statement:
  L1 ∧ L2 ∧ L3 ∧ R ≥ 1 ⟹ E_{z~ZR(g,x,R,T)}[dist(f*(z), f*(x))] = 0

Proof: Induction on R, using L1+L2 for base case and L3 for inductive step. -/
abbrev multi_round_preservation := @multi_round_proper

/-- Named alias used for fixed-partition extensions:
instantiate `multi_round_proper` at a deterministic tree map `x ↦ T(x)`. -/
abbrev fixed_partition_extension_instantiation := @multi_round_proper

/-- Coupling-form distortion equals document-level `Δ_R_ZR` when `μ_X = pure(x)`. -/
abbrev coupling_delta_eq_delta_r_zr := @coupling_Δ_eq_Δ_R_ZR

/-- Under local laws, the document-level distortion `Δ_R_ZR` is exactly zero. -/
abbrev delta_r_zr_zero_of_local_laws := @Δ_R_eq_zero_of_local_laws

/-!
## Theorem 2: Preference Learning Equivalence

**Statement**: When a preference loss is oracle-measurable and the pair/group
generator is oracle-indexed, zero distortion implies equal expected loss.

**Significance**: This abstracts over ALL preference learning methods. Any method
where the loss depends on documents only through oracle values inherits the
equivalence property.
-/

/-- General preference learning equivalence under zero distortion.

When summaries preserve oracle values (dist(f*(z), f*(x)) = 0 for all z in
summary support), any oracle-measurable preference learning method achieves
identical expected loss on summaries vs. originals.

This theorem abstracts over:
- DPO (pairwise, Bradley-Terry)
- GRPO (k-wise, Plackett-Luce)
- GRPO-RL (clipped surrogate + KL)
- Any future method satisfying oracle-measurability -/
abbrev preference_learning_equiv := @preference_learning_equivalence

/-!
## Theorem 3: DPO Training Soundness

**Statement**: When local laws hold, DPO training on summarized data produces
the same optimal policy as training on original data.

**Significance**: Concrete instantiation for the widely-used DPO method.
-/

/-- DPO equivalence: local laws imply identical training outcomes.

The gap between DPO loss on original data and DPO loss on summarized data
is exactly zero when L1, L2, L3 hold.

Corollary: argmin_{π measurable} L_DPO(π; X) = argmin_{π measurable} L_DPO(π; Z^R) -/
abbrev dpo_training_sound := @dpo_equivalence

/-- Under exact theorem-backed reduction, uniform oracle error turns exact DPO
oracle-argmins on `ZR` into `2ε`-optimal policies for the true objective. -/
abbrev dpo_true_epsilon_argmin_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement :=
  @FormalProofs.OPT.dpo_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement

/-- Pointwise value form of
`dpo_true_epsilon_argmin_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement`. -/
abbrev dpo_true_loss_le_best_plus_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement :=
  @FormalProofs.OPT.dpo_true_loss_le_best_plus_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement

/-- High-probability exact-DPO corollary: if optimizer selection lands in the
surrogate argmin set on a good event with failure probability at most `δ`, then
failure of true `2ε`-optimality is also bounded by `δ`. -/
abbrev dpo_true_epsilon_argmin_failure_prob_le_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement :=
  @FormalProofs.OPT.dpo_true_epsilon_argmin_failure_prob_le_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement

/-- Generic two-stage perturbation calculus for oracle-measurable argmins:
uniform truth-to-oracle error plus uniform oracle-to-surrogate transport error
yields `2(ε₁+ε₂)` near-optimality for surrogate minimizers. -/
abbrev oracle_measurable_argmin_two_stage_uniform_perturbation :=
  @FormalProofs.OPT.oracleMeasurableParamArgmin_subset_epsilonArgmin_of_uniform_two_stage_loss_perturbation

/-- Generic two-stage perturbation calculus with parameter-dependent oracle and
transport slack. -/
abbrev oracle_measurable_argmin_two_stage_pointwise_perturbation :=
  @FormalProofs.OPT.oracleMeasurableParamArgmin_subset_pointwiseEpsilonArgmin_of_nonuniform_two_stage_loss_perturbation

/-- Generic expected-tree surrogate perturbation with parameter-dependent slack:
small expected absolute loss gap implies near-optimality transfer from the
expected tree objective to the true objective. -/
abbrev oracle_measurable_argmin_expected_tree_pointwise_perturbation :=
  @FormalProofs.OPT.oracleMeasurableParamArgmin_subset_pointwiseEpsilonArgmin_of_expectedTree_loss_perturbation

/-- Uniform expected-tree surrogate perturbation. -/
abbrev oracle_measurable_argmin_expected_tree_uniform_perturbation :=
  @FormalProofs.OPT.oracleMeasurableParamArgmin_subset_epsilonArgmin_of_expectedTree_loss_perturbation

/-- Generic high-probability wrapper for pointwise-slack argmin transfer on a
good event. -/
abbrev oracle_measurable_argmin_pointwise_failure_prob_transfer :=
  @FormalProofs.OPT.oracleMeasurableParamArgmin_failure_prob_le_of_good_event_pointwiseTransfer

/-- Generic high-probability expected-tree wrapper: if a choice rule lands in
the argmin set of the expected tree objective on a good event, then failure of
the transported pointwise near-optimality statement is no more likely than
failure of that event. -/
abbrev oracle_measurable_argmin_expected_tree_pointwise_failure_prob_transfer :=
  @FormalProofs.OPT.oracleMeasurableParamArgmin_failure_prob_le_of_good_event_expectedTree_pointwiseTransfer

/-- Approximate-bundle DPO near-optimality: if an entire oracle-measurable
policy class shares one Lipschitz envelope and one audited approximate-local-law
bundle, DPO argmins on `ZR` are near-optimal for the true objective. -/
abbrev dpo_true_epsilon_argmin_via_ZR_of_approxBundle_and_uniformOracleMeasurement :=
  @FormalProofs.OPT.dpo_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_approxBundle_and_uniformOracleMeasurement

/-- Exact theorem-backed DPO optimizer perturbation with policy-dependent oracle
measurement error. -/
abbrev dpo_true_pointwise_epsilon_argmin_via_ZR_of_exactTheoremBacked_and_pointwiseOracleMeasurement :=
  @FormalProofs.OPT.dpo_oracle_argmin_subset_true_pointwiseEpsilonArgmin_via_ZR_of_exactTheoremBacked_and_pointwiseOracleMeasurement

/-- Approximate-bundle DPO optimizer perturbation with policy-dependent oracle
measurement error. -/
abbrev dpo_true_pointwise_epsilon_argmin_via_ZR_of_approxBundle_and_pointwiseOracleMeasurement :=
  @FormalProofs.OPT.dpo_oracle_argmin_subset_true_pointwiseEpsilonArgmin_via_ZR_of_approxBundle_and_pointwiseOracleMeasurement

/-- Audit-event specialization of
`dpo_true_epsilon_argmin_via_ZR_of_approxBundle_and_uniformOracleMeasurement`. -/
abbrev dpo_true_epsilon_argmin_via_ZR_of_nodewiseEmpiricalAudit_and_uniformOracleMeasurement :=
  @FormalProofs.OPT.dpo_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_nodewiseEmpiricalAudit_and_uniformOracleMeasurement

/-- Exact theorem-backed GRPO-PL optimizer perturbation. -/
abbrev grpo_pl_true_epsilon_argmin_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement :=
  @FormalProofs.OPT.grpo_pl_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement

/-- Approximate-bundle GRPO-PL optimizer perturbation. -/
abbrev grpo_pl_true_epsilon_argmin_via_ZR_of_approxBundle_and_uniformOracleMeasurement :=
  @FormalProofs.OPT.grpo_pl_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_approxBundle_and_uniformOracleMeasurement

/-- Exact theorem-backed GRPO-PL optimizer perturbation with policy-dependent
oracle measurement error. -/
abbrev grpo_pl_true_pointwise_epsilon_argmin_via_ZR_of_exactTheoremBacked_and_pointwiseOracleMeasurement :=
  @FormalProofs.OPT.grpo_pl_oracle_argmin_subset_true_pointwiseEpsilonArgmin_via_ZR_of_exactTheoremBacked_and_pointwiseOracleMeasurement

/-- Approximate-bundle GRPO-PL optimizer perturbation with policy-dependent
oracle measurement error. -/
abbrev grpo_pl_true_pointwise_epsilon_argmin_via_ZR_of_approxBundle_and_pointwiseOracleMeasurement :=
  @FormalProofs.OPT.grpo_pl_oracle_argmin_subset_true_pointwiseEpsilonArgmin_via_ZR_of_approxBundle_and_pointwiseOracleMeasurement

/-- High-probability exact-GRPO-PL optimizer perturbation. -/
abbrev grpo_pl_true_epsilon_argmin_failure_prob_le_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement :=
  @FormalProofs.OPT.grpo_pl_true_epsilon_argmin_failure_prob_le_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement

/-- Exact theorem-backed GRPO-RL optimizer perturbation. -/
abbrev grpo_rl_true_epsilon_argmin_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement :=
  @FormalProofs.OPT.grpo_rl_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement

/-- Approximate-bundle GRPO-RL optimizer perturbation. -/
abbrev grpo_rl_true_epsilon_argmin_via_ZR_of_approxBundle_and_uniformOracleMeasurement :=
  @FormalProofs.OPT.grpo_rl_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_approxBundle_and_uniformOracleMeasurement

/-- Exact theorem-backed GRPO-RL optimizer perturbation with policy-dependent
oracle measurement error. -/
abbrev grpo_rl_true_pointwise_epsilon_argmin_via_ZR_of_exactTheoremBacked_and_pointwiseOracleMeasurement :=
  @FormalProofs.OPT.grpo_rl_oracle_argmin_subset_true_pointwiseEpsilonArgmin_via_ZR_of_exactTheoremBacked_and_pointwiseOracleMeasurement

/-- Approximate-bundle GRPO-RL optimizer perturbation with policy-dependent
oracle measurement error. -/
abbrev grpo_rl_true_pointwise_epsilon_argmin_via_ZR_of_approxBundle_and_pointwiseOracleMeasurement :=
  @FormalProofs.OPT.grpo_rl_oracle_argmin_subset_true_pointwiseEpsilonArgmin_via_ZR_of_approxBundle_and_pointwiseOracleMeasurement

/-- High-probability exact-GRPO-RL optimizer perturbation. -/
abbrev grpo_rl_true_epsilon_argmin_failure_prob_le_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement :=
  @FormalProofs.OPT.grpo_rl_true_epsilon_argmin_failure_prob_le_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement

/-!
## Theorem 4: GRPO-RL Equivalence (DeepSeek-R1 Style)

**Statement**: The GRPO-RL objective (clipped surrogate + KL penalty) is
equivalent on original vs. summarized data when oracle-measurability holds.

**Significance**: Captures the exact objective used by DeepSeek-R1:
  J_GRPO(θ) = E[1/G Σ min(r_i·A_i, clip(r_i)·A_i) - β·D_KL(π_θ || π_ref)]
where A_i = (reward_i - mean) / std (z-score normalized advantage).
-/

/-- GRPO-RL equivalence: DeepSeek-R1 style training is sound on summaries.

The GRPO-RL loss includes:
- Group sampling (k candidates per prompt)
- Z-score normalized advantages: A_i = (r_i - mean) / std
- PPO-style clipping: min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)
- KL penalty: β · D_KL(π_θ || π_ref)

When policies, rewards, and group generators are oracle-measurable,
training on summaries equals training on originals. -/
abbrev grpo_rl_training_sound := @grpo_rl_equivalence

/-!
## Theorem 5: Listwise GRPO Equivalence (Plackett-Luce)

**Statement**: GRPO with Plackett-Luce ranking loss (k > 2 group comparisons)
is equivalent on original vs. summarized data.

**Significance**: Generalizes DPO (k=2, Bradley-Terry) to arbitrary k.
-/

/-- Plackett-Luce GRPO equivalence: listwise ranking is sound on summaries.

The Plackett-Luce model generalizes Bradley-Terry from pairs to rankings:
  P(ranking) = ∏_{i=1}^{k} exp(s_i) / Σ_{j≥i} exp(s_j)

When policy and ranker are oracle-measurable, GRPO training on
summarized data equals training on original data. -/
abbrev grpo_plackett_luce_sound := @grpo_equivalence

/-!
## Theorem 6: TreePO Oracle Utility Bounds

**Statement**: TreePO gives oracle utility gap bounds in terms of expected tree
distortion, with IPW estimators and optional label noise.

**Significance**: This is the bridge between the tree sampling model and
oracle-utility preservation, including the IPW form used for evaluation.
-/

/-- TreePO oracle utility gap bound via expected tree distortion. -/
abbrev treepo_oracle_utility_gap := @tree_oracle_utility_gap_bounded

/-- TreePO oracle utility gap bound via IPW estimator. -/
abbrev treepo_oracle_utility_gap_ipw := @tree_oracle_utility_gap_bounded_ipw

/-- TreePO oracle utility gap bound with noisy truth labels. -/
abbrev treepo_oracle_utility_gap_noisy := @tree_oracle_utility_gap_noisy_bounded

/-- TreePO oracle utility gap bound with noisy truth labels (IPW form). -/
abbrev treepo_oracle_utility_gap_noisy_ipw := @tree_oracle_utility_gap_noisy_bounded_ipw

/-!
## Theorem 6.25: Regularized Oracle-Risk Objective

These definitions make the optimization problem itself explicit: a global
oracle-risk term, a summary-budget term, and an approximate-local-law
regularizer. The distortion term still transports downstream objectives via the
existing Lipschitz theorems; the regularizer is the optimization-side control.
-/

/-- Weight bundle for the regularized oracle objective. -/
abbrev regularized_oracle_objective_weights := @FormalProofs.OPT.RegularizedObjectiveWeights

/-- Relative shares used inside the local-law part of the regularizer. -/
abbrev regularized_oracle_law_shares := @FormalProofs.OPT.LawComponentShares

/-- Uniform local-law shares (`1/3` each). -/
abbrev uniform_regularized_oracle_law_shares := @FormalProofs.OPT.uniformLawComponentShares

/-- One-parameter summary-to-law frontier for the regularized objective. -/
abbrev frontier_regularized_oracle_weights :=
  @FormalProofs.OPT.frontierRegularizedObjectiveWeights

/-- Fixed simulation-facing default weights (`0.75` on distortion, `0.25`
split across summary/law penalties). -/
abbrev simulation_default_regularized_oracle_weights :=
  @FormalProofs.OPT.simulationDefaultRegularizedObjectiveWeights

/-- The fixed simulation default is the frontier point
`regularizerWeight = 0.25`, `lawStrength = 0.5`, uniform law shares. -/
abbrev simulation_default_regularized_oracle_weights_eq_frontier :=
  @FormalProofs.OPT.simulationDefaultRegularizedObjectiveWeights_eq_frontier

/-- Expected summary-cost term under the `ZR` output distribution. -/
abbrev expected_summary_cost := @FormalProofs.OPT.expectedSummaryCost

/-- Certified local-law penalty built from an approximate/audited law bundle. -/
abbrev certified_law_penalty := @FormalProofs.OPT.certifiedLawPenalty

/-- Population oracle-risk objective: distortion plus summary cost. -/
abbrev oracle_risk_objective := @FormalProofs.OPT.oracleRiskObjective

/-- Certified regularized objective: oracle risk plus approximate-local-law penalty. -/
abbrev certified_regularized_objective := @FormalProofs.OPT.certifiedRegularizedObjective

/-- Hard-budget constraints for the constrained optimization formulation. -/
abbrev regularized_objective_constraints := @FormalProofs.OPT.RegularizedObjectiveConstraints

/-- Feasibility predicate for the constrained formulation. -/
abbrev satisfies_regularized_constraints := @FormalProofs.OPT.SatisfiesRegularizedConstraints

/-- Unconstrained minimizer predicate for the certified regularized objective. -/
abbrev certified_regularized_minimizer := @FormalProofs.OPT.IsCertifiedRegularizedMinimizer

/-- Constrained minimizer predicate for the certified regularized objective. -/
abbrev constrained_certified_regularized_minimizer :=
  @FormalProofs.OPT.IsConstrainedCertifiedRegularizedMinimizer

/-- `ε`-optimal version of the certified regularized minimizer predicate for a
true objective over summarizer/law pairs. -/
abbrev certified_regularized_epsilon_minimizer :=
  @FormalProofs.OPT.IsCertifiedRegularizedEpsilonMinimizer

/-- Constrained `ε`-optimal version of the certified regularized minimizer
predicate. -/
abbrev constrained_certified_regularized_epsilon_minimizer :=
  @FormalProofs.OPT.IsConstrainedCertifiedRegularizedEpsilonMinimizer

/-- Distortion-side term of the regularized objective bounded by an approximate
local-law bundle. -/
abbrev oracle_risk_objective_from_approx_bundle :=
  @FormalProofs.OPT.oracleRiskObjective_le_of_approx_bundle

/-- Full certified regularized objective bounded by the approximate-local-law
bundle and the summary-cost term. -/
abbrev certified_regularized_objective_from_approx_bundle :=
  @FormalProofs.OPT.certifiedRegularizedObjective_le_of_approx_bundle

/-- Uniform perturbation turns an exact certified-regularized minimizer into a
`2ε`-minimizer for the true objective. -/
abbrev certified_regularized_epsilon_minimizer_of_uniform_perturbation :=
  @FormalProofs.OPT.certifiedRegularized_epsilonMinimizer_of_uniform_perturbation

/-- Constrained version of
`certified_regularized_epsilon_minimizer_of_uniform_perturbation`. -/
abbrev constrained_certified_regularized_epsilon_minimizer_of_uniform_perturbation :=
  @FormalProofs.OPT.constrainedCertifiedRegularized_epsilonMinimizer_of_uniform_perturbation

/-- High-probability wrapper for certified-regularized minimizer transfer on a
confidence event. -/
abbrev certified_regularized_epsilon_minimizer_failure_prob_le_of_good_event :=
  @FormalProofs.OPT.certifiedRegularized_epsilonMinimizer_failure_prob_le_of_good_event

/-- High-probability constrained wrapper for certified-regularized minimizer
transfer on a confidence event. -/
abbrev constrained_certified_regularized_epsilon_minimizer_failure_prob_le_of_good_event :=
  @FormalProofs.OPT.constrainedCertifiedRegularized_epsilonMinimizer_failure_prob_le_of_good_event

/-- Non-uniform objective perturbation turns an exact certified-regularized
minimizer into a pointwise-slack minimizer for the true objective. -/
abbrev certified_regularized_pointwise_epsilon_minimizer_of_nonuniform_perturbation :=
  @FormalProofs.OPT.certifiedRegularized_pointwiseEpsilonMinimizer_of_nonuniform_perturbation

/-- Constrained pointwise-slack version of
`certified_regularized_pointwise_epsilon_minimizer_of_nonuniform_perturbation`. -/
abbrev constrained_certified_regularized_pointwise_epsilon_minimizer_of_nonuniform_perturbation :=
  @FormalProofs.OPT.constrainedCertifiedRegularized_pointwiseEpsilonMinimizer_of_nonuniform_perturbation

/-!
## Theorem 6.5: Exact Utility-Transport Suite

These are the exact-control wrappers used by the new exact utility-transport
simulations. They separate two layers:

1. any **oracle-indexed objective** transports under zero distortion;
2. any utility on an **exact mergeable latent state** is preserved exactly by
   tree reduction.
-/

/-- Generic oracle-indexed feature/state objective transport under zero distortion. -/
abbrev oracle_indexed_objective_transport :=
  @FormalProofs.OPT.featureIndexedObjective_eq_of_zero_dist

/-- Direct supervised-state learning is a special case of oracle-indexed
objective transport. -/
abbrev supervised_state_objective_transport :=
  @FormalProofs.OPT.supervisedStateExpectedLoss_eq_of_zero_dist

/-- Normalized exact-state utility: zero regret iff zero state error. -/
abbrev normalized_state_utility_zero_regret_iff_zero_error :=
  @FormalProofs.OPT.normalizedErrorUtility_zero_regret_iff_zero_error

/-- Any utility on an exact mergeable latent state is preserved by the tree exactly. -/
abbrev exact_mergeable_state_utility_on_tree :=
  @FormalProofs.OPT.mergeableStateUtility_exact_on_tree

/-- Markov exact-state utility preservation (count/endpoints lane). -/
abbrev markov_state_utility_exact_on_tree :=
  @FormalProofs.OPT.markovStateUtility_exact_on_tree

/-- Markov count-only exact-control utility reaches its optimum on the exact tree fold. -/
abbrev markov_count_only_exact_on_tree :=
  @FormalProofs.OPT.markovCountOnlyUtility_exact_on_tree

/-- Markov count-plus-endpoints exact-control utility reaches its optimum on the exact tree fold. -/
abbrev markov_count_endpoints_exact_on_tree :=
  @FormalProofs.OPT.markovCountEndpointsUtility_exact_on_tree

/-- Nonseparable complementarity state utility preservation. -/
abbrev complementarity_state_utility_exact_on_tree :=
  @FormalProofs.OPT.complementarityStateUtility_exact_on_tree

/-- Threshold complementarity downstream utility preservation. -/
abbrev complementarity_threshold_exact_on_tree :=
  @FormalProofs.OPT.complementarityThresholdUtility_exact_on_tree

/-- Topic unigram+boundary state utility preservation. -/
abbrev topic_state_utility_exact_on_tree :=
  @FormalProofs.OPT.topicSketchUtility_exact_on_tree

/-- Topic mass-only control utility preservation. -/
abbrev topic_mass_only_exact_on_tree :=
  @FormalProofs.OPT.topicMassUtility_exact_on_tree

/-- Topic-plus-boundary oracle score preservation. -/
abbrev topic_plus_boundary_exact_on_tree :=
  @FormalProofs.OPT.topicOracleFromSketch_exact_on_tree

/-!
## Theorem 9: Adversarial Chunking Failure Control (Non-Uniform WOR)

This is a concrete self-normalized failure-event composition theorem for
adversarial chunking regimes, built on top of the explicit non-uniform
without-replacement model.
-/

/-- Adversarial chunking: mean-scale self-normalized failure-event bound. -/
abbrev adversarial_chunking_failure_bound :=
  @FormalProofs.OPT.AdversarialChunkingInstance.failure_bound

/-- Conditional Hoeffding bridge (bounded + conditional mean-zero =>
conditional sub-Gaussian). -/
abbrev conditional_hoeffding_bridge :=
  @FormalProofs.OPT.hasCondSubgaussianMGF_of_mem_Icc_of_condExp_eq_zero

/-- Azuma-Hoeffding from bounded increments + conditional mean-zero. -/
abbrev azuma_from_conditional_hoeffding :=
  @FormalProofs.OPT.azuma_hoeffding_of_mem_Icc_of_condExp_eq_zero

/-- Two-sided Azuma-Hoeffding wrapper for bounded centered increments. -/
abbrev azuma_abs_from_conditional_hoeffding :=
  @FormalProofs.OPT.azuma_hoeffding_abs_of_mem_Icc_of_condExp_eq_zero

/-- Random-permutation without-replacement Azuma specialization. -/
abbrev azuma_abs_random_permutation_wor :=
  @FormalProofs.OPT.azuma_hoeffding_abs_of_random_permutation

/-!
## Theorem 7: OPS ↔ Mergeable Summary Bridge
-/

/-- OPS global assumptions imply merge-closure in the mergeable-summary interface. -/
abbrev ops_mergeable_mergeClosed := @ops_mergeClosed_of_global

/-- OPS global assumptions imply hierarchical mergeability over arbitrary merge trees. -/
abbrev ops_mergeable_hierarchical := @ops_hierarchical_mergeable_of_global

/-- Reduction of OPS global assumptions to the classical mergeable-summary statement. -/
abbrev ops_mergeable_classical := @ops_reduction_to_classical_mergeable

/-!
## Theorem 7.5: Sketch→Summary Local-Law Bridge

This packages the reusable bridge for learned/neural sketch operators:
under leaf preservation, merge compatibility, and decode/summary compatibility,
the induced deterministic summary operator satisfies `L1`, `L2`, `L3`.
-/

/-- Sketch-level assumptions imply a full local-law bundle for the induced
deterministic summary operator. -/
abbrev local_laws_bundle_from_sketch := @local_laws_bundle_of_sketch

/-- Sketch-level assumptions imply zero multi-round distortion for the induced
deterministic summary operator (typeclass-bound form). -/
abbrev multi_round_preservation_from_sketch := @multi_round_typeclass_of_sketch

/-- DPO equivalence recovered from generic sketch assumptions. -/
abbrev dpo_equivalence_from_sketch := @dpo_equivalence_of_sketch

/-- GRPO-PL equivalence via ZR recovered from generic sketch assumptions. -/
abbrev grpo_equivalence_from_sketch := @grpo_equivalence_via_ZR_of_sketch

/-- GRPO-RL equivalence via ZR recovered from generic sketch assumptions. -/
abbrev grpo_rl_equivalence_from_sketch := @grpo_rl_equivalence_via_ZR_of_sketch

/-- One-line local-law template for identity sketch + encoded feature oracle. -/
abbrev local_laws_from_identity_encoded_feature :=
  @local_laws_of_identity_encoded_feature

/-- One-line pairwise-equivalence template for identity sketch + encoded feature oracle. -/
abbrev pairwise_equivalence_from_identity_encoded_feature :=
  @preference_learning_equivalence_via_ZR_of_identity_encoded_feature

/-- One-line local-law template for the paired non-identity sketch + encoded feature oracle. -/
abbrev local_laws_from_paired_encoded_feature :=
  @local_laws_of_paired_encoded_feature

/-- One-line pairwise-equivalence template for the paired non-identity sketch +
encoded feature oracle. -/
abbrev pairwise_equivalence_from_paired_encoded_feature :=
  @preference_learning_equivalence_via_ZR_of_paired_encoded_feature

/-- Approximate leaf-preservation sketch assumption (nodewise budgeted). -/
abbrev approx_sketch_leaf_assumption :=
  @SketchLeafApproxPreserving

/-- Approximate merge-compatibility sketch assumption (nodewise budgeted). -/
abbrev approx_sketch_merge_assumption :=
  @SketchMergeApproxCompatible

/-- Approximate sketch assumptions imply nodewise approximate local laws. -/
abbrev approx_nodewise_laws_from_sketch :=
  @approx_nodewise_local_laws_of_sketch

/-- Approximate sketch assumptions plus idempotence budget imply an approximate
local-law bundle. -/
abbrev approx_bundle_from_sketch :=
  @approx_bundle_of_sketch

/-- Concrete Markov encoded-feature instantiation: sketch assumptions recover
local laws automatically for Markov sketch states. -/
abbrev markov_local_laws_from_encoded_feature :=
  @FormalProofs.OPT.markov_local_laws_of_encoded_feature

/-- DGP-level Markov path support: exact path encoder is a congruent feature. -/
abbrev markov_path_local_laws_of_encoded_state :=
  @FormalProofs.OPT.MarkovPath.local_laws_of_encoded_state

/-- DGP-level Markov path exact-state utility preservation. -/
abbrev markov_path_state_exact_on_tree :=
  @FormalProofs.OPT.MarkovPath.state_exact_on_tree

/-- DGP-level Markov path exact changepoint-count preservation. -/
abbrev markov_path_count_exact_on_tree :=
  @FormalProofs.OPT.MarkovPath.count_exact_on_tree

/-- DGP-level counterexample: count-only summaries are not compositionally sufficient. -/
abbrev markov_countOnly_mergeFold_counterexample :=
  @FormalProofs.OPT.MarkovPath.countOnly_mergeFold_counterexample

/-- Concrete topic encoded-feature instantiation: sketch assumptions recover
local laws automatically for topic unigram+bigram features. -/
abbrev topic_local_laws_from_encoded_feature :=
  @FormalProofs.OPT.topic_local_laws_of_encoded_feature

/-- Concrete length-feature instantiation on token lists (identity sketch). -/
abbrev length_local_laws_from_encoded_feature :=
  @FormalProofs.OPT.length_local_laws_of_encoded_feature

/-- Concrete length-feature instantiation on token lists (paired non-identity sketch). -/
abbrev length_local_laws_from_paired_encoded_feature :=
  @FormalProofs.OPT.length_local_laws_of_paired_encoded_feature

/-- Concrete length-feature instantiation on token lists (genuinely lossy length sketch). -/
abbrev length_local_laws_from_lossy_encoded_feature :=
  @FormalProofs.OPT.length_local_laws_of_lossy_encoded_feature

/-- Pairwise preference equivalence via ZR for encoded length feature. -/
abbrev length_pairwise_equivalence_from_encoded_feature :=
  @FormalProofs.OPT.length_preference_equivalence_via_ZR_of_encoded_feature

/-- Pairwise preference equivalence via ZR for encoded length feature under the
genuinely lossy length sketch. -/
abbrev length_pairwise_equivalence_from_lossy_encoded_feature :=
  @FormalProofs.OPT.length_preference_equivalence_via_ZR_of_lossy_encoded_feature

/-- End-to-end DPO gap bound for lossy length sketch under stochastic adaptive
approximate laws (support-tree view). -/
abbrev length_dpo_gap_from_lossy_stochastic_adaptive_approx :=
  @FormalProofs.OPT.length_dpo_gap_of_stochastic_adaptive_approx

/-- End-to-end DPO gap bound for lossy length sketch with an added
oracle-measurement term. -/
abbrev length_dpo_gap_from_lossy_stochastic_adaptive_approx_with_oracleMeasurement :=
  @FormalProofs.OPT.length_dpo_gap_of_stochastic_adaptive_approx_with_oracleMeasurement

/-- End-to-end GRPO-PL gap bound for lossy length sketch under stochastic
adaptive approximate laws (support-tree view). -/
abbrev length_grpo_pl_gap_from_lossy_stochastic_adaptive_approx :=
  @FormalProofs.OPT.length_grpo_pl_gap_of_stochastic_adaptive_approx

/-- End-to-end GRPO-PL gap bound for lossy length sketch with an added
oracle-measurement term. -/
abbrev length_grpo_pl_gap_from_lossy_stochastic_adaptive_approx_with_oracleMeasurement :=
  @FormalProofs.OPT.length_grpo_pl_gap_of_stochastic_adaptive_approx_with_oracleMeasurement

/-- End-to-end GRPO-RL gap bound for lossy length sketch under stochastic
adaptive approximate laws (support-tree view). -/
abbrev length_grpo_rl_gap_from_lossy_stochastic_adaptive_approx :=
  @FormalProofs.OPT.length_grpo_rl_gap_of_stochastic_adaptive_approx

/-- End-to-end GRPO-RL gap bound for lossy length sketch with an added
oracle-measurement term. -/
abbrev length_grpo_rl_gap_from_lossy_stochastic_adaptive_approx_with_oracleMeasurement :=
  @FormalProofs.OPT.length_grpo_rl_gap_of_stochastic_adaptive_approx_with_oracleMeasurement

/-- The lossy length sketch encoder is non-injective on nontrivial alphabets. -/
abbrev length_sketch_encoder_noninjective :=
  @lengthSketch_encode_not_injective

/-!
## Theorem 7.6: Approximate Local-Law Recovery
-/

/-- Nodewise approximate leaf law (`ε` per leaf). -/
abbrev approx_leaf_law_nodewise := @FormalProofs.OPT.L1εNode

/-- Nodewise approximate merge law (`ε` per internal merge). -/
abbrev approx_merge_law_nodewise := @FormalProofs.OPT.L2εNode

/-- Approximate leaf budget law (`ε`-leaf). -/
abbrev approx_leaf_law := @FormalProofs.OPT.L1ε

/-- Approximate merge budget law (`ε`-merge). -/
abbrev approx_merge_law := @FormalProofs.OPT.L2ε

/-- Approximate idempotence budget law (`ε`-idempotence). -/
abbrev approx_idemp_law := @FormalProofs.OPT.L3ε

/-- Nodewise laws imply an aggregate approximate-local-law bundle. -/
abbrev approx_bundle_from_nodewise := @FormalProofs.OPT.approx_bundle_of_nodewise

/-- Audited aggregate upper bounds transfer directly to an approximate local-law
bundle. -/
abbrev approx_bundle_from_audited_upper_bounds :=
  @FormalProofs.OPT.approx_bundle_of_audited_upper_bounds

/-- Confidence-event form: if the audit event holds, we recover an approximate
local-law bundle. -/
abbrev approx_bundle_from_audited_confidence_event :=
  @FormalProofs.OPT.approx_bundle_of_audited_confidence_event

/-- Nodewise empirical audit certificate with concentration margins (Hoeffding /
Serfling style). -/
abbrev nodewise_empirical_audit_certificate :=
  @FormalProofs.OPT.NodewiseEmpiricalAuditCertificate

/-- Nodewise empirical certificate lifts to aggregate audited upper bounds. -/
abbrev audited_upper_bounds_from_nodewise_empirical_certificate :=
  @FormalProofs.OPT.audited_upper_bounds_of_nodewise_empirical_certificate

/-- Nodewise empirical certificate lifts directly to an approximate local-law
bundle. -/
abbrev approx_bundle_from_nodewise_empirical_certificate :=
  @FormalProofs.OPT.approx_bundle_of_nodewise_empirical_certificate

/-- Confidence-event wrapper for nodewise empirical audit certificates. -/
abbrev nodewise_empirical_audit_with_confidence :=
  @FormalProofs.OPT.NodewiseEmpiricalAuditWithConfidence

/-- Under empirical audit confidence event, recover an approximate local-law
bundle. -/
abbrev approx_bundle_from_nodewise_empirical_confidence_event :=
  @FormalProofs.OPT.approx_bundle_of_nodewise_empirical_confidence_event

/-- Approximate local laws imply a quantitative `Δ_R_ZR` bound. -/
abbrev delta_r_zr_from_approx_local_laws := @FormalProofs.OPT.Δ_R_ZR_le_of_approx_local_laws

/-- Bundle-driven quantitative `Δ_R_ZR` bound under approximate local laws. -/
abbrev delta_r_zr_from_approx_bundle := @FormalProofs.OPT.Δ_R_ZR_le_of_approx_bundle

/-- DPO gap bound recovered from approximate local laws. -/
abbrev dpo_gap_from_approx_local_laws := @FormalProofs.OPT.dpo_gap_via_approx_local_laws

/-- Bundle-driven DPO gap bound under approximate local laws. -/
abbrev dpo_gap_from_approx_bundle := @FormalProofs.OPT.dpo_gap_via_approx_bundle

/-- Confidence-event lift from audited approximate bundle to DPO gap control. -/
abbrev dpo_gap_from_audited_confidence_event :=
  @FormalProofs.OPT.dpo_gap_via_audited_confidence_event

/-- GRPO-PL gap bound recovered from approximate local laws. -/
abbrev grpo_gap_from_approx_local_laws := @FormalProofs.OPT.grpo_pl_gap_via_approx_local_laws

/-- Bundle-driven GRPO-PL gap bound under approximate local laws. -/
abbrev grpo_gap_from_approx_bundle := @FormalProofs.OPT.grpo_pl_gap_via_approx_bundle

/-- Confidence-event lift from audited approximate bundle to GRPO-PL gap control. -/
abbrev grpo_gap_from_audited_confidence_event :=
  @FormalProofs.OPT.grpo_pl_gap_via_audited_confidence_event

/-- GRPO-RL gap bound recovered from approximate local laws. -/
abbrev grpo_rl_gap_from_approx_local_laws := @FormalProofs.OPT.grpo_rl_gap_via_approx_local_laws

/-- Bundle-driven GRPO-RL gap bound under approximate local laws. -/
abbrev grpo_rl_gap_from_approx_bundle := @FormalProofs.OPT.grpo_rl_gap_via_approx_bundle

/-- Confidence-event lift from audited approximate bundle to GRPO-RL gap control. -/
abbrev grpo_rl_gap_from_audited_confidence_event :=
  @FormalProofs.OPT.grpo_rl_gap_via_audited_confidence_event

/-!
## Theorem 7.7: Adaptive Chunking Bridge
-/

/-- Multi-round zero-distortion recovered along adaptive tree policies. -/
abbrev multi_round_from_adaptive_chunking := @FormalProofs.OPT.multi_round_typeclass_of_adaptive

/-- DPO equivalence recovered along adaptive tree policies. -/
abbrev dpo_equivalence_from_adaptive_chunking := @FormalProofs.OPT.dpo_equivalence_of_adaptive

/-- Adaptive approximate laws imply per-document `Δ_R_ZR` bounds. -/
abbrev delta_r_zr_from_adaptive_approx := @FormalProofs.OPT.Δ_R_ZR_le_of_adaptive_approx_local_laws

/-- Bundle-driven adaptive approximate bound with cleaner interface. -/
abbrev delta_r_zr_from_adaptive_approx_bundle := @FormalProofs.OPT.Δ_R_ZR_le_of_adaptive_approx_bundle

/-- Multi-round zero-distortion recovered for any support tree of a stochastic
adaptive policy. -/
abbrev multi_round_from_stochastic_adaptive_chunking :=
  @FormalProofs.OPT.multi_round_typeclass_of_stochastic_adaptive

/-- DPO equivalence recovered for any support tree of a stochastic adaptive policy. -/
abbrev dpo_equivalence_from_stochastic_adaptive_chunking :=
  @FormalProofs.OPT.dpo_equivalence_of_stochastic_adaptive

/-- Stochastic adaptive approximate laws imply per-support-tree `Δ_R_ZR` bounds. -/
abbrev delta_r_zr_from_stochastic_adaptive_approx :=
  @FormalProofs.OPT.Δ_R_ZR_le_of_stochastic_adaptive_approx_local_laws

/-- Exact stochastic adaptive local laws imply zero expected `Δ_R_ZR` over the
tree-policy distribution. -/
abbrev expected_delta_r_zr_zero_from_stochastic_adaptive :=
  @FormalProofs.OPT.Exp_Δ_R_ZR_eq_zero_of_stochastic_adaptive_local_laws

/-- Approximate stochastic adaptive local laws imply an expected `Δ_R_ZR` budget
bound over the tree-policy distribution. -/
abbrev expected_delta_r_zr_from_stochastic_adaptive_approx :=
  @FormalProofs.OPT.Exp_Δ_R_ZR_le_of_stochastic_adaptive_approx_local_laws

/-- Bounded-wrapper expected `Δ_R_ZR` theorem for stochastic adaptive approximate
laws (summability discharged by uniform bounds). -/
abbrev expected_delta_r_zr_from_stochastic_adaptive_approx_bounded :=
  @FormalProofs.OPT.Exp_Δ_R_ZR_le_of_stochastic_adaptive_approx_local_laws_bounded

/-- Expected DPO gap bound over stochastic adaptive tree policies under approximate
local laws. -/
abbrev expected_dpo_gap_from_stochastic_adaptive_approx :=
  @FormalProofs.OPT.Exp_dpo_gap_le_of_stochastic_adaptive_approx_local_laws

/-- Bounded-wrapper expected DPO gap theorem over stochastic adaptive tree
policies. -/
abbrev expected_dpo_gap_from_stochastic_adaptive_approx_bounded :=
  @FormalProofs.OPT.Exp_dpo_gap_le_of_stochastic_adaptive_approx_local_laws_bounded

/-- Generic stochastic-adaptive expected-gap lift with oracle measurement. -/
abbrev expected_gap_from_stochastic_adaptive_with_oracleMeasurement :=
  @FormalProofs.OPT.Exp_loss_gap_le_of_stochastic_adaptive_oracleMeasurement

/-- Generic stochastic-adaptive expected-gap lift with tree-indexed oracle
measurement uncertainty. -/
abbrev expected_gap_from_stochastic_adaptive_with_pointwiseOracleMeasurement :=
  @FormalProofs.OPT.Exp_loss_gap_le_of_stochastic_adaptive_pointwiseOracleMeasurement

/-- Expected DPO gap bound over stochastic adaptive tree policies with an
additional oracle-measurement term. -/
abbrev expected_dpo_gap_from_stochastic_adaptive_approx_with_oracleMeasurement :=
  @FormalProofs.OPT.Exp_dpo_gap_le_of_stochastic_adaptive_approx_local_laws_with_oracleMeasurement

/-- Expected DPO gap bound over stochastic adaptive tree policies with a
tree-indexed oracle-measurement envelope. -/
abbrev expected_dpo_gap_from_stochastic_adaptive_approx_with_pointwiseOracleMeasurement :=
  @FormalProofs.OPT.Exp_dpo_gap_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement

/-- Tree-level DPO optimizer transfer over the expected stochastic-adaptive
tree objective with tree-indexed oracle uncertainty. -/
abbrev expected_tree_dpo_pointwise_epsilon_argmin_from_stochastic_adaptive_approx :=
  @FormalProofs.OPT.dpo_expected_tree_argmin_subset_true_pointwiseEpsilonArgmin_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement

/-- High-probability tree-level DPO optimizer transfer over the expected
stochastic-adaptive tree objective. -/
abbrev expected_tree_dpo_pointwise_epsilon_argmin_failure_prob_from_stochastic_adaptive_approx :=
  @FormalProofs.OPT.dpo_expected_tree_true_pointwiseEpsilonArgmin_failure_prob_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement

/-- Bounded-wrapper expected DPO gap theorem over stochastic adaptive tree
policies with oracle measurement. -/
abbrev expected_dpo_gap_from_stochastic_adaptive_approx_bounded_with_oracleMeasurement :=
  @FormalProofs.OPT.Exp_dpo_gap_le_of_stochastic_adaptive_approx_local_laws_bounded_with_oracleMeasurement

/-- Expected GRPO-PL gap bound over stochastic adaptive tree policies under
approximate local laws. -/
abbrev expected_grpo_gap_from_stochastic_adaptive_approx :=
  @FormalProofs.OPT.Exp_grpo_pl_gap_le_of_stochastic_adaptive_approx_local_laws

/-- Expected GRPO-PL gap bound over stochastic adaptive tree policies with an
additional oracle-measurement term. -/
abbrev expected_grpo_gap_from_stochastic_adaptive_approx_with_oracleMeasurement :=
  @FormalProofs.OPT.Exp_grpo_pl_gap_le_of_stochastic_adaptive_approx_local_laws_with_oracleMeasurement

/-- Expected GRPO-PL gap bound over stochastic adaptive tree policies with a
tree-indexed oracle-measurement envelope. -/
abbrev expected_grpo_gap_from_stochastic_adaptive_approx_with_pointwiseOracleMeasurement :=
  @FormalProofs.OPT.Exp_grpo_pl_gap_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement

/-- Tree-level GRPO-PL optimizer transfer over the expected stochastic-adaptive
tree objective with tree-indexed oracle uncertainty. -/
abbrev expected_tree_grpo_pointwise_epsilon_argmin_from_stochastic_adaptive_approx :=
  @FormalProofs.OPT.grpo_pl_expected_tree_argmin_subset_true_pointwiseEpsilonArgmin_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement

/-- High-probability tree-level GRPO-PL optimizer transfer over the expected
stochastic-adaptive tree objective. -/
abbrev expected_tree_grpo_pointwise_epsilon_argmin_failure_prob_from_stochastic_adaptive_approx :=
  @FormalProofs.OPT.grpo_pl_expected_tree_true_pointwiseEpsilonArgmin_failure_prob_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement

/-- Expected GRPO-RL gap bound over stochastic adaptive tree policies under
approximate local laws. -/
abbrev expected_grpo_rl_gap_from_stochastic_adaptive_approx :=
  @FormalProofs.OPT.Exp_grpo_rl_gap_le_of_stochastic_adaptive_approx_local_laws

/-- Expected GRPO-RL gap bound over stochastic adaptive tree policies with an
additional oracle-measurement term. -/
abbrev expected_grpo_rl_gap_from_stochastic_adaptive_approx_with_oracleMeasurement :=
  @FormalProofs.OPT.Exp_grpo_rl_gap_le_of_stochastic_adaptive_approx_local_laws_with_oracleMeasurement

/-- Expected GRPO-RL gap bound over stochastic adaptive tree policies with a
tree-indexed oracle-measurement envelope. -/
abbrev expected_grpo_rl_gap_from_stochastic_adaptive_approx_with_pointwiseOracleMeasurement :=
  @FormalProofs.OPT.Exp_grpo_rl_gap_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement

/-- Tree-level GRPO-RL optimizer transfer over the expected stochastic-adaptive
tree objective with tree-indexed oracle uncertainty. -/
abbrev expected_tree_grpo_rl_pointwise_epsilon_argmin_from_stochastic_adaptive_approx :=
  @FormalProofs.OPT.grpo_rl_expected_tree_argmin_subset_true_pointwiseEpsilonArgmin_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement

/-- High-probability tree-level GRPO-RL optimizer transfer over the expected
stochastic-adaptive tree objective. -/
abbrev expected_tree_grpo_rl_pointwise_epsilon_argmin_failure_prob_from_stochastic_adaptive_approx :=
  @FormalProofs.OPT.grpo_rl_expected_tree_true_pointwiseEpsilonArgmin_failure_prob_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement

/-!
## Theorem 7.8: RUM Assumption Discharge (Sufficient Conditions)
-/

/-- Pointwise group-loss Lipschitz implies expected-group Lipschitz (RUM assumption). -/
abbrev expected_group_lipschitz_from_pointwise :=
  @FormalProofs.OPT.expected_group_loss_lipschitz_of_pointwise

/-- Finite-index interface: pointwise group-loss Lipschitz implies expected-group
Lipschitz without manual summability obligations. -/
abbrev expected_group_lipschitz_from_pointwise_finite :=
  @FormalProofs.OPT.expected_group_loss_lipschitz_of_pointwise_finite

/-- Bundle-driven GRPO-PL quantitative gap interface. -/
abbrev grpo_pl_gap_bundle_interface :=
  @grpo_pl_gap_bundle

/-- Bundle-driven GRPO-RL quantitative gap interface. -/
abbrev grpo_rl_gap_bundle_interface :=
  @grpo_rl_gap_bundle

/-!
## Theorem 11: Surjective Global↔Local Collapse (Master)

This is a citation-ready packaging of the strongest deterministic bridge under
surjectivity: global assumptions `(A1 ∧ A2)` collapse to local laws
`(L3 ∧ L2-on-all-trees)` when `A3` is available.
-/

/-- Master surjective collapse theorem (deterministic, with A3):
`(A1 ∧ A2) ↔ (L3 ∧ L2 on all trees)`. -/
abbrev surjective_global_local_master :=
  @A1_A2_iff_L3_and_L2_on_all_trees_of_A3_surjective

/-- Two-leaf test-family variant of `surjective_global_local_master`. -/
abbrev surjective_global_local_master_two_leaf :=
  @A1_A2_iff_L3_and_L2_on_two_leaf_trees_of_A3_surjective

/-!
## Theorem 10: Sketch-Flip-Merge Bridge (Dual-Target Collapse)

If the same deterministic merge route is required to preserve both the base
target and a second target, then the two targets must already be
oracle-equivalent. This is the C-TreePO-side analogue of why Corollary 4.11
adds a genuinely stronger condition.
-/

/-- One-route/two-target collapse: both targets must coincide at oracle level. -/
abbrev one_route_two_targets_collapse := @same_route_two_targets_force_oracle_equiv

/-- Contrapositive: an oracle-distinguishable second target is incompatible with
the same-route requirement. -/
abbrev one_route_second_target_impossible := @no_two_distinguished_targets_on_one_route

/-!
## Theorem 8: Unbounded Oracle Utility (Summability-Only)

These versions remove boundedness assumptions by requiring summability of the
utility and distortion series instead.
-/

abbrev utility_bound_ZR_summable_unbounded :=
  @expected_utility_bound_ZR_summable_unbounded

abbrev utility_bound_with_noise_ZR_summable_unbounded :=
  @expected_utility_bound_with_noise_ZR_summable_unbounded

/-!
## Corollary: Gap Composition

The distillation/two-stage results in TrainingPipeline.lean are corollaries
that compose the above theorems via triangle inequality.

These are "trivial" in the sense that they follow from standard analysis once
the deep theorems above are established.
-/

/-- Two-stage gap bound: gaps compose additively.

For a two-stage pipeline (Oracle → Teacher → Student):
  |L_S(orig) - L_L(orig)| ≤ 2·ε_stage1 + ε_stage2

When local laws hold exactly, ε_stage1 = 0, giving pure distillation gap. -/
abbrev gap_composition := @training_path_gap_bound

/-- Abstract two-stage gap composition with optional oracle measurement on the
final objective. -/
abbrev gap_composition_abstract_with_oracleMeasurement :=
  @training_path_gap_bound_abstract_with_oracleMeasurement

/-- DPO two-stage gap composition with oracle measurement on the final
objective. -/
abbrev gap_composition_with_oracleMeasurement := @training_path_gap_bound_with_oracleMeasurement

/-- Bundle-driven DPO two-stage gap composition with oracle measurement. -/
abbrev gap_composition_bundle_with_oracleMeasurement :=
  @training_path_bundle_gap_with_oracleMeasurement

/-- If the teacher is oracle-optimal for the true objective, the noisy two-stage
training-path bound yields an epsilon-optimal student. -/
abbrev training_path_epsilon_optimal_with_oracleMeasurement :=
  @_root_.training_path_epsilon_optimal_with_oracleMeasurement

/-- Bundle-driven version of
`training_path_epsilon_optimal_with_oracleMeasurement`. -/
abbrev training_path_bundle_epsilon_optimal_with_oracleMeasurement :=
  @_root_.training_path_bundle_epsilon_optimal_with_oracleMeasurement

/-- High-probability wrapper for `EpsilonOptimalForOracle` on a confidence event. -/
abbrev epsilon_optimal_failure_prob_le_of_good_event :=
  @_root_.epsilonOptimal_failure_prob_le_of_good_event

/-- High-probability training-path wrapper: if the noisy two-stage certificate
holds on a good event with failure probability at most `δ`, then failure of the
student's `ε(ω)`-optimality statement is also bounded by `δ`. -/
abbrev training_path_epsilon_optimal_failure_prob_le_with_oracleMeasurement :=
  @_root_.training_path_epsilon_optimal_failure_prob_le_with_oracleMeasurement

/-- GRPO two-stage gap composition with oracle measurement on the final
objective. -/
abbrev grpo_gap_composition_with_oracleMeasurement :=
  @grpo_training_path_gap_bound_with_oracleMeasurement

/-!
## Unified Framework: The Common Mathematical Core

All preference learning gap bounds (DPO, GRPO-PL, GRPO-RL, and future methods)
follow from a **single unified template**:

```
Gap ≤ Lipschitz_Constant × Expected_Distortion
```

**Theorem (Unified Preference Gap):** For any expected loss E[L] over a
distribution μ where the inner expectation E_gen is L-Lipschitz in oracle distance,

  |E_X[E_gen] - E_Z[E_gen]| ≤ L × Δ_R

where Δ_R = E_{x,z}[dist(f*(x), f*(z))] is the expected distortion.

**Proof Structure (Method-Agnostic):**
1. `coupling_expansion_bounded` rewrites E_X - E_Z as double sum over product measure
2. Pointwise Lipschitz bound controls each term: |E_gen(x) - E_gen(z)| ≤ L⋅dist(...)
3. `coupling_bound_ineq_bounded` + Fubini gives the final bound

**Method-Specific Instantiations:**

| Method | Lipschitz Constant | E_gen Structure |
|--------|-------------------|-----------------|
| DPO | L = 2\|β\|L_pol | Expected -log σ over pairs |
| GRPO-PL | L = L_grpo | Expected Plackett-Luce over k-groups |
| GRPO-RL | L = L_grpo_rl | Expected clipped advantage + KL over groups |

The unified theorem captures the shared mathematical structure, while the
specific instantiations demonstrate how different loss functions plug into
the framework.
-/

/-- Unified preference gap theorem.

This is the mathematical core shared by all preference learning methods.
Any expected loss with a Lipschitz inner expectation satisfies the standard
gap bound: Gap ≤ Lipschitz × Distortion.

Instantiations:
- DPO: L = 2|β|L_pol
- GRPO-PL: L = L_grpo
- GRPO-RL: L = L_grpo_rl -/
abbrev unified_gap := @unified_preference_gap_bounded

end MainTheorems

end
