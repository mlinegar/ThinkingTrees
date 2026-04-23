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
import FormalProofs.OPT.NeuralOperatorTheoremBridge
import FormalProofs.OPT.NeuralOperatorPreferenceBridge
import FormalProofs.ML.NeuralOperatorArchitecture
import FormalProofs.ML.TransformerAsNeuralOperator
import FormalProofs.OPT.NeuralOperatorSpaces
import FormalProofs.OPT.RegularizedObjective
import FormalProofs.OPT.ClassicalSketchLocalLaws
import FormalProofs.OPT.OptimizationPerturbation
import FormalProofs.OPT.AdaptiveChunkingBridge
import FormalProofs.OPT.RUMSufficientConditions
import FormalProofs.OPT.GlobalAssumptions
import FormalProofs.DSL.TreeIPW
import FormalProofs.DSL.TreePOEndToEnd
import FormalProofs.DSL.MergeableCertificates
import FormalProofs.OPT.OracleUtility
import FormalProofs.OPT.ExactUtilityTransport
import FormalProofs.OPT.NodeIndexedLatentState
import FormalProofs.OPT.ExactUtilityTransportInstances
import FormalProofs.OPT.ApproxOracleRecovery
import FormalProofs.OPT.LipschitzReadoutFactorization
import FormalProofs.OPT.OracleFiberRelations
import FormalProofs.OPT.FeatureFiberLaws
import FormalProofs.OPT.FiberPreservingObjective
import FormalProofs.OPT.FeatureClassObjectives
import FormalProofs.OPT.LabelScoreObjectives
import FormalProofs.OPT.TwoStageOracleSurrogate
import FormalProofs.OPT.TwoStageDecomposition
import FormalProofs.OPT.UnifiedOracleRoute
import FormalProofs.OPT.TwoStageLabelScoreObjectives
import FormalProofs.OPT.ProductScoreFiber
import FormalProofs.OPT.ReadoutAlignment
import FormalProofs.OPT.SharedFeatureMultihead
import FormalProofs.OPT.FixedBinaryTreeDiffusion
import FormalProofs.OPT.CoverageNormalizedObjective
import FormalProofs.OPT.DiscountedTreeMetaObjective
import FormalProofs.OPT.DiscountedIPWObjective
import FormalProofs.OPT.AdversarialChunkingExample
import FormalProofs.OPT.MarkovPathDGP
import FormalProofs.OPT.MarkovSimulationValidation
import FormalProofs.OPT.SerflingAudit
import FormalProofs.OPT.NamespaceCompat
import FormalProofs.OPT.InformationSufficiency
import FormalProofs.OPT.OracleEntropy
import FormalProofs.OPT.OracleSufficientCompression
import FormalProofs.OPT.MarkovSufficiency
import FormalProofs.OPT.MarkovCarrierProjection
import FormalProofs.OPT.MarkovObservedTokenRecoverability
import FormalProofs.OPT.MarkovRepresentationLearnability
import FormalProofs.OPT.MarkovMergeSupervision

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
## Coverage-Normalized Objective Exports

These aliases expose the coverage-normalization results used to fix
budgeted-document supervision in the Markov tree trainer.
-/

/-- The current batch objective contains a hidden coverage-rate multiplier on
the root/document term. -/
abbrev coverage_normalized_current_objective_identity :=
  @FormalProofs.OPT.currentCoverageScaledTreeObjective_eq_coverageRate_mul_selectedRootMean

/-- The corrected batch objective removes the hidden coverage multiplier and
keeps the root/document term at the supervised-subset mean. -/
abbrev coverage_normalized_corrected_objective_identity :=
  @FormalProofs.OPT.correctedCoverageNormalizedTreeObjective_eq_rootWeight_mul_selectedRootMean

/-- At full coverage, the corrected objective coincides exactly with the
full-supervision objective. -/
abbrev coverage_normalized_full_coverage_equiv :=
  @FormalProofs.OPT.correctedCoverageNormalizedTreeObjective_eq_fullSupervision_at_fullCoverage

/-- Under constant inclusion probability equal to the realized coverage rate,
the HT document-mean estimator equals the selected-subset mean. -/
abbrev coverage_normalized_ht_eq_selected_mean :=
  @FormalProofs.OPT.constantInclusionHTRootMean_eq_selectedDocumentMean

/-- The HT-corrected document/root mean is unbiased for the full population
document mean under constant inclusion probability. -/
abbrev coverage_normalized_expected_root_unbiased :=
  @FormalProofs.OPT.finiteExpectation_constantInclusionHTRootMean_eq_documentMean

/-- The corrected expected objective matches the full-supervision objective. -/
abbrev coverage_normalized_expected_objective_equiv :=
  @FormalProofs.OPT.finiteExpectation_correctedCoverageNormalizedTreeObjective_eq_fullSupervision

/-- The corrected expected objective has the same parameter argmin set as the
full-supervision objective. -/
abbrev coverage_normalized_expected_same_param_argmin :=
  @FormalProofs.OPT.coverageNormalized_expectedObjective_same_paramArgmin

/-!
## Discounted-IPW Objective Exports

These aliases show that RL-style depth discounting composes cleanly with the
repo's HT/IPW supervision logic and current tree weighting scheme.
-/

/-- Generic weighted HT/IPW objective equals the corresponding full population
objective under constant marginal inclusion probabilities. -/
abbrev ipw_weighted_objective_equiv :=
  @FormalProofs.OPT.expectedIPWWeightedDocumentObjective_eq_fullWeightedDocumentObjective

/-- RL-style discounted HT/IPW objective equals the corresponding full
discounted objective under constant marginal inclusion probabilities. -/
abbrev ipw_discounted_objective_equiv :=
  @FormalProofs.OPT.expectedIPWDiscountedDocumentObjective_eq_fullDiscountedDocumentObjective

/-- Therefore the discounted HT/IPW objective has the same parameter argmin set
as the full discounted objective. -/
abbrev ipw_discounted_objective_same_param_argmin :=
  @FormalProofs.OPT.ipwDiscountedObjective_same_paramArgmin

/-- The current root / C1 / C2 / C3 weighting scheme is exactly the generic
weighted-document objective instantiated at the four supervision channels. -/
abbrev current_tree_weighting_scheme_as_weighted_objective :=
  @FormalProofs.OPT.fullWeightedDocumentObjective_eq_fullSupervisionTreeObjectiveFn

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

/-- Named alias used for deterministic tree-policy extensions:
instantiate `multi_round_proper` at a document-indexed tree map `x ↦ T(x)`. -/
abbrev fixed_partition_extension_instantiation := @multi_round_proper

/-- Coupling-form distortion equals document-level `Δ_R_ZR` when `μ_X = pure(x)`. -/
abbrev coupling_delta_eq_delta_r_zr := @coupling_Δ_eq_Δ_R_ZR

/-- Under local laws, the document-level distortion `Δ_R_ZR` is exactly zero. -/
abbrev delta_r_zr_zero_of_local_laws := @Δ_R_eq_zero_of_local_laws

/-!
## Neural-Operator Bridge Exports

These aliases expose the Lean-backed interface that connects Section 9-style
neural-operator approximation assumptions to the existing approximate
theorem-backed route.
-/

/-- Uniform approximation on compact realized-call sets together with explicit
transfer assumptions yields an approximate-local-law bundle. -/
abbrev neural_operator_approx_local_laws_bundle :=
  @FormalProofs.OPT.approxLocalLawsBundle_of_uniformApproxExactTheoremBacked

/-- Main neural-operator realization bridge: exact theorem-backedness for an
ideal deterministic summarizer plus uniform approximation and transfer
assumptions implies approximate theorem-backedness for the realized operator. -/
abbrev neural_operator_theorem_backed_bridge :=
  @FormalProofs.OPT.approxTheoremBacked_of_uniformApproxExactTheoremBacked

/-- Kovachki Lemma-21 compact-union theorem surface used to obtain stability of
finite-rank input approximants on compact realized-call sets. -/
abbrev kovachki_lemma21_compact_union :=
  @ML.kovachki_compact_iUnion_image_of_uniform_limit

/-- The finite-coordinate approximation-property surface used by the
finite-dimensionalization proof is equivalent to the standard
finite-rank-operator AP statement. -/
abbrev finite_rank_ap_iff_standard :=
  @ML.finiteRankApproximationProperty_iff_standard

/-- Uniformly continuous target operators discharge the Lemma-21 stability
premise used by the finite-dimensionalization theorem. -/
abbrev kovachki_lemma21_stability_of_uniform_continuous :=
  @ML.kovachkiLemma21Stability_of_uniformContinuous

/-- Kovachki Lemma-22 finite-dimensionalization theorem surface: continuous
operators on compact AP-space call sets admit encoder--map--decoder
realizations to arbitrary tolerance. -/
abbrev kovachki_lemma22_finite_dimensionalization :=
  @ML.kovachki_finiteDimensionalization_on_compact

/-- Lemma-22 finite-dimensionalization with Lemma-21 stability discharged by
uniform continuity of the target operator. -/
abbrev kovachki_lemma22_finite_dimensionalization_uniform_continuous :=
  @ML.kovachki_finiteDimensionalization_on_compact_of_uniformContinuous

/-- A Lemma-22 finite-dimensionalization witness induces the uniform compact-set
approximation predicate consumed by the C-TreePO local-law bridge. -/
abbrev uniform_operator_approx_of_kovachki_finite_dimensionalization :=
  @ML.uniformOperatorApproxOnCompact_of_kovachkiFiniteDimensionalization

/-- Uniform-continuity version: AP plus compactness and uniform continuity yield
the uniform compact-set approximation predicate consumed by the local-law
bridge. -/
abbrev uniform_operator_approx_exists_of_kovachki_uniform_continuous :=
  @ML.uniformApproxOnCompact_exists_of_kovachkiFiniteDimensionalization_of_uniformContinuous

/-- Type-erased finite-dimensionalization certificates for represented C-TreePO
call sites feed the existing theorem-backedness bridge. -/
abbrev neural_operator_theorem_backed_bridge_of_kovachki_finite_dimensionalization :=
  @FormalProofs.OPT.approxTheoremBacked_of_kovachkiFiniteDimensionalization

/-- Exact neural-operator class membership plus exact theorem-backedness is the
certificate consumed by preference objectives. -/
abbrev neural_operator_preference_exact_bridge :=
  @FormalProofs.OPT.ExactNeuralOperatorPreferenceBridge

/-- Uniform neural-operator approximation bridge for preference objectives. -/
abbrev neural_operator_preference_uniform_bridge :=
  @FormalProofs.OPT.ApproxNeuralOperatorPreferenceBridge

/-- Finite-dimensionalization neural-operator bridge for preference objectives. -/
abbrev neural_operator_preference_fd_bridge :=
  @FormalProofs.OPT.FDNeuralOperatorPreferenceBridge

/-- Generic exact expected-loss transport through an exact neural-operator bridge. -/
abbrev expected_loss_generic_eq_via_neural_operator_exact_bridge :=
  @FormalProofs.OPT.expectedLossGeneric_eq_via_neuralOperatorExactBridge

/-- Generic exact compositional preference-loss transport through an exact
neural-operator bridge. -/
abbrev expected_pref_loss_eq_via_neural_operator_exact_bridge :=
  @FormalProofs.OPT.expectedPrefLoss_eq_via_neuralOperatorExactBridge

/-- Exact nested preference-program transport through an exact neural-operator bridge. -/
abbrev expected_pref_loss_prog_eq_via_neural_operator_exact_bridge :=
  @FormalProofs.OPT.expectedPrefLossProg_eq_via_neuralOperatorExactBridge

/-- DPO zero-gap transport through an exact neural-operator bridge. -/
abbrev dpo_equivalence_via_neural_operator_exact_bridge :=
  @FormalProofs.OPT.dpo_equivalence_via_neuralOperatorExactBridge

/-- Camel-case discoverability alias for DPO zero-gap transport through an
exact neural-operator preference bridge. -/
abbrev neuralOperatorDPOExactPreferenceBridge :=
  @FormalProofs.OPT.dpo_equivalence_via_neuralOperatorExactBridge

/-- GRPO-PL zero-gap transport through an exact neural-operator bridge. -/
abbrev grpo_pl_equivalence_via_neural_operator_exact_bridge :=
  @FormalProofs.OPT.grpo_pl_equivalence_via_neuralOperatorExactBridge

/-- Camel-case discoverability alias for GRPO-PL zero-gap transport through an
exact neural-operator preference bridge. -/
abbrev neuralOperatorGRPOPLExactPreferenceBridge :=
  @FormalProofs.OPT.grpo_pl_equivalence_via_neuralOperatorExactBridge

/-- GRPO-RL zero-gap transport through an exact neural-operator bridge. -/
abbrev grpo_rl_equivalence_via_neural_operator_exact_bridge :=
  @FormalProofs.OPT.grpo_rl_equivalence_via_neuralOperatorExactBridge

/-- Camel-case discoverability alias for GRPO-RL zero-gap transport through an
exact neural-operator preference bridge. -/
abbrev neuralOperatorGRPORLExactPreferenceBridge :=
  @FormalProofs.OPT.grpo_rl_equivalence_via_neuralOperatorExactBridge

/-- Generic Lipschitz expected-objective gap bound through the uniform
neural-operator bridge. -/
abbrev expected_objective_gap_via_neural_operator_uniform_bridge :=
  @FormalProofs.OPT.expectedObjectiveGap_via_neuralOperatorUniformBridge

/-- Generic Lipschitz expected-objective gap bound through the
finite-dimensionalization neural-operator bridge. -/
abbrev expected_objective_gap_via_neural_operator_fd_bridge :=
  @FormalProofs.OPT.expectedObjectiveGap_via_neuralOperatorFDBridge

/-- DPO gap bound through the uniform neural-operator bridge. -/
abbrev dpo_gap_via_neural_operator_uniform_bridge :=
  @FormalProofs.OPT.dpo_gap_via_neuralOperatorUniformBridge

/-- Camel-case discoverability alias for the DPO uniform neural-operator
preference-gap theorem. -/
abbrev neuralOperatorDPOUniformPreferenceGap :=
  @FormalProofs.OPT.dpo_gap_via_neuralOperatorUniformBridge

/-- DPO gap bound through the finite-dimensionalization neural-operator bridge. -/
abbrev dpo_gap_via_neural_operator_fd_bridge :=
  @FormalProofs.OPT.dpo_gap_via_neuralOperatorFDBridge

/-- Camel-case discoverability alias for the DPO finite-dimensionalization
neural-operator preference-gap theorem. -/
abbrev neuralOperatorDPOFDPreferenceGap :=
  @FormalProofs.OPT.dpo_gap_via_neuralOperatorFDBridge

/-- GRPO-PL gap bound through the uniform neural-operator bridge. -/
abbrev grpo_pl_gap_via_neural_operator_uniform_bridge :=
  @FormalProofs.OPT.grpo_pl_gap_via_neuralOperatorUniformBridge

/-- Camel-case discoverability alias for the GRPO-PL uniform neural-operator
preference-gap theorem. -/
abbrev neuralOperatorGRPOPLUniformPreferenceGap :=
  @FormalProofs.OPT.grpo_pl_gap_via_neuralOperatorUniformBridge

/-- GRPO-PL gap bound through the finite-dimensionalization neural-operator bridge. -/
abbrev grpo_pl_gap_via_neural_operator_fd_bridge :=
  @FormalProofs.OPT.grpo_pl_gap_via_neuralOperatorFDBridge

/-- Camel-case discoverability alias for the GRPO-PL finite-dimensionalization
neural-operator preference-gap theorem. -/
abbrev neuralOperatorGRPOPLFDPreferenceGap :=
  @FormalProofs.OPT.grpo_pl_gap_via_neuralOperatorFDBridge

/-- GRPO-RL gap bound through the uniform neural-operator bridge. -/
abbrev grpo_rl_gap_via_neural_operator_uniform_bridge :=
  @FormalProofs.OPT.grpo_rl_gap_via_neuralOperatorUniformBridge

/-- Camel-case discoverability alias for the GRPO-RL uniform neural-operator
preference-gap theorem. -/
abbrev neuralOperatorGRPORLUniformPreferenceGap :=
  @FormalProofs.OPT.grpo_rl_gap_via_neuralOperatorUniformBridge

/-- GRPO-RL gap bound through the finite-dimensionalization neural-operator bridge. -/
abbrev grpo_rl_gap_via_neural_operator_fd_bridge :=
  @FormalProofs.OPT.grpo_rl_gap_via_neuralOperatorFDBridge

/-- Camel-case discoverability alias for the GRPO-RL finite-dimensionalization
neural-operator preference-gap theorem. -/
abbrev neuralOperatorGRPORLFDPreferenceGap :=
  @FormalProofs.OPT.grpo_rl_gap_via_neuralOperatorFDBridge

/-- ASCII alias for the approximate-local-law distortion theorem used in the
paper's neural-operator discussion. -/
abbrev delta_r_zr_bound_of_approx_bundle :=
  @FormalProofs.OPT.Δ_R_ZR_le_of_approx_bundle

/-- Equation-(6)-style finite neural-operator architecture surface. -/
abbrev equation6_neural_operator := @ML.Equation6NeuralOperator

/-- Function class represented by equation-(6)-style neural operators. -/
abbrev equation6_neural_operator_class := @ML.Equation6NeuralOperatorClass

/-- Uniform compact-set approximation interface for equation-(6) neural
operators. -/
abbrev equation6_universal_approx_uniform := @ML.Equation6UniversalApproxUniform

/-- Expected-`L²` approximation interface for equation-(6) neural operators. -/
abbrev equation6_universal_approx_l2 := @ML.Equation6UniversalApproxL2

/-- Single-head attention is definitionally the associated discretized kernel
layer in the formalized Kovachki-Proposition-6 interface. -/
abbrev single_head_attention_eq_discretized_kernel_layer :=
  @ML.singleHeadAttention_eq_discretizedKernelLayer

/-- A transformer block built from attention plus pointwise/residual maps is a
neural-operator layer. -/
abbrev transformer_block_is_neural_operator_layer :=
  @ML.transformerBlock_is_neuralOperatorLayer

/-- Finite transformer encoder stacks instantiate the equation-(6)
neural-operator architecture. -/
abbrev transformer_encoder_is_equation6_neural_operator :=
  @ML.transformerEncoder_is_equation6NeuralOperator

/-- The realized finite transformer encoder belongs to the equation-(6)
representable function class. -/
abbrev transformer_encoder_mem_equation6_class :=
  @ML.transformerEncoder_mem_equation6Class

/-- Ambient neural-operator function class used for subspace/intersection
claims. -/
abbrev neural_operator_class := @NeuralOperatorSpaces.NeuralOperatorClass

/-- Certified subfamilies represent intersections with a proof predicate. -/
abbrev neural_operator_certified_subfamily := @NeuralOperatorSpaces.CertifiedSubfamily

/-- Operators induced by mergeable sketches with exact sketch-local witnesses. -/
abbrev mergeable_sketch_summary_class :=
  @NeuralOperatorSpaces.MergeableSketchSummaryClass

/-- Exact C1/C2/C3 local-law subspace inside a chosen neural-operator class. -/
abbrev exact_local_law_neural_operators :=
  @NeuralOperatorSpaces.ExactLocalLawNeuralOperators

/-- Approximate C1/C2/C3 local-law subspace inside a chosen neural-operator
class. -/
abbrev approx_local_law_neural_operators :=
  @NeuralOperatorSpaces.ApproxLocalLawNeuralOperators

/-- Intersection of a chosen neural-operator class with exact mergeable-sketch
summaries. -/
abbrev neural_operator_mergeable_sketch_overlap :=
  @NeuralOperatorSpaces.NeuralOperatorMergeableSketchOverlap

/-- Sketch-induced operators with exact sketch-local witnesses lie in the exact
local-law subspace on every tree. -/
abbrev mergeable_sketch_summary_subset_exact_local_law_subspace :=
  @NeuralOperatorSpaces.mergeableSketchSummaryClass_subset_exactLocalLawSubspace

/-- Certified mergeable-sketch/neural-operator overlap lies inside the exact
local-law neural-operator subspace. -/
abbrev mergeable_sketch_overlap_subset_exact_local_law_neural_operators :=
  @NeuralOperatorSpaces.mergeableSketch_overlap_subset_exactLocalLawNeuralOperators

/-- Proposition-1-style mergeable neural operators lie in the exact local-law
subspace. -/
abbrev mergeable_neural_operator_mem_exact_local_law_neural_operators :=
  @NeuralOperatorSpaces.paper_mergeableNeuralOperator_mem_exactLocalLawNeuralOperators

/-- Statement that local-law weights are a projection onto the exact local-law
subspace. -/
abbrev local_law_weights_are_projection :=
  @NeuralOperatorSpaces.LocalLawWeightsAreProjection

/-- Class-restricted statement that local-law weights project a chosen
neural-operator class onto its exact local-law subspace. -/
abbrev local_law_weights_are_projection_on :=
  @NeuralOperatorSpaces.LocalLawWeightsAreProjectionOn

/-- Assumption that the approximation-error zero set is exactly the exact
local-law subspace. -/
abbrev approximation_error_structured_by_local_laws :=
  @NeuralOperatorSpaces.ApproximationErrorStructuredByLocalLaws

/-- Class-restricted structured-approximation-error assumption. -/
abbrev approximation_error_structured_by_local_laws_on :=
  @NeuralOperatorSpaces.ApproximationErrorStructuredByLocalLawsOn

/-- Iff: local-law weights are a projection exactly when approximation error is
structured by the local-law zero set. -/
abbrev local_law_projection_iff_structured_approximation_error :=
  @NeuralOperatorSpaces.localLawWeightsAreProjection_iff_approximationErrorStructuredByLocalLaws

/-- Class-restricted iff between projection weights and structured
approximation error. -/
abbrev local_law_projection_on_iff_structured_approximation_error_on :=
  @NeuralOperatorSpaces.localLawWeightsAreProjectionOn_iff_approximationErrorStructuredByLocalLawsOn

/-- The faithful-penalty structure is equivalent to the structured-error
assumption. -/
abbrev faithful_projection_penalty_iff_structured_approximation_error :=
  @NeuralOperatorSpaces.faithfulProjectionPenalty_iff_approximationErrorStructuredByLocalLaws

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

/-- Generic optimizer-certificate transfer: a surrogate `ε`-minimizer plus a
uniform objective perturbation gives a true `(ε + 2δ)`-minimizer. -/
abbrev surrogate_optimizer_certificate_uniform_transfer :=
  @FormalProofs.OPT.surrogateOptimizerCertificate_true_epsilonArgmin_of_uniform_loss_perturbation

/-- Oracle-measurable version of
`surrogate_optimizer_certificate_uniform_transfer`. -/
abbrev oracle_measurable_surrogate_optimizer_certificate_uniform_transfer :=
  @FormalProofs.OPT.oracleMeasurableSurrogateOptimizerCertificate_true_epsilonArgmin_of_uniform_loss_perturbation

/-- Generic two-stage optimizer-certificate transfer for oracle-measurable
surrogates. -/
abbrev oracle_measurable_surrogate_optimizer_certificate_two_stage_uniform_transfer :=
  @FormalProofs.OPT.oracleMeasurableSurrogateOptimizerCertificate_true_epsilonArgmin_of_uniform_two_stage_loss_perturbation

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

/-- First-principles fixed-ranker Plackett-Luce GRPO-PL TreePO gap certificate:
the expected-loss Lipschitz hypothesis is discharged internally from the proved
fixed-ranker Plackett-Luce route. -/
abbrev grpo_pl_tree_gap_ipw_plackett_luce_fixed_ranker :=
  @grpo_pl_tree_gap_bounded_ipw_plackettLuce_fixed_ranker

/-- Fixed-ranker Plackett-Luce GRPO-PL sketch-upper transport with no explicit
expected-Lipschitz hypothesis at the API boundary. -/
abbrev grpo_pl_tree_gap_sketch_upper_plackett_luce_fixed_ranker :=
  @grpo_pl_tree_gap_bounded_by_sketch_upper_plackettLuce_fixed_ranker

/-- End-to-end fixed-ranker Plackett-Luce GRPO-PL TreePO certificate with the
expected-loss Lipschitz condition discharged from first principles. -/
abbrev grpo_pl_treepo_end_to_end_plackett_luce_fixed_ranker :=
  @grpo_pl_treepo_end_to_end_certificate_plackettLuce_fixed_ranker

/-- Fixed-ranker Plackett-Luce GRPO-PL end-to-end certificate with an optional
oracle-measurement layer above the oracle-indexed target. -/
abbrev grpo_pl_treepo_end_to_end_plackett_luce_fixed_ranker_with_oracleMeasurement :=
  @grpo_pl_treepo_end_to_end_certificate_plackettLuce_fixed_ranker_with_oracleMeasurement

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

/-- One-parameter oracle/projection weights: `(1-λ)` on oracle distortion and
`λρ_i` on C1/C2/C3 local-law budgets. -/
abbrev oracle_projection_objective_weights :=
  @FormalProofs.OPT.oracleProjectionObjectiveWeights

/-- `λ=0` gives pure oracle distortion weights. -/
abbrev oracle_projection_objective_weights_lam_zero :=
  @FormalProofs.OPT.oracleProjectionObjectiveWeights_lam_zero

/-- `λ=1` gives pure local-law projection weights. -/
abbrev oracle_projection_objective_weights_lam_one :=
  @FormalProofs.OPT.oracleProjectionObjectiveWeights_lam_one

/-- Nonnegativity of the oracle/projection weights on the simplex. -/
abbrev oracle_projection_objective_weights_nonneg :=
  @FormalProofs.OPT.oracleProjectionObjectiveWeights_nonneg

/-- Unit-mass identity for oracle/projection weights when law shares sum to one. -/
abbrev oracle_projection_objective_weights_total_mass :=
  @FormalProofs.OPT.oracleProjectionObjectiveWeights_total_mass

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

/-- Direct paper-facing oracle/projection objective
`(1-λ)L_oracle + λ∑ρ_i ε_i`. -/
abbrev oracle_projection_objective := @FormalProofs.OPT.oracleProjectionObjective

/-- Direct oracle/projection objective equals the certified regularized objective
with zero summary-cost weight. -/
abbrev oracle_projection_objective_eq_certified_regularized_objective :=
  @FormalProofs.OPT.oracleProjectionObjective_eq_certifiedRegularizedObjective

/-- `λ=0` endpoint is pure oracle risk. -/
abbrev oracle_projection_objective_lam_zero :=
  @FormalProofs.OPT.oracleProjectionObjective_lam_zero

/-- `λ=1` endpoint is pure local-law projection penalty. -/
abbrev oracle_projection_objective_lam_one :=
  @FormalProofs.OPT.oracleProjectionObjective_lam_one

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

/-- Neural-operator realization bridge for the oracle/projection objective. -/
abbrev oracle_projection_objective_from_neural_operator_bridge :=
  @FormalProofs.OPT.oracleProjectionObjective_le_of_uniformApproxExactTheoremBacked

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

/-- Exact node-indexed / context-conditioned latent-state families recover the
canonical span state by tree induction. -/
abbrev node_indexed_state_exact_on_tree :=
  @FormalProofs.OPT.nodeIndexedStateEval_eq_feature_of_exact

/-- Any downstream readout of an exact node-indexed latent-state family agrees
with the readout of the canonical span state on the full tree span. -/
abbrev node_indexed_state_utility_exact_on_tree :=
  @FormalProofs.OPT.nodeIndexedStateUtility_exact_on_tree

/-- If the node-indexed family collapses to one exact mergeable latent-state
operator, it recovers the same exact-control theorem surface. -/
abbrev node_indexed_state_utility_exact_on_tree_of_mergeable_feature :=
  @FormalProofs.OPT.nodeIndexedStateUtility_exact_on_tree_of_mergeable_feature

/-- Exact node-indexed latent-state recovery plus approximate theorem-backed
transport yields a root-state utility bound against the node-indexed state
itself. -/
abbrev node_indexed_state_utility_bound_of_approx_backed :=
  @FormalProofs.OPT.expected_nodeIndexedStateUtility_bound_of_exactNodeIndexed_and_approxBacked

/-- Markov exact-state utility preservation (count/endpoints lane). -/
abbrev markov_state_utility_exact_on_tree :=
  @FormalProofs.OPT.markovStateUtility_exact_on_tree

/-- Markov count-only exact-control utility reaches its optimum on the exact tree fold. -/
abbrev markov_count_only_exact_on_tree :=
  @FormalProofs.OPT.markovCountOnlyUtility_exact_on_tree

/-- Markov count-plus-endpoints exact-control utility reaches its optimum on the exact tree fold. -/
abbrev markov_count_endpoints_exact_on_tree :=
  @FormalProofs.OPT.markovCountEndpointsUtility_exact_on_tree

/-- Exact carrier merge preserves the theorem-facing Markov sketch projection. -/
abbrev markov_carrier_projection_merge_exact :=
  @FormalProofs.OPT.MarkovCarrierState.proj_mul

/-- Opaque-carrier exact-sketch merge preserves the theorem-facing Markov sketch
projection. This is the runtime-facing alias for the same carrier theorem. -/
abbrev markov_opaque_carrier_exact_sketch_merge_exact :=
  @FormalProofs.OPT.MarkovCarrierState.proj_mul

/-- Exact carrier states with exact projected merge give zero root distortion by
Theorem 1. -/
abbrev markov_carrier_projection_root_distortion_zero :=
  @FormalProofs.OPT.MarkovCarrierState.root_distortion_zero

/-- Opaque-carrier exact-sketch states with exact projected merge give zero
root distortion by Theorem 1. -/
abbrev markov_opaque_carrier_exact_sketch_root_distortion_zero :=
  @FormalProofs.OPT.MarkovCarrierState.root_distortion_zero

/-- Projection-preserving re-encoding yields C2/L3 on the theorem-facing Markov
oracle, regardless of how the residual carrier evolves. -/
abbrev markov_carrier_projection_L3_of_projection_preserving_reencode :=
  @FormalProofs.OPT.MarkovCarrierState.L3_of_proj_preserving_reencode

/-- Projection-preserving re-encoding yields C2/L3 on the theorem-facing Markov
oracle for the opaque-carrier exact-sketch lane. -/
abbrev markov_opaque_carrier_exact_sketch_L3_of_projection_preserving_reencode :=
  @FormalProofs.OPT.MarkovCarrierState.L3_of_proj_preserving_reencode

/-- Worked 4-leaf Markov carrier example: the carrier projection recovers the
same `2`-changepoint answer as the exact sketch tree. -/
abbrev markov_carrier_projection_example_oracle_correct :=
  @FormalProofs.OPT.carrierExampleTree_oracle_correct

/-- Worked 4-leaf opaque-carrier example: the certified sketch projection
recovers the same `2`-changepoint answer as the exact sketch tree. -/
abbrev markov_opaque_carrier_exact_sketch_example_oracle_correct :=
  @FormalProofs.OPT.carrierExampleTree_oracle_correct

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
## Theorem 6.6: Readout Alignment for Theorem-Backed Routes

These lemmas isolate the assumption gap between three practically important
regimes:

1. **same-surface / theorem-routed** objectives, which use exactly the
   theorem-bearing feature;
2. **factored auxiliary readouts**, which are still sound if they are obtained
   by deterministic post-processing of that feature; and
3. **unfactored auxiliary heads**, which fall outside the theorem route if they
   distinguish states that the theorem-bearing feature identifies.

Implementation note:

- the tree-neural `shared_feature` and `shared_feature_adapters` routes are
  intended to realize regime (2): a learned theorem feature `φ(z)` is first
  extracted from the latent state, and all theorem-facing/root supervision is
  required to factor through `φ`;
- `slotwise` is a stronger architectural bias that also fits regime (2), but by
  predeclaring a particular theorem surface inside the latent state; and
- any full-state auxiliary root head is theory-aligned only if it can be shown
  to factor through the same learned theorem feature.
-/

/-- Same-surface routing is the minimal theory-aligned special case. -/
abbrev same_surface_implies_factored_readout :=
  @FormalProofs.OPT.sameReadoutSurface_implies_factorsThroughFeature

/-- A factored readout is constant on fibers of the theorem-bearing feature. -/
abbrev factored_readout_respects_theorem_feature :=
  @FormalProofs.OPT.readoutFactorsThroughFeature_respects_feature_fibers

/-- If an auxiliary head separates points on one theorem-feature fiber, it is
not theory-aligned via that feature. -/
abbrev separated_auxiliary_head_not_theory_aligned :=
  @FormalProofs.OPT.not_readoutFactorsThroughFeature_of_distinguished_feature_fibers

/-- Exact theorem-backed transport for any loss indexed by a readout that
factors through the theorem-bearing feature. -/
abbrev factored_readout_expected_loss_transport :=
  @FormalProofs.OPT.expected_loss_eq_via_ZR_of_exactTheoremBacked_and_factoredReadout

/-- Exact theorem-backed transport for direct supervised learning of a factored
root readout. -/
abbrev factored_readout_supervised_transport :=
  @FormalProofs.OPT.supervisedReadoutLoss_eq_via_ZR_of_exactTheoremBacked_and_factoredReadout

/-- Exact theorem-backed transport for the minimal same-surface root objective. -/
abbrev same_surface_supervised_transport :=
  @FormalProofs.OPT.supervisedReadoutLoss_eq_via_ZR_of_exactTheoremBacked_and_sameSurface

/-!
## Theorem 6.6b: Fixed-Binary Tree Diffusion Packaging

These aliases package the fixed-tree diffusion surface without introducing a
new joint stochastic state:

- `TextCheckpoint` is the round-indexed text distribution `ZR g x r T`;
- `LatentCheckpoint` is the exact mergeable latent fold `mergeFold encode merge T`;
- the transport theorems reuse the existing theorem-backed route for exact or
  bounded downstream readouts.
-/

/-- Fixed-tree round-indexed text checkpoint. -/
abbrev fixed_binary_text_checkpoint :=
  @FormalProofs.OPT.TextCheckpoint

/-- Fixed-tree exact latent checkpoint. -/
abbrev fixed_binary_latent_checkpoint :=
  @FormalProofs.OPT.LatentCheckpoint

/-- Bundled fixed-binary-tree diffusion spec. -/
abbrev fixed_binary_tree_diffusion_spec :=
  @FormalProofs.OPT.FixedBinaryTreeDiffusionSpec

/-- Exact fixed-tree text-checkpoint preservation from a `LocalLawsBundle`. -/
abbrev fixed_binary_text_checkpoint_zero_of_local_laws :=
  @FormalProofs.OPT.textCheckpoint_distortion_zero_of_localLaws

/-- Approximate fixed-tree text-checkpoint distortion bound from an
`ApproxLocalLawsBundle`. -/
abbrev fixed_binary_text_checkpoint_bound_of_approx_local_laws :=
  @FormalProofs.OPT.textCheckpoint_distortion_le_of_approxLocalLaws

/-- Exact fixed-tree latent-checkpoint recovery of the theorem feature. -/
abbrev fixed_binary_latent_checkpoint_exact :=
  @FormalProofs.OPT.latentCheckpoint_eq_feature_of_exactMergeable

/-- Exact downstream-transport theorem for readouts that factor through the
fixed-tree theorem feature. -/
abbrev fixed_binary_factored_readout_transport :=
  @FormalProofs.OPT.factoredReadout_expectedLoss_eq_via_textCheckpoint_of_localLaws

/-- Exact supervised transport for factored fixed-tree root readouts. -/
abbrev fixed_binary_factored_readout_supervised_transport :=
  @FormalProofs.OPT.factoredReadout_supervisedLoss_eq_via_textCheckpoint_of_localLaws

/-- Quantitative shared-feature bound on realized fixed-tree text checkpoints. -/
abbrev fixed_binary_paired_readout_support_bound :=
  @FormalProofs.OPT.pairedApproxReadoutBound_on_textCheckpointSupport_of_localLaws

/-- Worked fixed-tree Markov example: the exact latent checkpoint recovers the
canonical Markov theorem state. -/
abbrev fixed_binary_markov_latent_checkpoint_example :=
  @FormalProofs.OPT.markovPath_latentCheckpoint_exact_example

/-- Worked fixed-tree Markov counterexample: count-only summaries are not
compositionally sufficient. -/
abbrev fixed_binary_markov_count_only_counterexample :=
  @FormalProofs.OPT.markovPath_count_only_counterexample

/-!
## Theorem 6.7: Feature Fibers, Covered Pair Supervision, and Shared-Feature Objectives

These results restate the theorem route in the language of a learned theorem
feature `Φ`, its induced equivalence classes, and the sparse covered-pair
relations used in practice to train it.
-/

/-- C2-style equivalence relation induced by a learned theorem feature. -/
abbrev same_feature_fiber :=
  @FormalProofs.OPT.SameFeatureFiber

/-- Relation-first primitive: two inputs lie on the same oracle fiber when the
oracle identifies them exactly. -/
abbrev same_oracle_fiber :=
  @FormalProofs.OPT.SameOracleFiber

/-- Exact oracle recovery is exactly the statement that the learned theorem
feature is constant on oracle fibers. -/
abbrev oracle_feature_recovery_respects_same_oracle_fiber :=
  @FormalProofs.OPT.oracleRecoversFeature_iff_respects_sameOracleFiber

/-- Approximate oracle recovery is exactly the statement that the learned
theorem feature has bounded diameter on each oracle fiber. -/
abbrev approx_feature_recovery_bounded_on_same_oracle_fiber :=
  @FormalProofs.OPT.approxOracleRecoversFeature_iff_bounded_on_sameOracleFiber

/-- Exact theorem-backed realized leaves stay on the same oracle fiber as their
source leaf. -/
abbrev leaf_support_same_oracle_fiber :=
  @FormalProofs.OPT.leaf_support_sameOracleFiber_of_exactTheoremBacked

/-- Exact theorem-backed realized merges stay on the same oracle fiber as their
raw subtree. -/
abbrev merge_support_same_oracle_fiber :=
  @FormalProofs.OPT.merge_support_sameOracleFiber_of_exactTheoremBacked

/-- Exact theorem-backed on-range re-summaries stay on the same oracle fiber as
their input theorem object. -/
abbrev idempotent_support_same_oracle_fiber :=
  @FormalProofs.OPT.idempotent_support_sameOracleFiber_of_exactTheoremBacked

/-- Exact theorem-backed multi-round reductions stay on the same oracle fiber
as the original document. -/
abbrev zr_support_same_oracle_fiber :=
  @FormalProofs.OPT.zr_support_sameOracleFiber_of_exactTheoremBacked

/-- Exact bridge: encoded-feature zero distortion is equivalent to lying on the
same feature fiber. -/
abbrev same_feature_fiber_iff_encodedOracle_zero :=
  @FormalProofs.OPT.sameFeatureFiber_iff_encodedOracle_zero

/-- Exact theorem-backed leaf support stays inside one feature fiber whenever
the oracle identifies the feature. -/
abbrev leaf_support_same_feature_fiber :=
  @FormalProofs.OPT.leaf_support_sameFeatureFiber_of_exactTheoremBacked

/-- Exact theorem-backed merge support stays inside one feature fiber whenever
the oracle identifies the feature. -/
abbrev merge_support_same_feature_fiber :=
  @FormalProofs.OPT.merge_support_sameFeatureFiber_of_exactTheoremBacked

/-- Exact theorem-backed on-range idempotence stays inside one feature fiber
whenever the oracle identifies the feature. -/
abbrev idempotent_support_same_feature_fiber :=
  @FormalProofs.OPT.idempotent_support_sameFeatureFiber_of_exactTheoremBacked

/-- Approximate feature-fiber distortion inherits the oracle distortion budget
through a Lipschitz theorem feature. -/
abbrev feature_fiber_distortion_le_of_featureLipschitzFromOracle :=
  @FormalProofs.OPT.expected_featureFiberDistortion_le_of_featureLipschitzFromOracle

/-- Restricted approximate oracle recovery is the right surface when only a
covered relation of oracle-labeled pairs is available. -/
abbrev covered_approx_oracle_feature_recovery :=
  @FormalProofs.OPT.ApproxOracleRecoversFeatureOn

/-- Zero contrastive risk on a covered same-fiber pair distribution forces
exact oracle-feature recovery on that covered relation. -/
abbrev covered_zero_contrastive_risk_implies_oracle_feature_recovery :=
  @FormalProofs.OPT.oracleRecoversFeatureOn_of_zero_contrastive_risk

/-- Approximate factorization bounds how much a readout can vary inside one
learned theorem-feature fiber. -/
abbrev approx_readout_factorization_fiber_bound :=
  @FormalProofs.OPT.approxReadoutFactorsThroughFeature_fiber_bound

/-- Approximate shared-feature readout stability on oracle fibers. -/
abbrev combined_approx_readout_bound_on_oracle_fibers :=
  @FormalProofs.OPT.combined_readout_bound_on_oracle_fibers

/-- Covered-pair version of the approximate shared-feature readout bound. -/
abbrev combined_approx_readout_bound_on_covered_oracle_fibers :=
  @FormalProofs.OPT.combined_readout_bound_on_covered_oracle_fibers

/-- Two simultaneous heads remain theory-aligned whenever each head factors
through the same theorem-bearing feature. -/
abbrev paired_factored_readouts :=
  @FormalProofs.OPT.pairedReadoutFactorsThroughFeature

/-- If two heads approximately factor through the same theorem feature, then
both are stable on each oracle fiber. -/
abbrev paired_approx_readout_bound_on_same_oracle_fiber :=
  @FormalProofs.OPT.paired_approxReadoutBound_on_sameOracleFiber_of_sharedFeature

/-- The previous paired oracle-fiber bound can be read as simultaneous
approximate oracle recovery for both heads. -/
abbrev paired_approx_readout_recovery :=
  @FormalProofs.OPT.paired_approxOracleRecoversReadouts_of_sharedFeature

/-- Covered-pair simultaneous approximate recovery for both shared-feature
heads. -/
abbrev paired_approx_readout_recovery_on_covered_pairs :=
  @FormalProofs.OPT.paired_approxOracleRecoversReadoutsOn_of_sharedFeature

/-- Exact theorem-backed reductions inherit the paired approximate readout
bound on every realized `ZR` support event. -/
abbrev zr_support_paired_approx_readout_bound :=
  @FormalProofs.OPT.zr_support_paired_approxReadoutBound_of_exactTheoremBacked_and_sharedFeature

/-- Exact theorem-backed reductions inherit the covered-pair shared-feature
bound whenever the realized `ZR` support stays inside the covered relation. -/
abbrev zr_support_paired_approx_readout_bound_on_covered_pairs :=
  @FormalProofs.OPT.zr_support_paired_approxReadoutBound_of_exactTheoremBacked_and_sharedFeature_on

/-- Direct bridge from zero covered-pair contrastive risk to simultaneous
stability of approximately factored task and summary heads. -/
abbrev paired_approx_readout_bound_on_covered_same_oracle_fiber_of_zero_contrastive_risk :=
  @FormalProofs.OPT.paired_approxReadoutBound_on_coveredSameOracleFiber_of_zero_contrastiveRisk

/-- Exact theorem-backed transport for hard same-class feature objectives. -/
abbrev same_feature_class_transport :=
  @FormalProofs.OPT.expected_sameFeatureClassUtility_eq_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature

/-- Exact theorem-backed transport for hard different-class feature objectives. -/
abbrev different_feature_class_transport :=
  @FormalProofs.OPT.expected_differentFeatureClassUtility_eq_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature

/-- Approximate theorem-backed transport for same-class feature objectives under
the usual feature-Lipschitz and utility regularity hypotheses. -/
abbrev same_feature_class_transport_approx :=
  @FormalProofs.OPT.expected_sameFeatureClassUtility_with_measurement_error_via_ZR_of_approxTheoremBacked_and_featureLipschitz

/-- Approximate theorem-backed transport for different-class feature objectives
under the usual feature-Lipschitz and utility regularity hypotheses. -/
abbrev different_feature_class_transport_approx :=
  @FormalProofs.OPT.expected_differentFeatureClassUtility_with_measurement_error_via_ZR_of_approxTheoremBacked_and_featureLipschitz

/-- Exact theorem-backed transport for arbitrary real-valued scores on decoded
labels. -/
abbrev label_score_transport :=
  @FormalProofs.OPT.expected_labelScoreUtility_eq_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature

/-- Exact theorem-backed transport for arbitrary decoded-label scores with a
noisy observation of the truth feature. -/
abbrev label_score_transport_exact_measurement_error :=
  @FormalProofs.OPT.expected_labelScoreUtility_with_measurement_error_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature

/-- Approximate theorem-backed transport for arbitrary decoded-label scores
under the standard feature-Lipschitz and utility regularity assumptions. -/
abbrev label_score_transport_approx :=
  @FormalProofs.OPT.expected_labelScoreUtility_with_measurement_error_via_ZR_of_approxTheoremBacked_and_featureLipschitz

/-!
## Theorem 6.8: Two-Stage Learned-Oracle Route

These lemmas package the practical workflow where one first learns or queries an
expensive surrogate oracle `f̂`, and then learns the tree summary relative to
`f̂` rather than directly relative to the true oracle `f*`.

The reduction-side consequence is additive:

- exact theorem-backedness for `f̂` yields true-oracle distortion at most `2ε`
  when `f̂` is uniformly within `ε` of `f*`; and
- approximate theorem-backedness for `f̂` yields the usual audited local-law
  transport budget plus the same additive `2ε` surrogate slack.
-/

/-- Uniform stage-1 learned-oracle approximation. -/
abbrev uniform_oracle_surrogate_approximation :=
  @FormalProofs.OPT.UniformOracleApproximation

/-- If two documents are identical under the learned surrogate oracle `f̂`,
their true-oracle values differ by at most `2ε`. -/
abbrev same_surrogate_fiber_implies_true_oracle_close :=
  @FormalProofs.OPT.sameSurrogateFiber_implies_trueOracleClose_of_uniformOracleApproximation

/-- Exact theorem-backed reductions for the surrogate oracle inherit a supportwise
true-oracle distortion bound with additive stage-1 slack. -/
abbrev zr_support_true_oracle_close_via_exact_surrogate :=
  @FormalProofs.OPT.zr_support_trueOracleDist_le_of_exactTheoremBacked_on_surrogate_and_uniformOracleApproximation

/-- Exact theorem-backedness for the surrogate oracle implies a `2ε` bound on
expected true-oracle distortion. -/
abbrev delta_r_zr_true_via_exact_surrogate :=
  @FormalProofs.OPT.Δ_R_ZR_true_le_of_exactTheoremBacked_on_surrogate_and_uniformOracleApproximation

/-- Any Lipschitz true-oracle utility inherits the exact-surrogate `2ε`
transport bound. This is the reduction-side theorem for "learn the expensive
oracle first, then train summaries against it". -/
abbrev true_oracle_utility_gap_via_exact_surrogate :=
  @FormalProofs.OPT.expected_trueOracleUtility_bound_via_ZR_of_exactTheoremBacked_on_surrogate_and_uniformOracleApproximation

/-- Approximate theorem-backedness for the surrogate oracle implies true-oracle
distortion bounded by the surrogate local-law budget plus `2ε`. -/
abbrev delta_r_zr_true_via_approx_surrogate :=
  @FormalProofs.OPT.Δ_R_ZR_true_le_of_approxTheoremBacked_on_surrogate_and_uniformOracleApproximation

/-- Any Lipschitz true-oracle utility inherits the approximate-surrogate budget:
surrogate local-law transport plus additive stage-1 oracle slack. -/
abbrev true_oracle_utility_gap_via_approx_surrogate :=
  @FormalProofs.OPT.expected_trueOracleUtility_bound_via_ZR_of_approxTheoremBacked_on_surrogate_and_uniformOracleApproximation

/-!
## Theorem 6.9: Two-Stage Label-Score Tradeoffs

These lemmas specialize the two-stage teacher-first route to arbitrary
real-valued scores on decoded labels, and they make the broader tradeoff surface
explicit.

- The direct route composes a learned surrogate oracle with arbitrary
  label-score objectives on the true oracle outputs.
- The layered route exposes the full end-to-end budget:
  stage-2 transport + stage-2 fiber error + measurement error + stage-1
  substitution cost.
- The comparison lemmas say when two-stage beats single-stage, and how errors
  accumulate across multiple surrogate/distillation stages.
-/

/-- Exact theorem-backedness for a learned surrogate oracle yields a direct
label-score gap bound on the true oracle outputs. -/
abbrev true_label_score_gap_via_exact_surrogate :=
  @FormalProofs.OPT.expected_trueLabelScoreUtility_bound_via_ZR_of_exactTheoremBacked_on_surrogate_and_uniformOracleApproximation

/-- Approximate theorem-backedness for a learned surrogate oracle yields a true
label-score gap bounded by the surrogate local-law budget plus additive stage-1
slack. -/
abbrev true_label_score_gap_via_approx_surrogate :=
  @FormalProofs.OPT.expected_trueLabelScoreUtility_bound_via_ZR_of_approxTheoremBacked_on_surrogate_and_uniformOracleApproximation

/-- Stage-2 label-score transport in the learned surrogate feature space. -/
abbrev label_score_stage2_transport_in_surrogate_space :=
  @FormalProofs.OPT.expected_labelScoreUtility_bound_in_surrogateFeatureSpace

/-- Full two-stage end-to-end label-score decomposition with an explicit
stage-1 substitution term. -/
abbrev label_score_two_stage_end_to_end_bound :=
  @FormalProofs.OPT.expected_labelScoreUtility_two_stage_end_to_end_bound

/-!
## Theorem 6.10: Explicit Product-State Score-Fiber Route

These lemmas specialize the shared-feature and two-stage routes to the intended
structured implementation:

- a learned theorem-bearing state of the form `Score × FiberState`,
- an exact scalar score readout from the first coordinate, and
- summary / downstream auxiliary readouts that use the full product state.
-/

/-- The scalar score readout in the factorized score-fiber route is the exact
first-coordinate projection. -/
abbrev score_readout_factors_through_product_score_fiber :=
  @FormalProofs.OPT.scoreReadoutFactorsThroughProductScoreFiber_firstCoordinate

/-- One shared product-state theorem feature supports an exact scalar score
head and an approximate summary head simultaneously. -/
abbrev paired_approx_readout_recovery_of_product_score_fiber :=
  @FormalProofs.OPT.paired_approxOracleRecoversReadouts_of_productScoreFiber

/-- Full two-stage end-to-end label-score decomposition specialized to a
product-state `Score × FiberState`. -/
abbrev label_score_two_stage_end_to_end_bound_of_product_score_fiber :=
  @FormalProofs.OPT.expected_labelScoreUtility_two_stage_end_to_end_bound_of_productScoreFiber

/-- Algebraic breakeven condition for when the two-stage route beats the direct
single-stage route. -/
abbrev two_stage_breakeven_tradeoff :=
  @FormalProofs.OPT.two_stage_breakeven_condition

/-- Multi-stage distillation amplifies upstream error by the downstream
Lipschitz factor. -/
abbrev multi_stage_distillation_error :=
  @FormalProofs.OPT.distillation_chain_error

/-- Distillation packaged as a direct approximate-recovery theorem for the
downstream representation. -/
abbrev distillation_chain_recovers_feature :=
  @FormalProofs.OPT.approxOracleRecoversFeature_of_distillation_chain

/-- Contractive distillation stages do not amplify upstream oracle-approximation
error. -/
abbrev contractive_distillation_error :=
  @FormalProofs.OPT.contractive_distillation_chain

/-- Contractive distillation packaged as a direct approximate-recovery theorem
for the downstream representation. -/
abbrev contractive_distillation_recovers_feature :=
  @FormalProofs.OPT.approxOracleRecoversFeature_of_contractive_distillation_chain

/-- A packaged stage-1 two-stage approximation can be pushed through another
Lipschitz distillation layer and still recover the true oracle approximately. -/
abbrev two_stage_bundle_recovers_feature_after_distillation :=
  @FormalProofs.OPT.approxOracleRecoversFeature_of_twoStageOracleApproximation_and_distillation

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

/-- Simulation-facing exact contract: every support tree of a stochastic Markov
path policy inherits exact local laws for the encoded exact state. -/
abbrev markov_path_stochastic_policy_local_laws :=
  @FormalProofs.OPT.markovPath_stochastic_policy_local_laws

/-- Simulation-facing exact contract: every support tree of a stochastic Markov
path policy is exact theorem-backed for the encoded exact state. -/
abbrev markov_path_exactTheoremBacked_on_support :=
  @FormalProofs.OPT.markovPath_exactTheoremBacked_on_support

/-- Exact-collapse sanity-check policy: one leaf containing the full Markov
document. -/
abbrev one_leaf_markov_path_policy :=
  @FormalProofs.OPT.oneLeafMarkovPathPolicy

/-- The one-leaf/full-document Markov policy is sound. -/
abbrev one_leaf_markov_path_policy_sound :=
  @FormalProofs.OPT.oneLeafMarkovPathPolicy_sound

/-- Exact Markov simulation contract for the one-leaf/full-document regime. -/
abbrev one_leaf_markov_path_exact_contract :=
  @FormalProofs.OPT.oneLeafMarkovPathExactContract

/-- Under the exact Markov simulation contract, every realized support tree
preserves the full exact Markov sketch state. -/
abbrev markov_path_state_exact_on_support_of_contract :=
  @FormalProofs.OPT.ExactMarkovPathSimulationContract.state_exact_on_support

/-- Under the exact Markov simulation contract, any downstream utility on the
exact Markov sketch state is preserved on support trees. -/
abbrev markov_path_state_utility_exact_on_support_of_contract :=
  @FormalProofs.OPT.ExactMarkovPathSimulationContract.state_utility_exact_on_support

/-- Under the exact Markov simulation contract, realized support trees preserve
the changepoint-count target exactly. -/
abbrev markov_path_changepoint_count_exact_on_support_of_contract :=
  @FormalProofs.OPT.ExactMarkovPathSimulationContract.changepoint_count_exact_on_support

/-- Count-only summaries cannot certify general topology claims on the Markov
benchmark. -/
abbrev markov_countOnly_not_exact_on_all_trees :=
  @FormalProofs.OPT.markov_countOnly_not_exact_on_all_trees

/-- Checked runtime audit artifacts on Markov support trees recover the exact
approximate-local-law bundle induced by their empirical certificate. -/
abbrev runtime_audited_markov_path_approx_bundle_on_support :=
  @FormalProofs.OPT.RuntimeAuditedMarkovPathSimulationContract.approx_bundle_eq_on_support

/-- Checked runtime audit artifacts on Markov support trees induce approximate
theorem-backedness for the encoded exact state. -/
abbrev runtime_audited_markov_path_approxTheoremBacked_on_support :=
  @FormalProofs.OPT.RuntimeAuditedMarkovPathSimulationContract.approxTheoremBacked_on_support

/-- Checked runtime audit artifacts compile to stochastic adaptive approximate
local laws for the encoded exact Markov state. -/
abbrev runtime_audited_markov_path_stochastic_approx_local_laws :=
  @FormalProofs.OPT.RuntimeAuditedMarkovPathSimulationContract.stochastic_approx_local_laws

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

/-- Finite-support discharge of the GRPO-RL expected-loss Lipschitz interface
from a primitive pointwise bound on `GRPORLLossPointwise`. -/
abbrev grpo_rl_expected_lipschitz_from_pointwise_finite :=
  @ExpectedGRPORLLossLipschitz_of_pointwise_finite

/-- Bundle-driven GRPO-PL quantitative gap interface. -/
abbrev grpo_pl_gap_bundle_interface :=
  @grpo_pl_gap_bundle

/-- Bundle-driven GRPO-RL quantitative gap interface. -/
abbrev grpo_rl_gap_bundle_interface :=
  @grpo_rl_gap_bundle

/-- GRPO-RL quantitative gap interface with the abstract expected-loss
Lipschitz assumption discharged from a primitive finite-support pointwise
Lipschitz hypothesis. -/
abbrev grpo_rl_gap_pointwise_interface :=
  @grpo_rl_gap_bounded_of_pointwise

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

/-!
## Information-Sufficiency Bridge

The information-theoretic bridge exported from the main theorem surface is
deliberately narrow. C-TreePO is formalized here as an **oracle-sufficient
reduction** for oracle-indexed tasks, not as a claim about generic lossless
compression of raw text.

The Lean-backed surface is:
- local laws induce oracle equality almost surely under the raw/summary joint law,
- this yields a.e. oracle factorization through the realized summary,
- oracle-indexed supervision then has zero task-relevant KLIC on that joint law,
- deterministic summary collisions across oracle-distinct points block decoding.

Full Shannon and mutual-information statements remain optional context rather
than part of the main theorem ladder.
-/

/-- Oracle-indexed conditional densities for task-relevant KLIC statements. -/
abbrev oracle_indexed_task_density :=
  @FormalProofs.OPT.OracleIndexedConditionalDensity

/-- Deterministic tree-policy bridge: local laws imply oracle equality a.e. under the
raw/summary joint law. -/
abbrev stochastic_local_laws_oracle_eq_ae :=
  @FormalProofs.OPT.jointTreeSummaryLaw_oracle_eq_ae_of_localLaws

/-- Deterministic tree-policy bridge: local laws imply oracle sufficiency a.e.
through summaries. -/
abbrev oracle_sufficiency_joint_law_ae :=
  @FormalProofs.OPT.jointTreeSummaryLaw_oracle_factorizationAE_of_localLaws

/-- Deterministic tree-policy bridge: score transport under the raw/summary
joint law. -/
abbrev score_transport_joint_law :=
  @FormalProofs.OPT.jointTreeSummaryLaw_score_transport_of_localLaws

/-- Deterministic tree-policy bridge: zero task-relevant KLIC for
oracle-indexed supervision. -/
abbrev zero_task_relevant_klic_joint_law_ae :=
  @FormalProofs.OPT.jointTreeSummaryLaw_taskRelevantKLIC_zero_ae_of_localLaws

/-- Stochastic tree-policy bridge: supportwise local laws imply oracle equality
a.e. under the induced raw/summary joint law. -/
abbrev stochastic_tree_policy_local_laws_oracle_eq_ae :=
  @FormalProofs.OPT.stochasticJointTreeSummaryLaw_oracle_eq_ae_of_localLaws

/-- Stochastic tree-policy bridge: supportwise local laws imply oracle
sufficiency a.e. through summaries. -/
abbrev oracle_sufficiency_stochastic_tree_policy_ae :=
  @FormalProofs.OPT.stochasticJointTreeSummaryLaw_oracle_factorizationAE_of_localLaws

/-- Stochastic tree-policy bridge: score transport under the induced
raw/summary joint law. -/
abbrev score_transport_stochastic_tree_policy :=
  @FormalProofs.OPT.stochasticJointTreeSummaryLaw_score_transport_of_localLaws

/-- Stochastic tree-policy bridge: zero task-relevant KLIC for oracle-indexed
supervision. -/
abbrev zero_task_relevant_klic_stochastic_tree_policy_ae :=
  @FormalProofs.OPT.stochasticJointTreeSummaryLaw_taskRelevantKLIC_zero_ae_of_localLaws

/-- Deterministic impossibility: a summary collision across oracle-distinct inputs blocks decoding. -/
abbrev summary_collision_impossibility :=
  @FormalProofs.OPT.no_oracle_decoder_of_summary_collision

/-- Markov task-facing sufficiency notion: the summary determines every
two-sided changepoint-count query. -/
abbrev markov_count_query_sufficient :=
  @FormalProofs.OPT.MarkovCountQuerySufficient

/-- Markov task-facing sufficiency forces exact-sketch equality on collisions. -/
abbrev markov_sufficiency_collision_implies_exact_sketch_eq :=
  @FormalProofs.OPT.markov_count_query_sufficient_collision_implies_exact_sketch_eq

/-- Markov task-facing sufficiency yields a decoder back to the exact sketch
`(count, first, last)`. -/
abbrev markov_sufficiency_has_exact_sketch_decoder :=
  @FormalProofs.OPT.markov_count_query_sufficient_has_decoder

/-- Markov count-only summaries are not sufficient for arbitrary topology /
context-sensitive changepoint-count queries. -/
abbrev markov_countOnly_not_sufficient :=
  @FormalProofs.OPT.markov_countOnly_not_query_sufficient

/-- Exact parent full-sketch supervision recovers Markov `L2/C3`. -/
abbrev markov_exact_parent_fullSketch_implies_L2 :=
  @FormalProofs.OPT.markov_exact_parent_fullSketch_implies_L2

/-- Exact leaves plus exact parent full sketches imply zero root distortion on
the Markov sketch. -/
abbrev markov_exact_leaf_and_parent_fullSketch_zero_root_distortion :=
  @FormalProofs.OPT.markov_exact_leaf_and_parent_fullSketch_zero_root_distortion

/-- Count-only parent supervision is not sufficient for Markov merge
correctness in general. -/
abbrev markov_parent_countOnly_not_sufficient :=
  @FormalProofs.OPT.markov_parent_countOnly_not_sufficient

/-- Positive node weights preserve the exact zero-loss optimum of nodewise
nonnegative Markov exact-sketch supervision. -/
abbrev markov_positive_weighted_nodewise_zero_iff :=
  @FormalProofs.OPT.positive_weighted_nodewise_zero_iff

/-- Clean disjoint-palette observed tokens deterministically recover the latent
Markov regime path. -/
abbrev piecewise_disjoint_palette_observed_tokens_recover_latent_path :=
  @FormalProofs.OPT.piecewise_disjoint_palette_observed_tokens_recover_latent_path

/-- Clean disjoint-palette observed tokens deterministically recover the exact
theorem-domain Markov sketch. -/
abbrev piecewise_disjoint_palette_observed_tokens_recover_exact_sketch :=
  @FormalProofs.OPT.piecewise_disjoint_palette_observed_tokens_recover_exact_sketch

/-- Clean disjoint-palette observed tokens recover the changepoint-count target
with zero Bayes error in the supportwise decoder sense. -/
abbrev piecewise_disjoint_palette_zero_bayes_error :=
  @FormalProofs.OPT.piecewise_disjoint_palette_zero_bayes_error

/-- Exact recovery of the theorem-domain Markov sketch by a learned
representation implies task-facing query sufficiency. -/
abbrev markov_representation_exact_recovery_implies_query_sufficient :=
  @FormalProofs.OPT.markov_representation_exact_recovery_implies_query_sufficient

/-- Exact recovery of the theorem-domain Markov sketch forces zero root
changepoint-count error. -/
abbrev markov_representation_exact_recovery_zero_root_count_error :=
  @FormalProofs.OPT.markov_representation_exact_recovery_zero_root_count_error

/-- Changepoint-count error is upper-bounded by discrete exact-sketch error. -/
abbrev markov_count_error_le_exact_sketch_error :=
  @FormalProofs.OPT.markov_count_error_le_exact_sketch_error

/-- Optional finite-support context: oracle log-cardinality is bounded by source log-cardinality. -/
abbrev oracle_log_card_le_source :=
  @FormalProofs.OPT.InformationTheory.OracleLogCard_le_SourceLogCard

end MainTheorems

end
