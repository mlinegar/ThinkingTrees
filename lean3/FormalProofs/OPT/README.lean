/-!
# OPT Module: Oracle Preference Training

## Overview

This module formalizes the theory of **oracle-preserving summarization** for preference learning.
The key insight is that **local testable conditions** (L1, L2, L3) on a summarizer imply
**global training equivalence** for DPO, GRPO, and other preference learning methods.

## Main Results

| Theorem | File | Statement |
|---------|------|-----------|
| `multi_round_proper` | ExpectationTheory | L1+L2+L3 ⟹ E[dist(f*(Z), f*(x))] = 0 |
| `dpo_equivalence` | PreferenceLearning | Zero distortion ⟹ DPO loss equivalent |
| `grpo_equivalence` | PreferenceLearning | Zero distortion ⟹ GRPO-PL loss equivalent |
| `grpo_rl_equivalence` | PreferenceLearning | Zero distortion ⟹ GRPO-RL loss equivalent |
| `dpo_gap_bounded` | PreferenceBounds | Quantitative gap: |L(X) - L(Z)| ≤ L × Δ |
| `grpo_pl_gap_bounded` | PreferenceBounds | GRPO-PL quantitative gap |
| `grpo_rl_gap_bounded` | PreferenceBounds | GRPO-RL quantitative gap |
| `unified_preference_gap_bounded` | PreferenceBounds | Unified framework for all gap bounds |
| `oracleRiskObjective` | RegularizedObjective | Population oracle risk = distortion + summary cost |
| `certifiedRegularizedObjective` | RegularizedObjective | Oracle risk + approximate local-law penalty |
| `oracleProjectionObjectiveWeights` | RegularizedObjective | Oracle/projection weights: `(1-λ)` oracle distortion and `λρ_i` local-law projection budgets |
| `oracleProjectionObjective` | RegularizedObjective | Paper-facing objective `(1-λ)L_oracle + λ∑ρ_i ε_i` |
| `oracleProjectionObjective_le_of_uniformApproxExactTheoremBacked` | RegularizedObjective | Neural-operator realization plus transfer assumptions yields a certified oracle/projection bound through the approximate-gap theorem |
| `certifiedRegularized_epsilonMinimizer_of_uniform_perturbation` | OptimizationPerturbation | Uniform oracle/objective error turns an exact certified regularized minimizer into a `2ε`-minimizer for the true objective |
| `certifiedRegularized_epsilonMinimizer_failure_prob_le_of_good_event` | OptimizationPerturbation | Confidence-event wrapper for certified-regularized minimizer transfer |
| `constrainedCertifiedRegularized_epsilonMinimizer_failure_prob_le_of_good_event` | OptimizationPerturbation | Confidence-event wrapper for constrained certified-regularized minimizer transfer |
| `frontierRegularizedObjectiveWeights` | RegularizedObjective | One-parameter frontier from legacy summary-only to law-only regularization |
| `certifiedRegularizedObjective_le_of_approx_bundle` | RegularizedObjective | Distortion term substituted by approximate local-law bundle |
| `ops_mergeClosed_of_global` | MergeableReduction | A1/A2/A3 imply merge closure in mergeable-summary form |
| `ops_hierarchical_mergeable_of_global` | MergeableReduction | A1/A2/A3 imply hierarchical mergeability on merge trees |
| `ops_reduction_to_classical_mergeable` | MergeableReduction | OPS reduction to classical mergeable summaries |
| `L3_of_reencodeExact` | HLLIdempotence | Exact sketch re-encoding implies OPS `L3` for the induced deterministic summary |
| `L3ε_of_onRangeViolationBound_deterministic` | HLLIdempotence | Audited on-range re-summary stability implies approximate OPS `L3ε` on any deterministic merge tree |
| `succMax_not_L3` | HLLIdempotence | Merge idempotence alone does not imply OPS `L3` |
| `hllRegisterOperator_merge_idempotent` | HLLIdempotence | HLL-style register-max merge is exactly idempotent |
| `hllRegisterOperator_L3` | HLLIdempotence | HLL register states satisfy OPS `L3` when theorem-domain objects are already register states |
| `ExactTheoremBacked.ofLocalLaws` | TheoremBackingAssumptions | Broadest exact sufficient interface: a `LocalLawsBundle` is enough for theorem-backed correctness |
| `ApproxTheoremBacked.ofApproxLocalLaws` | TheoremBackingAssumptions | Broadest approximate sufficient interface: an `ApproxLocalLawsBundle` is enough for approximate theorem-backed correctness |
| `SketchCodecExactAssumptions.toExactTheoremBacked` | TheoremBackingAssumptions | A supplied encode/merge/decode codec with exact sketch obligations induces exact theorem-backedness |
| `SketchCodecApproxAssumptions.toApproxTheoremBacked` | TheoremBackingAssumptions | A supplied encode/merge/decode codec with audited leaf/merge/idempotence budgets induces approximate theorem-backedness |
| `exactTheoremBacked_of_globalPreservation` | TheoremBackingAssumptions | The stronger global `A1/A2/A3` route compiles to exact theorem-backedness on any tree |
| `approxLocalLawsBundle_of_uniformApproxExactTheoremBacked` | NeuralOperatorTheoremBridge | Uniform neural-operator approximation plus explicit transfer assumptions compiles to an `ApproxLocalLawsBundle` |
| `approxTheoremBacked_of_uniformApproxExactTheoremBacked` | NeuralOperatorTheoremBridge | Exact theorem-backedness for an ideal summarizer plus uniform neural-operator approximation yields approximate theorem-backedness for the realized operator |
| `Equation6NeuralOperator` | ML.NeuralOperatorArchitecture | Finite equation-(6)-style neural-operator architecture surface |
| `Equation6UniversalApproxUniform` | ML.NeuralOperatorArchitecture | Uniform compact-set approximation interface for equation-(6) neural operators |
| `Equation6UniversalApproxL2` | ML.NeuralOperatorArchitecture | Expected-`L²` approximation interface for equation-(6) neural operators |
| `singleHeadAttention_eq_discretizedKernelLayer` | ML.TransformerAsNeuralOperator | Kovachki-Proposition-6 surface: single-head attention is the associated discretized kernel layer |
| `transformerEncoder_is_equation6NeuralOperator` | ML.TransformerAsNeuralOperator | Finite transformer encoder stacks instantiate equation-(6) neural operators |
| `MergeableSketchSummaryClass` | NeuralOperatorSpaces | Operators induced by mergeable sketches with exact sketch-local witnesses |
| `ExactLocalLawNeuralOperators` | NeuralOperatorSpaces | Exact C1/C2/C3 subspace inside a chosen neural-operator class |
| `ApproxLocalLawNeuralOperators` | NeuralOperatorSpaces | Approximate C1/C2/C3 subspace inside a chosen neural-operator class |
| `localLawWeightsAreProjection_iff_approximationErrorStructuredByLocalLaws` | NeuralOperatorSpaces | Local-law weights are a projection iff the approximation-error zero set is exactly the local-law subspace |
| `localLawWeightsAreProjectionOn_iff_approximationErrorStructuredByLocalLawsOn` | NeuralOperatorSpaces | Class-restricted iff between projection weights and structured approximation error |
| `mergeableSketchSummaryClass_subset_exactLocalLawSubspace` | NeuralOperatorSpaces | Mergeable sketches with exact sketch-local witnesses induce exact local-law membership |
| `mergeableSketch_overlap_subset_exactLocalLawNeuralOperators` | NeuralOperatorSpaces | The mergeable-sketch/neural-operator overlap lies inside the exact local-law subspace |
| `ClassicalSketchLocalLaws` | ClassicalSketchLocalLaws | Unified status/crosswalk for HLL, Count-Min, KLL/GK, bigram, and Markov-count sketches |
| `exactTheoremBacked_nonempty_iff_supportExactTheoremBacked` | TheoremBackingStructure | Existence of an exact theorem-backed witness is equivalent to zero oracle distortion on every realized support event |
| `exactTheoremBackedAllTrees_iff_A1_A2_of_A3` | TheoremBackingStructure | For deterministic summaries, exact theorem-backedness on all trees collapses to `A1 ∧ A2` once `A3` is supplied |
| `exactTheoremBackedAllTrees_iff_globalAssumptions` | TheoremBackingStructure | Exact theorem-backedness on all trees plus `A3` is equivalent to the full global `A1/A2/A3` regime |
| `SketchCodecExactAssumptions.toDirectSummaryExactAssumptions` | TheoremBackingStructure | Exact sketch/codec assumptions are a special case of the broad direct-summary exact interface |
| `SketchCodecApproxAssumptions.toDirectSummaryApproxAssumptions` | TheoremBackingStructure | Approximate sketch/codec assumptions are a special case of the broad direct-summary approximate interface |
| `sketchCodecExactAssumptions_imply_classical_mergeable` | TheoremBackingStructure | Under `A3`, exact sketch/codec assumptions induce the classical mergeable-summary interface |
| `expected_loss_eq_via_ZR_of_exactTheoremBacked` | TheoremBackingConsequences | Exact theorem-backedness is enough for any oracle-measurable expected loss on `PMF.pure x` versus `ZR` |
| `expected_pref_loss_prog_eq_via_ZR_of_exactTheoremBacked` | TheoremBackingConsequences | Exact theorem-backedness is enough for nested oracle-indexed preference programs |
| `dpo_equivalence_via_ZR_of_exactTheoremBacked` | TheoremBackingConsequences | DPO expected loss is preserved under exact theorem-backed reduction |
| `dpo_exact_metric_via_ZR_of_exactTheoremBacked` | TheoremBackingConsequences | DPO oracle-measurable argmins are preserved under exact theorem-backed reduction |
| `dpo_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement` | OptimizationPerturbation | Uniform oracle error turns exact DPO oracle-argmins on `ZR` into `2ε`-optimal policies for the true objective |
| `dpo_true_epsilon_argmin_failure_prob_le_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement` | OptimizationPerturbation | High-probability exact-DPO corollary: optimizer selections that are surrogate argmins on a good event fail true `2ε`-optimality with no more probability than that event fails |
| `oracleMeasurableParamArgmin_subset_epsilonArgmin_of_uniform_two_stage_loss_perturbation` | OptimizationPerturbation | Generic two-stage perturbation calculus: truth-to-oracle error plus oracle-to-surrogate transport gives near-optimal surrogate argmins |
| `oracleMeasurableParamArgmin_subset_pointwiseEpsilonArgmin_of_expectedTree_loss_perturbation` | OptimizationPerturbation | Generic expected-tree perturbation calculus: small expected absolute tree-gap transfers argmins of the expected tree objective to near-optimality for truth |
| `oracleMeasurableParamArgmin_failure_prob_le_of_good_event_expectedTree_pointwiseTransfer` | OptimizationPerturbation | Generic confidence-event wrapper for expected-tree pointwise argmin transfer |
| `dpo_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_approxBundle_and_uniformOracleMeasurement` | OptimizationPerturbation | Approximate local-law bundle plus a uniform policy-class Lipschitz envelope gives near-optimal DPO argmins on `ZR` |
| `dpo_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_nodewiseEmpiricalAudit_and_uniformOracleMeasurement` | OptimizationPerturbation | On a nodewise empirical audit confidence event, DPO argmins on `ZR` are near-optimal for the true objective |
| `dpo_expected_tree_argmin_subset_true_pointwiseEpsilonArgmin_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement` | OptimizationPerturbation | Under stochastic adaptive approximate local laws, argmins of the expected tree-DPO objective are near-optimal for the true objective with expected oracle and transport slack |
| `dpo_expected_tree_true_pointwiseEpsilonArgmin_failure_prob_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement` | OptimizationPerturbation | High-probability tree-level DPO optimizer transfer for expected stochastic-adaptive tree objectives |
| `grpo_pl_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement` | OptimizationPerturbation | Exact theorem-backed reduction plus uniform oracle error turns GRPO-PL oracle argmins on `ZR` into near-optimal policies for the true objective |
| `grpo_pl_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_approxBundle_and_uniformOracleMeasurement` | OptimizationPerturbation | Approximate local-law bundle plus a class-level GRPO-PL Lipschitz envelope gives near-optimal oracle argmins on `ZR` |
| `grpo_pl_true_epsilon_argmin_failure_prob_le_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement` | OptimizationPerturbation | High-probability exact-GRPO-PL corollary for selected surrogate argmins |
| `grpo_pl_expected_tree_argmin_subset_true_pointwiseEpsilonArgmin_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement` | OptimizationPerturbation | Tree-level GRPO-PL optimizer transfer for expected stochastic-adaptive tree objectives with oracle uncertainty |
| `grpo_pl_expected_tree_true_pointwiseEpsilonArgmin_failure_prob_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement` | OptimizationPerturbation | High-probability tree-level GRPO-PL optimizer transfer for expected stochastic-adaptive tree objectives |
| `grpo_rl_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement` | OptimizationPerturbation | Exact theorem-backed reduction plus uniform oracle error turns GRPO-RL oracle argmins on `ZR` into near-optimal policies for the true objective |
| `grpo_rl_oracle_argmin_subset_true_epsilonArgmin_via_ZR_of_approxBundle_and_uniformOracleMeasurement` | OptimizationPerturbation | Approximate local-law bundle plus a class-level GRPO-RL Lipschitz envelope gives near-optimal oracle argmins on `ZR` |
| `grpo_rl_true_epsilon_argmin_failure_prob_le_via_ZR_of_exactTheoremBacked_and_uniformOracleMeasurement` | OptimizationPerturbation | High-probability exact-GRPO-RL corollary for selected surrogate argmins |
| `grpo_rl_expected_tree_argmin_subset_true_pointwiseEpsilonArgmin_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement` | OptimizationPerturbation | Tree-level GRPO-RL optimizer transfer for expected stochastic-adaptive tree objectives with oracle uncertainty |
| `grpo_rl_expected_tree_true_pointwiseEpsilonArgmin_failure_prob_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement` | OptimizationPerturbation | High-probability tree-level GRPO-RL optimizer transfer for expected stochastic-adaptive tree objectives |
| `grpo_equivalence_via_ZR_of_exactTheoremBacked` | TheoremBackingConsequences | GRPO-Plackett-Luce expected loss is preserved under exact theorem-backed reduction |
| `grpo_rl_equivalence_via_ZR_of_exactTheoremBacked` | TheoremBackingConsequences | GRPO-RL expected loss is preserved under exact theorem-backed reduction |
| `expected_loss_eq_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature` | TheoremBackingMeasurementError | If the oracle identifies a latent feature, exact theorem-backedness transports any feature-indexed oracle-measurable loss exactly |
| `expected_pref_loss_eq_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature` | TheoremBackingMeasurementError | Feature-indexed preferences are preserved exactly when the latent feature is oracle-identified |
| `expected_feature_utility_with_measurement_error_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature` | TheoremBackingMeasurementError | Exact theorem-backed transport plus noisy latent-state observation yields a pure measurement-error bound |
| `feature_distortion_le_of_featureLipschitzFromOracle` | TheoremBackingApproxMeasurementError | A latent feature Lipschitz in the oracle inherits expected-distortion control from the oracle distortion |
| `expected_feature_utility_with_measurement_error_via_ZR_of_approxTheoremBacked_and_featureLipschitz` | TheoremBackingApproxMeasurementError | Approximate theorem-backed transport plus latent-state measurement error yields a transport term plus a pure measurement-error term |
| `ApproxOracleRecoversFeature` | ApproxOracleRecovery | Relation-first approximate oracle recovery: same oracle fiber implies the learned feature lands in an `ε`-ball |
| `ApproxOracleRecoversFeatureOn` | ApproxOracleRecovery | Covered-pair version of approximate oracle recovery for sparse labeled-pair supervision |
| `approxReadoutFactorsThroughFeature_fiber_bound` | LipschitzReadoutFactorization | Approximate factorization bounds how much a readout can vary inside one learned theorem-feature fiber |
| `oracleRecoversFeatureOn_of_zero_contrastive_risk` | FiberPreservingObjective | Zero contrastive risk on a covered same-fiber pair distribution forces exact oracle-feature recovery on the covered relation |
| `SameOracleFiber` | OracleFiberRelations | The primitive equivalence relation: two inputs have the same oracle value |
| `zr_support_sameOracleFiber_of_exactTheoremBacked` | OracleFiberRelations | Exact theorem-backed reductions stay on the same oracle fiber as the original document |
| `sameFeatureFiber_iff_encodedOracle_zero` | FeatureFiberLaws | Exact encoded-feature zero distortion is equivalent to remaining on one theorem-feature fiber |
| `leaf_support_sameFeatureFiber_of_exactTheoremBacked` | FeatureFiberLaws | Exact theorem-backed leaves remain inside one oracle-identified theorem-feature fiber |
| `merge_support_sameFeatureFiber_of_exactTheoremBacked` | FeatureFiberLaws | Exact theorem-backed internal reductions remain inside one oracle-identified theorem-feature fiber |
| `expected_featureFiberDistortion_le_of_featureLipschitzFromOracle` | FeatureFiberLaws | Approximate feature-fiber distortion is controlled by oracle distortion through a Lipschitz theorem feature |
| `expected_sameFeatureClassUtility_eq_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature` | FeatureClassObjectives | Hard same-class objectives on a learned theorem feature transport exactly |
| `expected_sameFeatureClassUtility_with_measurement_error_via_ZR_of_approxTheoremBacked_and_featureLipschitz` | FeatureClassObjectives | Same-class feature objectives admit the standard approximate transport bound once their utility is regular enough |
| `sameReadoutSurface_implies_factorsThroughFeature` | ReadoutAlignment | Same-surface root supervision is the minimal theory-aligned special case |
| `pairedReadoutFactorsThroughFeature` | ReadoutAlignment | Two simultaneous heads are theory-aligned whenever both factor through the same theorem feature |
| `not_readoutFactorsThroughFeature_of_distinguished_feature_fibers` | ReadoutAlignment | A root head that separates theorem-feature fibers is not justified by the theorem route |
| `expected_loss_eq_via_ZR_of_exactTheoremBacked_and_factoredReadout` | ReadoutAlignment | Any root loss indexed by a readout that factors through the theorem feature transports exactly |
| `supervisedReadoutLoss_eq_via_ZR_of_exactTheoremBacked_and_sameSurface` | ReadoutAlignment | Same-surface supervised root learning transports exactly under exact theorem-backedness |
| `paired_approxOracleRecoversReadouts_of_sharedFeature` | SharedFeatureMultihead | One shared learned theorem feature can support two approximate downstream heads with separate quantitative recovery bounds |
| `paired_approxOracleRecoversReadoutsOn_of_sharedFeature` | SharedFeatureMultihead | Covered-pair version of simultaneous approximate recovery for task and summary heads through one shared theorem feature |
| `zr_support_paired_approxReadoutBound_of_exactTheoremBacked_and_sharedFeature` | SharedFeatureMultihead | Exact theorem-backed reductions inherit simultaneous approximate stability for both task and summary heads |
| `paired_approxReadoutBound_on_coveredSameOracleFiber_of_zero_contrastiveRisk` | SharedFeatureMultihead | Zero covered-pair contrastive risk plus approximate factorization yields simultaneous task/summary stability on the covered oracle fibers |
| `expected_labelScoreUtility_eq_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature` | LabelScoreObjectives | Arbitrary real-valued scores on decoded labels transport exactly through an oracle-identified theorem feature |
| `expected_labelScoreUtility_with_measurement_error_via_ZR_of_approxTheoremBacked_and_featureLipschitz` | LabelScoreObjectives | Arbitrary decoded-label scores inherit the standard approximate transport-plus-measurement-error bound |
| `UniformOracleApproximation` | TwoStageOracleSurrogate | Stage-1 learned-oracle surrogate stays uniformly within `ε` of the true oracle |
| `sameSurrogateFiber_implies_trueOracleClose_of_uniformOracleApproximation` | TwoStageOracleSurrogate | Equality under the learned surrogate oracle implies `2ε` closeness under the true oracle |
| `Δ_R_ZR_true_le_of_exactTheoremBacked_on_surrogate_and_uniformOracleApproximation` | TwoStageOracleSurrogate | Exact theorem-backedness for the learned surrogate oracle yields expected true-oracle distortion at most `2ε` |
| `expected_trueOracleUtility_bound_via_ZR_of_exactTheoremBacked_on_surrogate_and_uniformOracleApproximation` | TwoStageOracleSurrogate | Any Lipschitz true-oracle utility inherits the exact-surrogate `2ε` transport bound |
| `Δ_R_ZR_true_le_of_approxTheoremBacked_on_surrogate_and_uniformOracleApproximation` | TwoStageOracleSurrogate | Approximate theorem-backedness for the surrogate oracle yields true-oracle distortion bounded by surrogate transport budget plus `2ε` |
| `expected_trueOracleUtility_bound_via_ZR_of_approxTheoremBacked_on_surrogate_and_uniformOracleApproximation` | TwoStageOracleSurrogate | Any Lipschitz true-oracle utility inherits the approximate-surrogate budget: audited local-law transport plus additive surrogate slack |
| `expected_trueLabelScoreUtility_bound_via_ZR_of_exactTheoremBacked_on_surrogate_and_uniformOracleApproximation` | TwoStageLabelScoreObjectives | The teacher-first exact-surrogate route transfers arbitrary true-oracle label scores with additive `2ε` stage-1 slack |
| `expected_trueLabelScoreUtility_bound_via_ZR_of_approxTheoremBacked_on_surrogate_and_uniformOracleApproximation` | TwoStageLabelScoreObjectives | The teacher-first approximate-surrogate route transfers arbitrary true-oracle label scores with surrogate transport budget plus `2ε` |
| `expected_labelScoreUtility_bound_in_surrogateFeatureSpace` | TwoStageLabelScoreObjectives | Stage-2 transport in the learned surrogate feature space for arbitrary scores on decoded labels |
| `expected_labelScoreUtility_two_stage_end_to_end_bound` | TwoStageLabelScoreObjectives | Full layered two-stage label-score bound with explicit transport, fiber, measurement, and substitution terms |
| `scoreReadoutFactorsThroughProductScoreFiber_firstCoordinate` | ProductScoreFiber | In the explicit score-fiber product state, the scalar score head is the exact first-coordinate readout |
| `paired_approxOracleRecoversReadouts_of_productScoreFiber` | ProductScoreFiber | A product-state `Score × FiberState` supports an exact scalar score head and an approximate summary head simultaneously |
| `expected_labelScoreUtility_two_stage_end_to_end_bound_of_productScoreFiber` | ProductScoreFiber | The full two-stage label-score decomposition specialized to the explicit product-state score-fiber route |
| `two_stage_breakeven_condition` | TwoStageDecomposition | Two-stage beats single-stage when stage-2 savings exceed stage-1 substitution cost and extra slack |
| `distillation_chain_error` | TwoStageDecomposition | Additional distillation stages multiply upstream approximation error by downstream Lipschitz factors |
| `approxOracleRecoversFeature_of_distillation_chain` | TwoStageDecomposition | A downstream distilled representation remains approximately oracle-recovering, with error scaled by the downstream Lipschitz factor |
| `contractive_distillation_chain` | TwoStageDecomposition | Contractive intermediate stages do not amplify upstream approximation error |
| `approxOracleRecoversFeature_of_contractive_distillation_chain` | TwoStageDecomposition | Contractive distillation preserves the original approximate oracle-recovery budget |
| `approxOracleRecoversFeature_of_twoStageOracleApproximation_and_distillation` | TwoStageDecomposition | A packaged stage-1 learned surrogate can be pushed through one more Lipschitz distillation layer while retaining a direct approximate oracle-recovery guarantee |

The intended tree-neural implementation of this readout-alignment class is a
shared learned theorem feature `φ(z)`: all theorem-facing/root heads read only
from `φ`, even though the underlying latent state and merger remain generic.
The `shared_feature` and `shared_feature_adapters` routes are direct
implementations of this class. This is more general than a fixed slotwise
theorem surface, but it stays inside the same theory class because every
supervised readout is required to factor through the same learned feature.

The newer feature-fiber files make the intended C2 reading explicit: the
theorem-facing object is the learned feature `Φ`, and the relevant invariance is
membership in a `Φ`-equivalence class rather than raw latent-state equality.
This is the formal handle for `shared_feature`, `shared_feature_adapters`, and
multi-head factored readout routes.

The newer oracle-fiber files push this one step further: before picking any
particular `Φ`, the primitive object is the oracle equivalence relation itself.
Learned theorem features, contrastive same/different objectives, and paired
factored heads are all then read as ways of realizing or approximately
respecting that relation.

The new two-stage surrogate-oracle file captures the practical teacher-first
workflow explicitly: first learn or query an expensive oracle surrogate `f̂`,
then learn summaries relative to `f̂`. The formal price of doing so is clean and
additive. If `f̂` is uniformly within `ε` of the true oracle `f*`, then exact
surrogate theorem-backedness gives true-oracle distortion at most `2ε`, while
approximate surrogate theorem-backedness adds the audited local-law transport
budget on top of that same `2ε` stage-1 slack.

The layered two-stage label-score route generalizes this again. Instead of
stopping at a surrogate oracle, it exposes the full decomposition for arbitrary
scores on decoded labels: stage-2 transport in the learned surrogate space,
stage-2 fiber error, measurement error at the root, and a stage-1 substitution
cost between the stage-2 learned feature and the reference feature you actually
care about. The breakeven and distillation lemmas make the tradeoff explicit:
two-stage helps only when the stage-2 budget savings exceed the stage-1
substitution cost, and extra distillation layers are safe only to the extent
their intermediate maps are contractive or low-Lipschitz.

The explicit product-state score-fiber file fixes one especially practical
specialization of that route: a learned feature of the form `Score × FiberState`,
where `Score` is a bounded one-dimensional score coordinate. The first
coordinate is reserved for the scalar oracle score, while the full product state
is still available to summary and auxiliary heads. This is the formal target
for the `factorized_score_fiber` implementation route.

| `same_route_two_targets_force_oracle_equiv` | SketchFlipMergeBridge | Requiring one merge route to preserve two targets forces oracle-equivalence |
| `local_laws_bundle_of_sketch` | SketchSummaryOperators | Sketch-level assumptions imply `L1/L2/L3` for the induced deterministic summary operator |
| `multi_round_typeclass_of_sketch` | SketchRecovery | Sketch assumptions recover multi-round zero-distortion theorem |
| `sketchReduce_countSketch_eq_bagOfWords` | BagOfWordsLDARecovery | Exact tree reduction with count sketches recovers the full bag-of-words histogram |
| `histogramUtility_exact_on_tree` | BagOfWordsLDARecovery | Any histogram-based document utility is exactly preserved by count-sketch tree reduction |
| `ldaDocumentLikelihood_exact_on_tree` | BagOfWordsLDARecovery | Ordinary bag-of-words LDA document likelihood is exactly preserved by count-sketch tree reduction |
| `linearUtility_weightedMean_eq_sum` | LeafLocalMixtureUtilityGap | The linear local-mixture utility commutes exactly with leaf averaging |
| `affineQuadratic_gap_eq_quadratic_gap` | LeafLocalMixtureUtilityGap | For `h(π)=θᵀπ+λπᵀWπ`, the pooled-vs-leaf gap is exactly the quadratic gap scaled by `λ` |
| `featureIndexedObjective_eq_of_zero_dist` | ExactUtilityTransport | Any oracle-indexed feature/state objective transports under zero distortion |
| `supervisedStateExpectedLoss_eq_of_zero_dist` | ExactUtilityTransport | Direct supervised-state learning is a zero-distortion special case |
| `normalizedErrorUtility_zero_regret_iff_zero_error` | ExactUtilityTransport | For normalized exact-state utilities, zero regret iff zero state error |
| `mergeableStateUtility_exact_on_tree` | ExactUtilityTransport | Any utility on an exact mergeable latent state is preserved exactly by the tree |
| `markovStateUtility_exact_on_tree` | ExactUtilityTransportInstances | Any utility on the exact Markov state is preserved exactly |
| `markov_path_state_exact_on_tree` | MarkovPathDGP | Any utility on the exact Markov sketch state is preserved exactly for trees of realized regime sequences |
| `markov_path_count_exact_on_tree` | MarkovPathDGP | The exact Markov path encoder preserves changepoint count exactly on every tree |
| `markov_path_stochastic_policy_local_laws` | MarkovSimulationValidation | Any stochastic policy over realized Markov paths inherits exact local laws on each support tree for the encoded exact state |
| `markov_path_state_exact_on_support_of_contract` | MarkovSimulationValidation | Under a sound stochastic Markov path policy, every realized support tree preserves the full exact Markov sketch state |
| `markov_path_changepoint_count_exact_on_support_of_contract` | MarkovSimulationValidation | Under a sound stochastic Markov path policy, every realized support tree preserves the changepoint-count target exactly |
| `nodeIndexedStateEval_eq_feature_of_exact` | NodeIndexedLatentState | Exact node-indexed leaf/merge families recover the canonical latent span state by tree induction |
| `nodeIndexedStateUtility_exact_on_tree` | NodeIndexedLatentState | Any readout of an exact node-indexed latent-state family is preserved on the tree |
| `nodeIndexedStateUtility_exact_on_tree_of_mergeable_feature` | NodeIndexedLatentState | Node-indexed exact families collapse to the same exact mergeable-state route when their realized operators are coherent |
| `expected_nodeIndexedStateUtility_bound_of_exactNodeIndexed_and_approxBacked` | NodeIndexedLatentState | Exact node-indexed latent-state recovery plus approximate theorem-backed text transport yields a root-state utility bound |
| `complementarityStateUtility_exact_on_tree` | ExactUtilityTransportInstances | Any utility on the exact complementarity state is preserved exactly |
| `topicSketchUtility_exact_on_tree` | ExactUtilityTransportInstances | Any utility on the exact topic mass + boundary state is preserved exactly |
| `topicOracleFromSketch_exact_on_tree` | ExactUtilityTransportInstances | The boundary-sensitive topic oracle score is exactly preserved on the tree |
| `local_laws_of_identity_encoded_feature` | SketchRecovery | One-line template: identity sketch + encoded feature ⇒ local laws |
| `preference_learning_equivalence_via_ZR_of_identity_encoded_feature` | SketchRecovery | One-line template: identity sketch + encoded feature ⇒ pairwise equivalence via `ZR` |
| `local_laws_of_paired_encoded_feature` | SketchRecovery | One-line template: paired non-identity sketch + encoded feature ⇒ local laws |
| `preference_learning_equivalence_via_ZR_of_paired_encoded_feature` | SketchRecovery | One-line template: paired non-identity sketch + encoded feature ⇒ pairwise equivalence via `ZR` |
| `dpo_equivalence_of_sketch` | SketchRecovery | Sketch assumptions recover DPO equivalence on `PMF.pure x` vs `ZR` |
| `grpo_equivalence_via_ZR_of_sketch` | SketchRecovery | Sketch assumptions recover GRPO-PL equivalence via `ZR` |
| `grpo_rl_equivalence_via_ZR_of_sketch` | SketchRecovery | Sketch assumptions recover GRPO-RL equivalence via `ZR` |
| `markov_local_laws_of_encoded_feature` | SketchRecoveryInstances | Concrete Markov encoded-feature instantiation |
| `markov_path_local_laws_of_encoded_state` | MarkovPathDGP | DGP-level Markov path support instantiation: exact path encoder gives exact local laws |
| `markov_countOnly_mergeFold_counterexample` | MarkovPathDGP | Concrete tree-level counterexample showing count-only summaries are not compositionally sufficient |
| `markov_countOnly_not_exact_on_all_trees` | MarkovSimulationValidation | Count-only summaries cannot certify general topology claims on the Markov benchmark |
| `topic_local_laws_of_encoded_feature` | SketchRecoveryInstances | Concrete topic-bigram encoded-feature instantiation |
| `length_local_laws_of_encoded_feature` | SketchRecoveryInstances | Concrete list-length encoded-feature instantiation |
| `length_local_laws_of_paired_encoded_feature` | SketchRecoveryInstances | Concrete list-length instantiation on paired non-identity sketch |
| `length_local_laws_of_lossy_encoded_feature` | SketchRecoveryInstances | Concrete list-length instantiation on genuinely lossy length sketch |
| `length_preference_equivalence_via_ZR_of_lossy_encoded_feature` | SketchRecoveryInstances | Pairwise equivalence via `ZR` for encoded length under lossy sketch |
| `length_dpo_gap_of_stochastic_adaptive_approx_with_oracleMeasurement` | SketchRecoveryInstances | Lossy-length DPO support-tree gap with explicit oracle-measurement term |
| `length_grpo_pl_gap_of_stochastic_adaptive_approx_with_oracleMeasurement` | SketchRecoveryInstances | Lossy-length GRPO-PL support-tree gap with explicit oracle-measurement term |
| `length_grpo_rl_gap_of_stochastic_adaptive_approx_with_oracleMeasurement` | SketchRecoveryInstances | Lossy-length GRPO-RL support-tree gap with explicit oracle-measurement term |
| `L1ε_of_nodewise` | ApproximateLocalLaws | Nodewise leaf budgets imply aggregate `L1ε` budget law |
| `L2ε_of_nodewise` | ApproximateLocalLaws | Nodewise merge budgets imply aggregate `L2ε` budget law |
| `approx_bundle_of_nodewise` | ApproximateLocalLaws | Nodewise leaf/merge laws + idempotence budget package into `ApproxLocalLawsBundle` |
| `Δ_R_ZR_le_of_approx_local_laws` | ApproximateLocalLaws | Approximate local budgets imply quantitative distortion bound |
| `Δ_R_ZR_le_of_approx_bundle` | ApproximateLocalLaws | Bundle-driven quantitative distortion bound |
| `dpo_gap_via_approx_local_laws` | ApproximateLocalLaws | DPO gap bound under approximate local budgets |
| `dpo_gap_via_approx_bundle` | ApproximateLocalLaws | Bundle-driven DPO gap bound |
| `grpo_pl_gap_via_approx_local_laws` | ApproximateLocalLaws | GRPO-PL gap bound under approximate local budgets |
| `grpo_rl_gap_via_approx_local_laws` | ApproximateLocalLaws | GRPO-RL gap bound under approximate local budgets |
| `multi_round_typeclass_of_adaptive` | AdaptiveChunkingBridge | Adaptive tree policy bridge to fixed-tree multi-round theorem |
| `dpo_equivalence_of_adaptive` | AdaptiveChunkingBridge | Adaptive tree policy bridge to DPO equivalence |
| `multi_round_typeclass_of_stochastic_adaptive` | AdaptiveChunkingBridge | Stochastic adaptive policy bridge on support trees |
| `dpo_equivalence_of_stochastic_adaptive` | AdaptiveChunkingBridge | Stochastic adaptive DPO bridge on support trees |
| `Δ_R_ZR_le_of_stochastic_adaptive_approx_local_laws` | AdaptiveChunkingBridge | Stochastic adaptive approximate local-law bound on support trees |
| `runtime_audited_markov_path_stochastic_approx_local_laws` | MarkovSimulationValidation | Checked runtime nodewise audit artifacts compile to stochastic adaptive approximate local laws for the encoded exact Markov state |
| `Exp_loss_gap_le_of_stochastic_adaptive_oracleMeasurement` | AdaptiveChunkingBridge | Expected stochastic-adaptive oracle-indexed gap lifts to a true-target gap by adding one oracle-measurement term |
| `Exp_loss_gap_le_of_stochastic_adaptive_pointwiseOracleMeasurement` | AdaptiveChunkingBridge | Tree-indexed oracle-measurement version of the stochastic-adaptive expected-gap lift |
| `Exp_dpo_gap_le_of_stochastic_adaptive_approx_local_laws_with_oracleMeasurement` | AdaptiveChunkingBridge | Stochastic adaptive expected DPO gap bound with oracle measurement |
| `Exp_dpo_gap_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement` | AdaptiveChunkingBridge | Stochastic adaptive expected DPO gap bound with tree-indexed oracle measurement |
| `Exp_grpo_pl_gap_le_of_stochastic_adaptive_approx_local_laws_with_oracleMeasurement` | AdaptiveChunkingBridge | Stochastic adaptive expected GRPO-PL gap bound with oracle measurement |
| `Exp_grpo_pl_gap_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement` | AdaptiveChunkingBridge | Stochastic adaptive expected GRPO-PL gap bound with tree-indexed oracle measurement |
| `Exp_grpo_rl_gap_le_of_stochastic_adaptive_approx_local_laws_with_oracleMeasurement` | AdaptiveChunkingBridge | Stochastic adaptive expected GRPO-RL gap bound with oracle measurement |
| `Exp_grpo_rl_gap_le_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement` | AdaptiveChunkingBridge | Stochastic adaptive expected GRPO-RL gap bound with tree-indexed oracle measurement |
| `expected_group_loss_lipschitz_of_pointwise` | RUMSufficientConditions | Discharge expected-group Lipschitz from stronger pointwise Lipschitz |
| `expected_group_loss_lipschitz_of_pointwise_finite` | RUMSufficientConditions | Finite-index wrapper that discharges summability automatically |
| `AdversarialChunkingInstance.failure_bound` | AdversarialChunkingExample | Self-normalized adversarial chunking failure bound (non-uniform WOR) |
| `training_path_gap_bound_abstract_with_oracleMeasurement` | TrainingPipeline | Abstract two-stage teacher→student gap composition with separate oracle-measurement terms |
| `training_path_gap_bound_with_oracleMeasurement` | TrainingPipeline | DPO two-stage gap composition with oracle measurement |
| `training_path_epsilon_optimal_with_oracleMeasurement` | TrainingPipeline | Oracle-optimal teacher plus noisy two-stage gap bound implies an epsilon-optimal student |
| `training_path_epsilon_optimal_failure_prob_le_with_oracleMeasurement` | TrainingPipeline | Confidence-event wrapper: if the noisy two-stage certificate holds on a good event, failure of the student's `ε(ω)`-optimality statement is bounded by that event's failure probability |
| `epsilonOptimal_failure_prob_le_of_good_event` | TrainingPipeline | Generic confidence-event wrapper for epsilon-optimality statements |
| `discountedTreeMetaLoss_eq_neg_discountedTreeReturn_of_negated` | DiscountedTreeMetaObjective | Tree-depth discounting is exactly the RL discounted-return objective with reward `-loss` |
| `discountedTreeMetaLoss_sandwich` | DiscountedTreeMetaObjective | For nonnegative node losses and `0 ≤ γ ≤ 1`, the discounted tree objective lies between root-only supervision and the full undiscounted node sum |
| `expectedIPWDiscountedDocumentObjective_eq_fullDiscountedDocumentObjective` | DiscountedIPWObjective | HT/IPW-corrected depth-discounted objectives remain unbiased under constant marginal inclusion probabilities |
| `fullWeightedDocumentObjective_eq_fullSupervisionTreeObjectiveFn` | DiscountedIPWObjective | The current root/C1/C2/C3 weighting scheme is a special case of the generic weighted-IPW surface |
| `grpo_training_path_gap_bound_with_oracleMeasurement` | TrainingPipeline | GRPO two-stage gap composition with oracle measurement |

## File Structure

```
OPT/
├── CoreDefinitions.lean      # Basic types: BinTree, Summarizer, Policy
├── OracleMeasurable.lean     # Oracle-measurable predicates (lightweight)
├── PreferenceNoise.lean      # Abstract preference noise models
├── SamplingModel.lean        # Generative model for preference datasets
├── TreeProperties.lean       # Tree operations and counting lemmas
├── LocalLaws.lean            # L1, L2, L3 local consistency conditions
├── PreservationTheorems.lean # Tree reduction preserves oracle
├── ExpectationTheory.lean    # Multi-round preservation (main CLT-style result)
├── GlobalAssumptions.lean    # Global A1, A2, A3 and derivations
├── MergeableReduction.lean   # OPS → mergeable-summary bridge
├── HLLIdempotence.lean       # HLL-style register-max idempotence and the bridge to OPS `L3`
├── TheoremBackingAssumptions.lean # Broadest exact/approx theorem-backed interfaces + direct/sketch/global routes
├── TheoremBackingStructure.lean # Support-level characterization + exact all-tree collapse to A1/A2/A3 + sketch special-case structure
├── TheoremBackingConsequences.lean # Exact theorem-backedness ⇒ multi-round zero distortion + DPO/GRPO/preference-program equivalence
├── TheoremBackingMeasurementError.lean # Oracle-identified latent states + exact theorem-backed transport + noisy-state measurement-error bridge
├── TheoremBackingApproxMeasurementError.lean # Approximate theorem-backedness + oracle-to-feature Lipschitz map + latent-state measurement-error bound
├── ApproxOracleRecovery.lean   # ε-relaxation of oracle feature recovery on same-oracle-fiber pairs
├── LipschitzReadoutFactorization.lean # Approximate readout factorization and fiber-wise readout stability
├── OracleFiberRelations.lean   # Relation-first oracle-fiber equivalence and exact support-level corollaries
├── FeatureFiberLaws.lean     # Feature-fiber restatement of C1/C2/C3 support laws + encoded/approx bridges
├── FiberPreservingObjective.lean # Covered-pair contrastive objectives implying oracle-feature recovery on the covered relation
├── FeatureClassObjectives.lean # Same/different feature-class objectives transported through exact/approx theorem routes
├── LabelScoreObjectives.lean # Arbitrary decoded-label score objectives transported through exact/approx theorem routes
├── ReadoutAlignment.lean    # Same-surface vs factored root readouts; theorem-aligned transport criteria
├── SharedFeatureMultihead.lean # One shared theorem feature supporting multiple approximate downstream heads, including covered-pair/contrastive bridges
├── OptimizationPerturbation.lean # Uniform objective perturbation ⇒ near-optimality transfer for DPO and regularized objectives
├── SketchSummaryOperators.lean # Learned sketch → deterministic summary local-law bridge
├── SketchRecovery.lean       # Generic sketch assumptions ⇒ multi-round + objective equivalence
├── SketchRecoveryInstances.lean # Markov/topic/length concrete encoded-feature instantiations (identity, paired, lossy)
├── ApproximateLocalLaws.lean # ε-local-law budgets ⇒ Δ_R and objective gap bounds
├── RegularizedObjective.lean # Explicit oracle-risk objective + certified local-law regularizer
├── AdaptiveChunkingBridge.lean # Adaptive tree policy bridge to fixed-tree theory
├── RUMSufficientConditions.lean # Pointwise sufficient conditions for expected-group Lipschitz
├── BigramSketch.lean         # Concrete mergeable sketch example (bigrams + boundary tokens)
├── BagOfWordsLDARecovery.lean # Exact bag-of-words LDA recovery via mergeable count sketches
├── LeafLocalMixtureUtilityGap.lean # Linear cancellation + quadratic gap identity for local-mixture utilities
├── TopicBigramOracle.lean    # Topic unigram+bigram oracle (Segment‑LDA sim alignment)
├── ExactUtilityTransport.lean # Oracle-indexed supervised/objective transport + exact mergeable-state utilities
├── ExactUtilityTransportInstances.lean # Markov / complementarity / topic exact-state instantiations
├── RidgeRegressionToy.lean   # Ridge identities (simulation intuition; large-N toy)
├── SketchFlipMergeBridge.lean # Bridge to Sketch-Flip-Merge style dual-target constraints
├── PreferenceLearning.lean   # DPO, GRPO loss definitions and equivalence
├── PreferenceBounds.lean     # Quantitative gap bounds (Lipschitz)
├── AuditBounds.lean          # Violation probability bounds
├── AuditSizes.lean           # Tree-size rewrites + sample-size scaling
├── SerflingAudit.lean        # Conditional Hoeffding (Azuma-ready), Serfling tools (WIP)
├── AdversarialChunkingExample.lean # Non-uniform WOR adversarial chunking example
├── Audit.lean                # Empirical audit framework
├── MeasureTheoreticAudit.lean # Hoeffding inequality connection
├── TrainingPipeline.lean     # Multi-stage gap composition
├── CounterexampleExistence.lean # L3 is substantive (counterexample)
├── ScoreTransport.lean       # Score transport theory
└── MainTheorems.lean         # Curated exports with documentation
```

## Assumptions Used

This module uses **1 modeling assumption** (see `FormalProofs/Axioms.lean` for full documentation):

- `ExpectedGroupLossLipschitz` - Expected loss over groups is Lipschitz

The assumption itself lives in `FormalProbability/DSL/RUM.lean` and is re-exported by
`FormalProofs/OPT/PreferenceBounds.lean` for convenience.

This assumption is justified by the **Random Utility Model** (McFadden 1974).
Under continuous noise distributions, ranking ties have measure zero, so expected losses
are Lipschitz even though pointwise ranking functions are discontinuous.

The assumption is instantiated for specific loss functions:
- `ExpectedGRPOLossLipschitz` - GRPO-PL (Plackett-Luce ranking loss)
- `ExpectedGRPORLLossLipschitz` - GRPO-RL (PPO-style clipped surrogate)

## Key Concepts

### Local Laws (L1, L2, L3)

- **L1 (Sufficiency)**: Summarizing leaves preserves oracle: E[D(g(b), b)] = 0
- **L2 (Merge)**: Merge preserves oracle: E[D(g(u·v), g(g(u)·g(v)))] = 0
- **L3 (Idempotence)**: Re-summarizing is inert: E[D(g(Z), Z)] = 0 for Z ∈ range(g)

### Oracle-Measurability

A function is **oracle-measurable** if it depends on documents only through the oracle f*:
```
dist(f*(x), f*(x')) = 0 ⟹ h(x) = h(x')
```

### Training Equivalence

When local laws hold and losses are oracle-measurable:
```
L_DPO(π; μ_X) = L_DPO(π; μ_Z)
```
where μ_Z is the distribution over summaries.

## Entry Point

For a curated view of the main theorems, see `MainTheorems.lean`.
-/
