import FormalProofs.OPT.ExpectationTheory
import FormalProofs.OPT.PreferenceLearning
import FormalProofs.OPT.PreferenceBounds
import FormalProofs.OPT.TrainingPipeline
import FormalProofs.OPT.MergeableReduction
import FormalProofs.OPT.MergeableProjection
import FormalProofs.OPT.SketchFlipMergeBridge
import FormalProofs.OPT.SketchSummaryOperators
import FormalProofs.OPT.SketchRecovery
import FormalProofs.OPT.SketchRecoveryInstances
import FormalProofs.OPT.ApproximateLocalLaws
import FormalProofs.OPT.InfluenceWeightedLocalLaws
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
import FormalProofs.OPT.PreferenceScope
import FormalProofs.OPT.AgarwalNesting
import FormalProofs.OPT.FuterStateSurfaceFiberDetection
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
import FormalProofs.OPT.RootLocalObjective
import FormalProofs.OPT.ProxyOracleGap
import FormalProofs.OPT.NodeLocalLawAggregate
import FormalProofs.OPT.NodeAIPWLocalLawAdjustment
import FormalProofs.OPT.UnifiedLocalLawAdjustment
import FormalProofs.OPT.DoublyRobustLocalLawAdjustment
import FormalProofs.OPT.DiscountedIPWObjective
import FormalProofs.OPT.AdversarialChunkingExample
import FormalProofs.OPT.MarkovPathDGP
import FormalProofs.OPT.MarkovSimulationValidation
import FormalProofs.OPT.SerflingAudit
import FormalProofs.OPT.NamespaceCompat
import FormalProofs.OPT.UniformG
import FormalProofs.OPT.InformationSufficiency
import FormalProofs.OPT.ContextualQuerySufficiency
import FormalProofs.OPT.SlicedContextualSufficiency
import FormalProofs.OPT.UnifiedGEstimator
import FormalProofs.OPT.UnifiedGSufficientStatisticsLiterature
import FormalProofs.OPT.RandomSlicedContextualSufficiency
import FormalProofs.OPT.InformationRepresentationSufficiency
import FormalProofs.OPT.LikelihoodOnStateSufficiency
import FormalProofs.OPT.SurjectiveLikelihoodOnState
import FormalProofs.OPT.PosteriorOnStateSufficiency
import FormalProofs.OPT.FiniteBayesOnState
import FormalProofs.OPT.PosteriorConsistency
import FormalProofs.OPT.MathlibBayesBridge
import FormalProofs.OPT.BayesianPersuasion
import FormalProofs.OPT.BayesianPersuasionEconomics
import FormalProofs.OPT.BayesianPersuasionDirect
import FormalProofs.OPT.HybridSummarySufficiency
import FormalProofs.OPT.HybridInformationObjectives
import FormalProofs.OPT.DependenceObjectiveProxies
import FormalProofs.OPT.OracleEntropy
import FormalProofs.OPT.OracleSufficientCompression
import FormalProofs.OPT.BagOfWordsLDARecovery
import FormalProofs.OPT.LDAAggregateStatistics
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
  (PreferenceBounds.lean)            on summarized data
```

## Coverage of Modern Methods

The formalization captures:

| Method | File | Key Theorem |
|--------|------|-------------|
| DPO | PreferenceBounds.lean | `dpo_equivalence` |
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
## Nominal Root/Local Objective Exports

The paper-facing objective uses the nominal convex root/local objective. Any
oracle-label correction happens inside the local-law loss supplied to this
objective, not through a separate lambda-renormalization export.
-/

/-- Nominal root/local objective: `(1 - Lambda) * rootLoss + Lambda * lawLoss`. -/
abbrev nominal_root_local_objective :=
  @FormalProofs.OPT.nominalRootLocalObjective

/-- Root-pair surrogate bias is bounded by the two-sided proxy-oracle gap slack. -/
abbrev root_surrogate_bias_bound :=
  @FormalProofs.OPT.rootPairBias_abs_le_oracleRecoverySlack

/-!
## Node Bias Envelope Exports

These aliases expose the proxy-only node-level aggregation layer: node local-law
losses measured through `fhat` are aggregated with depth discounting, and their
surrogate bias is bounded by a discounted node-bias envelope.
-/

/-- Discounted aggregate surrogate-vs-true node law loss is bounded by the
discounted aggregate node-bias envelope. -/
abbrev discounted_node_bias_bound :=
  @FormalProofs.OPT.discountedNodeLawLoss_abs_sub_le_biasBound

/-!
## Node AIPW Adjustment Exports

These aliases expose the combined proxy-plus-node-oracle layer: proxy node
local-law losses form the baseline, and node-oracle observations correct the
proxy residual through the DSL/IPW adjusted-outcome formula. The resulting
adjusted local-law loss is what enters the nominal root/local objective.
-/

/-- If no node-oracle observations are present, the AIPW node law is exactly the
proxy-only discounted node law. -/
abbrev aipw_node_eq_proxy_when_all_unsampled :=
  @FormalProofs.OPT.discountedAIPWNodeLawLoss_eq_proxy_of_all_unsampled

/-- If every node is observed with unit propensity, the AIPW node law is exactly
the true-oracle discounted node law. -/
abbrev aipw_node_eq_oracle_when_all_sampled_pi_one :=
  @FormalProofs.OPT.discountedAIPWNodeLawLoss_eq_oracle_of_all_sampled_pi_one

/-- Matched propensities remove the scalar AIPW residual bias. -/
abbrev aipw_node_scalar_unbiased_matched_propensity :=
  @FormalProofs.OPT.nodeAIPWAdjustedLawLoss_expectation_eq_zero_of_matched_propensity

/-- Misspecified propensities leave the exact scalar residual
`(1 - pi_true / pi_used) * (proxy - oracle)`. -/
abbrev aipw_node_scalar_residual_misspecified_propensity :=
  @FormalProofs.OPT.nodeAIPWAdjustedLawLoss_expectation_eq_residual_of_misspecified_propensity

/-- Discounted aggregate adjusted-vs-true node law loss is bounded by the
discounted aggregate adjusted-error envelope. -/
abbrev aipw_node_adjusted_error_bound :=
  @FormalProofs.OPT.discountedAIPWNodeLawLoss_abs_sub_le_errorBound

/-!
## Unified Local-Law Adjustment Exports

These aliases expose the final local-law adjustment equation: proxy node laws
are corrected by node-oracle AIPW residuals, aggregated with depth discounting,
and then supplied to the nominal root/local objective. The propensity residual
remains explicit, and vanishes under matched logged propensities.
-/

/-- Unified adjusted local-law estimate reduces to the proxy-only discounted
node law when no node-oracle observations are present. -/
abbrev unified_adjusted_local_law_eq_proxy_when_all_unsampled :=
  @FormalProofs.OPT.unifiedAdjustedLocalLawEstimate_eq_proxy_of_all_unsampled

/-- Unified adjusted local-law estimate reduces to the true-oracle discounted
node law when every node is observed with unit propensity. -/
abbrev unified_adjusted_local_law_eq_oracle_when_all_sampled_pi_one :=
  @FormalProofs.OPT.unifiedAdjustedLocalLawEstimate_eq_oracle_of_all_sampled_pi_one

/-- The explicit propensity residual is zero under matched propensities. -/
abbrev unified_local_law_propensity_residual_zero_of_matched :=
  @FormalProofs.OPT.propensityMismatchResidual_eq_zero_of_matched

/-- The scalar AIPW expectation residual equals the explicit propensity
misspecification residual. -/
abbrev unified_local_law_propensity_residual_expectation_identity :=
  @FormalProofs.OPT.nodeAIPWAdjustedLawLoss_expectation_eq_propensityMismatchResidual

/-- Nominal root/local objective using the corrected local-law estimate as its
local channel. -/
abbrev unified_local_law_nominal_objective :=
  @FormalProofs.OPT.unifiedLocalLawNominalObjective

/-- The unified local-law nominal objective is definitionally the nominal
root/local objective applied to the corrected local-law estimate. -/
abbrev unified_local_law_nominal_objective_eq_nominal :=
  @FormalProofs.OPT.unifiedLocalLawNominalObjective_eq_nominal

/-!
## Doubly Robust Local-Law Exports

These aliases expose the classical AIPW/DSL double-robust reading of the
local-law adjustment: the scalar residual is zero if either the logged
propensity is correct or the proxy local law is exact.
-/

/-- Exact proxy local laws remove the explicit propensity residual. -/
abbrev dr_local_law_propensity_residual_zero_of_exact_proxy :=
  @FormalProofs.OPT.propensityMismatchResidual_eq_zero_of_exact_proxy

/-- The explicit propensity residual vanishes under either doubly robust route:
matched propensity or exact proxy local law. -/
abbrev dr_local_law_propensity_residual_zero :=
  @FormalProofs.OPT.propensityMismatchResidual_eq_zero_of_matched_or_exact_proxy

/-- Exact proxy local laws make the scalar AIPW adjusted law unbiased even when
the used propensity is misspecified. -/
abbrev dr_local_law_scalar_unbiased_exact_proxy :=
  @FormalProofs.OPT.nodeAIPWAdjustedLawLoss_expectation_eq_zero_of_exact_proxy

/-- Scalar doubly robust unbiasedness: matched propensity or exact proxy local
law is enough to remove the AIPW residual. -/
abbrev dr_local_law_scalar_unbiased :=
  @FormalProofs.OPT.nodeAIPWAdjustedLawLoss_expectation_eq_zero_of_matched_propensity_or_exact_proxy

/-- Pointwise exact-proxy endpoint for a node-level AIPW local law. -/
abbrev dr_local_law_pointwise_eq_oracle_exact_proxy :=
  @FormalProofs.OPT.nodeAIPWAdjustedLawLoss_eq_oracle_of_exact_proxy

/-- Discounted exact-proxy endpoint for the AIPW node-law aggregate. -/
abbrev dr_local_law_discounted_eq_oracle_exact_proxy :=
  @FormalProofs.OPT.discountedAIPWNodeLawLoss_eq_oracle_of_exact_proxy

/-- Unified exact-proxy endpoint for the adjusted local-law estimate. -/
abbrev dr_local_law_unified_eq_oracle_exact_proxy :=
  @FormalProofs.OPT.unifiedAdjustedLocalLawEstimate_eq_oracle_of_exact_proxy

/-- Under either doubly robust route, the unified envelope drops the explicit
propensity-residual term. -/
abbrev dr_local_law_envelope_no_propensity_residual :=
  @FormalProofs.OPT.adjustedLocalLawEnvelope_eq_no_propensity_residual_of_matched_propensity_or_exact_proxy

/-!
## Mergeable-Projection Exports

These aliases expose the Lean-only formalization of the projection reading used
for learned non-mergeable or approximately mergeable sketches: a neural operator
is selected from a representable class and projected, through local-law
penalties, toward the learnable mergeable/local-law-compatible set.
-/

/-- Learned projections minimize risk over the exact learnable mergeable set. -/
abbrev learned_mergeable_projection_risk_le :=
  @FormalProofs.OPT.learnedMergeableProjection_risk_le

/-- If the learnable mergeable set contains a zero-risk representative, a
projection has zero residual risk. -/
abbrev learned_mergeable_projection_zero_of_exact :=
  @FormalProofs.OPT.learnedMergeableProjection_zero_of_exact

/-- If every learnable mergeable representative has positive risk, a learned
projection has a structural residual gap. -/
abbrev learned_mergeable_projection_structural_gap :=
  @FormalProofs.OPT.learnedMergeableProjection_structural_gap

/-- Approximate local-law version of the learned projection risk bound. -/
abbrev learned_approx_mergeable_projection_risk_le :=
  @FormalProofs.OPT.learnedApproxMergeableProjection_risk_le

/-- At the `λ = 1` endpoint, a faithful nonnegative projection penalty selects
the exact learnable local-law set when that set is nonempty at zero penalty. -/
abbrev local_law_weight_endpoint_projection :=
  @FormalProofs.OPT.classRestrictedBalancedMinimizer_mem_exactLocalLawNeuralOperators_of_lam_one

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

/-- **⚠ Per-tree kernel only.** This alias is a literal rename of
`multi_round_proper`: it covers the fixed-tree kernel of the paper's
fixed-partition theorem, not the extension that names it. The actual
extension — deterministic partition rule `Π`, document distribution `μ_X`, and
the tower step — is formalized as `fixed_partition_population` (support and
expectation forms) in `FormalProofs/OPT/MergeTriangle.lean`; cite that for
Appendix C. -/
abbrev fixed_partition_extension_instantiation := @multi_round_proper

/-- Fixed-partition extension (Appendix C), with the partition rule and tower
step explicit. See `FormalProofs/OPT/MergeTriangle.lean`. -/
abbrev paper_fixed_partition_population := @fixed_partition_population

/-- Coupling-form distortion equals document-level `Δ_R_ZR` when `μ_X = pure(x)`. -/
abbrev coupling_delta_eq_delta_r_zr := @coupling_Δ_eq_Δ_R_ZR

/-!
## Merge Triangle and Compositional Preservation (curated exports)

The paper's central compositionality law `g(x·y) ~ g(g(x)·g(y))` and the
de-circularized preservation tier live in `FormalProofs/OPT/MergeTriangle.lean`;
these aliases give them stable paper-facing names.
-/

/-- The Merge Triangle law: `g(x·y) ~ g(g(x)·g(y))` (paper C3, link 2). -/
abbrev paper_merge_triangle := @MergeTriangle

/-- The triangle is derivable from the audited local laws. -/
abbrev paper_merge_triangle_of_local := @mergeTriangle_of_local

/-- Triangle + joint faithfulness transport the raw-span oracle value to the
merged summary. -/
abbrev paper_merge_faithful_of_triangle := @merge_faithful_of_triangle

/-- n-ary tree generalization of the Merge Triangle. -/
abbrev paper_tree_triangle := @tree_triangle

/-- Theorem 1 by genuine structural induction from one-call local laws and
context compatibility (non-circular). -/
abbrev paper_one_pass_of_local := @one_pass_of_local

/-- Theorem 2 from one-call local laws; no boundedness hypothesis. -/
abbrev paper_multi_round_of_local := @multi_round_of_local

/-- Bridge: one-call local laws + context compatibility derive the legacy
subtree-level `L2` (and a full `LocalLawsBundle`). -/
abbrev paper_L2_of_local := @L2_of_local

/-- Bridge producing the full legacy law bundle from local laws. -/
abbrev paper_local_laws_bundle_of_local := @localLawsBundle_of_local

/-- Corollary 1 with the same-partition hypothesis doing work. -/
abbrev paper_schedule_invariance_of_local := @schedule_invariance_of_local

/-- Corollary 2 with the two-level fold structure explicit. -/
abbrev paper_fold_of_folds_of_local := @fold_of_folds_of_local

/-- Population loss transport: the tower step for preference equivalence. -/
abbrev paper_population_loss_transport := @population_loss_transport

/-- Unified preference gap bound over an arbitrary document-summary coupling. -/
abbrev paper_unified_gap_coupled := @unified_preference_gap_bounded_coupled

/-- Under the local laws, the coupled population preference gap is exactly zero. -/
abbrev paper_population_gap_zero_of_local := @population_gap_zero_of_local

/-- Error-budget union bound (paper Equation `eq:error_budget`). -/
abbrev paper_error_budget_union_bound' := @paper_error_budget_union_bound

/-- Under local laws, the document-level distortion `Δ_R_ZR` is exactly zero. -/
abbrev delta_r_zr_zero_of_local_laws := @Δ_R_eq_zero_of_local_laws

/-!
## Stochastic and Deterministic Theorem-Backed Interfaces

The core C-TreePO semantics are stochastic: `Summarizer α = α → PMF α`.
Deterministic operators, including the current neural-operator realization
bridges, enter as the special case `deterministicSummarizer s`.
-/

/-- Paper-facing stochastic exact route: direct local laws for a PMF-valued
summarizer imply exact theorem-backedness. -/
abbrev stochastic_direct_exact_theorem_backed :=
  @FormalProofs.OPT.DirectSummaryExactAssumptions.toExactTheoremBacked

/-- Paper-facing stochastic approximate route: approximate local laws for a
PMF-valued summarizer imply approximate theorem-backedness. -/
abbrev stochastic_direct_approx_theorem_backed :=
  @FormalProofs.OPT.DirectSummaryApproxAssumptions.toApproxTheoremBacked

/-- Embedding of deterministic summary operators into the stochastic
`Summarizer` interface. -/
abbrev deterministic_summarizer_embedding :=
  @deterministicSummarizer

/-- Deterministic exact theorem-backedness from the global preservation
typeclass route. -/
abbrev deterministic_exact_theorem_backed_of_global_preservation :=
  @FormalProofs.OPT.exactTheoremBacked_of_globalPreservation

/-!
## Neural-Operator Bridge Exports

These aliases expose the Lean-backed interface that connects Section 9-style
neural-operator approximation assumptions to the existing approximate
theorem-backed route. These bridges certify deterministic realizers and then
embed them into the stochastic PMF-valued summarizer semantics above.
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

/-- Paper-facing aggregate transfer moduli
`ω_leaf, ω_merge, ω_idemp` for the neural-operator route. -/
abbrev neural_operator_transfer_moduli :=
  @FormalProofs.OPT.NeuralOperatorTransferModuli

/-- Paper formula
`ω_leaf(ε) + ω_merge(ε) + (R-1)ω_idemp(ε)`. -/
abbrev neural_operator_transfer_local_law_budget :=
  @FormalProofs.OPT.NeuralOperatorTransferModuli.localLawBudget

/-- Paper formula with method transport:
`C_meth * (ω_leaf(ε) + ω_merge(ε) + (R-1)ω_idemp(ε))`. -/
abbrev neural_operator_transfer_method_gap_budget :=
  @FormalProofs.OPT.NeuralOperatorTransferModuli.methodGapBudget

/-- Paper-facing local-law budget produced by a uniform neural-operator
realization bridge. -/
abbrev neural_operator_realization_local_law_budget :=
  @FormalProofs.OPT.ApproxNeuralOperatorPreferenceBridge.localLawBudget

/-- Uniform neural-operator bridge budgets match aggregate transfer moduli. -/
abbrev neural_operator_realization_matches_transfer_moduli :=
  @FormalProofs.OPT.ApproxNeuralOperatorPreferenceBridge.matchesTransferModuli

/-- Uniform bridge budget equals the transfer-modulus paper formula. -/
abbrev neural_operator_realization_budget_eq_transfer_moduli :=
  @FormalProofs.OPT.ApproxNeuralOperatorPreferenceBridge.localLawBudget_eq_transferModuliBudget

/-- Paper-facing `Δ_R` bound from a uniform neural-operator realization bridge
before method-specific preference transport is applied. -/
abbrev neural_operator_delta_r_bound :=
  @FormalProofs.OPT.ApproxNeuralOperatorPreferenceBridge.delta_R_ZR_le_localLawBudget

/-- Paper-form uniform `Δ_R` bound using
`ω_leaf(ε) + ω_merge(ε) + (R-1)ω_idemp(ε)`. -/
abbrev neural_operator_delta_r_transfer_moduli_bound :=
  @FormalProofs.OPT.ApproxNeuralOperatorPreferenceBridge.delta_R_ZR_le_transferModuliBudget

/-- Uniform neural-operator epsilon certificate: if the composed local-law
budget is at most the target threshold, then the tree distortion is certified
at that threshold. -/
abbrev neural_operator_delta_r_epsilon_certificate :=
  @FormalProofs.OPT.ApproxNeuralOperatorPreferenceBridge.delta_R_ZR_le_epsilon_of_localLawBudget_le

/-- Paper-facing uniform proxy-oracle gap assumption for a learned readout
`fhat` approximating the true oracle `fstar`. -/
abbrev oracle_recovered_within :=
  @FormalProofs.OPT.OracleRecoveredWithin

/-- Two-sided true-oracle transport slack from the proxy-oracle gap. -/
abbrev oracle_recovery_slack :=
  FormalProofs.OPT.OracleRecoverySlack

/-- Local-law budget plus proxy-oracle gap slack. -/
abbrev total_oracle_recovery_budget :=
  FormalProofs.OPT.TotalOracleRecoveryBudget

/-- Calibrated-readout uniform neural-operator bridge: local-law distortion for
`fhat` plus a uniform proxy-oracle gap for `fstar` gives true-oracle distortion with
additive `2ε_orc` slack. -/
abbrev neural_operator_true_oracle_delta_r_bound_calibrated :=
  @FormalProofs.OPT.trueOracle_delta_R_ZR_le_of_calibrated_neuralOperatorBridge

/-- Transfer-modulus form of the calibrated-readout uniform neural-operator
true-oracle distortion bridge. -/
abbrev neural_operator_true_oracle_delta_r_transfer_moduli_bound_calibrated :=
  @FormalProofs.OPT.trueOracle_delta_R_ZR_le_of_calibrated_neuralOperatorBridge_transferModuli

/-- Calibrated-readout uniform neural-operator epsilon certificate: the total
proxy-oracle gap budget is compared to one target threshold. -/
abbrev neural_operator_true_oracle_delta_r_epsilon_certificate_calibrated :=
  @FormalProofs.OPT.trueOracle_delta_R_ZR_le_epsilon_of_calibrated_neuralOperatorBridge

/-- Lipschitz true-oracle utility bound through the calibrated-readout uniform
neural-operator bridge. -/
abbrev neural_operator_true_oracle_utility_bound_calibrated :=
  @FormalProofs.OPT.expected_trueOracleUtility_bound_via_calibrated_neuralOperatorBridge

/-- Paper-facing local-law budget produced by a finite-dimensionalization
neural-operator bridge. -/
abbrev neural_operator_fd_realization_local_law_budget :=
  @FormalProofs.OPT.FDNeuralOperatorPreferenceBridge.localLawBudget

/-- Finite-dimensionalization bridge budgets match aggregate transfer moduli. -/
abbrev neural_operator_fd_realization_matches_transfer_moduli :=
  @FormalProofs.OPT.FDNeuralOperatorPreferenceBridge.matchesTransferModuli

/-- Finite-dimensionalization bridge budget equals the transfer-modulus paper
formula. -/
abbrev neural_operator_fd_realization_budget_eq_transfer_moduli :=
  @FormalProofs.OPT.FDNeuralOperatorPreferenceBridge.localLawBudget_eq_transferModuliBudget

/-- Paper-facing `Δ_R` bound from a finite-dimensionalization neural-operator
bridge before method-specific preference transport is applied. -/
abbrev neural_operator_fd_delta_r_bound :=
  @FormalProofs.OPT.FDNeuralOperatorPreferenceBridge.delta_R_ZR_le_localLawBudget

/-- Paper-form finite-dimensionalization `Δ_R` bound using transfer moduli. -/
abbrev neural_operator_fd_delta_r_transfer_moduli_bound :=
  @FormalProofs.OPT.FDNeuralOperatorPreferenceBridge.delta_R_ZR_le_transferModuliBudget

/-- Finite-dimensionalization neural-operator epsilon certificate. -/
abbrev neural_operator_fd_delta_r_epsilon_certificate :=
  @FormalProofs.OPT.FDNeuralOperatorPreferenceBridge.delta_R_ZR_le_epsilon_of_localLawBudget_le

/-- Calibrated-readout finite-dimensionalization neural-operator bridge:
local-law distortion for `fhat` plus a uniform proxy-oracle gap for `fstar` gives
true-oracle distortion with additive `2ε_orc` slack. -/
abbrev neural_operator_fd_true_oracle_delta_r_bound_calibrated :=
  @FormalProofs.OPT.trueOracle_delta_R_ZR_le_of_calibrated_neuralOperatorFDBridge

/-- Transfer-modulus form of the calibrated-readout finite-dimensionalization
neural-operator true-oracle distortion bridge. -/
abbrev neural_operator_fd_true_oracle_delta_r_transfer_moduli_bound_calibrated :=
  @FormalProofs.OPT.trueOracle_delta_R_ZR_le_of_calibrated_neuralOperatorFDBridge_transferModuli

/-- Calibrated finite-dimensionalization neural-operator epsilon certificate. -/
abbrev neural_operator_fd_true_oracle_delta_r_epsilon_certificate_calibrated :=
  @FormalProofs.OPT.trueOracle_delta_R_ZR_le_epsilon_of_calibrated_neuralOperatorFDBridge

/-- Lipschitz true-oracle utility bound through the calibrated-readout
finite-dimensionalization neural-operator bridge. -/
abbrev neural_operator_fd_true_oracle_utility_bound_calibrated :=
  @FormalProofs.OPT.expected_trueOracleUtility_bound_via_calibrated_neuralOperatorFDBridge

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

/-- Paper-form generic expected-objective gap bound through the uniform
neural-operator bridge using aggregate transfer moduli. -/
abbrev expected_objective_gap_via_neural_operator_transfer_moduli :=
  @FormalProofs.OPT.expectedObjectiveGap_via_neuralOperatorTransferModuli

/-- Paper-form generic expected-objective gap bound through the
finite-dimensionalization neural-operator bridge using aggregate transfer
moduli. -/
abbrev expected_objective_gap_via_neural_operator_fd_transfer_moduli :=
  @FormalProofs.OPT.expectedObjectiveGap_via_neuralOperatorFDTransferModuli

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

/-- Composed root-error budget from checked C1/C2/C3 residuals. -/
abbrev approx_local_laws_root_error_budget :=
  @FormalProofs.OPT.ApproxLocalLawsBundle.rootErrorBudget

/-- Paper-facing certificate predicate: the composed local-law root-error
budget is at most the target `ε`. -/
abbrev approx_local_laws_certified_at_epsilon :=
  @FormalProofs.OPT.ApproxLocalLawsBundle.CertifiedAtEpsilon

/-- Approximate-local-law epsilon certificate for `Δ_R_ZR`. -/
abbrev delta_r_zr_bound_of_approx_bundle_certified_at_epsilon :=
  @FormalProofs.OPT.Δ_R_ZR_le_of_approx_bundle_certifiedAtEpsilon

/-!
## Influence-Weighted Local-Law Certificates

These exports make the "no adversarially hidden needles" audit condition
paper-facing: local C1/C2/C3 rows carry influence weights `lambda`, the audit
policy logs row propensities `pi`, and finite-sample certificates bound the
influence-weighted local-law mass.
-/

/-- Local-law audit-row channel: C1 leaf, C2 idempotence, or C3 merge. -/
abbrev influence_local_law_channel := FormalProofs.OPT.LocalLawChannel

/-- Generic finite audit row carrying channel, node, and round identifiers. -/
abbrev influence_local_law_audit_row := FormalProofs.OPT.LocalLawAuditRow

/-- Deterministic local-law residual measured by `fstar` or a proxy `fhat`. -/
abbrev deterministic_local_law_residual :=
  @FormalProofs.OPT.deterministicLocalLawResidual

/-- Influence-weighted local-law mass `sum_a lambda(a) r(a)`. -/
abbrev influence_weighted_local_law_mass :=
  @FormalProofs.OPT.weightedLocalLawMass

/-- Influence-weighted HT summand `lambda(a) / pi(a) * r(a)`. -/
abbrev influence_ht_summand :=
  @FormalProofs.OPT.influenceHTSummand

/-- Empirical influence-weighted HT estimate from sampled audit rows. -/
abbrev empirical_influence_ht :=
  @FormalProofs.OPT.empiricalInfluenceHT

/-- Influence-weighted design-effect proxy `sum_a lambda(a)^2 / pi(a)`. -/
abbrev influence_design_effect :=
  @FormalProofs.OPT.influenceDesignEffect

/-- Worst influence-to-propensity ratio predicate. -/
abbrev influence_worst_ratio_bound :=
  @FormalProofs.OPT.influenceWorstRatioBound

/-- Influence-weighted audit overlap: consequential rows have non-tiny logged
propensity through bounded design effect and worst ratio. -/
abbrev influence_weighted_audit_overlap :=
  @FormalProofs.OPT.InfluenceWeightedAuditOverlap

/-- Uniform proxy calibration transfers proxy local-law residuals to true
oracle local-law residuals with a `2 * eps` row slack. -/
abbrev deterministic_local_law_residual_calibration :=
  @FormalProofs.OPT.deterministicLocalLawResidual_le_proxy_plus_two_calibration

/-- Weighted proxy-to-true calibration transfer:
true local-law mass is bounded by proxy mass plus `2 * eps * sum lambda`. -/
abbrev influence_weighted_oracle_mass_calibration :=
  @FormalProofs.OPT.weightedOracleMass_le_proxy_plus_calibration

/-- Root/document error controlled by influence-weighted local-law mass. -/
abbrev root_error_controlled_by_influence_mass :=
  @FormalProofs.OPT.RootErrorControlledByInfluenceMass

/-- Deterministic root-error bound from any upper bound on
influence-weighted local-law mass. -/
abbrev root_error_bound_from_influence_mass :=
  @FormalProofs.OPT.rootError_le_of_influence_weighted_mass_upper

/-- Root-error bound combining propagation, proxy estimation, and calibration. -/
abbrev root_error_bound_from_proxy_influence_certificate :=
  @FormalProofs.OPT.rootError_le_proxy_estimate_plus_stat_plus_calibration

/-- Packaged finite-sample influence-weighted error certificate. -/
abbrev influence_weighted_error_certificate :=
  @FormalProofs.OPT.InfluenceWeightedErrorCertificate

/-- Packaged certificate bounds root/document error by
`estimate + statRadius + calibrationRadius`. -/
abbrev influence_weighted_error_certificate_root_bound :=
  @FormalProofs.OPT.InfluenceWeightedErrorCertificate.rootError_le_totalBound

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

/-- Statement that local-law coefficients are a projection onto the exact local-law
subspace. -/
abbrev local_law_coefficients_are_projection :=
  @NeuralOperatorSpaces.LocalLawWeightsAreProjection

/-- Class-restricted statement that local-law coefficients project a chosen
neural-operator class onto its exact local-law subspace. -/
abbrev local_law_coefficients_are_projection_on :=
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

/-- Generic same-argmin export: equality of losses on every oracle-measurable
parameter gives equality of oracle-measurable argmin sets. -/
abbrev same_oracle_measurable_argmin_from_loss_eq :=
  @same_oracle_measurable_argmin_general_of_loss_eq

/-!
## Application Assumption Packages

Application sections, including the manifesto/RILE workflow, are not separate
theorems. They supply these predicate packages and then instantiate the generic
DPO/GRPO theorem surfaces.
-/

/-- DPO application package: current and reference policies are
oracle-measurable. Pair-generation oracle-indexing is exported separately. -/
abbrev dpo_application_oracle_measurable_policies :=
  @OracleMeasurablePolicies

/-- DPO application package: preference-pair generation is oracle-indexed. -/
abbrev dpo_application_oracle_indexed_pair_generator :=
  @OracleIndexedPairGen

/-- GRPO-PL application package: policy, ranker, and group generator satisfy
the oracle-measurability/indexing conditions used by `grpo_equivalence`. -/
abbrev grpo_pl_application_oracle_measurable_bundle :=
  @GRPOOracleMeasurableBundle

/-- GRPO-RL application package: policy measurability predicate. -/
abbrev grpo_rl_application_oracle_measurable_policy :=
  @GRPOOracleMeasurable

/-- GRPO-RL application package: reward measurability predicate. -/
abbrev grpo_rl_application_oracle_measurable_reward :=
  @OracleMeasurableReward

/-- GRPO-RL application package: group generator oracle-indexing predicate. -/
abbrev grpo_rl_application_oracle_indexed_group_generator :=
  @OracleIndexedGroupGen

/-- Bundled DPO application assumptions: the policies are oracle-measurable
and the preference-pair generator is oracle-indexed. -/
structure DPOApplicationAssumptionBundle {Strings A Y : Type*} [PseudoMetricSpace Y]
    (pol pol_ref : Policy Strings A) (gen : PairGenerator Strings A)
    (fstar : Strings → Y) where
  policies : OracleMeasurablePolicies pol pol_ref fstar
  pair_generator : OracleIndexedPairGen gen fstar

/-- A DPO application bundle supplies exactly the premises used by the generic
DPO theorem surfaces. -/
theorem dpo_application_bundle_supplies_premises
    {Strings A Y : Type*} [PseudoMetricSpace Y]
    {pol pol_ref : Policy Strings A} {gen : PairGenerator Strings A}
    {fstar : Strings → Y}
    (bundle : DPOApplicationAssumptionBundle pol pol_ref gen fstar) :
    OracleMeasurablePolicies pol pol_ref fstar ∧ OracleIndexedPairGen gen fstar :=
  ⟨bundle.policies, bundle.pair_generator⟩

/-- Bundled GRPO-PL application assumptions: policy, ranker, and group
generator satisfy the oracle-measurability/indexing predicates. -/
structure GRPOPLApplicationAssumptionBundle {Strings A Y : Type*} [PseudoMetricSpace Y]
    {k : ℕ} (pol : Policy' Strings A) (ranker : Strings → GroupRanker A k)
    (gen : GroupGenerator Strings A k) (fstar : Strings → Y) where
  oracle_bundle : GRPOOracleMeasurableBundle pol ranker gen fstar

/-- A GRPO-PL application bundle supplies exactly the bundled predicate used by
the GRPO-PL theorem surfaces. -/
theorem grpo_pl_application_bundle_supplies_premises
    {Strings A Y : Type*} [PseudoMetricSpace Y] {k : ℕ}
    {pol : Policy' Strings A} {ranker : Strings → GroupRanker A k}
    {gen : GroupGenerator Strings A k} {fstar : Strings → Y}
    (bundle : GRPOPLApplicationAssumptionBundle pol ranker gen fstar) :
    GRPOOracleMeasurableBundle pol ranker gen fstar :=
  bundle.oracle_bundle

/-- Bundled GRPO-RL application assumptions: current, old, and reference
policies are oracle-measurable, reward is oracle-measurable, and group
generation is oracle-indexed. -/
structure GRPORLApplicationAssumptionBundle {Strings A Y : Type*} [PseudoMetricSpace Y]
    {k : ℕ} (pol pol_old pol_ref : Policy' Strings A)
    (reward : Strings → A → ℝ) (gen : GroupGenerator Strings A k)
    (fstar : Strings → Y) where
  pol_measurable : GRPOOracleMeasurable pol fstar
  old_measurable : GRPOOracleMeasurable pol_old fstar
  ref_measurable : GRPOOracleMeasurable pol_ref fstar
  reward_measurable : OracleMeasurableReward reward fstar
  group_generator : OracleIndexedGroupGen gen fstar

/-- A GRPO-RL application bundle supplies exactly the premises used by the
GRPO-RL theorem surfaces. -/
theorem grpo_rl_application_bundle_supplies_premises
    {Strings A Y : Type*} [PseudoMetricSpace Y] {k : ℕ}
    {pol pol_old pol_ref : Policy' Strings A}
    {reward : Strings → A → ℝ} {gen : GroupGenerator Strings A k}
    {fstar : Strings → Y}
    (bundle : GRPORLApplicationAssumptionBundle pol pol_old pol_ref reward gen fstar) :
    GRPOOracleMeasurable pol fstar ∧
      GRPOOracleMeasurable pol_old fstar ∧
      GRPOOracleMeasurable pol_ref fstar ∧
      OracleMeasurableReward reward fstar ∧
      OracleIndexedGroupGen gen fstar :=
  ⟨bundle.pol_measurable, bundle.old_measurable, bundle.ref_measurable,
    bundle.reward_measurable, bundle.group_generator⟩

/-!
## Paper Preference Stack

The public preference statement is method-generic: once an application bundle
instantiates the DPO/GRPO theorem premises, exact theorem-backed objectives are
the same on the admissible class. If the paper uses a noisy or approximate
preference object, the residual field records the uniform objective drift.
-/

/-- Paper-facing preference-objective alignment stack.

`fullObjective` is the objective on originals, `summaryObjective` is the
objective on summaries or theorem-backed preference objects, `admissible` is the
policy/model class supplied by the application bundle, and `objective_gap`
records the uniform residual on that class. -/
structure PaperPreferenceStack (Θ : Type*) where
  fullObjective : Θ → ℝ
  summaryObjective : Θ → ℝ
  admissible : Θ → Prop
  residual : ℝ
  objective_gap : ∀ θ, admissible θ → |fullObjective θ - summaryObjective θ| ≤ residual

namespace PaperPreferenceStack

/-- Exact constrained argmin set for a paper preference objective. -/
def argminSet {Θ : Type*} (s : PaperPreferenceStack Θ) (objective : Θ → ℝ) : Set Θ :=
  FormalProofs.OPT.ConstrainedParamEpsilonArgmin objective s.admissible 0

/-- Constrained `ε`-argmin set for a paper preference objective. -/
def epsilonArgminSet {Θ : Type*} (s : PaperPreferenceStack Θ)
    (objective : Θ → ℝ) (ε : ℝ) : Set Θ :=
  FormalProofs.OPT.ConstrainedParamEpsilonArgmin objective s.admissible ε

/-- If the residual is zero, the full and summary constrained argmin sets are
identical. -/
theorem same_argmin_of_zero_residual {Θ : Type*} (s : PaperPreferenceStack Θ)
    (h_residual : s.residual = 0) :
    s.argminSet s.fullObjective = s.argminSet s.summaryObjective := by
  apply Set.Subset.antisymm
  · simpa [argminSet, h_residual, abs_sub_comm] using
      (FormalProofs.OPT.constrainedParamArgmin_subset_epsilonArgmin_of_uniform_loss_perturbation
        (lossTrue := s.summaryObjective)
        (lossSur := s.fullObjective)
        (feasible := s.admissible)
        (ε := 0)
        (hclose := by
          intro θ hθ
          simpa [h_residual, abs_sub_comm] using s.objective_gap θ hθ))
  · simpa [argminSet, h_residual] using
      (FormalProofs.OPT.constrainedParamArgmin_subset_epsilonArgmin_of_uniform_loss_perturbation
        (lossTrue := s.fullObjective)
        (lossSur := s.summaryObjective)
        (feasible := s.admissible)
        (ε := 0)
        (hclose := by
          intro θ hθ
          simpa [h_residual] using s.objective_gap θ hθ))

/-- Any exact summary-objective minimizer is `2 * residual`-optimal for the full
objective on the same admissible class. -/
theorem summary_argmin_full_epsilon {Θ : Type*} (s : PaperPreferenceStack Θ) :
    s.argminSet s.summaryObjective ⊆
      s.epsilonArgminSet s.fullObjective (2 * s.residual) := by
  simpa [argminSet, epsilonArgminSet] using
    (FormalProofs.OPT.constrainedParamArgmin_subset_epsilonArgmin_of_uniform_loss_perturbation
      (lossTrue := s.fullObjective)
      (lossSur := s.summaryObjective)
      (feasible := s.admissible)
      (ε := s.residual)
      (hclose := s.objective_gap))

end PaperPreferenceStack

/-- Public alias for the paper-facing preference stack. -/
abbrev paper_preference_stack := PaperPreferenceStack

/-- Public alias for exact constrained preference argmin sets. -/
abbrev paper_preference_stack_argminSet :=
  @PaperPreferenceStack.argminSet

/-- Public alias for constrained preference `ε`-argmin sets. -/
abbrev paper_preference_stack_epsilonArgminSet :=
  @PaperPreferenceStack.epsilonArgminSet

/-- Public alias: zero residual gives identical full/summary argmins. -/
abbrev paper_preference_stack_same_argmin :=
  @PaperPreferenceStack.same_argmin_of_zero_residual

/-- Public alias: exact summary minimizers are full-objective
`2 * residual`-minimizers. -/
abbrev paper_preference_stack_summary_argmin_full_epsilon :=
  @PaperPreferenceStack.summary_argmin_full_epsilon

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

/-- GRPO-PL same-argmin export: under zero oracle distortion on support, the
full-document and summary objectives have the same oracle-measurable policy
argmin set. -/
abbrev grpo_pl_same_argmin := @grpo_pl_exact_metric

/-- GRPO-RL same-argmin export: under zero oracle distortion on support, the
full-document and summary objectives have the same oracle-measurable current
policy argmin set. -/
abbrev grpo_rl_same_argmin := @grpo_rl_exact_metric

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

/-- End-to-end DPO TreePO certificate: HT objective unbiasedness plus the
method-specific distortion gap. -/
abbrev dpo_treepo_end_to_end :=
  @dpo_treepo_end_to_end_certificate

/-- DPO TreePO certificate with an optional oracle-measurement layer for noisy
or approximate preference targets. -/
abbrev dpo_treepo_end_to_end_with_oracleMeasurement :=
  @dpo_treepo_end_to_end_certificate_with_oracleMeasurement

/-- End-to-end GRPO-PL TreePO certificate: HT objective unbiasedness plus the
method-specific distortion gap. -/
abbrev grpo_pl_treepo_end_to_end :=
  @grpo_pl_treepo_end_to_end_certificate

/-- GRPO-PL TreePO certificate with an optional oracle-measurement layer for
noisy or approximate preference targets. -/
abbrev grpo_pl_treepo_end_to_end_with_oracleMeasurement :=
  @grpo_pl_treepo_end_to_end_certificate_with_oracleMeasurement

/-- End-to-end GRPO-RL TreePO certificate: HT objective unbiasedness plus the
method-specific distortion gap. -/
abbrev grpo_rl_treepo_end_to_end :=
  @grpo_rl_treepo_end_to_end_certificate

/-- GRPO-RL TreePO certificate with an optional oracle-measurement layer for
noisy or approximate preference targets. -/
abbrev grpo_rl_treepo_end_to_end_with_oracleMeasurement :=
  @grpo_rl_treepo_end_to_end_certificate_with_oracleMeasurement

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
## Audit Robustness Exports

These are the paper-facing anchors for the adversarial-sampling appendix:
logged marginal propensities give HT unbiasedness, while variance control is
reported through the explicit constrained-design proxy.
-/

/-- HT unbiasedness under arbitrary sampling with correct logged marginal
propensities. -/
abbrev ht_unbiased_of_logged_marginals :=
  @htExp_unbiased_of_logged_marginals

/-- Uniform finite-population HT unbiasedness under arbitrary sampling with
correct logged marginal propensities. -/
abbrev ht_uniform_mean_unbiased_of_logged_marginals :=
  @htUniformMean_unbiased_of_logged_marginals

/-- Constrained-design variance proxy
`N^{-2}∑ᵢ((1-πᵢ)/πᵢ)Yᵢ²`. -/
abbrev ht_uniform_mean_variance_proxy :=
  @htUniformMeanVarianceProxy

/-- Covariance-control condition connecting actual HT variance to the
Bernoulli-design proxy. -/
abbrev ht_uniform_mean_covariance_controlled :=
  @HTUniformMeanCovarianceControlled

/-- Algebraic constrained-design variance bound from `π_min` and `D_max`. -/
abbrev ht_uniform_mean_variance_proxy_bound :=
  @htUniformMeanVarianceProxy_le_constrained

/-- Actual constrained-design variance bound once covariance control is
available. -/
abbrev ht_uniform_mean_variance_bound_constrained :=
  @htUniformMean_variance_bound_of_constrained_design

/-- Bernoulli-product-measure surface for the same variance bound. -/
abbrev ht_uniform_mean_variance_bound_independent_bernoulli :=
  @htUniformMean_variance_bound_of_independent_bernoulli

/-- Independent Bernoulli product sampling satisfies the covariance-control
proxy used by the constrained-design variance theorem. -/
abbrev ht_uniform_mean_covariance_controlled_independent_bernoulli :=
  @htUniformMean_covarianceControlled_independent_bernoulli

/-- TreePO distortion HT unbiasedness under logged marginal propensities. -/
abbrev tree_audit_uniform_distortion_unbiased_logged :=
  @treeAuditUniformDistortion_unbiased_of_logged_marginals

/-- TreePO distortion constrained-design variance bound. -/
abbrev tree_audit_uniform_distortion_variance_bound_constrained :=
  @treeAuditUniformDistortion_variance_bound_of_constrained_design

/-- TreePO distortion Bernoulli-product-measure variance-bound surface. -/
abbrev tree_audit_uniform_distortion_variance_bound_independent_bernoulli :=
  @treeAuditUniformDistortion_variance_bound_of_independent_bernoulli

/-!
## Paper Error Certificate Exports

These aliases make the displayed finite-sample certificate formula a first-class
Lean object: transported local-law distortion, calibration, estimation, and
clipping.
-/

/-- Local-law budget object carrying `delta_R` and the method transport
constant. -/
abbrev paper_local_law_error_budget := PaperLocalLawErrorBudget

/-- Paper-facing error-certificate object. -/
abbrev paper_error_certificate := PaperErrorCertificate

/-- Paper-facing finite-sample error stack object. -/
abbrev paper_error_stack := PaperErrorStack

/-- Transported local-law distortion `C_meth * delta_R`. -/
abbrev paper_local_law_transported_distortion :=
  @PaperLocalLawErrorBudget.transportedDistortion

/-- Total certificate bound
`C_meth * delta_R + B_cal + B_est + B_clip`. -/
abbrev paper_error_certificate_total_bound :=
  @PaperErrorCertificate.totalObjectiveBound

/-- The paper error certificate expands definitionally to the displayed
formula. -/
abbrev paper_error_certificate_formula :=
  @PaperErrorCertificate.totalObjectiveBound_eq_paper_formula

/-- Deterministic objective-gap wrapper for the displayed certificate. -/
abbrev paper_error_certificate_objective_gap :=
  @PaperErrorCertificate.objective_gap_le_total

/-- High-probability objective-gap wrapper from calibration, estimation,
clipping, and local-law transport events. -/
abbrev paper_error_certificate_high_prob :=
  @PaperErrorCertificate.high_prob_total_of_events

/-- High-probability objective-gap wrapper for the bundled paper error stack. -/
abbrev paper_error_stack_high_prob :=
  @PaperErrorStack.high_prob_total

/-- Clipped inverse-propensity weight `min(w, w_max)`. -/
abbrev clipped_ipw_weight := @TreeSample.clippedWeight

/-- Deterministic clipping excess `w - min(w, w_max)`. -/
abbrev clipped_ipw_excess := @TreeSample.clippingExcess

/-- Aggregate clipping-bias envelope from total clipped mass. -/
abbrev clipped_ipw_bias_abs_bound := @clippingBiasAbsBound

/-- Deterministic aggregate clipping-bias bound. -/
abbrev clipped_ipw_total_bias_bound :=
  @totalClippingBias_abs_le

/-- Deterministic clipped-vs-unclipped Hajek gap bound. -/
abbrev clipped_hajek_gap_bound :=
  @clippedHajek_abs_diff_le

/-- Relative clipped-mass corollary for the Hajek gap. -/
abbrev clipped_hajek_gap_bound_relative_excess :=
  @clippedHajek_abs_diff_le_of_relative_excess

/-- Unit-range corollary for the clipped Hajek gap. -/
abbrev clipped_hajek_gap_bound_unit :=
  @clippedHajek_abs_diff_le_unit

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

/-!
## Preference Scope: State-Factored Preferences

These aliases expose the Lean-first scope statement: C-TreePO supports
preferences and losses that factor through a locally preserved task state /
oracle fiber.  Additive separability is a special case, not the boundary.
-/

/-- A downstream preference/readout depends on the document only through a task
state. -/
abbrev preference_factors_through_state :=
  @FormalProofs.OPT.PreferenceFactorsThroughState

/-- A downstream loss depends on the document only through a task state. -/
abbrev loss_factors_through_state :=
  @FormalProofs.OPT.LossFactorsThroughState

/-- A state has exact leaf encoding and an exact binary merge law. -/
abbrev exact_composable_state :=
  @FormalProofs.OPT.ExactComposableState

/-- General fiber/preimage of a map value. -/
abbrev map_fiber :=
  @FormalProofs.OPT.MapFiber

/-- General same-fiber relation induced by a map. -/
abbrev same_map_fiber :=
  @FormalProofs.OPT.SameMapFiber

/-- Same-fiber is an equivalence relation for every map. -/
abbrev same_map_fiber_equivalence :=
  @FormalProofs.OPT.sameMapFiber_equivalence

/-- Same-fiber equality is membership in one common value fiber. -/
abbrev same_map_fiber_iff_exists_common_value :=
  @FormalProofs.OPT.sameMapFiber_iff_exists_common_value

/-- Fiber of a state value: all documents mapped to that state value. -/
abbrev state_fiber :=
  @FormalProofs.OPT.StateFiber

/-- Same-state-fiber relation: two documents have the same state value. -/
abbrev same_state_fiber :=
  @FormalProofs.OPT.SameStateFiber

/-- Same-state-fiber is an equivalence relation. -/
abbrev same_state_fiber_equivalence :=
  @FormalProofs.OPT.sameStateFiber_equivalence

/-- Same-state-fiber equality is membership in one common state fiber. -/
abbrev same_state_fiber_iff_exists_common_state :=
  @FormalProofs.OPT.sameStateFiber_iff_exists_common_state

/-- Two points in the same named state fiber are in the same state-fiber
equivalence class. -/
abbrev same_state_fiber_of_state_fiber :=
  @FormalProofs.OPT.sameStateFiber_of_stateFiber

/-- Moving along same-state-fiber equality preserves named fiber membership. -/
abbrev state_fiber_of_same_state_fiber_left :=
  @FormalProofs.OPT.stateFiber_of_sameStateFiber_left

/-- Symmetric named-fiber membership transport. -/
abbrev state_fiber_of_same_state_fiber_right :=
  @FormalProofs.OPT.stateFiber_of_sameStateFiber_right

/-- Fiber of a downstream preference/readout value. -/
abbrev preference_fiber :=
  @FormalProofs.OPT.PreferenceFiber

/-- Same-preference-fiber relation. -/
abbrev same_preference_fiber :=
  @FormalProofs.OPT.SamePreferenceFiber

/-- Equality-based oracle-value fiber. -/
abbrev oracle_value_fiber :=
  @FormalProofs.OPT.OracleValueFiber

/-- Equality-based same-oracle-value fiber. -/
abbrev same_oracle_value_fiber :=
  @FormalProofs.OPT.SameOracleValueFiber

/-- Existing metric oracle fibers coincide with equality-based oracle-value
fibers for metric theorem oracles. -/
abbrev same_oracle_fiber_iff_same_oracle_value_fiber :=
  @FormalProofs.OPT.sameOracleFiber_iff_sameOracleValueFiber

/-- A readout respects state fibers when it is constant on same-state pairs. -/
abbrev readout_respects_state_fibers :=
  @FormalProofs.OPT.ReadoutRespectsStateFibers

/-- State fibers refine preference fibers when state equality never hides a
preference distinction. -/
abbrev state_fibers_refine_preference_fibers :=
  @FormalProofs.OPT.StateFibersRefinePreferenceFibers

/-- Explicit partition-language form is equivalent to readout constancy on
state fibers. -/
abbrev state_fibers_refine_preference_fibers_iff_respects_state_fibers :=
  @FormalProofs.OPT.stateFibersRefinePreferenceFibers_iff_respectsStateFibers

/-- State-factorization implies constancy on state fibers. -/
abbrev preference_factors_through_state_respects_fibers :=
  @FormalProofs.OPT.preferenceFactorsThroughState_respectsStateFibers

/-- Constancy on state fibers implies state factorization. -/
abbrev readout_respects_state_fibers_factors :=
  @FormalProofs.OPT.readoutRespectsStateFibers_factorsThroughState

/-- For inhabited readout codomains, state factorization is equivalent to
constancy on state fibers. -/
abbrev preference_factors_through_state_iff_respects_fibers :=
  @FormalProofs.OPT.preferenceFactorsThroughState_iff_respectsStateFibers

/-- State factorization is equivalently: state fibers refine preference
fibers. -/
abbrev preference_factors_through_state_iff_state_fibers_refine_preference_fibers :=
  @FormalProofs.OPT.preferenceFactorsThroughState_iff_stateFibersRefinePreferenceFibers

/-- Explicit preference shape: `pref x = readout (state x)`. -/
abbrev preference_readout_of_state :=
  @FormalProofs.OPT.PreferenceReadoutOfState

/-- Explicit state readout implies existential state factorization. -/
abbrev preference_readout_of_state_factors :=
  @FormalProofs.OPT.preferenceReadoutOfState_factorsThroughState

/-- A named state fiber maps into the corresponding preference fiber under an
explicit state readout. -/
abbrev state_fiber_subset_preference_fiber_of_readout :=
  @FormalProofs.OPT.stateFiber_subset_preferenceFiber_of_readout

/-- Same-state-fiber pairs are same-preference-fiber pairs under an explicit
state readout. -/
abbrev same_state_fiber_implies_same_preference_fiber_of_state_readout :=
  @FormalProofs.OPT.sameStateFiber_implies_samePreferenceFiber_of_stateReadout

/-- A deterministic summary operator `g` preserves the task state. -/
abbrev summary_preserves_state :=
  @FormalProofs.OPT.SummaryPreservesState

/-- State preservation by `g` is exactly same-state-fiber preservation. -/
abbrev summary_preserves_state_iff_same_state_fiber :=
  @FormalProofs.OPT.summaryPreservesState_iff_sameStateFiber

/-- A global task state decomposes over concatenation by a binary merge. -/
abbrev state_decomposes_by :=
  @FormalProofs.OPT.StateDecomposesBy

/-- The two-route `g (g x * g y)` law stated directly on task state. -/
abbrev summary_merge_preserves_state :=
  @FormalProofs.OPT.SummaryMergePreservesState

/-- Two-route state preservation is exactly same-state-fiber preservation along
the merge route. -/
abbrev summary_merge_preserves_state_iff_same_state_fiber :=
  @FormalProofs.OPT.summaryMergePreservesState_iff_sameStateFiber

/-- Pointwise state preservation plus state decomposability implies two-route
state preservation. -/
abbrev summary_merge_preserves_state_of_preserves_and_decomposes :=
  @FormalProofs.OPT.summaryMergePreservesState_of_preservesState_and_stateDecomposes

/-- State-preserving summaries preserve every preference read out from that
state. -/
abbrev summary_preserves_preference_of_state_readout :=
  @FormalProofs.OPT.summaryPreservesPreference_of_stateReadout

/-- Two-route state preservation preserves every preference read out from that
state along the two-route merge path. -/
abbrev summary_merge_preserves_preference_of_state_readout :=
  @FormalProofs.OPT.summaryMergePreservesPreference_of_stateReadout

/-- State preservation implies A1 for the encoded-state oracle. -/
abbrev summary_preserves_state_implies_A1_encoded_oracle :=
  @FormalProofs.OPT.summaryPreservesState_implies_A1_encodedOracle

/-- Two-route state preservation implies A2 for the encoded-state oracle. -/
abbrev summary_merge_preserves_state_implies_A2_encoded_oracle :=
  @FormalProofs.OPT.summaryMergePreservesState_implies_A2_encodedOracle

/-- If an oracle identifies a state, oracle-fiber equality implies state-fiber
equality. -/
abbrev same_state_fiber_of_same_oracle_fiber :=
  @FormalProofs.OPT.sameStateFiber_of_sameOracleFiber

/-- Abstract type carriers for Futer 2013 state-surface fiber detection. -/
abbrev futer2013_state_surface_types :=
  @FormalProofs.OPT.Futer2013.StateSurfaceTypes

/-- Predicate vocabulary for Futer 2013 state-surface fiber detection. -/
abbrev futer2013_state_surface_predicates :=
  @FormalProofs.OPT.Futer2013.StateSurfacePredicates

/-- Futer 2013 Theorem 1 statement: homogeneous state surface is a topological
fiber surface iff the reduced state graph is a tree. -/
abbrev futer2013_theorem1_statement :=
  @FormalProofs.OPT.Futer2013.theorem1_statement

/-- Futer 2013 Corollary 2, A-adequate half. -/
abbrev futer2013_corollary2_A_statement :=
  @FormalProofs.OPT.Futer2013.corollary2_A_statement

/-- Futer 2013 Corollary 2, B-adequate half. -/
abbrev futer2013_corollary2_B_statement :=
  @FormalProofs.OPT.Futer2013.corollary2_B_statement

/-- Generic detector schema: an object-level property is decided by an
associated certificate-level predicate. -/
abbrev detector_problem :=
  @FormalProofs.OPT.Futer2013.DetectorProblem

/-- Exact detector predicate for the generic detector schema. -/
abbrev exact_detector :=
  @FormalProofs.OPT.Futer2013.ExactDetector

/-- Futer Theorem 1 yields an exact detector: reduced graph tree-ness detects
topological fiber-surface status. -/
abbrev futer2013_theorem1_yields_exact_detector :=
  @FormalProofs.OPT.Futer2013.theorem1_yields_exact_detector

/-- C-TreePO analogue: a state-factored predicate is detected by its state. -/
abbrev state_factored_detector_problem :=
  @FormalProofs.OPT.Futer2013.state_factored_detector_problem

/-- State-factored predicates are exactly detected by their state certificate. -/
abbrev state_factored_detector_exact :=
  @FormalProofs.OPT.Futer2013.state_factored_detector_exact

/-- Local merge state realizes a global task state via encode/merge/decode. -/
abbrev local_state_realizes_global_state :=
  @FormalProofs.OPT.LocalStateRealizesGlobalState

/-- Decoding the folded local state recovers the direct global state. -/
abbrev local_state_decode_mergeFold_eq_global :=
  @FormalProofs.OPT.LocalStateRealizesGlobalState.decode_mergeFold_eq_global

/-- A preference is captured by a local state that realizes a global task state. -/
abbrev global_local_preference_shape :=
  @FormalProofs.OPT.GlobalLocalPreferenceShape

/-- A global/local preference shape admits a readout from the decoded folded
local state. -/
abbrev global_local_preference_readout_of_fold :=
  @FormalProofs.OPT.GlobalLocalPreferenceShape.readout_of_local_mergeFold

/-- A preference is captured by a mergeable state when the state is exactly
composable and the preference factors through that state. -/
abbrev mergeable_preference_shape :=
  @FormalProofs.OPT.MergeablePreferenceShape

/-- A mergeable preference admits a root readout from the folded state. -/
abbrev mergeable_preference_readout_of_fold :=
  @FormalProofs.OPT.MergeablePreferenceShape.readout_of_mergeFold

/-- Agarwal-style state-level nesting: merge summary states, preserve a
validity relation, then read out the preference at the root. -/
abbrev relational_mergeable_preference_shape :=
  @FormalProofs.OPT.RelationalMergeablePreferenceShape

/-- Relational state-level summaries are hierarchical over merge trees. -/
abbrev relational_mergeable_preference_hierarchical :=
  @FormalProofs.OPT.RelationalMergeablePreferenceShape.hierarchical

/-- Relational state-level summaries recover the root preference after merging
states up the tree. -/
abbrev relational_mergeable_preference_readout_of_tree :=
  @FormalProofs.OPT.RelationalMergeablePreferenceShape.readout_of_mergeTree

/-- Epsilon relational state-level nesting: root readout is within the task
metric threshold whenever the merged root state is valid. -/
abbrev epsilon_relational_mergeable_preference_shape :=
  @FormalProofs.OPT.EpsilonRelationalMergeablePreferenceShape

/-- Epsilon relational state-level summaries are hierarchical over merge trees. -/
abbrev epsilon_relational_mergeable_preference_hierarchical :=
  @FormalProofs.OPT.EpsilonRelationalMergeablePreferenceShape.hierarchical

/-- Epsilon relational summaries give a root task-metric error bound after
state merging. -/
abbrev epsilon_relational_mergeable_preference_readout_error_of_tree :=
  @FormalProofs.OPT.EpsilonRelationalMergeablePreferenceShape.readout_error_of_mergeTree

/-- Canonical/equality-valued relational summaries collapse to the existing
exact mergeable preference shape. -/
abbrev relational_mergeable_preference_to_exact_shape_of_canonical :=
  @FormalProofs.OPT.RelationalMergeablePreferenceShape.to_mergeablePreferenceShape_of_canonical

/-- Randomized root-readout correctness event for a merge tree. -/
abbrev randomized_tree_readout_success :=
  @FormalProofs.OPT.RandomizedTreeReadoutSuccess

/-- Valid randomized root states read out correctly with the same probability
lower bound. -/
abbrev randomized_tree_readout_success_of_randomized_tree_success :=
  @FormalProofs.OPT.randomizedTreeReadoutSuccess_of_randomizedTreeSuccess

/-- Randomized root epsilon-readout accuracy event for a merge tree. -/
abbrev randomized_tree_epsilon_readout_success :=
  @FormalProofs.OPT.RandomizedTreeEpsilonReadoutSuccess

/-- Randomized root validity transfers to root epsilon-readout accuracy with
the same probability lower bound. -/
abbrev randomized_tree_epsilon_readout_success_of_randomized_tree_success :=
  @FormalProofs.OPT.randomizedTreeEpsilonReadoutSuccess_of_randomizedTreeSuccess

/-- Randomized Agarwal-style state-level nesting: root validity with high
probability plus deterministic valid-state readout. -/
abbrev randomized_relational_mergeable_preference_shape :=
  @FormalProofs.OPT.RandomizedRelationalMergeablePreferenceShape

/-- Randomized relational summaries recover the root preference in probability
after merging states up the tree. -/
abbrev randomized_relational_mergeable_preference_readout_success_of_tree :=
  @FormalProofs.OPT.RandomizedRelationalMergeablePreferenceShape.readout_success_of_mergeTree

/-- Randomized epsilon Agarwal-style state-level nesting: root validity with
high probability plus deterministic epsilon valid-state readout. -/
abbrev randomized_epsilon_relational_mergeable_preference_shape :=
  @FormalProofs.OPT.RandomizedEpsilonRelationalMergeablePreferenceShape

/-- Randomized epsilon relational summaries recover root task accuracy in
probability after merging states up the tree. -/
abbrev randomized_epsilon_relational_mergeable_preference_readout_success_of_tree :=
  @FormalProofs.OPT.RandomizedEpsilonRelationalMergeablePreferenceShape.readout_success_of_mergeTree

/-- Every `StateLevelMergeableSummary` with query correctness instantiates the
relational C-TreePO preference shape. -/
abbrev agarwal_state_level_relational_shape :=
  @FormalProofs.OPT.stateLevelMergeableSummary_relationalShape

/-- `StateLevelMergeableSummary` root-readout theorem over arbitrary merge
trees. -/
abbrev agarwal_state_level_readout_of_merge_tree :=
  @FormalProofs.OPT.stateLevelMergeableSummary_readout_of_mergeTree

/-- Every `StateLevelMergeableSummary` with epsilon query correctness
instantiates the epsilon relational C-TreePO preference shape. -/
abbrev agarwal_state_level_epsilon_relational_shape :=
  @FormalProofs.OPT.stateLevelMergeableSummary_epsilonRelationalShape

/-- `StateLevelMergeableSummary` root epsilon-readout theorem over arbitrary
merge trees. -/
abbrev agarwal_state_level_readout_error_of_merge_tree :=
  @FormalProofs.OPT.stateLevelMergeableSummary_readout_error_of_mergeTree

/-- Canonical/equality-valued Agarwal summaries instantiate the existing exact
mergeable preference shape. -/
abbrev agarwal_state_level_to_exact_mergeable_shape_of_canonical :=
  @FormalProofs.OPT.stateLevelMergeableSummary_to_mergeablePreferenceShape_of_canonical

/-- Explicit adapter bundle listing the transformations needed to read a
C-TreePO state/readout construction as Agarwal's fixed-`ε` `S(D, ε)` summary
interface. -/
abbrev ctreepo_to_agarwal_transform :=
  @FormalProofs.OPT.CTreePOToAgarwalTransform

/-- Epsilon adapter bundle: validity plus root scorer give an `ε` metric
guarantee instead of exact equality. -/
abbrev ctreepo_to_agarwal_epsilon_transform :=
  @FormalProofs.OPT.CTreePOToAgarwalEpsilonTransform

/-- The adapter forgets size metadata to produce Agarwal's state-level
mergeable summary interface. -/
abbrev ctreepo_to_agarwal_state_level_summary :=
  @FormalProofs.OPT.CTreePOToAgarwalTransform.toStateLevelMergeableSummary

/-- The epsilon adapter forgets size metadata to produce Agarwal's state-level
mergeable summary interface. -/
abbrev ctreepo_to_agarwal_epsilon_state_level_summary :=
  @FormalProofs.OPT.CTreePOToAgarwalEpsilonTransform.toStateLevelMergeableSummary

/-- The adapter also produces the C-TreePO relational nesting shape. -/
abbrev ctreepo_to_agarwal_relational_shape :=
  @FormalProofs.OPT.CTreePOToAgarwalTransform.toRelationalShape

/-- The epsilon adapter produces the C-TreePO epsilon relational nesting shape. -/
abbrev ctreepo_to_agarwal_epsilon_relational_shape :=
  @FormalProofs.OPT.CTreePOToAgarwalEpsilonTransform.toEpsilonRelationalShape

/-- Leaf adapter: C-TreePO build gives a valid sized `S(D, ε)` state. -/
abbrev ctreepo_to_agarwal_build_valid_sized_state :=
  @FormalProofs.OPT.CTreePOToAgarwalTransform.buildValidSizedState

/-- Merge adapter: valid child `S(Dᵢ, ε)` states merge to a valid sized
`S(D₁ ++ D₂, ε)` state. -/
abbrev ctreepo_to_agarwal_merge_valid_sized_state :=
  @FormalProofs.OPT.CTreePOToAgarwalTransform.mergeValidSizedState

/-- Tree adapter: C-TreePO state merging over a tree gives a valid sized
`S(D, ε)` root state for the represented data. -/
abbrev ctreepo_to_agarwal_merge_tree_valid_sized_state :=
  @FormalProofs.OPT.CTreePOToAgarwalTransform.mergeTree_validSizedState

/-- Root adapter: after state merging, readout recovers the target preference. -/
abbrev ctreepo_to_agarwal_readout_of_merge_tree :=
  @FormalProofs.OPT.CTreePOToAgarwalTransform.readout_of_mergeTree

/-- Epsilon root adapter: after state merging, readout is within the target
task metric threshold. -/
abbrev ctreepo_to_agarwal_epsilon_readout_error_of_merge_tree :=
  @FormalProofs.OPT.CTreePOToAgarwalEpsilonTransform.readout_error_of_mergeTree

/-- Size-profile adapter: the merged root state satisfies `k(|D|, ε)`. -/
abbrev ctreepo_to_agarwal_merge_tree_size_bound :=
  @FormalProofs.OPT.CTreePOToAgarwalTransform.mergeTree_size_bound

/-- Randomized Agarwal summaries instantiate the randomized relational C-TreePO
preference shape. -/
abbrev agarwal_randomized_relational_shape :=
  @FormalProofs.OPT.randomizedMergeableSummary_relationalShape

/-- Randomized Agarwal summaries recover the root preference in probability
over arbitrary merge trees. -/
abbrev agarwal_randomized_readout_success_of_merge_tree :=
  @FormalProofs.OPT.randomizedMergeableSummary_readout_success_of_mergeTree

/-- Randomized Agarwal summaries instantiate the epsilon randomized relational
C-TreePO preference shape. -/
abbrev agarwal_randomized_epsilon_relational_shape :=
  @FormalProofs.OPT.randomizedMergeableSummary_epsilonRelationalShape

/-- Randomized Agarwal summaries recover root epsilon-readout accuracy in
probability over arbitrary merge trees. -/
abbrev agarwal_randomized_epsilon_readout_success_of_merge_tree :=
  @FormalProofs.OPT.randomizedMergeableSummary_epsilon_readout_success_of_mergeTree

/-- Explicit marker for the stronger scalar child-query merge law, which is not
required by Agarwal-style state-level nesting. -/
abbrev scalar_query_merge_law :=
  @FormalProofs.OPT.ScalarQueryMergeLaw

/-- Additive-state special case: concatenation maps to state addition. -/
abbrev additive_composable_state :=
  @FormalProofs.OPT.AdditiveComposableState

/-- Additive states instantiate exact composable states with merge `+`. -/
abbrev additive_state_to_exact_composable_state :=
  @FormalProofs.OPT.AdditiveComposableState.toExactComposableState

/-- Extra condition for classical additive separability: the readout is
additive on merged state values. -/
abbrev additive_state_readout :=
  @FormalProofs.OPT.AdditiveStateReadout

/-- Additively separable utilities are the additive-readout subcase of
state-factored preferences. -/
abbrev additively_separable_through_state :=
  @FormalProofs.OPT.AdditivelySeparableThroughState

/-- Additive state plus additive readout yields concatenation-additive utility. -/
abbrev additive_state_readout_yields_concat_additive :=
  @FormalProofs.OPT.additive_state_readout_yields_concat_additive

/-- Additively separable utilities factor through state, but state-factored
preferences need not be additive. -/
abbrev additively_separable_factors_through_state :=
  @FormalProofs.OPT.additivelySeparableThroughState_factorsThroughState

/-- C-TreePO exact theorem-backedness transports any state-factored objective
whose state is identified by the preserved oracle. -/
abbrev ctreepo_supports_state_factored_preference :=
  @FormalProofs.OPT.ctreepo_supports_state_factored_preference

/-- Exact mergeable state preservation supports arbitrary downstream utilities
on that state. -/
abbrev exact_mergeable_state_supports_any_downstream_utility :=
  @FormalProofs.OPT.exact_mergeable_state_supports_any_downstream_utility

/-- Supported nonseparable complementarity over exact left/right count state. -/
abbrev supported_nonseparable_complementarity :=
  @FormalProofs.OPT.supported_nonseparable_complementarity

/-- Supported boundary interaction through topic unigram plus boundary state. -/
abbrev supported_boundary_interaction :=
  @FormalProofs.OPT.supported_boundary_interaction

/-- Histogram state supports arbitrary histogram utilities, not only additive
linear utilities. -/
abbrev supported_histogram_state_any_utility :=
  @FormalProofs.OPT.supported_histogram_state_any_utility

/-- Bag-of-words LDA likelihood is preserved by the histogram/count-sketch
state. -/
abbrev supported_lda_likelihood_histogram_utility :=
  @FormalProofs.OPT.supported_lda_likelihood_histogram_utility

/-- Generic classical state-level sketch bridge: merge states first, then
read out/query at the root. -/
abbrev classical_state_level_mergeable_preference_shape :=
  @FormalProofs.OPT.classical_state_level_mergeable_preference_shape

/-- Additive linear sketches are fully mergeable. -/
abbrev additive_linear_sketch_preference_shape :=
  @FormalProofs.OPT.additive_linear_sketch_preference_shape

/-- Count-Min-style additive counter tables are state-level mergeable. -/
abbrev count_min_state_level_preference_shape :=
  @FormalProofs.OPT.count_min_state_level_preference_shape

/-- HLL-style max-register states are state-level mergeable. -/
abbrev hll_state_level_preference_shape :=
  @FormalProofs.OPT.hll_state_level_preference_shape

/-- Additive scalar preferences are mergeable when final oracle values compose
by addition. -/
abbrev additive_scalar_preference_is_mergeable :=
  @FormalProofs.OPT.additive_scalar_preference_is_mergeable

/-- Generic scalar-oracle obstruction: equal child oracle values cannot support
different parent oracle values under one global scalar merge. -/
abbrev scalar_oracle_concat_witness_not_expressible :=
  @FormalProofs.OPT.scalar_oracle_concat_witness_not_expressible

/-- Threshold-AND fails if each child is collapsed to its Boolean threshold
answer before merge. -/
abbrev scalar_threshold_and_not_expressible :=
  @FormalProofs.OPT.scalar_threshold_and_not_expressible

/-- Boundary bigrams fail if each child scalar omits boundary-token state. -/
abbrev scalar_boundary_bigram_not_expressible :=
  @FormalProofs.OPT.scalar_boundary_bigram_not_expressible

/-- Scalar child distinct counts are not a sufficient merge state. -/
abbrev insufficient_scalar_distinct_count_state :=
  @FormalProofs.OPT.insufficient_scalar_distinct_count_state

/-- Markov count-only state is insufficient for arbitrary topology claims. -/
abbrev insufficient_markov_count_only_state :=
  @FormalProofs.OPT.insufficient_markov_count_only_state

/-- C2/on-range idempotence is an independent operator requirement. -/
abbrev c2_idempotence_not_derivable :=
  @FormalProofs.OPT.c2_idempotence_not_derivable

/-- Public-shape C2 independence counterexample. -/
abbrev c2_independence_counterexample :=
  @FormalProofs.OPT.c2_independence_counterexample

/-- A preference that separates one theorem-state fiber is not state-factored. -/
abbrev preference_not_factored_through_state :=
  @FormalProofs.OPT.preference_not_factored_through_state

/-- If the oracle is injective, oracle-sufficient compression cannot help. -/
abbrev no_compression_when_everything_matters :=
  @FormalProofs.OPT.no_compression_when_everything_matters

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

/-- Exact proxy-oracle agreement is exactly the statement that the learned theorem
feature is constant on oracle fibers. -/
abbrev oracle_feature_recovery_respects_same_oracle_fiber :=
  @FormalProofs.OPT.oracleRecoversFeature_iff_respects_sameOracleFiber

/-- Approximate proxy-oracle agreement is exactly the statement that the learned
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

/-- Restricted approximate proxy-oracle agreement is the right surface when only a
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
approximate proxy-oracle agreement for both heads. -/
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

/-- OPS global assumptions imply merge-closure in the strict oracle-output
mergeable-summary interface. -/
abbrev ops_mergeable_mergeClosed := @ops_mergeClosed_of_global

/-- OPS global assumptions imply strict hierarchical mergeability over arbitrary
merge trees. -/
abbrev ops_mergeable_hierarchical := @ops_hierarchical_mergeable_of_global

/-- Reduction of OPS global assumptions to the strict oracle-output mergeable
summary statement. -/
abbrev ops_mergeable_classical := @ops_reduction_to_classical_mergeable

/-- Classical state-level sketch reduction: merge states first, then read out. -/
abbrev sketch_state_level_mergeable :=
  @sketch_state_level_reduction_to_classical_mergeable

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

/-- Fixed-ranker Plackett-Luce route discharging the GRPO-PL expected-loss
Lipschitz condition from policy Lipschitzness and the finite PL form. -/
abbrev grpo_pl_expected_lipschitz_from_plackett_luce_fixed_ranker :=
  @ExpectedGRPOLossLipschitz_plackettLuce_fixed_ranker_all

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

/-- Generic contextual-response signature of an input. -/
abbrev contextual_response_signature :=
  @FormalProofs.OPT.ResponseSignature

/-- Generic deterministic sufficiency: representation fibers refine
contextual-response fibers. -/
abbrev contextual_query_sufficient :=
  @FormalProofs.OPT.QuerySufficient

/-- Two-sided compositional contextual query
`query (left, right) x = fstar (left * x * right)`. -/
abbrev twoSided_context_query :=
  @FormalProofs.OPT.TwoSidedContextQuery

/-- Generic two-sided contextual sufficiency for a learned state map `g`. -/
abbrev twoSided_context_sufficient :=
  @FormalProofs.OPT.TwoSidedContextSufficient

/-- One shared compositional `g`: leaves and merge pairs are embedded into one
carrier space, then the same endomap `g : Carrier → Carrier` is applied at both
sites. -/
abbrev uniform_composable_g :=
  @FormalProofs.OPT.UniformG

/-- Leaf-state map induced by one shared compositional `g`. -/
abbrev uniform_composable_leaf :=
  @FormalProofs.OPT.UniformG.leaf

/-- Merge-state map induced by the same shared compositional `g`. -/
abbrev uniform_composable_merge :=
  @FormalProofs.OPT.UniformG.merge

/-- Finite sampled-context coverage of full contextual-response fibers. -/
abbrev finite_context_covers :=
  @FormalProofs.OPT.FiniteContextCovers

/-- Sufficiency is equivalent to existence of a readout from the state to every
contextual query response. -/
abbrev contextual_query_sufficient_iff_exists_contextReadout :=
  @FormalProofs.OPT.querySufficient_iff_exists_contextReadout

/-- A sufficient state map cannot collapse inputs distinguished by some
contextual response. -/
abbrev contextual_query_sufficient_no_bad_collision :=
  @FormalProofs.OPT.querySufficient_no_collision_of_distinguished_context

/-- If sampled contexts cover true response-signature fibers, zero loss on
that sampled set implies full contextual sufficiency. -/
abbrev finite_context_zeroLoss_implies_contextual_sufficiency :=
  @FormalProofs.OPT.finiteContext_zeroLoss_implies_querySufficient

/-- Algebraic leaf/merge/readout behavior implies two-sided contextual
sufficiency of the leaf state. This form does not by itself require leaf and
merge to be induced by the same learned `g`. -/
abbrev exact_composed_state_readout_implies_twoSided_contextual_sufficiency :=
  @FormalProofs.OPT.composedTwoSidedReadoutExact_implies_twoSidedContextSufficient

/-- Exact composed behavior for one shared `g` plus readout `f` implies
two-sided contextual sufficiency of the induced leaf state. -/
abbrev exact_shared_gf_implies_twoSided_contextual_sufficiency :=
  @FormalProofs.OPT.uniformComposedTwoSidedReadoutExact_implies_twoSidedContextSufficient

/-- Approximate query sufficiency: representation collisions cost at most `ε`
in any contextual query response (under a pseudometric on the response type).
Treats the empirical case where a learned `g` is sufficient up to slack. -/
abbrev contextual_query_sufficient_within :=
  @FormalProofs.OPT.QuerySufficientWithin

/-- Approximate finite-context sufficiency on a sampled context set. -/
abbrev contextual_query_sufficient_within_on :=
  @FormalProofs.OPT.QuerySufficientWithinOn

/-- Approximate finite-context cover: sampled context closeness implies full
contextual-response closeness. -/
abbrev finite_context_covers_within :=
  @FormalProofs.OPT.FiniteContextCoversWithin

/-- Approximate finite-context bridge into approximate contextual sufficiency. -/
abbrev finite_context_within_implies_contextual_sufficiency_within :=
  @FormalProofs.OPT.finiteContext_within_implies_querySufficientWithin

/-- Two-sided compositional approximate contextual sufficiency for a learned
state map `g` with slack `ε`. -/
abbrev twoSided_context_sufficient_within :=
  @FormalProofs.OPT.TwoSidedContextSufficientWithin

/-- Metric near-collision contextual sufficiency for continuous learned states. -/
abbrev contextual_query_sufficient_near_within :=
  @FormalProofs.OPT.QuerySufficientNearWithin

/-- Approximate contextual-response realization by a state readout. -/
abbrev contextual_readout_realizes_within :=
  @FormalProofs.OPT.ContextReadoutRealizesWithin

/-- Radius-local readout stability, the Lean surface for Lipschitz-style state
continuity. -/
abbrev contextual_readout_near_preserving :=
  @FormalProofs.OPT.ContextReadoutNearPreserving

/-- Exact readout plus radius-local stability implies metric contextual
sufficiency. -/
abbrev contextual_readout_near_preserving_implies_near_sufficiency :=
  @FormalProofs.OPT.contextReadoutNearPreserving_implies_querySufficientNearWithin

/-- Approximate readout plus radius-local stability implies metric contextual
sufficiency with readout error paid on both sides. -/
abbrev contextual_readout_approx_near_preserving_implies_near_sufficiency :=
  @FormalProofs.OPT.contextReadoutApproxNearPreserving_implies_querySufficientNearWithin

/-- Approximate algebraic leaf/merge/readout behavior implies approximate
two-sided contextual sufficiency of the leaf state with `2 * ε` slack. -/
abbrev approx_composed_state_readout_implies_twoSided_contextual_sufficiency :=
  @FormalProofs.OPT.composedTwoSidedReadoutWithinEps_implies_twoSidedContextSufficientWithin

/-- Approximate composed behavior for one shared `g` plus readout `f` implies
approximate two-sided contextual sufficiency of the induced leaf state with
`2 * ε` slack. This is the load-bearing shared-`g` bridge from a sampled-context
contextual training loss to a formal sufficiency guarantee. -/
abbrev approx_shared_gf_implies_twoSided_contextual_sufficiency :=
  @FormalProofs.OPT.uniformComposedTwoSidedReadoutWithinEps_implies_twoSidedContextSufficientWithin

/-- Zero-slack instance of the algebraic approximate bridge: exact composed
behavior implies zero-slack two-sided contextual sufficiency. -/
abbrev exact_composed_state_readout_implies_twoSided_contextual_sufficiency_within_zero :=
  @FormalProofs.OPT.composedTwoSidedReadoutExact_implies_twoSidedContextSufficientWithin_zero

/-!
### Package-Level Unified-g Estimator Surface

These aliases name the generic package-facing organization: a problem adapter
supplies the contextual query family, while estimator choices realize the same
shared `UniformG` contract.
-/

/-- Generic problem adapter: the contextual query family whose fibers a learned
state model must preserve. -/
abbrev unified_g_problem :=
  @FormalProofs.OPT.UnifiedGProblem

/-- Two-sided problem adapter `query (left,right) x = fstar (left * x * right)`. -/
abbrev unified_g_twoSided_problem :=
  @FormalProofs.OPT.UnifiedGProblem.twoSided

/-- Realized estimator: one shared `UniformG` endomap on a carrier space. -/
abbrev unified_g_estimator :=
  @FormalProofs.OPT.UnifiedGEstimator

/-- Family of swappable estimators, each realizing the same shared-`g` contract. -/
abbrev unified_g_estimator_family :=
  @FormalProofs.OPT.UnifiedGEstimatorFamily

/-- Leaf-state map induced by a realized unified-`g` estimator. -/
abbrev unified_g_leaf_state :=
  @FormalProofs.OPT.UnifiedGEstimator.leafState

/-- Merge-state map induced by the same realized unified-`g` estimator. -/
abbrev unified_g_merge_state :=
  @FormalProofs.OPT.UnifiedGEstimator.mergeState

/-- Problem-level sufficiency of a realized unified-`g` estimator. -/
abbrev unified_g_query_sufficient :=
  @FormalProofs.OPT.UnifiedGQuerySufficient

/-- Approximate problem-level sufficiency of a realized unified-`g` estimator. -/
abbrev unified_g_query_sufficient_within :=
  @FormalProofs.OPT.UnifiedGQuerySufficientWithin

/-- Exact theorem-state decoding certifies problem-level sufficiency. This is
the generic Markov/HLL exact-state route. -/
abbrev unified_g_exact_state_decoder_implies_contextual_sufficiency :=
  @FormalProofs.OPT.unifiedG_exactStateDecoder_implies_querySufficient

/-- Finite contextual-response preservation plus a cover condition certifies
problem-level sufficiency. -/
abbrev unified_g_finite_context_zeroLoss_implies_contextual_sufficiency :=
  @FormalProofs.OPT.unifiedG_finiteContext_zeroLoss_implies_querySufficient

/-- Finite selected-slice preservation plus a slice-cover condition certifies
problem-level sufficiency. -/
abbrev unified_g_finite_sliced_zeroLoss_implies_contextual_sufficiency :=
  @FormalProofs.OPT.unifiedG_finiteSliced_zeroLoss_implies_querySufficient

/-- Exact composed two-sided readout behavior certifies problem-level
sufficiency for the realized unified-`g` estimator. -/
abbrev unified_g_composed_twoSided_readout_exact_implies_contextual_sufficiency :=
  @FormalProofs.OPT.unifiedG_composedTwoSidedReadoutExact_implies_querySufficient

/-- Approximate composed two-sided readout behavior certifies approximate
problem-level sufficiency with the existing `2ε` slack. -/
abbrev unified_g_composed_twoSided_readout_within_implies_contextual_sufficiency :=
  @FormalProofs.OPT.unifiedG_composedTwoSidedReadoutWithin_implies_querySufficientWithin

/-!
### Unified-g Literature Method Certificates

These aliases organize the main paper routes for learning or certifying a
unified `g` sufficient statistic for `f*`.
-/

/-- Named method families for unified-`g` sufficient-statistic learning. -/
abbrev unified_g_sufficient_statistic_method :=
  @FormalProofs.OPT.UnifiedGSufficientStatisticMethod

/-- NASS-style dependence-proxy certificate for a selected unified-`g` estimator. -/
abbrev dependence_proxy_unified_g_certificate :=
  @FormalProofs.OPT.DependenceProxyUnifiedGCertificate

/-- NASS/dependence-proxy loss minimization yields symbolic proxy maximization
under an order-reversal assumption. -/
abbrev nass_dependence_proxy_certificate_proxy_argmax :=
  @FormalProofs.OPT.nass_dependenceProxy_certificate_proxyArgmax

/-- Uniform proxy error turns exact proxy optimality into deterministic
near-optimality for the target information objective. -/
abbrev nass_dependence_proxy_certificate_information_epsilon_argmax :=
  @FormalProofs.OPT.nass_dependenceProxy_certificate_informationEpsilonArgmax

/-- The deterministic readout part of a NASS-style certificate proves
problem-level unified-`g` sufficiency. -/
abbrev nass_dependence_proxy_readout_certificate_implies_unified_g_sufficient :=
  @FormalProofs.OPT.nass_dependenceProxy_readout_certificate_implies_unifiedG_sufficient

/-- SSS/NASSS finite selected-slice certificate for exact unified-`g`
sufficiency. -/
abbrev nasss_finite_slices_certificate_implies_unified_g_sufficient :=
  @FormalProofs.OPT.nasss_finiteSlices_certificate_implies_unifiedG_sufficient

/-- SSS/NASSS finite selected-slice certificate for approximate unified-`g`
sufficiency. -/
abbrev nasss_finite_slices_within_certificate_implies_unified_g_sufficient_within :=
  @FormalProofs.OPT.nasss_finiteSlices_within_certificate_implies_unifiedG_sufficientWithin

/-- Likelihood family obtained by evaluating a likelihood head on the learned
unified-`g` state. -/
abbrev unified_g_likelihood_on_state_family :=
  @FormalProofs.OPT.UnifiedGLikelihoodOnStateFamily

/-- SSNL/SNLE deterministic core: likelihood-on-unified-state is likelihood
family sufficient. -/
abbrev ssnl_unified_g_likelihood_on_state_family_sufficient :=
  @FormalProofs.OPT.ssnl_unifiedG_likelihoodOnState_family_sufficient

/-- A unified-`g` state with a decoded learned state is sufficient for every
likelihood-on-decoded-state family. -/
abbrev ssnl_unified_g_state_readout_likelihood_on_state_family_sufficient :=
  @FormalProofs.OPT.ssnl_unifiedG_stateReadout_likelihoodOnState_family_sufficient

/-- Makinen-style hybrid summary specialized to a base statistic plus the
unified-`g` leaf state. -/
abbrev unified_g_hybrid_summary :=
  @FormalProofs.OPT.UnifiedGHybridSummary

/-- The hybrid product refines the learned unified-`g` state. -/
abbrev hybrid_unified_g_summary_sufficient_for_unified_g_state :=
  @FormalProofs.OPT.hybrid_unifiedG_summary_sufficient_for_unifiedG_state

/-- A hybrid response readout proves likelihood-free sufficiency of the hybrid
product `(base(x), g(x))`. -/
abbrev hybrid_unified_g_response_readout_implies_hybrid_sufficient :=
  @FormalProofs.OPT.hybrid_unifiedG_response_readout_implies_hybrid_sufficient

/-- If the base statistic is readable from `g(x)`, a hybrid readout collapses
back to ordinary unified-`g` sufficiency. -/
abbrev hybrid_base_readout_response_readout_implies_unified_g_sufficient :=
  @FormalProofs.OPT.hybrid_baseReadout_responseReadout_implies_unifiedG_sufficient

/-- Approximate hybrid readout collapses back to approximate unified-`g`
sufficiency when the base statistic is readable from `g(x)`. -/
abbrev hybrid_base_readout_response_readout_within_implies_unified_g_sufficient_within :=
  @FormalProofs.OPT.hybrid_baseReadout_responseReadoutWithin_implies_unifiedG_sufficientWithin

/-- Two-sided oracle value on `(left, x, right)` triples. -/
abbrev unified_g_two_sided_oracle :=
  @FormalProofs.OPT.UnifiedGTwoSidedOracle

/-- Realized composed two-sided readout induced by one shared unified `g`. -/
abbrev unified_g_two_sided_composed_readout :=
  @FormalProofs.OPT.UnifiedGTwoSidedComposedReadout

/-- Neural-operator compact-uniform approximation over all two-sided triples
certifies approximate unified-`g` sufficiency. -/
abbrev neural_operator_uniform_approx_all_triples_implies_unified_g_sufficient_within :=
  @FormalProofs.OPT.neuralOperator_uniformApproxAllTriples_implies_unifiedG_sufficientWithin

/-- Direct composed-readout error theorem for neural-operator unified-`g`
certificates. -/
abbrev neural_operator_composed_readout_within_implies_unified_g_sufficient_within :=
  @FormalProofs.OPT.neuralOperator_composedReadoutWithin_implies_unifiedG_sufficientWithin

/-- SSS/NASSS-style sliced response signature: selected slice functions applied
to the full contextual response signature. -/
abbrev sliced_contextual_signature :=
  @FormalProofs.OPT.SlicedResponseSignature

/-- Exact sliced contextual sufficiency: representation collisions preserve all
selected slice values. -/
abbrev sliced_query_sufficient :=
  @FormalProofs.OPT.SlicedQuerySufficient

/-- Exact all-slice cover condition: equality of selected slice values implies
equality of full contextual response signatures. -/
abbrev slices_cover_response_fibers :=
  @FormalProofs.OPT.SlicesCoverResponseFibers

/-- Exact all-slice bridge: sliced sufficiency plus slice cover implies ordinary
contextual sufficiency. -/
abbrev sliced_sufficiency_implies_contextual_sufficiency :=
  @FormalProofs.OPT.slicedQuerySufficient_implies_querySufficient

/-- Finite selected-slice sufficiency matching empirical SSS/NASSS probes. -/
abbrev finite_sliced_query_sufficient :=
  @FormalProofs.OPT.SlicedQuerySufficientOn

/-- Finite selected-slice cover condition: equality on a sampled slice set
implies equality of full contextual response signatures. -/
abbrev finite_slices_cover_response_fibers :=
  @FormalProofs.OPT.FiniteSlicesCoverResponseFibers

/-- Finite selected-slice zero-loss bridge: if the sampled slice set covers full
response fibers, zero sliced collision loss implies contextual sufficiency. -/
abbrev finite_sliced_zeroLoss_implies_contextual_sufficiency :=
  @FormalProofs.OPT.finiteSliced_zeroLoss_implies_querySufficient

/-- Coordinate slice of a contextual response signature. -/
abbrev coordinate_slice :=
  @FormalProofs.OPT.CoordinateSlice

/-- All coordinate slices cover full response fibers. -/
abbrev coordinate_slices_cover_response_fibers :=
  @FormalProofs.OPT.coordinateSlices_cover_responseFibers

/-- The full finite set of coordinate slices covers full response fibers. -/
abbrev finite_coordinate_slices_univ_cover_response_fibers :=
  @FormalProofs.OPT.finiteCoordinateSlices_univ_cover_responseFibers

/-- Coordinate slices cover response fibers with the same metric slack. -/
abbrev coordinate_slices_cover_response_fibers_within :=
  @FormalProofs.OPT.coordinateSlices_cover_responseFibersWithin

/-- Full finite coordinate slices cover response fibers with the same metric
slack. -/
abbrev finite_coordinate_slices_univ_cover_response_fibers_within :=
  @FormalProofs.OPT.finiteCoordinateSlices_univ_cover_responseFibersWithin

/-- Left-invertible deterministic slice families cover response fibers. -/
abbrev left_invertible_slices_cover_response_fibers :=
  @FormalProofs.OPT.leftInvertibleSlices_cover_responseFibers

/-- Approximate sliced sufficiency: representation collisions keep all slice
values within slack. -/
abbrev sliced_query_sufficient_within :=
  @FormalProofs.OPT.SlicedQuerySufficientWithin

/-- Approximate all-slice cover condition from slice slack to contextual-response
slack. -/
abbrev slices_cover_response_fibers_within :=
  @FormalProofs.OPT.SlicesCoverResponseFibersWithin

/-- Approximate all-slice bridge into approximate contextual sufficiency. -/
abbrev sliced_within_implies_contextual_sufficiency_within :=
  @FormalProofs.OPT.slicedWithin_implies_querySufficientWithin

/-- Approximate finite selected-slice sufficiency. -/
abbrev finite_sliced_query_sufficient_within :=
  @FormalProofs.OPT.SlicedQuerySufficientWithinOn

/-- Approximate finite selected-slice cover condition. -/
abbrev finite_slices_cover_response_fibers_within :=
  @FormalProofs.OPT.FiniteSlicesCoverResponseFibersWithin

/-- Approximate finite selected-slice bridge into approximate contextual
sufficiency. -/
abbrev finite_sliced_within_implies_contextual_sufficiency_within :=
  @FormalProofs.OPT.finiteSlicedWithin_implies_querySufficientWithin

/-- Random finite-slice good event: a seed's selected slices cover response
fibers and the seed's learned representation has zero finite-slice loss. -/
abbrev random_finite_sliced_good_event :=
  @FormalProofs.OPT.RandomFiniteSlicedGoodEvent

/-- Random finite-slice seedwise bridge from the good event to contextual
sufficiency. -/
abbrev random_finite_sliced_good_event_implies_contextual_sufficiency :=
  @FormalProofs.OPT.randomFiniteSlicedGoodEvent_implies_querySufficient

/-- Event-level probability transport for exact random finite slices. -/
abbrev random_finite_sliced_contextual_sufficiency_failure_prob_le :=
  @FormalProofs.OPT.randomFiniteSliced_contextualSufficiency_failure_prob_le

/-- Approximate random finite-slice good event with slice slack and contextual
response slack. -/
abbrev random_finite_sliced_within_good_event :=
  @FormalProofs.OPT.RandomFiniteSlicedWithinGoodEvent

/-- Approximate random finite-slice seedwise bridge from the good event to
approximate contextual sufficiency. -/
abbrev random_finite_sliced_within_good_event_implies_contextual_sufficiency_within :=
  @FormalProofs.OPT.randomFiniteSlicedWithinGoodEvent_implies_querySufficientWithin

/-- Event-level probability transport for approximate random finite slices. -/
abbrev random_finite_sliced_within_contextual_sufficiency_failure_prob_le :=
  @FormalProofs.OPT.randomFiniteSlicedWithin_contextualSufficiency_failure_prob_le

/-- Representation-level sufficient statistic: representation fibers refine
target fibers. -/
abbrev target_sufficient_representation :=
  @FormalProofs.OPT.TargetSufficientRepresentation

/-- A target can be read out from a sufficient representation. -/
abbrev target_readout_realizes :=
  @FormalProofs.OPT.TargetReadoutRealizes

/-- Representation sufficiency is equivalent to existence of a target readout. -/
abbrev target_sufficient_iff_exists_readout :=
  @FormalProofs.OPT.targetSufficient_iff_exists_readout

/-- Target-measurable downstream quantities are constant on target fibers. -/
abbrev target_measurable :=
  @FormalProofs.OPT.TargetMeasurable

/-- A representation sufficient for a target also preserves every
target-measurable downstream quantity. -/
abbrev target_sufficient_preserves_target_measurable :=
  @FormalProofs.OPT.targetSufficient_preserves_targetMeasurable

/-- Likelihood-model sufficient statistic: the representation preserves the
whole likelihood family over parameters. -/
abbrev likelihood_family_sufficient :=
  @FormalProofs.OPT.LikelihoodFamilySufficient

/-- Likelihood-family sufficiency is contextual query sufficiency with
parameters as contexts. -/
abbrev likelihood_family_sufficient_iff_contextual_query_sufficient :=
  @FormalProofs.OPT.likelihoodFamilySufficient_iff_querySufficient

/-- Likelihood-family sufficiency is equivalent to factoring the likelihood
through a representation readout. -/
abbrev likelihood_family_sufficient_iff_exists_readout :=
  @FormalProofs.OPT.likelihoodFamilySufficient_iff_exists_readout

/-- Likelihood-family sufficiency forbids collisions distinguished by any
parameter likelihood. -/
abbrev likelihood_family_sufficient_no_collision_of_distinguished_likelihood :=
  @FormalProofs.OPT.likelihoodFamilySufficient_no_collision_of_distinguished_likelihood

/-- Likelihood family induced by evaluating a state-space likelihood head on a
learned state. -/
abbrev likelihood_on_state_family :=
  @FormalProofs.OPT.LikelihoodOnStateFamily

/-- The state-space likelihood head realizes the induced likelihood family from
the learned state. -/
abbrev likelihood_on_state_readout_realizes :=
  @FormalProofs.OPT.likelihoodOnState_readout_realizes

/-- Likelihood-on-state families are likelihood-family sufficient with respect
to the learned state. -/
abbrev likelihood_on_state_family_sufficient :=
  @FormalProofs.OPT.likelihoodOnState_family_sufficient

/-- A richer representation with a decoder to the learned state is sufficient
for any likelihood-on-state family. -/
abbrev rep_with_state_readout_likelihood_on_state_family_sufficient :=
  @FormalProofs.OPT.repWithStateReadout_likelihoodOnState_family_sufficient

/-- Likelihood-on-state sufficiency forbids state collisions that a
state-likelihood head can distinguish. -/
abbrev likelihood_on_state_no_collision_of_likelihood_distinct :=
  @FormalProofs.OPT.likelihoodOnState_no_collision_of_likelihood_distinct

/-- Approximate likelihood-family sufficiency with metric slack. -/
abbrev likelihood_family_sufficient_within :=
  @FormalProofs.OPT.LikelihoodFamilySufficientWithin

/-- Approximate likelihood readout realization. -/
abbrev likelihood_readout_realizes_within :=
  @FormalProofs.OPT.LikelihoodReadoutRealizesWithin

/-- Approximate likelihood readout implies approximate likelihood-family
sufficiency with readout error paid on both collapsed inputs. -/
abbrev likelihood_readout_within_implies_likelihood_family_sufficient_within :=
  @FormalProofs.OPT.likelihoodReadoutWithin_implies_likelihoodFamilySufficientWithin

/-- Approximate readout for a likelihood-on-state family implies approximate
likelihood-on-state sufficiency. -/
abbrev state_likelihood_readout_within_implies_likelihood_on_state_sufficient_within :=
  @FormalProofs.OPT.stateLikelihoodReadoutWithin_implies_likelihoodOnStateSufficientWithin

/-- Surjective state map: the set-theoretic fragment of a surjector/state
coverage claim. -/
abbrev surjective_state_map :=
  @FormalProofs.OPT.SurjectiveStateMap

/-- Likelihood factorization through a state-space likelihood head. -/
abbrev likelihood_factors_through_state :=
  @FormalProofs.OPT.LikelihoodFactorsThroughState

/-- Surjective-state factorization: state-fiber likelihood sufficiency yields a
state-space likelihood head. -/
abbrev surjective_state_likelihood_factorization :=
  @FormalProofs.OPT.surjectiveState_likelihood_factorization

/-- For a surjective state map, likelihood sufficiency is equivalent to
factorization through a state likelihood head. -/
abbrev surjective_state_likelihood_sufficient_iff_factors :=
  @FormalProofs.OPT.surjectiveState_likelihoodSufficient_iff_factors

/-- Approximate surjective-state readout: state-fiber likelihood slack yields a
state-space likelihood head within the same slack. -/
abbrev surjective_state_likelihood_readout_within :=
  @FormalProofs.OPT.surjectiveState_likelihoodReadoutWithin

/-- Posterior/readout sufficiency as target sufficiency for a posterior-like
object. -/
abbrev posterior_sufficient :=
  @FormalProofs.OPT.PosteriorSufficient

/-- A posterior-like object evaluated through a state is sufficient with
respect to that state. -/
abbrev posterior_on_state_sufficient :=
  @FormalProofs.OPT.posteriorOnState_sufficient

/-- A richer representation with a decoder to the frozen state is sufficient for
any posterior-like object evaluated through that state. -/
abbrev rep_with_state_readout_posterior_on_state_sufficient :=
  @FormalProofs.OPT.repWithStateReadout_posteriorOnState_sufficient

/-- Likelihood-family sufficiency transports to posterior/readout sufficiency
when the posterior-like object is determined by the likelihood family. -/
abbrev likelihood_sufficient_implies_posterior_sufficient :=
  @FormalProofs.OPT.likelihoodSufficient_implies_posteriorSufficient

/-- State-likelihood sufficiency transports to posterior/readout sufficiency
under an explicit posterior-determined-by-likelihood assumption. -/
abbrev likelihood_on_state_implies_posterior_sufficient :=
  @FormalProofs.OPT.likelihoodOnState_implies_posteriorSufficient

/-- Surjective-state posterior factorization from posterior sufficiency. -/
abbrev surjective_state_posterior_factorization :=
  @FormalProofs.OPT.surjectiveState_posterior_factorization

/-- For a surjective state, posterior sufficiency is equivalent to posterior
factorization through a state-space readout. -/
abbrev surjective_state_posterior_sufficient_iff_factors :=
  @FormalProofs.OPT.surjectiveState_posteriorSufficient_iff_factors

/-- Approximate posterior/readout sufficiency from an approximate posterior
readout. -/
abbrev posterior_readout_within_implies_posterior_sufficient_within :=
  @FormalProofs.OPT.posteriorReadoutWithin_implies_posteriorSufficientWithin

/-- Approximate surjective-state posterior readout from state-fiber posterior
slack. -/
abbrev surjective_state_posterior_readout_within :=
  @FormalProofs.OPT.surjectiveState_posteriorReadoutWithin

/-- Finite Bayes posterior for a fixed prior and likelihood family. -/
abbrev finite_bayes_posterior :=
  @FormalProofs.OPT.BayesPosterior

/-- Mathlib event-conditioned probability measure `μ[|s]`. -/
abbrev mathlib_conditional_probability :=
  @FormalProofs.OPT.mathlib_conditional_probability

/-- Mathlib event-level Bayes rule for conditional probabilities. -/
abbrev mathlib_conditional_bayes_rule :=
  @FormalProofs.OPT.mathlib_conditional_bayes_rule

/-- Mathlib conditional-probability application formula. -/
abbrev mathlib_conditional_probability_apply :=
  @FormalProofs.OPT.mathlib_conditional_probability_apply

/-- Mathlib conditioning-twice identity: condition on `s`, then `t`, equals
conditioning on `s ∩ t`. -/
abbrev mathlib_conditional_probability_condition_twice :=
  @FormalProofs.OPT.mathlib_conditional_probability_condition_twice

/-- Mathlib conditional-probability product/intersection identity. -/
abbrev mathlib_conditional_probability_mul_eq_inter :=
  @FormalProofs.OPT.mathlib_conditional_probability_mul_eq_inter

/-- Mathlib complement form of the law of total probability. -/
abbrev mathlib_conditional_probability_total_complement :=
  @FormalProofs.OPT.mathlib_conditional_probability_total_complement

/-- Mathlib finite-fiber law of total probability for a random variable. -/
abbrev mathlib_conditional_probability_finite_fiber_total :=
  @FormalProofs.OPT.mathlib_conditional_probability_finite_fiber_total

/-- Mathlib condition making a conditional measure a probability measure. -/
abbrev mathlib_conditional_probability_is_probability :=
  @FormalProofs.OPT.mathlib_conditional_probability_is_probability

/-- Mathlib absolute-continuity of conditional measures with respect to the
original measure. -/
abbrev mathlib_conditional_probability_absolutely_continuous :=
  @FormalProofs.OPT.mathlib_conditional_probability_absolutely_continuous

/-- Mathlib conditional expectation `μ[f|m]`. -/
abbrev mathlib_conditional_expectation :=
  @FormalProofs.OPT.mathlib_conditional_expectation

/-- Mathlib additivity of conditional expectation for integrable summands. -/
abbrev mathlib_conditional_expectation_add :=
  @FormalProofs.OPT.mathlib_conditional_expectation_add

/-- Mathlib conditional expectation of constants. -/
abbrev mathlib_conditional_expectation_const :=
  @FormalProofs.OPT.mathlib_conditional_expectation_const

/-- Mathlib a.e. congruence for conditional expectation. -/
abbrev mathlib_conditional_expectation_congr_ae :=
  @FormalProofs.OPT.mathlib_conditional_expectation_congr_ae

/-- Mathlib integral preservation for conditional expectation. -/
abbrev mathlib_integral_conditional_expectation :=
  @FormalProofs.OPT.mathlib_integral_conditional_expectation

/-- Mathlib set-integral preservation for conditional expectation. -/
abbrev mathlib_set_integral_conditional_expectation :=
  @FormalProofs.OPT.mathlib_set_integral_conditional_expectation

/-- Mathlib strong measurability of conditional expectation. -/
abbrev mathlib_strongly_measurable_conditional_expectation :=
  @FormalProofs.OPT.mathlib_strongly_measurable_conditional_expectation

/-- Mathlib integrability of conditional expectation. -/
abbrev mathlib_integrable_conditional_expectation :=
  @FormalProofs.OPT.mathlib_integrable_conditional_expectation

/-- Mathlib identity for already conditioning-measurable functions. -/
abbrev mathlib_conditional_expectation_of_strongly_measurable :=
  @FormalProofs.OPT.mathlib_conditional_expectation_of_strongly_measurable

/-- Mathlib conditional expectation indicator identity. -/
abbrev mathlib_conditional_expectation_indicator :=
  @FormalProofs.OPT.mathlib_conditional_expectation_indicator

/-- Mathlib conditional-expectation/Radon-Nikodym bridge. -/
abbrev mathlib_rn_deriv_ae_eq_conditional_expectation :=
  @FormalProofs.OPT.mathlib_rn_deriv_ae_eq_conditional_expectation

/-- Mathlib conditional expectation under independence. -/
abbrev mathlib_conditional_expectation_independent_eq_integral :=
  @FormalProofs.OPT.mathlib_conditional_expectation_independent_eq_integral

/-- Mathlib scalar multiplication for conditional expectation. -/
abbrev mathlib_conditional_expectation_smul :=
  @FormalProofs.OPT.mathlib_conditional_expectation_smul

/-- Mathlib monotonicity of conditional expectation. -/
abbrev mathlib_conditional_expectation_mono :=
  @FormalProofs.OPT.mathlib_conditional_expectation_mono

/-- Mathlib kernel/disintegration posterior `posterior κ μ`, notation `κ†μ`. -/
abbrev mathlib_kernel_posterior :=
  @FormalProofs.OPT.mathlib_kernel_posterior

/-- Mathlib kernel posterior defining identity for the swapped joint law. -/
abbrev mathlib_kernel_posterior_compProd_eq_map_swap :=
  @FormalProofs.OPT.mathlib_kernel_posterior_compProd_eq_map_swap

/-- Mathlib countable-parameter posterior density/Bayes formula. -/
abbrev mathlib_kernel_posterior_with_density_countable :=
  @FormalProofs.OPT.mathlib_kernel_posterior_with_density_countable

/-- Mathlib posterior density/Bayes formula under an absolute-continuity
assumption. -/
abbrev mathlib_kernel_posterior_eq_with_density :=
  @FormalProofs.OPT.mathlib_kernel_posterior_eq_with_density

/-- Mathlib posterior Radon-Nikodym derivative identity. -/
abbrev mathlib_kernel_posterior_rn_deriv :=
  @FormalProofs.OPT.mathlib_kernel_posterior_rn_deriv

/-- Mathlib posterior uniqueness up to a.e. equality from the swapped joint-law
identity. -/
abbrev mathlib_kernel_posterior_unique_ae :=
  @FormalProofs.OPT.mathlib_kernel_posterior_unique_ae

/-- Mathlib theorem that composing a kernel with its posterior recovers the
prior measure. -/
abbrev mathlib_kernel_posterior_comp_self :=
  @FormalProofs.OPT.mathlib_kernel_posterior_comp_self

/-- Mathlib theorem that posterior inversion is involutive up to a.e. equality. -/
abbrev mathlib_kernel_posterior_posterior :=
  @FormalProofs.OPT.mathlib_kernel_posterior_posterior

/-- Mathlib theorem that posterior kernels compose contravariantly. -/
abbrev mathlib_kernel_posterior_comp :=
  @FormalProofs.OPT.mathlib_kernel_posterior_comp

/-- Mathlib `HasPDF` class for dominated random variables. -/
abbrev mathlib_has_pdf :=
  @FormalProofs.OPT.mathlib_has_pdf

/-- Mathlib PDF as a Radon-Nikodym derivative of the pushforward law. -/
abbrev mathlib_pdf :=
  @FormalProofs.OPT.mathlib_pdf

/-- Mathlib law-as-with-density theorem for random variables with PDFs. -/
abbrev mathlib_pdf_map_eq_with_density :=
  @FormalProofs.OPT.mathlib_pdf_map_eq_with_density

/-- Mathlib setwise density formula for measurable sets. -/
abbrev mathlib_pdf_map_eq_set_lintegral :=
  @FormalProofs.OPT.mathlib_pdf_map_eq_set_lintegral

/-- Mathlib nonnegative LOTUS theorem for PDFs. -/
abbrev mathlib_pdf_lintegral_lotus :=
  @FormalProofs.OPT.mathlib_pdf_lintegral_lotus

/-- Mathlib discrete probability mass function type. -/
abbrev mathlib_probability_mass_function :=
  @FormalProofs.OPT.mathlib_probability_mass_function

/-- Mathlib PMF-to-measure construction. -/
abbrev mathlib_pmf_to_measure :=
  @FormalProofs.OPT.mathlib_pmf_to_measure

/-- Mathlib finite-type formula for PMF-induced measures. -/
abbrev mathlib_pmf_to_measure_apply_fintype :=
  @FormalProofs.OPT.mathlib_pmf_to_measure_apply_fintype

/-- Mathlib injectivity of the PMF-to-measure construction. -/
abbrev mathlib_pmf_to_measure_inj :=
  @FormalProofs.OPT.mathlib_pmf_to_measure_inj

/-- Mathlib convergence-in-measure predicate used for convergence in
probability in this repo's posterior-consistency layer. -/
abbrev mathlib_tendsto_in_measure :=
  @FormalProofs.OPT.mathlib_tendsto_in_measure

/-- Mathlib congruence theorem for convergence in measure. -/
abbrev mathlib_tendsto_in_measure_congr :=
  @FormalProofs.OPT.mathlib_tendsto_in_measure_congr

/-- Mathlib subsequence-a.e. theorem from convergence in measure. -/
abbrev mathlib_tendsto_in_measure_exists_seq_tendsto_ae :=
  @FormalProofs.OPT.mathlib_tendsto_in_measure_exists_seq_tendsto_ae

/-- Finite Bayes posterior normalizes when evidence is nonzero. -/
abbrev finite_bayes_posterior_sum_eq_one :=
  @FormalProofs.OPT.bayesPosterior_sum_eq_one

/-- Finite Bayes posterior masses are nonnegative under prior, likelihood, and
evidence positivity assumptions. -/
abbrev finite_bayes_posterior_nonneg :=
  @FormalProofs.OPT.bayesPosterior_nonneg

/-- Finite real-valued score maximizer predicate used for MAP decisions. -/
abbrev finite_score_map :=
  @FormalProofs.OPT.IsFiniteScoreMAP

/-- Finite real-valued score minimizer predicate used for Bayes actions. -/
abbrev finite_score_argmin :=
  @FormalProofs.OPT.IsFiniteScoreArgmin

/-- MAP predicate for unnormalized finite Bayes numerators. -/
abbrev finite_bayes_numerator_map :=
  @FormalProofs.OPT.BayesNumeratorMAP

/-- MAP predicate for normalized finite Bayes posteriors. -/
abbrev finite_bayes_posterior_map :=
  @FormalProofs.OPT.BayesPosteriorMAP

/-- MAP predicate for unnormalized state Bayes numerators. -/
abbrev state_finite_bayes_numerator_map :=
  @FormalProofs.OPT.StateBayesNumeratorMAP

/-- MAP predicate for normalized state finite Bayes posteriors. -/
abbrev state_finite_bayes_posterior_map :=
  @FormalProofs.OPT.StateBayesPosteriorMAP

/-- Positive normalization preserves finite Bayes MAP decisions. -/
abbrev finite_bayes_posterior_map_iff_numerator_map :=
  @FormalProofs.OPT.bayesPosteriorMAP_iff_bayesNumeratorMAP

/-- Positive state normalization preserves state finite Bayes MAP decisions. -/
abbrev state_finite_bayes_posterior_map_iff_numerator_map :=
  @FormalProofs.OPT.stateBayesPosteriorMAP_iff_stateBayesNumeratorMAP

/-- Posterior odds cancel the evidence and equal Bayes-numerator odds. -/
abbrev finite_bayes_posterior_odds_eq_numerator_odds :=
  @FormalProofs.OPT.bayesPosterior_odds_eq_bayesNumerator_odds

/-- State posterior odds cancel the state evidence and equal state
Bayes-numerator odds. -/
abbrev state_finite_bayes_posterior_odds_eq_numerator_odds :=
  @FormalProofs.OPT.stateBayesPosterior_odds_eq_stateBayesNumerator_odds

/-- Finite posterior expectation/readout for posterior functionals. -/
abbrev finite_bayes_posterior_expectation :=
  @FormalProofs.OPT.BayesPosteriorExpectation

/-- State finite posterior expectation/readout for posterior functionals. -/
abbrev state_finite_bayes_posterior_expectation :=
  @FormalProofs.OPT.StateBayesPosteriorExpectation

/-- Likelihood-on-state finite posterior expectations equal state posterior
expectations. -/
abbrev finite_bayes_posterior_expectation_likelihood_on_state_eq_state :=
  @FormalProofs.OPT.bayesPosteriorExpectation_likelihoodOnState_eq_state

/-- Fixed-prior finite posterior expectations are determined by the likelihood
family. -/
abbrev finite_bayes_posterior_expectation_determined_by_likelihood :=
  @FormalProofs.OPT.bayesPosteriorExpectation_determinedByLikelihood

/-- Likelihood-on-state finite posterior expectations are state-sufficient. -/
abbrev finite_bayes_posterior_expectation_likelihood_on_state_sufficient :=
  @FormalProofs.OPT.bayesPosteriorExpectation_likelihoodOnState_sufficient

/-- Finite posterior predictive likelihood. -/
abbrev finite_bayes_posterior_predictive :=
  @FormalProofs.OPT.BayesPosteriorPredictive

/-- State finite posterior predictive likelihood. -/
abbrev state_finite_bayes_posterior_predictive :=
  @FormalProofs.OPT.StateBayesPosteriorPredictive

/-- Likelihood-on-state finite posterior predictive likelihoods equal state
posterior predictive likelihoods. -/
abbrev finite_bayes_posterior_predictive_likelihood_on_state_eq_state :=
  @FormalProofs.OPT.bayesPosteriorPredictive_likelihoodOnState_eq_state

/-- For a fixed future observation, likelihood-on-state finite posterior
predictives are sufficient in the observed learned state. -/
abbrev finite_bayes_posterior_predictive_likelihood_on_state_sufficient_observed :=
  @FormalProofs.OPT.bayesPosteriorPredictive_likelihoodOnState_sufficient_observed

/-- Finite posterior Bayes risk of an action. -/
abbrev finite_bayes_posterior_risk :=
  @FormalProofs.OPT.BayesPosteriorRisk

/-- State-space finite posterior Bayes risk of an action. -/
abbrev state_finite_bayes_posterior_risk :=
  @FormalProofs.OPT.StateBayesPosteriorRisk

/-- Finite Bayes-action predicate. -/
abbrev finite_bayes_action :=
  @FormalProofs.OPT.BayesAction

/-- State-space finite Bayes-action predicate. -/
abbrev state_finite_bayes_action :=
  @FormalProofs.OPT.StateBayesAction

/-- Likelihood-on-state finite Bayes risks equal state posterior risks. -/
abbrev finite_bayes_posterior_risk_likelihood_on_state_eq_state :=
  @FormalProofs.OPT.bayesPosteriorRisk_likelihoodOnState_eq_state

/-- Likelihood-on-state finite Bayes risks are state-sufficient. -/
abbrev finite_bayes_posterior_risk_likelihood_on_state_sufficient :=
  @FormalProofs.OPT.bayesPosteriorRisk_likelihoodOnState_sufficient

/-- Bayes-action optimality transports across likelihood-on-state
factorization. -/
abbrev finite_bayes_action_likelihood_on_state_iff_state_action :=
  @FormalProofs.OPT.bayesAction_likelihoodOnState_iff_stateBayesAction

/-- Finite posterior mass assigned to a parameter event. -/
abbrev finite_bayes_posterior_set_mass :=
  @FormalProofs.OPT.BayesPosteriorSetMass

/-- State-space finite posterior mass assigned to a parameter event. -/
abbrev state_finite_bayes_posterior_set_mass :=
  @FormalProofs.OPT.StateBayesPosteriorSetMass

/-- Finite credible/acceptance-set predicate at a posterior-mass level. -/
abbrev finite_bayes_credible_at_level :=
  @FormalProofs.OPT.BayesCredibleAtLevel

/-- State-space finite credible/acceptance-set predicate. -/
abbrev state_finite_bayes_credible_at_level :=
  @FormalProofs.OPT.StateBayesCredibleAtLevel

/-- Likelihood-on-state posterior event masses equal state posterior event
masses. -/
abbrev finite_bayes_posterior_set_mass_likelihood_on_state_eq_state :=
  @FormalProofs.OPT.bayesPosteriorSetMass_likelihoodOnState_eq_state

/-- Likelihood-on-state credible/acceptance-set claims are equivalent to
state-space claims. -/
abbrev finite_bayes_credible_at_level_likelihood_on_state_iff_state :=
  @FormalProofs.OPT.bayesCredibleAtLevel_likelihoodOnState_iff_state

/-- Likelihood-on-state posterior event masses are state-sufficient. -/
abbrev finite_bayes_posterior_set_mass_likelihood_on_state_sufficient :=
  @FormalProofs.OPT.bayesPosteriorSetMass_likelihoodOnState_sufficient

/-- Evidence-ratio remainder for finite Bayes target-posterior algebra. -/
abbrev finite_bayes_evidence_ratio_remainder :=
  @FormalProofs.OPT.BayesEvidenceRatioRemainder

/-- State-space evidence-ratio remainder. -/
abbrev state_finite_bayes_evidence_ratio_remainder :=
  @FormalProofs.OPT.StateBayesEvidenceRatioRemainder

/-- Target posterior mass equals inverse one-plus evidence-ratio remainder. -/
abbrev finite_bayes_posterior_target_eq_inv_one_plus_evidence_ratio_remainder :=
  @FormalProofs.OPT.bayesPosterior_target_eq_inv_one_plus_evidenceRatioRemainder

/-- State-space target posterior mass equals inverse one-plus evidence-ratio
remainder. -/
abbrev state_finite_bayes_posterior_target_eq_inv_one_plus_evidence_ratio_remainder :=
  @FormalProofs.OPT.stateBayesPosterior_target_eq_inv_one_plus_evidenceRatioRemainder

/-- Finite Bayes posterior packaged as a mathlib `PMF` under positivity
assumptions. -/
abbrev finite_bayes_posterior_pmf :=
  @FormalProofs.OPT.BayesPosteriorPMF

/-- The finite Bayes PMF's induced measure has singleton masses matching the
posterior mass function. -/
abbrev finite_bayes_posterior_pmf_to_measure_singleton :=
  @FormalProofs.OPT.bayesPosteriorPMF_toMeasure_singleton

/-- The finite Bayes PMF's induced measure has arbitrary-event masses matching
the finite sum of posterior masses. -/
abbrev finite_bayes_posterior_pmf_to_measure_set :=
  @FormalProofs.OPT.bayesPosteriorPMF_toMeasure_set

/-- State finite Bayes posterior packaged as a mathlib `PMF` under positivity
assumptions. -/
abbrev state_finite_bayes_posterior_pmf :=
  @FormalProofs.OPT.StateBayesPosteriorPMF

/-- The state finite Bayes PMF's induced measure has singleton masses matching
the posterior mass function. -/
abbrev state_finite_bayes_posterior_pmf_to_measure_singleton :=
  @FormalProofs.OPT.stateBayesPosteriorPMF_toMeasure_singleton

/-- The state finite Bayes PMF's induced measure has arbitrary-event masses
matching the finite sum of posterior masses. -/
abbrev state_finite_bayes_posterior_pmf_to_measure_set :=
  @FormalProofs.OPT.stateBayesPosteriorPMF_toMeasure_set

/-- If the likelihood factors through state, the raw finite Bayes PMF equals
the state finite Bayes PMF. -/
abbrev finite_bayes_posterior_pmf_likelihood_on_state_eq_state_pmf :=
  @FormalProofs.OPT.bayesPosteriorPMF_likelihoodOnState_eq_stateBayesPosteriorPMF

/-- Fixed-prior finite Bayes posterior is determined by the likelihood family. -/
abbrev finite_bayes_posterior_determined_by_likelihood :=
  @FormalProofs.OPT.bayesPosterior_determinedByLikelihood

/-- Likelihood sufficiency implies finite-Bayes posterior sufficiency. -/
abbrev likelihood_sufficient_implies_finite_bayes_posterior_sufficient :=
  @FormalProofs.OPT.likelihoodSufficient_implies_bayesPosteriorSufficient

/-- A state likelihood induces a posterior-on-state readout by finite Bayes. -/
abbrev finite_bayes_posterior_likelihood_on_state_eq_posterior_on_state :=
  @FormalProofs.OPT.bayesPosterior_likelihoodOnState_eq_posteriorOnState

/-- Finite-Bayes posterior sufficiency for likelihood-on-state families. -/
abbrev finite_bayes_posterior_likelihood_on_state_sufficient :=
  @FormalProofs.OPT.bayesPosterior_likelihoodOnState_sufficient

/-- A richer representation with a decoder to the state is sufficient for the
finite-Bayes posterior induced by a state likelihood. -/
abbrev rep_with_state_readout_finite_bayes_posterior_likelihood_on_state_sufficient :=
  @FormalProofs.OPT.repWithStateReadout_bayesPosterior_likelihoodOnState_sufficient

/-- Surjective-state factorization for finite-Bayes posteriors. -/
abbrev surjective_state_finite_bayes_posterior_factorization :=
  @FormalProofs.OPT.surjectiveState_bayesPosterior_factorization

/-- Posterior consistency as convergence in probability in a posterior metric
space. -/
abbrev posterior_consistent :=
  @FormalProofs.OPT.PosteriorConsistent

/-- Posterior consistency is exactly mathlib `TendstoInMeasure` along
`Filter.atTop`. -/
abbrev posterior_consistent_iff_mathlib_tendsto_in_measure :=
  @FormalProofs.OPT.posteriorConsistent_iff_mathlib_tendstoInMeasure

/-- Finite-parameter posterior mass concentrates on a target parameter. -/
abbrev finite_posterior_mass_concentrates_at :=
  @FormalProofs.OPT.FinitePosteriorMassConcentratesAt

/-- Finite posterior mass concentration is exactly mathlib `TendstoInMeasure`
for the target parameter's posterior mass. -/
abbrev finite_posterior_mass_concentrates_at_iff_mathlib_tendsto_in_measure :=
  @FormalProofs.OPT.finitePosteriorMassConcentratesAt_iff_mathlib_tendstoInMeasure

/-- Finite Bayes posterior sequence induced by raw observations. -/
abbrev finite_bayes_posterior_seq :=
  @FormalProofs.OPT.FiniteBayesPosteriorSeq

/-- Finite Bayes posterior sequence induced by learned states. -/
abbrev state_finite_bayes_posterior_seq :=
  @FormalProofs.OPT.StateFiniteBayesPosteriorSeq

/-- Evidence-ratio remainder sequence for finite Bayes observations. -/
abbrev finite_bayes_evidence_ratio_remainder_seq :=
  @FormalProofs.OPT.FiniteBayesEvidenceRatioRemainderSeq

/-- State-space evidence-ratio remainder sequence. -/
abbrev state_finite_bayes_evidence_ratio_remainder_seq :=
  @FormalProofs.OPT.StateFiniteBayesEvidenceRatioRemainderSeq

/-- Posterior-transform concentration for finite evidence-ratio remainders. -/
abbrev finite_bayes_evidence_ratio_posterior_transform_concentrates_at_one :=
  @FormalProofs.OPT.FiniteBayesEvidenceRatioPosteriorTransformConcentratesAtOne

/-- Posterior-transform concentration for state evidence-ratio remainders. -/
abbrev state_finite_bayes_evidence_ratio_posterior_transform_concentrates_at_one :=
  @FormalProofs.OPT.StateFiniteBayesEvidenceRatioPosteriorTransformConcentratesAtOne

/-- Finite likelihood-ratio/evidence-ratio sufficient-condition bundle. -/
abbrev finite_bayes_likelihood_ratio_consistency_condition :=
  @FormalProofs.OPT.FiniteBayesLikelihoodRatioConsistencyCondition

/-- State-space likelihood-ratio/evidence-ratio sufficient-condition bundle. -/
abbrev state_finite_bayes_likelihood_ratio_consistency_condition :=
  @FormalProofs.OPT.StateFiniteBayesLikelihoodRatioConsistencyCondition

/-- Evidence-ratio posterior-transform concentration implies finite posterior
mass concentration. -/
abbrev finite_bayes_posterior_mass_concentration_of_evidence_ratio_transform :=
  @FormalProofs.OPT.finiteBayesPosteriorMassConcentration_of_evidenceRatioTransform

/-- Finite likelihood-ratio/evidence-ratio condition implies posterior mass
concentration. -/
abbrev finite_bayes_posterior_mass_concentration_of_likelihood_ratio_condition :=
  @FormalProofs.OPT.finiteBayesPosteriorMassConcentration_of_likelihoodRatioCondition

/-- State evidence-ratio posterior-transform concentration implies state
finite posterior mass concentration. -/
abbrev state_finite_bayes_posterior_mass_concentration_of_evidence_ratio_transform :=
  @FormalProofs.OPT.stateFiniteBayesPosteriorMassConcentration_of_evidenceRatioTransform

/-- State likelihood-ratio/evidence-ratio condition implies state posterior
mass concentration. -/
abbrev state_finite_bayes_posterior_mass_concentration_of_likelihood_ratio_condition :=
  @FormalProofs.OPT.stateFiniteBayesPosteriorMassConcentration_of_likelihoodRatioCondition

/-- Pointwise equality preserves posterior consistency. -/
abbrev posterior_consistency_of_pointwise_equal :=
  @FormalProofs.OPT.posteriorConsistency_of_pointwise_equal

/-- Pointwise equality preserves finite posterior mass concentration. -/
abbrev finite_posterior_mass_concentration_of_pointwise_equal :=
  @FormalProofs.OPT.finitePosteriorMassConcentration_of_pointwise_equal

/-- Explicit assumption bundle for finite Bayes posterior consistency. -/
abbrev finite_bayes_posterior_consistency_assumption :=
  @FormalProofs.OPT.FiniteBayesPosteriorConsistencyAssumption

/-- Explicit assumption bundle for state-space finite Bayes posterior
consistency. -/
abbrev state_finite_bayes_posterior_consistency_assumption :=
  @FormalProofs.OPT.StateFiniteBayesPosteriorConsistencyAssumption

/-- For likelihood-on-state finite Bayes, raw posterior concentration is
equivalent to state posterior concentration. -/
abbrev finite_bayes_consistency_likelihood_on_state_iff :=
  @FormalProofs.OPT.finiteBayesConsistency_likelihoodOnState_iff

/-- Exact state readout transports finite Bayes posterior concentration. -/
abbrev state_readout_finite_bayes_consistency :=
  @FormalProofs.OPT.stateReadout_finiteBayesConsistency

/-!
## Bayesian Persuasion / Information Design Exports

These aliases expose the finite Kamenica--Gentzkow persuasion algebra.  The
canonical article is Kamenica and Gentzkow (AER 2011), not Econometrica.
The Lean layer is finite and assumption-backed: experiments induce Bayes
posteriors, valid full-support experiments induce Bayes-plausible posterior
distributions, receiver best responses are finite Bayes actions for negative
utility loss, and concavification is represented by a symbolic optimal-value
witness.
-/

/-- Finite probability-vector predicate for beliefs and signal distributions. -/
abbrev bayesian_persuasion_finite_probability :=
  @FormalProofs.OPT.IsFiniteProbability

/-- Finite signal experiment validity: each state has a signal-probability
vector. -/
abbrev bayesian_persuasion_signal_experiment_valid :=
  @FormalProofs.OPT.SignalExperimentValid

/-- Signal distribution induced by a prior and finite experiment. -/
abbrev bayesian_persuasion_signal_distribution :=
  @FormalProofs.OPT.SignalDistribution

/-- Finite posterior belief after a signal realization. -/
abbrev bayesian_persuasion_posterior_after_signal :=
  @FormalProofs.OPT.PosteriorAfterSignal

/-- Full-support predicate for retained signal realizations. -/
abbrev bayesian_persuasion_signal_full_support :=
  @FormalProofs.OPT.SignalDistributionFullSupport

/-- Posterior-after-signal equals the repo's finite Bayes posterior. -/
abbrev bayesian_persuasion_posterior_eq_finite_bayes :=
  @FormalProofs.OPT.posteriorAfterSignal_eq_bayesPosterior

/-- A valid finite experiment induces a finite probability distribution over
signals. -/
abbrev bayesian_persuasion_signal_distribution_probability :=
  @FormalProofs.OPT.signalDistribution_isFiniteProbability

/-- Bayes-plausibility barycenter condition for finite posterior
distributions. -/
abbrev bayesian_persuasion_bayes_plausible_distribution :=
  @FormalProofs.OPT.BayesPlausiblePosteriorDistribution

/-- Valid full-support experiments induce Bayes-plausible posterior
distributions. -/
abbrev bayesian_persuasion_valid_signal_bayes_plausible :=
  @FormalProofs.OPT.validSignalExperiment_bayesPlausible_of_fullSupport

/-- Valid full-support experiments induce feasible finite persuasion schemes. -/
abbrev bayesian_persuasion_valid_signal_scheme_feasible :=
  @FormalProofs.OPT.validSignalExperiment_persuasionSchemeFeasible_of_fullSupport

/-- Strong finite persuasion scheme where posterior labels are finite
probability vectors. -/
abbrev bayesian_persuasion_scheme_belief_feasible :=
  @FormalProofs.OPT.PersuasionSchemeBeliefFeasible

/-- Explicit finite splitting experiment implementing a Bayes-plausible
posterior decomposition. -/
abbrev bayesian_persuasion_splitting_experiment :=
  @FormalProofs.OPT.SplittingExperiment

/-- Finite splitting construction: a Bayes-plausible posterior distribution
with positive prior support defines a valid signal experiment. -/
abbrev bayesian_persuasion_splitting_experiment_valid :=
  @FormalProofs.OPT.splittingExperiment_valid_of_bayesPlausible

/-- The splitting experiment induces the supplied signal weights. -/
abbrev bayesian_persuasion_splitting_signal_distribution_eq_weight :=
  @FormalProofs.OPT.signalDistribution_splittingExperiment_eq_weight

/-- Positive-weight signals from the splitting construction recover the
supplied posterior beliefs. -/
abbrev bayesian_persuasion_splitting_posterior_eq :=
  @FormalProofs.OPT.posteriorAfterSignal_splittingExperiment_eq_posterior

/-- Receiver expected utility at a posterior belief. -/
abbrev bayesian_persuasion_receiver_expected_utility :=
  @FormalProofs.OPT.ReceiverExpectedUtility

/-- Sender expected utility at a posterior belief and receiver action. -/
abbrev bayesian_persuasion_sender_expected_utility :=
  @FormalProofs.OPT.SenderExpectedUtility

/-- Receiver best-response predicate. -/
abbrev bayesian_persuasion_receiver_optimal_action :=
  @FormalProofs.OPT.ReceiverOptimalAction

/-- Sender-preferred receiver best response under optimistic tie-breaking. -/
abbrev bayesian_persuasion_sender_preferred_receiver_best_response :=
  @FormalProofs.OPT.SenderPreferredReceiverBestResponse

/-- Posterior risk for negative receiver utility is negative receiver expected
utility. -/
abbrev bayesian_persuasion_negative_utility_risk_eq :=
  @FormalProofs.OPT.bayesPosteriorRisk_negativeReceiverUtility_eq_neg_expectedUtility

/-- Receiver best responses are exactly finite Bayes actions for negative
receiver utility loss. -/
abbrev bayesian_persuasion_receiver_bayes_action_iff_best_response :=
  @FormalProofs.OPT.bayesAction_negativeReceiverUtility_iff_receiverOptimalAction

/-- Feasible finite persuasion scheme. -/
abbrev bayesian_persuasion_scheme_feasible :=
  @FormalProofs.OPT.PersuasionSchemeFeasible

/-- Expected value of a finite persuasion scheme. -/
abbrev bayesian_persuasion_scheme_value :=
  @FormalProofs.OPT.PersuasionSchemeValue

/-- Symbolic concavification witness for the finite persuasion optimum. -/
abbrev bayesian_persuasion_concavification_witness :=
  @FormalProofs.OPT.ConcavificationWitness

/-- Symbolic optimal persuasion value predicate. -/
abbrev bayesian_persuasion_optimal_value :=
  @FormalProofs.OPT.IsOptimalPersuasionValue

/-- A supplied finite concavification witness is exactly an optimal persuasion
value witness in the symbolic surface. -/
abbrev bayesian_persuasion_concavification_iff_optimal_value :=
  @FormalProofs.OPT.concavificationWitness_iff_optimalPersuasionValue

/-!
### Bayesian-persuasion economic-formulation exports
-/

/-- Posterior-belief state induced by a signal experiment. -/
abbrev bayesian_persuasion_belief_state :=
  @FormalProofs.OPT.PersuasionBeliefState

/-- Receiver posterior loss is negative receiver expected utility at a belief. -/
abbrev bayesian_persuasion_receiver_posterior_loss :=
  @FormalProofs.OPT.ReceiverPosteriorLoss

/-- Receiver posterior loss over signals factors through the induced posterior
belief state. -/
abbrev bayesian_persuasion_receiver_loss_factors_through_belief :=
  @FormalProofs.OPT.receiverPosteriorLoss_factorsThroughBeliefState

/-- Belief-indexed receiver best-response selector. -/
abbrev bayesian_persuasion_receiver_best_response_selector :=
  @FormalProofs.OPT.ReceiverBestResponseSelector

/-- Belief-indexed sender-preferred receiver best-response selector. -/
abbrev bayesian_persuasion_sender_tie_breaking_selector :=
  @FormalProofs.OPT.SenderTieBreakingSelector

/-- Sender tie-breaking selectors are receiver best-response selectors. -/
abbrev bayesian_persuasion_sender_tie_breaking_implies_best_response_selector :=
  @FormalProofs.OPT.senderTieBreakingSelector_receiverBestResponseSelector

/-- Sender indirect value induced by a belief-indexed receiver-action
selector. -/
abbrev bayesian_persuasion_sender_indirect_value_of_selector :=
  @FormalProofs.OPT.SenderIndirectValueOfSelector

/-- Sender indirect value over signals factors through the induced posterior
belief state. -/
abbrev bayesian_persuasion_sender_indirect_value_factors_through_belief :=
  @FormalProofs.OPT.senderIndirectValue_factorsThroughBeliefState

/-- Sender value of a concrete signal experiment and signal-indexed action
rule. -/
abbrev bayesian_persuasion_signal_experiment_sender_value :=
  @FormalProofs.OPT.SignalExperimentSenderValue

/-- Sender value of a signal experiment under a belief-indexed action
selector. -/
abbrev bayesian_persuasion_signal_experiment_indirect_value :=
  @FormalProofs.OPT.SignalExperimentIndirectValue

/-- Indirect experiment value is exactly persuasion-scheme value for the
experiment-induced posterior distribution. -/
abbrev bayesian_persuasion_signal_indirect_value_eq_scheme_value :=
  @FormalProofs.OPT.signalExperimentIndirectValue_eq_persuasionSchemeValue

/-- Concrete experiment value equals indirect value when the action rule is
generated by a belief selector. -/
abbrev bayesian_persuasion_signal_sender_value_eq_indirect_of_selector :=
  @FormalProofs.OPT.signalExperimentSenderValue_eq_indirectValue_of_selector

/-- Direct recommendation obedience predicate. -/
abbrev bayesian_persuasion_receiver_obedient_recommendation :=
  @FormalProofs.OPT.ReceiverObedientRecommendation

/-- Direct recommendation obedience iff finite Bayes-action optimality under
negative receiver utility loss. -/
abbrev bayesian_persuasion_receiver_obedient_iff_bayes_action :=
  @FormalProofs.OPT.receiverObedientRecommendation_iff_bayesAction_negativeUtility

/-- Sender value of a direct recommendation experiment. -/
abbrev bayesian_persuasion_direct_recommendation_sender_value :=
  @FormalProofs.OPT.DirectRecommendationSenderValue

/-- Direct recommendation value is concrete signal-experiment value with the
identity action rule. -/
abbrev bayesian_persuasion_direct_recommendation_value_eq_signal_value :=
  @FormalProofs.OPT.directRecommendationSenderValue_eq_signalExperimentSenderValue

/-- Same signal-indexed distribution over posterior beliefs for two
experiments. -/
abbrev bayesian_persuasion_same_posterior_distribution :=
  @FormalProofs.OPT.SamePosteriorDistribution

/-- Same posterior distribution is reflexive. -/
abbrev bayesian_persuasion_same_posterior_distribution_refl :=
  @FormalProofs.OPT.samePosteriorDistribution_refl

/-- Belief-based indirect value is invariant under same posterior
distribution. -/
abbrev bayesian_persuasion_indirect_value_eq_of_same_posterior_distribution :=
  @FormalProofs.OPT.signalExperimentIndirectValue_eq_of_samePosteriorDistribution

/-- Concrete signal-experiment value is invariant under same posterior
distribution when the signal-indexed action rule is the same. -/
abbrev bayesian_persuasion_signal_value_eq_of_same_posterior_distribution :=
  @FormalProofs.OPT.signalExperimentSenderValue_eq_of_samePosteriorDistribution

/-!
### Bayesian-persuasion direct-recommendation exports
-/

/-- Action-valued direct recommendation experiment obtained by pooling original
signals through a deterministic signal-to-action rule. -/
abbrev bayesian_persuasion_direct_recommendation_from_experiment :=
  @FormalProofs.OPT.DirectRecommendationFromExperiment

/-- Pooled direct recommendations are valid experiments when the original
signal experiment is valid. -/
abbrev bayesian_persuasion_direct_recommendation_from_experiment_valid :=
  @FormalProofs.OPT.directRecommendationFromExperiment_valid

/-- Ex-ante sender value of a signal experiment. -/
abbrev bayesian_persuasion_signal_ex_ante_sender_value :=
  @FormalProofs.OPT.SignalExperimentExAnteSenderValue

/-- Ex-ante sender value of a direct recommendation experiment. -/
abbrev bayesian_persuasion_direct_recommendation_ex_ante_sender_value :=
  @FormalProofs.OPT.DirectRecommendationExAnteSenderValue

/-- Inner regrouping identity for direct-recommendation sender value. -/
abbrev bayesian_persuasion_direct_recommendation_inner_sender_value_eq :=
  @FormalProofs.OPT.directRecommendationFromExperiment_inner_senderValue_eq

/-- Pooling signals by their receiver action preserves ex-ante sender value. -/
abbrev bayesian_persuasion_direct_recommendation_ex_ante_sender_value_eq :=
  @FormalProofs.OPT.directRecommendationFromExperiment_exAnte_senderValue_eq

/-- Posterior signal-experiment value equals ex-ante sender value under
full-support signal probabilities. -/
abbrev bayesian_persuasion_signal_value_eq_ex_ante_sender_value :=
  @FormalProofs.OPT.signalExperimentSenderValue_eq_exAnteSenderValue

/-- Posterior direct-recommendation value equals ex-ante sender value under
full-support recommendation probabilities. -/
abbrev bayesian_persuasion_direct_recommendation_value_eq_ex_ante_sender_value :=
  @FormalProofs.OPT.directRecommendationSenderValue_eq_exAnteSenderValue

/-- Full-support posterior-value version of direct-recommendation value
preservation. -/
abbrev bayesian_persuasion_direct_recommendation_sender_value_eq :=
  @FormalProofs.OPT.directRecommendationFromExperiment_senderValue_eq

/-- Likelihood-free sufficient representation: the representation preserves all
simulator/probe/contextual responses. -/
abbrev likelihood_free_response_sufficient :=
  @FormalProofs.OPT.LikelihoodFreeResponseSufficient

/-- Likelihood-free response sufficiency is contextual query sufficiency with
probes as contexts. -/
abbrev likelihood_free_response_sufficient_iff_contextual_query_sufficient :=
  @FormalProofs.OPT.likelihoodFreeResponseSufficient_iff_querySufficient

/-- Likelihood-free response sufficiency is equivalent to factoring the response
family through a representation readout. -/
abbrev likelihood_free_response_sufficient_iff_exists_readout :=
  @FormalProofs.OPT.likelihoodFreeResponseSufficient_iff_exists_readout

/-- Likelihood-free sufficiency forbids collisions distinguished by any probe. -/
abbrev likelihood_free_response_sufficient_no_collision_of_distinguished_probe :=
  @FormalProofs.OPT.likelihoodFreeResponseSufficient_no_collision_of_distinguished_probe

/-- Two-sided contextual sufficiency is likelihood-free response sufficiency
with two-sided contexts as probes. -/
abbrev twoSided_context_sufficient_iff_likelihood_free_response_sufficient :=
  @FormalProofs.OPT.twoSidedContextSufficient_iff_likelihoodFreeResponseSufficient

/-- Sliced sufficiency plus cover implies likelihood-free response sufficiency. -/
abbrev sliced_sufficiency_implies_likelihood_free_response_sufficient :=
  @FormalProofs.OPT.slicedQuerySufficient_implies_likelihoodFreeResponseSufficient

/-- Finite sliced zero-loss plus cover implies likelihood-free response
sufficiency. -/
abbrev finite_sliced_zeroLoss_implies_likelihood_free_response_sufficient :=
  @FormalProofs.OPT.finiteSliced_zeroLoss_implies_likelihoodFreeResponseSufficient

/-- Exact shared-`g` instance for ordinary bag-of-words LDA over histogram
leaves. -/
abbrev bagOfWords_exact_g :=
  @FormalProofs.OPT.bagOfWordsExactG

/-- The exact bag-of-words shared `g` folds histogram leaves by histogram
addition. -/
abbrev bagOfWords_exact_g_treeEval_eq_bagOfWordsTree :=
  @FormalProofs.OPT.bagOfWordsExactG_treeEval_eq_bagOfWordsTree

/-- Ordinary bag-of-words LDA likelihood is recovered exactly by the shared
endomorphic `g` on histogram observations. -/
abbrev lda_histogram_likelihood_exact_uniformG :=
  @FormalProofs.OPT.ldaHistogramLikelihood_exact_uniformG

/-- Bag-of-words LDA likelihood over a tree factors as the product of leaf
likelihoods. -/
abbrev lda_histogram_likelihood_bagOfWordsTree_eq_leaf_prod :=
  @FormalProofs.OPT.ldaHistogramLikelihood_bagOfWordsTree_eq_leaf_prod

/-- The exact shared bag-of-words `g` recovers the product-of-leaf likelihood
factorization. -/
abbrev lda_histogram_likelihood_uniformG_eq_leaf_prod :=
  @FormalProofs.OPT.ldaHistogramLikelihood_uniformG_eq_leaf_prod

/-- Root token mass is the sum of leaf token masses for a bag-of-words tree. -/
abbrev histogram_token_mass_bagOfWordsTree_eq_leaf_sum :=
  @FormalProofs.OPT.histogramTokenMass_bagOfWordsTree_eq_leaf_sum

/-- Root LDA log-likelihood is the sum of leaf LDA log-likelihoods. -/
abbrev lda_histogram_log_likelihood_bagOfWordsTree_eq_leaf_sum :=
  @FormalProofs.OPT.ldaHistogramLogLikelihood_bagOfWordsTree_eq_leaf_sum

/-- The exact shared bag-of-words `g` recovers the sum-of-leaf LDA
log-likelihood. -/
abbrev lda_histogram_log_likelihood_uniformG_eq_leaf_sum :=
  @FormalProofs.OPT.ldaHistogramLogLikelihood_uniformG_eq_leaf_sum

/-- Average document log-likelihood is the token-weighted average of leaf
average log-likelihoods. -/
abbrev lda_average_log_likelihood_bagOfWordsTree_eq_token_weighted_leaf_average :=
  @FormalProofs.OPT.ldaAverageLogLikelihood_bagOfWordsTree_eq_tokenWeightedLeafAverage

/-- The exact shared bag-of-words `g` recovers the token-weighted average
log-likelihood decomposition. -/
abbrev lda_average_log_likelihood_uniformG_eq_token_weighted_leaf_average :=
  @FormalProofs.OPT.ldaAverageLogLikelihood_uniformG_eq_tokenWeightedLeafAverage

/-- Document topic proportions are token-weighted averages of leaf topic
proportions. -/
abbrev lda_topic_proportion_eq_token_weighted_leaf_average :=
  @FormalProofs.OPT.lda_topicProportion_eq_tokenWeightedLeafAverage

/-- Document word proportions are token-weighted averages of leaf word
proportions. -/
abbrev lda_word_proportion_eq_token_weighted_leaf_average :=
  @FormalProofs.OPT.lda_wordProportion_eq_tokenWeightedLeafAverage

/-- Document word-topic joint proportions are token-weighted averages of leaf
word-topic joint proportions. -/
abbrev lda_word_topic_joint_proportion_eq_token_weighted_leaf_average :=
  @FormalProofs.OPT.lda_wordTopicJointProportion_eq_tokenWeightedLeafAverage

/-- Topic-conditional word proportions are topic-mass weighted averages of leaf
topic-conditional word proportions. -/
abbrev lda_word_given_topic_proportion_eq_topic_mass_weighted_leaf_average :=
  @FormalProofs.OPT.lda_wordGivenTopicProportion_eq_topicMassWeightedLeafAverage

/-- LDA likelihood readout from a bag-of-words histogram. -/
abbrev lda_likelihood_readout :=
  @FormalProofs.OPT.ldaLikelihoodReadout

/-- LDA likelihood family on token lists. -/
abbrev lda_likelihood_family :=
  @FormalProofs.OPT.ldaLikelihoodFamily

/-- Bag-of-words histograms realize the LDA likelihood family by readout. -/
abbrev lda_likelihood_readout_realizes_bagOfWords :=
  @FormalProofs.OPT.ldaLikelihoodReadout_realizes_bagOfWords

/-- Bag-of-words histograms are sufficient for the ordinary LDA likelihood
family. -/
abbrev bagOfWords_lda_likelihood_family_sufficient :=
  @FormalProofs.OPT.bagOfWords_ldaLikelihoodFamilySufficient

/-- Any hybrid that includes bag-of-words as its base statistic remains
sufficient for the ordinary LDA likelihood family. -/
abbrev lda_bow_hybrid_likelihood_sufficient :=
  @FormalProofs.OPT.lda_bowHybrid_likelihoodFamilySufficient

/-- For order/contextual probes inside a bag-of-words fiber, hybrid response
sufficiency forces the neural component to separate probe-distinct documents. -/
abbrev lda_bow_hybrid_neural_separates_response_within_bagOfWords :=
  @FormalProofs.OPT.lda_bowHybrid_neural_separates_response_within_bagOfWords

/-- Product summary combining a base/domain statistic with a neural statistic. -/
abbrev hybrid_summary :=
  @FormalProofs.OPT.HybridSummary

/-- Symbolic pointwise argmax predicate for information objectives. -/
abbrev hybrid_information_argmax :=
  @FormalProofs.OPT.IsArgmax

/-- Symbolic pointwise argmin predicate for loss objectives. -/
abbrev hybrid_information_argmin :=
  @FormalProofs.OPT.IsArgmin

/-- Symbolic chain-rule interface for hybrid MI:
`I((t,s);theta) = I(s;theta|t) + I(t;theta)`. -/
abbrev hybrid_mi_chain_rule :=
  @FormalProofs.OPT.HybridMIChainRule

/-- Under the symbolic hybrid MI chain rule, maximizing conditional MI beyond
the base summary is equivalent to maximizing joint hybrid MI. -/
abbrev hybrid_cmi_argmax_iff_joint_mi_argmax :=
  @FormalProofs.OPT.hybridCMI_argmax_iff_jointMI_argmax

/-- EPE/posterior-style negated information losses have minimizers exactly at
information maximizers. -/
abbrev hybrid_epe_loss_min_iff_information_max :=
  @FormalProofs.OPT.hybridEPELoss_argmin_iff_information_argmax

/-- Classifier/JSD-style losses are optimizer-equivalent to their information
proxy whenever the loss reverses the proxy order. -/
abbrev hybrid_classifier_loss_min_iff_information_proxy_max :=
  @FormalProofs.OPT.hybridClassifierLoss_argmin_iff_informationProxy_argmax

/-- Symbolic epsilon-argmax predicate for dependence proxies and information
objectives. -/
abbrev dependence_proxy_epsilon_argmax :=
  @FormalProofs.OPT.IsEpsilonArgmax

/-- Uniform deterministic proxy error transports exact proxy maximizers to
near-maximizers of the target information objective. -/
abbrev uniform_proxy_error_argmax_implies_information_epsilon_argmax :=
  @FormalProofs.OPT.uniformProxyError_argmax_implies_informationEpsilonArgmax

/-- MINE/DV-style loss minimization is equivalent to symbolic proxy
maximization under the supplied order-reversal assumption. -/
abbrev mine_dv_loss_min_iff_proxy_max :=
  @FormalProofs.OPT.mineDV_lossArgmin_iff_proxyArgmax

/-- Deep InfoMax/JSD-style loss minimization is equivalent to symbolic proxy
maximization under the supplied order-reversal assumption. -/
abbrev deep_infomax_jsd_loss_min_iff_proxy_max :=
  @FormalProofs.OPT.deepInfoMaxJSD_lossArgmin_iff_proxyArgmax

/-- InfoNCE/CPC-style loss minimization is equivalent to symbolic proxy
maximization under the supplied order-reversal assumption. -/
abbrev infonce_loss_min_iff_proxy_max :=
  @FormalProofs.OPT.infoNCE_lossArgmin_iff_proxyArgmax

/-- Distance-correlation objectives are exposed as symbolic dependence-proxy
maximization, not an independence-characterization theorem. -/
abbrev distance_correlation_proxy_max :=
  @FormalProofs.OPT.distanceCorrelation_proxyMax

/-- Wasserstein-dependency objectives are exposed as symbolic dependence-proxy
maximization, not an optimal-transport duality theorem. -/
abbrev wasserstein_dependency_proxy_max :=
  @FormalProofs.OPT.wassersteinDependency_proxyMax

/-- EPE losses based on joint hybrid MI also maximize conditional MI beyond the
fixed base summary under the symbolic chain rule. -/
abbrev hybrid_epe_loss_min_iff_conditional_mi_max :=
  @FormalProofs.OPT.hybridEPELoss_argmin_iff_conditionalMI_argmax

/-- Hybrid product summaries refine their base component. -/
abbrev hybrid_summary_sufficient_for_base :=
  @FormalProofs.OPT.hybridSummary_sufficient_for_base

/-- Hybrid product summaries refine their neural component. -/
abbrev hybrid_summary_sufficient_for_neural :=
  @FormalProofs.OPT.hybridSummary_sufficient_for_neural

/-- Within-base target sufficiency: the neural summary resolves target
distinctions inside each base-summary fiber. -/
abbrev within_base_target_sufficient :=
  @FormalProofs.OPT.WithinBaseTargetSufficient

/-- Neural summary separates target distinctions that remain inside a
base-summary fiber. -/
abbrev neural_separates_target_within_base :=
  @FormalProofs.OPT.NeuralSeparatesTargetWithinBase

/-- Hybrid target sufficiency is equivalent to within-base target sufficiency. -/
abbrev hybrid_target_sufficient_iff_within_base_target_sufficient :=
  @FormalProofs.OPT.hybridTargetSufficient_iff_withinBaseTargetSufficient

/-- Within-base target sufficiency is equivalent to neural separation of
within-base target distinctions. -/
abbrev within_base_target_sufficient_iff_neural_separates_target_within_base :=
  @FormalProofs.OPT.withinBaseTargetSufficient_iff_neuralSeparatesTargetWithinBase

/-- Hybrid target sufficiency forces neural separation of target distinctions
left unresolved by the base summary. -/
abbrev hybrid_target_sufficient_neural_separates_target_within_base :=
  @FormalProofs.OPT.hybridTargetSufficient_neuralSeparatesTargetWithinBase

/-- A target-sufficient hybrid cannot collapse a target-distinct pair in both
components. -/
abbrev hybrid_collision_impossible_of_distinguished_target :=
  @FormalProofs.OPT.hybridTargetSufficient_no_base_neural_collision_of_distinguished_target

/-- Target readout from a hybrid product summary. -/
abbrev hybrid_target_readout_realizes :=
  @FormalProofs.OPT.HybridTargetReadoutRealizes

/-- Hybrid target readout implies target sufficiency. -/
abbrev hybrid_target_readout_implies_target_sufficient :=
  @FormalProofs.OPT.hybridTargetReadout_implies_targetSufficient

/-- Target sufficiency of the base statistic lifts to the hybrid product. -/
abbrev hybrid_target_sufficient_of_base_sufficient :=
  @FormalProofs.OPT.hybridTargetSufficient_of_baseSufficient

/-- Target sufficiency of the neural statistic lifts to the hybrid product. -/
abbrev hybrid_target_sufficient_of_neural_sufficient :=
  @FormalProofs.OPT.hybridTargetSufficient_of_neuralSufficient

/-- Within-base likelihood sufficiency: the neural summary resolves likelihood
distinctions inside each base-summary fiber. -/
abbrev within_base_likelihood_sufficient :=
  @FormalProofs.OPT.WithinBaseLikelihoodSufficient

/-- Neural summary separates likelihood distinctions that remain inside a
base-summary fiber. -/
abbrev neural_separates_likelihood_within_base :=
  @FormalProofs.OPT.NeuralSeparatesLikelihoodWithinBase

/-- Hybrid likelihood sufficiency is equivalent to within-base likelihood
sufficiency. -/
abbrev hybrid_likelihood_sufficient_iff_within_base_likelihood_sufficient :=
  @FormalProofs.OPT.hybridLikelihoodSufficient_iff_withinBaseLikelihoodSufficient

/-- Within-base likelihood sufficiency is equivalent to neural separation of
within-base likelihood distinctions. -/
abbrev within_base_likelihood_sufficient_iff_neural_separates_likelihood_within_base :=
  @FormalProofs.OPT.withinBaseLikelihoodSufficient_iff_neuralSeparatesLikelihoodWithinBase

/-- Hybrid likelihood sufficiency forces neural separation of likelihood
distinctions left unresolved by the base summary. -/
abbrev hybrid_likelihood_sufficient_neural_separates_likelihood_within_base :=
  @FormalProofs.OPT.hybridLikelihoodSufficient_neuralSeparatesLikelihoodWithinBase

/-- A likelihood-sufficient hybrid cannot collapse a likelihood-distinct pair in
both components. -/
abbrev hybrid_collision_impossible_of_distinguished_likelihood :=
  @FormalProofs.OPT.hybridLikelihoodSufficient_no_base_neural_collision_of_distinguished_likelihood

/-- Likelihood-family readout from a hybrid product summary. -/
abbrev hybrid_likelihood_readout_realizes :=
  @FormalProofs.OPT.HybridLikelihoodReadoutRealizes

/-- Hybrid likelihood readout implies likelihood-family sufficiency. -/
abbrev hybrid_likelihood_readout_implies_likelihood_sufficient :=
  @FormalProofs.OPT.hybridLikelihoodReadout_implies_likelihoodSufficient

/-- Likelihood sufficiency of the base statistic lifts to the hybrid product. -/
abbrev hybrid_likelihood_sufficient_of_base_sufficient :=
  @FormalProofs.OPT.hybridLikelihoodSufficient_of_baseSufficient

/-- Likelihood sufficiency of the neural statistic lifts to the hybrid product. -/
abbrev hybrid_likelihood_sufficient_of_neural_sufficient :=
  @FormalProofs.OPT.hybridLikelihoodSufficient_of_neuralSufficient

/-- Likelihood-on-hybrid-state is a direct likelihood-on-state instance. -/
abbrev hybrid_likelihood_on_state_family_sufficient :=
  @FormalProofs.OPT.hybridLikelihoodOnState_family_sufficient

/-- Within-base likelihood-free response sufficiency. -/
abbrev within_base_response_sufficient :=
  @FormalProofs.OPT.WithinBaseResponseSufficient

/-- Neural summary separates probe-response distinctions that remain inside a
base-summary fiber. -/
abbrev neural_separates_response_within_base :=
  @FormalProofs.OPT.NeuralSeparatesResponseWithinBase

/-- Hybrid response sufficiency is equivalent to within-base response
sufficiency. -/
abbrev hybrid_response_sufficient_iff_within_base_response_sufficient :=
  @FormalProofs.OPT.hybridResponseSufficient_iff_withinBaseResponseSufficient

/-- Within-base response sufficiency is equivalent to neural separation of
within-base probe-response distinctions. -/
abbrev within_base_response_sufficient_iff_neural_separates_response_within_base :=
  @FormalProofs.OPT.withinBaseResponseSufficient_iff_neuralSeparatesResponseWithinBase

/-- Hybrid response sufficiency forces neural separation of probe distinctions
left unresolved by the base summary. -/
abbrev hybrid_response_sufficient_neural_separates_response_within_base :=
  @FormalProofs.OPT.hybridResponseSufficient_neuralSeparatesResponseWithinBase

/-- A response-sufficient hybrid cannot collapse a probe-distinct pair in both
components. -/
abbrev hybrid_collision_impossible_of_distinguished_response :=
  @FormalProofs.OPT.hybridResponseSufficient_no_base_neural_collision_of_distinguished_response

/-- Likelihood-free response readout from a hybrid product summary. -/
abbrev hybrid_response_readout_realizes :=
  @FormalProofs.OPT.HybridResponseReadoutRealizes

/-- Hybrid response readout implies likelihood-free response sufficiency. -/
abbrev hybrid_response_readout_implies_likelihood_free_sufficient :=
  @FormalProofs.OPT.hybridResponseReadout_implies_likelihoodFreeSufficient

/-- Approximate within-base target sufficiency. -/
abbrev within_base_target_sufficient_within :=
  @FormalProofs.OPT.WithinBaseTargetSufficientWithin

/-- Approximate hybrid target readout. -/
abbrev hybrid_target_readout_realizes_within :=
  @FormalProofs.OPT.HybridTargetReadoutRealizesWithin

/-- Approximate hybrid target readout implies approximate within-base target
sufficiency. -/
abbrev hybrid_target_readout_within_implies_within_base_target_sufficient_within :=
  @FormalProofs.OPT.hybridTargetReadoutWithin_implies_withinBaseTargetSufficientWithin

/-- Approximate within-base likelihood sufficiency. -/
abbrev within_base_likelihood_sufficient_within :=
  @FormalProofs.OPT.WithinBaseLikelihoodSufficientWithin

/-- Approximate hybrid likelihood sufficiency is equivalent to approximate
within-base likelihood sufficiency. -/
abbrev hybrid_likelihood_sufficient_within_iff_within_base_likelihood_sufficient_within :=
  @FormalProofs.OPT.hybridLikelihoodSufficientWithin_iff_withinBaseLikelihoodSufficientWithin

/-- Approximate hybrid likelihood readout. -/
abbrev hybrid_likelihood_readout_realizes_within :=
  @FormalProofs.OPT.HybridLikelihoodReadoutRealizesWithin

/-- Approximate hybrid likelihood readout implies approximate within-base
likelihood sufficiency. -/
abbrev hybrid_likelihood_readout_within_implies_within_base_likelihood_sufficient_within :=
  @FormalProofs.OPT.hybridLikelihoodReadoutWithin_implies_withinBaseLikelihoodSufficientWithin

/-- Approximate within-base likelihood-free response sufficiency. -/
abbrev within_base_response_sufficient_within :=
  @FormalProofs.OPT.WithinBaseResponseSufficientWithin

/-- Approximate hybrid response sufficiency is equivalent to approximate
within-base response sufficiency. -/
abbrev hybrid_response_sufficient_within_iff_within_base_response_sufficient_within :=
  @FormalProofs.OPT.hybridResponseSufficientWithin_iff_withinBaseResponseSufficientWithin

/-- Approximate hybrid likelihood-free response readout. -/
abbrev hybrid_response_readout_realizes_within :=
  @FormalProofs.OPT.HybridResponseReadoutRealizesWithin

/-- Approximate hybrid response readout implies approximate within-base
likelihood-free response sufficiency. -/
abbrev hybrid_response_readout_within_implies_within_base_response_sufficient_within :=
  @FormalProofs.OPT.hybridResponseReadoutWithin_implies_withinBaseResponseSufficientWithin

/-- Approximate Markov-count query sufficiency over real-valued counts. -/
abbrev markov_count_query_sufficient_within :=
  @FormalProofs.OPT.MarkovCountQuerySufficientWithin

/-- Approximate Markov-count query sufficiency is generic two-sided approximate
contextual sufficiency for real-valued counts. -/
abbrev markov_count_query_sufficient_within_iff_twoSided_context_sufficient_within :=
  @FormalProofs.OPT.markovCountQuerySufficientWithin_iff_twoSidedContextSufficientWithin

/-- Exact Markov sketches are zero-slack sufficient for real-valued count
queries. -/
abbrev exact_markov_sketch_twoSided_context_sufficient_within_zero_real :=
  @FormalProofs.OPT.exact_markov_sketch_twoSidedContextSufficientWithin_zero_real

/-- Composed approximate readout bridge specialized to real-valued Markov count
queries. -/
abbrev markov_composed_readout_within_implies_twoSided_context_sufficient_within_real :=
  @FormalProofs.OPT.markov_composedReadoutWithin_implies_twoSidedContextSufficientWithin_real

/-- Finite sliced approximate bridge specialized to real-valued Markov count
queries. -/
abbrev markov_finite_sliced_within_implies_count_query_sufficient_within :=
  @FormalProofs.OPT.markov_finiteSlicedWithin_implies_countQuerySufficientWithin

#check FormalProofs.OPT.QuerySufficient
#check FormalProofs.OPT.ResponseSignature
#check FormalProofs.OPT.TwoSidedContextQuery
#check FormalProofs.OPT.TwoSidedContextSufficient
#check FormalProofs.OPT.UniformG
#check FormalProofs.OPT.UniformG.leaf
#check FormalProofs.OPT.UniformG.merge
#check FormalProofs.OPT.FiniteContextCovers
#check FormalProofs.OPT.querySufficient_iff_exists_contextReadout
#check FormalProofs.OPT.querySufficient_no_collision_of_distinguished_context
#check FormalProofs.OPT.finiteContext_zeroLoss_implies_querySufficient
#check FormalProofs.OPT.QuerySufficientWithin
#check FormalProofs.OPT.TwoSidedContextSufficientWithin
#check @FormalProofs.OPT.composedTwoSidedReadoutWithinEps_implies_twoSidedContextSufficientWithin
#check @FormalProofs.OPT.uniformComposedTwoSidedReadoutWithinEps_implies_twoSidedContextSufficientWithin
#check @FormalProofs.OPT.uniformComposedTwoSidedReadoutExact_implies_twoSidedContextSufficient
#check @FormalProofs.OPT.composedTwoSidedReadoutExact_implies_twoSidedContextSufficientWithin_zero
#check FormalProofs.OPT.SlicedResponseSignature
#check FormalProofs.OPT.SlicedQuerySufficient
#check FormalProofs.OPT.FiniteSlicesCoverResponseFibers
#check @FormalProofs.OPT.finiteSliced_zeroLoss_implies_querySufficient
#check FormalProofs.OPT.SlicedQuerySufficientWithin
#check FormalProofs.OPT.FiniteSlicesCoverResponseFibersWithin
#check @FormalProofs.OPT.finiteSlicedWithin_implies_querySufficientWithin
#check FormalProofs.OPT.RandomFiniteSlicedGoodEvent
#check @FormalProofs.OPT.randomFiniteSliced_contextualSufficiency_failure_prob_le
#check @FormalProofs.OPT.randomFiniteSlicedWithin_contextualSufficiency_failure_prob_le
#check FormalProofs.OPT.QuerySufficientWithinOn
#check FormalProofs.OPT.FiniteContextCoversWithin
#check @FormalProofs.OPT.finiteContext_within_implies_querySufficientWithin
#check FormalProofs.OPT.QuerySufficientNearWithin
#check FormalProofs.OPT.ContextReadoutNearPreserving
#check @FormalProofs.OPT.contextReadoutApproxNearPreserving_implies_querySufficientNearWithin
#check FormalProofs.OPT.CoordinateSlice
#check @FormalProofs.OPT.finiteCoordinateSlices_univ_cover_responseFibers
#check @FormalProofs.OPT.leftInvertibleSlices_cover_responseFibers
#check FormalProofs.OPT.TargetSufficientRepresentation
#check FormalProofs.OPT.TargetMeasurable
#check @FormalProofs.OPT.targetSufficient_iff_exists_readout
#check FormalProofs.OPT.LikelihoodFamilySufficient
#check @FormalProofs.OPT.likelihoodFamilySufficient_iff_exists_readout
#check @FormalProofs.OPT.likelihoodFamilySufficient_no_collision_of_distinguished_likelihood
#check FormalProofs.OPT.LikelihoodOnStateFamily
#check @FormalProofs.OPT.likelihoodOnState_family_sufficient
#check @FormalProofs.OPT.repWithStateReadout_likelihoodOnState_family_sufficient
#check @FormalProofs.OPT.likelihoodReadoutWithin_implies_likelihoodFamilySufficientWithin
#check FormalProofs.OPT.SurjectiveStateMap
#check @FormalProofs.OPT.surjectiveState_likelihood_factorization
#check @FormalProofs.OPT.surjectiveState_likelihoodSufficient_iff_factors
#check @FormalProofs.OPT.surjectiveState_likelihoodReadoutWithin
#check FormalProofs.OPT.LikelihoodFreeResponseSufficient
#check @FormalProofs.OPT.likelihoodFreeResponseSufficient_iff_exists_readout
#check @FormalProofs.OPT.twoSidedContextSufficient_iff_likelihoodFreeResponseSufficient
#check @FormalProofs.OPT.bagOfWords_ldaLikelihoodFamilySufficient
#check @FormalProofs.OPT.lda_bowHybrid_likelihoodFamilySufficient
#check FormalProofs.OPT.HybridSummary
#check FormalProofs.OPT.HybridMIChainRule
#check @FormalProofs.OPT.hybridCMI_argmax_iff_jointMI_argmax
#check @FormalProofs.OPT.hybridEPELoss_argmin_iff_information_argmax
#check @FormalProofs.OPT.hybridClassifierLoss_argmin_iff_informationProxy_argmax
#check FormalProofs.OPT.WithinBaseTargetSufficient
#check @FormalProofs.OPT.hybridTargetSufficient_iff_withinBaseTargetSufficient
#check @FormalProofs.OPT.withinBaseTargetSufficient_iff_neuralSeparatesTargetWithinBase
#check @FormalProofs.OPT.hybridLikelihoodSufficient_iff_withinBaseLikelihoodSufficient
#check @FormalProofs.OPT.hybridLikelihoodSufficient_neuralSeparatesLikelihoodWithinBase
#check @FormalProofs.OPT.hybridLikelihoodSufficient_no_base_neural_collision_of_distinguished_likelihood
#check @FormalProofs.OPT.hybridLikelihoodReadout_implies_likelihoodSufficient
#check @FormalProofs.OPT.hybridLikelihoodOnState_family_sufficient
#check @FormalProofs.OPT.hybridResponseReadout_implies_likelihoodFreeSufficient
#check @FormalProofs.OPT.hybridLikelihoodReadoutWithin_implies_withinBaseLikelihoodSufficientWithin
#check @FormalProofs.OPT.hybridResponseReadoutWithin_implies_withinBaseResponseSufficientWithin
#check FormalProofs.OPT.MarkovCountQuerySufficientWithin
#check @FormalProofs.OPT.markov_finiteSlicedWithin_implies_countQuerySufficientWithin
#check @FormalProofs.OPT.markov_countEndpointHybrid_twoSidedContextSufficient

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

/-- Markov task-facing sufficiency is the generic two-sided contextual
sufficiency condition specialized to changepoint-count contexts. -/
abbrev markov_count_query_sufficient_iff_twoSided_context_sufficient :=
  @FormalProofs.OPT.markovCountQuerySufficient_iff_twoSidedContextSufficient

/-- Exact Markov `(count, first, last)` sketches satisfy the generic two-sided
contextual sufficiency condition. -/
abbrev exact_markov_sketch_twoSided_context_sufficient :=
  @FormalProofs.OPT.exact_markov_sketch_twoSidedContextSufficient

/-- Markov endpoint residual paired with count-only state. -/
abbrev markov_endpoint_residual :=
  @FormalProofs.OPT.markovEndpointResidual

/-- Makinen-style Markov hybrid summary: `(count-only, endpoint residual)`. -/
abbrev markov_count_endpoint_hybrid :=
  @FormalProofs.OPT.markovCountEndpointHybrid

/-- The Markov count-plus-endpoint hybrid is sufficient for all two-sided
changepoint-count queries. -/
abbrev markov_count_endpoint_hybrid_query_sufficient :=
  @FormalProofs.OPT.markov_countEndpointHybrid_query_sufficient

/-- The Markov count-plus-endpoint hybrid satisfies generic two-sided
contextual sufficiency. -/
abbrev markov_count_endpoint_hybrid_two_sided_sufficient :=
  @FormalProofs.OPT.markov_countEndpointHybrid_twoSidedContextSufficient

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

/-- Markov count-only summaries also fail the generic two-sided contextual
sufficiency condition. -/
abbrev markov_countOnly_not_twoSided_context_sufficient :=
  @FormalProofs.OPT.markov_countOnly_not_twoSidedContextSufficient

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
