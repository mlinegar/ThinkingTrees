# TreePO: Tree-Based Preference Optimization (Formalization Map)

This document records the formal definitions, invariance results, and sampling theory behind TreePO. It ties together the Lean proofs in this repository and the design-based sampling framework in `../FormalProbability/`.

## Scope

TreePO is not a new optimization algorithm. It is a data-collection and objective-estimation strategy that lets us train DPO, GRPO, or PPO-style objectives using tree summaries and a random oracle-labeled subset, while retaining formal guarantees about loss equivalence or bounded gaps.

## Core Objects and Notation

- `Strings` is the document space with monoid structure.
- `Y` is the oracle space with a pseudo-metric.
- `f* : Strings → Y` is the oracle function.
- `A` is the action space (responses, summaries, or candidate outputs).
- `μ : PMF Strings` is the document distribution.
- `gen : Strings → PMF (A × A)` is the pair generator for pairwise preferences.
- `gen_k : Strings → PMF (Fin k → A)` is the group generator for listwise preferences.
- `πθ : Strings → A → ℝ` is a policy, with `π_ref` and `π_old` when needed.

Lean references:
- `lean3/FormalProofs/OPT/PreferenceLearning.lean`
- `lean3/FormalProofs/OPT/PreferenceBounds.lean`

## Preference Learning Methods (Formal Definitions)

DPO, GRPO, and GRPO-RL are already formalized in Lean as pointwise losses and expected objectives.

- DPO logit and loss: `DPOLogit`, `DPOLossPointwise`, `ExpectedDPOLoss` in `lean3/FormalProofs/OPT/PreferenceLearning.lean`.
- GRPO-PL (Plackett-Luce): `PlackettLuceLogProb`, `GRPOLossPointwise`, `ExpectedGRPOLoss` in `lean3/FormalProofs/OPT/PreferenceLearning.lean`.
- GRPO-RL (clipped surrogate + KL): `GRPOClip`, `GRPOAdvantage`, `GRPOGroupKL`, `GRPORLLossPointwise`, `ExpectedGRPORLLoss` in `lean3/FormalProofs/OPT/PreferenceLearning.lean`.
- PPO is treated as a specialization of the GRPO-RL objective with a fixed group structure and externally provided advantage. The clipping and KL machinery is already present in the GRPO-RL formalization.

## TRL Runtime Realization (Python)

The Lean objectives above are now wired into the Python TRL stack in
`src/training/trl_training.py` with explicit propensity/IPW semantics:

- `train_dpo` maps pairwise preferences to TRL `DPOTrainer`.
- `train_reward_model` maps pairwise preferences to TRL `RewardTrainer`.
- `train_grpo` maps prompt distributions to TRL `GRPOTrainer` with online rewards.

Design-based weighting implementation:

- Every exported training example carries `sample_weight = 1 / joint_propensity`
  (with global-uniform fallback, so default weight is `1`).
- DPO and reward training support native weighted losses:
  - DPO loss is reduced as weighted mean over per-example DPO losses.
  - Reward-model pairwise logistic loss is reduced as weighted mean.
- GRPO uses weighted advantages in the trainer path, i.e. per-example
  advantages are scaled by normalized sample weights before surrogate-loss
  evaluation.
- Optional weighted resampling remains available as a fallback
  (`multinomial`, `pps_systematic`, `stratified_multinomial`), but is no
  longer required when native weighting is enabled.

This is the direct computational analogue of the design-based estimators in
`lean3/FormalProofs/DSL/IPWTheory.lean` and `lean3/FormalProofs/DSL/TreeIPW.lean`:
the runtime objective minimizes a weighted empirical approximation to the same
population risk targeted by HT/Hajek-style estimators.

## Oracle-Measurability and Loss Equivalence

The fundamental invariance is already formalized: if the loss and generator are oracle-indexed, then the expected loss is unchanged when oracle values are preserved.

- Generic invariance: `expected_loss_eq_of_zero_dist_generic` in `lean3/FormalProofs/OPT/PreferenceLearning.lean`.
- Pairwise specialization: `expected_loss_eq_of_zero_dist` in `lean3/FormalProofs/OPT/PreferenceLearning.lean`.
- Group specialization: `expected_group_loss_eq_of_zero_dist` in `lean3/FormalProofs/OPT/PreferenceLearning.lean`.
- DPO, GRPO-PL, GRPO-RL equivalence: `dpo_equivalence`, `grpo_equivalence`, `grpo_rl_equivalence_via_pref` in `lean3/FormalProofs/OPT/PreferenceLearning.lean`.

## Exact Utility-Transport Surface

The exact-control simulation suite now has a Lean-facing theorem surface that is
broader than pairwise preference learning alone.

Generic exact-utility wrappers:

- `featureIndexedObjective_eq_of_zero_dist` in `lean3/FormalProofs/OPT/ExactUtilityTransport.lean`
  gives zero-distortion transport for any objective that factors through an
  exact feature/state oracle.
- `supervisedStateExpectedLoss_eq_of_zero_dist` in
  `lean3/FormalProofs/OPT/ExactUtilityTransport.lean`
  packages direct supervised-state learning as a special case.
- `normalizedErrorUtility_zero_regret_iff_zero_error` in
  `lean3/FormalProofs/OPT/ExactUtilityTransport.lean`
  records the exact-control condition used in the new simulations:
  for normalized exact-state utilities, zero utility regret coincides with zero
  state error.
- `mergeableStateUtility_exact_on_tree` in
  `lean3/FormalProofs/OPT/ExactUtilityTransport.lean`
  states the strongest exact-control result: if a latent state is represented by
  an exact mergeable fold, then any downstream utility on that state is
  preserved exactly by the tree.

Concrete exact lanes:

- `markovStateUtility_exact_on_tree`,
  `markovCountOnlyUtility_exact_on_tree`,
  `markovCountEndpointsUtility_exact_on_tree` in
  `lean3/FormalProofs/OPT/ExactUtilityTransportInstances.lean`
- `complementarityStateUtility_exact_on_tree`,
  `complementarityThresholdUtility_exact_on_tree` in
  `lean3/FormalProofs/OPT/ExactUtilityTransportInstances.lean`
- `topicSketchUtility_exact_on_tree`,
  `topicMassUtility_exact_on_tree`,
  `topicOracleFromSketch_exact_on_tree` in
  `lean3/FormalProofs/OPT/ExactUtilityTransportInstances.lean`

This is the exact theorem surface that the new Markov / nonseparable /
boundary-topic utility simulations should cite.

## Tree Summarization and Zero Distortion

Tree summarization is encoded via a summarizer `g`, binary trees `T`, and multi-round reduction `ZR g x R T`. The local laws L1, L2, L3 imply zero expected distortion, which triggers the preference-loss invariance above.

- Local laws and the link to zero distortion: `multi_round_proper` and the summaries in `lean3/FormalProofs/Assumptions.lean`.
- ZR connection for preference learning: `preference_learning_equivalence_via_ZR` and `grpo_equivalence_via_ZR` in `lean3/FormalProofs/OPT/PreferenceLearning.lean`.
- Quantitative gaps when distortion is nonzero: `dpo_gap_bounded` and the GRPO Lipschitz axioms in `lean3/FormalProofs/OPT/PreferenceBounds.lean` and `lean3/FormalProofs/Axioms.lean`.

## Approximate + Audited + Stochastic Theorem Chain

The non-ideal (practical) proof chain is now explicit and connected end-to-end:

1. Approximate local-law layer:
   - `L1εNode`, `L2εNode`, `ApproxLocalLawsBundle`,
     `approx_bundle_of_nodewise`
   - file: `lean3/FormalProofs/OPT/ApproximateLocalLaws.lean`
2. Audited transfer layer:
   - `AuditedApproxUpperBounds`,
     `approx_bundle_of_audited_upper_bounds`,
     `AuditedBoundsWithConfidence`,
     `approx_bundle_of_audited_confidence_event`
   - empirical-margin wrappers:
     `NodewiseEmpiricalAuditCertificate`,
     `audited_upper_bounds_of_nodewise_empirical_certificate`,
     `NodewiseEmpiricalAuditWithConfidence`
   - file: `lean3/FormalProofs/OPT/ApproximateLocalLaws.lean`
3. Objective-gap lifts from audited confidence events:
   - `dpo_gap_via_audited_confidence_event`
   - `grpo_pl_gap_via_audited_confidence_event`
   - `grpo_rl_gap_via_audited_confidence_event`
   - file: `lean3/FormalProofs/OPT/ApproximateLocalLaws.lean`
4. Stochastic adaptive expected bounds:
   - `Exp_Δ_R_ZR_eq_zero_of_stochastic_adaptive_local_laws`
   - `Exp_Δ_R_ZR_le_of_stochastic_adaptive_approx_local_laws`
   - `Exp_dpo_gap_le_of_stochastic_adaptive_approx_local_laws`
   - `Exp_grpo_pl_gap_le_of_stochastic_adaptive_approx_local_laws`
   - `Exp_grpo_rl_gap_le_of_stochastic_adaptive_approx_local_laws`
   - bounded wrappers:
     `Exp_Δ_R_ZR_le_of_stochastic_adaptive_approx_local_laws_bounded`,
     `Exp_dpo_gap_le_of_stochastic_adaptive_approx_local_laws_bounded`
   - file: `lean3/FormalProofs/OPT/AdaptiveChunkingBridge.lean`
5. Export surface:
   - consolidated aliases in `lean3/FormalProofs/OPT/MainTheorems.lean`.

## Design-Based Sampling and IPW

Random oracle labeling is formalized via design-based sampling and inverse probability weighting. This supports unbiased estimation of violation rates and loss gaps from a labeled subset.

- Sampling probability and positivity: `DesignBasedSampling` in `../FormalProbability/FormalProbability/DSL/SamplingTheory.lean`.
- Oracle access on sampled points: `OracleAccess` in `lean3/FormalProofs/DSL/CoreDefinitions.lean` and `../FormalProbability/FormalProbability/DSL/CoreDefinitions.lean`.
- IPW Bernoulli HT unbiasedness: `ht_expectation` in `FormalProbability/DSL/IPWTheory.lean`
  (repackaged as `htExp_unbiased` in `lean3/FormalProofs/DSL/IPWTheory.lean`).
- HT estimator boundedness: `htExpEstimator_abs_le` / `htExpEstimator_abs_sq_le` and TreePO specializations `ipw_tree_distortion_abs_le` / `ipw_tree_distortion_abs_sq_le` in `lean3/FormalProofs/DSL/IPWTheory.lean` and `lean3/FormalProofs/DSL/TreeIPW.lean`.
- Tree-level propensities and samples: `TreePropensity` in `lean3/FormalProofs/DSL/IPWTheory.lean` and `TreeSample` in `lean3/FormalProofs/DSL/TreeIPW.lean`.
- Tree DSL bound: `DSLBound` and `computeDSLBound` in `lean3/FormalProofs/DSL/TreeIPW.lean`.
- Generator stability (doc-dependent gen): `GroupGeneratorLipschitzL1` in `lean3/FormalProofs/OPT/PreferenceLearning.lean`.
- Self-normalized (empirical Bernstein) primitives + event interfaces:
  `weightedVariance`, `empiricalBernsteinRadius`,
  `empiricalBernstein_bound_ennreal_of_event` in `lean3/FormalProofs/DSL/IPWTheory.lean`.
- Tree-level empirical Bernstein wrapper for violation rates (non-Bernoulli sampling): 
  `ipw_violation_rate_empirical_bernstein` and `ipwViolationEmpiricalBernsteinCI`
  in `lean3/FormalProofs/DSL/TreeIPW.lean`.
- Tree-level empirical Bernstein wrapper for preference losses (non-Bernoulli sampling):
  `ipw_preference_loss_empirical_bernstein` and `ipwPreferenceEmpiricalBernsteinCI`
  in `lean3/FormalProofs/DSL/TreeIPW.lean`.
- Honest sample splitting helpers (train vs eval by document):
  `SampleSplit` in `lean3/FormalProofs/DSL/Honesty.lean` and
  `honestIPWViolationRate` / `honestIPWPreferenceLoss` in `lean3/FormalProofs/DSL/TreeIPW.lean`.
- Three-layer honesty helpers for chunker/summarizer/oracle:
  `ThreeLayerSplit`, `filterThreeEval`, `ThreeLayerHonestTraining`,
  `ThreeLayerHonestEvaluation`, `ParallelSafeTraining`,
  `threeLayer_eval_bound` in `lean3/FormalProofs/DSL/Honesty.lean`.
- Single-oracle two-view honesty helper:
  `SingleOracleTwoViewHonesty` in `lean3/FormalProofs/DSL/Honesty.lean`.
- Dual-oracle honesty helpers (teacher + proxy tied to oracle split):
  `filterOracleEval`, `DualOracleHonestTraining`,
  `DualOracleHonestEvaluation` in `lean3/FormalProofs/DSL/Honesty.lean`.
- K-fold honest evaluation helpers:
  `KFoldSplit` in `lean3/FormalProofs/DSL/Honesty.lean` and
  `kFoldIPWViolationRate` / `kFoldIPWPreferenceLoss` in `lean3/FormalProofs/DSL/TreeIPW.lean`.
- Honest bound lifting + K-fold aggregation:
  `honest_eval_bound` and `kfold_avg_bound` in `lean3/FormalProofs/DSL/Honesty.lean`.
- K-fold TreeIPW concentration wrappers:
  `kFoldIPWViolationRate_bound` and `kFoldIPWPreferenceLoss_bound` in
  `lean3/FormalProofs/DSL/TreeIPW.lean`.
- Honest empirical Bernstein corollaries:
  `honest_ipw_violation_empirical_bernstein` and
  `honest_ipw_preference_empirical_bernstein` in
  `lean3/FormalProofs/DSL/TreeIPW.lean`.
- K-fold empirical Bernstein corollaries + CI helpers:
  `kFoldIPWViolationRate_empirical_bernstein`,
  `kFoldIPWPreferenceLoss_empirical_bernstein`,
  `kFoldViolationEmpiricalBernsteinCI`,
  `kFoldPreferenceEmpiricalBernsteinCI`
  in `lean3/FormalProofs/DSL/TreeIPW.lean`.
- K-fold union-bound EB + CI bundle:
  `kFoldIPWUnionBound_empirical_bernstein`,
  `kFoldIPWUnionBound_empirical_bernstein_from_components`,
  `kFoldUnionBoundEmpiricalBernsteinCI`,
  `kFoldUnionBoundEmpiricalBernsteinBound`
  in `lean3/FormalProofs/DSL/TreeIPW.lean`.
- Worst-case calibration/estimation/clipping envelopes:
  `treepo_gap_with_calibration_and_estimation`,
  `treepo_gap_with_calibration_estimation_clipping`,
  `dsl_abs_gap_bound_from_estimate`,
  `dsl_abs_gap_bound_from_clipped_estimate`
  in `lean3/FormalProofs/DSL/TreeIPW.lean`.
- Three-layer deterministic/probabilistic worst-case envelopes:
  `threeLayer_abs_envelope`, `threeLayer_error_union_bound`
  in `lean3/FormalProofs/DSL/Honesty.lean`.

## Honest Adaptive Chunking: Current Status

This repository now has both formal hooks and Python wiring for honest adaptive chunking.

Lean status:

- Split primitives: `SampleSplit`, `KFoldSplit` (`lean3/FormalProofs/DSL/Honesty.lean`).
- Honest bound lifting: `honest_eval_bound`, `kfold_avg_bound` (`lean3/FormalProofs/DSL/Honesty.lean`).
- Three-layer split primitives and bound lifting:
  `ThreeLayerSplit`, `filterThreeEval`, `ParallelSafeTraining`,
  `threeLayer_eval_bound` (`lean3/FormalProofs/DSL/Honesty.lean`).
- Adaptive exploration-floor interface:
  `adaptiveMixtureProb`, `AdaptiveSamplingAxioms`,
  `floor_lower_bound`, `mixedProb_pos`, `mixedWeight` (`lean3/FormalProofs/DSL/Honesty.lean`).

Python status:

- Chunk-policy objects and split-aware memory:
  `AdaptiveChunkingConfig`, `HonestChunkingPolicy`, `AdaptiveChunkMemory`,
  `assign_honest_split` (`src/preprocessing/chunker.py`).
- Tree build path applies boundary-only feedback under honest policy
  (`src/tree/builder.py`, `src/training/run_pipeline.py`).
- Evaluation now reports held-out split metrics in `final_stats.json` via
  `train.honest_split_metrics` / `test.honest_split_metrics`
  (`src/training/run_pipeline.py`).
- Three-layer runtime integration now affects training objectives:
  scorer optimization train/val splits are filtered by oracle roles, and
  preference-based training subsets can be filtered to train-role docs across
  chunk/summarizer/oracle layers (`src/training/run_pipeline.py`).

Lean label-source + robust propensity status:

- Mixed truth/prediction label primitives:
  `TruthLabelSource`, `ApproxLabelSource`, `LabelObservation`
  (`lean3/FormalProofs/DSL/CoreDefinitions.lean`).
- Unknown/heterogeneous propensity workaround layer:
  `LoggedJointPropensity`, `TreeSampleWithProvenance`,
  `TreePreferenceSampleWithProvenance`,
  `ipwViolationRateRobust`, `ipwPreferenceLossRobust`
  (`lean3/FormalProofs/DSL/TreeIPW.lean`).

## Worst-Case Certificate Stack (Now Explicit)

For applied reporting, the tightest generic stack is now formalized as:

1. Structural TreePO bound (`TreeIPW` / `PreferenceBounds`):
   `|G^J| ≤ B_tree` from IPW/tree distortion transport.
2. Calibration envelope:
   `|G* - G^J| ≤ B_cal`.
3. Estimation envelope:
   `|G^J - G^E| ≤ B_est`.
4. Optional clipping envelope:
   `|G^E - G^C| ≤ B_clip`.

This yields:
\[
|G^*| \le |G^E| + B_{\mathrm{cal}} + B_{\mathrm{est}},
\quad
|G^*| \le |G^C| + B_{\mathrm{cal}} + B_{\mathrm{est}} + B_{\mathrm{clip}}.
\]

Lean anchors:
- `treepo_gap_with_calibration_and_estimation`
- `treepo_gap_with_calibration_estimation_clipping`
- `dsl_abs_gap_bound_from_estimate`
- `dsl_abs_gap_bound_from_clipped_estimate`
- `dsl_abs_gap_bound_from_estimate_high_prob`
- `dsl_abs_gap_bound_from_clipped_estimate_high_prob`
- `dsl_abs_gap_bound_from_estimate_high_prob_total`
- `dsl_abs_gap_bound_from_clipped_estimate_high_prob_total`
- `dsl_bound_valid_from_events`
- `dsl_bound_valid_from_events_total`
- `computeDSLBound_valid_from_events`
in `lean3/FormalProofs/DSL/TreeIPW.lean`.

Equivalent one-shot tail form:
\[
\mathbb{P}\!\left(|G^*| \ge |G^E| + B_{\mathrm{cal}} + B_{\mathrm{est}}\right)
\le \delta_{\mathrm{cal}} + \delta_{\mathrm{est}},
\]
\[
\mathbb{P}\!\left(|G^*| \ge |G^C| + B_{\mathrm{cal}} + B_{\mathrm{est}} + B_{\mathrm{clip}}\right)
\le \delta_{\mathrm{cal}} + \delta_{\mathrm{est}} + \delta_{\mathrm{clip}}.
\]

For triple-honesty component errors `(chunk, summarizer, oracle)`:
- deterministic envelope: `threeLayer_abs_envelope`
- failure-event union bound: `threeLayer_error_union_bound`
in `lean3/FormalProofs/DSL/Honesty.lean`.

### Which Concentration Bound Where?

Short answer: not “instead of”, but “by layer”.

- Use empirical Bernstein at the document/IPW layer (non-uniform weights, heteroskedastic outcomes).
- Use Serfling for within-document chunk auditing under uniform without-replacement sampling.
- Use BM/Azuma/Freedman-style martingale bounds when chunk selection is adaptive/sequential and not simple WOR.

Recommended applied certificate:

- Radius 1 (within-doc): Serfling or BM, depending on query design.
- Radius 2 (across-doc/IPW): empirical Bernstein.
- Radius 3 (model mismatch): calibration/clipping envelopes.
- Final one-shot bound: union composition via the DSL high-probability envelopes above.

Assumption status in this stack:

- TreePO DSL validity theorem is now non-tautological and event-based.
- Empirical-Bernstein concentration in TreePO wrappers is now event-based.
- Calibration transfer in TreePO wrappers is now event-based (`h_rmse_upper`), with
  `*_from_axioms` compatibility wrappers retained.

Open formal gaps:

- We have honesty interfaces and concentration wrappers, but not yet a full
  consistency/asymptotic theorem for the adaptive chunk-boundary learner itself.
- Finite-population without-replacement instantiation for chunk audits is still
  partially pending (see Serfling/Azuma section below).

## Chunk Sampling: IID vs Without Replacement (Serfling/Azuma)

Within a single document, “chunks/leaves” form a **finite population**. Treating chunk labels as IID draws is usually the wrong mental model:

- The chunk values are fixed (after partitioning); the randomness is in *which chunks we query*.
- Sampling without replacement induces **dependence** across queried chunks (negative dependence).
- If informative content is “front-loaded/back-loaded”, IID assumptions can understate worst cases; finite-population concentration avoids this.

Importantly, this is **not** assuming a separable/global-additive utility over chunks. The typical use is:

- define a *bounded* leaf-level diagnostic or proxy quantity (e.g. “leaf law violation”, “local distortion indicator”, or a bounded score against the oracle),
- estimate its finite-population mean from sampled leaves,
- then transport that estimate to a **document-level** or **preference-objective** claim using the already-formal Lipschitz/transport theorems.

**Core setup (one doc).**

- Let a document induce `N` leaves with bounded per-leaf quantity `u₁,…,u_N ∈ [a,b]` (e.g. a leaf-level distortion indicator, or a bounded utility gap proxy).
- Define the doc-level target as the finite-population mean `μ_doc := (1/N) * ∑ u_i`.
- Query `m ≤ N` leaves **uniformly without replacement** and compute the sample mean `μ̂_doc`.

Then (classically) `μ̂_doc` is unbiased for `μ_doc` and obeys Serfling/Hoeffding-style tail bounds for finite populations. The key point is: *no “IID chunks” assumption is needed*; the population is fixed.

One standard form (Serfling-style finite population correction) is:

\[
\mathbb{P}\big(|\hat{\mu}_\text{doc} - \mu_\text{doc}| \ge \varepsilon \big)
\;\le\;
2 \exp\!\left(
\frac{-2 m \varepsilon^2}{(1 - (m-1)/N)\,(b-a)^2}
\right).
\]

Ignoring the finite-population correction term gives the familiar Hoeffding scaling `m ≳ (b-a)^2 * log(2/δ) / (2 ε^2)`; the correction improves as `m/N` grows.

**Lean status.**

We have the main Azuma-ready glue:

- `lean3/FormalProofs/OPT/SerflingAudit.lean` proves a conditional Hoeffding lemma
  (`hasCondSubgaussianMGF_of_mem_Icc_of_condExp_eq_zero`) and a convenience Azuma wrapper
  (`azuma_hoeffding_of_mem_Icc_of_condExp_eq_zero`).
- Interface cleanup: canonical theorem names remain under legacy `OPT.*`, and
  the consolidated export surface is now provided in
  `lean3/FormalProofs/OPT/MainTheorems.lean` via stable aliases
  (`conditional_hoeffding_bridge`, `azuma_from_conditional_hoeffding`,
  `azuma_abs_from_conditional_hoeffding`, `azuma_abs_random_permutation_wor`,
  `adversarial_chunking_failure_bound`).
- The same concentration bridge is now mirrored in
  `../FormalProbability/FormalProbability/DSL/SamplingConcentration.lean` so
  TreePO proofs and the shared DSL stack can import a common theorem layer.
  New concrete wrappers there now include:
  - `UniformWithoutReplacementLeafAudit.abs_tail_bound`
  - `UniformWithoutReplacementLeafAudit.serfling_style_abs_tail_bound`
  - `UniformWithoutReplacementLeafAudit.serfling_style_mean_tail_bound`

What’s still pending is the concrete instantiation for “uniformly sample `m` leaves without replacement” (random permutation / simple random sample), i.e. constructing the martingale differences for the sampling process and discharging the boundedness + conditional-mean-zero hypotheses.

## Two-Level Sampling: #Docs vs #Leaves per Doc

For corpus-level statements, you almost always have a **two-stage design**:

1. Sample `D` documents from the corpus distribution.
2. Within each sampled document, sample `m` leaves/chunks (often without replacement).

If the per-doc target is a mean over leaves, the total error naturally splits into:

- **Between-doc error** (how many documents you sampled): scales like `O(1/√D)` for bounded doc-level means.
- **Within-doc error** (how many leaves per doc you audited): scales like `O(1/√m)` (with finite-population correction when sampling without replacement).

One clean (but slightly conservative) way to make this explicit is to target an overall tolerance
`ε_total = ε_between + ε_within` with confidence `1-δ`, and split failure probability:

- Doc sampling: `δ_between := δ/2` gives `| (1/D)∑ μ_doc(d_j) - 𝔼[μ_doc] | ≤ O(√(log(1/δ)/D))`.
- Within-doc sampling: `δ_within := δ/(2D)` per document + union bound ensures
  `max_j |μ̂_doc(d_j) - μ_doc(d_j)| ≤ O(√(log(D/δ)/m))`, hence the average within-doc error is ≤ that same bound.

This makes the tradeoff transparent: for large corpora, `D` is what buys you generalization across documents; `m` is what buys you faithful measurement *within* a document.

This yields a practical budgeting rule for a fixed oracle-query budget `Q ≈ D*m`:

- Increase `m` until within-doc error is “small enough”, then spend remaining budget on increasing `D`.
- If documents are highly heterogeneous (large between-doc variance), `D` matters more.
- If each doc is internally heterogeneous or your tree has many leaves, `m` matters more at first, but shows diminishing returns as `m/N` grows (finite-population correction).

Formalizing this decomposition in Lean is still pending; the clean route is:

- (i) a finite-population (Serfling/Azuma) bound for `μ̂_doc - μ_doc` within each doc,
- (ii) a standard bounded-mean concentration bound over `D` IID documents,
- (iii) a union bound / confidence split to make the within-doc guarantees hold simultaneously over the `D` sampled docs.

The generic composition core for step (iii) is now formalized in
`../FormalProbability/FormalProbability/DSL/SamplingConcentration.lean` as:

- `abs_tail_union_bound_of_abs_le_add`
- `abs_tail_union_bound_delta_split_of_abs_le_add`

The explicit sample-complexity rule is also formalized there as:

- `docsRequired`
- `leavesPerDocRequired`
- `docs_vs_leaves_sample_complexity_rule`

## Outlet-Indexed Slant (Documents as Distributions)

The outlet-slant layer is now formalized in
`../FormalProbability/FormalProbability/DSL/OutletSlant.lean`.

Core objects:

- `DocumentKnowledgeDGP`, `KnowledgeQuery`, `pmfMean`
- `outletMoment`, `outletSlant`, `outletMomentVec`, `outletSlantVec`, `projectedSlant`
- linearity/composition lemmas: `outletSlant_add`, `outletSlant_smul`, `outletSlant_chain`
- `ZeroDistInvariant`, `RespectsZeroDist`, `oracle_composed_zeroDistInvariant_of_respects`

Estimator and concentration layer:

- `outletMomentEstimator`, `outletSlantEstimator`
- `slant_error_concentration_of_two_level`
- `slant_error_docs_vs_leaves_sample_complexity`
- `multi_outlet_docs_vs_leaves_sample_complexity`
- `multi_outlet_docs_vs_leaves_sample_complexity_total_budget`
- non-uniform IPW concentration hooks (in `SamplingConcentration`):
  `NonUniformIPWLeafAudit.abs_tail_bound`,
  `NonUniformIPWLeafAudit.abs_tail_bound_with_variance_proxy`,
  `NonUniformIPWLeafAudit.mean_abs_tail_bound_with_variance_proxy`
- explicit non-uniform without-replacement transfer:
  `NonUniformWithoutReplacementIPWLeafAudit.toNonUniformIPWLeafAudit`,
  `NonUniformWithoutReplacementIPWLeafAudit.abs_tail_bound`,
  `NonUniformWithoutReplacementIPWLeafAudit.mean_abs_tail_bound_with_variance_proxy`
- self-normalized IPW layer (in `IPWTheory`):
  `weightedVariance`,
  `empiricalBernsteinRadius`,
  `empiricalBernstein_bound_ennreal_of_event`,
  `empiricalBernsteinCI_coverage_of_axioms`
- proved failure-event split for self-normalized slant bounds:
  `outletSlant_empiricalBernstein_failure_split`,
  `outletSlant_empiricalBernstein_failure_bound_of_small_radius_and_tail`
- non-uniform Azuma + self-normalized transfer helpers:
  `NonUniformIPWLeafAudit.radius_failure_bound_of_small_radius_and_abs_tail`,
  `NonUniformIPWLeafAudit.radius_failure_bound_of_small_radius_and_mean_abs_tail`
- condExp-derived non-uniform without-replacement model:
  `NonUniformWithoutReplacementIPWLeafAuditFromCondExp`,
  `increment_cond_zero_derived`,
  `toNonUniformWithoutReplacementIPWLeafAudit`
- condExp-derived self-normalized transfer wrappers:
  `NonUniformWithoutReplacementIPWLeafAuditFromCondExp.radius_failure_bound_of_small_radius_and_abs_tail`,
  `NonUniformWithoutReplacementIPWLeafAuditFromCondExp.radius_failure_bound_of_small_radius_and_mean_abs_tail`
- condExp canonical variance proxy + auto tails:
  `NonUniformWithoutReplacementIPWLeafAuditFromCondExp.stepVarianceProxy`,
  `NonUniformWithoutReplacementIPWLeafAuditFromCondExp.mean_abs_tail_bound`,
  `NonUniformWithoutReplacementIPWLeafAuditFromCondExp.radius_failure_bound_of_small_radius_and_mean_abs_tail_auto`
- generic mean-tail plug-in point (for future Freedman/adaptive tails):
  `NonUniformIPWLeafAudit.radius_failure_bound_of_small_radius_and_mean_tail`,
  `NonUniformWithoutReplacementIPWLeafAuditFromCondExp.radius_failure_bound_of_small_radius_and_mean_tail`
- slant-error ↔ condExp/WOR partial-sum bridge theorems:
  `outletSlant_empiricalBernstein_failure_bound_of_condExpWOR_mean_identification`,
  `outletSlant_empiricalBernstein_failure_bound_of_condExpWOR_mean_identification_and_variance_floor_neff_window`
- auto-tail slant bridge + one-call composition:
  `outletSlant_empiricalBernstein_failure_bound_of_condExpWOR_mean_identification_auto_tail`,
  `outletSlant_empiricalBernstein_failure_bound_of_condExpWOR_mean_identification_and_variance_floor_neff_window_auto_tail`
- multi-outlet slant bridge composition:
  `multi_outlet_outletSlant_empiricalBernstein_failure_bound_of_condExpWOR_mean_identification`,
  `multi_outlet_outletSlant_empiricalBernstein_failure_bound_of_condExpWOR_mean_identification_uniform`,
  `multi_outlet_outletSlant_empiricalBernstein_failure_bound_of_condExpWOR_mean_identification_auto_tail_uniform`
- `D` vs `m` specialization with condExp/WOR within-doc non-uniform noise:
  `slant_error_docs_vs_leaves_sample_complexity_of_condExpWOR_within`

Mergeable-sketch bridge (for chunked/distributed auditing):

- `MomentSketchState`, `SlantSketchState`
- `buildMomentSketchState_append`, `buildSlantSketchState_append`
- `querySlantSketchState_append_eq_merge_query`
- `slantSizedMergeableQuerySketch`
- `slantSketch_hierarchicalMergeable`
- `slantSketch_query_mergeTree_eval_eq_direct`
- second-moment mergeable sketches for self-normalized bounds:
  `MomentEBSketchState`, `SlantEBSketchState`,
  `slantEBSizedMergeableQuerySketch`,
  `slantEBSketch_query_mergeTree_eval_eq_direct`,
  `slantEBSketch_radius_mergeTree_eval_eq_direct`

ThinkingTrees bridge modules (conservative attribution split):

- `lean3/FormalProofs/OPT/MergeableReduction.lean`
  - `streamConcat`, `opsBuildDet`, `opsValidDet`, `opsMergeDet`
  - `ops_mergeClosed_of_global`
  - `ops_hierarchical_mergeable_of_global`
  - `ops_reduction_to_classical_mergeable`
  - `ops_merge_commutative_oracle`
- `lean3/FormalProofs/DSL/MergeableCertificates.lean`
  - `tree_gap_bound_transport_upper`
  - `tree_gap_bound_transport_upper_prob`
  - `dpo_tree_gap_bounded_by_sketch_upper`
  - `grpo_pl_tree_gap_bounded_by_sketch_upper`
  - `grpo_rl_tree_gap_bounded_by_sketch_upper`
  - `kll_hierarchical_mergeability_available`
  - `gk_one_way_mergeability_available`
  - `gk_chunk_fold_ingestion_available`

Attribution split used in this bridge:

- Mergeable-summary lineage (Agarwal et al. 2012/2013 and descendants): closure/compositional correctness over merge trees and one-way ingestion interfaces.
- TreePO-specific contribution: oracle-semantic preservation and objective/IPW certificate transport, implemented here as upper-bound substitution on already-proved TreeIPW gap certificates.

## Importance Sampling Over Leaves (Practical)

Uniform leaf sampling is rarely optimal. If some leaves are more likely to contain “decision-critical” information (or more likely to violate local laws), you can bias sampling towards them and correct via IPW/HT.

Practical template:

- Choose a leaf-sampling policy `q(u | x)` (possibly adaptive) that over-samples “likely important” leaves.
- Ensure **positivity**: `q(u|x) > 0` for all leaves you might want to make population claims about.
- Use Horvitz–Thompson style estimators to recover unbiased doc-level/corpus-level means.

What remains to be formalized cleanly is concentration for these *non-uniform, without-replacement* designs (and for adaptive `q`); the existing Lean already contains:

- IPW unbiasedness primitives (`FormalProbability/DSL/IPWTheory.lean` and `lean3/FormalProofs/DSL/IPWTheory.lean`)
- Tree-level IPW wrappers + empirical Bernstein interfaces (`lean3/FormalProofs/DSL/TreeIPW.lean`)

The next step is to connect leaf-level (within-doc) sampling designs to those IPW abstractions and add a concentration layer (Azuma/empirical Bernstein) that respects tree-structured dependence.

## TreePO Objective (Formal Template)

TreePO is a preference-learning objective where preferences are generated at nodes or spans of a summarization tree.

A clean formal template is:

\[
\mathcal{L}_\text{TreePO}(\theta) =
\mathbb{E}_{x \sim \mu}
\mathbb{E}_{u \sim q(\cdot \mid x)}
\mathbb{E}_{p \sim gen(S(u))}
\ell_\theta(S(u), p)
\]

Key points:

- `S(u)` is the span at node `u`.
- `q(u|x)` is a node sampler, which should be oracle-indexed if invariance is desired.
- `gen` is the preference generator (pairwise or group). This can be expressed as a `PrefProgram` and proved oracle-indexed via `OracleIndexedProgram` in `lean3/FormalProofs/OPT/PreferenceLearning.lean`.
- If only a subset of nodes is oracle-labeled, use IPW with joint propensity `π(x,u,p)` to produce an unbiased objective estimate.

## Concrete Lean Extension (Now Implemented)

This section shows the concrete symbols to use when wiring TreePO end-to-end in Lean.
The packaged method-level certificates are in:

- `lean3/FormalProofs/DSL/TreePOEndToEnd.lean`

### 1) Tree-level sampling model

Use the model in `lean3/FormalProofs/OPT/SamplingModel.lean`:

```lean
def NodeSampler (Strings Node : Type*) := Strings → PMF Node
def NodeGroupGenerator (Node A : Type*) (k : ℕ) := Node → PMF (Fin k → A)

def OracleIndexedNodeSampler {Strings Y Node : Type*} [PseudoMetricSpace Y]
    (fstar : Strings → Y) (q : NodeSampler Strings Node) : Prop := ...

structure TreePreferenceSamplingModel (Strings Node A : Type*) (k : ℕ) where
  docDist : PMF Strings
  nodeSampler : NodeSampler Strings Node
  nodeSpan : Node → Strings
  groupGen : NodeGroupGenerator Node A k
```

TreePO population objective:

```lean
noncomputable def ExpectedTreePreferenceLoss {Strings Node A : Type*} {k : ℕ}
    (model : TreePreferenceSamplingModel Strings Node A k)
    (loss : Strings → (Fin k → A) → ℝ) : ℝ := ...
```

### 2) Programmatic generator composition

Use `PrefProgram` in `lean3/FormalProofs/OPT/PreferenceLearning.lean` to encode nested sampling.

```lean
inductive PrefProgram (Strings : Type*) (α : Type*) : Type _ where
  | pure : α → PrefProgram Strings α
  | sample : {β : Type*} → (Strings → PMF β) →
      (β → PrefProgram Strings α) → PrefProgram Strings α

def PrefProgram.run : PrefProgram Strings α → Strings → PMF α := ...
def OracleIndexedProgram (fstar : Strings → Y) :
    PrefProgram Strings α → Prop := ...

lemma oracle_indexed_run ... :
    OracleIndexedGenComb (PrefProgram.run prog) fstar := ...
```

Zero-distortion transport for program-defined objectives:

```lean
lemma expected_pref_loss_prog_eq_of_zero_dist ... :
  ExpectedPrefLossProg loss μ_X prog = ExpectedPrefLossProg loss μ_Z prog := ...
```

### 3) Sampled objective with IPW

For partial oracle labeling, use `TreePreferenceSample` and `ipwPreferenceLoss` in
`lean3/FormalProofs/DSL/TreeIPW.lean`.

```lean
structure TreePreferenceSample (Strings Node A : Type*) (k : ℕ) where
  doc : Strings
  node : Node
  group : Fin k → A
  loss : ℝ
  propensity : TreePropensity
  policy_version : ℕ
  is_oracle_labeled : Bool

def ipwPreferenceLoss
    (samples : List (TreePreferenceSample Strings Node A k)) : ℝ := ...
```

Unbiasedness connection theorem (Bernoulli HT form):

```lean
theorem ipw_preference_loss_connection ... :
  ∫ ω, htExpEstimator p pi loss ω ∂bernoulliProductMeasure pi hpi_pos hpi_le
    = Exp p loss := ...
```

### 4) Gap-to-objective bridge used by TreePO

Use the generic and method-specific wrappers in `lean3/FormalProofs/DSL/TreeIPW.lean`:

```lean
theorem tree_gap_bounded_by_ipw ...
theorem dpo_tree_gap_bounded_ipw ...
theorem grpo_pl_tree_gap_bounded_ipw ...
theorem grpo_rl_tree_gap_bounded_ipw ...
```

These are the direct theorem hooks for turning sampled tree audits into bounds on
training-objective mismatch.

## Theoretical Benefit (Formal Chain)

TreePO gains are formalized as a chain of implications.

1. IPW yields unbiased estimates of the preference objective under design-based sampling.
2. Local laws L1, L2, L3 imply zero distortion for `ZR`, yielding exact loss equivalence.
3. When distortion is not exactly zero, Lipschitz losses yield quantitative gap bounds.
4. Tree-level IPW estimates provide valid upper bounds on the distortion term, and
   `tree_gap_bound_transport_upper`/`tree_gap_bound_transport_upper_prob` lift
   those upper bounds into objective-gap certificates.
5. Node-level group generators may vary with the span: if `groupGen u = gen (nodeSpan u)`, the TreePO gap bound still holds via `tree_gap_bounded_ipw_gen`.
6. If the doc-level generator is stable in L1 (PMF shift bounded by oracle distance), then GRPO-PL admits a TreePO gap bound with an extra `M * L_gen` term; see `grpo_pl_tree_gap_bounded_ipw_gen`.

The above uses existing results in `lean3/FormalProofs/OPT/PreferenceLearning.lean`, `lean3/FormalProofs/OPT/PreferenceBounds.lean`, `lean3/FormalProofs/OPT/AuditBounds.lean`, and `lean3/FormalProofs/DSL/TreeIPW.lean`.

## Missing Formalization (Minimal Additions)

These are the smallest additions needed to make TreePO fully formal without axioms beyond the current design-based sampling axioms.

- IPW unbiasedness for bounded preference losses (DONE: `ipw_preference_loss_connection` in `lean3/FormalProofs/DSL/TreeIPW.lean`).
- Tree preference sampling model (DONE: `TreePreferenceSamplingModel` in `lean3/FormalProofs/OPT/SamplingModel.lean`).
- Oracle-indexed node sampler predicate (DONE: `OracleIndexedNodeSampler` in `lean3/FormalProofs/OPT/SamplingModel.lean`).
- Tree-size rewrites + sample-size scaling for leaf/merge audits (DONE: `lean3/FormalProofs/OPT/AuditSizes.lean`).
- Without-replacement (Serfling/Azuma) chunk sampling: conditional Hoeffding glue is DONE
  (`lean3/FormalProofs/OPT/SerflingAudit.lean` and
  `../FormalProbability/FormalProbability/DSL/SamplingConcentration.lean`), with a concrete
  finite-population audit model now available via
  `UniformWithoutReplacementLeafAudit`. The remaining work is a fully explicit
  random-permutation/simple-random-sample construction discharging that model’s
  assumptions from first principles.
- Two-level sample-size tradeoff lemma (`#docs` vs `#leaves/doc`) combining:
  doc-level concentration + within-doc (finite population) concentration + union bound
  (DONE at the rule level via `docs_vs_leaves_sample_complexity_rule`; concrete
  tail-model instantiations for a specific application remain user/model specific).
- Outlet-indexed slant formalization with weaker zero-distance invariance and mergeable
  sketch composition (DONE in `../FormalProbability/FormalProbability/DSL/OutletSlant.lean`).
- Concentration for importance-weighted (non-uniform) leaf sampling:
  the Azuma-centered martingale layer is now DONE in
  `../FormalProbability/FormalProbability/DSL/SamplingConcentration.lean`
  via `NonUniformIPWLeafAudit`; still PENDING are sharper/self-normalized
  empirical-Bernstein/Freedman concentration proofs (beyond the current
  radius-event decomposition + axiom-wrapper layer) and fully model-derived
  adaptive-policy instantiations.
- A final lemma that combines IPW-based distortion bounds with the Lipschitz preference-gap bound
  (DONE: `tree_gap_bounded_by_ipw` and method-specific wrappers
  `dpo_tree_gap_bounded_ipw`, `grpo_pl_tree_gap_bounded_ipw`, `grpo_rl_tree_gap_bounded_ipw`
  in `lean3/FormalProofs/DSL/TreeIPW.lean`, plus sketch-upper transport wrappers
  `dpo_tree_gap_bounded_by_sketch_upper`,
  `grpo_pl_tree_gap_bounded_by_sketch_upper`,
  `grpo_rl_tree_gap_bounded_by_sketch_upper`
  in `lean3/FormalProofs/DSL/MergeableCertificates.lean`).

## Applied Synthesis: Single-Oracle Triple-Honesty TreePO (Optional Proxy)

For applied deployments, truth labels (human and/or trusted dataset labels) are limited;
the learned oracle is a budget-saving approximation and is itself adaptive.

Use these truth/prediction objects:

- `Y*`: latent human preference/label function (target truth).
- `O_t`: learned oracle approximator at round `t`.
- `O_t^online`: update/adaptation view of `O_t`.
- `O_t^eval`: frozen/OOF evaluation view of `O_t`.

and one query policy:

- `Q_t` with logged propensity `pi_t` and exploration floor.

### Why this matters

When `O` is trained, interference is bidirectional:

- `O -> C` (oracle residuals drive chunking),
- `C -> S` (chunking changes summarizer inputs),
- `S -> O` (summarizer shifts oracle train/eval distribution),
- `Q -> {C,S,O}` (adaptive labeling changes observed distribution).

This is why three-layer honesty is not just cosmetic in this setting.

### Operational role split (single oracle)

- `O^online`: used for chunk adaptation, query prioritization, and training-time feedback.
- `O^eval`: used for calibration diagnostics, OOF nuisance prediction for DR estimators, and reporting.
- Final reported claims are anchored to truth labels (human and/or trusted dataset labels), not oracle-only scores.

Optional speed layer:
- Introduce `P_t` (small LM or embedding+head) for fast chunk/query heuristics.
- `P_t` is optional and does not change core theorem assumptions.

### Minimal single-oracle honesty contract

Keep the existing `ThreeLayerSplit` semantics with roles `(chunk, summarizer, oracle)`,
and bind oracle updates/evaluation to separate views under the oracle role:

1. `C` updates only from `chunk=train` docs.
2. `S` updates only from `summarizer=train` docs.
3. `O^online` updates only from `oracle=train` truth-labeled docs.
4. `O^eval` is frozen/OOF on `oracle=eval` docs.
5. Oracle predictions used by `C`/`S` are out-of-fold with respect to `oracle` role.
6. Final reporting is on `E = E_C ∩ E_S ∩ E_O`.
7. Logged propensities satisfy positivity (`pi >= epsilon > 0`).

If a proxy `P_t` is used, treat it as an auxiliary scorer under the same oracle split discipline:
- updates from oracle-train only,
- eval diagnostics from oracle-eval only,
- no proxy-only final claims.

### Truth label provenance

Truth labels can come from:
- human annotation (`TruthLabelSource.human`), or
- trusted dataset labels (`TruthLabelSource.dataset`).

Lean primitives for this are in `lean3/FormalProofs/DSL/CoreDefinitions.lean`
via `TruthLabelSource`, `ApproxLabelSource`, and `LabelObservation`.

### Unknown/heterogeneous propensity workaround

When per-unit propensities are missing or heterogeneous, use a robust logged
propensity interface with a positivity floor:

- `LoggedJointPropensity`,
- `TreeSampleWithProvenance`,
- `TreePreferenceSampleWithProvenance`,
- `ipwViolationRateRobust`, `ipwPreferenceLossRobust`.

These provide stable weighting via effective propensity `max(floor, logged)` and
preserve overlap assumptions for design-based inference.

### Estimation target and estimator

For frozen round-`t` policy state, target:

\[
J_t = \mathbb{E}\left[\ell_{\theta_t}(x, u, p, Y^*(x,u,p))\right].
\]

Under logged `pi`, estimate `J_t` with HT/IPW or cross-fitted DR on honest eval docs.
This is the empirical deployment objective aligned with the existing TreeIPW theorems.

### Suggested theorem packaging (next Lean step)

Package a corollary that composes:

- `threeLayer_eval_bound` (honest evaluation isolation),
- TreeIPW unbiasedness/concentration wrappers,
- DR/OOF interface assumptions for oracle nuisance models (`O^eval`),
- and (optional) a proxy-transfer term for `P` if used.

This yields a bound decomposition of the form:

\[
\text{total error}
\le
\text{sampling/estimation error}
 + \text{oracle approximation gap}
 + \text{optional proxy-transfer gap}
 + \text{residual adaptation leakage},
\]

with the leakage term removed under the honesty contract above.

### Runtime plan (implementation-facing)

1. Log full lineage keys: doc id, round id, split roles, policy versions, propensities, oracle model ids.
2. Train/update `O^online` on truth labels (`oracle=train`).
3. Build `O^eval` as frozen snapshot or strict OOF oracle-eval predictions.
4. Optional: train proxy `P` for high-throughput chunk/query decisions.
5. Compute OOF `O^eval` predictions for estimator/nuisance use.
6. Report honest eval metrics plus ESS/calibration/seed-variance diagnostics.
7. Run budget-matched ablations: single-oracle vs single-oracle+proxy.

## External Repository References

The design-based sampling theory also lives in the sibling repo `../FormalProbability` and should be treated as a shared dependency for TreePO’s sampling guarantees.

Key paths:

- `../FormalProbability/FormalProbability/DSL/CoreDefinitions.lean`
- `../FormalProbability/FormalProbability/DSL/SamplingTheory.lean`
- `../FormalProbability/FormalProbability/DSL/SamplingConcentration.lean`
- `../FormalProbability/FormalProbability/DSL/OutletSlant.lean`
- `../FormalProbability/FormalProbability/DSL/IPWTheory.lean`
