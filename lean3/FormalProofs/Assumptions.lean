import FormalProofs.OPT.LocalLaws
import FormalProofs.OPT.GlobalAssumptions
import FormalProofs.OPT.ScoreTransport
import FormalProofs.OPT.InformationSufficiency
import FormalProofs.OPT.PreferenceLearning
import FormalProofs.OPT.PreferenceBounds
import FormalProofs.OPT.InfluenceWeightedLocalLaws
import FormalProofs.DSL.Honesty
import FormalProofs.DSL.DocumentStructure
import FormalProofs.DSL.UnifiedLearningCertificate

/-!
# FormalProofs/Assumptions.lean

## Paper Assumptions and Their Implications

This file collects all assumptions from the paper in one place and states
which theorems they imply. It serves as a "map" for readers to understand
the logical structure of the formalization.

## Summary Table

| Assumption Class | Theorems Implied | File |
|------------------|------------------|------|
| L1 + L2 + L3 (Local Laws) | Multi-round preservation: E[D(Z^R, x)] = 0 | ExpectationTheory |
| + Influence-weighted audit overlap | Informative finite-sample local-law certificates | InfluenceWeightedLocalLaws |
| + Oracle-Measurability | Preference learning equivalence | PreferenceLearning |
| + Lipschitz | Quantitative gap bounds (DPO) | PreferenceBounds |
| + Random Utility Model | GRPO-PL/RL gap bounds (expected Lipschitz) | PreferenceBounds |

## Paper Notation Correspondence

| Lean Name | Paper Name | Description |
|-----------|------------|-------------|
| L1 | C1 (Sufficiency) | Leaf preserves oracle: E[D(g(b), b)] = 0 |
| L2 | C3 (Merge) | Merge preserves oracle |
| L3 | C2 (Idempotence) | Re-summary is inert: E[D(g(Z), Z)] = 0 for Z ∈ range(g) |
| A1_global | Global Sufficiency | ∀ z, D(g z, z) = 0 |
| A2_global | Two-Route Identity | ∀ u v, D(u*v, g(g u * g v)) = 0 |
| A3_global | Strict Oracle Merge | ∃ M : Y → Y → Y with properties |
| TopLevelIID / TopLevelExchangeable | Sampling unit assumption | IID/exchangeability applies to `(X_i, Y_i*)`, not rows |
| TopLevelSplit | Honest unit split | Train/eval roles live on top-level units, not derived rows |
| ParentOf | Derived-row parent map | Leaves/nodes/audit rows inherit roles from their top-level unit |
| ChunkerObjectiveTerms | Chunker objective | Downstream loss + law mass + radius + cost + boundary regularization |
| Span / AdmissiblePartition | Document support | Finite ordered spans and non-overlapping covering chunk partitions |
| RunManifestContract | Run manifest | Parent IDs, artifacts, propensities, influence weights, and split roles |
| UnifiedLearningErrorCertificate | Unified certificate | Reported estimate plus law/calibration/estimation/clipping radii |
| UnifiedLearningPaperAssumptions | Final paper theorem context | Sampling + honesty + chunk/manifest contracts |
| RootErrorControlledByInfluenceMass | Influence propagation | root error ≤ weighted local-law mass |
| InfluenceWeightedAuditOverlap | Audit overlap | consequential rows have non-tiny logged propensity |

## Theorem Dependency Structure

```
                      ┌─────────────────┐
                      │  Local Laws     │
                      │  L1 + L2 + L3   │
                      └────────┬────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │     multi_round_proper         │
              │  E[D(Z^R, x)] = 0 for all R    │
              │     (ExpectationTheory)        │
              └────────────────┬───────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
┌────────────────┐   ┌────────────────┐   ┌────────────────┐
│ DPO Equivalence│   │ GRPO-PL Equiv  │   │ GRPO-RL Equiv  │
│ + Oracle-Meas  │   │ + Oracle-Meas  │   │ + Oracle-Meas  │
└────────┬───────┘   └────────┬───────┘   └────────┬───────┘
         │                     │                     │
         ▼                     ▼                     ▼
┌────────────────┐   ┌────────────────┐   ┌────────────────┐
│ + Lipschitz    │   │ + Lipschitz    │   │ + Lipschitz    │
└────────┬───────┘   └────────┬───────┘   └────────┬───────┘
         │                     │                     │
         ▼                     ▼                     ▼
    dpo_gap_bounded     grpo_pl_gap_bounded   grpo_rl_gap_bounded
```

-/

set_option linter.mathlibStandardSet false

open scoped BigOperators Real Nat Classical Pointwise

noncomputable section

namespace PaperAssumptions

variable {Strings : Type*} [Monoid Strings]
variable {A : Type*}
variable {Y : Type*} [PseudoMetricSpace Y]

/-!
## Section 0: Top-Level Units, Derived Rows, and Chunker Objective

The paper's statistical train/eval split is over top-level cases/documents
`X_i`.  Sub-document spans, tree nodes, and local-law rows are dependent derived
objects, so their roles are inherited through a parent map rather than treated
as separate IID samples.
-/

/-- **Top-level honest split**:
the train/eval split lives on the case/document unit supporting the reported
generalization claim. -/
abbrev TopLevelUnitSplit := @DSL.TopLevelSplit

/-- **Top-level paired observation**:
the case/document together with its truth target. -/
abbrev TopLevelObservation := @DSL.topLevelObservation

/-- **Top-level IID assumption**:
IID is asserted over paired top-level observations `(X_i, Y_i*)`, not over
spans, nodes, or local-law rows. -/
abbrev TopLevelIIDAssumption := @DSL.TopLevelIID

/-- **Top-level exchangeability assumption**:
finite-dimensional exchangeability is asserted over paired top-level
observations `(X_i, Y_i*)`, not over derived rows. -/
abbrev TopLevelExchangeabilityAssumption := @DSL.TopLevelExchangeable

/-- **Derived-row parent map**:
maps spans, tree nodes, and local-law audit rows back to their top-level unit. -/
abbrev DerivedRowParent := @DSL.ParentOf

/-- **Sibling relation for derived rows**:
two rows are siblings when they share the same top-level unit. -/
abbrev SameTopLevelUnit := @DSL.SameTopLevelUnit

/-- **Inherited split role**:
a derived row's train/eval role is determined by its parent top-level unit. -/
abbrev InheritedSplitRole := @DSL.inheritedSplitRole

/-- **Derived-row honest training**:
training over rows may depend only on rows whose parent units are in train. -/
abbrev DerivedRowHonestTraining := @DSL.DerivedRowHonestTraining

/-- **Derived-row honest evaluation**:
evaluation over rows may depend only on rows whose parent units are in eval. -/
abbrev DerivedRowHonestEvaluation := @DSL.DerivedRowHonestEvaluation

/-- **K-fold honest artifact training**:
fold-`k` artifacts are trained only on top-level units outside fold `k`. -/
abbrev KFoldHonestArtifactTraining := @DSL.KFoldHonestTraining

/-- **K-fold honest evaluation**:
fold-`k` evaluation depends only on top-level units inside fold `k`. -/
abbrev KFoldHonestArtifactEvaluation := @DSL.KFoldHonestEvaluation

/-- **Chunker policy**:
maps a top-level unit and a frozen artifact bundle to a partition. -/
abbrev ChunkerPolicy := @DSL.ChunkerPolicy

/-- **Chunker objective terms**:
downstream loss, local-law residual mass, certificate radius, compute/query
cost, and boundary regularization with explicit weights. -/
abbrev ChunkerObjectiveTerms := @DSL.ChunkerObjectiveTerms

/-- **Chunker objective minimizer**:
the selected partition minimizes the weighted objective over admissible
partitions for a fixed top-level unit and frozen artifacts. -/
abbrev ChunkerObjectiveMinimizer := @DSL.IsChunkerObjectiveMinimizer

/-- **Unified learning honesty**:
chunker, `g`, and oracle/readout training are each honest on their three-layer
split, and final evaluation uses a frozen artifact bundle on the joint eval
view. -/
abbrev UnifiedLearningHonesty := @DSL.UnifiedLearningHonesty

/-- **Document span**:
half-open support interval used for chunks, nodes, and audit rows. -/
abbrev DocumentSpan := DSL.Span

/-- **Admissible chunk partition**:
nonempty valid spans that cover the finite ordered document without overlap. -/
abbrev AdmissibleChunkPartition := DSL.AdmissiblePartition

/-- **Finite top-level unit**:
top-level case/document with a positive finite ordered length. -/
abbrev FiniteTopLevelUnit := DSL.FiniteTopLevelUnit

/-- **Chunk partition contract**:
each top-level unit receives an admissible chunk partition. -/
abbrev ChunkPartitionContract := DSL.ChunkPartitionContract

/-- **Run manifest contract**:
the theorem-facing log with parent unit IDs, artifact lineage, propensities,
effective propensities, and influence weights. -/
abbrev RunManifestContract := DSL.RunManifestContract

/-- **Manifest roles consistent with the three-layer split**:
logged chunker/g/oracle roles agree with top-level split roles. -/
abbrev ManifestRolesConsistent := @DSL.ManifestRolesConsistent

/-- **Unified learning error certificate**:
reported honest estimate plus local-law, calibration, estimation, and clipping
radii for the final paper-facing envelope. -/
abbrev UnifiedLearningErrorCertificate := DSL.UnifiedLearningErrorCertificate

/-- **Unified learning paper assumptions**:
the final theorem context bundling top-level sampling, honesty, admissible
chunking, manifest consistency, and row-support validity. -/
abbrev UnifiedLearningPaperAssumptions := @DSL.UnifiedLearningPaperAssumptions

/-- **Unified component evidence**:
provenance records for local-law, calibration, estimation, and clipping radii. -/
abbrev UnifiedLearningComponentEvidence := @DSL.UnifiedLearningComponentEvidence

/-!
## Section 1: Local Laws (Consistency Conditions)

The local laws are the core testable conditions on a summarizer g.
They can be audited empirically by sampling documents and checking distortion.
-/

/-- **Paper Condition C1** (Sufficiency): Summarizing leaves preserves oracle.

Mathematical statement: E[D(g(b), b)] = 0 for all leaves b in tree T.

This is Lean's L1. -/
abbrev C1_Sufficiency := @L1

/-- **Paper Condition C2** (Idempotence): Re-summarizing is inert.

Mathematical statement: E[D(g(Z), Z)] = 0 for all Z in range(g).

This is Lean's L3 (note the numbering swap from paper). -/
abbrev C2_Idempotence := @L3

/-- **Paper Condition C3** (Merge Consistency): Merging preserves oracle.

Mathematical statement: E[D(reduce g (node T_L T_R), S(node T_L T_R))] = 0

This is Lean's L2 (note the numbering swap from paper). -/
abbrev C3_MergeConsistency := @L2

/-!
## Section 1b: Influence-Weighted Audit and Bounds Assumptions

Approximate local laws alone give a valid finite-depth sum bound, but the bound
can be uninformative if all root-relevant error is concentrated on a row with
vanishing audit probability.  The influence-weighted layer makes the additional
finite-sample assumption explicit:

1. root/document error is controlled by an influence-weighted local-law mass;
2. the audit design assigns enough logged propensity to consequential rows.

This is weaker than a "no hidden needles" assumption.  Needles may exist; they
must not be adversarially hidden in rows whose `lambda / pi` ratio is arbitrarily
large.
-/

/-- **Influence-weighted local-law mass**:
`sum_a lambda(a) * residual(a)` over finite C1/C2/C3 audit rows. -/
abbrev InfluenceWeightedLocalLawMass :=
  @FormalProofs.OPT.weightedLocalLawMass

/-- **Influence-weighted design effect**:
`sum_a lambda(a)^2 / pi(a)`, the variance/design-effect proxy for the audit. -/
abbrev InfluenceWeightedDesignEffect :=
  @FormalProofs.OPT.influenceDesignEffect

/-- **Worst influence-to-propensity ratio**:
`lambda(a) / pi(a) <= W` for every audit row. -/
abbrev InfluenceWeightedWorstRatioBound :=
  @FormalProofs.OPT.influenceWorstRatioBound

/-- **Influence-weighted audit overlap**:
positive logged propensities plus bounded design effect and worst ratio.

This is the formal "no adversarially hidden consequential needles" assumption. -/
abbrev InfluenceWeightedAuditOverlapAssumption :=
  @FormalProofs.OPT.InfluenceWeightedAuditOverlap
abbrev InfluenceWeightedAuditOverlapAxiom :=
  @InfluenceWeightedAuditOverlapAssumption

/-- **Influence propagation assumption**:
root/document error is bounded by influence-weighted local-law residual mass. -/
abbrev RootErrorControlledByInfluenceMassAssumption :=
  @FormalProofs.OPT.RootErrorControlledByInfluenceMass
abbrev RootErrorControlledByInfluenceMassAxiom :=
  @RootErrorControlledByInfluenceMassAssumption

/-- Uniform proxy calibration transfers proxy local-law residuals to true
oracle residuals with `2 * eps` row slack. -/
abbrev InfluenceWeightedCalibrationTransfer :=
  @FormalProofs.OPT.weightedOracleMass_le_proxy_plus_calibration

/-- Root-error certificate from any influence-weighted local-law mass upper
bound. -/
abbrev InfluenceWeightedRootErrorBound :=
  @FormalProofs.OPT.rootError_le_of_influence_weighted_mass_upper

/-- Root-error certificate combining proxy estimation, statistical radius, and
calibration radius. -/
abbrev InfluenceWeightedProxyRootErrorBound :=
  @FormalProofs.OPT.rootError_le_proxy_estimate_plus_stat_plus_calibration

/-- Packaged finite-sample influence-weighted error certificate. -/
abbrev InfluenceWeightedErrorCertificateAssumptionSurface :=
  @FormalProofs.OPT.InfluenceWeightedErrorCertificate

/-!
## Section 2: Global Assumptions

The global assumptions (A1, A2, A3) are STRONGER than local laws.
They imply the local laws for ANY tree structure.

`A3_global` is the strict oracle-output case: it requires the oracle values
themselves to compose. Classical mergeable sketches are more general because
bounded sketch states can merge before applying a final readout.

**Key Derivation** (GlobalAssumptions.lean):
  A1_global + A2_global + A3_global ⟹ L1 + L2 + L3 for any tree
-/

/-- **Global Sufficiency (A1)**: Oracle distortion is zero for ALL strings.

Stronger than C1 which only requires this for leaves. -/
abbrev A1_GlobalSufficiency := @A1_global

/-- **Two-Route Identity (A2)**: Joint and disjoint summarization are equivalent.

∀ u v : Strings, D(u*v, g(g u * g v)) = 0 -/
abbrev A2_TwoRouteIdentity := @A2_global

/-- **Strict Merge Function Existence (A3)**: Oracle-level merge function exists.

∃ M : Y → Y → Y such that f*(g(g u * g v)) = M(f*(g u), f*(g v)) -/
abbrev A3_MergeFunction := @A3_global

/-!
## Section 2b: Deterministic Global↔Local IFF Bridges

These aliases expose the strongest deterministic equivalences currently
formalized between global assumptions and local laws.
-/

/-- Deterministic L1 IFF: local C1 on a tree is exactly leafwise oracle sufficiency. -/
abbrev det_l1_iff_leafwise := @L1_deterministic_iff_leafwise

/-- Deterministic L3 IFF: local C2/L3 is exactly in-range oracle sufficiency. -/
abbrev det_l3_iff_inrange := @L3_deterministic_iff_inRange

/-- Global A1 IFF deterministic L1 over all trees. -/
abbrev a1_iff_l1_all_trees := @A1_iff_L1_for_all_trees

/-- Deterministic C3/L2 IFF on a two-leaf tree equals pointwise A2 at `(u,v)`. -/
abbrev det_l2_two_leaf_iff_a2_pointwise := @L2_deterministic_two_leaf_iff_A2_pointwise

/-- Global A2 IFF deterministic C3/L2 on all two-leaf trees. -/
abbrev a2_iff_l2_two_leaf_trees := @A2_iff_L2_on_two_leaf_trees

/-- Under A1+A3, global A2 IFF deterministic C3/L2 on all trees. -/
abbrev a2_iff_l2_all_trees_given_a1a3 := @A2_iff_L2_on_all_trees_of_A1_A3

/-- Under A1+A3, deterministic C3/L2 on all trees IFF checking only two-leaf trees. -/
abbrev l2_all_trees_iff_two_leaf_trees_given_a1a3 := @L2_on_all_trees_iff_two_leaf_trees_of_A1_A3

/-- Under surjectivity, deterministic C1/L1(all trees) IFF deterministic C2/L3. -/
abbrev l1_all_trees_iff_l3_surjective := @L1_on_all_trees_iff_L3_of_surjective

/-- Under A3+surjectivity, `(A1 ∧ A2)` IFF `(C2/L3 ∧ C3/L2 on all trees)`. -/
abbrev a1a2_iff_l3_l2_all_trees_given_a3_surjective :=
  @A1_A2_iff_L3_and_L2_on_all_trees_of_A3_surjective

/-- Under A3+surjectivity, `(A1 ∧ A2)` IFF `(C2/L3 ∧ C3/L2 on two-leaf trees)`. -/
abbrev a1a2_iff_l3_l2_two_leaf_given_a3_surjective :=
  @A1_A2_iff_L3_and_L2_on_two_leaf_trees_of_A3_surjective

/-- Weaker deterministic A1: global sufficiency only on the summary range. -/
abbrev A1_OnSummaryRange := @A1_on_summary_range

/-- Non-surjective converse: deterministic C2/L3 IFF A1 on summary range. -/
abbrev a1_on_summary_range_iff_l3 := @L3_iff_A1_on_summary_range

/-- Under surjective summaries, global A1 IFF deterministic C2/L3. -/
abbrev a1_iff_l3_surjective := @A1_iff_L3_of_surjective

/-!
## Section 3: Oracle-Measurability Assumptions

For preference learning equivalence, we need loss functions and generators
to depend on documents ONLY through their oracle values f*(x).
-/

/-- **Oracle-Measurable Policy**: Policy depends on x only through f*(x).

dist(f*(x), f*(x')) = 0 ⟹ pol(x,a) = pol(x',a) for all actions a. -/
abbrev OracleMeasurablePolicy := @DPO.OracleMeasurable

/-- **Oracle-Indexed Pair Generator**: Pairs depend on x only through f*(x).

dist(f*(x), f*(x')) = 0 ⟹ gen(x) = gen(x') -/
abbrev OracleIndexedPairs := @OracleIndexedPairGen

/-- **Oracle-Indexed Group Generator**: Groups depend on x only through f*(x). -/
abbrev OracleIndexedGroups := @OracleIndexedGroupGen

/-- **Oracle-Indexed Ranker**: Rankings depend on x only through f*(x). -/
abbrev OracleIndexedRanking := @OracleIndexedRanker

/-- **Oracle-Measurable Reward**: Reward depends on x only through f*(x). -/
abbrev OracleMeasurableRewardFn := @OracleMeasurableReward

/-!
## Section 3b: Sufficient-Statistic IFF Bridges (Doob-Dynkin)

These aliases expose the strongest "if and only if" connections currently formalized
for oracle sufficiency/transport in the OPT module.
-/

/-- Pointwise oracle factorization through summaries: `f*(X) = h(Z)` for measurable `h`. -/
abbrev OracleFactorizationPointwise := @OracleFactorization'

/-- Oracle sigma containment: `σ(f*(X)) ⊆ σ(Z)`. -/
abbrev OracleSigmaContainment := @OracleSigmaSubset'

/-- Doob-Dynkin oracle IFF (pointwise): factorization iff sigma containment. -/
abbrev doob_dynkin_oracle_iff := @oracle_factorization_iff_sigma_subset

/-- Doob-Dynkin oracle IFF (a.e.): factorization a.e. iff sigma-Z a.e.-measurability. -/
abbrev doob_dynkin_oracle_ae_iff := @oracle_factorization_ae_iff_aestronglyMeasurable

/-- Oracle-indexed conditional densities for task-relevant KLIC statements. -/
abbrev OracleIndexedTaskDensity := @FormalProofs.OPT.OracleIndexedConditionalDensity

/-- Stochastic fixed-partition bridge: local laws imply oracle equality a.e. under the joint law. -/
abbrev stochastic_local_laws_oracle_eq_ae :=
  @FormalProofs.OPT.jointTreeSummaryLaw_oracle_eq_ae_of_localLaws

/-- Stochastic fixed-partition bridge: local laws imply a.e. oracle factorization through summaries. -/
abbrev oracle_sufficiency_joint_law_ae :=
  @FormalProofs.OPT.jointTreeSummaryLaw_oracle_factorizationAE_of_localLaws

/-- Zero task-relevant KLIC under oracle-indexed supervision and the fixed-partition joint law. -/
abbrev zero_task_relevant_klic_joint_law_ae :=
  @FormalProofs.OPT.jointTreeSummaryLaw_taskRelevantKLIC_zero_ae_of_localLaws

/-- Deterministic collision impossibility: merging oracle-distinct inputs blocks any decoder. -/
abbrev summary_collision_impossibility :=
  @FormalProofs.OPT.no_oracle_decoder_of_summary_collision

/-!
## Section 4: Lipschitz Assumptions (for Quantitative Bounds)

For quantitative gap bounds (not just equivalence), we need Lipschitz conditions.
-/

/-- **Policy Lipschitz**: Log-ratio is Lipschitz in oracle distance.

|log(pol(x,a)/ref(x,a)) - log(pol(x',a)/ref(x',a))| ≤ L × dist(f*(x), f*(x')) -/
abbrev PolicyLipschitzCondition := @PolicyLipschitz

/-- **GRPO Policy Lipschitz**: Policy log-prob is Lipschitz. -/
abbrev GRPOPolicyLipschitzCondition := @GRPOPolicyLipschitz

/-- **Reward Lipschitz**: Reward is Lipschitz in oracle value.

|R(y,a) - R(y',a)| ≤ L × dist(y, y') -/
abbrev RewardLipschitzCondition := @RewardLipschitz

/-!
## Section 4b: Random Utility Model Assumption (for Expected Lipschitz)

The quantitative bounds for GRPO-PL and GRPO-RL require **expected Lipschitz** bounds,
not pointwise Lipschitz bounds. This is because rankings/orderings are discontinuous
pointwise (they jump at ties) but are continuous **in expectation** under the Random
Utility Model.

### Random Utility Model (McFadden, 1974)

In Random Utility Models, choices/rankings arise from:
  U_i = V_i + ε_i
where V_i is a continuous deterministic utility and ε_i is i.i.d. noise.

**Key properties:**
1. Ties (where rankings change) have measure zero when ε has continuous density
2. Expected loss is an integral over the noise distribution
3. This integral is continuous in V by dominated convergence
4. With Lipschitz components, the expected loss is Lipschitz

**Examples:**
- Gumbel noise → Multinomial Logit / Plackett-Luce
- Normal noise → Probit model
- Any continuous noise → Expected Lipschitz holds

### Assumptions

The following assumptions state that expected GRPO losses are Lipschitz.
They are **modeling assumptions**, not technical lemmas, and live in
`FormalProbability/DSL/RUM.lean` (re-exported here).

Reference: McFadden, D. (1974). "Conditional logit analysis of qualitative choice behavior"
in Frontiers in Econometrics. Zarembka, P. (ed.), Academic Press.
-/

/-- **Random Utility Model Assumption**: Expected group loss is Lipschitz.

Under continuous underlying utilities with i.i.d. noise (e.g., Gumbel → Plackett-Luce),
the expected loss over groups is Lipschitz in oracle distance.

This is the **single foundational assumption** for preference learning bounds.
It replaces the unprovable pointwise Lipschitz bound. -/
abbrev ExpectedGroupLossLipschitzAssumption
    {Strings A Y : Type*} [PseudoMetricSpace Y] {k : ℕ} :=
  @ExpectedGroupLossLipschitz Strings A Y _ k
abbrev ExpectedGroupLossLipschitzAxiom := @ExpectedGroupLossLipschitzAssumption

/-- Instantiation for GRPO-PL (Plackett-Luce ranking loss). -/
abbrev ExpectedGRPOLossLipschitzAssumption
    {Strings A Y : Type*} [PseudoMetricSpace Y] {k : ℕ} :=
  @ExpectedGRPOLossLipschitz Strings A Y _ k
abbrev ExpectedGRPOLossLipschitzAxiom := @ExpectedGRPOLossLipschitzAssumption

/-- Instantiation for GRPO-RL (PPO-style clipped surrogate + KL). -/
abbrev ExpectedGRPORLLossLipschitzAssumption
    {Strings A Y : Type*} [PseudoMetricSpace Y] :=
  @ExpectedGRPORLLossLipschitz Strings A Y _
abbrev ExpectedGRPORLLossLipschitzAxiom := @ExpectedGRPORLLossLipschitzAssumption

/-!
## Section 5: Main Implication Theorems

These theorems show: Assumptions ⟹ Conclusions
-/

/-- **Main Theorem 1: Multi-Round Preservation**

**Assumptions**: L1, L2, L3 for summarizer g on tree T

**Conclusion**: E[D(Z^R, x)] = 0 for all rounds R ≥ 1

This is the foundational result: local laws imply zero expected distortion
after any number of summarization rounds. -/
abbrev local_laws_imply_zero_distortion := @multi_round_proper

/-- **C2/L3 Round-Inertness Characterization**:
on-range idempotence is equivalent to one-step normalization inertness. -/
abbrev l3_iff_round_inert := @L3_iff_RoundInert

/-- **ZR Step Inertness from L3**:
under C2/L3, the `R -> R+1` normalization term vanishes on `ZR`. -/
abbrev l3_implies_zr_step_inert := @L3_implies_ZR_step_inert

/-- **Main Theorem 2: DPO Training Equivalence**

**Assumptions**:
- Local laws L1, L2, L3 (via ZR having dist 0 from original)
- Policy and reference policy oracle-measurable
- Pair generator oracle-indexed

**Conclusion**: argmin L_DPO(π; Z^R) = argmin L_DPO(π; X)

Training on summaries yields the same optimal policy as training on originals. -/
abbrev dpo_training_equivalence := @dpo_equivalence

/-- **Main Theorem 3: GRPO-PL Training Equivalence**

**Assumptions**:
- Zero expected distortion (from local laws)
- Policy oracle-measurable
- Ranker oracle-indexed
- Group generator oracle-indexed

**Conclusion**: Training on summaries = training on originals for GRPO. -/
abbrev grpo_pl_training_equivalence := @grpo_equivalence

/-- **Main Theorem 4: GRPO-RL Training Equivalence (DeepSeek-R1 Style)**

**Assumptions**:
- Zero expected distortion
- All policies oracle-measurable (current, old, reference)
- Reward oracle-measurable
- Group generator oracle-indexed

**Conclusion**: Training on summaries = training on originals for GRPO-RL. -/
abbrev grpo_rl_training_equivalence := @grpo_rl_equivalence

/-- **Main Theorem 5: DPO Quantitative Gap Bound**

**Assumptions**:
- Local laws L1, L2, L3
- Oracle-measurability
- Policy Lipschitz with constant L_pol
- Bounded oracle distances (D_max) and bounded DPO loss (Loss_max)
- Constant pair generator (or use the oracle-indexed variant)

**Conclusion**: |L_DPO(X) - L_DPO(Z^R)| ≤ 2|β|L_pol × Δ_R

where Δ_R is the expected distortion. When local laws hold exactly, Δ_R = 0. -/
abbrev dpo_quantitative_gap := @dpo_gap_bounded

/-- **Main Theorem 6: GRPO-PL Quantitative Gap Bound**

**Assumptions**:
- Oracle-measurability conditions
- Policy Lipschitz with constant L_grpo
- Bounded oracle distances and bounded group loss

**Conclusion**: |L_GRPO(X) - L_GRPO(Z)| ≤ L_grpo × Δ_R -/
abbrev grpo_pl_quantitative_gap := @grpo_pl_gap_bounded

/-- **Main Theorem 7: GRPO-RL Quantitative Gap Bound**

**Assumptions**:
- Oracle-measurability conditions
- Policy and reward Lipschitz with constant L_grpo_rl
- Bounded oracle distances and bounded reward/group losses

**Conclusion**: |L_GRPO_RL(X) - L_GRPO_RL(Z)| ≤ L_grpo_rl × Δ_R -/
abbrev grpo_rl_quantitative_gap := @grpo_rl_gap_bounded

/-- **Main Theorem 8: Unified Preference Gap**

**This theorem captures the common mathematical structure of ALL gap bounds.**

**Assumptions**:
- Any expected loss with Lipschitz inner expectation E_gen
- Bounded oracle metric space (diameter D_max)
- Explicit bound on |E_gen|

**Conclusion**: |E_X[E_gen] - E_Z[E_gen]| ≤ L × Δ_R

DPO, GRPO-PL, and GRPO-RL are all instances of this unified bound. -/
abbrev unified_preference_gap_bound := @unified_preference_gap_bounded

/-!
## Section 6: Necessity Results

These theorems show the assumptions are NECESSARY, not just sufficient.

**C2 independence counterexample**: Located in CounterexampleExistence.lean

There exists a summarizer g_bad satisfying C1/L1 and fresh-input C3/L2 but
violating C2/L3. This shows idempotence is an independent axiom, not derivable
from the one-pass fresh-input laws.

The construction builds g_bad such that:
- For fresh inputs b ≠ POS, NEG: D(g_bad(b), b) = 0 (satisfies C1/L1)
- For fresh merge inputs: the C3/L2 merge chain is oracle-preserving
- For POS in range: D(g_bad(POS), POS) > 0 (violates L3)

See `ex_c2_independent_formalized` in CounterexampleExistence.lean.
-/

end PaperAssumptions

end
