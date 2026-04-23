-- ============================================================================
-- Shared Infrastructure
-- ============================================================================
import FormalProofs.Shared.Config
import FormalProofs.Shared.BoundedMetricSpace

-- ============================================================================
-- CLT Module: Central Limit Theorem
-- ============================================================================
import FormalProofs.CLT

-- ============================================================================
-- Econometrics Module: direct import only
-- ============================================================================
--
-- `FormalProofs.Econometrics` uses the local `Econometrics.*` namespace, while
-- the newer semiparametric / Chapter 3 coverage route used by the DSL imports
-- `FormalProbability.Econometrics.*` under the same namespace root. Importing
-- both umbrellas into this single top-level module causes a namespace clash
-- (`Econometrics.PotentialOutcomes`).
--
-- Keep the legacy/local Econometrics bundle available as its own direct import,
-- but do not re-export it from this umbrella module.

-- ============================================================================
-- ML Module: Supervised Learning Foundations
-- ============================================================================
import FormalProofs.ML

-- ============================================================================
-- OPT Module: Oracle Preference Training
-- ============================================================================

-- Layer 1: Core Foundations
import FormalProofs.OPT.CoreDefinitions
import FormalProofs.OPT.OracleMeasurable
import FormalProofs.OPT.PreferenceNoise
import FormalProofs.OPT.SamplingModel
import FormalProofs.OPT.TreeProperties

-- Layer 2: Local Laws
import FormalProofs.OPT.LocalLaws

-- Layer 3: Theorems
import FormalProofs.OPT.PreservationTheorems
import FormalProofs.OPT.ExpectationTheory

-- Layer 4: Global Theory
import FormalProofs.OPT.GlobalAssumptions
import FormalProofs.OPT.MergeableReduction
import FormalProofs.OPT.NeuralOperatorSpaces
import FormalProofs.OPT.SketchFlipMergeBridge
import FormalProofs.OPT.SketchSummaryOperators
import FormalProofs.OPT.SketchRecovery
import FormalProofs.OPT.SketchRecoveryInstances
import FormalProofs.OPT.HLLIdempotence
import FormalProofs.OPT.ClassicalSketchLocalLaws
import FormalProofs.OPT.TheoremBackingAssumptions
import FormalProofs.OPT.TheoremBackingStructure
import FormalProofs.OPT.TheoremBackingConsequences
import FormalProofs.OPT.NeuralOperatorTheoremBridge
import FormalProofs.OPT.TheoremBackingMeasurementError
import FormalProofs.OPT.TheoremBackingApproxMeasurementError
import FormalProofs.OPT.ApproxOracleRecovery
import FormalProofs.OPT.LipschitzReadoutFactorization
import FormalProofs.OPT.OracleFiberRelations
import FormalProofs.OPT.FeatureFiberLaws
import FormalProofs.OPT.FiberPreservingObjective
import FormalProofs.OPT.FeatureClassObjectives
import FormalProofs.OPT.LabelScoreObjectives
import FormalProofs.OPT.TwoStageOracleSurrogate
import FormalProofs.OPT.TwoStageLabelScoreObjectives
import FormalProofs.OPT.ProductScoreFiber
import FormalProofs.OPT.ReadoutAlignment
import FormalProofs.OPT.SharedFeatureMultihead
import FormalProofs.OPT.FixedBinaryTreeDiffusion
import FormalProofs.OPT.OptimizationPerturbation
import FormalProofs.OPT.ApproximateLocalLaws
import FormalProofs.OPT.RegularizedObjective
import FormalProofs.OPT.AdaptiveChunkingBridge
import FormalProofs.OPT.RUMSufficientConditions
import FormalProofs.OPT.BigramSketch
import FormalProofs.OPT.BagOfWordsLDARecovery
import FormalProofs.OPT.LeafLocalMixtureUtilityGap
import FormalProofs.OPT.TopicBigramOracle
import FormalProofs.OPT.MarkovPathDGP
import FormalProofs.OPT.ExactUtilityTransport
import FormalProofs.OPT.NodeIndexedLatentState
import FormalProofs.OPT.ExactUtilityTransportInstances
import FormalProofs.OPT.CoverageNormalizedObjective
import FormalProofs.OPT.DiscountedTreeMetaObjective
import FormalProofs.OPT.DiscountedIPWObjective
import FormalProofs.OPT.RidgeRegressionToy
import FormalProofs.OPT.SegmentLDAPipelineToy

-- Layer 5: Applications
import FormalProofs.OPT.ScoreTransport
import FormalProofs.OPT.PreferenceLearning
import FormalProofs.OPT.PreferenceBounds
import FormalProofs.OPT.TrainingPipeline
import FormalProofs.OPT.CutBudgetGuidance
import FormalProofs.OPT.CounterexampleExistence
import FormalProofs.OPT.AuditBounds
import FormalProofs.OPT.AuditSizes
import FormalProofs.OPT.SerflingAudit
import FormalProofs.OPT.AdversarialChunkingExample

-- Layer 5b: Information-Sufficiency / Finite-Support Context
import FormalProofs.OPT.InformationSufficiency
import FormalProofs.OPT.OracleEntropy
import FormalProofs.OPT.OracleSufficientCompression

-- Layer 6: Main Theorems (curated exports)
import FormalProofs.OPT.MainTheorems

-- Layer 7: Empirical Audit Framework
import FormalProofs.OPT.Audit
import FormalProofs.OPT.MeasureTheoreticAudit

-- ============================================================================
-- DSL Module: Debiased/Double Machine Learning
-- ============================================================================
import FormalProofs.DSL
import FormalProofs.DSL.IPWTheory
import FormalProofs.DSL.ClusteredVariance
import FormalProofs.DSL.JudgeCalibration
import FormalProofs.DSL.TreeIPW
import FormalProofs.DSL.MergeableCertificates

-- ============================================================================
-- Documentation
-- ============================================================================
import FormalProofs.Assumptions
import FormalProofs.TechnicalAxioms

/-!
# FormalProofs - Modular Formalization of Oracle Preference Training

This file re-exports the main active modules in dependency order, organized into
five main sections:
- **CLT**: Central Limit Theorem and probability theory
- **Econometrics**: available as a separate direct import (`FormalProofs.Econometrics`)
- **ML**: Supervised learning primitives
- **DSL**: Debiased/Double Machine Learning
- **OPT**: Oracle Preference Training (main results)

## Proof Status

✅ **656+ theorems/lemmas** - Core preservation / transport stack formalized
✅ **No live placeholder declarations** in the active Lean modules re-exported here

## Quick Navigation

| Module | Entry Point | What It Proves |
|--------|-------------|----------------|
| **CLT** | `FormalProofs.CLT` | Central Limit Theorem |
| **Econometrics** | `FormalProofs.Econometrics` | Separate direct import: local potential-outcomes/IPW foundations |
| **ML** | `FormalProofs.ML` | Supervised learning basics |
| **DSL** | `FormalProofs.DSL` | Debiased ML, IPW, clustered SEs |
| **OPT** | `FormalProofs.OPT.MainTheorems` | Local laws → training equivalence |

## Assumption Bundles

See `FormalProofs/Axioms.lean` for the centralized registry of model-level
assumption bundles used by some asymptotic and systems-facing modules.
-/
