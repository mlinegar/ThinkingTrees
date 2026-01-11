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
-- Econometrics Module: Foundations for IPW
-- ============================================================================
import FormalProofs.Econometrics

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

-- Layer 5: Applications
import FormalProofs.OPT.ScoreTransport
import FormalProofs.OPT.PreferenceLearning
import FormalProofs.OPT.PreferenceBounds
import FormalProofs.OPT.TrainingPipeline
import FormalProofs.OPT.CounterexampleExistence
import FormalProofs.OPT.AuditBounds

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

-- ============================================================================
-- Documentation
-- ============================================================================
import FormalProofs.Assumptions
import FormalProofs.TechnicalAxioms

/-!
# FormalProofs - Modular Formalization of Oracle Preference Training

This file re-exports all modules in dependency order, organized into five main sections:
- **CLT**: Central Limit Theorem and probability theory
- **Econometrics**: Potential outcomes + IPW foundations
- **ML**: Supervised learning primitives
- **DSL**: Debiased/Double Machine Learning
- **OPT**: Oracle Preference Training (main results)

## Proof Status

✅ **656+ theorems/lemmas** - All proved (no sorry)
⚠️ **1 axiom** - Modeling assumption: Random Utility Model (McFadden 1974)

## Quick Navigation

| Module | Entry Point | What It Proves |
|--------|-------------|----------------|
| **CLT** | `FormalProofs.CLT` | Central Limit Theorem |
| **Econometrics** | `FormalProofs.Econometrics` | Potential outcomes + IPW foundations |
| **ML** | `FormalProofs.ML` | Supervised learning basics |
| **DSL** | `FormalProofs.DSL` | Debiased ML, IPW, clustered SEs |
| **OPT** | `FormalProofs.OPT.MainTheorems` | Local laws → training equivalence |

## Axioms

See `FormalProofs/Axioms.lean` for the centralized axiom registry with full justification.
-/
