-- Core definitions and types
import FormalProofs.DSL.CoreDefinitions

-- Sampling theory and Assumption 1
import FormalProofs.DSL.SamplingTheory
import FormalProofs.DSL.IPWMeasurementError

-- Cross-fitting
import FormalProofs.DSL.CrossFitting
import FormalProofs.DSL.Honesty

-- Moment functions for GLMs
import FormalProofs.DSL.MomentFunctions

-- Core DSL estimator
import FormalProofs.DSL.DSLEstimator

-- Asymptotic theory
import FormalProofs.DSL.AsymptoticCore
import FormalProofs.DSL.AsymptoticTheory
import FormalProofs.DSL.ConcreteCoverage

-- Bias analysis
import FormalProofs.DSL.BiasAnalysis
import FormalProofs.DSL.NonclassicalExpectationMismatch

-- Variance decomposition and power
import FormalProofs.DSL.VarianceDecomposition

-- Specific models
import FormalProofs.DSL.LinearRegression
import FormalProofs.DSL.LogisticRegression
import FormalProofs.DSL.MultinomialLogistic
import FormalProofs.DSL.InstrumentalVariables
import FormalProofs.DSL.RegressionDiscontinuity
import FormalProofs.DSL.CategoryProportion
import FormalProofs.DSL.FixedEffects

-- Main theorems summary
import FormalProofs.DSL.RuntimeCertificates
import FormalProofs.DSL.MergeableCertificates
import FormalProofs.DSL.MainTheorems
import FormalProofs.DSL.TreePOEndToEnd
import FormalProofs.DSL.LabelRateBounds

/-!
# FormalProofs/DSL.lean

## Design-based Supervised Learning (DSL)

This is the root import file for the DSL formalization.

### Paper Reference

Egami, Hinck, Stewart, Wei - "Design-based Supervised Learning"

### Overview

DSL is a framework for using LLM/ML predicted variables in downstream statistical
analyses without bias from prediction errors. The key insight is that prediction
errors are **non-random** - they correlate with variables in the analysis. Even
with 90%+ accuracy, ignoring these errors leads to substantial bias and invalid
confidence intervals.

### Solution

DSL combines:
1. **Automated annotations** (N documents) - predictions for all data
2. **Expert annotations** (n sampled) - ground truth for a subset
3. **Doubly robust estimation** - corrects for prediction errors

Key assumptions:
1. Researcher controls sampling process (Assumption 1: Design-Based Sampling)
2. Oracle access for sampled documents (expert labels are ground truth)

### Module Structure

| Module | Description |
|--------|-------------|
| CoreDefinitions | Basic types, documents, oracle access |
| SamplingTheory | Design-based sampling, Assumption 1 |
| MomentFunctions | GLM moment functions |
| DSLEstimator | Core estimator, design-adjusted outcomes |
| AsymptoticCore | Shared asymptotic-limit and coverage definitions |
| AsymptoticTheory | Consistency, normality, variance |
| BiasAnalysis | Bias from ignoring prediction errors |
| VarianceDecomposition | Power analysis, efficiency |
| LinearRegression | DSL for OLS |
| LogisticRegression | DSL for logistic models |
| CategoryProportion | DSL for estimating proportions |
| FixedEffects | DSL for panel data |
| MainTheorems | Summary of main results |

### Key Formulas

**Design-Adjusted Outcome (Equation 4)**:
  Ỹ = Ŷ - (R/π)(Ŷ - Y)

**Key Property**:
  E[Ỹ - Y | X] = 0 (unbiasedness regardless of prediction accuracy)

**DSL Linear Regression (Equation 6)**:
  β̂_DSL = (X'X)⁻¹X'Ỹ

### Main Theorems

1. **Valid Inference**: DSL provides valid inference regardless of prediction accuracy
2. **Efficiency**: Better predictions → smaller standard errors
3. **Naive is Invalid**: Ignoring errors invalidates inference (even with 90%+ accuracy)
-/
