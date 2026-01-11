/-!
# DSL Module: Debiased/Double Machine Learning

## Overview

This module formalizes **debiased machine learning** (DML) and related statistical methods
for causal inference and robust estimation. It provides infrastructure for:

- Inverse probability weighting (IPW)
- Cross-fitting and sample splitting
- Clustered standard errors
- M-estimation theory

## Main Results

| Theorem | File | Statement |
|---------|------|-----------|
| IPW unbiasedness | IPWTheory | E[IPW estimator] = true causal effect |
| Clustered SE validity | ClusteredVariance | Clustered SEs valid under correlation |
| Judge calibration | JudgeCalibration | Surrogate error bounds for judge models |
| Cross-fitting properties | CrossFitting | Bias reduction via sample splitting |

## File Structure

```
DSL/
├── CoreDefinitions.lean       # Basic types for DML
├── SamplingTheory.lean        # Sampling and probability foundations
├── CrossFitting.lean          # Sample splitting for bias reduction
├── MomentFunctions.lean       # Moment conditions for estimation
├── DSLEstimator.lean          # Debiased/double ML estimator
├── AsymptoticTheory.lean      # Asymptotic normality results
├── BiasAnalysis.lean          # Bias decomposition
├── VarianceDecomposition.lean # Variance estimation
├── LinearRegression.lean      # OLS as DML special case
├── LogisticRegression.lean    # Logistic regression
├── MultinomialLogistic.lean   # Multinomial logit
├── FixedEffects.lean          # Panel data models
├── InstrumentalVariables.lean # IV estimation
├── RegressionDiscontinuity.lean # RDD design
├── CategoryProportion.lean    # Categorical outcomes
├── MEstimationCore.lean       # General M-estimation
├── MainTheorems.lean          # Curated exports
├── IPWTheory.lean             # Inverse probability weighting
├── ClusteredVariance.lean     # Clustered standard errors
├── JudgeCalibration.lean      # Judge model calibration
└── TreeIPW.lean               # IPW for tree-structured data
```

## Key Concepts

### Inverse Probability Weighting (IPW)

IPW reweights samples to correct for selection bias:
```
E[Y·w(X)] = E[Y | treated]
```
where w(X) = 1/P(treated | X).

### Cross-Fitting

Split data into K folds, fit nuisance parameters on K-1 folds,
evaluate on held-out fold. Reduces bias from overfitting.

### Clustered Standard Errors

When observations are correlated within clusters:
```
Var(θ̂) = Σ_c Var(Σ_{i∈c} ψ_i)
```
rather than assuming independence.

## Connection to OPT Module

The DSL module provides statistical infrastructure used by the OPT module:
- IPW for design-based inference on tree-structured preference data
- Clustered SEs for dependent preference comparisons
- Judge calibration for surrogate reward models

## Assumptions

This module bundles assumptions as **structures** (explicit theorem parameters)
rather than Lean `axiom` declarations. This makes them explicit parameters to theorems,
which is cleaner for a formalization that aims to be modular.

| Structure | Location | Purpose | Justification |
|-----------|----------|---------|---------------|
| `OracleAccess` | CoreDefinitions | Expert labels = oracle labels | Design assumption |
| `MEstimationAxioms` | AsymptoticTheory | M-estimation asymptotics | Newey & McFadden 1994 |
| `CoverageAxioms` | AsymptoticTheory | CI coverage properties | Standard asymptotic theory |

See `FormalProofs/Axioms.lean` for detailed documentation of each assumption structure.
-/
