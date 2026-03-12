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
| `misspecified_ht_weight_expectation_eq_ratio` | IPWMeasurementError | Using the wrong propensity in HT weighting introduces the exact bias factor `π_true / π_used` |
| `proxyAware_ht_weight_expectation_eq_one_of_adjusted` | IPWMeasurementError | Proxy-aware / nonclassical sampling is harmless for HT if the IPW propensity is adjusted to the actual proxy-aware design |
| `proxyAwareIgnorable_iff_noExpectationMismatch_of_observedIgnorable` | NonclassicalExpectationMismatch | Under observed-only ignorability, proxy-aware ignorability is equivalent to there being no expectation mismatch between the coarse and rich information sets |
| `proxyAware_designAdjustedOutcome_expectation_eq_residual` | NonclassicalExpectationMismatch | If the design-adjusted outcome uses a propensity that ignores the richer proxy-aware information set, its expectation retains the exact residual term `(1 - π_true/π_used) × error` |
| `designAdjustedOutcome_expectation_eq_residual_plus_oracleMeasurementError` | NonclassicalExpectationMismatch | Full scalar decomposition: sampling-law mismatch residual plus oracle measurement error |
| `DSLMoment_expectation_eq_residual_plus_oracleMeasurementError` | NonclassicalExpectationMismatch | Full moment-level decomposition: proxy-aware sampling mismatch plus oracle measurement error |
| `treepo_gap_with_oracleMeasurement_calibration_and_estimation` | TreeIPW | Four-layer deterministic envelope `true -> oracle -> judge -> estimate`; exact-oracle is the `oracle_err = 0` special case |
| `dsl_bound_valid_with_oracleMeasurement` | TreeIPW | High-probability `computeDSLBound` validity for a true target when oracle labels may themselves have bounded measurement error |
| `clusteredConfidenceInterval_coverage_of_error_event` | ClusteredVariance | Generic clustered confidence-interval coverage follows from any event controlling the estimation error by `z × SE` |
| `judgeBiasConfidenceInterval_coverage_of_error_event` | JudgeCalibration | Judge-bias confidence-interval coverage reduces to the corresponding bias-error event |
| `abs_trueBias_le_absbiasUpperBound_of_mem_biasConfidenceInterval` | JudgeCalibration | If the true bias lies in the judge-bias confidence interval, the absolute-bias envelope is valid |
| `judgeClusteredBiasConfidenceInterval_coverage_of_error_event` | JudgeCalibration | Clustered judge-bias confidence-interval coverage reduces to the corresponding clustered bias-error event |
| `computeDSLBound_valid_from_joint_interval_event` | TreeIPW | `computeDSLBound` is valid on any event where the judge-side gap lies in its clustered confidence interval and calibration is controlled |
| `computeDSLBound_valid_from_joint_interval_event_with_oracleMeasurement` | TreeIPW | Oracle-measurement version of the joint interval/calibration validity theorem |
| `treepo_loss_gap_with_oracleMeasurement` | TreePOEndToEnd | Generic end-to-end bridge from true loss to oracle loss to tree objective; exact-oracle is the zero-error special case |
| `dpo_treepo_end_to_end_certificate_with_oracleMeasurement` | TreePOEndToEnd | DPO end-to-end TreePO certificate lifted to a possibly noisy oracle target |
| `grpo_pl_treepo_end_to_end_certificate_with_oracleMeasurement` | TreePOEndToEnd | GRPO-PL end-to-end TreePO certificate lifted to a possibly noisy oracle target |
| `grpo_pl_treepo_end_to_end_gen_certificate_with_oracleMeasurement` | TreePOEndToEnd | GRPO-PL generator-based end-to-end TreePO certificate lifted to a possibly noisy oracle target |
| `grpo_rl_treepo_end_to_end_certificate_with_oracleMeasurement` | TreePOEndToEnd | GRPO-RL end-to-end TreePO certificate lifted to a possibly noisy oracle target |
| Clustered SE validity | ClusteredVariance | Clustered SEs valid under correlation |
| Judge calibration | JudgeCalibration | Surrogate error bounds for judge models |
| Cross-fitting properties | CrossFitting | Bias reduction via sample splitting |
| TreePO end-to-end certificates | TreePOEndToEnd | HT unbiasedness + method-specific gap bounds |
| Mergeable certificate transport | MergeableCertificates | Sketch upper bounds transported to TreePO gap certificates |

## File Structure

```
DSL/
├── CoreDefinitions.lean       # Basic types for DML
├── SamplingTheory.lean        # Sampling and probability foundations
├── IPWMeasurementError.lean   # Exact bias-factor statement for misspecified IPW under proxy-aware / nonclassical sampling
├── NonclassicalExpectationMismatch.lean # Wrong-information-set expectations + exact residual bias in design-adjusted outcomes
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
├── TreeIPW.lean               # IPW for tree-structured data
├── MergeableCertificates.lean # Sketch-to-TreePO certificate transport
└── TreePOEndToEnd.lean        # End-to-end TreePO method certificates
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
| `MEstimatorConsistencyAssumption` / `MEstimatorAsymptoticNormalAssumption` | AsymptoticTheory | Decomposed M-estimation assumptions | Same theory, explicit split |
| `CoverageFromAsymptoticNormal` (`CoverageAxioms` alias) | AsymptoticTheory | CI coverage transfer | Standard asymptotic theory |
| `CalibrationRMSEBound` (`CalibrationAxioms` alias) | JudgeCalibration | Calibration RMSE bound | Representativeness of calibration set |

See `FormalProofs/Axioms.lean` for detailed documentation of each assumption structure.
-/
