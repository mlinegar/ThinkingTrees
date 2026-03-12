# Theorem-Backed Reduction as a Measurement-Error Problem

This note connects the theorem-backed reduction framework in `lean3/FormalProofs/OPT`
to the measurement-error literature.

## Main Correspondence

Our setup has three layers:

1. A document `x` with latent state `feature x`.
2. A reduction pipeline producing a summary distribution `ZR g x R T`.
3. A possibly noisy observation `featureHat x` of the target state.

The exact theorem-backed regime says the reduction itself introduces no oracle
distortion on support. The approximate regime says the reduction introduces a
controlled amount of oracle distortion. The additional question is whether the
oracle is rich enough to identify, or at least stably control, the latent state.

That is precisely a measurement-error question.

## Exact Regime

The exact bridge is formalized in:

- `TheoremBackingMeasurementError.lean`
- `OracleRecoversFeature`
- `expected_feature_utility_with_measurement_error_via_ZR_of_exactTheoremBacked_and_oracleRecoversFeature`

Interpretation:

- `OracleRecoversFeature` is an identification assumption.
- If oracle distortion is zero, then the latent state is exactly recovered.
- The only remaining utility gap is the discrepancy between the noisy proxy
  `featureHat x` and the true latent state `feature x`.

This is the zero-measurement-error special case for the reduction step plus an
ordinary measurement-error term for the final proxy.

## Approximate Regime

The approximate bridge is formalized in:

- `TheoremBackingApproxMeasurementError.lean`
- `FeatureLipschitzFromOracle`
- `feature_distortion_le_of_featureLipschitzFromOracle`
- `expected_feature_utility_with_measurement_error_via_ZR_of_approxTheoremBacked_and_featureLipschitz`

Interpretation:

- the reduction pipeline creates an upstream distortion budget in oracle space;
- `FeatureLipschitzFromOracle` converts that oracle-space distortion into latent
  state distortion; and
- the final bound separates into:
  - a transport term from approximate reduction error; and
  - a pure measurement-error term from `featureHat`.

So the approximate theorem-backed regime is a structured measurement-error model
with two sources of discrepancy:

1. reduction-induced representation error; and
2. observation error in the target latent state.

## Relation to Standard Measurement-Error Models

The literature usually distinguishes:

- classical error: the observed measurement is the truth plus noise;
- Berkson error: the truth varies around an assigned or proxy measurement; and
- nonclassical error: the error can depend on the truth or on other variables.

Our `featureHat` term is agnostic about the mechanism. The Lean theorem only
needs a metric discrepancy `dist (featureHat x) (feature x)` and Lipschitz
control of utility in that argument. This is deliberately broader than the
standard parametric classical-error model.

`OracleRecoversFeature` is closest to a validation/gold-standard regime: the
oracle is informative enough to identify the latent target exactly. The
approximate `FeatureLipschitzFromOracle` route is closer to a calibrated
surrogate setting, where the oracle does not exactly identify the latent state
but does control it quantitatively.

## Relation to Correction Methods

Regression calibration and SIMEX are procedures for correcting bias once a
measurement-error model is specified or estimated from validation / replicate
data. Our theorems live one layer above that:

- they do not estimate the measurement model;
- they state what conclusions follow once the relevant assumptions are supplied;
- they isolate the reduction error from the final measurement-error term.

In that sense, theorem-backedness provides a structural decomposition:

`total utility gap <= reduction transport gap + measurement error gap`

and the measurement-error literature provides concrete ways to estimate or bound
the second term in applications.

## Practical Reading

For our purposes, the literature suggests the following interpretation.

- If you have a gold-standard or validation-style oracle, target the exact
  theorem-backed route plus `OracleRecoversFeature`.
- If you only have a calibrated surrogate oracle, target the approximate route
  plus `FeatureLipschitzFromOracle`.
- If the proxy noise is modeled from replicate or validation data, the Lean
  theorems tell you exactly how that estimated noise budget enters the overall
  reduction guarantee.

## Canonical References

- Wayne A. Fuller, `Measurement Error Models` (1987): the classical foundational
  reference for errors-in-variables models.
- Raymond J. Carroll, David Ruppert, Leonard A. Stefanski, and Ciprian M.
  Crainiceanu, `Measurement Error in Nonlinear Models: A Modern Perspective`
  (2nd ed., 2006): the broad modern reference spanning functional versus
  structural views, generalized models, and semiparametric settings.
- John Bound, Charles Brown, and Nancy Mathiowetz, `Measurement Error in Survey
  Data` (2001): emphasizes that real measurement error is often nonclassical and
  that validation data can be used to assess or bound resulting bias.
- STRATOS guidance on measurement error and misclassification (2020): practical
  review of regression calibration, validation/calibration studies, and SIMEX.

These references align with the current formalization as follows:

- `OracleRecoversFeature` corresponds to an ideal validation / gold-standard
  identification condition.
- `FeatureLipschitzFromOracle` corresponds to a calibrated surrogate condition.
- `dist (featureHat x) (feature x)` is the application-specific measurement
  error term that can be estimated or bounded using the standard tools above.
