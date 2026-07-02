# Axiom De-Formalization Manifest

Status as of 2026-03-22: closed for the information-sufficiency surface, with
the main implementation-facing DSL bridge pushed further from abstract
interfaces toward executable certificate inputs.

The earlier axiomatized Shannon-entropy stub in
`FormalProofs/OPT/OracleEntropy.lean` has been removed. The active Lean-backed
surface now consists of:

- `FormalProofs/OPT/InformationSufficiency.lean`
  Oracle sufficiency, a.e. factorization, score transport, and zero
  task-relevant KLIC.
- `FormalProofs/OPT/OracleEntropy.lean`
  A proved finite-support log-cardinality envelope with no `axiom`
  declarations.
- `FormalProofs/OPT/OracleSufficientCompression.lean`
  Deterministic oracle-sufficiency and collision/impossibility statements with
  no `sorry`.

What changed:

- Full Shannon / mutual-information formalization is no longer represented as a
  partially axiomatized module.
- The optional information-theory context has been narrowed to statements we can
  prove directly in Lean today.
- The fixed-ranker Plackett-Luce GRPO-PL path now exposes first-principles
  public certificates, so that route no longer requires the abstract
  `ExpectedGRPOLossLipschitz` interface at the API boundary.
- The GRPO-RL finite-support lane now also exposes a first-principles route:
  a primitive pointwise Lipschitz bound on `GRPORLLossPointwise` discharges the
  abstract `ExpectedGRPORLLossLipschitz` interface and propagates through the
  quantitative gap / TreePO certificate surface.
- The DSL coverage layer now has a concrete one-dimensional route in
  `FormalProofs/DSL/ConcreteCoverage.lean`: cdf convergence to the standard
  normal law plus an explicit coverage-event equivalence implies asymptotic
  coverage, and the OLS Wald theorem is now a corollary of that generic lemma.
- The DSL coverage layer also now has a concrete multivariate coordinatewise
  route in `FormalProofs/DSL/ConcreteCoverage.lean`: weak convergence of the
  full studentized statistic vector implies coordinatewise interval coverage
  after projection to each coordinate via the continuous mapping theorem and
  Portmanteau on boundary-null intervals.
- `FormalProofs/DSL/AsymptoticTheory.lean` and
  `FormalProofs/DSL/MainTheorems.lean` now expose that route on the main DSL
  surface via `DSL_valid_coverage_coordStdNormal*`,
  `DSL_valid_inference_coordStdNormal*`, and
  `DSL_CI_coverage_coordStdNormal*`. The remaining normalization input there is
  just positive diagonal variance together with an event identity for the
  diagonally studentized statistic; the standard-normal coordinate marginals
  are derived from the `NormalLimit` witness itself. The `*_symm` wrappers
  package the common symmetric `[-z, z]` critical-value case.
- `FormalProofs/DSL/AsymptoticTheory.lean` and
  `FormalProofs/DSL/MainTheorems.lean` now also expose a plug-in diagonal Wald
  route via `DSL_valid_coverage_pluginStdNormal*`,
  `DSL_valid_inference_pluginStdNormal*`, and
  `DSL_CI_coverage_pluginStdNormal*`: implementations can studentize by a
  diagonal covariance estimator `V̂ₙ` and discharge validity from asymptotic
  normality, covariance consistency, positive limiting diagonal variance, and
  explicit plug-in interval-event identities.
- `FormalProofs/DSL/AsymptoticCore.lean` and
  `FormalProofs/DSL/ConcreteCoverage.lean` now also expose a generic
  constructive coverage interface:
  `CoverageEventWitness`, `CoordinateCoverageLimitWitness`, and
  `NormalCoverageConstruction`. These are threaded into the DSL theorem surface
  by `DSL_valid_coverage_from_construction*` and
  `DSL_valid_inference_from_construction*`.
- `FormalProofs/DSL/JudgeCalibration.lean` now discharges the calibration RMSE
  interface from held-out evidence: population-RMSE events or true-bias
  confidence-interval events imply `CalibrationRMSEBound`, and those events can
  be pushed directly into the PMF surrogate-gap validity surface.
- `FormalProofs/DSL/TreeIPW.lean` now includes stopped-time wrappers: a
  scheduled family of fixed-horizon IPW bad-event bounds yields an anytime-valid
  bound at an arbitrary stopping rule by a direct union-bound argument.
- `FormalProofs/DSL/RuntimeCertificates.lean` now packages the existing
  `computeDSLBound` and approximate-local-law theorem surfaces as soundness
  theorems for checked runtime artifacts, giving the implementation a concrete
  certificate object format rather than only theorem-level statements.
- The manuscript-facing bridge is therefore aligned with the formal surface:
  task-relevant preservation, not general source coding.

If a later phase adds genuine Shannon entropy, conditional entropy, or mutual
information, it should land as a new fully proved finite-support module rather
than by reintroducing axioms.

What still remains assumption-driven:

- Fully generic DSL coverage transfer for arbitrary confidence-interval
  constructions when no explicit `NormalCoverageConstruction` witness is
  supplied, and the broader M-estimation large-sample layer, which are
  represented as regularity/limit structures rather than Lean axioms.
- Plug-in covariance estimation itself is still taken as an input convergence
  fact (`V̂ₙ →p V`) rather than derived from a specific estimator family.
- Held-out calibration validity still starts from whichever finite-sample event
  controls the true bias or population RMSE; the theorem surface now transports
  those events, but does not yet derive them from a concrete concentration
  inequality for a chosen sampling design.
- Sequential audit validity is currently a scheduled-union wrapper around
  fixed-horizon events, not yet a fully sharpened martingale/confidence-sequence
  development.
- Runtime certificates currently check stored equalities and theorem premises;
  they do not yet connect to a serialized on-disk format or an external
  checker implementation.
- GRPO-RL beyond the finite-support pointwise route: a full first-principles
  derivation of the primitive pointwise bound for the clipped surrogate itself
  still depends on additional modeling detail.
