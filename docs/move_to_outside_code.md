# Move To Outside Code

Date: 2026-05-19

This note is an implementation roadmap for reusing mature outside code wherever
that is statistically honest and engineering-safe. The intended stance is
reference-first reuse:

- keep the Lean/Python TreePO contracts canonical;
- use outside packages as reference implementations, optional adapters, and
  parity checks;
- never let package-native objects silently replace TreePO estimands,
  manifests, local-law rows, or final certificates.

Primary references:

- GRF honesty: `grf` `causal_forest` and algorithm docs:
  <https://grf-labs.github.io/grf/reference/causal_forest.html> and
  <https://grf-labs.github.io/grf/REFERENCE.html>.
- Design-based supervised learning: R package `dsl`:
  <https://naokiegami.com/dsl/>.
- Optional secondary Python DSL candidate: `dsl-kit`:
  <https://pypi.org/project/dsl-kit/>.

## Executive Position

Use outside code to reduce bespoke statistical implementation where possible,
but only behind adapters that preserve our theorem-facing contracts. External
packages can supply estimates, standard errors, model diagnostics, and parity
checks. They cannot define the unit of analysis or change which rows enter the
TreePO certificate.

The canonical objects remain local:

- top-level case/document unit `X_i`;
- derived spans, nodes, and local-law audit rows;
- C1/C2/C3 local-law semantics;
- logged propensities and influence weights;
- `RunManifestContract`;
- `UnifiedLearningComponentEvidence`;
- `UnifiedLearningErrorCertificate`;
- final Lean theorem surface:
  `DSL.unified_learning_final_paper_certificate` and
  `DSL.unified_learning_final_paper_certificate_high_prob`.

This is the same reason honest causal forests do not remove the need to state
the estimand. GRF can teach and validate split/estimation separation, but our
chunker/g/oracle roles and local-law audit rows are TreePO-specific.

## Component Matrix

| Local component | Outside candidate | Reuse mode | What remains local | Required parity check |
|---|---|---|---|---|
| Top-level honesty split | GRF honesty docs and simulations | Reference model and vocabulary for split-building vs estimation samples | `TopLevelIID`, `TopLevelExchangeable`, `UnifiedLearningHonesty`, role tuple `(r_C, r_G, r_O)` | Verify artifacts trained on split-building units cannot consume eval labels, residuals, or law failures |
| K-fold / cross-fit artifacts | GRF out-of-bag and honest-subsample conventions | Reference design for fold-specific frozen artifacts | Fold assignment, artifact IDs, parent unit IDs, and final eval set `E_C intersect E_G intersect E_O` | Round-trip fold IDs and artifact lineage through the manifest |
| Downstream DSL estimator | R package `dsl` | Optional reference estimator for top-level corrected estimates and standard errors | Export schema, TreePO local-law rows, manifest, and final certificate | Compare local DSL/IPW estimate and SE against R `dsl` on a tiny fixed dataset |
| Python DSL path | `dsl-kit` | Secondary reference only after R `dsl` parity is understood | Same as above | Compare against both local path and R `dsl`; do not make core depend on it |
| Local-law sampled objective | None as full replacement | Keep local implementation; external code may validate generic IPW arithmetic | `treepo/src/treepo/training/local_law.py`, C1/C2/C3 row construction, influence weights | Local master objective matches manifest rows and known synthetic truth |
| IPW / HT sampling arithmetic | R `dsl`, survey-style estimators, existing Lean IPW layer | Reference estimates and variance diagnostics where the estimand matches | Propensity definition, node weights, hidden-needle overlap, effective propensity floor | Compare HT/IPW estimates, ESS, and max-weight diagnostics |
| Classical mergeable summaries | `datasketches` and similar libraries | Runtime implementation and parity baseline | Lean sketch contracts, oracle readout, local-law mapping | External sketch output satisfies the same C1/C2/C3 or approximate-law certificate |
| Learned `f/g` and nuisance models | scikit-learn, torch, GRF-style forests where useful | Optional model backends behind frozen artifact IDs | Training/eval roles, artifact lineage, calibration and estimation evidence | Model outputs map into `UnifiedLearningComponentEvidence` |
| Final certificate | No outside replacement | Outside outputs may fill component radii | `UnifiedLearningErrorCertificate` and Lean theorem surface | External radii populate certificate fields without bypassing the manifest |

## Must-Have Contracts Before Reuse

Any external package bridge must accept or produce normalized data that satisfies
these contracts.

1. Top-level unit identity
   - Stable `top_level_unit_id`.
   - Declared top-level unit type: document, section, panel, sequence, or other
     estimand-level object.
   - Truth target `Y_i*` or documented approximation/calibration source.

2. Derived-row identity
   - Stable `row_id`.
   - Parent top-level unit ID.
   - Row kind: C1 leaf, C2 idempotence, C3 merge, top-level loss, calibration
     row, or estimator row.
   - Support span or node/pair IDs when applicable.

3. Honest role and fold lineage
   - `fold_id` and `split_seed`.
   - Role tuple `(r_C, r_G, r_O)`.
   - Artifact IDs for chunker, `g`, `f`, online oracle, eval oracle, query
     policy, and proxy.
   - A guarantee that eval-unit labels, residuals, and observed law failures
     did not update artifacts used to report that same unit.

4. Sampling and overlap
   - Observed indicator.
   - Logged propensity in `(0, 1]`.
   - Effective propensity after any exploration floor.
   - Influence weight.
   - Effective sample size and max influence-to-propensity diagnostics.

5. Certificate provenance
   - Local-law radius and its source, preferably
     `InfluenceWeightedErrorCertificate`.
   - Calibration radius and event/provenance source.
   - Statistical estimation radius and event/provenance source.
   - Clipping or floor radius, if any.
   - Failure-probability split across components.

6. Version and reproducibility
   - Package name, version, language/runtime, and seed.
   - Input export hash and output import hash.
   - Adapter version.
   - Exact estimator function and arguments used.

## External Reuse Targets

### GRF For Honesty

Use GRF as the outside reference for honesty language and split discipline.
Its useful concepts are:

- split-building sample versus estimation sample;
- `honesty = TRUE`;
- `honesty.fraction`;
- pruning/skipping behavior when an estimation leaf is empty;
- out-of-bag or subsample discipline for predictions and variance.

Do not directly outsource TreePO honesty to GRF. GRF trees split rows for a
forest estimand; TreePO splits top-level units across chunker, `g`, and oracle
roles. The adapter should therefore be a parity harness:

- construct synthetic top-level units;
- assign honest split roles;
- train a GRF-style reference model only on split-building units;
- compute predictions on held-out estimation units;
- assert that changing held-out labels/residuals does not change trained
  split-building artifacts.

Acceptance criteria:

- a deterministic fixture catches deliberate eval-label leakage;
- fold and artifact IDs round-trip through the manifest;
- the test explains any GRF-specific behavior that has no TreePO analogue.

### R `dsl` For Design-Based Estimation

Use R `dsl` as the primary outside reference for downstream design-based
estimation with predicted variables. The bridge should be optional and should
operate at the top-level unit table, not the local-law row table.

Minimum adapter contract:

- export a CSV/Parquet table with top-level IDs, predictions, truth labels or
  sampled labels, treatment/covariate columns if used, sample indicators, and
  inclusion probabilities;
- run an R script that calls `dsl` with explicit arguments;
- import normalized estimates, standard errors, confidence intervals, and
  package metadata;
- map the statistical radius into `UnifiedLearningComponentEvidence.estimation`
  or a task-specific calibration/estimation component.

Acceptance criteria:

- a tiny synthetic dataset has matching point estimates between local DSL/IPW
  and R `dsl`;
- standard errors agree within a documented tolerance when both packages are
  computing the same estimand;
- mismatched estimands fail loudly rather than producing a certificate.

### Optional Python DSL Reference

Treat `dsl-kit` as secondary until its API and estimands are checked against the
R package and our Lean contracts. It may become useful for CI where R is not
available, but it should not define the canonical estimator.

Acceptance criteria:

- documented estimator equivalence or documented difference from R `dsl`;
- no mandatory dependency in `pyproject.toml`;
- tests skip cleanly when the optional package is absent.

### Classical Sketch Libraries

Use libraries such as `datasketches` for classical mergeable summary runtime
implementations when they match our sketch contracts. They are good candidates
for HLL, quantiles, heavy hitters, and related classical sketches.

What remains local:

- Lean sketch theorem assumptions;
- mapping from sketch state to oracle/readout;
- C1/C2/C3 audit row construction;
- proof that the external sketch output satisfies the relevant exact or
  approximate local-law interface.

Acceptance criteria:

- external sketch output matches a local reference on deterministic fixtures;
- merge associativity/compatibility tests pass;
- certificate construction is identical to the local path after the sketch
  output is normalized.

### General ML And Statistical Packages

Use scikit-learn, torch, GRF-style forests, or other ML packages only as model
backends behind frozen artifact IDs. They can produce predictions, nuisance
models, uncertainty proxies, or calibration curves. They must not decide the
theorem-facing estimand.

Acceptance criteria:

- every fitted object has artifact lineage;
- training data is restricted by the declared role;
- predictions on evaluation rows are reproducible from frozen artifacts;
- outputs map into the same component-radius evidence fields as local models.

## Do Not Outsource

These pieces must remain TreePO-owned:

- definition of the top-level statistical unit;
- derived-row parent mapping;
- C1/C2/C3 row construction and paper numbering;
- local-law candidate generation from chunked trees;
- influence weights and hidden-needle overlap diagnostics;
- manifest validity;
- final Lean certificate;
- decision of whether rows are top-level estimator rows or local-law audit rows;
- distinction between training diagnostics and honest evaluation metrics.

An external package result is certificate-eligible only after it has been
normalized into the local manifest and evidence objects.

## Recommended Adapter Design

Add optional adapters later under a clearly named external-reference area, for
example:

```text
src/external_reference/
  grf_honesty/
  dsl_r/
  sketches/
```

Do not add mandatory R dependencies to core packages. Prefer one of:

- script boundary: Python exports files, R script runs, Python imports JSON;
- `rpy2` optional extra for local interactive use;
- containerized reference run for CI or paper artifact reproduction.

Each adapter should expose the same normalized shape:

```text
input_manifest_hash
external_package
external_package_version
adapter_version
estimator_name
estimator_arguments
point_estimate
standard_error_or_radius
confidence_level_or_delta
component_target
row_count
top_level_unit_count
diagnostics
```

The adapter output should be rejected unless:

- all input rows have stable top-level IDs;
- all sampled rows have positive propensities;
- artifact lineage is complete;
- component target is one of local-law, calibration, estimation, or clipping;
- the output can populate `UnifiedLearningComponentEvidence`.

## Required Parity Checks

1. GRF-style honesty parity
   - Construct a small dataset with known train/eval roles.
   - Fit a split-building reference model using only train-role units.
   - Mutate held-out labels/residuals and verify the trained artifact is
     unchanged.
   - Verify predictions and reported metrics use only frozen artifacts.

2. DSL estimator parity
   - Build a tiny top-level table with truth labels, model predictions, sampled
     labels, inclusion indicators, and propensities.
   - Compute local corrected/IPW estimate.
   - Export to R `dsl` and compare point estimate and SE.
   - Fail if the adapter cannot prove the same estimand is being computed.

3. Sampling parity
   - Compare logged propensities against realized sampling design.
   - Verify HT/IPW estimates on known finite populations.
   - Report ESS, max weight, and max influence-to-propensity ratio.
   - Include zero-observed-row and high-weight diagnostics.

4. Manifest parity
   - Every external call round-trips:
     `top_level_unit_id`, `fold_id`, `row_id`, role tuple, artifact IDs,
     propensity, effective propensity, influence weight, support span, and law
     kind.
   - Missing or altered fields fail before estimates are accepted.

5. Certificate parity
   - External outputs populate `UnifiedLearningComponentEvidence`.
   - Final report constructs `UnifiedLearningErrorCertificate`.
   - No adapter is allowed to emit a final certificate directly.

## Implementation Sequence

1. Documentation and schema alignment
   - Keep this file and `docs/unified_learning_theorem_map.md` in sync.
   - Decide the external adapter output JSON schema.
   - Add examples of valid and invalid manifest rows.

2. DSL reference bridge
   - Implement export/import scripts for a tiny top-level dataset.
   - Add an optional R script using package `dsl`.
   - Add skipped tests when R or `dsl` is unavailable.

3. Honesty parity harness
   - Implement synthetic train/eval leakage fixtures.
   - Add GRF-backed parity only as an optional reference path.
   - Keep the local honesty test independent of GRF availability.

4. Classical sketch adapter checks
   - Add deterministic sketch fixtures where external libraries already exist.
   - Normalize outputs into local sketch and local-law contracts.

5. Certificate integration
   - Add a fixture showing external estimation/calibration radii populate
     `UnifiedLearningComponentEvidence`.
   - Verify `DSL.unified_learning_final_paper_certificate_high_prob` is the
     theorem cited by the report.

## Acceptance Criteria

The move toward outside code is complete enough for paper use when:

- every external result can be traced to a manifest row set and adapter version;
- every external estimate has a declared TreePO component target;
- local tests catch eval-label leakage;
- local and external DSL estimates agree on a fixed small example;
- any mismatch in estimands produces a hard failure;
- the final paper table can be reconstructed as a
  `UnifiedLearningErrorCertificate`;
- no core training path requires R or optional external reference packages.

## Open Risks

- GRF honesty is conceptually aligned but not structurally identical to
  three-role TreePO honesty.
- R `dsl` may expose estimators whose estimand or variance convention differs
  from our local DSL/IPW path.
- Some external sketch libraries may merge states correctly but lack the
  task-specific readout needed for oracle preservation.
- Optional adapters can create reproducibility drift unless package versions,
  seeds, and input hashes are logged.

The mitigation is to keep all outside code behind adapters that emit normalized
component evidence and never bypass the local theorem-facing certificate.
