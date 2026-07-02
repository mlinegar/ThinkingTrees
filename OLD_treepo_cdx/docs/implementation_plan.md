# treepo_cdx Implementation Plan

This is the working plan for the parallel package lane. The principle is to
promote working code only when the package boundary is already clear, and to add
new abstractions only where the existing repo has repeated glue.

## Stable Package Spine

Status: implemented.

- Public import must stay dependency-light.
- Contract nouns live in `manifest.py`, `objective.py`, `audit.py`,
  `certificate.py`, `sampling.py`, `honesty.py`, and `folds.py`.
- JSON round trips and deterministic digests are part of the public contract.
- `release.audit_release()` is the local gate for static heavy imports and
  public import side effects.

## Single Fit Facade

Status: implemented for the existing paper/runtime/learning shapes.

`treepo_cdx.fit()` accepts one config object and dispatches to the working
monorepo lanes:

- paper exercises: `treepo.bench.runner.run_single`
- runtime exercises: `treepo.runtime.run_runtime_eval`
- f/g learning specs: `src.ctreepo.learning.fit`
- local-law audit fixtures: package-native corrected objective path

Every fit writes `fit_result.json`. Runtime fits also write
`run_manifest.json`, because runtime predictions already expose enough row
shape to build a theorem-facing manifest.

Next:

- Add manifest sidecars for paper exercises where the runner emits row-level
  observations.
- Preserve upstream learning manifests when `src.ctreepo.learning.fit` returns
  one, and add a validator smoke test for that path once a small fixture exists.

## Local-Law Objective Lane

Status: first dep-light implementation is in place.

- `local_law.py` exposes the corrected DR/IPW objective and sampled IPW
  fallback.
- `adapters.py` turns manifest rows and simple mappings into
  `LocalLawAuditRow` values.
- Existing canonical implementations in `treepo/src/treepo/training` and
  `src/core` remain the numerical reference until parity fixtures are added.

Next:

- Add a parity fixture against `treepo.src.treepo.training.local_law` for the
  two-row DR/IPW case and a small sampled tree case.
- Add row builders for trace tables, sketch states, and f/g local-law outputs as
  those surfaces are promoted.

## Honesty And Folds

Status: deterministic package-native helpers are in place.

- `honesty.py` mirrors the three-role vocabulary.
- `folds.py` gives deterministic hash folds and disjoint train/eval views.

Next:

- Add a source parity test against `src.training.run_pipeline` three-layer role
  assignment once the exact seed/hash contract is locked.
- Decide whether the promoted helper should preserve the old SHA256 role cutoffs
  byte-for-byte or standardize on the smaller `treepo_cdx` contract.

## Backends

Status: capability contract implemented; native HLL wrapper implemented.

The minimal package should not copy backend internals. Backends enter through a
small protocol:

- `state_shape_contract()`
- `supported_supervisions()`
- `fit` or `run` entry point
- artifact references that can be recorded in manifests

`backends.py` now contains the dep-light capability contract:
`StateShapeContract`, `SupervisionSpec`, and `backend_capabilities()`.

The native HLL wrapper in `sketches.py` is the first backend adapter. It lazily
uses the existing `treepo.hll` implementation and exposes the same capability
metadata as other runtimes.

Next:

- Promote the broader classical-sketch comparison runner behind the same
  capability surface.
- Add adapters for LLM/DSPy/FNO/TRL only after the manifest output shape is
  stable for the smaller sketch lane.

## Certificates And Release Gates

Status: first primitives are in place.

- `certificate.py` keeps component evidence explicit and rejects hidden scalar
  collapse.
- `release.py` is intentionally small and local.

Next:

- Build certificates from actual local-law objective summaries and manifest
  validation reports.
- Add a verification matrix command that runs unit tests, release audit, and
  selected parity fixtures.

## Defer

- TRL alternation and GRPO-specific training loops.
- DSPy per-node auxiliary channels.
- DataSketches as a default dependency.
- External R/Python parity subprocesses until the package-native row schema is
  stable.
