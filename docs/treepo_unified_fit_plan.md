# treepo unified `fit()` plan — v1 minimal

> Supersedes the additive scope in `docs/treepo_final_package_map.md` (v5) and
> the parallel plan at `~/.claude/plans/i-ve-been-working-with-glistening-porcupine.md`.
> Both proposals enumerated good reuse but kept growing new package surfaces
> (`treepo.manifest`, `treepo.audit`, `treepo.certificate`, `treepo.folds`,
> Lean-noun dataclass sprawl, external adapters). This plan cuts all of that
> from v1 and anchors on one call signature that already runs every paper
> exercise.

> **Status update (post-cdx-comparison):** Local-law and audit math
> ported into `treepo_cld` as both an in-loop training signal and a
> post-hoc audit hook. `corrected_local_law_loss`,
> `local_law_objective_summary` (corrected + IPW modes, gamma-depth
> discount), and `compute_influence_weighted_overlap` (observed-rows
> only — fixes cdx's propensity-floor bias) are exported from the
> package root. `LearnableConstantFamily` demonstrates an in-loop
> training step driven by the IPW objective; verified unbiased under
> confounded sampling. `fit()` accepts
> `backend_config["law_audit_rows"]` and attaches the audit summary to
> `result.summary["audit"]`. **43 tests pass.** Honest scope on FNO /
> DSPy / TRL: those families still consume objective knobs via their
> own configs; routing them through `treepo_cld.corrected_local_law_loss`
> is a per-family refactor, not a v1 wrap.
>
> **Status (2026-05-19):** Phases 1-6 implemented in
> [`treepo_cld/`](../treepo_cld/) as a parallel workspace. **29 tests pass**,
> including 6 real-synthetic-data end-to-end checks (oracle MAE=0,
> classical HLL within bounded error, leaf-local-mixture LDA against
> closed-form truth). Manifest sidecar written to
> `output_dir/treepo_cld_run_manifest.json`; `ObjectiveSpec` accepted on
> `backend_config['objective']` and recorded in the manifest. **Honest
> scope:** v1 covers `oracle` + classical sketch (`hll` / `count_min`)
> paths end-to-end; `fno`/`dspy`/`trl` are *registered* but their full
> paper exercises need servers/configs/migrations called out in
> [`../treepo_cld/README.md`](../treepo_cld/README.md). Custom training
> loops (e.g. `scripts/probe_clean_unified_no.py`, the LDA recovery
> baselines) are explicit Phase 7+ migrations, not v1 wraps. Kill list
> locked. Merge into [`treepo/`](../treepo/) is a mechanical move of
> ~8 files.

## 1. The anchor

```python
from treepo import fit
from src.ctreepo.contracts import CTreePOLearningSpec, CTreePOFitResult

result: CTreePOFitResult = fit(spec: CTreePOLearningSpec)
```

Both dataclasses already exist:

- `CTreePOLearningSpec` — [src/ctreepo/contracts.py:2152](src/ctreepo/contracts.py#L2152). Fields: `space_kind`, `family`, `schedule`, `initial_artifacts`, `train_data`, `eval_data`, `backend_config`, `axis`.
- `CTreePOFitResult` — [src/ctreepo/contracts.py:2198](src/ctreepo/contracts.py#L2198). Fields: `status`, `metrics`, `artifacts`, `history`, `summary`, `manifest_path`.

The implementation of `fit()` is a thin orchestrator over
[`run_alternating_family`](src/ctreepo/alternating.py#L1201), which already
defines the full kwargs surface (`family`, `f_init`, `g_init`, `traces`,
`eval_trees`, `max_iterations`, `output_dir`, `axis_kind`, `leaf_count`,
`leaf_size_tokens`, `first_train_side`, `initial_f_degree`,
`initial_g_degree`, `stage_naming`, `artifact_namer`). Backends dispatch
through the existing
[`FamilyRuntime`](src/ctreepo/alternating.py#L250) protocol
(`train_f` / `train_g` / `score_roots_with_f` / `validate_artifact`) and the
[`BundleAwareFamilyRuntime`](src/ctreepo/alternating.py#L309) extension
(`expected_bundle()`, `supported_inits()`, `resolve_init`,
`share_state_axes`).

There is no second signature. There is no `treepo.audit.fit`. There is no
`treepo.certificate.fit`. There is one `fit()`.

## 2. The six paper exercises through one signature

| # | Exercise | `spec.family` | f | g | Laws | Anchor script |
|---|---|---|---|---|---|---|
| 1 | Manifesto LLM distillation | `"dspy"` / `"trl"` | LLM scorer (Gemma) | LLM summarizer | C1+C3 | [run_manifesto_fg_real_training_grid.py](scripts/run_manifesto_fg_real_training_grid.py) |
| 2 | HLL sketch + learned merge | `"hll"` | register estimator | pointwise-max merge | C1+C3 | [run_hll_sampled_node_rate_grid.py](scripts/run_hll_sampled_node_rate_grid.py) |
| 3 | FNO on Markov change-point | `"fno"` | FNO forward map | FNO encoder | C1+C3 | [run_markov_fno_round2_campaign.py](scripts/run_markov_fno_round2_campaign.py) |
| 4 | LDA topic-mixture tree | `"oracle_lda"` | quadratic topic utility | topic posterior | C1+C3 | [run_lda_tree_recovery_simulation.py](scripts/run_lda_tree_recovery_simulation.py) |
| 5 | Count-Min Sketch | `"count_min"` | point-freq query | additive merge | C1+C3 | subset of [run_tree_root_only_parity_diagnosis.py](scripts/run_tree_root_only_parity_diagnosis.py) |
| 6 | Classical parity baseline | `"oracle"` | oracle readout | identity | C1 only | [report_tree_root_only_parity_pdf.py](scripts/report_tree_root_only_parity_pdf.py) |

Local laws ride **inside** `family.train_f` / `family.train_g`, parameterized
by `spec.backend_config["objective"]` (an existing
[`ObjectiveSpec`](src/ctreepo/contracts.py#L1273) — `objective_family`,
`local_law_estimator`, `local_law_weight`, `root_share`,
`local_law_component_weights`, `terms`). No separate law surface lives
above `fit()`.

## 3. Kill list — what v1 does NOT build

These appeared in proposal 1 and/or proposal 2 and are dropped from v1:

- `UnifiedLearningErrorCertificate` / `InfluenceWeightedErrorCertificate` builders
- `InfluenceWeightedAuditOverlap` (`D_λ` / `W_λ` overlap object)
- `ManifestValidationReport`, `validate_manifest_roles_consistent`, `validate_propensity_floor`, `validate_artifact_lineage`
- Schema-anchor parity test parsing `docs/unified_learning_theorem_map.md`
- Leak-detection fixture as a standalone surface (the prototypes in `src/training/` already exercise this)
- Verification matrix runner; `treepo/src/treepo/release.py` stays minimal
- R `dsl::dsl()` subprocess adapter, `dsl_kit` adapter, EconML adapter, GRF parity fixture, ranger reference, DataSketches optional extras
- `state_shape_contract()` / `supported_supervisions()` declarations on backends — the existing `BundleAwareFamilyRuntime.expected_bundle()` / `supported_inits()` already cover this
- Paper-table reconstruction module
- Phase 0 "reuse inventory lock" / `migration_inventory.yaml` ceremony
- `treepo.audit`, `treepo.certificate`, `treepo.folds`, `treepo.manifest` sub-packages
- Lean-noun dataclasses (`TopLevelUnit`, `DerivedRow`, `Span`, `ChunkPartition`, `RoleTuple`, `ManifestRow`, `RunManifestContract`) — `ArtifactRef` is the only one v1 needs and it already exists at [src/ctreepo/contracts.py:2091](src/ctreepo/contracts.py#L2091)

## 4. Trigger rules for re-introducing deferred items

Each deferred item is added back **only** when a concrete event fires:

| Deferred item | Trigger to add it back |
|---|---|
| Lean-noun dataclass | A paper section *consumes* it in a table or claim (not merely *mentions* it) |
| Manifest validator | An actual experiment fails because of a malformed manifest in prod |
| `D_λ` / `W_λ` audit overlap | A paper table reports the number |
| External adapter (GRF/EconML/dsl/dsl_kit) | A paper claim cites its output as evidence |
| Certificate builder | Lean ↔ Python parity test for a specific theorem requires it |
| Verification matrix runner | Release gate blocks on a regression that the runner would catch |
| State-shape contract declaration | A second consumer beyond `BundleAwareFamilyRuntime` reads it |

If none of the 6 exercises trips a trigger, the item never enters the package.

## 5. Promote, do not rewrite

Everything below is imported directly. No wrappers, no shims, no parallel surfaces:

| What | From | Used by |
|---|---|---|
| `run_alternating_family` | [src/ctreepo/alternating.py:1201](src/ctreepo/alternating.py#L1201) | `treepo.fit` body |
| `FamilyRuntime`, `BundleAwareFamilyRuntime` | [src/ctreepo/alternating.py:250-380](src/ctreepo/alternating.py#L250) | backend protocol |
| `CTreePOLearningSpec`, `CTreePOFitResult`, `CTreePOProgramSpec`, `ArtifactRef`, `ObjectiveSpec`, `TreeBundleManifest`, `RunManifest` | [src/ctreepo/contracts.py](src/ctreepo/contracts.py) | fit() I/O |
| `FNOFamily` | [src/ctreepo/fno_family.py:170](src/ctreepo/fno_family.py#L170) | exercise 3 |
| `DSPyFamily` | [src/ctreepo/dspy_family.py:374](src/ctreepo/dspy_family.py#L374) | exercise 1 |
| `TRLFamily` | [src/ctreepo/trl_family.py:96](src/ctreepo/trl_family.py#L96) | exercise 1 |
| `OracleFamilyRuntime` | [src/ctreepo/oracles/runtime.py:27](src/ctreepo/oracles/runtime.py#L27) | exercises 4, 6 |
| `corrected_local_law_loss`, `local_law_objective_mean`, `aggregate_local_law_observations` | [src/core/local_law_adjustment.py](src/core/local_law_adjustment.py) | inside families |
| `corrected_local_law_loss_tensor`, `local_law_objective_target_mse` | [src/training/supervision/local_law_torch.py](src/training/supervision/local_law_torch.py) | inside FNO family |
| `treepo_reduce`, `SketchAdapter` | [treepo/src/treepo/sketches/](treepo/src/treepo/sketches/) | sketch FamilyRuntime |
| `HyperLogLogSketch` | [treepo/src/treepo/hll.py](treepo/src/treepo/hll.py) | exercise 2 |
| three-layer honesty split helpers | `src/training/run_pipeline.py` (`assign_three_layer_split`, `assign_three_layer_roles`, `ThreeLayerHonestyConfig` ~lines 2806–2846) | inside `fit()` when honesty is requested |

## 6. Genuinely new code for v1

| File | What | LOC est. |
|---|---|---|
| [treepo/src/treepo/learning.py](treepo/src/treepo/learning.py) | `fit(spec)` orchestrator: extract kwargs from spec, call `run_alternating_family`, package records into `CTreePOFitResult` | ~150 |
| [treepo/src/treepo/families.py](treepo/src/treepo/families.py) | `resolve_family(name, backend_config)` registry: maps `"fno"`/`"dspy"`/`"trl"`/`"oracle"`/`"oracle_lda"`/`"hll"`/`"count_min"` to `FamilyRuntime` instance; uses `_optional.py` guards | ~120 |
| [treepo/src/treepo/sketches/family.py](treepo/src/treepo/sketches/family.py) | `ClassicalSketchFamilyRuntime`: wraps any `SketchAdapter` to satisfy `FamilyRuntime` (train_f/g no-op, `score_roots_with_f` invokes `treepo_reduce`) | ~80 |
| [treepo/src/treepo/tests/test_fit_exercises.py](treepo/src/treepo/tests/test_fit_exercises.py) | one tiny end-to-end test per exercise (asserts `status == "success"`, checks shape of `metrics`) | ~200 |

**Total new code: ~550 LOC.** Everything new is on the call path between
`fit(spec)` and `CTreePOFitResult`. Delete any of the 4 files and one
paper exercise stops working. Nothing is defensive.

## 7. Phased implementation

### Phase 1 — Wire the fit() shell
- Write `treepo/src/treepo/learning.py::fit(spec)`. Call `run_alternating_family` with kwargs derived from `spec.axis` (`max_iterations`, `axis_kind`, `leaf_count`, `leaf_size_tokens`, `stage_naming`) and `spec.backend_config` (`first_train_side`, `initial_f_degree`, `initial_g_degree`).
- Re-export `fit` from `treepo/src/treepo/__init__.py`.
- **DoD:** `from treepo import fit; result = fit(spec)` runs and returns `CTreePOFitResult(status="success", ...)` for a no-op `OracleFamilyRuntime` with one stage. One test covers this.

### Phase 2 — Family registry + FNO exercise
- Add `treepo/src/treepo/families.py::resolve_family(name, backend_config)`.
- Register `"fno"`, `"dspy"`, `"trl"`, `"oracle"` with optional-import guards.
- **DoD:** exercise 3 (`spec.family="fno"`) runs end-to-end on a tiny Markov fixture (small `t`, small `train_docs`) and reports `metrics["root_mae"]`.

### Phase 3 — Sketch families through fit()
- Add `treepo/src/treepo/sketches/family.py::ClassicalSketchFamilyRuntime`.
- Register `"hll"`, `"count_min"`, `"oracle_hll"` in the registry.
- **DoD:** exercises 2, 5, 6 each run through `fit()` on tiny fixtures (~10 leaves).

### Phase 4 — LLM exercises through fit()
- DSPy and TRL families already implement `FamilyRuntime`; only the registry entry is new.
- **DoD:** exercise 1 (manifesto) runs through `fit()` with a 3-doc / 1-dimension fixture using the `teacher` backend (no real LLM call).

### Phase 5 — LDA exercise
- Reuse `OracleFamilyRuntime` if it covers the LDA case; otherwise add a thin `LDAFamilyRuntime` that follows the same shape.
- **DoD:** exercise 4 runs through `fit()` on a 1024-doc synthetic LDA bundle.

### Phase 6 — Stop. Lock the kill list.
- Add a one-paragraph note to `treepo/README.md` referencing this plan and the trigger rules.
- No new code. Re-evaluate every proposed addition against §3 (kill list) and §4 (triggers).

## 8. Reconciliation with prior proposals

| Item from prior proposals | Disposition |
|---|---|
| Promote `local_law` from `treepo/src/treepo/training/local_law.py` + parity siblings | Adopted (imported, not rewrapped) |
| Promote `FamilyRuntime` / `BundleAwareFamilyRuntime` | Adopted (it's the dispatch surface) |
| Promote `ObjectiveSpec` v1 multiplicative form | Adopted (lives in `spec.backend_config["objective"]`) |
| Promote sketch adapters | Adopted (wrapped once by `ClassicalSketchFamilyRuntime`) |
| Promote three-layer honesty + `assign_three_layer_split` | Adopted as-is; called from inside `fit()` only when `spec.backend_config["honesty"]` is set |
| Promote k-fold orchestration | Deferred until an exercise needs CV inside a single `fit()` call (manifesto sweep handles CV at the script level today) |
| Promote `release.py` release gate | Keep as-is in `treepo/`; no extension in v1 |
| New `treepo.manifest`/`treepo.audit`/`treepo.certificate`/`treepo.folds` packages | Dropped |
| New 9 Lean-noun dataclasses | Dropped (8 of 9); `ArtifactRef` already exists |
| New `InfluenceWeightedAuditOverlap` | Dropped (trigger: paper table) |
| New manifest validators (3) | Dropped (trigger: prod failure) |
| New schema-anchor parity test | Dropped (trigger: Lean ↔ Python claim) |
| New paper-table reconstruction | Dropped (trigger: paper section consumes it) |
| New `state_shape_contract()` / `supported_supervisions()` on backends | Dropped (already covered by `BundleAwareFamilyRuntime`) |
| New external adapters (GRF / EconML / R dsl / dsl_kit / DataSketches / ranger) | All dropped (trigger: paper claim cites adapter output) |
| Phase 0 "reuse inventory lock" / `migration_inventory.yaml` | Dropped |
| New `treepo.fit()` shell | Adopted (this plan's anchor) |
| `f*` artifact reification edits in `alternating.py` | Deferred unless a downstream exercise reads `f*` artifacts; current backends already serialize artifacts |

## 9. What "minimal and non-defensive" means here, concretely

After Phase 5:

- The public surface of `treepo` for fit-style work is: `fit`, `CTreePOLearningSpec`, `CTreePOFitResult`, `ArtifactRef`, `ObjectiveSpec`, the family-name strings, and the `FamilyRuntime` protocol re-export. That is the entire API.
- There is no second way to do the same thing.
- There is no validator, certifier, or auditor sitting between the caller and the loop.
- There is no optional-extra dead-code path.
- The 6 paper exercises run.

If something in §3 ever does need to come back, it will be because a real
experiment or paper table demanded it — and at that point we'll know the
exact shape it needs to take, which we don't today.
