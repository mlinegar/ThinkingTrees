# Canonical Local-Law / F-G Consolidation Master Plan

Date: 2026-06-25

Status: superseded comparison draft. The consensus plan is now
`docs/local_law_single_path_master_plan.md`; keep this file only as an audit
trail of the Codex draft used during collaboration.

Original status: Codex combined draft for collaboration. This file merges the plan in
`docs/local_law_single_path_master_plan.md` with the earlier dated Codex plan,
the current living plan in `docs/local_law_single_path_plan.md`, the review in
`docs/local_law_canonical_path_review_2026-06-25.md`, and the sampling contract
in `docs/local_law_sampling_contract.md`.

This draft intentionally leaves `docs/local_law_single_path_master_plan.md`
untouched so another LLM can compare both versions and produce a further
reconciliation.

## Executive Stance

The goal is one canonical local-law/objective path shared by every method and
model in ThinkingTrees: FNO/NO, embedding models, Markov/sim runners, DSPy/LLM
families, TRL scaffolds, and future families.

Canonical math lives in `/home/mlinegar/treepo`. ThinkingTrees keeps only
repo-specific adapters that build rows from trees, traces, labels, and model
outputs.

First deliverable: correctness and unification, not new metrics. Pause
Benoit/FNO publication sweeps until the corrected, unified objective path is
active across the relevant FNO/NO surfaces.

The plans merge cleanly because:

- The non-dated plan contributes the compact executable phase list, the explicit
  "FNO A2 is already corrected" status, and the re-attribution warning for the
  old `a2state` result.
- The dated Codex plan contributes a fuller API consolidation path, source
  inventory phase, row-adapter details, acceptance matrix, documentation phase,
  and more explicit open decisions.
- Both plans agree on the DSPy/GEPA constraint: GEPA/MIPRO score examples, not
  whole-tree batches, so DSPy training keeps a shared per-example helper while
  canonical AIPW/local-law aggregation runs as offline eval/audit.

## Target End State

There is one way to express and consume local-law supervision:

```text
repo-specific row adapters
  -> treepo.training.local_law for training objectives
  -> treepo.local_law for strict audit/certificate rows
  -> treepo.objective.ObjectiveSpec for root/local resolution
  -> FamilyRuntime training adapters
  -> InferenceEngine execution surfaces where model calls are needed
```

Canonical objective:

```text
J = (1 - Lambda) * root_loss + Lambda * local_law_loss

local_law_loss =
  mean_v node_weight_v * gamma^depth(v)
    * [proxy_loss_v + observed_v / propensity_v * (oracle_loss_v - proxy_loss_v)]
```

Conventions:

- `depth(root) = 0`.
- `gamma in [0, 1]`.
- `Lambda = local_law_weight`; `root_share = 1 - Lambda`.
- Role weights become row `node_weight`s or component weights resolved by
  `ObjectiveSpec`; they do not create private objective formulas.
- A2 / merge preservation means the merge-route readout agrees with an
  independent reading of the actual parent text, not with the already-merged
  parent state.
- A3 / readout factorization and g-associativity remain separate diagnostics or
  projections. They are not evidence for A2.

## Canonical Ownership

| Concern | Canonical home | Everything else becomes |
|---|---|---|
| Training arithmetic: AIPW, depth discount, node weights, root/local split | `/home/mlinegar/treepo/src/treepo/training/local_law.py` | Delegating shim or caller |
| Training row type that permits proxy-only unobserved rows | New `treepo.training.local_law.LocalLawTrainingRow` | Replacement for ad hoc objective rows |
| Strict audit/certificate rows | `/home/mlinegar/treepo/src/treepo/local_law.py::LocalLawAuditRow` | Caller only |
| Public objective contract | `/home/mlinegar/treepo/src/treepo/objective.py::ObjectiveSpec` | ThinkingTrees compatibility facade |
| Root/local and component resolver | Prefer upstream `treepo.objective`; temporary `src/ctreepo/objective_resolution.py` allowed during migration | `CompositeObjectiveSpec` thin sim adapter |
| Repo-specific row construction | New `src/ctreepo/local_law_rows.py` | Family callers build rows here |
| f/g training interface | `src/ctreepo/alternating.py::FamilyRuntime` | FNO, DSPy, TRL, oracle adapters |
| Execution and inference | `src/core/inference_engine.py::InferenceEngine` | No duplicate execution surface |

## Row Contract

Every family emits the same logical row fields:

```text
prediction
proxy_target | proxy_loss
oracle_target | oracle_loss
observed
propensity
depth
node_weight
law_kind
global_axiom
state_kind
law_channel
doc_id
node_id
metadata
```

Rules:

- Family adapters build rows only. They do not implement AIPW denominators,
  depth discounting, or root/local mixing.
- Training rows and audit rows are separate. Training rows may include
  `observed=False, propensity=0` proxy-only rows when no division occurs. Audit
  rows with nonzero influence require positive logged propensities.
- Uniform all-node supervision has one logical row per node. A full binary tree
  with `L` leaves produces `2L - 1` rows.
- The root is counted exactly once. If a trace has an explicit root row and a
  cumulative merge row whose final row is also the root, drop the cumulative
  duplicate for all-node objectives.
- Fixed-size uniform sampling logs `propensity = q / N`; Bernoulli sampling logs
  the Bernoulli rate.
- No per-document "at least one node" adjustment unless the estimand explicitly
  says so.
- Rate-grid supervision uses persistent sampled masks across epochs.

## Current State To Preserve

- [x] FNO A2 self-comparison has already been corrected per
  `docs/local_law_single_path_plan.md` and
  `docs/local_law_canonical_path_review_2026-06-25.md`.
- [x] Corrected A2 compares `f(parent text)` to
  `f(merge(child states))`.
- [x] FNO f/g paths are at least partly routed through
  `treepo.training.local_law.local_law_objective_from_losses`.
- [x] `gamma_depth` convention is root depth 0.
- [x] `g_a2_weight` is deprecated as a no-op; A2 share is governed by
  `local_law_weight`.
- [x] `a3_factorization_weight` is separate from A2.
- [x] `docs/local_law_sampling_contract.md` establishes the sampling/IPW
  contract.
- [x] `src/core/inference_engine.py` is already the central execution surface.
- [ ] Re-attribute any prior "a2state win" as an associativity-penalty result,
  not A2 evidence.
- [ ] `LocalLawTrainingRow` does not yet exist upstream in `treepo`.
- [ ] `src/training/supervision/local_law_torch.py` is still an implementation
  copy.
- [ ] `src/core/local_law_adjustment.py` still owns scalar adjustment behavior
  and diagnostics.
- [ ] `src/ctreepo/contracts.py::ObjectiveSpec` still duplicates the upstream
  objective contract.
- [ ] `src/ctreepo/sim/composite_objective.py` still owns root/local resolution.
- [ ] `src/ctreepo/local_law_rows.py` does not yet centralize row construction.
- [ ] DSPy and TRL are not fully on the canonical row/objective path.

## Disposition Policy

Use quarantine first. Delete only after import scans and parity tests prove the
replacement.

### Keep Canonical

- `/home/mlinegar/treepo/src/treepo/training/local_law.py`
- `/home/mlinegar/treepo/src/treepo/local_law.py`
- `/home/mlinegar/treepo/src/treepo/objective.py`
- `src/ctreepo/alternating.py::FamilyRuntime`
- `src/core/inference_engine.py`
- New `src/ctreepo/local_law_rows.py`
- Temporary `src/ctreepo/objective_resolution.py` only if upstream resolver
  migration cannot land immediately.

### Quarantine

- `g_a2_weight`, marked deprecated/no-op.
- Old q-sentence local-law rewards, marked `diagnostic_only`.
- Embedding-FNO role-weighted MSE, retained as `legacy_weighted_mse` diagnostic.
- Scaffolded TRL/SFT paths until true current-f reward/GRPO exists.
- A3 factorization and g-associativity diagnostics, kept separate from A2.

### OLD Archive

Archive only after zero-live-import scans:

- `treepo/src/treepo/_research/ctreepo/*`
- `treepo_cdx/`
- Pure re-export shims such as `src/tree/neural_operator.py`
- Duplicated local-law/objective implementation bodies after shim parity.

Never touch unrelated dirty worktree state, existing `OLD_*` trees,
paper/report script churn, or broad Markov configs outside migration gates.

## Phase 0 - Source Inventory And Guardrails

Goal: make the migration measurable before behavior changes.

- [ ] Inventory all objective specs, local-law objective functions,
  root/local mixers, row builders, reward helpers, and archive mirrors.
- [ ] Record live import graph for:
  `src/training/supervision/local_law_torch.py`,
  `src/core/local_law_adjustment.py`,
  `src/ctreepo/sim/composite_objective.py`,
  `src/ctreepo/contracts.py::ObjectiveSpec`,
  `treepo._research`,
  `treepo_cdx`.
- [ ] Add allowlist-based source guard rejecting new bespoke AIPW,
  propensity-denominator, depth-discount, or root/local mixer code outside
  canonical homes and approved shims/adapters.
- [ ] Add archive-import guard that fails on imports from officially archived
  `OLD_*`, `treepo_cdx`, or `_research` paths.
- [ ] Baseline focused tests before wiring changes:
  `tests/training/test_local_law_torch.py`,
  `tests/core/test_local_law_adjustment.py`,
  `tests/ctreepo/test_objective_weights.py`,
  `tests/ctreepo/test_composite_objective.py`,
  `tests/ctreepo/test_neural_operator_baselines.py`,
  FNO A2 tests, and upstream `treepo/tests/training/test_local_law.py`.

Gate:

- [ ] Inventory note exists.
- [ ] Guard tests pass with current duplicates explicitly allowlisted.

## Phase 1 - Canonical Foundation

Goal: make upstream `treepo` a superset of live ThinkingTrees requirements, and
add the opt-in row/resolver adapters without changing family behavior.

Upstream `treepo`:

- [ ] Add `LocalLawTrainingRow` in `treepo.training.local_law`.
- [ ] Add scalar training aggregate helpers mirroring useful behavior from
  `src/core/local_law_adjustment.py`, excluding ThinkingTrees-only diagnostics.
- [ ] Keep `LocalLawAuditRow` strict in `treepo.local_law`.
- [ ] Move live-only helpers upstream, including
  `corrected_local_law_target_mse` if callers still require it.
- [ ] Extend `treepo.objective.ObjectiveSpec` with live ThinkingTrees estimator
  modes, including `oracle_state` and `external_passthrough`.
- [ ] Preserve strict convex validation by default.
- [ ] Add or promote `resolve_root_local_objective_weights(...)` in the
  upstream objective API.
- [ ] Accept legacy ThinkingTrees objective payloads on read; emit canonical
  `treepo` schema on new writes.

ThinkingTrees additive adapters:

- [ ] Add `src/ctreepo/objective_resolution.py` only if the resolver cannot be
  consumed directly from upstream yet.
- [ ] Add `src/ctreepo/local_law_rows.py` with `build_local_law_rows(...)` and
  `classify_node_role(...)`.
- [ ] Keep `CompositeObjectiveSpec` as a thin internal sim adapter over the
  canonical resolver.
- [ ] Do not wire families yet.

Tests:

- [ ] Training rows accept unobserved proxy-only `propensity=0`.
- [ ] Audit rows reject invalid observed/nonzero-influence propensities.
- [ ] Scalar and tensor helper parity against current ThinkingTrees behavior.
- [ ] `gamma_depth > 1` rejected.
- [ ] Node weights and zero-sample cases covered.
- [ ] ObjectiveSpec legacy payload load and canonical payload round-trip.
- [ ] New row adapter emits `2L - 1` rows, root once, correct roles, correct
  `q / N`, persistent masks, and law metadata.

Gate:

- [ ] Upstream `treepo` tests pass.
- [ ] ThinkingTrees focused tests still pass before rewiring.

## Phase 2 - Collapse Duplicate Arithmetic And Contracts

Goal: replace implementation copies with compatibility shims.

- [ ] Convert `src/training/supervision/local_law_torch.py` into a thin shim
  over `treepo.training.local_law`.
- [ ] Keep public names/signatures where feasible:
  `corrected_local_law_target_mse`,
  `local_law_objective_from_losses`, and test-pinned symbols.
- [ ] Convert `src/core/local_law_adjustment.py` scalar loss,
  normalization helpers, and constants to delegate to `treepo`.
- [ ] Keep only ThinkingTrees-specific diagnostics/dataclasses in
  `src/core/local_law_adjustment.py`.
- [ ] Convert `src/ctreepo/contracts.py::ObjectiveSpec` to a re-export facade
  or compatibility wrapper over `treepo.objective.ObjectiveSpec`.
- [ ] Accept legacy `ctreepo.objective.v1` payloads; emit canonical upstream
  schema on new writes.
- [ ] Route `src/ctreepo/sim/composite_objective.py` through the canonical
  resolver.

Tests:

- [ ] Shim parity proves bit-identical or tolerance-identical scalar/tensor
  returns on deterministic fixtures.
- [ ] Legacy objective payload migration tests.
- [ ] Composite objective root/local resolution tests.
- [ ] Existing downstream imports still work.

Gate:

- [ ] No behavior deltas except explicitly documented serialization
  normalization.
- [ ] Focused local-law/objective suite green.

## Phase 3 - FNO And Neural Operator Migration

Goal: route all FNO/NO surfaces through canonical rows/objective while
preserving the corrected A2 behavior.

`src/ctreepo/fno_family.py`:

- [x] Preserve corrected A2: `f(parent text)` versus `f(merge(child states))`.
- [x] Keep A3 factorization and associativity separate from A2.
- [ ] Replace remaining private root/law split helper with canonical resolver.
- [ ] Emit rows through `src/ctreepo/local_law_rows.py`.
- [ ] Keep current behavior behind parity fixtures until row path is proven.

`src/ctreepo/embedding_fno.py`:

- [ ] Replace `_batch_loss` role-weighted MSE with central rows plus
  `treepo.training.local_law`.
- [ ] Retain old weighted-MSE path as `legacy_weighted_mse` diagnostic only.
- [ ] Map existing config knobs to `ObjectiveSpec`, component weights, or row
  `node_weight`s.

`src/ctreepo/sim/core/markov_neural_operator_baselines.py`:

- [ ] Replace in-repo `local_law_torch` imports with `treepo.training.local_law`.
- [ ] Replace or shim `_fno_single_lambda_objective_loss` with canonical
  resolver plus canonical objective helpers.
- [ ] Preserve current tensor materialization path, tree-model-version behavior,
  and residual-decomposition behavior.
- [ ] Preserve hot-path performance rule: no per-node `.cpu()` or `.item()` in
  long merge chains.

Other wrappers:

- [ ] Fold or archive redundant `EmbeddingFNOModelConfig`, `FNOFamilyConfig`,
  and `ManifestoRileEmbeddingObjective` wrappers after import scans and parity.

Tests:

- [ ] Deterministic-seed loss parity across live `tree_model_version` variants
  and `use_residual_decomposition` modes.
- [ ] Existing FNO A2 non-vacuity tests remain green.
- [ ] New `embedding_fno._batch_loss` canonical-row test.
- [ ] Markov FNO import/objective parity.
- [ ] Performance smoke catches hot-path sync regressions.

Gate:

- [ ] Canonical path is default for FNO/NO.
- [ ] Old path is diagnostic-only, shimmed, or archived.

## Phase 4 - DSPy, LLM, And TRL Migration

Goal: LLM methods emit canonical rows and share reward helpers without claiming
GEPA sees whole-tree AIPW objectives.

DSPy:

- [ ] DSPy record builders emit canonical row metadata through
  `src/ctreepo/local_law_rows.py`.
- [ ] Collapse base scalar reward, q-sentence vector reward, and inline
  distillation reward copies into one shared per-example helper.
- [ ] Shared helper uses the same component names and objective metadata as
  `ObjectiveSpec`, but stays per-example for GEPA/MIPRO compatibility.
- [ ] Mark q-sentence C1/C3a/C3b reward variants `diagnostic_only` when they
  are not corrected/IPW paper-facing objectives.
- [ ] Run canonical AIPW/local-law aggregation as offline eval/audit on emitted
  DSPy rows.
- [ ] Route distillation builders through central rows instead of bespoke
  q-sentence inline builders.

TRL:

- [ ] Keep scaffolded teacher-SFT paths quarantined until true alternating
  current-f reward/GRPO exists.
- [ ] Define a GRPO reward function over canonical row outputs.
- [ ] Add a smoke test that LoRA weights change and row-based rewards are logged.

Tests:

- [ ] DSPy row emission tests.
- [ ] DSPy per-example reward parity tests.
- [ ] Distillation/q-sentence builder parity tests.
- [ ] Offline AIPW audit smoke on a tiny tree set.
- [ ] TRL reward smoke once live trainer exists.

Gate:

- [ ] DSPy training behavior preserved where GEPA constraints require it.
- [ ] Offline canonical audit available for DSPy outputs.

## Phase 5 - Public API Consolidation

Goal: the four public surfaces are coherent and not duplicated.

`CTreePOLearningSpec`:

- [ ] Decide whether canonical ownership remains in
  `src/ctreepo/contracts.py` or moves upstream to `treepo`.
- [ ] If moved upstream, keep ThinkingTrees as a compatibility import plus
  legacy payload loader.
- [ ] Ensure `fit(...)` and `preflight(...)` accept canonical specs and legacy
  mappings.

`ObjectiveSpec`:

- [ ] Canonical owner is `treepo.objective.ObjectiveSpec`.
- [ ] ThinkingTrees imports delegate to or wrap upstream.
- [ ] New writes use upstream schema.

`FamilyRuntime`:

- [ ] Keep one protocol boundary for FNO, DSPy, TRL, oracle, classical, and
  learnable-constant families.
- [ ] Keep bundle-aware extensions additive and optional.

`InferenceEngine`:

- [x] Keep `src/core/inference_engine.py` as the execution surface.
- [ ] Remove or archive duplicate execution interfaces only after live import
  scans.
- [ ] Ensure model families call execution through adapters rather than
  reimplementing transport.

Tests:

- [ ] Public import tests.
- [ ] Legacy payload migration tests.
- [ ] Fit/dispatch smoke tests.
- [ ] Runtime/inference engine smoke tests.

Gate:

- [ ] New code has one documented import path per public surface.

## Phase 6 - Archive And Delete Bodies

Goal: remove duplicate bodies only after tests prove they are inactive.

- [ ] Import scan proves `treepo._research.ctreepo` has no live callers.
- [ ] Archive `_research/ctreepo` with `OLD_` prefix or agreed archive header.
- [ ] Import scan proves `treepo_cdx` has no live callers.
- [ ] Archive `treepo_cdx`.
- [ ] Import scan proves pure re-export neural-operator wrappers are unused.
- [ ] Archive or delete those wrappers by repo convention.
- [ ] Delete duplicate local-law/objective bodies only after shims and parity
  tests have landed.
- [ ] Tighten source guards so archived imports fail tests.

Gate:

- [ ] No import from archived mirrors.
- [ ] Full focused suite green.

## Phase 7 - Acceptance Test Matrix

No publication-scale sweeps are required for this refactor. Acceptance is
deterministic unit, parity, source-guard, and tiny smoke coverage.

Upstream `treepo`:

- [ ] `LocalLawTrainingRow` tests.
- [ ] `LocalLawAuditRow` strict-propensity tests.
- [ ] Tensor/scalar objective parity tests.
- [ ] `ObjectiveSpec` validation and legacy payload tests.

ThinkingTrees:

- [ ] `tests/training/test_local_law_torch.py`
- [ ] `tests/core/test_local_law_adjustment.py`
- [ ] `tests/ctreepo/test_objective_weights.py`
- [ ] `tests/ctreepo/test_composite_objective.py`
- [ ] New `tests/ctreepo/test_local_law_rows.py`
- [ ] New source-guard tests.

FNO/NO:

- [ ] Existing FNO A2 consistency tests.
- [ ] `tests/test_fno_null_space_law.py`
- [ ] `tests/test_fno_extent_latent.py`
- [ ] `tests/test_fno_merge_can_learn_average.py`
- [ ] `tests/ctreepo/test_neural_operator_baselines.py`
- [ ] New embedding-FNO canonical objective parity test.

DSPy/LLM:

- [ ] DSPy record-weight and objective-summary tests.
- [ ] Q-sentence/distillation row-emission tests.
- [ ] Offline AIPW audit smoke.

Runtime/API:

- [ ] `CTreePOLearningSpec` compatibility tests.
- [ ] `FamilyRuntime` dispatch tests.
- [ ] `InferenceEngine` smoke tests.

Tiny smokes:

- [ ] Markov FNO objective run.
- [ ] Manifesto/FNO canonical-row run.
- [ ] DSPy offline-audit run using emitted rows.

## Phase 8 - Documentation And Handoff

- [ ] Update `docs/local_law_sampling_contract.md` if `LocalLawTrainingRow`
  changes row semantics.
- [ ] Update `docs/local_law_single_path_plan.md` with completed phase markers
  or replace it with the final master plan after collaboration.
- [ ] Update `docs/ctreepo_python_code_map_for_llms.md` only after re-running a
  source inventory, AST parse sweep, and targeted path searches.
- [ ] Document archived paths and replacement imports.
- [ ] Add a short migration guide for new family authors: build rows, call the
  canonical objective, never reimplement AIPW.

## Open Decisions

- [ ] DSPy: confirm constrained unification, or split per-tree batch training
  into a separate project. Default: constrained unification.
- [ ] Sync: Phase 1 changes land in real `/home/mlinegar/treepo`; Phase 6
  archives the dead `_research` mirror. Default: archive the mirror, do not
  re-sync it.
- [ ] Resolver: canonical-only in `treepo.objective`, or temporary
  `src/ctreepo/objective_resolution.py` during migration?
- [ ] `CTreePOLearningSpec`: upstream owner or ThinkingTrees public facade?
- [ ] Corrected A2 parent-text read: descendant-leaf embedding pooling by
  default, or re-embed concatenated parent text for root/interiors?
- [ ] Root A2 row: excluded from `local_law_loss` because root has the separate
  `(1 - Lambda)` term, or included with explicit no-double-counting metadata?
- [ ] f/g objective split: one shared `Lambda`, or phase-specific shares?
- [ ] Undefined `delta` knob: define it or drop it. Candidate meanings are role
  ratio, sampling-rate floor, or proxy/oracle blend, but no canonical definition
  exists.
- [ ] TRL true alternating: part of this cleanup or separate follow-up?

## Implementation Checklist

- [x] Read attached plans.
- [x] Read `docs/local_law_single_path_master_plan.md`.
- [x] Read current sampling contract, living plan, and canonical review.
- [x] Produce this combined Codex draft.
- [ ] Phase 0: source inventory and guards.
- [ ] Phase 1: upstream foundation plus opt-in row/resolver adapters.
- [ ] Phase 2: compatibility shims and contract collapse.
- [ ] Phase 3: FNO/NO migration.
- [ ] Phase 4: DSPy/LLM/TRL migration.
- [ ] Phase 5: public API consolidation.
- [ ] Phase 6: archive/dead-code pass.
- [ ] Phase 7: acceptance tests.
- [ ] Phase 8: docs and handoff updates.

## Recommended First Coding Pass

1. Run Phase 0 source inventory and create guard allowlists.
2. Add upstream `LocalLawTrainingRow`, scalar helper parity, and objective
   resolver.
3. Shim `local_law_torch.py`, `local_law_adjustment.py`, and `ObjectiveSpec`.
4. Add central row adapter tests with no family rewiring.
5. Migrate FNO/NO first because corrected A2 and partial canonical routing are
   already present.
6. Migrate DSPy/LLM row emission and offline audit.
7. Archive only after import guards are active and parity tests are green.

This order keeps behavior-preserving foundation work ahead of family rewrites
and avoids deleting anything before the canonical path covers live behavior.
