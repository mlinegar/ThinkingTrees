# Canonical Local-Law / F-G Consolidation — Consensus Master Plan — 2026-06-25

**Single agreed plan.** Best-of-both synthesis of the two source plans and both
LLM master-plan drafts, reconciled with the correctness record in
`docs/local_law_single_path_plan.md`, `docs/local_law_sampling_contract.md`, and
`docs/local_law_canonical_path_review_2026-06-25.md`. Both reconciling LLMs concur
this is the canonical combined version; the dated draft
`docs/local_law_single_path_master_plan_2026-06-25.md` is retained only as a
superseded comparison draft. Convergence transcript: `docs/COLLABORATION.md`.

One canonical local-law/objective path for **every** method and model: FNO/NO,
embedding models, Markov/sim runners, DSPy/LLM families, TRL. Canonical math
lives in `/home/mlinegar/treepo`; ThinkingTrees keeps only repo-specific adapters
that build rows from its trees, traces, and model outputs.

**First deliverable is correctness + unification, NOT new metrics.** Pause
Benoit/FNO publication sweeps until the corrected, unified objective is active.

## Why the plans merged cleanly

- **Plan A** contributed the training-row vs audit-row two-layer split, named
  central modules + a static-guard test banning new bespoke arithmetic, and
  widening the target to the four public interfaces + the TRL family.
- **Plan B** contributed quarantine-first disposition, legacy serialized-payload
  acceptance for `ObjectiveSpec`, and crisp per-family routing.
- Both agree on the **GEPA per-example constraint**: GEPA/MIPRO score per example
  and never see whole trees, so the canonical AIPW (a per-tree aggregate) cannot
  be the DSPy *training* reward. Training reward stays a shared per-example
  helper; canonical AIPW runs as offline eval/audit on DSPy outputs.

## Target end state — the one "single way"

```text
rows from repo-specific adapters
  -> treepo training objective / audit objective
  -> ObjectiveSpec root-local resolution
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

Conventions: `depth(root)=0`; `gamma in [0,1]`; `Lambda = local_law_weight`,
`root_share = 1 - Lambda`; role/component weights become row `node_weight`s or
spec-resolved component weights (never private formulas); A2/merge-preservation =
the merge-route readout agrees with an **independent reading of the actual parent
text**, not with the already-merged parent state.

| Concern | Canonical home | Everything else becomes |
|---|---|---|
| Tensor/scalar training arithmetic, AIPW, node weights, depth discount | `treepo/src/treepo/training/local_law.py` | delegating shim / caller |
| Strict certificate/audit rows, positive-propensity math | `treepo/src/treepo/local_law.py` (`LocalLawAuditRow`) | caller only |
| Training rows allowing proxy-only unobserved rows | new `treepo.training.local_law.LocalLawTrainingRow` | replaces ad hoc observation dataclasses as objective input |
| Root/local + component-weight resolution | `treepo.objective` resolver (`resolve_root_local_objective_weights`) | `src/ctreepo/sim/composite_objective.py` → thin sim adapter |
| Public objective contract | `treepo.objective.ObjectiveSpec` (frozen, strict-convex) | `src/ctreepo/contracts.py::ObjectiveSpec` compatibility facade; `treepo_cdx` archived |
| Repo-specific tree/trace/model row construction | new `src/ctreepo/local_law_rows.py` | all family-specific row builders delegate here |
| f/g training adapter interface | `src/ctreepo/alternating.py::FamilyRuntime` | FNO, DSPy, TRL, oracle families plug in as adapters |
| Model execution surface | `src/core/inference_engine.py::InferenceEngine` | no duplicate chat/embedding/operator/diffusion/symbolic transport |

## Single row/tensor contract (every family emits these; nothing else)

Per node: `prediction`, `proxy_target|proxy_loss`, `oracle_target|oracle_loss`,
`observed`, `propensity`, `depth`, `node_weight`, `law_kind`, `global_axiom`,
`state_kind`, `law_channel`, `doc_id`, `node_id`, + optional `metadata`. Family
adapters ONLY build rows; they must not implement their own AIPW / depth
weighting / Λ mixing.

## Invariants

- Training rows and audit rows are **separate types**. Training may include
  unobserved proxy-only rows with `observed=False`, `propensity=0` (no division
  occurs). Audit rows with nonzero influence require positive logged propensities.
- All sampled objectives follow `docs/local_law_sampling_contract.md`.
- Full binary tree, `L` leaves → exactly `2L−1` logical rows (`L` leaves, `L−2`
  non-root internal, 1 root). **Root counted exactly once**; if a trace has both
  an explicit root row and a cumulative-merge root row, drop the cumulative one.
- Fixed-size uniform sampling logs `propensity = q/N`; Bernoulli logs its rate; no
  "at least one node per doc" rule unless the estimand says so. Rate grids use
  **persistent masks** across epochs (R10 = ~10% ever-labeled, not redrawn).
- `gamma_depth>1`, negative weights, invalid propensities, and mixed
  `root_share`+`local_law_weight` payloads **fail fast**.
- No family implements private AIPW denominators, depth-discount arithmetic, or
  root/local convex mixing (enforced by the static guard).

## Disposition policy (three tiers — quarantine-first, never delete first)

1. **Quarantine** (live callers / useful diagnostics): tag
   `legacy_quarantined`/`diagnostic_only` + guard, keep body, route new work to
   canonical. For: `g_a2_weight` (deprecated no-op), old q-sentence law rewards,
   `embedding_fno` role-weighted MSE (as `legacy_weighted_mse`), TRL SFT scaffold,
   A3 readout-factorization + g-associativity diagnostics (kept separate from A2).
2. **`OLD_`-archive** (confirmed dead, imported nowhere — per
   `feedback_old_prefix_for_legacy`): rename `OLD_` + header note. For:
   `treepo/src/treepo/_research/ctreepo/*`, `treepo_cdx`, pure re-export shims
   (`src/tree/neural_operator.py`).
3. **Delete body → shim** (only after static-search + parity proves no behavioral
   caller): overlap bodies of `local_law_torch.py` / `local_law_adjustment.py`,
   duplicated scalar aggregates, obsolete role-classification helpers.

**Never touch:** unrelated dirty worktree, existing `OLD_*` trees, paper/report
scripts, broad Markov configs outside the narrow migration gates.

## Current state to preserve (already DONE)

- [x] FNO A2 corrected: merge route vs **independent parent-text reading** (not
  the vacuous `f(parent_state)−f(merge(l,r))≡0` self-comparison).
- [x] FNO f/g route through `treepo.training.local_law.local_law_objective_from_losses`
  (AIPW + γ^depth); Λ wired to `(1−Λ)·root + Λ·law` for both phases.
- [x] γ depth-convention fixed (root = depth 0).
- [x] `g_a2_weight` deprecated no-op; `a3_factorization_weight` separate from A2;
  `g_assoc_weight` diagnostic only.
- [x] `docs/local_law_sampling_contract.md` establishes row + IPW contract.
- [x] `src/core/inference_engine.py` is the central execution surface.
- [x] 30 FNO tests green; Benoit smoke at (Λ=0.5, γ=0.5) and defaults.
- [ ] **Re-attribute prior "a2state win" → associativity penalty** in
  handoff/memory; the honest (γ,Λ) sweep must be re-run after the fix.

---

## Phases

### Phase 0 — Source inventory + guardrails (measure before changing)
- [ ] Inventory objective specs, local-law objective fns, root/local mixers, row
  builders, family reward helpers, archive mirrors.
- [ ] Record live import graph for `local_law_torch.py`, `local_law_adjustment.py`,
  `sim/composite_objective.py`, `contracts.py::ObjectiveSpec`, `treepo._research`,
  `treepo_cdx`.
- [ ] Allowlist source-guard test: reject new bespoke AIPW / depth-discount /
  propensity-denominator / root-local mixer code outside
  `treepo.training.local_law`, `treepo.objective`, approved shims/adapters.
- [ ] Archive-import guard: fail on imports from `OLD_*`, `treepo_cdx`,
  `_research` once each is archived.
- [ ] Baseline focused tests before edits: `tests/training/test_local_law_torch.py`,
  `tests/core/test_local_law_adjustment.py`, `tests/ctreepo/test_objective_weights.py`,
  `tests/ctreepo/test_neural_operator_baselines.py`, FNO A2 tests,
  `treepo/tests/training/test_local_law.py`.
- **Gate:** inventory note exists; guard tests pass with allowlist matching known
  duplicates.

### Phase 1 — Upstream canonical foundation in `treepo`
- [ ] Add `LocalLawTrainingRow` (full §contract; proxy-only `propensity=0` allowed,
  no division).
- [ ] Keep `LocalLawAuditRow` strict (positive propensities for
  observed/nonzero-influence rows).
- [ ] Move live-only helpers upstream (incl. `corrected_local_law_target_mse` if
  callers need target-level MSE).
- [ ] Add scalar training-aggregate helpers mirroring `local_law_adjustment.py`
  (not ThinkingTrees diagnostics).
- [ ] Extend `treepo.objective.ObjectiveSpec` with `oracle_state`,
  `external_passthrough` estimator modes; strict convex validation by default.
- [ ] Add/promote `resolve_root_local_objective_weights(...)` to `treepo.objective`.
- [ ] Preserve legacy payload reading; emit canonical `treepo` schema on write.
- **Tests:** training rows accept `propensity=0`; audit rows reject invalid
  propensities; scalar↔tensor parity; `gamma_depth>1` rejected; node-weight +
  zero-sample cases; legacy payload load + canonical round-trip.
- **Gate:** upstream `treepo` tests pass; ThinkingTrees still green before wiring.

### Phase 2 — ThinkingTrees compatibility shims
- [ ] `local_law_torch.py` → thin re-export/delegating shim (preserve public
  names/signatures + test-pinned symbols).
- [ ] `local_law_adjustment.py` → delegate scalar loss/normalize/constants; keep
  only ThinkingTrees diagnostics/dataclasses.
- [ ] `contracts.py::ObjectiveSpec` → compatibility facade over
  `treepo.objective.ObjectiveSpec`; accept legacy `ctreepo.objective.v1` payloads.
- [ ] `sim/composite_objective.py` → route through canonical resolver;
  `CompositeObjectiveSpec` stays an internal sim adapter.
- **Tests:** shim parity (bit/tolerance-identical on deterministic fixtures);
  ObjectiveSpec migration/serialization; composite root/local resolution;
  downstream imports still work.
- **Gate:** no behavior deltas except documented legacy serialization
  normalization; focused suite green.

### Phase 3 — Central row adapter `src/ctreepo/local_law_rows.py`
- [ ] `classify_node_role(node, *, root_id=None, tree_shape=None)`.
- [ ] `build_local_law_rows(...)` generic constructor (full §contract fields).
- [ ] Helpers: full-binary-tree population; cumulative-merge traces dropping
  duplicate root rows; sampling-policy (fixed-size uniform, Bernoulli, persistent
  mask, full-obs); metadata normalization; optional predictions/targets→losses.
- **Tests:** `2L−1` rows; root once; stable leaf/internal/root roles; `q/N` and
  Bernoulli propensities; persistent masks across epochs; missing oracle →
  proxy-only training row (not audit row); law metadata present for A1/A2/A3/root.
- **Gate:** unit-tested, opt-in only; no family migration yet.

### Phase 4 — FNO / neural-operator migration
- [x] Corrected A2 preserved; A3 separate; `g_assoc_weight` diagnostic only.
- [ ] `fno_family.py`: replace remaining private root/law split helper with the
  canonical resolver; emit rows via `local_law_rows.py`; keep current behavior
  behind a parity fixture until proven.
- [ ] `embedding_fno._batch_loss`: canonical row emission + `treepo.training.local_law`;
  retain old weighted-MSE only as `legacy_weighted_mse`; map config knobs to
  `ObjectiveSpec`/component weights/`node_weight`s.
- [ ] `markov_neural_operator_baselines.py` (FNOCountSketch): swap in-repo
  `local_law_torch` → `treepo.training.local_law`; replace
  `_fno_single_lambda_objective_loss` with canonical resolver+objective (or shim);
  preserve `tree_model_version` + `use_residual_decomposition`; **no per-node
  `.cpu()`/`.item()` in long merge chains**.
- [ ] Fold/archive redundant `EmbeddingFNOModelConfig`, `FNOFamilyConfig`,
  `ManifestoRileEmbeddingObjective` after import scan + parity.
- **Tests:** deterministic loss parity across `tree_model_version` × residual on/off;
  FNO A2 non-vacuity green; `embedding_fno._batch_loss` canonical test; Markov FNO
  parity; perf smoke (no hot-path sync regression).
- **Gate:** canonical path default for FNO/NO; old path diagnostic-only/archived.

### Phase 5 — DSPy / LLM family migration (GEPA preserved)
- **Hard constraint:** GEPA/MIPRO metrics are per-example; do NOT force GEPA
  training reward to be the full per-tree AIPW aggregate.
- [ ] DSPy record builders emit canonical rows via `local_law_rows.py`.
- [ ] Collapse base scalar reward + q-sentence vector reward + inline distillation
  reward into one shared per-example helper using `ObjectiveSpec` component names.
- [ ] Mark q-sentence C1/C3a/C3b reward variants `diagnostic_only` when not
  corrected/IPW paper-facing.
- [ ] Offline eval/audit computes canonical AIPW/local-law aggregate from emitted
  DSPy rows.
- [ ] Route distillation builders through central rows (not bespoke inline builders).
- [ ] TRL: keep SFT scaffold quarantined; define GRPO reward over canonical row
  outputs; smoke that LoRA weights change + rewards logged — **separate project,
  not blocking**.
- **Tests:** DSPy row emission; per-example reward parity; distillation/q-sentence
  builder parity; offline AIPW audit smoke; TRL reward smoke (once live).
- **Gate:** DSPy training behavior preserved where GEPA requires; offline canonical
  audit available.

### Phase 6 — Public API consolidation
- [ ] `CTreePOLearningSpec`: **recommend ThinkingTrees facade now**, upstream owner
  long-term; `fit(...)`/`preflight(...)` accept canonical + legacy specs.
- [ ] `ObjectiveSpec`: canonical owner `treepo.objective`; ThinkingTrees delegates;
  new writes use upstream schema.
- [ ] `FamilyRuntime`: single protocol boundary for FNO/DSPy/TRL/oracle/classical/
  learnable-constants; bundle-aware extensions additive/optional.
- [x] `InferenceEngine`: keep `src/core/inference_engine.py`; remove duplicate
  execution interfaces only after import scans; families call via adapters.
- **Tests:** public import; legacy payload migration; fit/dispatch smoke; runtime
  smoke. **Gate:** one documented import path per public surface.

### Phase 7 — Archive + delete bodies (only after guards active)
- [ ] Import scan proves `treepo._research.ctreepo` dead → `OLD_`-archive.
- [ ] Import scan proves `treepo_cdx` dead → archive.
- [ ] Pure re-export neural-operator wrappers unused → archive/delete per convention.
- [ ] Delete local-law duplicate bodies only after shims + parity landed.
- [ ] Update source guards so archived imports fail. **Gate:** no archived imports;
  focused suite green.

### Phase 8 — Acceptance test matrix (no publication sweeps)
- [ ] Upstream: `LocalLawTrainingRow`, strict `LocalLawAuditRow`, tensor/scalar
  parity, `ObjectiveSpec` validation + legacy payloads.
- [ ] ThinkingTrees: `tests/training/test_local_law_torch.py`,
  `tests/core/test_local_law_adjustment.py`, `tests/ctreepo/test_objective_weights.py`,
  `tests/ctreepo/test_composite_objective.py`, new `tests/ctreepo/test_local_law_rows.py`,
  new source-guard tests.
- [ ] FNO/NO: FNO A2 consistency, `tests/test_fno_null_space_law.py`,
  `tests/test_fno_extent_latent.py`, `tests/test_fno_merge_can_learn_average.py`,
  `tests/ctreepo/test_neural_operator_baselines.py`, embedding-FNO canonical parity.
- [ ] DSPy/LLM: record-weight + objective-summary, q-sentence/distillation row
  emission, offline AIPW audit smoke.
- [ ] Runtime/API: `CTreePOLearningSpec` compat, `FamilyRuntime` dispatch,
  `InferenceEngine` smoke.
- [ ] Smoke: tiny Markov FNO objective; tiny manifesto/FNO canonical rows; tiny
  DSPy offline-audit.

### Phase 9 — Documentation + handoff
- [ ] Update `docs/local_law_sampling_contract.md` if `LocalLawTrainingRow` changes
  row semantics.
- [ ] Mark phase completion in `docs/local_law_single_path_plan.md` (or retire it
  to this master plan as the single living file).
- [ ] Update `docs/ctreepo_python_code_map_for_llms.md` only after a fresh source
  inventory + AST sweep.
- [ ] Document archived paths + replacement imports; add a new-family-author
  migration guide (build rows, call canonical objective, never reimplement AIPW).
- [ ] Correct memory/handoff "a2state win" attribution.

## Implementation order (first coding pass)

1. Phase 0 inventory + guard allowlist.
2. Phase 1 upstream `LocalLawTrainingRow`, helper parity, objective resolver.
3. Phase 2 shims (`local_law_torch.py`, `local_law_adjustment.py`, `ObjectiveSpec`).
4. Phase 3 row adapter with tests, no family rewiring.
5. Phase 4 FNO/NO (most corrected objective work already done).
6. Phase 5 DSPy/LLM row emission + offline audit.
7. Phase 7 archive pass only after import guards active.

Behavior-preserving foundation precedes family rewrites; nothing is deleted before
parity tests prove the canonical path covers live behavior.

**PR batching (keeps the 10 phases executable):** Phases 0–1 land together
(inventory + upstream foundation); Phases 2–3 together (shims + row adapter);
Phases 4, 5, and 7 each standalone; Phases 6/8/9 fold into whichever PR touches
the relevant surface.

## Decisions (RESOLVED — both reconciling LLMs agree)

Formerly "open"; closed during reconciliation (`docs/COLLABORATION.md`). Revisit
only if implementation evidence contradicts.

1. **`CTreePOLearningSpec`** — ThinkingTrees **facade now**; upstream move is a
   later, separate change, out of scope here.
2. **Root/local resolver** — canonical owner `treepo.objective`. A temporary
   `src/ctreepo/objective_resolution.py` is allowed ONLY as a re-export shim
   (never a second implementation), only if upstream churn blocks landing
   directly; removed in Phase 7 if created.
3. **Corrected-A2 read** — descendant-leaf pooling is the default; re-embedding
   the concatenated parent text is opt-in, **recommended for the root** (subject
   to the two-leaf no-truncation rule), parity-tested.
4. **f/g Λ** — **shared Λ by default**; `ObjectiveSpec` allows an optional
   phase-specific override, enabled only after the shared path is stable.
5. **Root A2 row** — **excluded** from `local_law_loss` (root is the `(1−Λ)`
   term); enforced by the "drop cumulative-merge root row" invariant + explicit
   `no_double_count` metadata.
6. **`delta` knob** — **dropped**; no canonical definition, reserve no plumbing.
7. **TRL true-alternating** — **out of scope**; quarantine the SFT scaffold.

## Assumptions

- `/home/mlinegar/treepo` is the canonical editable package; small upstream API
  additions allowed.
- Quarantine-first, not immediate deletion.
- GEPA/MIPRO replacement with a per-tree batch optimizer is out of scope.
- Validation bar = deterministic unit + parity + tiny smoke + source-guard tests;
  no publication-scale sweeps for the refactor.
- Public API centers on `CTreePOLearningSpec`, `ObjectiveSpec`, `FamilyRuntime`,
  `InferenceEngine`; dirty worktree preserved.

## Work checklist (tracking)

- [x] Read both source plans + both LLM master drafts.
- [x] Check sampling contract + living plan + canonical-path review.
- [x] Produce merged master plan (this file) + `docs/COLLABORATION.md`.
- [x] Converge with the other LLM on this single canonical file; 7 decisions
  resolved; dated draft marked superseded.
- [x] Phase 0: source inventory + guards. (`docs/local_law_phase0_inventory.md`;
  `tests/ctreepo/test_local_law_source_guards.py` (Codex, fragment) +
  `tests/ctreepo/test_local_law_arithmetic_ownership_guard.py` (Claude, def-ownership
  + archive-import). Baseline recorded; one pre-existing WIP FNO failure noted.)
- [x] Phase 1: upstream `treepo` foundation. (Codex: `LocalLawTrainingRow`,
  `LocalLawTrainingAggregate`, `resolve_root_local_objective_weights`,
  `oracle_state`/`external_passthrough` estimators. Reviewed PASS by Claude.)
- [x] Phase 2: ThinkingTrees shims. (Codex: `local_law_torch.py`,
  `local_law_adjustment.py`, `sim/composite_objective.py` route through canonical.
  `contracts.py::ObjectiveSpec` facade DONE by Claude — both dual-live copies now
  serialize `treepo.objective.v1`, source estimator vocab + resolver from
  `treepo.objective`, read legacy payloads. Full validation delegation deferred
  (needs `composite_objective` producer fix for masked law packages).)
- [x] Phase 3: central row adapter. (Codex: `src/ctreepo/local_law_rows.py` +
  `tests/ctreepo/test_local_law_rows.py`. Reviewed PASS by Claude; consumers must
  filter the `no_double_count` root row.)
- [ ] Phase 4: FNO/NO migration. (A2 DONE. Remaining: resolver wiring + row
  emission + parity; reconcile `fno_family.py` sync drift — COLLABORATION risk #2.)
- [ ] Phase 5: DSPy/LLM migration.
- [ ] Phase 6: public API consolidation. (Includes `contracts.py` facade.)
- [ ] Phase 7: archive/dead-code pass.
- [~] Phase 8: acceptance tests. Foundation verified (137 green). Cross-family ×
  leaf-size acceptance matrix landed: `tests/ctreepo/test_treepo_families_across_leaf_sizes.py`
  (18 green) — oracle/learnable_constant/fno fit end-to-end over leaves {2,4,8,16}
  via `treepo.methods`; dspy/trl/diffusion wiring contracts. Remaining: live
  dspy/diffusion/trl across leaf sizes (integration-gated).
- [ ] Phase 9: docs + handoff. ("a2state win" re-attribution already correct in
  memory `project_local_law_single_canonical_path`.)
