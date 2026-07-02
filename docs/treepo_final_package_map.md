# TreePO Final Package Map, v6

Date: 2026-05-19

This is the internal map and roadmap for turning `treepo/` into the final
publication-ready C-TreePO package. It combines the Lean-first package spine
from the first map with the more concrete phase plan from the parallel review.
v3 added the comparison layer against existing implementations, so the package
migration preserves the local-law, sampling, error-decomposition, and
neural-operator bug fixes already learned in the workspace. v4 adds the
source-grounded reconciliation with the existing C-TreePO objective and
multi-backend runtime contracts. v5 adds the explicit reuse map: what we
promote from our own verified code, what we wrap from research code, what we
use from outside packages, and what is genuinely new package work. v6 makes
`treepo.fit(...)` part of the first package spine rather than a late phase:
the single call signature wraps the working paper exercise patterns
(`treepo-bench` suites, runtime role configs, and existing f/g ladders) while
manifest, audit, and certificate sidecars remain TreePO-owned.

The organizing decision is:

```text
treepo/src/treepo/ is the canonical package surface.
src/ remains research and migration scaffolding until it imports or emits
treepo package contracts.
```

External packages are useful as reference implementations, parity checks, and
optional adapters. They should never define the TreePO estimand, replace
manifest contracts, or emit final certificates directly.

## Non-Negotiable Foundations

1. Lean-aligned learning with `f*`, `f`, and `g`
   - `f*` is the truth or full-information oracle target.
   - `g` is the summary/state operator over leaves and merges.
   - `f` is the readout on states.
   - Training and evaluation route through one package-level `fit()` entry
     point that can dispatch to paper bench exercises, LongBench/runtime
     methods, DSPy/FNO/TRL/FamilyRuntime ladders, symbolic/sketch runtimes, and
     external-state backends while preserving artifact lineage.

2. Sampling from first principles
   - Top-level document/case sampling and node/summary sampling are separate
     lanes.
   - Every sampled theorem-facing row needs an observed indicator, logged
     propensity, effective propensity, influence weight, and diagnostics.
   - The package must report ESS, max weight, and max influence-to-propensity
     ratios where they affect the certificate.

3. Honest chunking as a first-class package object
   - The split is over top-level units, not chunks sampled after the fact.
   - The role tuple `(r_C, r_G, r_O)` records chunker, `g`, and oracle/readout
     train/eval status.
   - Adaptive chunking is certificate-ready only when driven by fold-specific
     frozen artifacts for held-out top-level units.

4. Folds as shared infrastructure
   - Cross-fitting, honest split, K-fold error bars, robustness folds, and
     DSL-style nuisance estimation should use one fold representation.
   - Fold-specific artifacts must have stable IDs and must round-trip through
     the manifest.

5. Certificates are assembled locally
   - Outside adapters can fill component estimates or radii.
   - Final `UnifiedLearningErrorCertificate` construction remains TreePO-owned.
   - Paper tables should be reconstructable from saved manifests and component
     evidence.

6. Objective and backend contracts are promotions, not reinventions
   - `src/ctreepo/contracts.py` already has an `ObjectiveSpec` v1 surface with
     canonical `root` and `local_law_corrected` terms.
   - Existing tests assert that `oracle_gap` is not an objective term and that
     legacy public fields such as `gap_weight`, `oracle_gap_weight`,
     `lambda_eff`, and `reliability` are rejected.
   - If a compatibility path later accepts legacy gap weights, they must be
     quarantined as ignored metadata/evidence, never reintroduced as a third
     additive objective term.
   - `src/ctreepo/alternating.py` already defines the backend runtime
     protocol. `treepo.fit(...)` should close that abstraction over manifests,
     folds, objective specs, and f-star lineage rather than invent a parallel
     training interface.

## Source Inventory

### Already clean in `treepo/`

- `treepo/src/treepo/core/experiment.py`
  - `ExperimentContext`, `SamplingPlan`, `NormalizedOutput`, and canonical
    sidecars.
  - Gap: experiment sidecars do not yet carry theorem-facing row lineage,
    fold roles, propensities, or certificate evidence.

- `treepo/src/treepo/core/roles.py`
  - Public role vocabulary: `scorer`, `summarizer`, `oracle`, `embedder`,
    `state_model`.
  - Gap: roles are not yet tied to top-level train/eval assignments or
    artifact admissibility.

- `treepo/src/treepo/core/refs.py`
  - `BenchmarkRef`, `MethodRef`, and `ResultRow`.
  - Gap: good for public metadata, insufficient for Lean-facing manifests.

- `treepo/src/treepo/sketches/protocol.py` and
  `treepo/src/treepo/sketches/tree_reducer.py`
  - Sketch adapter contract and tree fold/reduce surface.
  - Gap: sketch outputs are not yet normalized into C1/C2/C3 local-law rows.

- `treepo/src/treepo/training/local_law.py`
  - Canonical corrected and sampled-IPW local-law objective implementation.
  - This is the master objective that runners should call after building
    normalized rows.

- `treepo/src/treepo/bench/` and `treepo/src/treepo/runtime/`
  - Migrated benchmark, report, suite, and runtime scaffolding.
  - Gap: these are not yet certificate-ready unless they emit the new manifest
    and evidence objects.

### Richer contract code still outside the package

- `docs/unified_learning_theorem_map.md`
  - Canonical Lean object list and theorem surface.
  - v2 requirement: schema parity tests should be generated from this file
    instead of hand-written from memory.

- `docs/unified_learning_procedure.md`
  - Operational protocol for top-level units, chunker, `g`, `f`, oracle,
    query policy, folds, logged propensities, and report decomposition.

- `docs/local_law_sampling_contract.md`
  - Canonical sampled local-law row contract.
  - Explicitly routes experiment-specific code through
    `treepo.training.local_law`.

- `src/core/logged_supervision.py`
  - Mature `SamplingMetadata`, `ObservationUnitKind`, and logged observation
    records.
  - Promotion target for `treepo.sampling` or `treepo.audit`.

- `src/tree/ipw.py` and `src/tree/full_tree_ipw.py`
  - HT/Hajek estimators, fold helpers, empirical Bernstein CIs, full-tree
    node records, ESS, max-weight diagnostics.
  - Promotion target for package sampling and audit utilities.

- `src/training/run_pipeline.py`
  - Existing three-layer honesty implementation:
    `ThreeLayerHonestyConfig`, `assign_three_layer_split`,
    `assign_three_layer_roles`, `filter_items_by_three_layer_role`.
  - This is a promotion job, not a rewrite.
  - `AdaptiveChunkingConfig.crossfit_folds` is already threaded through parts
    of this workspace pipeline; final package work should reuse/promote that
    flag rather than introduce a second cross-fit knob.

- `src/preprocessing/chunker.py`
  - `AdaptiveChunkingConfig`, `HonestChunkingPolicy`,
    `assign_honest_split`, and adaptive feedback memory.
  - Gap: package-level certificate readiness should depend on cross-fitted
    top-level fold lineage, not only a boundary/evaluation feedback split.

- `src/ctreepo/contracts.py`
  - Law IDs, objective specs, tree bundle manifests, run manifests, learning
    specs, and program specs.
  - Promotion target, but final package contracts should be smaller and mirror
    Lean objects directly.

- `src/ctreepo/alternating.py`, `src/ctreepo/dspy_family.py`,
  `src/ctreepo/fno_family.py`, `src/ctreepo/trl_family.py`
  - Current research f/g ladder and backend families.
  - Existing backend machinery under the immediate `treepo.fit()` facade;
    protocol completion remains a later gap-closing task.

### Lean anchors

- `lean3/FormalProofs/DSL/Honesty.lean`
  - `TopLevelIID`, `TopLevelExchangeable`, `ParentOf`,
    `DerivedRowHonestyContract`, `KFoldHonestTraining`,
    `KFoldHonestEvaluation`, `UnifiedLearningHonesty`,
    `ChunkerObjectiveTerms`.

- `lean3/FormalProofs/DSL/DocumentStructure.lean`
  - `Span`, `AdmissiblePartition`, `ChunkPartitionContract`,
    `RunManifestContract`, `ManifestRolesConsistent`,
    `ManifestSupportsValid`.

- `lean3/FormalProofs/OPT/InfluenceWeightedLocalLaws.lean`
  - `LocalLawAuditRow`, `InfluenceWeightedAuditOverlap`,
    `InfluenceWeightedErrorCertificate`.

- `lean3/FormalProofs/DSL/UnifiedLearningCertificate.lean`
  - `UnifiedLearningErrorCertificate`,
    `UnifiedLearningComponentEvidence`,
    `UnifiedLearningPaperAssumptions`,
    `unified_learning_final_paper_certificate`,
    `unified_learning_final_paper_certificate_high_prob`.

### Outside-code lanes

Use the external work documented in `docs/outside_code.md` and
`docs/move_to_outside_code.md` as optional reference infrastructure:

- GRF, EconML, and CausalML for honesty and cross-fit vocabulary/parity.
- R `dsl` as the primary design-based supervised-learning reference.
- Python `dsl-kit` as a secondary optional reference.
- `datasketches` as optional sketch backends behind local sketch contracts.
- `ranger` as a performance/design reference, not a core dependency.

GPL/R-backed packages must stay outside core imports unless the project makes a
separate license decision.

## Lean-First Package Spine

The final package should expose these nouns before treating any run as
certificate-ready.

| Package noun | Lean / doc anchor | Current source | Final package home |
| --- | --- | --- | --- |
| Top-level unit | `DSL.TopLevelIID`, `DSL.TopLevelExchangeable` | `src/tree/ipw.py`, docs | `treepo.manifest` or `treepo.contracts` |
| Derived row | `DSL.ParentOf`, `DSL.DerivedRowHonestyContract` | `src/core/logged_supervision.py` | `treepo.manifest` |
| Role tuple | `DSL.UnifiedLearningHonesty` | `src/training/run_pipeline.py` | `treepo.honesty` |
| Fold assignment | `DSL.KFoldHonestTraining`, `DSL.KFoldHonestEvaluation` | `src/tree/ipw.py`, scripts | `treepo.folds` |
| Support span | `DSL.Span` | chunker and tree code | `treepo.manifest` |
| Chunk partition | `DSL.AdmissiblePartition`, `DSL.ChunkPartitionContract` | `src/preprocessing/chunker.py` | `treepo.chunking` or `treepo.manifest` |
| Run manifest | `DSL.RunManifestContract` | `src/ctreepo/contracts.py`, `ExperimentContext` | `treepo.manifest` |
| Local-law row | `LocalLawAuditRow`, C1/C2/C3 contract | `src/core/ops_checks.py`, local-law docs | `treepo.audit` |
| Sampling metadata | local-law sampling contract, TreeIPW | `src/core/logged_supervision.py`, `src/tree/ipw.py` | `treepo.sampling` |
| Influence overlap | `InfluenceWeightedAuditOverlap` | Lean only, scattered diagnostics | `treepo.audit` |
| Error certificate | `UnifiedLearningErrorCertificate` | Lean only, legacy audit certs | `treepo.certificate` |
| Component evidence | `UnifiedLearningComponentEvidence` | Lean only, report fragments | `treepo.certificate` |
| Program artifacts | unified learning procedure | `src/ctreepo/*`, `treepo.core.roles` | `treepo.fit` / `treepo.artifacts` |

## Package Gaps To Close

The package has a good release shell but does not yet have theorem-facing
contracts. A certificate-ready manifest needs, at minimum:

- stable `top_level_unit_id`;
- stable `row_id`;
- parent top-level unit ID for every derived row;
- `fold_id` and `split_seed`;
- role tuple `(r_C, r_G, r_O)`;
- artifact IDs for chunker, `g`, `f`, oracle/readout, query policy, and proxy;
- support spans or node/pair IDs;
- local-law kind using the C1/C2/C3 and Lean L1/L3/L2 mapping;
- observed indicator;
- logged propensity in `(0, 1]` for observed rows;
- effective propensity after floors/clipping;
- influence weight;
- truth and approximation label sources;
- component evidence links for local-law, calibration, estimation, and
  clipping/floor terms.

The current `ExperimentContext` records broad sampling metadata but cannot
prove these invariants. The final package should add validators first and only
then migrate training/runtime code behind them.

## What We Already Have And Must Preserve

The package migration is not a greenfield rewrite. Several corrected
implementations already encode the semantics that the final package must make
explicit and theorem-facing.

| Existing source | What it already gets right | Package implication |
| --- | --- | --- |
| `treepo/src/treepo/training/local_law.py` | Canonical package implementation for corrected local-law objectives, sampled IPW objectives, persistent masks, observed masks, propensities, and node weights. | New row builders must feed this objective rather than reimplementing sampled local-law arithmetic. |
| `src/core/local_law_adjustment.py` | Dependency-light corrected estimator `proxy + R/pi * (oracle - proxy)` plus aggregate diagnostics for proxy total, residual correction, corrected total, ESS, and max IPW weight. | The contract layer must preserve proxy/oracle/residual fields and make the correction reconstructable from saved rows. |
| `src/training/supervision/local_law_torch.py` | Torch analogue of the corrected estimator with observed-propensity validation and unobserved-row handling. | Package tensor builders should match this validation behavior and fail observed rows with invalid propensities. |
| `src/ctreepo/contracts.py` and `tests/ctreepo/test_tree_bundle_contract.py` | `ObjectiveSpec` v1 already normalizes canonical `root` and `local_law_corrected` terms and rejects additive `oracle_gap`/legacy gap public fields. | Add `treepo.objective` by promotion; do not loosen the no-additive-gap guardrail. |
| `src/ctreepo/alternating.py` | `FamilyRuntime` and `BundleAwareFamilyRuntime` already define the multi-backend f/g trampoline, init-spec grammar, and artifact validation surface. | `treepo.fit()` should be a manifest/fold/objective shell over this runtime contract. |
| `src/ctreepo/dspy_family.py`, `src/ctreepo/fno_family.py`, `src/ctreepo/trl_family.py`, and `src/ctreepo/oracles/runtime.py` | Existing families cover LLM, FNO, TRL, and oracle surfaces; FNO and Oracle already expose bundle-aware behavior. | Phase 7 is gap-closing across existing backends, not a new backend architecture. |
| `treepo/src/treepo/sketches/adapters/` and `treepo/src/treepo/sketches/protocol.py` | Classical sketch adapters already expose encode/merge/query surfaces. | Add a sketch `FamilyRuntime` wrapper only after preserving the sketch adapter contract. |
| `docs/local_laws_unification_handoff_2026-04-18.md` | Shared objective shape: root loss plus `local_law_weight` times the `rho_C1`/`rho_C2`/`rho_C3` weighted local-law objective; strict local laws depend on per-node prediction channels such as `forward_aux`. | `fit()` must preserve an auxiliary per-node prediction surface for strict law checks, not only root predictions. |
| `docs/unified_learning_procedure.md` and `lean3/FormalProofs/DSL/UnifiedLearningCertificate.lean` | Final certificate splits the target gap into local-law, calibration, estimation, and clipping components. | Certificate assembly must retain separate evidence components and reject collapsed one-scalar summaries. |
| `docs/unified_local_law_handoff_2026-03-09.md` | Resolved local-law learnability bugs in Markov and LDA stress paths, including correct baselines and selected learned programs. | Those bug cases become migration regression fixtures, not historical notes. |
| `docs/hll_jax_local_law_handoff_2026-05-08.md` | HLL experiments separated register/state local-law metrics from raw cardinality readout metrics and tracked estimate-aware auxiliaries. | Sketch certificates need both state/register evidence and scalar readout evidence with provenance. |
| `docs/markov_data_scaling_g_ablation_handoff_2026-05-06.md` | Large Markov/FNO paths require mini-batched merge supervision and chunked evaluation; the g-side was not the main bottleneck in the documented grid. | Neural-operator adapters must preserve performance-safe eval/training paths and domain-specific readout metrics. |
| `docs/markov_fno_local_law_bridge.md` and `docs/minimal_unified_gf_contract_2026-05-03.md` | The strict `g`/`f` factorization and PyTorch/JAX bridge evidence are useful evidence surfaces, but not theorem replacements. | Package reports should expose backend evidence while keeping Lean-facing claims tied to local contracts. |

The final package should therefore promote existing semantics into contracts:
normalize rows, validate manifests, and call the canonical objective. It should
not re-derive IPW, local-law, calibration, or f-vs-f-star semantics from
scratch.

## Error Decomposition Contract

The local-law residual is only one component of the final error story. The
package must keep these evidence components separate:

| Component | Meaning | Examples of required evidence |
| --- | --- | --- |
| `f*` truth error | Difference between the ideal target and the available truth source or reduction target. | `truth_source`, target definition, exact/analytic/dataset/teacher provenance, domain readout metric. |
| Oracle/proxy measurement error | Difference between oracle-observed loss and proxy or judge loss under the sampling design. | `proxy_loss`, `oracle_loss`, `observed`, `propensity`, estimator mode, calibration source. |
| Learned `f` readout error | Error in the readout from state/summary/sketch representation to the target scale. | `approx_source`, readout artifact ID, theta/count/regime/cardinality/RILE metrics, held-out fold ID. |
| Sampling/IPW error | Design-based uncertainty from observed rows and propensities. | `node_weight`, effective propensity, influence weight, ESS, max weight, `D_lambda`, `W_lambda`. |
| Clipping/floor error | Bias or radius introduced by propensity floors, clipping, or range restrictions. | floor policy, clipping policy, pre-clip estimate, reported estimate, clipping radius. |

Implementation rows should map to the Lean-facing certificate decomposition:

```text
targetGap -> oracleGap -> judgeGap -> estimateBeforeClip -> reportedEstimate
```

The component evidence mapping is:

- local-law evidence: `targetGap - oracleGap`;
- calibration evidence: `oracleGap - judgeGap`;
- estimation evidence: `judgeGap - estimateBeforeClip`;
- clipping evidence: `estimateBeforeClip - reportedEstimate`.

Every certificate-ready local-law or derived row should therefore carry, or
link to evidence carrying, at least:

- `truth_source`;
- `approx_source`;
- `proxy_loss`;
- `oracle_loss`;
- `observed`;
- `propensity`;
- `effective_propensity`;
- `node_weight`;
- `influence_weight`;
- domain-specific readout metrics.

The package must explicitly reject evidence that treats "f vs f*" as a single
undifferentiated residual unless it also provides provenance for local-law,
calibration, estimation, and clipping components. Local-law residuals can
bound one transport step, but they should not silently absorb oracle/proxy
measurement error, learned-readout error, and sampling uncertainty.

## Resolved Bug Ledger

These are known mistakes the final package should encode as regression checks
before migration:

| Area | Resolved behavior to preserve | Regression check |
| --- | --- | --- |
| Markov law-stress | Compare local-law variants against the matched `root_only` baseline, not an `undersupported` baseline. | A Markov law-stress fixture must fail if the comparison row is pulled from the wrong baseline family. |
| LDA law-stress | Score the selected `learned_g`, not the first candidate in a candidate list; primary LDA utility should use downstream target error rather than a local fallback. | A fixture with multiple candidate `learned_g` artifacts must select by recorded artifact ID. |
| HLL/data sketches | Report register/local-law metrics separately from raw cardinality readout; estimate-aware auxiliary loss is optional and provenance-tracked. | HLL reports must include both state/register metrics and scalar readout metrics, with auxiliary-loss flags recorded. |
| Uniform all-node sampling | Avoid duplicate root rows and use the actual inclusion probability, not a display or nominal sampling rate. | Full binary tree conversion should yield `2L - 1` rows and logged propensities should match the actual design. |
| Large Markov/FNO paths | Use chunked evaluation and batched merge supervision to avoid OOM and throughput regressions. | Large fixture smoke tests should exercise chunked eval and mini-batched merge-loss paths. |
| Adaptive/per-node telemetry | Avoid per-node `.cpu()` or `.item()` calls in hot paths; collect full traces only when requested. | Hot-path tests should assert telemetry defaults do not materialize per-node CPU scalars. |

This ledger is part of the acceptance criteria. A new package implementation
that passes formal-looking schema tests but reintroduces one of these bugs is
not acceptable for final C-TreePO use.

## Domain Generality Matrix

The final package should be general enough for LLMs, Markov processes, data
sketches, LDA/learned sketches, and neural operators without flattening their
different error surfaces.

| Domain | `f*` truth target | Learned/approx `f` | `g` operator | Local-law target | Certificate evidence |
| --- | --- | --- | --- | --- | --- |
| LLM/RILE | Human label, dataset label, or calibrated teacher score on the full top-level unit. | Judge/readout on summaries, chunks, or tree states. | Summary program over chunks and merges. | Per-leaf and per-merge RILE preservation, with honest evaluator/calibration evidence. | RILE error, judge calibration, sampling/IPW diagnostics, chunk/fold artifact lineage. |
| Markov | Analytic state, theta, count, or regime readout under the simulated process. | Learned decoder/readout from state representation. | Learned merge/state map over fragments and nodes. | State-transition, merge-consistency, contextual-sufficiency, and root readout residuals. | Theta/count/regime metrics, local-law eps metrics, f/g artifact IDs, chunked-eval provenance. |
| HLL/data sketches | Exact sketch/register/readout contract for the finite stream. | Formula readout or learned decoder from sketch state. | Sketch update/merge operator over leaves and internal nodes. | Register/state preservation and merge laws, separate from scalar cardinality error. | Register MAE, law eps, raw/relative cardinality error, auxiliary estimate-loss provenance. |
| LDA/learned sketches | Downstream utility or theta target for the selected topic/sketch program. | Selected learned readout for the recorded program. | Selected `learned_g` state/update program. | Program-consistency and downstream utility preservation for the selected artifact. | Selected artifact ID, target utility error, local-law residuals, candidate-selection provenance. |
| Neural operators/FNO | Domain-specific analytic or teacher target used to define the theorem-facing row. | Neural readout head or decoder. | FNO/CleanUnifiedNO merge or state evolution surface. | Explicit f/g factorization metrics plus strict local-law metrics when available. | Backend config, `forward_aux` or equivalent per-node channels, local-law losses, readout metrics. |

CleanUnifiedNO/FNO paths are important implementation and evidence surfaces.
They should not be described as theorem replacements unless their rows,
propensities, truth sources, readouts, and component evidence can reconstruct
the same Lean-facing manifest and certificate.

## Objective Spec Non-Regression Contract

The final package needs a first-class objective contract because the local-law
loss and the f-vs-f-star decomposition are easy to regress.

Current source facts to preserve:

- `src/ctreepo/contracts.py` exposes `ObjectiveSpec` v1.
- Canonical public objective terms are `root` and `local_law_corrected`.
- `tests/ctreepo/test_tree_bundle_contract.py` asserts `oracle_gap` is not a
  term in normalized objective metadata.
- `tests/ctreepo/test_tree_bundle_contract.py` and
  `tests/ctreepo/test_objective_weights.py` reject legacy public fields such
  as `gap_weight`, `oracle_gap_weight`, `lambda_eff`, `lambda_effective`,
  `lambda_nominal`, and `reliability`.

Package rules:

- Promote this surface into `treepo.objective.spec`.
- Keep root loss and corrected local-law loss as the only objective terms for
  certificate-ready runs.
- Do not add an additive `oracle_gap`/`gap_weight` term. The f-vs-f-star gap
  belongs in reliability, calibration, or component evidence, not as a third
  training term that double-counts the same learned readout.
- Certificate-ready manifests must record the effective local-law weight
  actually used. Today that can be reconstructed from `root_share`,
  `local_law_weight`, and component weights; if a future reliability-scaled
  `lambda_effective` is introduced, it must be logged as manifest/evidence
  metadata and must be present whenever a corrected local-law estimator is
  active.
- If historical configs with `gap_weight` are accepted for migration, the
  value must be routed to `metadata["ignored_legacy_gap_weight"]` or an
  equivalent quarantine field and excluded from objective arithmetic.

This contract refines the v3 error decomposition: local-law, calibration,
estimation, and clipping evidence are certificate components; the optimizer
still sees only the root/corrected-local-law objective family.

## Backend Runtime Contract And Gaps

The multi-backend abstraction already exists in `src/ctreepo/alternating.py`.
The package should promote and close it.

`FamilyRuntime` requires:

- `name`;
- `train_f(...)`;
- `train_g(...)`;
- `score_roots_with_f(...)`;
- `validate_artifact(...)`.

`BundleAwareFamilyRuntime` adds:

- `default_f`;
- `default_g`;
- `expected_bundle()`;
- `supported_inits()`;
- `resolve_init(...)`;
- `share_state_axes()`.

The existing init-spec grammar already covers sentinel specs such as
`identity`, `raw`, `raw_concat`, `external_passthrough`,
`pretuned_scorer`, and `bare_scorer`, plus `oracle:<name>` and
`artifact:<path>`.

Current backend state:

| Surface | Existing source | Package gap |
| --- | --- | --- |
| DSPy text families | `src/ctreepo/dspy_family.py`, `src/ctreepo/joint_dspy_family.py`, `src/ctreepo/manifesto_qsentence_dspy_family.py` | Declare supported supervision depths and bundle-aware init/shape behavior explicitly. |
| FNO family | `src/ctreepo/fno_family.py` | Preserve bundle-aware behavior, full-tree trace exports, and state-shape constraints. |
| TRL family | `src/ctreepo/trl_family.py` | k>=1 remains partial/stubbed; certificate-ready runs must fail loudly or wire `train_g` through the intended GRPO path. |
| Oracle family | `src/ctreepo/oracles/runtime.py` | Treat f-star/oracle handles as first-class artifacts in package manifests. |
| Classical sketches | `treepo/src/treepo/sketches/adapters/` | Add `ClassicalSketchFamilyRuntime` only as a wrapper over existing sketch adapters. |
| Markov/FNO/CleanUnifiedNO simulation surfaces | `src/ctreepo/sim/core/*` | Keep them as evidence surfaces with explicit f/g factorization, not theorem replacements. |

Phase 7 should add two declarations that are missing from the current protocol:

- `state_shape_contract() -> Mapping[str, Any]`, so shape errors surface at
  initialization rather than during a backend forward pass.
- `supported_supervisions() -> frozenset[str]`, so a backend states whether it
  can train on `root`, `leaf`, `merge`, and range/idempotence rows.

The release rule is simple: a backend that does not declare bundle, shape,
supervision, f-star source, and artifact lineage can still run diagnostics,
but it cannot produce certificate-ready evidence.

## Reuse Map: What To Reuse, Wrap, Or Build

The package should be built by promotion and adapter work wherever possible.
New implementation should be limited to contract glue, validators, row
builders, release gates, and tests that were missing from the research code.

### Reuse Our Own Code Directly Or By Promotion

| Package need | Reuse source | Reuse mode | New work still needed |
| --- | --- | --- | --- |
| Corrected local-law objective | `treepo/src/treepo/training/local_law.py` | Keep as the canonical package objective; call it from row builders. | Build normalized row-to-tensor adapters and parity fixtures. |
| Dependency-light corrected estimator diagnostics | `src/core/local_law_adjustment.py` | Promote scalar helper behavior or keep as a parity oracle while package code stabilizes. | Normalize diagnostics into certificate evidence fields. |
| Torch local-law parity | `src/training/supervision/local_law_torch.py` | Use as the torch analogue and validation reference. | Ensure package tensor builders match observed-propensity checks. |
| Sampling metadata and logged labels | `src/core/logged_supervision.py` | Promote `SamplingMetadata`, `ObservationUnitKind`, logged observations, and artifact summaries into `treepo.sampling`. | Add theorem-facing IDs, role/fold lineage, and manifest validation. |
| IPW/HT/Hajek/ESS utilities | `src/tree/ipw.py` and `src/tree/full_tree_ipw.py` | Promote estimators, ESS/max-weight diagnostics, and full-tree node records. | Add Lean-facing `D_lambda`/`W_lambda` overlap object and row validators. |
| PPS/systematic sampling | `src/stats/sampling.py` | Reuse allocation and sampling primitives. | Add package-level propensity records and finite-population tests. |
| Objective metadata | `src/ctreepo/contracts.py` | Promote `ObjectiveSpec`, objective normalization, and digest logic into `treepo.objective`. | Keep no-additive-gap guardrails and add package-light import tests. |
| Tree/run bundle metadata | `src/ctreepo/contracts.py` | Reuse `TreeBundleManifest`, `RunManifest`, and lineage ideas as migration input. | Split into smaller Lean-facing package schemas with stricter validators. |
| Three-layer honesty | `src/training/run_pipeline.py` | Promote `ThreeLayerHonestyConfig`, `assign_three_layer_split`, `assign_three_layer_roles`, and filtering helpers. | Add release gates, manifest role tuples, and leakage fixtures. |
| Adaptive cross-fit knob | `src/preprocessing/chunker.py` and `src/training/run_pipeline.py` | Reuse `AdaptiveChunkingConfig.crossfit_folds` rather than adding another flag. | Move fold lineage into package artifacts and mark non-cross-fit adaptive chunking diagnostic-only. |
| K-fold scaffolding | `scripts/run_kfold_cv.py`, `scripts/run_governed_kfold_cv.py`, `scripts/phase1_optimize_scorer_kfold.py` | Extract deterministic fold assignment patterns and artifact layout conventions. | Build dependency-light `treepo.folds` with manifest-addressable fold IDs. |
| Backend runtime | `src/ctreepo/alternating.py` | Promote/adapt `FamilyRuntime`, `BundleAwareFamilyRuntime`, init-spec parsing, and helper dispatch. | Add shape/supervision declarations and certificate-ready backend gates. |
| Existing backend families | `src/ctreepo/dspy_family.py`, `src/ctreepo/fno_family.py`, `src/ctreepo/trl_family.py`, `src/ctreepo/oracles/runtime.py` | Wrap them under `treepo.fit(...)`; do not rewrite training algorithms first. | Make bundle-aware and supervision-depth behavior explicit; keep partial TRL diagnostic-only. |
| Sketch protocol and adapters | `treepo/src/treepo/sketches/protocol.py`, `treepo/src/treepo/sketches/tree_reducer.py`, `treepo/src/treepo/sketches/adapters/` | Keep the protocol and adapters; add a runtime wrapper only after preserving adapter semantics. | Add `ClassicalSketchFamilyRuntime` and sketch local-law row builders. |
| HLL package implementation | `treepo/src/treepo/hll.py` and `treepo/src/treepo/sketches/adapters/hll_native.py` | Keep local HLL as deterministic reference for sketch fixtures. | Separate register/state metrics from scalar cardinality readout evidence. |
| Markov/LDA/FNO regression surfaces | `src/ctreepo/sim/core/*`, especially Markov, LDA, and neural-operator files | Use as regression fixtures and evidence surfaces. | Do not migrate wholesale into core; extract small deterministic fixtures and expected metrics. |
| Lean schema anchors | `docs/unified_learning_theorem_map.md` and Lean files under `lean3/FormalProofs/` | Generate schema parity expectations from these sources. | Add a small parser/test harness and keep package names synchronized. |

### External Code Reuse

External packages should reduce statistical and engineering duplication, but
only after TreePO rows, manifests, and evidence objects are local.

| Need | External source | Reuse mode | Boundary |
| --- | --- | --- | --- |
| Honesty vocabulary and split discipline | GRF in `outside_data/method_reference_repos/grf` (`grf 2.6.1`, GPL-3) | Reference model and optional subprocess/test-only parity harness. | No core import, no copied GPL implementation, no direct TreePO estimand outsourcing. |
| Python cross-fit and causal forest reference | EconML in `outside_data/method_reference_repos/econml` (`econml 0.16.0`, MIT/BSD notices) | Optional Python parity tests for cross-fit/frozen nuisance patterns. | Optional extra/test-only import; TreePO still owns role tuple and manifest. |
| Causal tree/forest baselines | CausalML in `outside_data/method_reference_repos/causalml` (`causalml 0.16.0`, Apache-2.0) | Optional benchmark/parity baseline for tree-style learners. | Never a final certificate builder. |
| Design-based supervised learning | R `dsl` in `outside_data/method_reference_repos/dsl-r` (`dsl 0.1.0`, GPL-2) | Primary external reference for top-level DSL estimates and standard errors through an R subprocess. | Keep out of core imports; compare only after estimand metadata matches. |
| Python DSL reference | `dsl-kit` in `outside_data/method_reference_repos/dsl-python` (`dsl_kit 0.2.0`, MIT) | Secondary Python reference, useful for CI after R `dsl` parity is understood. | Optional extra; cannot define the canonical DSL estimand alone. |
| Sketch runtimes | Apache DataSketches through existing optional adapters under `treepo/src/treepo/sketches/adapters/` | Runtime backend and parity baseline for cardinality, frequency, quantile, tuple/sampling sketches. | Normalize outputs into local sketch state/readout/local-law evidence. |
| Random forest performance reference | ranger in `outside_data/method_reference_repos/ranger` (`ranger 0.18.0`, GPL-3) | Performance/design reference and optional benchmark only. | No core dependency and no copied GPL implementation. |
| Generic ML backends | scikit-learn, torch, DSPy, TRL, FNO code already in workspace | Model training backends behind artifact IDs. | Heavy imports stay outside core contracts and certificate assembly. |

The external inventory in `docs/outside_code.md` records tested versions,
commits, and caveats. The reuse rule is: external outputs can populate
component evidence only after a manifest hash, adapter version, package
version, estimator arguments, seed/thread controls, and input/output hashes are
recorded.

### What Is Genuinely New

These pieces are not already present as robust, package-native code and should
be implemented in the final package:

- dependency-light `treepo.manifest` schemas that mirror Lean names directly;
- `treepo.objective` promotion with core import tests and no-additive-gap
  release gates;
- manifest validators for parent rows, spans, role tuples, fold IDs,
  propensities, artifact lineage, and evidence links;
- row builders from full tree traces, sampled node traces, cumulative merge
  traces, top-level DSL tables, and sketch states into one normalized contract;
- `treepo.audit.InfluenceWeightedAuditOverlap` with `D_lambda`, `W_lambda`,
  ESS, max weight, and floor/clipping diagnostics;
- `treepo.folds` with a shared representation for honesty, cross-fitting,
  K-fold CV, robustness, and adaptive chunking artifacts;
- `treepo.certificate` assembly, component-evidence validation, and paper-table
  reconstruction;
- optional adapter output JSON schema for GRF/EconML/CausalML/DSL/DataSketches;
- release gates that classify runs as certificate-ready or diagnostic-only;
- regression fixtures for the resolved bug ledger and reuse-parity checks.

### Reuse Order

1. Promote local, dependency-light contracts and arithmetic first:
   local-law objective, objective spec, logged supervision, sampling/IPW, and
   honesty helpers.
2. Add validators and row builders that call those promoted surfaces.
3. Wrap existing backends through `FamilyRuntime` and make their support
   declarations explicit.
4. Add external adapters only after the local manifest/evidence JSON shape is
   stable.
5. Use external packages mostly for parity and standard-error/reference checks,
   not as sources of theorem-facing definitions.

## Unified `fit()` Plan From Working Paper Patterns

The package-level `fit()` should follow the entrypoints that already support
paper exercises rather than invent a new training grammar. The current working
patterns are:

| Working pattern | Existing source | What `treepo.fit(...)` preserves |
| --- | --- | --- |
| Config-driven paper suites | `treepo/src/treepo/bench/runner.py`, `treepo/src/treepo/bench/suites/paper.py`, and `treepo/examples/*.yaml` | A small config dictionary, experiment ID, JSON/CSV outputs, and existing paper smoke/grid behavior for cardinality, HLL merge learning, classical sketches, LDA, and LongBench runtime. |
| Runtime role configs | `treepo/src/treepo/runtime/eval.py` and runtime examples | `benchmark`, `methods`, role configs (`scorer`, `summarizer`, `embedder`, `state_model`, `oracle`), predictions, calls, and method metrics. |
| f/g ladder over `FamilyRuntime` | `src/ctreepo/learning.py` and `src/ctreepo/alternating.py` | `space_kind`, `family`, `schedule`, `initial_artifacts`, `backend_config`, train/eval traces, ladder manifests, and f/g lineage. |
| Sketch runtime/program facade | `src/ctreepo/runtime.py`, `treepo/src/treepo/sketches/*` | Existing sketch adapter semantics and state/query/merge behavior; package wrappers should not redefine HLL or DataSketches logic. |
| LLM/DSPy manifesto ladders | `scripts/run_alternating_ladder.py`, `scripts/run_manifesto_teacher_fg_leaf_grid.py`, manifesto task code | Existing tree-bundle inputs, leaf axes, teacher/oracle artifacts, warm-start preflights, and f-vs-f-star diagnostics. |

The public call shape is intentionally narrow:

```python
from treepo import fit

result = fit(
    config,
    task=None,
    backend=None,
    output_dir="outputs/run",
    train_data=None,
    eval_data=None,
    **backend_options,
)
```

`config` is a mapping or `FitConfig`. The dispatcher infers the lane from the
same shapes already used in the repo:

- `{"experiment": "hll-merge-learning", "config": {...}}` or
  `fit(..., task="hll-merge-learning")` routes through the existing
  `treepo-bench run` experiment registry.
- Runtime configs with `benchmark` and `methods` route through
  `treepo.runtime.run_runtime_eval`.
- Learning specs with `family` and `schedule` route through
  `src.ctreepo.learning.fit` when the monorepo backend is available.

The return shape is also narrow:

```python
FitResult(
    status="ok",
    metrics={...},
    artifacts={...},
    history=(...),
    summary={...},
    manifest_path="...",
    mode="runtime|paper_experiment|learning",
)
```

This is deliberately not a new backend API. Backend-specific config remains in
the existing working config dictionaries. The new package responsibility is to
standardize the entrypoint, sidecars, role/objective/manifest/certificate
contracts, and release gates.

### Minimal `fit()`-First Phase Order

The prior two plans both wanted the Lean-facing contract spine before full
training migration. The reconciliation is to add `fit()` immediately, but make
it a thin dispatcher until the contract sidecars catch up.

1. **Fit facade and contract skeleton**: add dependency-light
   `treepo.learning.fit`, `treepo.manifest`, `treepo.objective`,
   `treepo.audit`, `treepo.sampling`, `treepo.honesty`, and
   `treepo.certificate`. The facade runs existing paper/runtime/ladder lanes
   without copying backend logic.
2. **Sidecar normalization**: every `fit()` lane writes or returns a
   `FitResult`; paper exercises keep their JSON/CSV files, runtime keeps
   predictions/calls, and learning keeps ladder manifests. The next step is to
   attach `RunManifestContract` sidecars beside those outputs.
3. **Certificate-ready classification**: runs remain diagnostic unless their
   manifest validates top-level units, row parents, spans, role tuples, folds,
   artifact lineage, propensities, and component evidence.
4. **Backend declarations**: after the facade is stable, extend existing
   `FamilyRuntime` families with `state_shape_contract()` and
   `supported_supervisions()`. This closes protocol gaps without rewriting DSPy,
   FNO, TRL, or sketch code.
5. **Paper exercise migration**: convert one proven script at a time into a
   `fit()` smoke while preserving old CLI behavior. The first candidates are
   runtime mock LongBench, HLL merge learning, classical sketches with
   `execution_backend=treepo`, and one manifesto f/g ladder row.
6. **External adapters last**: R `dsl`, EconML/GRF parity, and DataSketches
   parity consume local manifest/evidence JSON. They never emit final
   certificates directly.

The invariant is that `fit()` is the public handle, while certificate readiness
is earned by sidecar validation. A backend that can run but cannot emit
manifest/evidence stays useful and diagnostic without being promoted to a final
paper claim.

## Package Module Plan

The following module names are proposed so Phase 1 can be split into small PRs.
They are intentionally dependency-light and should not import torch, pandas,
OpenAI, DSPy, transformers, vLLM, R packages, or GPL-backed packages.

### `treepo.manifest`

Purpose: Lean-facing unit, row, span, partition, artifact, and manifest
schemas.

Core objects:

- `TopLevelUnit`
- `Span`
- `ChunkPartition`
- `ArtifactRef`
- `RoleTuple`
- `ManifestRow`
- `RunManifestContract`
- `ManifestValidationReport`

Required behavior:

- JSON-safe `to_dict()` / `from_dict()`;
- deterministic manifest digest;
- row parent validation;
- role/support validation;
- positive-propensity validation for observed rows;
- artifact lineage completeness checks;
- schema parity test generated from `docs/unified_learning_theorem_map.md`.

### `treepo.objective`

Purpose: package-native objective schema for root plus corrected local-law
training, preserving the current `ObjectiveSpec` v1 guardrails.

Core objects/functions:

- `ObjectiveSpec`
- `ObjectiveTerm`
- `normalize_objective_spec`
- `validate_objective_spec`
- `objective_spec_digest`
- `objective_metadata`

Source of truth:

- `src/ctreepo/contracts.py`;
- `tests/ctreepo/test_tree_bundle_contract.py`;
- `tests/ctreepo/test_objective_weights.py`.

Required behavior:

- only canonical objective terms `root` and `local_law_corrected`;
- reject additive `oracle_gap` terms;
- reject or quarantine legacy gap fields, never train on them;
- preserve C1/C2/C3 law IDs and component weights;
- record the effective local-law weight used for certificate-ready runs;
- keep calibration/reliability evidence outside objective arithmetic.

### `treepo.honesty`

Purpose: promote existing three-layer honesty into package-level API.

Core objects/functions:

- `ThreeLayerHonestyConfig`
- `assign_three_layer_split`
- `assign_three_layer_roles`
- `filter_items_by_three_layer_role`
- `honest_eval_unit_ids`
- `validate_manifest_roles_consistent`

Source of truth:

- promote from `src/training/run_pipeline.py`;
- keep deterministic SHA256-style assignment;
- preserve user-facing role labels `train` and `eval`.

Required behavior:

- top-level split only;
- no chunk-level train/eval leakage;
- manifest role tuple round-trip;
- leak-detection fixture that mutates held-out labels and verifies frozen
  split-building artifact hashes do not change.

### `treepo.sampling`

Purpose: shared top-level and derived-row sampling/IPW layer.

Core objects/functions:

- `SamplingMetadata`
- `ObservationUnitKind`
- `DocumentSamplingRow`
- `NodeSamplingRow`
- `hajek_mean`
- `horvitz_thompson_mean`
- `effective_sample_size`
- `max_weight`
- `validate_propensity_floor`

Source of truth:

- mirror or migrate from `src/core/logged_supervision.py`, `src/tree/ipw.py`,
  and `src/tree/full_tree_ipw.py`.

Required behavior:

- document-level lane for DSL-style supervised estimation;
- node/summary-level lane for local-law rows;
- support fixed-size, Bernoulli, PPS, and persistent-mask designs;
- no silent `pi = 0` for observed or consequential rows.

### `treepo.audit`

Purpose: local-law rows and influence-weighted overlap diagnostics.

Core objects/functions:

- `LawKind`
- `LocalLawAuditRow`
- `InfluenceWeightedAuditOverlap`
- `InfluenceWeightedErrorCertificate`
- `build_local_law_loss_tensors`
- `validate_local_law_rows`

Source of truth:

- Lean `OPT/InfluenceWeightedLocalLaws.lean`;
- `src/core/ops_checks.py` law naming;
- `treepo.training.local_law` objective arithmetic.

Required behavior:

- C1/C2/C3 row construction;
- Lean L1/L3/L2 mapping preserved in metadata;
- compute `D_lambda = sum(lambda(a)^2 / pi(a))`;
- compute `W_lambda = max(lambda(a) / pi(a))`;
- route sampled objective values through `treepo.training.local_law`.

### `treepo.folds`

Purpose: one fold representation for cross-fitting, honest split, K-fold CV,
and robustness folds.

Core objects/functions:

- `FoldSpec`
- `FoldAssignment`
- `make_kfold_assignments`
- `crossfit_splits`
- `validate_fold_disjointness`
- `fold_artifact_id`

Source of truth:

- workspace K-fold scripts;
- `src/tree/ipw.py` fold helpers;
- EconML-style crossfit mechanics as reference only.

Required behavior:

- every fold has train/eval unit sets;
- fold artifacts are stable and manifest-addressable;
- train/test disjointness is asserted;
- `AdaptiveChunkingConfig.crossfit_folds` is consumed/promoted rather than
  duplicated.

### `treepo.certificate`

Purpose: assemble component evidence and final paper certificates.

Core objects/functions:

- `UnifiedLearningErrorCertificate`
- `UnifiedLearningComponentEvidence`
- `CertificateBuildInput`
- `build_error_certificate`
- `validate_certificate_components`
- `reconstruct_paper_table_row`

Source of truth:

- Lean `DSL/UnifiedLearningCertificate.lean`;
- `docs/unified_learning_theorem_map.md`.

Required behavior:

- fields match the theorem surface:
  `reported_estimate`, `local_law_radius`, `calibration_radius`,
  `estimation_radius`, `clipping_radius`;
- high-probability deltas are component-specific;
- outside adapter outputs can populate component evidence only after manifest
  validation;
- no adapter can emit a final certificate directly.

### `treepo.fit` or `treepo.learning`

Purpose: immediate unification layer for all paper-facing training and
evaluation workloads.

Core objects/functions:

- `FitConfig`
- `FitResult`
- `fit`
- backend dispatch to existing DSPy/FNO/TRL/symbolic paths;
- f-star/f/g artifact lineage.

This starts as a thin dispatcher over existing working implementations. It can
run diagnostic paper exercises before every certificate sidecar is present, but
certificate-ready status requires valid manifest, honesty, sampling, fold, and
component-evidence outputs.

## Roadmap

### Phase 0 - Reuse inventory lock

Goal: before moving code, freeze the list of sources that will be promoted,
wrapped, or used as external references.

PR-sized steps:

1. Add a small checked-in reuse inventory that maps each package module to a
   source file or an explicit "new implementation" reason.
2. Add path-existence checks for every promoted source and optional external
   reference named in the inventory.
3. Add import-boundary checks so dependency-light modules do not import heavy
   backends, R bridges, or GPL-backed packages.
4. Add a rule that any replacement of a promoted implementation needs a parity
   fixture against the original source first.

Acceptance:

- Every Phase 1 module has a reuse decision: promote, wrap, adapter, reference,
  or new.
- No GPL/R-backed code is copied or imported by core modules.
- The reuse map and `docs/outside_code.md` agree on external pins and caveats.

### Phase 1 - Contract surface in `treepo/`

Goal: make theorem-facing data structures importable from the package without
moving training code yet.

PR-sized steps:

1. Add `treepo.manifest` with `TopLevelUnit`, `Span`, `ArtifactRef`,
   `RoleTuple`, `ManifestRow`, and `RunManifestContract`.
2. Add `treepo.objective.spec` by promoting `ObjectiveSpec` v1 from
   `src/ctreepo/contracts.py`.
3. Add serialization, digest, and validation reports.
4. Add a generated schema-anchor test that parses
   `docs/unified_learning_theorem_map.md` and asserts package objects exist for
   each required Lean-facing noun.
5. Add manifest round-trip tests: write JSON, read JSON, exact equality.
6. Add invalid manifest tests for missing parent ID, missing role tuple,
   missing artifact ID, invalid propensity, and invalid span.
7. Add objective invalid tests for additive `oracle_gap`, `gap_weight`,
   `oracle_gap_weight`, missing corrected-estimator weight provenance, and
   non-canonical law IDs.

Acceptance:

- `import treepo.manifest` is light.
- `import treepo.objective` is light.
- Schema parity is derived from the theorem map.
- ObjectiveSpec keeps `oracle_gap` out of objective terms.
- Existing package tests still pass.

### Phase 2 - Certificate and audit primitives

Goal: make the final certificate and influence diagnostics concrete before
training emits them.

PR-sized steps:

1. Add `treepo.certificate` with `UnifiedLearningErrorCertificate` and
   `UnifiedLearningComponentEvidence`.
2. Add `treepo.audit` with `LawKind`, `LocalLawAuditRow`, and
   `InfluenceWeightedAuditOverlap`.
3. Add tests for total-bound arithmetic and component delta summation.
4. Add tests for `D_lambda`, `W_lambda`, ESS, zero-row behavior, and propensity
   floor rejection.
5. Add a fixture that builds local-law tensors from rows and verifies parity
   with `treepo.training.local_law`.
6. Add a comparison fixture that reproduces
   `src/core/local_law_adjustment.py` and
   `src/training/supervision/local_law_torch.py` on the same observed, proxy,
   and oracle rows.

Acceptance:

- Final certificate fields match the Lean theorem map.
- Certificate component evidence preserves the
  `targetGap -> oracleGap -> judgeGap -> estimateBeforeClip ->
  reportedEstimate` chain.
- Influence-weighted local-law diagnostics can be computed without importing
  heavy optional dependencies.
- Package audit rows reproduce existing corrected local-law behavior and keep
  local-law, calibration, estimation, and clipping evidence separate.

### Phase 3 - Promote three-layer honesty

Goal: make `(r_C, r_G, r_O)` a package contract, not a buried training helper.

PR-sized steps:

1. Add `treepo.honesty` by promoting the implementation from
   `src/training/run_pipeline.py`.
2. Add role-tuple serialization into `treepo.manifest.ManifestRow`.
3. Add `ManifestRolesConsistent`-style validation.
4. Add leak-detection fixture: mutate held-out labels/residuals/law failures
   and assert split-building artifact hash remains unchanged.
5. Update workspace code to import the package honesty helpers where practical,
   leaving compatibility shims if needed.

Acceptance:

- Existing deterministic role assignment is preserved.
- Fold ID, split seed, and role tuple round-trip through the manifest.
- Certificate-ready evaluation set is `E_C intersect E_G intersect E_O`.

### Phase 4 - Sampling lanes and DSL parity

Goal: separate top-level document sampling from node/summary local-law
sampling, while sharing diagnostics.

PR-sized steps:

1. Add `treepo.sampling` by mirroring/migrating `SamplingMetadata` and
   observation-unit kinds.
2. Add document-level sampling rows for DSL-style estimation:
   `top_level_unit_id`, label observed indicator, inclusion probability,
   prediction, truth label, and covariates/metadata.
3. Add node-level sampling rows compatible with local-law audit rows.
4. Add HT/Hajek/ESS/max-weight utilities and tests on known finite
   populations.
5. Add optional R `dsl` subprocess adapter test on a tiny fixture; skip cleanly
   when R or `dsl` is unavailable.
6. Add hard failure for mismatched estimand metadata.

Acceptance:

- Document-level and node-level estimands are not conflated.
- Propensity/IPW diagnostics appear in manifest-compatible output.
- External DSL is reference-only.

### Phase 5 - Fold infrastructure and adaptive chunking cross-fit

Goal: make folds the shared mechanism for cross-fitting, honest split, CV, and
adaptive chunking.

PR-sized steps:

1. Add `treepo.folds` with fold specs, assignments, disjointness validation,
   and stable fold artifact IDs.
2. Connect `treepo.honesty` role assignment to fold views.
3. Promote the existing `crossfit_folds` behavior from workspace adaptive
   chunking into package fold config.
4. Mark adaptive chunking as certificate-ready only when it uses fold-specific
   frozen artifacts for held-out top-level units.
5. Add tests that saved fold artifacts reproduce the same point estimate.

Acceptance:

- Train and eval unit sets are disjoint for every fold.
- `crossfit_folds` is a real package-level setting.
- Adaptive chunking without fold lineage is diagnostic only.

### Phase 6 - Local-law row pipeline

Goal: ensure every sampled C1/C2/C3 objective goes through the package master
objective.

PR-sized steps:

1. Add row builders for full tree traces, sampled node traces, and cumulative
   merge traces.
2. Drop duplicated root rows when converting cumulative merge traces into the
   uniform all-node estimand.
3. Support node weights, depths, observed masks, propensities, and persistent
   masks.
4. Add parity tests against `treepo.training.local_law`.
5. Add comparison tests against `src/core/local_law_adjustment.py` and
   `src/training/supervision/local_law_torch.py` for the corrected estimator
   `proxy + R/pi * (oracle - proxy)`.
6. Add regression fixtures for duplicate-root handling and actual inclusion
   probabilities in uniform all-node sampling.
7. Update migrated benchmarks to emit local-law row sidecars when relevant.

Acceptance:

- No package runner implements independent IPW arithmetic for sampled
  local-law objectives.
- Package row builders match the existing corrected local-law semantics across
  package, dependency-light, and torch implementations.
- Uniform all-node and root-guaranteed estimands are explicitly named and not
  compared as if identical.
- Known root-duplication and displayed-rate propensity bugs are covered before
  migration.

### Phase 7 - Complete package `fit()` backend contracts

Goal: complete certificate-ready backend declarations after the thin
`treepo.fit()` facade is already routing working paper patterns.

PR-sized steps:

1. Extend `treepo.learning.fit()` with fold spec, objective spec, manifest
   sidecars, and backend readiness classification.
2. Wrap existing `src/ctreepo/alternating.py` and backend families rather than
   rewriting them.
3. Promote or adapt `FamilyRuntime` and `BundleAwareFamilyRuntime` as the
   package backend contract.
4. Add `state_shape_contract()` and `supported_supervisions()` declarations.
5. Add `ClassicalSketchFamilyRuntime` over existing sketch adapters.
6. Make non-bundle-aware backends diagnostic-only until they declare bundle,
   shape, supervision, and artifact lineage.
7. Track `f*` lineage as a first-class artifact source.
8. Emit f-vs-f-star gap and fold-specific artifact IDs per ladder step.
9. Preserve `forward_aux` or an equivalent per-node prediction channel so
   strict local laws can consume non-root predictions.
10. Rewrite one existing training script as a thin `treepo.fit(...)` smoke.

Acceptance:

- `treepo.fit(...)` can run a minimal backend and emit the same manifest schema
  as non-fit paths.
- f-star/f/g lineage is visible in sidecars and certificates.
- Strict local-law checks can consume per-node predictions, not only a final
  root prediction.
- Every certificate-ready backend declares supported init specs, expected
  bundle constraints, state shape, supported supervision depths, and artifact
  validation behavior.
- TRL k>=1 either works through the intended `train_g` path or fails as
  diagnostic-only before certificate assembly.

### Phase 8 - Certificate assembly and paper-table reconstruction

Goal: make final reporting mechanically reconstructable.

PR-sized steps:

1. Add `treepo.certificate.build_error_certificate(...)`.
2. Map local-law radius from influence-weighted audit evidence.
3. Map calibration radius from honest evaluator evidence.
4. Map estimation radius from fold/IPW/DSL evidence.
5. Map clipping/floor radius from propensity floor diagnostics.
6. Add report reconstruction check: saved manifest plus component evidence
   must rebuild the paper table row.
7. Reject evidence that collapses calibration, local-law, estimation, and
   clipping errors into one scalar without component provenance.

Acceptance:

- The final table can be rebuilt from artifacts without re-running training.
- Mismatched component targets or estimands hard-fail.
- Evidence that loses the Lean-facing component chain hard-fails.
- External adapters populate evidence, not final certificates.

### Phase 9 - Optional outside-code adapters

Goal: add reference adapters only after local contracts are stable.

PR-sized steps:

1. Add optional GRF/EconML honesty and cross-fit parity fixtures.
2. Add R `dsl` and Python `dsl-kit` reference adapters behind subprocess or
   optional extras.
3. Add `datasketches` adapter parity against local sketch fixtures.
4. Record external package versions, commit SHAs, seeds, input hashes, output
   hashes, command lines, and adapter versions.

Acceptance:

- Optional adapters skip cleanly when unavailable.
- Core package imports remain light.
- GPL/R-backed code is not imported by core modules.

## Verification Matrix

| Risk | Required test |
| --- | --- |
| Map/document drift | ASCII-only check plus path-existence check for cited repo files. |
| Lean/Python schema drift | Generate schema-anchor expectations from `docs/unified_learning_theorem_map.md`. |
| Additive objective regression | Assert normalized `ObjectiveSpec` has only `root` and `local_law_corrected` terms; reject `oracle_gap`, `gap_weight`, and `oracle_gap_weight` as training terms. |
| Missing effective objective provenance | Corrected-estimator runs must record reconstructable root/local-law weights, and future reliability-scaled runs must record effective local-law weight metadata. |
| Eval label leakage | Mutate held-out labels/residuals/law failures and verify frozen artifact hash unchanged. |
| Role lineage loss | Round-trip `fold_id`, `split_seed`, and `(r_C, r_G, r_O)` through manifest JSON. |
| Bad propensities | Reject observed rows with non-finite or non-positive propensities. |
| Hidden high-influence rows | Report ESS, max weight, `D_lambda`, and `W_lambda`; fail when configured floors are violated. |
| Local-law objective drift | Build tensors from rows and compare against `treepo.training.local_law`. |
| Corrected-estimator regression | Compare package rows against `treepo.training.local_law`, `src/core/local_law_adjustment.py`, and `src/training/supervision/local_law_torch.py` on shared fixtures. |
| Duplicate root row | Test cumulative merge trace conversion yields `2L - 1` rows for full binary trees. |
| f-vs-f-star collapse | Assert component evidence reconstructs `targetGap -> oracleGap -> judgeGap -> estimateBeforeClip -> reportedEstimate`. |
| Domain provenance loss | Markov, HLL, LLM/RILE, and sketch fixtures each identify truth source, proxy source, readout, local-law target, and certificate component. |
| Known migration bugs | Regression fixtures cover Markov matched `root_only`, LDA selected `learned_g`, HLL state/readout split, actual all-node propensities, chunked FNO eval, and hot-path telemetry defaults. |
| Backend protocol drift | Each certificate-ready backend satisfies bundle-aware init, shape, supervision-depth, artifact-validation, and f-star-lineage checks. |
| Sketch wrapper overreach | `ClassicalSketchFamilyRuntime` delegates to existing sketch adapters and does not redefine sketch semantics. |
| Adaptive chunking leakage | Require cross-fitted frozen artifacts for certificate-ready adaptive chunking. |
| DSL estimand mismatch | Tiny R `dsl` parity fixture plus hard failure for incompatible metadata. |
| Heavy import regression | Core import tests confirm no torch, pandas, OpenAI, DSPy, transformers, vLLM, or R packages are loaded. |
| Adapter overreach | External adapters can emit component evidence only, never final certificates. |
| Report irreproducibility | Reconstruct paper table from manifest plus component evidence. |

## End-To-End Done Definition

The package unification is complete enough for final C-TreePO use when one
script can:

1. Build a `RunManifestContract` from a top-level document table.
2. Validate an `ObjectiveSpec` with canonical root/corrected-local-law terms
   and no additive `oracle_gap` training term.
3. Assign `(r_C, r_G, r_O)` and `fold_id` deterministically from a seed.
4. Pass the leakage fixture.
5. Sample top-level documents with logged propensities and report document-lane
   diagnostics.
6. Build node/local-law rows with logged propensities and report audit-lane
   diagnostics.
7. Reproduce the existing corrected local-law estimator behavior from package,
   dependency-light, and torch fixtures.
8. Preserve f-vs-f-star decomposition into local-law, calibration, estimation,
   and clipping evidence.
9. Pass R `dsl` parity on a tiny synthetic top-level dataset.
10. Run cross-fitted training with per-fold frozen artifacts.
11. Run a minimal `treepo.fit(...)` backend through the existing
    `FamilyRuntime` contract.
12. Assemble `UnifiedLearningComponentEvidence`.
13. Build `UnifiedLearningErrorCertificate`.
14. Reconstruct the paper table from saved artifacts.

## Operating Defaults

- The package should prefer theorem alignment and reproducibility over fast
  external reuse.
- Fixed or weakly adaptive chunking remains the cold-start default.
- Adaptive chunking is certificate-ready only with cross-fitted top-level fold
  lineage.
- Local node/span sampling is design-based querying, not the honesty split.
- Training curves, local-law losses, and proxy-only estimates are diagnostics,
  not final claims.
- `oracle_gap` and f-vs-f-star diagnostics are evidence/reliability inputs,
  not additive objective terms.
- Local-law residuals do not absorb all f-vs-f-star error; truth, proxy,
  readout, sampling, and clipping evidence remain separate.
- Domain-specific readout metrics stay visible beside local-law metrics.
- Existing `FamilyRuntime` backends are the starting point for `treepo.fit`;
  package work closes protocol gaps instead of duplicating backend machinery.
- Workspace code should migrate only after it can satisfy package contracts and
  release gates.
- External packages are optional references and parity harnesses until their
  outputs are normalized into local manifest and evidence objects.

## First Implementation Target: Phase 1 PR Sequence

A practical first milestone is three small PRs:

1. Reuse inventory and import boundary
   - Add the module-to-source reuse inventory.
   - Add path-existence and dependency-light import tests.

2. `treepo.manifest` skeleton
   - Add core dataclasses and round-trip tests.
   - No integration with training code.

3. Manifest validation
   - Add validation report, invalid-fixture tests, and manifest digest.
   - Add the theorem-map-derived schema anchor test.

4. Release-gate integration
   - Extend `treepo-bench check launch` with an optional manifest-contract
     check over package fixtures.
   - Keep it small: one valid fixture and three invalid fixtures are enough.

This gets the package onto the right axis without touching the heavier
training, chunking, or outside-code adapters.
