# Phase 0 Source Inventory — Local-Law Canonical Path (2026-06-25)

Measure-before-changing inventory for
`docs/local_law_single_path_master_plan.md`. Captured before any Phase 1+ edits.

## Canonical arithmetic homes (treepo, authoritative)

- `treepo/src/treepo/training/local_law.py` — tensor AIPW + γ^depth:
  `corrected_local_law_loss_tensor`, `local_law_objective_from_losses`,
  `local_law_objective_target_mse`, `_depth_discount_weights`,
  `sampled_uniform_node_ipw_mean_loss`, `observed_uniform_node_ipw_mean_loss`.
  γ validated to `[0, 1]` (fails fast on `>1`).
- `treepo/src/treepo/local_law.py` — scalar + strict audit rows: `LawKind`,
  `corrected_local_law_loss`, `LocalLawAuditRow` (positive propensity required),
  `local_law_objective_summary`, influence-overlap helpers.
- `treepo/src/treepo/objective.py` — `ObjectiveSpec` (frozen, strict-convex by
  default via `allow_nonconvex_objective`), `canonical_law_component_weights`,
  `normalize_objective_spec`. **No `resolve_root_local_objective_weights` yet** —
  Phase 1 adds it.

## In-repo arithmetic to shim (Phase 2)

Canonical local-law functions are **defined** in exactly two ThinkingTrees files
(everything else only imports them):

- `src/training/supervision/local_law_torch.py` — tensor helpers
  (`corrected_local_law_loss_tensor`, `local_law_objective_from_losses`,
  `local_law_objective_target_mse`, `corrected_local_law_target_mse`,
  `_depth_discount_weights`). NOTE: local `_depth_discount_weights` allows
  `gamma >= 0` (no upper bound) — upstream is stricter (`[0,1]`); shim must
  preserve test-visible behavior or migrate tests.
- `src/core/local_law_adjustment.py` — scalar helpers + diagnostics
  (`corrected_local_law_loss`, `depth_discount`, `local_law_objective_mean`,
  `normalize_local_law_objective_mode`, `LocalLawObservation`,
  `LocalLawAggregate`, `aggregate_local_law_observations`). NOTE: scalar
  `LocalLawObservation` allows `propensity` in `[0,1]` and proxy-only
  unobserved rows — this is the *training* relaxation (audit rows stay strict).

## Live import graph (who consumes the shims)

- `local_law_torch`: `src/training/tree_model_v2_trainer.py`,
  `src/ctreepo/sim/core/markov_neural_operator_baselines.py`,
  `scripts/run_fno_mergeable_sketch_diagnostic.py`, tests.
- `local_law_adjustment`: `src/core/__init__.py`, `src/training/run_pipeline.py`,
  `src/training/tree_model_v2_trainer.py`,
  `src/training/supervision/optimizer_metadata.py`,
  `src/training/supervision/local_law_torch.py`,
  `src/ctreepo/sim/core/markov_neural_operator_baselines.py`,
  `src/ctreepo/sim/core/markov_changepoint_ops_count.py`,
  `src/ctreepo/sim/core/markov_local_laws.py`, `src/tree/full_tree_ipw.py`, tests.
- `sim/composite_objective`: `src/ctreepo/sim/{local_law_backfill,expectations}.py`,
  `src/ctreepo/sim/core/{leaf_local_mixture_utility,markov_changepoint_ops_count}.py`,
  `src/ctreepo/sim/suite/law_stress_builders.py`,
  `scripts/build_tree_relevant_lda_local_law_cmds.py`, tests.
- `contracts.ObjectiveSpec`: `src/ctreepo/fno_family.py`,
  `src/ctreepo/sim/composite_objective.py`,
  `src/ctreepo/sim/core/markov_changepoint_ops_count.py`,
  `src/ctreepo/sim/local_law_backfill.py`,
  `src/ctreepo/sim/core/leaf_local_mixture_utility.py`,
  `src/ctreepo/distillation.py`, tests. (Note: ThinkingTrees
  `contracts.ObjectiveSpec` uses `n` = root weight, not upstream `root_share`.)

## Bespoke arithmetic flagged (numpy / non-canonical; quarantine or allowlist)

- `src/ctreepo/sim/core/markov_changepoint_ops_count.py` — numpy Monte-Carlo IPW
  (`np.sum(y_s * w_s) / N`, DSL correction). Sim diagnostic; allowlist for now.
- `src/tree/private_sfm_comparison.py`, `src/training/trl_training.py` — unrelated
  numeric code (false-positive on AIPW grep); not local-law arithmetic.
- `src/training/supervision/optimizer_metadata.py` — `base_weight * gamma**depth`
  depth weight; candidate to route through canonical depth discount later.

## Archive mirrors (Phase 7 targets, dead-import scan pending)

- `treepo/src/treepo/_research/ctreepo/*` (sync mirror of ThinkingTrees src;
  partial after the 585→369 prune). Still live in upstream `treepo.methods.*`
  and cannot be archived yet.
- `OLD_treepo_cdx/` (archived 2026-06-25 after import scan found no live callers
  outside docs/tests).
- `src/tree/neural_operator.py` (pure re-export shim — verify).

## Baseline test state (before edits)

- GREEN: `tests/training/test_local_law_torch.py` (3),
  `tests/core/test_local_law_adjustment.py` (7),
  `tests/ctreepo/test_objective_weights.py` (27),
  `tests/ctreepo/test_composite_objective.py` (4) — 41 passed.
- GREEN upstream: `treepo/tests/training/test_local_law.py` (10).
- **PRE-EXISTING FAILURE** (not introduced by this work; FNO/Phase-4 domain):
  `tests/ctreepo/test_neural_operator_baselines.py::TestModelSmoke::`
  `test_tree_fno_shared_feature_local_supervision_bypasses_summary_spec_terms`
  — `AssertionError: shared-feature local supervision should not use summary-spec
  replay`. Flagged to the FNO-owning LLM in `COLLABORATION.md`. 143 passed / 1
  failed in that file; `tests/test_fno_a2_consistency.py` green.

## Guards added this phase

- `tests/ctreepo/test_local_law_source_guard.py` — allowlists the only files
  permitted to *define* canonical local-law arithmetic / root-local mixing, and
  bans imports from `OLD_*`, `treepo_cdx`, and `treepo._research` in `src/`.
