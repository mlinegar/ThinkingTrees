# Test-Suite Drift Ledger — 2026-07-02

Status of the ThinkingTrees test suite after the 2026-07-02 formalization
pass, for whoever ports the remaining tests. Baseline before the pass:
2,492 passed / 124 failed / 56 errors. Fixed this pass:

- **56 setup errors** — `tests/conftest.py` had been replaced by treepo's
  collection guard, orphaning the shared fixtures (`simple_tree`,
  `single_node_tree`, `medium_text`, …). Restored the fixture conftest and
  kept the guard (commit `f922d44610`). All 133 tests in the affected files
  pass.
- **Missing `transformers`** — it lives in the `trl` extra and was absent
  from the venv; installed with `uv pip install` (no full resync, to
  preserve ad-hoc installs). Recovered 2 tests outright; the
  `test_unified_fg_ladder_contract.py` cluster now runs 34/43.
- **Structural-cell legacy aliases** — the sticky-structural-v2 rename
  (`r12_seg10to12` → `r12_p079` etc.) declared `legacy_aliases` in
  `markov_hazard_panels.py` but the pipeline validator never consulted
  them; wired `canonicalize_structural_v2_cell_id` into the
  supervision-recovery cell check in
  `scripts/run_markov_optimization_tradeoff_pipeline.py`. Recovered 3
  tests; legacy configs keep working.

## Remaining failures: contract drift, needs research-intent porting

These tests encode pre-rename contracts. Updating them means porting
expectations to the current schemes, which the owning research thread
should do deliberately (the expected values are claims about experiment
behavior). Counts from the pre-fix inventory; the three fixes above
reduce them somewhat.

| Cluster | Count | Root cause |
| --- | --- | --- |
| `tests/tree/test_markov_optimization_tradeoff_pipeline.py` | 22 | Expected sets/keys still use old cell spellings (`r12_seg10to12__…`) and removed config keys (`task_objective_weight`, `supervision_recovery_tree_family`); several mocks exhaust (`StopIteration`) because the pipeline now makes different calls. |
| `tests/test_pipeline_phases.py` | 11 | `PlanMergeTask` no longer exists in `src/core/batch_orchestrator.py` (current surface: `DocumentState`, `BatchTreeOrchestrator`); tests target the removed batching API. |
| `tests/tree/test_markov_optimization_tradeoffs_report.py` | 10 | Report schema drift (sibling of the pipeline cluster). |
| `tests/tree/test_full_doc_anchor_diagnostics.py` | 10 | Diagnostics return `None` where tests expect `0.0` — default/optional-metric behavior changed. |
| `tests/ctreepo/test_unified_fg_ladder_contract.py` | 9 | Contract assertions fail on substance (observed-row IPW trace expectations) after ladder changes. |
| `tests/tree/test_markov_changepoint_ops_count_simulation.py` | 8 | Simulation output drift. |
| Long tail (28 files) | ~40 | One-to-five failures each; mixture of the above schemes plus isolated API drift. |

Full failing-test list captured at the time of the pass:
`pytest tests/ -q --tb=no -rf` (13 min). The `pytest_ignore_collect` guard
in `tests/conftest.py` still excludes anything importing
`treepo._research`.

## Manifesto pass (same day, second round)

The manifesto suites (`tests/tasks/test_manifesto_*`, `test_phase3_*`) are
now 60 passed / 1 skipped / 1 failed:

- `pyreadr` added to the `manifesto` extra (Benoit .rda readers);
  `transformers` pinned `<5` in the `trl` extra (v5 changes model-loading
  pickle defaults). Both installed in the venv.
- `dspy.load` calls in `run_manifesto_full_doc_dspy_global_f.py` pass
  `allow_pickle=True` (local, self-produced program artifacts).
- `phase3_full_pipeline_optimize._load_scorer_component` handles the
  current `DimensionScorer.save()` layout (top-level `predictor` key)
  alongside the older `scorer`/`score`/`scorer.score` layouts; the
  warm-start test builds its fixture with `save()` so it tracks the format.
- `test_fg_ladder_exports_contract_fit_artifacts` is skipped: its subject
  is the archived `scripts/OLD_build_manifesto_fg_ladder_legacy.py`;
  current fg-ladder runs go through `run_manifesto_fg_real_training_grid.py`.
- Remaining: `test_teacher_fg_leaf_grid_writes_external_summary_bundle_metadata`
  — the external-summary bundle path writes 0 of 4 labeled trees; behavior
  drift in the bundle writer, needs the owning thread.

treepo's own manifesto guard: the full Manifesto Project corpus integration
test passes against v0.1.1
(`TREEPO_RUN_MANIFESTO_PROJECT_FULL=1 TREEPO_MANIFESTO_PROJECT_ROOT=data/raw/manifesto_project_full`).

## Conventions

- Port, don't delete: update expectations to the current schemes when the
  owning thread confirms them; `OLD_`-archive only tests whose subject was
  itself retired.
- Environment: `transformers` is required by the fg-ladder and judge
  suites; it comes from the `trl` extra.
