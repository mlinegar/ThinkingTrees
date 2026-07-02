# Local-Law Phase 0 Inventory - 2026-06-25

Scope: first implementation pass for `docs/local_law_single_path_master_plan.md`.
This inventory was intentionally focused on the foundation files touched by the
pass, not a replacement for `docs/ctreepo_python_code_map_for_llms.md`.

## Canonical Owners

- Tensor/scalar training rows and sampled objective arithmetic:
  `/home/mlinegar/treepo/src/treepo/training/local_law.py`.
- Strict audit rows and influence-weighted audit overlap:
  `/home/mlinegar/treepo/src/treepo/local_law.py`.
- Root/local objective resolution and public upstream objective schema:
  `/home/mlinegar/treepo/src/treepo/objective.py`.

## ThinkingTrees Compatibility Surfaces

- `src/core/local_law_adjustment.py`: now a delegating shim over
  `treepo.training.local_law`; retains old scalar names.
- `src/training/supervision/local_law_torch.py`: now a delegating shim over
  `treepo.training.local_law`; retains old torch names.
- `src/ctreepo/sim/composite_objective.py`: keeps sim-specific
  `CompositeObjectiveSpec` and evaluation helpers; imports the upstream
  `ResolvedObjectiveWeights` and `resolve_root_local_objective_weights`.
- `src/ctreepo/local_law_rows.py`: new opt-in adapter that builds
  `LocalLawTrainingRow` instances from repo-specific tree/trace nodes.

## Live Import Graph Notes

Initial focused `rg` scan found live ThinkingTrees imports of:

- `src.core.local_law_adjustment` from IPW helpers, Markov local-law builders,
  training pipeline helpers, and tests.
- `src.training.supervision.local_law_torch` from FNO/Markov neural-operator
  paths, `tree_model_v2_trainer`, diagnostic scripts, and tests.
- `src.ctreepo.sim.composite_objective.resolve_root_local_objective_weights`
  from Markov, LDA, law-stress, and objective-weight tests.

Because these are live, this pass used shims/delegation instead of archive or
delete.

Follow-up import-collapse pass:

- Live `src/` and `scripts/` imports of `src.core.local_law_adjustment` and
  `src.training.supervision.local_law_torch` now import
  `treepo.training.local_law` directly.
- Remaining imports of those shims are compatibility tests only.
- `treepo_cdx/` had no live imports outside docs/tests and was archived to
  `OLD_treepo_cdx/`.
- `/home/mlinegar/treepo/src/treepo/_research` remains live and cannot be
  archived yet because upstream `treepo.methods.*`, `treepo.learning`, and tests
  still import it.

## Guard Status

- Added `tests/ctreepo/test_local_law_source_guards.py` to pin the two
  ThinkingTrees local-law modules as delegating shims and keep the new row
  adapter free of `treepo_cdx`, `OLD_*`, and `_research` imports.
- Strengthened the same guard so live `src/` and `scripts/` code cannot import
  the old ThinkingTrees local-law shims outside `src/core/__init__.py` and the
  shim files themselves.
- Broad archive-import guards are deferred until Phase 7 because the current
  upstream `/home/mlinegar/treepo` package still has live `_research` imports.

## Baseline Test Target

Focused suites for this pass:

- `/home/mlinegar/treepo/tests/training/test_local_law.py`
- `/home/mlinegar/treepo/tests/test_unified_contracts.py`
- `tests/training/test_local_law_torch.py`
- `tests/core/test_local_law_adjustment.py`
- `tests/ctreepo/test_objective_weights.py`
- `tests/ctreepo/test_composite_objective.py`
- `tests/ctreepo/test_local_law_rows.py`
- `tests/ctreepo/test_local_law_source_guards.py`
