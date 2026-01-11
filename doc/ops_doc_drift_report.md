# Docs / CLI Drift Report (OPT)

This report lists mismatches between documentation and the current code layout.
Fixing these will make the package shippable without tribal knowledge.

## High Priority (User-facing drift)

### 1) `docs/ARTIFACTS.md`
- Describes `python main.py train|infer|audit` but `main.py` only builds a tree.
- Training entrypoint is `src/training/run_pipeline.py` (or `scripts/run_training_pipeline.sh`).

### 2) `doc/oracle_preference_training.md`
- References `src/ops_engine/...` which no longer exists.
- Actual modules live in `src/training/preference/` and `src/training/judges/`.

### 3) `doc/OPTIMIZATION_GUIDE.md`
- Uses `src/ops_engine/...` paths; should map to `src/training/...` and `src/core/strategy.py`.

### 4) `README.md`
- Mostly accurate, but should point to the preferred CLI entrypoints (shell script vs module).

## Medium Priority (Internal drift)

### 5) `doc/architecture.md`
- Old paths (`src/ops_engine`, `src/core/scorers.py`, etc.).
- Conflicts with `ARCHITECTURE.md` which reflects the current layout.

### 6) `doc/test_plan.md` and `doc/testing_documentation.md`
- Reference `tests/ops_engine/*` which no longer exists.
- Should point to `tests/tree/`, `tests/core/`, `tests/training/` etc.

### 7) `doc/GLOSSARY.md`
- Old module locations for builder/auditor/training.

## Low Priority / Archive

### 8) `doc/gemini_implementation_plan.txt`
- Early design plan with old module names.
- Should be archived or moved to `doc/old/`.

## Recommended Fix Strategy

1) Choose a single doc root (`docs/` or `doc/`).
2) Update paths to current layout (remove `ops_engine` references).
3) Align CLI descriptions with:
   - `scripts/run_training_pipeline.sh`
   - `python -m src.training.run_pipeline`
   - `python main.py` for tree building only
4) Deprecate or move stale design docs to `doc/old/`.

