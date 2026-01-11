# OPT Split Plan (Draft)

## Goals
- Make OPT shippable as a coherent project with clear boundaries.
- Reduce cross-coupling between code, proof, and paper.
- Keep each artifact buildable and releasable on its own.

## Proposed Repository Split

### 1) Python package (runtime + training)
**Working name**: `thinking-trees` (keeps existing pyproject name)

**Contents**
- `src/` (core, tree, training, tasks, datasets)
- `scripts/` (server + pipeline entrypoints)
- `config/` (runtime settings)
- `tests/`
- `README.md`, `LICENSE`, `CHANGELOG.md`, `pyproject.toml`

**Optional add-ons**
- `experiments/` (keep if you want repro scripts in-repo)
- `docs/` (only if you want package-level docs here)

### 2) Lean formalization
**Working name**: `opt-formal` or `formalproofs`

**Contents**
- Move `lean3/` to repo root
- Keep `FormalProofs/` and `FormalProofs.lean`
- `lakefile.toml`, `lean-toolchain`, `README.md`

**Notes**
- Keep `Deprecated/` or move it to a separate branch/tag if you want a clean build.
- Single axiom in `FormalProofs/OPT/PreferenceBounds.lean` should be documented in repo README.

### 3) Paper repo
**Working name**: `opt-paper` or `oracle-preference-training`

**Contents**
- `main.tex` and `main.pdf`
- `refs.bib` (currently in `doc/old/refs.bib`)
- figures, tables, and any paper-specific scripts

**Notes**
- This repo should cite both the Python and Lean repos.

### Optional 4) Data + experiments
If you want to keep datasets/results separate, split:
- `data/`, `outputs/`, `logs/`, `experiments/` into a private or internal repo
- This avoids bloating the release artifacts.

---

## Mapping From Current Tree

- Python package
  - `src/` -> python repo
  - `scripts/` -> python repo
  - `config/` -> python repo
  - `tests/` -> python repo
  - `README.md` -> python repo (package-level)

- Lean
  - `lean3/` -> lean repo root
  - `lean/`, `lean2/`, `lean3_backup/` -> drop or archive

- Paper
  - `doc/main.tex` -> paper repo
  - `doc/main.pdf` and/or `main.pdf` -> paper repo
  - `doc/old/refs.bib` -> paper repo as `refs.bib`

---

## Migration Sequence (Low-Risk)

### Phase 0: Freeze and tag
- Tag current monorepo as `opt-pre-split` so the state is recoverable.

### Phase 1: Extract Python package
- Copy `src/`, `scripts/`, `config/`, `tests/`, `README.md`, `pyproject.toml`.
- Update package imports if you change the top-level module name.
- Add CI for unit tests and a basic smoke run.

### Phase 2: Extract Lean
- Move `lean3/` to its own repo.
- Confirm `lake build FormalProofs` works.
- Add CI with `lake build FormalProofs`.

### Phase 3: Extract paper
- Move `doc/main.tex` and `doc/old/refs.bib`.
- Rename `refs.bib` at repo root and verify `pdflatex` build.

### Phase 4: Cross-link
- Add cross-repo links in each README (paper -> code -> proofs).
- Ensure citation info (`CITATION.cff`) exists in Python repo.

---

## Decisions Needed (Before The Split)
- Final repo names and GitHub org/user.
- Python import name (`thinking_trees` vs `thinkingtrees` vs `ops`).
- Whether to keep `docs/` in the Python repo or split it into the paper repo.
- Whether `experiments/` stays with code or moves to data repo.

---

## Risks And Mitigations
- **Doc drift**: choose one canonical doc folder and delete/move the rest.
- **Proof status claims**: align README + paper + Lean docs on axioms and any `sorry`.
- **Import churn**: avoid renaming `src` imports unless you must.

