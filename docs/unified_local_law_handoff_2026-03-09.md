# Unified Local-Law Handoff (2026-03-09)

## Goal

The current paper-facing objective is a single simulation story across the two main DGP families:

- Markov changepoint / ops-count
- tree-relevant LDA / local-mixture utility

The intended claim is:

> when local laws (C1/C2/C3) hold strongly enough, we can learn a DGP-specific oracle summary function `g`, and the learned `g` improves the downstream/oracle target relative to a matched baseline.

This repo now has two layers that matter:

1. a unified **local-law learnability** protocol / manifest / backfill / report stack
2. a shared **law-stress** classification layer that tries to summarize when the learned `g` actually beats the baseline and which laws improved


## What Exists Now

### 1. Unified local-law learnability protocol

This was already implemented before the most recent law-stress pass.

Key files:

- `src/ctreepo/sim/local_law_learnability.py`
- `src/ctreepo/sim/local_law_backfill.py`
- `src/ctreepo/sim/local_law_report_common.py`
- `src/ctreepo/sim/expectations.py`
- `scripts/organize_existing_local_law_runs.py`
- `scripts/report_local_law_meta.py`

What this does:

- defines a common additive `local_law_learnability` schema for Markov and LDA
- serializes `oracle_g`, `baseline_g`, `learned_g`, candidate policies, support budgets, split IDs, selection info, thresholds, and artifact references
- backfills older pre-unification JSONs into that schema in memory
- lets the shared report and expectation code consume both legacy and new outputs through one interface

Current orchestration for the paper-facing learnability sweeps is now also centralized:

- `src/ctreepo/sim/suite/identifiable_zero_policy.py`
- `src/ctreepo/sim/suite/identifiable_zero.py`
- `src/ctreepo/sim/suite/identifiable_zero_publication_policy.py`
- `src/ctreepo/sim/suite/identifiable_zero_publication.py`
- `src/ctreepo/sim/suite/publication_policy.py`
- `src/ctreepo/sim/suite/publication_lanes.py`
- `src/ctreepo/sim/suite/publication_ctreepo.py`
- `src/ctreepo/sim/suite/learnability_policy.py`
- `src/ctreepo/sim/suite/identifiable_zero_learnability.py`
- `venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-learnability ...`
- `venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero ...`
- `venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-publication --profile publication_clean ...`
- `venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-publication --profile longrun_equiv_v1 ...`

That suite layer is now the canonical way to build and run the paper-facing identifiable-zero families.
It keeps the Markov and C-TreePO train-doc grids, label-rate grids, held-out counts, lane definitions,
and seed sets synchronized, and it carries the matched no-tree baselines through the canonical APIs:

- Markov doc-level baseline (`--include-doc-level-baseline`)
- C-TreePO full-document theta baseline (`--include-full-doc-theta-baseline`)

The same suite layer now also owns the remaining publication-facing identifiable-zero packs:

- `venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-publication --profile publication_clean ...`
- `venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-publication --profile longrun_equiv_v1 ...`

Those two profiles replace duplicated builder logic for:

- publication-clean CPU/GPU packs
- longrun oracle-equivalence / scale packs
- bounded pilot-manifest derivation for scheduler calibration

The legacy publication-clean and longrun builder scripts are now retired migration stubs; the shared suite is the only canonical entrypoint.

Recommended entrypoint:

```bash
venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-learnability build \
  --output-root outputs/identifiable_zero_learnability_v1_<stamp>

venv/bin/python -m src.ctreepo.cli sim suite identifiable-zero-learnability run \
  --output-root outputs/identifiable_zero_learnability_v1_<stamp> \
  --jobs 64 \
  --gpu-tokens auto
```

The same `ctreepo sim suite ...` layer now also has a unified law-stress entrypoint:

```bash
venv/bin/python -m src.ctreepo.cli sim suite law-stress build \
  --output-root outputs/law_stress_suite_<stamp> \
  --groups "markov_sanity_suite lda_sanity_suite"

venv/bin/python -m src.ctreepo.cli sim suite law-stress run \
  --output-root outputs/law_stress_suite_<stamp> \
  --jobs 32 \
  --gpu-tokens auto
```

The Markov and LDA law-stress script builders are now thin wrappers over:

- `src/ctreepo/sim/suite/law_stress_builders.py`

### 2. Shared law-stress classification

Key files:

- `src/ctreepo/sim/core/law_stress_common.py`
- `src/ctreepo/sim/core/lda_law_stress.py`
- `scripts/report_cross_dgp_law_stress.py`

Important semantic choice:

- `primary_pass` is the success criterion
- local laws C1/C2/C3 are diagnostics / mechanism indicators
- `bundle_full_success` is now only a backward-compatible alias for primary success, not “primary + all laws”
- `bundle_status` is one of:
  - `full_success`
  - `primary_only`
  - `laws_only`
  - `failure`

That is a meaningful conceptual change relative to the older Markov “bundle” language.


## What Changed In This Session

Two correctness fixes were made to the new law-stress layer.

### Fix 1. Markov law-stress is now matched against `root_only`, not `undersupported`

The previous version had a serious comparability bug:

- the Markov runner wrote `metrics["law_stress"]` by comparing the learned package against `undersupported`
- that was not the same baseline used by the unified learnability protocol, which treats `root_only` as the matched baseline
- this produced misleading cross-DGP aggregates

What changed:

- `src/ctreepo/sim/core/markov_changepoint_ops_count.py`
  - the old single-run comparison was renamed to `metrics["diagnostic_law_stress_vs_undersupported"]`
  - it is now explicitly diagnostic only
- `src/ctreepo/sim/local_law_backfill.py`
  - added `collect_law_stress_assessments(...)`
  - for Markov, it now pairs non-`root_only` runs against matched `root_only` runs across summaries using scenario/config keys
  - canonical cross-DGP law-stress now comes from that paired comparison, not from the old in-run `undersupported` comparison
- `src/ctreepo/sim/local_law_report_common.py`
  - shared report core now uses the paired law-stress collection path
- `scripts/report_cross_dgp_law_stress.py`
  - unified/manifest mode now uses the paired collection path too

### Fix 2. LDA law-stress now uses the selected `learned_g`, not the first candidate

The previous bridge could silently score the wrong candidate.

Specifically:

- `local_law.law_stress` in LDA stores per-policy assessments
- the old bridge often grabbed the first policy in the dict
- the unified summary fallback also often took the first `candidate_g`
- this could score `law_calibrated_naive` instead of the selected `learned_g`

What changed:

- `src/ctreepo/sim/local_law_backfill.py`
  - `compute_law_stress_for_summary(...)` now:
    - prefers `learned_g`
    - otherwise uses `selection.selected_candidate`
    - only then falls back to a generic candidate
  - if raw LDA payload already contains `local_law.law_stress`, it now selects the entry for the chosen candidate instead of the first entry

Additional bug fixed while doing this:

- LDA primary law-stress classification now uses downstream `oracle_target_abs_error`
- the older fallback path was incorrectly using local `root_error` (`mean_root_c3_error`) in some cases


## Current Artifacts

### Inventory / manifest

Primary unified inventory root:

- `outputs/existing_local_law_inventory_20260309`

Important files:

- `outputs/existing_local_law_inventory_20260309/existing_local_law_manifest.jsonl`
- `outputs/existing_local_law_inventory_20260309/existing_local_law_inventory_summary.json`
- `outputs/existing_local_law_inventory_20260309/existing_local_law_inventory.md`

Current primary-manifest totals:

- included roots: `3`
- included config-json runs: `15,910`
- direct unified runs: `6,761`
- backfilled legacy runs: `9,149`

Primary included roots:

- `outputs/markov_law_stress_20260308_qa_smoke`
- `outputs/tree_relevant_lda_local_law_20260308_210436`
- `outputs/tree_relevant_lda_local_law_smoke_20260308`

Exploratory Markov roots are still excluded from the primary manifest by default, but can be promoted with `--include-root`.

### Meta report

- `outputs/existing_local_law_inventory_20260309/meta_report/local_law_meta_report.pdf`
- `outputs/existing_local_law_inventory_20260309/meta_report/local_law_meta_report.md`
- `outputs/existing_local_law_inventory_20260309/meta_report/local_law_meta_report_summary.json`

### Cross-DGP law-stress report

This was regenerated after the two fixes above.

- `outputs/existing_local_law_inventory_20260309/cross_dgp_report/cross_dgp_law_stress_report.pdf`
- `outputs/existing_local_law_inventory_20260309/cross_dgp_report/cross_dgp_law_stress_table.txt`
- `outputs/existing_local_law_inventory_20260309/cross_dgp_report/cross_dgp_law_stress_summary.json`


## Current Cross-DGP Numbers (Corrected)

The corrected report was built with:

- manifest:
  - `outputs/existing_local_law_inventory_20260309/existing_local_law_manifest.jsonl`
- extra exploratory Markov root:
  - `outputs/markov_local_law_journal_suite_20260308_063826`

Current summary table:

```text
DGP                                 Package                N  Prim%   C1%   C2%   C3%  Laws  PrimGain
-----------------------------------------------------------------------------------------------------
markov_ops_count                    all_laws              81  0.0% 100%   0%  75%   1.8    -7.4%
markov_ops_count                    all_laws_plus_sched     8  0.0% 100%   0% 100%   2.0   -11.8%
markov_ops_count                    c1_only                2  0.0% 100%   0%  50%   1.5     3.2%
markov_ops_count                    c1c3                   6  0.0% 100%   0% 100%   2.0   -16.9%
markov_ops_count                    c2_only                6  0.0%  17% 100%  83%   2.0     0.9%
markov_ops_count                    c3_only                2  0.0%   0%   0% 100%   1.0   -16.0%
markov_ops_count                    sched_only             1  0.0%   0%   0%   0%   0.0     1.5%
tree_relevant_lda_local_law         all_laws           15651 19.6%  36%  82%  36%   1.4     7.2%
```

Interpretation:

- the earlier “Markov all_laws: ~98% primary pass” number is obsolete and was driven by the bad `undersupported` baseline comparison
- the corrected matched-baseline Markov comparison is much harsher
- the comparable Markov sample is also much smaller (`81` matched `all_laws` comparisons), because most of the reusable Markov material is older exploratory data and does not come pre-organized as matched `root_only` / learned pairs in the main manifest


## Current Scientific State

### LDA

LDA is the stronger story operationally right now:

- huge existing run volume in `tree_relevant_lda_local_law_20260308_210436`
- unified protocol/backfill works
- selected-policy law-stress is now scored correctly
- current corrected cross-DGP aggregate:
  - `all_laws`
  - `N = 15,651`
  - `primary_pass_rate = 19.6%`
  - `C1 pass = 36%`
  - `C2 pass = 82%`
  - `C3 pass = 36%`
  - `mean primary gain = 7.2%`

### Markov

Markov is more mixed right now:

- the new law-stress stack itself exists and has smoke outputs
- the big reusable root that materially changes cross-DGP coverage is still the older exploratory journal root:
  - `outputs/markov_local_law_journal_suite_20260308_063826`
- once matched against `root_only`, the current comparable Markov cross-DGP table no longer shows broad primary success

That does **not** necessarily mean the Markov program is scientifically dead. It means:

- the old exploratory journal root does not support the stronger cross-DGP claim under the corrected matched-baseline definition
- and/or the comparable matched coverage is too sparse / too old / too heterogeneous to trust as the main paper dataset

In practice, if the paper needs a clean cross-DGP Markov comparison under the new protocol, it likely needs fresh matched law-stress runs rather than relying on the old exploratory journal root.


## Important Open Issues

### 1. LDA `lambda=0` null-control anomaly is still unresolved

This remains the main substantive unresolved issue.

Symptoms:

- the unified meta report still flags the LDA `lambda=0` null-control as failing
- representative file:
  - `outputs/tree_relevant_lda_local_law_20260308_210436/results/suite_a_exact_controls/mode_aligned/tau_1/lam_0/seed_0.json`
- in that regime, `oracle_true_summary` has near-zero oracle-target absolute error, but `mean_aux_oracle_target_delta` is still materially non-zero

Why this matters:

- under the theorem-facing story, `lambda=0` should largely kill the downstream relevance of better local summaries
- the current emitted `Delta` metric does not yet cleanly reflect that expectation

Likely investigation target:

- `src/ctreepo/sim/core/leaf_local_mixture_utility.py`
- especially the downstream utility / pooled-vs-local delta path

### 2. Markov comparable coverage is limited

The corrected cross-DGP Markov table is now comparable, but sparse.

Current reality:

- primary inventory manifest includes only the law-stress smoke root for Markov
- richer Markov coverage in the cross-DGP report comes from explicitly adding:
  - `outputs/markov_local_law_journal_suite_20260308_063826`
- after matched pairing, only a modest subset of those runs are actually comparable

If a strong Markov cross-DGP story is needed, the likely fix is not more backfill code. It is a fresh matched run plan.

### 3. No dedicated tests yet for `report_cross_dgp_law_stress.py`

Relevant logic is indirectly covered through protocol/report tests, but there is no dedicated smoke test for:

- cross-DGP report generation
- paired Markov baseline aggregation
- corrected row counts / table emission


## What Is Already Implemented On The LDA Side

An earlier summary said “Phases B–F not started.” That is no longer true.

Already implemented in code:

- `law_package` config surface in LDA
- `exact_family` config surface in LDA
- LDA exact-family calibrators in:
  - `src/ctreepo/sim/core/lda_law_stress.py`
- LDA `local_law.law_stress` emission in:
  - `src/ctreepo/sim/core/leaf_local_mixture_utility.py`
- CLI flags for those fields in:
  - `scripts/run_leaf_local_mixture_utility_simulation.py`

What is still missing is not the basic code surface. It is:

- clean theorem-facing interpretation
- stronger testing around the new law-stress layer
- a settled paper-facing report structure around those fields


## Verification Completed

Targeted verification run after the fixes:

```bash
source venv/bin/activate
PYTHONPATH=. pytest -q \
  tests/tree/test_markov_law_stress_report.py \
  tests/ctreepo/test_leaf_local_mixture_utility_local_law.py \
  tests/ctreepo/test_local_law_learnability_protocol.py
```

Result:

- `17 passed`

Also verified:

- `venv/bin/python -m py_compile` on the touched shared/report files passed
- the corrected cross-DGP report regenerated successfully


## Regeneration Commands

### Rebuild inventory manifest

```bash
source venv/bin/activate
PYTHONPATH=. python scripts/organize_existing_local_law_runs.py \
  --output-dir outputs/existing_local_law_inventory_20260309
```

### Rebuild unified meta report

```bash
source venv/bin/activate
PYTHONPATH=. python scripts/report_local_law_meta.py \
  --manifest outputs/existing_local_law_inventory_20260309/existing_local_law_manifest.jsonl \
  --output-dir outputs/existing_local_law_inventory_20260309/meta_report
```

### Rebuild corrected cross-DGP report

```bash
source venv/bin/activate
PYTHONPATH=. python scripts/report_cross_dgp_law_stress.py \
  --manifest outputs/existing_local_law_inventory_20260309/existing_local_law_manifest.jsonl \
  --unified-root outputs/markov_local_law_journal_suite_20260308_063826 \
  --output-dir outputs/existing_local_law_inventory_20260309/cross_dgp_report
```


## Recommended Next Steps

1. Investigate the LDA `lambda=0` null-control failure in `leaf_local_mixture_utility.py`.
2. Decide whether to trust the old exploratory Markov journal root at all for paper-facing claims under the corrected matched-baseline protocol.
3. If Markov needs to support the same cross-DGP story, run fresh matched `root_only` vs learned-package law-stress suites rather than leaning on the old exploratory root.
4. Add a direct smoke test for `scripts/report_cross_dgp_law_stress.py`.
5. Only after the above, update manuscript-facing summaries / paper figures with the corrected cross-DGP numbers.


## Repo State Reminder

The repo is still dirty.

- Do not assume untracked/modified files are safe to clean.
- The worktree contains many unrelated changes.
- Avoid destructive cleanup.
