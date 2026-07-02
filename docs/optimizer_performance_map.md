# Optimizer Performance Map

This note separates three different claims that now coexist in the repo:

1. Lean certifies transfer of good surrogate optimization to the target objective.
2. The DSPy runtime audit records what each optimizer wrapper actually did.
3. The recoverable Markov diagnostics explain whether a failure is about information, objective choice, optimization, representation, or an implementation path.

## What Lean Certifies

Lean does not prove that one DSPy teleprompter is a better search procedure than another.
It proves optimizer-output transfer statements of the form:

- if `θ̂` is an `ε`-minimizer of a surrogate objective, and
- the surrogate objective is uniformly within `δ` of the target objective,
- then `θ̂` is an `(ε + 2δ)`-minimizer of the target objective.

Public aliases:

- `MainTheorems.surrogate_optimizer_certificate_uniform_transfer`
- `MainTheorems.oracle_measurable_surrogate_optimizer_certificate_uniform_transfer`
- `MainTheorems.oracle_measurable_surrogate_optimizer_certificate_two_stage_uniform_transfer`

Implementation:

- `lean3/FormalProofs/OPT/OptimizationPerturbation.lean`
- `lean3/FormalProofs/OPT/MainTheorems.lean`

The Lean surface is algorithm-agnostic by design. It certifies transfer once a runtime optimizer has produced a surrogate certificate. It does not certify GEPA, MIPRO, bootstrap, or labeled few-shot as search algorithms.

## What The Runtime Audit Records

Every scorer, leaf summarizer, merge summarizer, and comparison-module optimization attempt now emits a normalized run record.

Saved surfaces:

- `final_stats.json` under `optimizer_diagnostics`
- `optimizer_audit_manifest.json` for each pipeline run
- `optimizer_audit_manifest.json` and `optimizer_performance_summary.{json,md}` for audit-grid sweeps

Primary fields:

- `optimizer_requested`, `optimizer_used`
- `component`, `dataset_size`, `dataset_regime`, `budget_mode`, `seed`
- `compile_status ∈ {completed, skipped, fallback, noop, failed}`
- `skip_reason`, `fallback_reason`, `exception_summary`
- `metric_before`, `metric_after`, `heldout_gain`, `train_gain`
- `input_mutation_flags`
- `comparison_control_flag`

Current explicit semantic mismatches:

- `bootstrap_random_search` may fall back to basic bootstrap if the DSPy teleprompter is unavailable.
- `labeled_fewshot` can surface as `noop` when the teleprompter is unavailable.
- `mipro` records train/val example compaction, including truncation or optional-field dropping.
- comparison-module training is always GEPA and is reported as `forced_control`, not ranked with the phase-2 optimizer families.

Cell-level classification rules are implemented in `src/training/optimization/performance.py`.

## How To Read The Markov Recoverable Diagnostics

The recoverable full-document lane now reports:

- `gap_to_ridge_control`
- `gap_to_exact_witness`
- `train_val_gap`
- `val_test_gap`
- `selection_metric_curve_summary`
- backend/device metadata
- `objective_variant`
- `cause_code`

Cause-code interpretation:

- `information_barrier`: only when the exact witness and ridge control also fail on the same bundle.
- `objective_mismatch`: auxiliary objective variants underperform the simpler primary objective on matched settings.
- `optimization_limit`: train-side fit exists and more data measurably narrows the held-out gap.
- `representation_limit`: both 1x and larger-scale runs remain far above the witness without meaningful fit.
- `implementation_path_issue`: backend, device, or fallback behavior correlates with degradation.

Interpretation boundary:

- exact Markov-state theorems show the task is recoverable on the right state surface;
- count-only counterexamples show some summaries cannot work in principle;
- when `ridge_control` is near exact on the recoverable bundle, poor neural performance is not an information-loss excuse.

## Pre-Rerun Lean Validation Gates

Before rerunning theorem-facing Markov simulations, the intended contract is now:

- Exact-collapse / one-leaf lane:
  certify a sound one-leaf policy and use the exact-state support theorem
  `markov_path_changepoint_count_exact_on_support_of_contract`.
- Real-topology exact lane:
  certify a sound sampled tree policy and use the same support theorem, since it
  is topology-agnostic once `S T = x` holds on support.
- Count-only controls:
  treat them as negative controls only.
  Lean now records this explicitly via
  `markov_countOnly_not_exact_on_all_trees`.
- Approximate local-law topology lanes:
  require checked runtime nodewise audit artifacts.
  Lean then compiles them to stochastic adaptive approximate local laws via
  `runtime_audited_markov_path_stochastic_approx_local_laws`.

Relevant files:

- `lean3/FormalProofs/OPT/MarkovSimulationValidation.lean`
- `lean3/FormalProofs/DSL/RuntimeCertificates.lean`
- `lean3/FormalProofs/OPT/MainTheorems.lean`

## Reporting Entry Points

DSPy optimizer sweep:

- `scripts/run_optimizer_performance_audit.py`
- `scripts/report_optimizer_performance_audit.py`

Recoverable Markov diagnostics:

- `src/ctreepo/sim/core/full_doc_anchor_diagnostics.py`

The optimizer report produces:

- a DSPy optimizer matrix by component and dataset regime
- an optional Markov witness-gap table by model family
