# Markov Report Archive

The following legacy Markov report entrypoints are archived and no longer part of
the supported v3 reporting surface:

- `scripts/report_full_doc_anchor_diagnostics_pdf.py`
- `scripts/report_tree_fno_tuning_pdf.py`
- `scripts/report_tree_root_only_parity_pdf.py`
- `scripts/report_markov_partial_capacity_update.py`
- `scripts/report_markov_cohort_compare.py`
- `scripts/run_tree_root_only_parity_diagnosis.py`
- `scripts/report_markov_capability_map.py`
- `scripts/report_markov_changepoint_ops_count_run.py`
- `scripts/report_markov_law_stress.py`
- `scripts/report_markov_local_law_learnability.py`
- `scripts/report_publication_clean_markov_ctreepo_appendix.py`

Supported v3-compatible replacements:

- `scripts/report_markov_optimization_tradeoffs.py`
  Use for the canonical supervision-recovery, family-grid, parity, tradeoff, and
  audit-aware report path.
- `scripts/run_markov_optimization_tradeoff_pipeline.py`
  Use when you want the pipeline to generate the v3 report artifacts end-to-end.
- `scripts/run_markov_publication_bundle.py`
  Use for the publication bundle. Legacy PDF phases were removed from the default
  phase set so the bundle only emits supported v3 artifacts.
- `scripts/report_markov_parity_self_contained.py`
  Use only when you specifically need a self-contained parity appendix sourced
  from parity-grid artifacts. This surface is v3-aware and filters quarantined
  rows from headline views.
- `scripts/report_markov_supervision_recovery_paper_audit.py`
  Use when you want a paper-readiness audit derived from the canonical v3
  supervision-recovery summary and report payloads.
- `scripts/report_tree_oracle_budget_frontier_pdf.py`
  Use only for the dedicated oracle-budget frontier appendix fed by the current
  v3 budget-frontier summary payload.
- `scripts/report_appendix_walkthrough_narrative_deck.py`
  Use for the maintained appendix/deck narrative surface. The older publication-
  clean appendix script is archived.

Operational policy:

- New Markov evidence should come from v3 outputs carrying modern provenance.
- Legacy report CLIs are hard-disabled to avoid silently producing non-canonical
  figures.
- Historical implementations remain recoverable through git history if needed
  for archaeology.
