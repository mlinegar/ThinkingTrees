# Markov Alignment Audit

This note defines the strict semantic alignment surface for the Markov simulations used in the C-TreePO paper-facing diagnostics.

## Paper-To-Lean Local-Law Mapping

- paper C1 = Lean L1
- paper C2 = Lean L3
- paper C3 = Lean L2

These are the only theorem-facing local laws in the Markov full-doc alignment story. Schedule-consistency and schedule-spread diagnostics remain proxy-only.

## Surface Distinction

The Markov full-doc normalized task/local-law weighting surface is identified in emitted metadata as:

- `markov_full_doc_normalized_task_local_law_surface`

It is not the same objective family as the separate TreePO regularized objective note.
It is not the same objective family as the TreePO Regularized Objective note:

- `treepo_regularized_objective`

The Markov full-doc surface is an empirical full-document objective used for the full-doc tree/FNO simulations, parity runs, capacity-tagged runs, and budget-share runs. The TreePO Regularized Objective note describes a separate formal/empirical objective surface and must not be cited as if it were the same weighting family.

## Surface Semantics

- `markov_observed_token`
  Underlying problem: observed-token root-count prediction with sampled local-law supervision.
  Status: mixed theorem anchors plus approximate/audited learned operators.
- `markov_full_doc_anchor_diagnostics`
  Underlying problem: paper-facing fixed-bundle full-document learning, with theorem-facing C1/C2/C3 totals only for the tree-local-law baselines.
  Status: approximate/audited or proxy-only publication baselines.
- `markov_full_doc_anchor_ladder`
  Underlying problem: provenance and reproduction alignment for the doc-sequence full-doc baseline.
  Status: proxy-only provenance surface.
- `markov_full_tree_ipw_grid`
  Underlying problem: realized full-tree node mean loss under Bernoulli realized-node sampling with naive/HT/Hajek point estimators.
  Status: approximate/audited estimand diagnostic; point-estimation only unless an explicit CI wrapper is added.

## Supervision And Proxy Discipline

- Theorem-facing local-law totals include only C1/L1, C2/L3, and C3/L2.
- For the changepoint-count Markov task, the primary runtime C2 diagnostic is
  `c2_count_drift_r1_mae`: one-step re-summary should not change the predicted
  changepoint count. The legacy `c2_idempotence_mae` field is retained only as
  a compatibility alias for that same quantity.
- For the exact-sketch lane, decoded `(count, first, last)` recovery is treated
  as a Markov sufficiency witness, not just a codec convenience. The Lean
  theorem `markov_count_query_sufficient_has_decoder` says that any summary
  sufficient for all two-sided changepoint-count queries admits a decoder back
  to an equivalent exact sketch.
- `c2_on_range_exact_match` is a stricter exact-sketch witness on decoded
  `(count, first, last)` replay, not the default runtime Markov C2 score.
- `c2_state_replay_mse` is proxy-only and is used to debug latent replay
  instability rather than theorem-facing correctness.
- The failure-attribution field
  `exact_sketch_markov_sufficiency_gap_score` is the task-facing alias for the
  existing exact-sketch decode gap: large values mean the learned summary is
  failing an empirical sufficiency witness for the theorem-domain Markov sketch.
- `schedule_consistency` and `schedule spread` are associativity proxies only.
- In the budget-share surface, `root_only` and `doc_sequence` are two consumption modes of the same paid full-document label.
- In the full-tree IPW grid, the document channel is `always_observed_document_top_loss` and the node channel is `sampled_realized_tree_nodes` with `unit_propensity`.
