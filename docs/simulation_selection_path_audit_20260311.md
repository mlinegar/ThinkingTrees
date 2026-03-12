# Simulation Selection-Path Audit (2026-03-11)

This note records which simulation families now use actual held-out checkpoint
selection and which only expose explicit no-validation semantics.

## Validation-backed selection now in place

- `src/ctreepo/sim/core/markov_changepoint_ops_count.py`
  - `val_docs > 0`: after each epoch, evaluate the exact held-out optimization
    objective, restore the best epoch, and emit `training_selection_*` fields.
  - `val_docs = 0`: emit `training_selection_mode = final_epoch_no_validation`.

- `src/ctreepo/sim/core/lda_tree_recovery_learned.py`
  - `full_doc_operator` now restores the best checkpoint by held-out MSE when
    a validation split exists.
  - Metadata fields: `selection_mode`, `selection_split`,
    `selection_metric_name`, `selection_metric_value`, `best_epoch`.

- `src/ctreepo/sim/core/lda_tree_utility_vector.py`
  - `full_doc_mlp_diag` now restores the best checkpoint by held-out MSE when
    a validation split exists.
  - Metadata fields mirror the learned LDA recovery family.

## Explicit no-validation semantics now in place

- `src/ctreepo/sim/core/segment_lda_ops_weight_recovery.py`
  - neural topic refinement still trains without a validation split, but now
    reports `final_step_no_validation` instead of implying held-out selection.

- `src/ctreepo/sim/core/segmented_lda_ctreepo.py`
  - `leaf_theta_estimator="mlp"` now reports
    `selection_mode = final_epoch_no_validation`.
  - `leaf_theta_estimator="rf"` now reports
    `selection_mode = rf_fit_no_validation`.

## Still missing a true held-out selection path

- `src/ctreepo/sim/core/exact_utility_common.py`
  - `train_neural_tree_policy`
  - `train_flat_policy`
  - `train_flat_span_policy`

These preference-policy trainers still train on the supplied `train_docs`
without an internal validation split or checkpoint-selection step. They do not
currently surface selection metadata because they only return models.

If we want the same training-selection guarantees there, the next step is to
thread an explicit `val_docs` split and a held-out objective evaluator through
the exact-utility simulation stack.
