# Simulation Theory Alignment Status (2026-03-11)

This note records the current theory-facing read on the main paper simulation suites after comparing them against the Lean theorem surface and the family-level expectation checks.

## Canonical artifacts

- Primary suite expectation merge:
  - `outputs/formal_reruns_20260310_062551/paper_reports/primary_suite_expectations.json`
  - `outputs/formal_reruns_20260310_062551/paper_reports/primary_suite_expectations.md`
- Primary suite theory alignment:
  - `outputs/formal_reruns_20260310_062551/paper_reports/primary_suite_theory_alignment.json`
  - `outputs/formal_reruns_20260310_062551/paper_reports/primary_suite_theory_alignment.md`
- Underlying suite reports used in the merge:
  - `outputs/formal_reruns_20260310_062551/paper_reports/cpu_megasweep_expectations.json`
  - `outputs/formal_reruns_20260310_062551/paper_reports/simulation_buildout_expectations.json`
  - `outputs/formal_reruns_20260310_062551/paper_reports/publication_clean_expectations.json`

## Current read

- `mergeable_ablation`: aligned
- `local_law_learnability`: provisionally aligned
- `markov_ops_count`: misaligned
- `segment_lda_ops_weight_recovery`: misaligned
- `segmented_lda_ctreepo`: misaligned

This is the right high-level interpretation:

1. The exact mergeable-sketch subcase is currently the cleanest theorem-aligned simulation family.
2. The local-law learnability layer is promising, but still depends on a partial learnability suite.
3. The main learned / approximate paper families still contain theory-facing monotonicity or separation failures and should not yet be described as fully clean demonstrations of the Lean story.

## Highest-priority mismatches

### Markov OPS Count

The main failures are not the exact ceiling. Those remain in the right place. The failures are in the learned lane:

- learned `root_mae` does not always improve with audit fraction at max train support
- in the buildout slice, learned `root_mae` also fails the expected train-support monotonicity in at least one neural regime
- some merge-level checks only pass at warning level rather than cleanly

Lean anchor:

- `exactSketch_root_distortion_zero`
- `markov_local_laws_from_encoded_feature`
- `not_L3_gFlip`

Interpretation:

- the exact theorem-backed control looks right
- the learned lane is not consistently approaching that control in the expected order
- this is a simulation calibration/slice-selection problem, not a contradiction of the Lean theorem

### Segment-LDA OPS Weight Recovery

This family has the heaviest mismatch burden.

The main recurring failures are:

- ridge `merge_mae` does not always improve with audit fraction at max train support
- some ridge trends with train support also fail
- at least one boundary-sensitive contrast about internal-node labels fails for the online-TensorLDA path

Lean anchor:

- `sketchReduce_countSketch_eq_bagOfWords`
- `ldaDocumentLikelihood_exact_on_tree`
- `affineQuadratic_gap_eq_quadratic_gap`

Interpretation:

- the exact bag-of-words control is still the right theorem anchor
- the learned/inferred estimator paths are not yet giving a clean monotone approximation ladder
- boundary-sensitive slices likely need to be narrowed or re-bucketed by estimator quality before they are paper-clean

### Segmented-LDA C-TreePO

This family currently fails a smaller number of checks, but they are central ones:

- budgeted `root_l1_mean` does not always improve with calibration labels

Lean anchor:

- `dpo_expected_tree_argmin_subset_true_pointwiseEpsilonArgmin_of_stochastic_adaptive_approx_local_laws_with_pointwiseOracleMeasurement`
- `training_path_bundle_epsilon_optimal_with_oracleMeasurement`
- `computeDSLBound_valid_from_joint_interval_event_with_oracleMeasurement_export`

Interpretation:

- the exact oracle-tree control is still the right theorem-facing baseline
- the approximate/audited lane is not yet consistently showing the expected calibration/budget improvement pattern
- this is exactly the family where the paper should be most careful to separate theorem-backed control from empirical approximation behavior

## What is safe to claim now

- The exact mergeable-sketch family is a clean special case of the theorem-backed framework.
- The local-law learnability story has positive evidence and is compatible with the theorem-backed interpretation, but still depends on partially completed suite coverage.
- The broader learned/approximate families are useful evidence and mechanism studies, but they currently need tighter slicing, better calibration, or revised expectation definitions before they should be presented as clean theorem-aligned demonstrations.

## Recommended next cleanup pass

1. Revisit the failing monotonicity expectations for Markov learned `root_mae`.
2. Split Segment-LDA trend checks by estimator regime and by boundary-sensitive vs bag-of-words-safe targets.
3. Re-check the C-TreePO calibration-label trend on the budgeted lane and verify whether the paper-facing slice should be narrowed.
4. Only after those are stable, rerun the combined `report_simulation_theory_alignment.py` pass on the full canonical rerun root.
