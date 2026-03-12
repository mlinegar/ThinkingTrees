# Tree-Relevant LDA Coarse Leaf Publication Note (2026-03-09)

This note records the current publication-facing status of the coarse leaf-size extension to the tree-relevant LDA follow-up.

## Main Result

The coarse-size extension resolves the open question left by the March 7 follow-up:

- increasing leaf size helps through `50%` of the document (`192` tokens)
- the benefit does **not** continue to the whole document
- the `100%` case (`384` tokens) is the one-leaf pooled control, so its `Delta` is exactly `0`

The strongest paper-facing statement is therefore:

> per-section analysis benefits from coarser sections up to a practical optimum, but that benefit saturates before the whole document; in this family the best coarse point is `50%`, while the `100%` one-leaf case collapses exactly to pooling.

## Key Thresholds

At `lambda=2`, `doc_topic_concentration=0.6`, `train_docs=512`:

- `4%` leaves (`16` tokens): last positive `tau = 1`
- `8%` leaves (`32` tokens): last positive `tau = 2`
- `17%` leaves (`64` tokens): last positive `tau = 4`
- `25%` leaves (`96` tokens): last positive `tau = 8`
- `50%` leaves (`192` tokens): last positive `tau = 16`
- `100%` leaves (`384` tokens): never positive; identical to pooling

For the coarse-only lambda onset:

- at `tau=1`, the onset improves from `0.5` at `25%` to `0.25` at `50%`
- at `tau=8`, the onset improves from `1.5` at `25%` to `1.0` at `50%`
- `100%` has no onset because it is the pooled null

For the boundary check at `tau=16`:

- `25%` remains negative at both `train_docs=512` and `train_docs=2048`
- `50%` remains positive at both `train_docs=512` and `train_docs=2048`
- `100%` remains exactly `0` at both training sizes

## Final Artifacts

Main analytical report:

- `outputs/tree_relevant_lda_followup_20260309_proportion_extension/report/tree_relevant_lda_proportion_extension_report.md`
- `outputs/tree_relevant_lda_followup_20260309_proportion_extension/report/tree_relevant_lda_proportion_extension_report.pdf`
- `outputs/tree_relevant_lda_followup_20260309_proportion_extension/report/tree_relevant_lda_proportion_extension_report_summary.json`

Publication package:

- `outputs/tree_relevant_lda_followup_20260309_proportion_extension/report/publication/tree_relevant_lda_proportion_extension_publication_report.md`
- `outputs/tree_relevant_lda_followup_20260309_proportion_extension/report/publication/tree_relevant_lda_proportion_extension_publication_report.pdf`
- `outputs/tree_relevant_lda_followup_20260309_proportion_extension/report/publication/tree_relevant_lda_proportion_extension_publication_diagnostics.json`

Clean figure assets:

- `outputs/tree_relevant_lda_followup_20260309_proportion_extension/report/publication/figures/publication_figures/figure_A_tau_frontier.png`
- `outputs/tree_relevant_lda_followup_20260309_proportion_extension/report/publication/figures/publication_figures/figure_B_last_positive_tau.png`
- `outputs/tree_relevant_lda_followup_20260309_proportion_extension/report/publication/figures/publication_figures/figure_C_lambda_onset_coarse.png`
- `outputs/tree_relevant_lda_followup_20260309_proportion_extension/report/publication/figures/publication_figures/figure_D_boundary_train_docs.png`
- `outputs/tree_relevant_lda_followup_20260309_proportion_extension/report/publication/figures/publication_figures/figure_E_null_control.png`

## Rerun Command

```bash
source venv/bin/activate
python3 scripts/report_tree_relevant_lda_proportion_extension_publication.py \
  --baseline-root outputs/tree_relevant_lda_followup_20260307_094903 \
  --extension-root outputs/tree_relevant_lda_followup_20260309_proportion_extension \
  --output-dir outputs/tree_relevant_lda_followup_20260309_proportion_extension/report/publication \
  --snapshot-label "Coarse Leaf-Size Publication Report (March 9, 2026)"
```

## Remaining Gaps

This closes the reporting gap for the coarse leaf-size question itself.

What still remains at the broader journal-package level is separate:

- integrate this coarse-size result into the main tree-relevant LDA paper/report stack if it should replace the older `96`-token-only headline
- decide whether Stage 3 realism (`weighting`, `boundary mismatch`, `adaptive labeling`) should be presented as an appendix to this main coarse-size story
- decide how this LDA publication result should be paired with the corrected Markov cross-DGP report in the unified local-law narrative
