# Paper Simulation Map

This file is the current paper-facing map for the formal simulation reruns. It is meant to answer two questions:

1. Which simulation roots are canonical for the paper?
2. Which report script turns each root into the paper-facing draft artifact?

The current canonical rerun root is:

- `outputs/formal_reruns_20260310_062551`

## Core Paper Layers

### 1. Baseline anchor

- Root: `outputs/formal_reruns_20260310_062551/cpu_megasweep`
- Purpose: broad baseline sweep across Markov, Segment-LDA OPS recovery, Segmented-LDA C-TreePO, and mergeable controls
- Paper role: baseline anchor figures and reference diagnostics
- Main report scripts:
  - `scripts/report_cpu_megasweep.py`
  - `scripts/report_cpu_megasweep_readable.py`

### 2. Buildout layer

- Root: `outputs/formal_reruns_20260310_062551/simulation_buildout`
- Purpose: the focused "why the method works / where it breaks" layer
- Paper role: buildout/stress section
- Expected sub-roots:
  - `hard_regimes`
  - `estimator_stress`
  - `guidance_frontier`
  - `ipw_expanded`
- Main report script:
  - `scripts/report_simulation_buildout.py`

### 3. Clean core publication slice

- Root: `outputs/formal_reruns_20260310_062551/identifiable_zero_longrun_clean`
- Purpose: compact cross-family oracle-equivalence story for the main paper
- Paper role: the main clean cross-family publication figures
- Main report script:
  - `scripts/report_identifiable_zero_suite_publication_clean.py`
- Fixed slices:
  - Segment-LDA OPS: `train_docs=12000`, `local-law lambda=1.0`
  - Segmented-LDA C-TreePO: `train_docs=4096`
  - Markov: `train_docs=8000`, `leaf_query_rate=1.0`, `include_root_query=true`

### 3a. Figure B Reading Rule

- Figure B is now paired by family: the left column is raw error and should be read within family only.
- The right column is normalized progress and is the cross-family comparison column.
- `B1` and `B3` are not directly unit-comparable: C-TreePO is plotted in root `L1`, while Markov additive is plotted in root `MAE`.
- The paper-safe cross-family read is therefore `B2` vs `B4`, not `B1` vs `B3`.
- On that normalized read, C-TreePO is not simply "worse"; it is more decision-time dependent. Its gap closes sharply once decision-time visibility is available, while the Markov additive lane improves more gradually.
- When summarizing the figure in prose, describe C-TreePO as "late-closing / decision-time dependent", not as uniformly underperforming the Markov lane.

### 3b. Normalization / `N/A` Rule

- `N/A` in the clean publication report is never a large bad value.
- It means normalization is undefined because the observed baseline and observed ceiling are already numerically indistinguishable, i.e. `baseline - ceiling <= 1e-12`.
- Valid normalized values above `1.2` may still be clipped for display; that is a different case from `N/A`.
- In reader-facing prose, call these cases "undefined normalization" or "no observed improvable gap", not "high normalized error".

### 3c. Figure Layout Rule

- Prefer taller figures with fewer columns when the panels mix raw and normalized views.
- Keep cross-family interpretation in the markdown memo rather than inside each subplot.
- Use figures for structure and values; keep the paper-safe takeaways in the report text so the layout can be reused across Markov and LDA families.

### 4. Publication C-TreePO suite

- Root: `outputs/formal_reruns_20260310_062551/identifiable_zero_publication_ctreepo`
- Purpose: richer publication-profile C-TreePO / LDA sweep, including lane-wise partial progress
- Paper role: expanded LDA/C-TreePO results and progress diagnostics while the suite is still in flight
- Main report script:
  - `scripts/report_identifiable_zero_publication_ctreepo_progress.py`

## Appendix / Robustness Layers

### 6. Learnability

- Expected root: `outputs/formal_reruns_20260310_062551/identifiable_zero_learnability`
- Purpose: appendix-quality learnability story
- Main report script:
  - `scripts/report_identifiable_zero_learnability.py`

### 7. Neural operator overnight

- Root: `outputs/formal_reruns_20260310_062551/identifiable_zero_neural_operator_v2`
- Purpose: operator-capacity / operator-guidance robustness
- Main report script:
  - `scripts/report_identifiable_zero_neural_operator_overnight.py`

### 8. LDA leaf-noise progression

- Expected root: `outputs/formal_reruns_20260310_062551/identifiable_zero_lda_leafnoise`
- Purpose: appendix-style LDA degradation / leaf-noise story
- Main report script:
  - `scripts/report_identifiable_zero_lda_leafnoise_progression.py`

### 9. DTM-LDA

- Expected root: `outputs/formal_reruns_20260310_062551/identifiable_zero_dtm_lda`
- Purpose: additional appendix / robustness coverage
- Report status: no dedicated paper report script yet; current handling is bundle-level placeholder status only

## Diagnostic-Only Layers

### 10. LDA tree recovery production suite

- Root: `outputs/formal_reruns_20260310_062551/lda_tree_recovery_production`
- Purpose: dedicated LDA tree-recovery experiment
- Paper role: diagnostic only; not the paper-facing local-law/learnability LDA counterpart
- Main report script:
  - `scripts/report_lda_tree_recovery_progress.py`

## Objective / IPW Contract

The current reruns are intended to reflect the realized optimized objective, not report-time defaults.

- Markov and LDA local-law runs now serialize the realized objective payload.
- Estimator-aware variants such as `configured_objective_ht` and `configured_objective_hajek` are first-class.
- Report layers should consume realized weights and estimator-aware objective fields whenever they exist.
- The clean Markov publication slice is pinned to the historical regime that was actually optimized in the older publication root:
  - task weight `1.0`
  - C3 weight `0.2`
  - C1/C2 off

## Lean Alignment

- `lean3/FormalProofs/OPT/MarkovCountSketchExample.lean`: exact mergeable Markov count sketch and local-law control; anchors the Markov exact/additive ceilings and why C1/C3-style errors are theorem-facing diagnostics.
- `lean3/FormalProofs/OPT/BagOfWordsLDARecovery.lean`: exact bag-of-words histogram recovery; anchors the pooled-counts / exact-merge control for Segment-LDA and related LDA baseline stories.
- `lean3/FormalProofs/OPT/LeafLocalMixtureUtilityGap.lean`: pooled-vs-leaf gap under nonlinear local utility; applies to diagnostic local-mixture LDA roots whose knob is a quadratic utility weight rather than the paper local-law lambda.

## Practical Workflow

For the current rerun root, use:

- `scripts/generate_paper_simulation_report_bundle.py --formal-root outputs/formal_reruns_20260310_062551`
- `scripts/report_simulation_theory_alignment.py --formal-root outputs/formal_reruns_20260310_062551 --bundle-manifest outputs/formal_reruns_20260310_062551/paper_reports/paper_report_bundle_manifest.json`

That driver:

- runs the existing paper-facing report scripts where data exists
- suppresses optional PDF emission for the pandoc-based reports
- writes placeholder draft reports for missing or still-empty suites
- builds a bundle-level status index under `<formal-root>/paper_reports`

The theory-alignment report writes:

- `<formal-root>/paper_reports/simulation_expectations.json`
- `<formal-root>/paper_reports/simulation_expectations.md`
- `<formal-root>/paper_reports/simulation_theory_alignment.json`
- `<formal-root>/paper_reports/simulation_theory_alignment.md`

Use that report as the canonical crosswalk from simulation families and paper suites to the Lean theorem surface. It is the quickest way to check whether a rerun root still matches the exact / approximate / proxy distinctions in the formalization.

For the current status read on the primary paper suites, see:

- `docs/simulation_theory_alignment_status_20260311.md`

## Notes

- The clean publication report is stricter than the progress reports because it expects a fixed-slice cross-family comparison.
- Partial roots are still worth reporting; the bundle generator records draft or pending status instead of forcing backward-looking defaults.
- Excluded from the paper-facing bundle: `outputs/tree_relevant_lda_local_law_20260308_210436`. In that reused root, `quadratic_utility_weight` (historically serialized as `lambda_multiplier`) is a latent quadratic-utility multiplier rather than a normalized local-law weight in `[0,1]`, so it should not be used as the paper LDA lambda comparison.
- If a future rerun root supersedes `formal_reruns_20260310_062551`, update this file and rerun the bundle generator against the new root.
